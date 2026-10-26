"""Authorized causal replay using the frozen strategy and next-open paper engine.

This module never opens a ledger. Callers supply a versioned database and an
already verified snapshot. Retrospectively retrieved quotes are not represented
as a point-in-time vendor archive or as a plan created before the market open.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_KEY = 'execution_v3_policy'
VERSION = 'causal_replay_v3'
ORIGIN = 'authorized_causal_replay'
STATUS = 'authorized_causal_replay_executed'


def _date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise RuntimeError('A naive session date is required')
    return day


def _utc(value, label):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise RuntimeError(label + ' must include a timezone')
    return stamp.tz_convert('UTC')


def _previous(state, day, calendar, cfg):
    day = _date(day)
    dates = calendar.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None:
        raise RuntimeError('Calendar requires naive session dates')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise RuntimeError('Calendar sessions must be unique and ordered')
    if not dates.equals(dates.normalize()) or day not in dates:
        raise RuntimeError('Invalid calendar session')
    i = dates.get_loc(day)
    if i == 0:
        raise RuntimeError('Prior signal session required')
    previous = dates[i - 1]
    last = state.get('last_date')
    if last is None:
        eligible = dates[dates >= _date(cfg['paper_start'])]
        if len(eligible) == 0 or day != eligible[0] or state.get('positions'):
            raise RuntimeError('Initial state is only valid for the first paper session')
    elif _date(last) != previous:
        raise RuntimeError('State must end on the immediately preceding session')
    return previous


def _policy(db, now):
    row = db.execute('SELECT value FROM meta WHERE key=?', (POLICY_KEY,)).fetchone()
    if not row:
        raise RuntimeError('Standing causal replay authorization is required')
    try:
        policy = json.loads(row[0])
        if not isinstance(policy, dict) or policy.get('version') != VERSION:
            raise ValueError('wrong policy version')
        authorized = _utc(policy['authorized_at_utc'], 'Policy authorization')
        if authorized > now:
            raise ValueError('authorization is in the future')
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise RuntimeError('Invalid standing causal replay authorization') from exc
    return authorized.isoformat()


def _validated_splits(history, symbol, cutoff):
    ix = history.index
    if (not isinstance(ix, pd.DatetimeIndex) or ix.tz is not None or
            not ix.is_unique or not ix.is_monotonic_increasing or
            not ix.equals(ix.normalize()) or len(ix) == 0 or ix[-1] > cutoff or
            'Stock Splits' not in history):
        raise RuntimeError('Invalid/incomplete signal history: ' + symbol)
    splits = pd.to_numeric(history['Stock Splits'], errors='coerce')
    if not np.isfinite(splits.to_numpy()).all() or (splits < 0).any():
        raise RuntimeError('Invalid split history: ' + symbol)
    return splits


def _signal_views(frames, manifest, previous, cfg):
    """Undo only actions after P, then expose no rows later than P.

    Yahoo raw OHLC/dividends and volume are split-adjusted. Adj Close also has
    dividend adjustments. Anchoring Adj Close at P removes the common future
    adjustment while retaining relative historical total returns through P.
    """
    cutoff = _date(manifest['as_of'])
    if cutoff < previous or manifest.get('failures'):
        raise RuntimeError('Incomplete snapshot for prior-close signals')
    if not isinstance(manifest.get('snapshot_id'), str) or not manifest['snapshot_id']:
        raise RuntimeError('Snapshot identity is required')
    stocks = list(cfg['symbols'])
    if len(stocks) != len(set(stocks)):
        raise RuntimeError('Duplicate configured symbols')
    sectors = list(dict.fromkeys(cfg['sector_etfs'].values()))
    allstocks = list(dict.fromkeys(stocks + ['SPY', 'QQQ']))
    required = list(dict.fromkeys(allstocks + sectors))
    fields = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    views, factors = {}, {}
    for symbol in required:
        if symbol not in frames:
            raise RuntimeError('Missing signal history: ' + symbol)
        history = frames[symbol]
        ix = history.index
        splits = _validated_splits(history, symbol, cutoff)
        if previous not in ix or any(field not in history for field in fields):
            raise RuntimeError('Invalid/incomplete signal history: ' + symbol)
        later = splits.loc[ix > previous]
        factor = float(later.where(later > 0, 1.).prod())
        if not math.isfinite(factor) or factor <= 0:
            raise RuntimeError('Invalid future split factor: ' + symbol)
        view = history.loc[ix <= previous, fields].copy().astype(float)
        last = view.loc[previous]
        positive = last[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        # Match frozen features: H/L must enclose Close; Open is a separate
        # positive quote, not an additional H/L-based admission filter.
        if (not np.isfinite(positive.to_numpy()).all() or not (positive > 0).all() or
                not math.isfinite(float(last['Dividends'])) or
                last['Low'] > last['Close'] + 1e-6 or
                last['High'] < last['Close'] - 1e-6):
            raise RuntimeError('Invalid prior-session OHLC/volume: ' + symbol)
        close, adjusted = float(last['Close']), float(last['Adj Close'])
        if not all(math.isfinite(v) and v > 0 for v in (close, adjusted, close * factor)):
            raise RuntimeError('Invalid prior-close price basis: ' + symbol)
        view[['Open', 'High', 'Low', 'Close', 'Dividends']] *= factor
        view['Volume'] /= factor
        view['Adj Close'] *= close * factor / adjusted
        views[symbol] = view
        factors[symbol] = factor
    ix = views['SPY'].index
    def matrix(field, symbols):
        return pd.DataFrame({symbol: views[symbol][field].reindex(ix) for symbol in symbols}, index=ix)
    # Use the original data container; neither its loaders nor research runners run.
    from backtest_swing import Inputs
    data = Inputs(matrix('Close', allstocks), matrix('Open', allstocks),
                  matrix('Dividends', allstocks), matrix('Volume', allstocks),
                  {**cfg['sectors'], 'SPY': 'ETF', 'QQQ': 'ETF'}, matrix('Close', sectors))
    raw = {'adj_close': matrix('Adj Close', stocks), 'high': matrix('High', stocks),
           'low': matrix('Low', stocks), 'sector_adjusted': matrix('Adj Close', sectors)}
    return data, raw, factors


def build_decision(m, state, day, calendar, frames, manifest, cfg):
    """Pure prior-close decision; the caller assigns origin and creation time."""
    previous = _previous(state, day, calendar, cfg)
    source_hashes = manifest.get('hashes', {})
    if not isinstance(source_hashes, dict):
        raise RuntimeError('Snapshot hashes must be the verified loader mapping')
    data, raw, factors = _signal_views(frames, manifest, previous, cfg)
    features = m.features(data, raw)
    decision = m.plan(copy.deepcopy(state), _date(day), calendar.index, data.close,
                      features['ma5'], features['atr'], features['masks'][cfg['admission']],
                      features['score'], cfg['sectors'], cfg)
    expected = str(previous.date())
    if decision.get('session') != str(_date(day).date()) or decision.get('signal_date') != expected:
        raise RuntimeError('Strategy returned an inconsistent decision date')
    decision.update(execution_policy=VERSION, decision_state_sha256=hashlib.sha256(m.dumps(state).encode()).hexdigest(),
                    decision_as_of=expected, data_as_of=expected,
                    parent_snapshot=manifest['snapshot_id'], parent_snapshot_as_of=manifest['as_of'],
                    parent_snapshot_hashes_sha256=hashlib.sha256(m.dumps(source_hashes).encode()).hexdigest(),
                    parent_snapshot_retrieved_at_utc=manifest.get('retrieved_at_utc'),
                    data_vintage='retrospectively_retrieved',
                    signal_price_basis='prior_session_split_units_adj_close_anchored_at_prior_close',
                    future_split_factors={s: v for s, v in factors.items() if v != 1.},
                    strategy_version=cfg['version'], cost_bps=cfg['cost_bps'],
                    fill_model='next_session_open_proxy_not_actual_broker_fill')
    return decision


def _validate_record(db, record, day, previous, calendar, now, authorized):
    created = _utc(record[0], 'Plan creation')
    decision = json.loads(record[1])
    ds, ps = str(day.date()), str(previous.date())
    if not isinstance(decision, dict) or decision.get('session') != ds:
        raise RuntimeError('Recorded plan session mismatch')
    if not isinstance(decision.get('buys'), list) or not isinstance(decision.get('sells'), list):
        raise RuntimeError('Invalid recorded orders')
    if created > now:
        raise RuntimeError('Recorded plan creation is in the future')
    for key in ('signal_date', 'decision_as_of', 'data_as_of'):
        if key in decision and decision[key] != ps:
            raise RuntimeError('Recorded plan is not based on the previous session')
    opening = _utc(calendar.at[day, 'open'], 'Market open')
    if created >= opening - pd.Timedelta(minutes=1):
        origin = decision.get('origin')
        if origin == 'user_authorized_retrospective':
            row = db.execute('SELECT value FROM meta WHERE key=?', ('authorized_retrospective_' + ds,)).fetchone()
            if not row or not row[0]:
                raise RuntimeError('Late legacy plan lacks its explicit daily authorization')
        elif origin == ORIGIN:
            if (decision.get('execution_policy') != VERSION or
                    decision.get('policy_authorized_at_utc') != authorized or
                    decision.get('decision_as_of') != ps or
                    decision.get('signal_date') != ps or
                    decision.get('data_as_of') != ps or
                    decision.get('data_vintage') != 'retrospectively_retrieved' or
                    not decision.get('parent_snapshot')):
                raise RuntimeError('Late replay plan lacks causal authorization/provenance')
        else:
            raise RuntimeError('Late plan lacks authorized causal replay origin')
    return decision


def reconcile(m, db, cfg, frames, manifest, now):
    """Replay missing completed sessions, atomically committing each entire day."""
    now = _utc(now, 'Reconciliation time')
    authorized = _policy(db, now)
    if db.in_transaction:
        raise RuntimeError('Reconciliation requires no caller transaction')
    state = m.latest_state(db)
    calendar = m.schedule(now)
    cutoff = _date(manifest['as_of'])
    if cutoff not in calendar.index or manifest.get('failures'):
        raise RuntimeError('Invalid/incomplete reconciliation snapshot')
    if _utc(calendar.at[cutoff, 'close'], 'Market close') + pd.Timedelta(minutes=15) > now:
        raise RuntimeError('Snapshot session is not complete')
    if state.get('last_date') and _date(state['last_date']) > cutoff:
        raise RuntimeError('Snapshot precedes the latest ledger state')
    days = calendar.index[(calendar.index >= _date(cfg['paper_start'])) & (calendar.index <= cutoff)]
    if state.get('last_date'):
        days = days[days > _date(state['last_date'])]
    count = 0
    for day in days:
        previous = _previous(state, day, calendar, cfg)
        ds = str(day.date())
        record = db.execute('SELECT created_utc,payload FROM plans WHERE session=?', (ds,)).fetchone()
        if record:
            decision = _validate_record(db, record, day, previous, calendar, now, authorized)
            if ((decision.get('execution_policy') == VERSION or decision.get('origin') == ORIGIN) and
                    'decision_state_sha256' not in decision):
                raise RuntimeError('V3 plan lacks its account state hash')
            if ('decision_state_sha256' in decision and decision['decision_state_sha256'] !=
                    hashlib.sha256(m.dumps(state).encode()).hexdigest()):
                raise RuntimeError('Recorded plan was built from a different account state')
        else:
            decision = build_decision(m, state, day, calendar, frames, manifest, cfg)
            decision.update(origin=ORIGIN, created_utc=now.isoformat(), execution_policy=VERSION,
                            policy_authorized_at_utc=authorized)
        status = STATUS if decision.get('origin') == ORIGIN else (
            'user_authorized_retrospective_executed' if decision.get('origin') == 'user_authorized_retrospective'
            else 'recorded_plan_executed')
        needed = set(state['positions']) | {r['symbol'] for r in decision['buys']}
        for symbol in needed:
            splits = _validated_splits(frames[symbol], symbol, cutoff)
            later = splits.loc[splits.index > previous]
            factor = float(later.where(later > 0, 1.).prod())
            if not math.isfinite(factor) or factor <= 0:
                raise RuntimeError('Invalid execution split factor: ' + symbol)
        # Preserve the recorded signal plan, including its ATR, exactly as stored.
        execution = copy.deepcopy(decision)
        unit_adjustments = []
        for buy in execution['buys']:
            if 'reference_close' in buy and 'signal_date' in execution:
                original_close = m.session_bar(frames[buy['symbol']], execution['signal_date'])['close']
                reference = float(buy['reference_close'])
                if not all(math.isfinite(v) and v > 0 for v in (reference, original_close)):
                    raise RuntimeError('Invalid plan price basis')
                scale = reference / original_close
                buy['atr'] /= scale
                if abs(scale - 1.) > 1e-8:
                    unit_adjustments.append(dict(symbol=buy['symbol'], plan_atr_unit_scale=scale))
        bars = {}
        for symbol in needed:
            history = frames[symbol]
            if history.index[-1] > cutoff:
                raise RuntimeError('Execution history extends past snapshot cutoff')
            bar = m.session_bar(history, day)
            split = float(bar.get('split', 0.))
            if not math.isfinite(split) or split < 0:
                raise RuntimeError('Invalid execution split')
            bars[symbol] = bar
        new, daily, trades, orders, actions = m.execute(
            state, execution, bars, calendar.index, cfg['cost_bps'], cfg['sectors'])
        actions.extend(unit_adjustments)
        # A failed insert rolls back the new plan as well as every execution row.
        with db:
            if not record:
                snapshot = str(Path(m.CACHE) / manifest['snapshot_id'])
                db.execute('INSERT INTO plans VALUES (?,?,?,?,?)',
                           (ds, now.isoformat(), m.dumps(decision), snapshot, m.fingerprint()))
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',
                       (ds, m.dumps(daily), m.dumps(new), status, manifest['snapshot_id']))
            db.executemany('INSERT INTO trades(data) VALUES (?)', [(m.dumps(t),) for t in trades])
            db.executemany('INSERT INTO orders(data) VALUES (?)', [(m.dumps(o),) for o in orders])
            db.executemany('INSERT INTO corporate_actions(session,data) VALUES (?,?)',
                           [(ds, m.dumps(a)) for a in actions])
        state = new
        count += 1
    return count
