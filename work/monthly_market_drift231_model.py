"""Pure current-market-conditioned daily drift, with whole-month common support.

No I/O, training labels, predictive models, portfolio scores or orders. The host
owns original calm-state verification and the one replacement selection hook.
"""
from collections import Counter
from pathlib import Path
import math
import numbers

import numpy as np
import pandas as pd

ARMS = ('incumbent_local67', 'market_conditioned_drift', 'unconditional_drift_control')
ARRAY_KEYS = ('candidate_raw', 'control_raw', 'candidate_effect', 'control_effect', 'ready')
COUNT_FIELDS = ('query_months', 'eligible_cells', 'supported_months', 'fallback_months',
    'initial_partial_months', 'domain_fallback_months', 'source_fallback_months',
    'state_support_fallback_months', 'checked_stock_cells', 'source_valid_stock_cells',
    'stock_price_observations', 'stock_log_returns', 'drift_estimates',
    'ranked_observations', 'market_state_evaluations', 'market_price_observations')
INITIAL = '2016-08-17'
SKIP, CALM_RETURNS, OTHER_RETURNS = 42, 210, 252
MARKET_WINDOW, MIN_MATCHING, SHRINKAGE = 200, 42, 63


def dependency_files():
    return (Path(__file__).resolve(),)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite arithmetic from finite source')
    return result


def _sum(values):
    try:
        return _finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite scalar sum') from exc


def _mean(values):
    total = _sum(values)
    result = _finite(total/len(values))
    if total != 0. and result == 0.:
        raise FloatingPointError('Nonzero mean underflowed')
    return result


def _prices(raw, size, kind):
    _require(np.shape(raw) == (size,), 'Exact '+kind+' price shape required')
    values = np.asarray(raw, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in values):
        return None, 'nonnumeric_'+kind+'_price'
    try:
        result = [float(x) for x in values]
    except (ValueError, OverflowError):
        return None, 'nonnumeric_'+kind+'_price'
    if any(not math.isfinite(x) or x <= 0. for x in result):
        return None, 'invalid_'+kind+'_price'
    return result, None


def market_state(prices):
    """Exactly200 prices; equality is the non-up state, with no epsilon."""
    p, reason = _prices(prices, MARKET_WINDOW, 'market')
    _require(reason is None, 'Complete positive market prices required')
    mean = _mean(p)
    return bool(p[-1] > mean), mean


def drift(prices, matching):
    """Fixed210/252 complete returns and >=42 matching prior-day states."""
    mask = np.asarray(matching)
    _require(mask.ndim == 1 and mask.dtype == bool and len(mask) in (CALM_RETURNS, OTHER_RETURNS),
             '210 or252 exact boolean state matches required')
    p, reason = _prices(prices, len(mask)+1, 'stock')
    _require(reason is None, 'Complete positive stock prices required')
    n = int(mask.sum())
    _require(n >= MIN_MATCHING, 'At least42 matching state days required')
    # The preregistered stable realization of log(Pd/Pprev) is log difference.
    logs = [math.log(x) for x in p]
    returns = [_finite(logs[i+1]-logs[i]) for i in range(len(mask))]
    mu = _mean(returns)
    mu_state = _mean([r for r, matched in zip(returns, mask) if matched])
    if n == len(mask) or mu_state == mu:
        candidate = mu
    else:
        weight = n/(n+float(SHRINKAGE))
        candidate = _finite(mu + _finite(weight*_finite(mu_state-mu)))
    return dict(unconditional_mean=mu, conditional_mean=mu_state,
                candidate_raw=candidate, control_raw=mu)


def effect_ranks(values):
    """Average exact ties in the full E domain; all-constant is exactly zero."""
    a = np.asarray(values)
    _require(a.ndim == 1 and len(a) >= 2, 'At least two effect values required')
    v = [_finite(x) for x in a]
    if min(v) == max(v):
        return np.zeros(len(v), dtype=float)
    result = []
    for x in v:
        lower, equal = sum(y < x for y in v), sum(y == x for y in v)
        average_rank = lower + (equal+1)/2.
        result.append((average_rank-1.)/(len(v)-1.)-.5)
    return np.asarray(result, dtype=float)


def _day(value):
    _require(isinstance(value, str), 'Canonical ISO date required')
    try:
        day = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Canonical ISO date required') from exc
    _require(not pd.isna(day) and day.tzinfo is None and day == day.normalize()
             and value == str(day.date()), 'Canonical ISO date required')
    return day


def _calendar(calendar):
    _require(not isinstance(calendar, (str, bytes)), 'Explicit official calendar required')
    days = [_day(x) for x in calendar]
    _require(bool(days) and all(x.weekday() < 5 for x in days)
             and all(b > a for a, b in zip(days, days[1:])),
             'Unique increasing weekday official calendar required')
    return pd.DatetimeIndex(days)


def _frame(frame):
    _require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex),
             'Adjusted-price DataFrame with DatetimeIndex required')
    idx = frame.index
    _require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing
             and idx.tz is None and idx.equals(idx.normalize()), 'Invalid source price index')
    _require(frame.columns.is_unique and all(isinstance(s, str) and s and s == s.strip()
                                            for s in frame.columns), 'Invalid security columns')


def _inputs(panel, meta, calendar, arrays, limit):
    _require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Exactly adj panel required')
    _frame(panel['adj'])
    sessions = _calendar(calendar)
    _require(isinstance(arrays, dict) and {'eligible', 'state_calm10'} <= set(arrays),
             'Original eligibility and calm state arrays required')
    eligible, calm = np.asarray(arrays['eligible']), np.asarray(arrays['state_calm10'])
    _require(eligible.ndim == 2 and eligible.dtype == bool, 'Eligibility must be a boolean matrix')
    rows, width = eligible.shape
    _require(rows == 0 or width > 0, 'Nonempty queries need positive width')
    _require(calm.dtype == bool and calm.shape in ((rows,), (rows, 1)), 'Exact boolean calm vector required')
    _require(limit is None or (type(limit) is int and 0 <= limit <= rows), 'Integer query prefix required')
    n = rows if limit is None else limit
    _require(isinstance(meta, dict) and {'dates', 'symbols'} <= set(meta), 'Explicit query metadata required')
    for name in ('dates', 'symbols'):
        _require(isinstance(meta[name], (list, tuple)) and len(meta[name]) == rows, 'Metadata axes differ')
    if 'ends' in meta:
        _require(isinstance(meta['ends'], (list, tuple)) and len(meta['ends']) == rows, 'End metadata axis differs')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries = []
    for i in range(n):
        day = _day(meta['dates'][i])
        _require(day in lookup and (not queries or day > queries[-1][0]), 'Unique ordered official query sessions required')
        t = lookup[day]
        if str(day.date()) == INITIAL:
            _require(i == 0, 'Initial partial can only be the first query')
        else:
            _require(t+1 < len(sessions) and sessions[t+1].to_period('M') != day.to_period('M'),
                     'Verified official monthend required')
        symbols = meta['symbols'][i]
        _require(isinstance(symbols, (list, tuple)) and len(symbols) == width
                 and all(isinstance(s, str) and s and s == s.strip() for s in symbols)
                 and len(set(symbols)) == width, 'Exact unique query security axis required')
        queries.append((day, list(symbols), t))
    return panel['adj'], sessions, queries, eligible[:n], calm.reshape(rows)[:n], width


def _cell(column, symbol):
    return dict(column=int(column), symbol=symbol, source_checked=False,
        source_available=False, reason='not_checked', price_start=None, price_end=None,
        unconditional_mean=None, conditional_mean=None, candidate_raw=None, control_raw=None,
        candidate_effect=0., control_effect=0.)


def _source_event(adj, sessions, t, L, cells, event, counts):
    """Validate all required market input, then every eligible stock source."""
    u = t-SKIP
    if t < L+SKIP+MARKET_WINDOW-1:
        return 'insufficient_calendar', None, None
    formation = sessions[u-L:u+1]
    first, last = u-L, u-1
    event.update(formation_start=str(formation[0].date()), formation_end=str(formation[-1].date()),
        historical_state_first=str(sessions[first].date()), historical_state_last=str(sessions[last].date()))
    if 'SPY' not in adj.columns:
        return 'missing_market_symbol', None, None
    historical_market = sessions[first-MARKET_WINDOW+1:last+1]
    query_market = sessions[t-MARKET_WINDOW+1:t+1]
    required = historical_market.union(query_market)
    if len(required.difference(adj.index)):
        return 'missing_market_session', None, None
    market, reason = _prices(adj.loc[required, 'SPY'].to_numpy(copy=True), len(required), 'market')
    if reason:
        return reason, None, None
    market_by_day = dict(zip(required, market))
    query_prices = [market_by_day[x] for x in query_market]
    query_up, query_mean = market_state(query_prices)
    states = []
    for position in range(first, last+1):
        window = [market_by_day[x] for x in sessions[position-MARKET_WINDOW+1:position+1]]
        states.append(market_state(window)[0])
    matching = np.asarray([s == query_up for s in states], dtype=bool)
    event.update(query_market_close=query_prices[-1], query_market_mean=query_mean,
                 query_state_up=query_up, historical_up_days=sum(states),
                 matching_state_days=int(matching.sum()))
    counts['market_state_evaluations'] = L+1
    counts['market_price_observations'] = MARKET_WINDOW*(L+1)
    stock_prices = {}
    missing_sessions = bool(len(formation.difference(adj.index)))
    for cell in cells:
        cell['source_checked'] = True
        counts['checked_stock_cells'] += 1
        symbol = cell['symbol']
        if symbol not in adj.columns:
            reason = 'missing_symbol'
        elif missing_sessions:
            reason = 'missing_stock_session'
        else:
            p, reason = _prices(adj.loc[formation, symbol].to_numpy(copy=True), L+1, 'stock')
            if reason is None:
                stock_prices[cell['column']] = p
                cell.update(source_available=True, price_start=p[0], price_end=p[-1])
                counts['source_valid_stock_cells'] += 1
                counts['stock_price_observations'] += L+1
        cell['reason'] = reason or 'source_valid'
    if not all(cell['source_available'] for cell in cells):
        return 'invalid_stock_source', None, None
    if int(matching.sum()) < MIN_MATCHING:
        return 'insufficient_matching_state', None, None
    return 'supported', stock_prices, matching


def build(panel, meta, calendar, arrays, limit=None):
    adj, sessions, queries, eligible, calm, width = _inputs(panel, meta, calendar, arrays, limit)
    shape = (len(queries), width)
    values = {key: np.full(shape, np.nan) for key in ('candidate_raw', 'control_raw')}
    values.update({key: np.zeros(shape, float) for key in ('candidate_effect', 'control_effect')})
    values['ready'] = np.zeros(len(queries), bool)
    records = []
    for i, (day, symbols, t) in enumerate(queries):
        ids = np.flatnonzero(eligible[i]).tolist()
        L = CALM_RETURNS if calm[i] else OTHER_RETURNS
        cells = [_cell(j, symbols[j]) for j in ids]
        counts = dict.fromkeys(COUNT_FIELDS, 0)
        counts.update(query_months=1, eligible_cells=len(ids))
        event = dict(row=i, date=str(day.date()), symbols=symbols, eligible_indices=ids,
            calm=bool(calm[i]), formation_returns=L, formation_start=None, formation_end=None,
            historical_state_first=None, historical_state_last=None, query_market_close=None,
            query_market_mean=None, query_state_up=None, historical_up_days=None,
            matching_state_days=None, ready=False, status=None, counts=counts, cells=cells)
        if event['date'] == INITIAL:
            status = 'initial_partial'
            counts['initial_partial_months'] = 1
        elif len(ids) < 2:
            status = 'fewer_than_two_eligible'
            counts['domain_fallback_months'] = 1
        else:
            status, stock, matching = _source_event(adj, sessions, t, L, cells, event, counts)
            if status == 'supported':
                values['ready'][i] = event['ready'] = True
                counts.update(supported_months=1, stock_log_returns=L*len(ids),
                              drift_estimates=2*len(ids), ranked_observations=2*len(ids))
                for cell in cells:
                    j = cell['column']
                    cell.update(drift(stock[j], matching))
                    for key in ('candidate_raw', 'control_raw'):
                        values[key][i, j] = cell[key]
                for arm in ('candidate', 'control'):
                    effects = effect_ranks(values[arm+'_raw'][i, ids])
                    values[arm+'_effect'][i, ids] = effects
                    for cell, effect in zip(cells, effects):
                        cell[arm+'_effect'] = float(effect)
            elif status == 'insufficient_matching_state':
                counts['state_support_fallback_months'] = 1
            else:
                counts['source_fallback_months'] = 1
        event['status'] = status
        counts['fallback_months'] = int(not event['ready'])
        records.append(event)
    total = {key: sum(r['counts'][key] for r in records) for key in COUNT_FIELDS}
    audit = dict(method='current_market_conditioned_skip42_daily_drift', arms=list(ARMS),
        events=len(queries), symbols_per_event=width, skip_sessions=SKIP,
        calm_returns=CALM_RETURNS, other_returns=OTHER_RETURNS, market_symbol='SPY',
        market_mean_sessions=MARKET_WINDOW, min_matching_days=MIN_MATCHING,
        shrinkage_days=SHRINKAGE,
        market_price_observation_counting='200 per evaluated state; overlapping windows counted repeatedly',
        counts=total, new_model_fits=0, predictive_model_fits=0, new_ridge_fits=0,
        new_tree_fits=0, reason_counts=dict(sorted(Counter(r['status'] for r in records).items())),
        event_history=records)
    return dict(arrays=values, audit=audit)
