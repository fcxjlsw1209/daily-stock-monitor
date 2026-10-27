"""Append only matured January--July 2026 labels to the frozen 169 inputs.

Caller supplies arrays, metadata, adjusted prices and the authenticated session
calendar. This module does not load data, fit a model, select holdings or replay
portfolio returns. Source identity/provenance and the experiment PLAN belong to
the caller. Full original cache bytes are copied/compared, including labels later
than an early query; only NEW label arithmetic is restricted by ``asof``. A
learner must independently enforce its training clock on the returned cache.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = 'monthly_matured_clock273_inputs_v1'
WIDTH, BASE_MONTHS, EVENT_MONTHS, FEATURES = 49, 180, 122, 10
BASE_ROWS = WIDTH * BASE_MONTHS
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
TRAIN_KEYS = ('train_plain', 'event_plain', 'train_eligible', 'train_good',
              'dates', 'ends', 'event_dates', 'offsets', 'eligible',
              'common_good', 'vol63') + tuple('target_' + x for x in TARGETS)
EVENT_KEYS = ('event_plain', 'event_dates', 'eligible')
ROW_KEYS = tuple(k for k in TRAIN_KEYS if k not in EVENT_KEYS + ('offsets',))
HELPER_SHA256 = {
    'outputs/monthly_core_expansion/ordinal_model.py':
        '4a8cd20e0be109c1e4f8175b5cb182941090dda3025703ac7870ebc1391abce3',
    'outputs/monthly_target_definition70/labels.py':
        '54662ea980b03e6088afa63318321da1316fd03cde1a72edffb33d4d2a26e618',
    'outputs/monthly_architecture_restart/residual_model.py':
        'ea294b9a0a1e9493799f1014dcec464effa93dce70f44fb5de53a604cd832886',
}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value, name):
    _need(type(value) is str, name + ' must be an ISO day string')
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(name + ' must be an ISO day string') from exc
    _need(result.isoformat() == value, name + ' is not canonical')
    return value


def _same(a, b, name):
    _need(isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
          a.shape == b.shape and a.dtype == b.dtype and
          a.tobytes(order='C') == b.tobytes(order='C'), name + ' bytes differ')


def _fingerprint(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def _snapshot(training):
    return {k: _fingerprint(training[k]) for k in TRAIN_KEYS}


def _metadata(meta, n, name):
    _need(type(meta) is dict and all(k in meta for k in ('dates', 'ends', 'symbols')),
          name + ' metadata fields missing')
    for key in ('dates', 'ends', 'symbols'):
        _need(type(meta[key]) is list and len(meta[key]) == n, name + ' axis mismatch')
    for i, (start, end, symbols) in enumerate(zip(meta['dates'], meta['ends'], meta['symbols'])):
        _day(start, name + '.date'); _day(end, name + '.end')
        _need(start < end, name + ' signal must precede label end')
        _need(type(symbols) is list and len(symbols) == WIDTH and
              all(type(s) is str and s and s.strip() == s for s in symbols) and
              symbols == sorted(set(symbols)), name + ' requires 49 sorted unique identities')
        if i:
            _need(meta['dates'][i - 1] < start, name + ' dates must increase')
    return meta


def _calendar(sessions):
    _need(type(sessions) in (list, tuple) and len(sessions) > 1, 'session calendar required')
    days = [_day(x, 'session') for x in sessions]
    _need(days == sorted(set(days)), 'sessions must be sorted and unique')
    _need(all(date.fromisoformat(x).weekday() < 5 for x in days), 'weekend session')
    return days, {x: i for i, x in enumerate(days)}


def _month_end(day, calendar, positions):
    _need(day in positions, 'month endpoint absent from session calendar: ' + day)
    pos = positions[day]
    _need(pos + 1 < len(calendar) and calendar[pos + 1][:7] != day[:7],
          'month endpoint lacks following-session witness: ' + day)


def _month_number(day):
    return int(day[:4]) * 12 + int(day[5:7]) - 1


def _validate(training, combined_meta, sessions, asof):
    _day(asof, 'asof')
    _need(type(training) is dict and set(training) == set(TRAIN_KEYS), 'training key set differs')
    _need(type(combined_meta) is dict and 'training' in combined_meta and 'monthly' in combined_meta,
          'combined metadata must contain training and monthly')
    tm = _metadata(combined_meta['training'], BASE_MONTHS, 'training')
    monthly = _metadata(combined_meta['monthly'], EVENT_MONTHS, 'monthly')
    _need(monthly['dates'][0] == '2016-08-17' and monthly['dates'][-1] == '2026-08-31',
          'original 122-event axis required')
    _need('etfs' in monthly and type(monthly['etfs']) is list and
          all(type(s) is str and s for s in monthly['etfs']) and
          len(set(monthly['etfs'])) == len(monthly['etfs']), 'monthly ETFs invalid')
    _need(all(set(monthly['etfs']).issubset(ss) for ss in monthly['symbols']), 'ETF axis mismatch')
    _need([d[:7] for d in tm['dates']] ==
          [str(x) for x in np.arange('2011-01', '2026-01', dtype='datetime64[M]')],
          'original 180 training months required; cannot append twice')
    _need([d[:7] for d in monthly['dates'][1:]] ==
          [str(x) for x in np.arange('2016-08', '2026-09', dtype='datetime64[M]')],
          'original formal monthly sequence required')
    _need(asof <= monthly['dates'][-1], 'asof exceeds the last permitted query')
    calendar, positions = _calendar(sessions)
    _need(asof in positions, 'asof absent from session calendar')
    for label, meta in (('training', tm), ('monthly', monthly)):
        for i, (start, end) in enumerate(zip(meta['dates'], meta['ends'])):
            if label == 'monthly' and i in (0, EVENT_MONTHS - 1):
                continue  # Initial partial and final evaluation stub are never appended.
            _need(_month_number(end) - _month_number(start) == 1, 'non-natural label month')
            _month_end(start, calendar, positions); _month_end(end, calendar, positions)
    bools = {'train_eligible', 'train_good', 'common_good', 'eligible'}
    days = {'dates', 'ends', 'event_dates'}
    for k in TRAIN_KEYS:
        a = training[k]
        shape = ((BASE_ROWS, FEATURES) if k == 'train_plain' else
                 (EVENT_MONTHS * WIDTH, FEATURES) if k == 'event_plain' else
                 (EVENT_MONTHS, WIDTH) if k == 'eligible' else
                 (EVENT_MONTHS * WIDTH,) if k == 'event_dates' else
                 (BASE_MONTHS + 1,) if k == 'offsets' else (BASE_ROWS,))
        _need(isinstance(a, np.ndarray) and a.shape == shape, k + ' shape mismatch')
        if k in bools:
            _need(a.dtype == np.dtype(bool), k + ' must be bool')
        elif k in days:
            _need(a.dtype == np.dtype('datetime64[D]'), k + ' must be datetime64[D]')
        elif k == 'offsets':
            _need(a.dtype == np.dtype('int64'), 'offsets must be int64')
        else:
            _need(a.dtype == np.dtype('float64'), k + ' must be float64')
    _same(training['offsets'], np.arange(BASE_MONTHS + 1, dtype=np.int64) * WIDTH, 'offsets')
    for key in ('dates', 'ends'):
        _same(training[key], np.repeat(np.array(tm[key], dtype='datetime64[D]'), WIDTH), key)
    _same(training['event_dates'], np.repeat(np.array(monthly['dates'], dtype='datetime64[D]'), WIDTH), 'event dates')
    # A shared signal must be exactly the same signal-time snapshot, not a later
    # reconstruction with different features, membership or eligibility.
    lookup = {d: j for j, d in enumerate(monthly['dates'])}
    shared = 0
    for i, day in enumerate(tm['dates']):
        if day not in lookup:
            continue
        j = lookup[day]
        _need(tm['ends'][i] == monthly['ends'][j] and tm['symbols'][i] == monthly['symbols'][j],
              'shared-month identity mismatch')
        _same(training['train_plain'][i * WIDTH:(i + 1) * WIDTH],
              training['event_plain'][j * WIDTH:(j + 1) * WIDTH], 'shared features')
        _same(training['train_eligible'][i * WIDTH:(i + 1) * WIDTH],
              training['eligible'][j], 'shared eligibility')
        shared += 1
    _need(shared == 113, 'expected 113 shared signal snapshots')
    candidates = [j for j, d in enumerate(monthly['dates']) if '2026-01' <= d[:7] <= '2026-07']
    _need(len(candidates) == 7, 'seven extension headers required')
    chosen = [j for j in candidates if monthly['ends'][j] <= asof]
    return tm, monthly, calendar, positions, chosen


def _load_helpers():
    root = Path(__file__).resolve().parents[1]
    for rel, expected in HELPER_SHA256.items():
        _need(hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected,
              'frozen label helper SHA mismatch: ' + rel)
    modules = []
    for name, rel in (('ordinal273', 'outputs/monthly_core_expansion/ordinal_model.py'),
                      ('labels273', 'outputs/monthly_target_definition70/labels.py')):
        spec = importlib.util.spec_from_file_location(name, root / rel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _price_window(panel, monthly, chosen, calendar, positions):
    _need(type(panel) is dict and 'adj' in panel, 'panel requires adj')
    adj = panel['adj']
    _need(isinstance(adj, pd.DataFrame) and isinstance(adj.index, pd.DatetimeIndex),
          'adj must be a DataFrame with DatetimeIndex')
    ix = adj.index
    _need(ix.tz is None and not ix.hasnans and ix.equals(ix.normalize()) and
          ix.is_unique and ix.is_monotonic_increasing and adj.columns.is_unique,
          'adjusted-price axes malformed')
    first, last = monthly['dates'][chosen[0]], monthly['ends'][chosen[-1]]
    begin, stop = positions[first] - 63, positions[last]
    _need(begin >= 0, 'fewer than 64 calendar sessions before first new label')
    expected = pd.DatetimeIndex(calendar[begin:stop + 1])
    actual = ix[(ix >= expected[0]) & (ix <= expected[-1])]
    _need(actual.equals(expected), 'consumed price sessions differ from official calendar')
    # Missing columns retain the original whole-month/row exclusions. No
    # imputation and no inspection of unrelated or post-maturity numeric cells.
    symbols = sorted({s for j in chosen for s in monthly['symbols'][j]} | {'SPY'})
    present = [s for s in symbols if s in adj.columns]
    window = adj.loc[expected, present].copy(deep=True)
    return window, dict(first_session=str(expected[0].date()),
                        last_session=str(expected[-1].date()),
                        sessions=len(expected), required_symbols=symbols,
                        present_symbols=present, copied_price_cells=int(window.size))


def _literal_oracle(adj, dates, ends, symbols, eligible, features):
    """Separate label construction; shares NumPy std/log/mean primitives only.

    Implements original eligible-whole-month path validity, each scalar endpoint
    formula, 63-return sample volatility, then common-cohort clip/centering.
    No model or frozen label implementation is called here.
    """
    n = len(dates)
    raw = {k: np.full(n, np.nan) for k in TARGETS}
    result = {k: np.full(n, np.nan) for k in TARGETS}
    vol = np.full(n, np.nan)
    path_good = np.zeros(n, dtype=bool)
    good = np.zeros(n, dtype=bool)
    endpoint = np.zeros(n, dtype=bool)
    limits = dict(simple_excess=1., log_excess=1., vol_scaled=5., loss_utility=1.)

    def values(symbol, positions):
        if symbol not in adj.columns:
            return None
        try:
            a = adj[symbol].iloc[positions].to_numpy(dtype=float)
        except (ValueError, TypeError, OverflowError):
            return None
        return a if np.all(np.isfinite(a) & (a > 0)) else None

    for start in range(0, n, WIDTH):
        rows = np.arange(start, start + WIDTH)
        sig, end = pd.Timestamp(dates[start]), pd.Timestamp(ends[start])
        lo, hi = adj.index.get_indexer([sig, end])
        selected = rows[eligible[rows]]
        names = list(symbols[selected])
        needed = names + ([] if 'SPY' in names else ['SPY'])
        complete = bool(len(selected)) and lo >= 0 and hi >= 0
        if complete:
            for name in needed:
                path = values(name, slice(lo, hi + 1))
                if path is None:
                    complete = False
                    break
                with np.errstate(all='ignore'):
                    payoff = path[-1] / path[0] - 1.0
                if not np.isfinite(payoff):
                    complete = False
                    break
        if complete:
            path_good[selected] = True
        spy = values('SPY', [lo, hi]) if lo >= 0 and hi >= 0 else None
        for row in rows:
            if lo >= 63:
                history = values(symbols[row], slice(lo - 63, lo + 1))
                if history is not None:
                    with np.errstate(all='ignore'):
                        daily = history[1:] / history[:-1] - 1.0
                        sigma = np.std(daily, ddof=1)
                    if np.isfinite(daily).all() and np.isfinite(sigma):
                        vol[row] = sigma
            prices = values(symbols[row], [lo, hi]) if lo >= 0 and hi >= 0 else None
            if spy is None or prices is None:
                continue
            endpoint[row] = True
            with np.errstate(all='ignore'):
                gross = prices[1] / prices[0]
                spy_gross = spy[1] / spy[0]
                ret = gross - 1.0
                excess = ret - (spy_gross - 1.0)
                raw['simple_excess'][row] = excess
                raw['log_excess'][row] = np.log(gross) - np.log(spy_gross)
                raw['vol_scaled'][row] = excess / (vol[row] * np.sqrt(21.0))
                raw['loss_utility'][row] = excess - 0.5 * max(0.0, -ret)
        common = (eligible[rows] & path_good[rows] & np.isfinite(features[rows]).all(axis=1)
                  & endpoint[rows] & np.isfinite(vol[rows]) & (vol[rows] > 0))
        for name in TARGETS:
            common &= np.isfinite(raw[name][rows])
        keep = rows[common]
        if len(keep) >= 2:
            good[keep] = True
            for name in TARGETS:
                clipped = np.clip(raw[name][keep], -limits[name], limits[name])
                result[name][keep] = clipped - clipped.mean()
    return dict(targets=result, good=good, vol63=vol, train_good=path_good)


def append_completed(training, combined_meta, panel, sessions, *, asof):
    """Return (15-key extended training, audit with ``training_metadata``).

    ``asof`` is a caller-authenticated physical query date, not the evaluation
    label end. Jan--Jul 2026 are the only possible new signal cohorts. An end
    equal to asof is mature. Full output has 187 months / 9163 rows. All source
    array shapes/dtypes/bytes and metadata are retained; new arrays own their
    memory. No extra August-to-September partial label is generated.
    """
    tm, monthly, calendar, positions, chosen = _validate(training, combined_meta, sessions, asof)
    before = _snapshot(training)
    metadata_before = deepcopy(combined_meta)
    out = {k: training[k].copy() for k in TRAIN_KEYS}
    extended_meta = deepcopy(tm)
    ordinal_audit = target_audit = window_audit = None
    oracle_checks = 0
    if chosen:
        ordinal, labels = _load_helpers()
        row_indices = np.concatenate([np.arange(j * WIDTH, (j + 1) * WIDTH) for j in chosen])
        features = training['event_plain'][row_indices].copy()
        eligible = training['eligible'][chosen].reshape(-1).copy()
        dates = np.repeat(np.array([monthly['dates'][j] for j in chosen], dtype='datetime64[D]'), WIDTH)
        ends = np.repeat(np.array([monthly['ends'][j] for j in chosen], dtype='datetime64[D]'), WIDTH)
        symbols = np.array([s for j in chosen for s in monthly['symbols'][j]])
        window, window_audit = _price_window(panel, monthly, chosen, calendar, positions)
        old_targets, ordinal_audit = ordinal.build_targets(window, dates, ends, symbols, eligible)
        train_good = eligible & np.isfinite(old_targets['return']) & np.isfinite(old_targets['ordinal'])
        new = labels.build_targets(window, dates, ends, symbols, eligible,
                                   train_good, np.isfinite(features).all(axis=1))
        oracle = _literal_oracle(window, dates, ends, symbols, eligible, features)
        for name, observed, expected in [('train_good', train_good, oracle['train_good']),
                                        ('common_good', new['good'], oracle['good']),
                                        ('vol63', new['vol63'], oracle['vol63'])] + [
                ('target_' + k, new['targets'][k], oracle['targets'][k]) for k in TARGETS]:
            _same(observed, expected, 'independent ' + name)
            oracle_checks += int(observed.size)
        additions = dict(train_plain=features, train_eligible=eligible, train_good=train_good,
                         dates=dates, ends=ends, common_good=new['good'], vol63=new['vol63'])
        additions.update({'target_' + k: new['targets'][k] for k in TARGETS})
        for k in ROW_KEYS:
            out[k] = np.concatenate((out[k], additions[k]), axis=0)
        out['offsets'] = np.concatenate((out['offsets'],
            np.arange(BASE_MONTHS + 1, BASE_MONTHS + len(chosen) + 1, dtype=np.int64) * WIDTH))
        for key in ('dates', 'ends', 'symbols'):
            extended_meta[key].extend(deepcopy([monthly[key][j] for j in chosen]))
        target_audit = new['audit']
    for key in ROW_KEYS:
        _same(out[key][:BASE_ROWS], training[key], 'preserved ' + key)
    _same(out['offsets'][:BASE_MONTHS + 1], training['offsets'], 'preserved offsets')
    for key in EVENT_KEYS:
        _same(out[key], training[key], 'preserved ' + key)
    _need(_snapshot(training) == before and combined_meta == metadata_before,
          'caller inputs changed during append')
    audit = dict(
        schema=SCHEMA, asof=asof, training_metadata=extended_meta,
        appended_months=[dict(signal_date=monthly['dates'][j], label_end=monthly['ends'][j],
                              event_index=j, first_row=BASE_ROWS + i * WIDTH, rows=WIDTH)
                         for i, j in enumerate(chosen)],
        original_arrays=before, original_arrays_unchanged=True, original_prefix_bytes_exact=True,
        event_arrays_unchanged=True, source_files_authenticated=False,
        helper_sha256=dict(HELPER_SHA256), ordinal=ordinal_audit, targets=target_audit,
        price_window=window_audit, independent_label_oracle_passed=True,
        independent_label_oracle_scope='separate literal formulas; shared NumPy std/log/clip/mean; exact bytes',
        counts=dict(original_months=BASE_MONTHS, original_rows=BASE_ROWS,
                    appended_months=len(chosen), appended_rows=len(chosen) * WIDTH,
                    total_months=BASE_MONTHS + len(chosen), total_rows=BASE_ROWS + len(chosen) * WIDTH,
                    shared_event_months_checked=113, new_feature_rows_reused=len(chosen) * WIDTH,
                    label_builder_calls=int(bool(chosen)), ordinal_builder_calls=int(bool(chosen)),
                    independent_label_routines=int(bool(chosen)), independent_scalar_comparisons=oracle_checks,
                    new_model_fits=0, predictive_model_fits=0),
        consumption_scope='Full original cache bytes copied/compared; new price arithmetic only matured extension windows. Calendar/header validation may read future metadata, not future price values.',
        portfolio_returns_computed=False, label_returns_computed=bool(chosen),
        original_labels_recomputed=False, trading_eligibility_modified=False)
    return out, audit
