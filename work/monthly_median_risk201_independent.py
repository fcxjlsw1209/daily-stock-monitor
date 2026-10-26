"""Independent median/mean absolute-deviation risk and half-core audit.

No producer calculation or real-data command is imported. Only the pinned,
frozen independent portfolio-accounting adapter is loaded for outcome audits.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'median_mad_core', 'mean_mad_control')
RAW = {'median_mad_core': 'candidate', 'mean_mad_control': 'control'}
BLEND = .5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('median201_frozen_independent_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | {Path(__file__).resolve()}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Panel keys differ')
    frame = panel['adj']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Adjusted frame type')
    idx = frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
            idx.equals(idx.normalize()), 'Invalid adjusted daily axis')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
            for s in frame.columns), 'Invalid adjusted symbol labels')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    positions = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for value, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(value)
        require(day in positions and (not queries or day > queries[-1][0]), 'Event session/order')
        t = positions[day]
        if queries or value != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event symbol')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


def empty(reason):
    return dict(source_available=False, available=False, reason=reason,
                start_price=None, formation_end_price=None, risk_start_price=None, end_price=None,
                numerator=None, median_return=None, median_deviation=None, mean_deviation=None,
                simple_returns=[], absolute_deviations=[], raw_candidate=None, raw_control=None,
                dispersion_estimates=0)


def measure_oracle(prices):
    require(np.shape(prices) == (295,), 'Exactly295 stored adjusted prices required')
    raw = list(prices)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return empty('nonnumeric_adjusted_price')
    try:
        p = [float(x) for x in raw]
    except (TypeError, ValueError, OverflowError):
        return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(x) or x <= 0 for x in p):
        return empty('invalid_adjusted_price')
    numerator = p[252] / p[0] - 1.
    returns = [p[j] / p[j-1] - 1. for j in range(232, 295)]
    if not math.isfinite(numerator) or not all(math.isfinite(r) for r in returns):
        raise FloatingPointError('Nonfinite gain or simple return from valid prices')
    median_return = sorted(returns)[31]
    deviations = [abs(r - median_return) for r in returns]
    if not all(math.isfinite(d) for d in deviations):
        raise FloatingPointError('Nonfinite centered absolute deviation')
    median_deviation = sorted(deviations)[31]
    try:
        mean_deviation = math.fsum(deviations) / 63.
    except OverflowError as exc:
        raise FloatingPointError('Nonfinite sum of absolute deviations') from exc
    if not math.isfinite(mean_deviation):
        raise FloatingPointError('Nonfinite mean absolute deviation')
    result = empty('zero_dispersion')
    result.update(source_available=True, numerator=numerator, median_return=median_return,
                  median_deviation=median_deviation, mean_deviation=mean_deviation,
                  start_price=p[0], formation_end_price=p[252], risk_start_price=p[231], end_price=p[294],
                  simple_returns=returns, absolute_deviations=deviations, dispersion_estimates=2)
    if median_deviation <= 0. or mean_deviation <= 0.:
        return result
    candidate, control = numerator / median_deviation, numerator / mean_deviation
    if not math.isfinite(candidate) or not math.isfinite(control):
        raise FloatingPointError('Nonfinite risk-adjusted ratio')
    result.update(available=True, reason='available', raw_candidate=candidate, raw_control=control)
    return result


def original_pct_ranks(raw, available):
    result = np.full(np.shape(raw), np.nan)
    for i in range(len(result)):
        members = [j for j, good in enumerate(available[i]) if good]
        n = len(members)
        for j in members:
            less = sum(raw[i, k] < raw[i, j] for k in members)
            equal = sum(raw[i, k] == raw[i, j] for k in members)
            result[i, j] = (less + (equal + 1) / 2.) / n - .5
    return result


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    shape = (len(queries), width)
    floats = ('candidate', 'control', 'raw_candidate', 'raw_control')
    values = {key: np.full(shape, np.nan) for key in floats}
    values['available'] = np.zeros(shape, bool)
    history, reasons, counts, sources = [], {}, [], []
    frame = panel['adj']
    for i, (day, names, t) in enumerate(queries):
        dates = sessions[t-294:t+1] if t >= 294 else []
        missing = [str(d.date()) for d in dates if d not in frame.index]
        cells = []
        for j, symbol in enumerate(names):
            reason = ('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                      else 'missing_official_session' if missing else None)
            cell = empty(reason) if reason else measure_oracle([frame.at[d, symbol] for d in dates])
            cell.update(symbol=symbol, date=str(day.date()), candidate=None, control=None)
            values['available'][i, j] = cell['available']
            for key in ('raw_candidate', 'raw_control'):
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        mask = [cell['available'] for cell in cells]
        source_mask = [cell['source_available'] for cell in cells]
        counts.append(sum(mask))
        sources.append(sum(source_mask))
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=str(dates[0].date()) if dates else None, price_end=str(dates[-1].date()) if dates else None,
            numerator_start=str(dates[0].date()) if dates else None,
            numerator_end=str(dates[252].date()) if dates else None,
            risk_price_start=str(dates[231].date()) if dates else None,
            risk_price_end=str(dates[294].date()) if dates else None,
            missing_official_sessions=missing,
            source_available=source_mask, available=mask, source_available_count=sum(source_mask),
            available_count=sum(mask), cells=cells))
    for key in ('candidate', 'control'):
        values[key] = original_pct_ranks(values['raw_' + key], values['available'])
    for i, event in enumerate(history):
        for j, cell in enumerate(event['cells']):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i, j]), control=float(values['control'][i, j]))
    available, checked = sum(counts), int(values['available'].size)
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=checked, price_count=295,
        skip_sessions=42, current_return_count=63, source_available_cells=sum(sources),
        source_available_by_event=sources, available_cells=available, available_by_event=counts,
        unavailable_cells=checked-available, reason_counts=reasons, valid_price_observations=295*sum(sources),
        simple_return_observations=63*sum(sources), dispersion_estimates=2*sum(sources),
        ratio_observations=2*available, ranked_score_observations=2*available,
        predictive_model_fits=0, new_model_fits=0, common_availability=True, ranking_performed=True,
        ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, event_history=history)
    return values, audit


def feature_arrays(values):
    keys = {'candidate', 'control', 'available', 'raw_candidate', 'raw_control'}
    require(isinstance(values, dict) and set(values) == keys, 'Feature keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean availability required')
    for key in keys-{'available'}:
        v = a[key]
        require(v.shape == mask.shape and np.issubdtype(v.dtype, np.floating), 'Float feature matrix required: '+key)
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v), mask), 'Common availability differs: '+key)
    for key in ('candidate', 'control'):
        require(np.array_equal(a[key], original_pct_ranks(a['raw_' + key], mask), equal_nan=True),
                'Original percentile ranks differ')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = feature_arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, limit)
    for key in wanted:
        require(np.array_equal(actual[key], wanted[key], equal_nan=True), 'Independent feature differs: '+key)
    record_check(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'source_available_cells',
        'source_available_by_event', 'available_cells', 'available_by_event', 'unavailable_cells',
                'valid_price_observations', 'simple_return_observations', 'dispersion_estimates',
                'ratio_observations', 'ranked_score_observations')
    return dict(passed=True, **{key: expected[key] for key in counters}, independent_scalar_rebuild=True,
        original_numerator_checked=True, median_center_checked=True, median_mean_deviations_checked=True,
        common_rank_domain_checked=True, original_pct_ranks_checked=True, full295_price_support_checked=True,
        official_session_windows_checked=True, monthend_queries_checked=True, exact_common_availability=True,
        audit_checked=True, no_future_or_outcome_inputs=True, predictive_model_fits=0, new_model_fits=0,
        computes_returns=False, numerical_comparison='exact binary64 ratios, sorted medians, fsum means, ranks and audit',
        outcome_adapter_sha256=FROZEN_HELPER_SHA)


def original_prefix(original, n, width):
    require(isinstance(original, dict) and all(k in original for k in ('core','core_weight','state_calm10')),
            'Original core/state fields required')
    rawcore = np.asarray(original['core'])
    require(rawcore.ndim == 2 and rawcore.shape[1] == width and len(rawcore) >= n, 'Original core shape')
    full = len(rawcore)
    state = np.asarray(original['state_calm10'])
    require(state.shape == (full,), 'Original calm shape')
    require(all(isinstance(x, (bool, np.bool_)) for x in state[:n]), 'Original calm must be boolean')
    calm = np.asarray(state[:n], dtype=bool)
    rawweight = np.asarray(original['core_weight'])
    require(rawweight.shape in ((full,), (full,1), (full,width)), 'Original core weight shape')
    if rawweight.ndim == 1:
        rawweight = rawweight[:,None]
    try:
        weight = np.broadcast_to(np.asarray(rawweight[:n], dtype=float), (n,width)).copy()
        core = np.asarray(rawcore[:n], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Nonnumeric original core/weights') from exc
    require(np.isfinite(core).all() and np.isfinite(weight).all(), 'Nonfinite original core/weights')
    require(np.array_equal(weight, np.broadcast_to(np.where(calm,.5,.52)[:,None], (n,width))),
            'Core weights must exactly follow original calm')
    require(np.all(np.abs(core) <= .5*weight), 'Original weighted core outside rank bounds')
    return core, weight, full


def independent_deltas(values, original, targets, blend=BLEND):
    require(type(blend) in (float,int) and blend == BLEND, 'Fixed half-core blend required')
    a = feature_arrays(values)
    n, width = a['available'].shape
    core, weight, full = original_prefix(original,n,width)
    require(isinstance(targets,dict) and set(targets) == set(RAW), 'Target arms differ')
    result = {}
    for arm, raw in RAW.items():
        target = np.asarray(targets[arm])
        require(target.ndim == 2 and target.shape in ((n,width),(full,width)) and
                np.issubdtype(target.dtype,np.floating), 'Full/prefix floating targets required')
        require(not np.isinf(target).any(), 'Infinite target rejected')
        if len(target)>n:
            require(np.isnan(target[n:]).all(), 'Unfilled target suffix must remain NaN')
        wanted = a[raw].copy()
        calm = np.asarray(original['state_calm10'][:n], bool)
        wanted[calm] = np.nan
        if n:
            wanted[0] = np.nan
        require(np.array_equal(target[:n], wanted, equal_nan=True), 'Target prefix, calm or first partial differs')
        delta = np.zeros((n,width),float)
        for i in range(n):
            for j in range(width):
                if math.isfinite(float(wanted[i,j])):
                    delta[i,j] = blend*(float(weight[i,j])*float(wanted[i,j])-float(core[i,j]))
        require(np.isfinite(delta).all() and np.all(np.abs(delta) <= .26), 'Half-core delta exceeds bound')
        result[arm] = delta
    return result


def verify_deltas(values, arrays, targets, deltas, blend=BLEND):
    require(isinstance(deltas,dict) and set(deltas) == set(RAW), 'Delta arms differ')
    wanted = independent_deltas(values, arrays, targets, blend)
    for arm, expected in wanted.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == expected.shape and np.issubdtype(actual.dtype,np.floating) and
                np.isfinite(actual).all() and np.array_equal(actual,expected), 'Exact half-core delta differs: '+arm)
    n = np.shape(values['available'])[0]
    return dict(passed=True,checked_events=n,blend=BLEND,max_absolute_delta=.26,
        half_core_formula_checked=True,original_pct_ranks_preserved=True,first_partial_unchanged=True,
        calm_rows_unchanged=True,
        missing_preserves_core=True,core_weights_checked=True,no_second_rank=True,independent_delta_rebuild=True,
        changed_cells={arm:int(np.count_nonzero(value)) for arm,value in wanted.items()},
        target_available_cells={arm:int(np.isfinite(np.asarray(targets[arm])[:n]).sum()) for arm in RAW})
