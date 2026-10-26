"""Independent source-window, scalar break search, rank-slot and account audit.

No producer kernel or strategy engine is imported. There is no data-loading or
execution entry point. Historical log changes here are feature observations,
not portfolio returns. Caller is responsible for frozen input/PLAN bindings.
"""
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'breakpoint_core', 'midpoint_core')
RAW = dict(breakpoint_core='candidate', midpoint_core='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .5
PENALTY = 2. * math.log(252.)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('break177_accounting', ACCOUNTING)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('break177_outcome', OUTCOME_HELPER)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    require(value.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected frozen outcome arms')
    value.ARMS = ARMS
    result = value.audit_outcomes(px, opening, orders, exports, results, meta)
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12, rtol=1e-11):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, label + ': shape')
    require(np.array_equal(np.isnan(a), np.isnan(b)) and not np.isinf(a).any()
            and not np.isinf(b).any(), label + ': nonfinite')
    good = np.isfinite(b)
    require(np.all(np.abs(a[good] - b[good]) <= atol + rtol * np.abs(b[good])), label + ': numbers')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': mapping')
        for k, v in expected.items():
            require(k in actual, label + ': missing ' + k)
            record_check(actual[k], v, label + '/' + k)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    d = date.fromisoformat(value)
    require(d.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(d)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Exact adj panel required')
    f = panel['adj']
    require(isinstance(f, pd.DataFrame) and isinstance(f.index, pd.DatetimeIndex), 'Daily frame required')
    require(f.index.is_unique and f.index.is_monotonic_increasing and not f.index.hasnans
            and f.index.tz is None and f.index.equals(f.index.normalize()), 'Invalid panel index')
    require(f.columns.is_unique and all(isinstance(s, str) and s.strip() == s and s for s in f.columns),
            'Invalid panel symbols')
    require(isinstance(meta, dict) and 'dates' in meta and 'symbols' in meta
            and len(meta['dates']) == len(meta['symbols']), 'Event metadata differs')
    sessions = [iso(d) for d in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions)
            and all(a < b for a, b in zip(sessions, sessions[1:])), 'Official session order')
    positions = {d: i for i, d in enumerate(sessions)}
    events, width = [], 0
    for d, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(d)
        require(day in positions and (not events or day > events[-1][0]), 'Event session/order')
        require(isinstance(symbols, (list, tuple)) and symbols and len(set(symbols)) == len(symbols)
                and all(isinstance(s, str) and s.strip() == s and s for s in symbols), 'Event identities')
        require(not events or width == len(symbols), 'Changing width')
        width = len(symbols); events.append((day, list(symbols), positions[day]))
    return sessions, events, width


def empty(reason):
    return dict(baseline=float('nan'), candidate=float('nan'), control=float('nan'),
        baseline_available=False, baseline_reason=reason, available=False, reason=reason,
        recent_volatility=None, endpoint_growth=None, full_mean=None, sse0=None,
        formation_volatility=None, numerically_constant=False, split_tie_tolerance=None,
        splits=[], candidate_fit=None, control_fit=None, constant_mean_estimates=0,
        split_mean_estimates=0, candidate_breaks=0, control_breaks=0)


def failure(result, reason):
    result['reason'] = reason
    if not result['baseline_available']:
        result['baseline_reason'] = reason
    return result


def scalar_log(a, b):
    # Independent scalar evaluation; avoids subtracting large close logarithms.
    r = (b - a) / a
    return math.log1p(r) if math.isfinite(r) and r > -1. else math.log(b) - math.log(a)


def mean_sse(values):
    mu = math.fsum(values) / len(values)
    return mu, math.fsum((x - mu) ** 2 for x in values)


def split_oracle(values, k):
    left, lsse = mean_sse(values[:k]); right, rsse = mean_sse(values[k:])
    return dict(k=k, pre_mean=left, post_mean=right, sse=lsse + rsse)


def fit_oracle(split, sse0, mean, growth, baseline, volatility, numerically_constant=False,
               selection_rule='minimum_sse_relative_tie_earliest'):
    perfect = sse0 > 0. and split['sse'] == 0.
    statistic = None if numerically_constant or perfect else (0. if sse0 == 0. else
        252. * (math.log(sse0) - math.log(split['sse'])))
    accepted = not numerically_constant and (perfect or (statistic is not None and statistic > PENALTY))
    predicted = split['post_mean'] if accepted else mean
    forecast = growth
    if accepted:
        try:
            forecast = math.expm1(252. * predicted)
        except OverflowError:
            forecast = None
    raw = baseline if not accepted else (forecast / volatility if forecast is not None else float('nan'))
    if forecast is not None and not math.isfinite(forecast):
        forecast, raw = None, float('nan')
    fit = dict(split, selection_rule=selection_rule, statistic=statistic, perfect_fit=perfect, accepted=accepted,
               predicted_mean=predicted, forecast_growth=forecast)
    return fit, raw


def measure_oracle(formation, current):
    require(np.shape(formation) == (253,) and np.shape(current) == (64,), '253/64 prices required')
    result = empty('available')
    try:
        cur = [float(v) for v in current]
    except (ValueError, TypeError, OverflowError):
        return failure(result, 'nonnumeric_current_price')
    if any(not math.isfinite(v) or v <= 0. for v in cur):
        return failure(result, 'invalid_current_price')
    changes = [b / a - 1. for a, b in zip(cur[:-1], cur[1:])]
    if any(not math.isfinite(v) for v in changes):
        return failure(result, 'nonfinite_current_return')
    avg = math.fsum(changes) / 63.
    try:
        variance = math.fsum((v - avg) ** 2 for v in changes) / 62.
    except OverflowError:
        return failure(result, 'zero_current_volatility')
    vol = math.sqrt(variance)
    if not math.isfinite(vol) or vol <= 0.:
        return failure(result, 'zero_current_volatility')
    result['recent_volatility'] = vol
    try:
        first, last = float(formation[0]), float(formation[-1])
    except (ValueError, TypeError, OverflowError):
        return failure(result, 'nonnumeric_endpoint_price')
    if any(not math.isfinite(v) or v <= 0. for v in (first, last)):
        return failure(result, 'invalid_endpoint_price')
    growth = last / first - 1.
    baseline = growth / vol
    if not math.isfinite(baseline):
        return failure(result, 'nonfinite_baseline')
    result.update(endpoint_growth=growth, baseline=baseline, baseline_available=True, baseline_reason='available')
    if vol <= 1e-12:
        return failure(result, 'volatility_floor')
    try:
        p = [float(v) for v in formation]
    except (ValueError, TypeError, OverflowError):
        return failure(result, 'nonnumeric_formation_price')
    if any(not math.isfinite(v) or v <= 0. for v in p):
        return failure(result, 'invalid_formation_price')
    changes = [scalar_log(a, b) for a, b in zip(p[:-1], p[1:])]
    mu, sse0 = mean_sse(changes)
    formation_vol = math.sqrt(sse0 / 251.)
    constant = formation_vol <= 1e-12
    splits = [split_oracle(changes, k) for k in range(63, 190)]
    tie_tolerance = 1e-12 * sse0
    minimum = min(row['sse'] for row in splits)
    best = splits[0] if constant else next(row for row in splits if row['sse'] <= minimum + tie_tolerance)
    cf, candidate = fit_oracle(best, sse0, mu, growth, baseline, vol, constant,
        'canonical_inactive_k63' if constant else 'minimum_sse_relative_tie_earliest')
    xf, control = fit_oracle(splits[126 - 63], sse0, mu, growth, baseline, vol, constant, 'fixed_midpoint')
    result.update(full_mean=mu, sse0=sse0, splits=splits, candidate_fit=cf, control_fit=xf,
        formation_volatility=formation_vol, numerically_constant=constant, split_tie_tolerance=tie_tolerance,
        constant_mean_estimates=1, split_mean_estimates=127,
        candidate_breaks=int(cf['accepted']), control_breaks=int(xf['accepted']))
    if not math.isfinite(candidate) or not math.isfinite(control):
        return failure(result, 'nonfinite_forecast')
    result.update(candidate=candidate, control=control, available=True)
    return result


def json_cell(result, symbol, days):
    cell = dict(result, symbol=symbol)
    for key in ('baseline', 'candidate', 'control'):
        if not math.isfinite(cell[key]): cell[key] = None
    def dated(row):
        if row is None: return None
        return dict(row, break_after_date=str(days[row['k']].date()),
                    post_first_return_date=str(days[row['k'] + 1].date()))
    cell['splits'] = [dated(row) for row in result['splits']]
    cell['candidate_fit'], cell['control_fit'] = dated(result['candidate_fit']), dated(result['control_fit'])
    return cell


def independent_features(panel, meta, calendar):
    sessions, events, width = schema(panel, meta, calendar)
    frame = panel['adj']
    values = {k: np.full((len(events), width), np.nan) for k in ('baseline', 'candidate', 'control')}
    values['available'] = np.zeros((len(events), width), dtype=bool)
    history, reasons = [], {}
    for i, (day, names, t) in enumerate(events):
        fd = sessions[t - 294:t - 41] if t >= 294 else []
        vd = sessions[t - 63:t + 1] if t >= 63 else []
        mf = [str(d.date()) for d in fd if d not in frame.index]
        mv = [str(d.date()) for d in vd if d not in frame.index]
        cells = []
        for j, symbol in enumerate(names):
            if t < 294: cell = empty('insufficient_history')
            elif symbol not in frame.columns: cell = empty('missing_symbol')
            elif mv: cell = empty('missing_volatility_session')
            elif fd[0] not in frame.index or fd[-1] not in frame.index: cell = empty('missing_endpoint_session')
            else:
                form = [frame.at[d, symbol] if d in frame.index else float('nan') for d in fd]
                cell = measure_oracle(form, [frame.at[d, symbol] for d in vd])
                if mf and cell['baseline_available']:
                    cell = failure(cell, 'missing_formation_session')
            for key in ('baseline', 'candidate', 'control'): values[key][i, j] = cell[key]
            values['available'][i, j] = cell['available']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(json_cell(cell, symbol, fd))
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            formation_start=str(fd[0].date()) if fd else None, formation_end=str(fd[-1].date()) if fd else None,
            volatility_start=str(vd[0].date()) if vd else None, volatility_end=str(vd[-1].date()) if vd else None,
            missing_formation_sessions=mf, missing_volatility_sessions=mv, cells=cells))
    audit = dict(events=len(events), symbols_per_event=width, available_cells=int(values['available'].sum()),
        skip_sessions=42, formation_price_count=253, formation_return_count=252,
        current_price_count=64, current_return_count=63, minimum_segment=63, split_candidates=127,
        split_min=63, split_max=189, midpoint=126, penalty=PENALTY,
        volatility_floor=1e-12, formation_standard_deviation_floor=1e-12, split_sse_relative_tolerance=1e-12,
        baseline_finite_cells=int(np.isfinite(values['baseline']).sum()),
        unavailable_cells=int((~values['available']).sum()), reason_counts=reasons, event_history=history,
        predictive_model_fits=0, common_availability=True, portfolio_returns_calculated=False,
        ranking_performed=False, trading_eligibility_modified=False)
    for key in ('constant_mean_estimates', 'split_mean_estimates', 'candidate_breaks', 'control_breaks'):
        audit[key] = sum(c[key] for e in history for c in e['cells'])
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'baseline', 'candidate', 'control', 'available'}, 'Value keys differ')
    a = {k: np.asarray(v) for k, v in values.items()}
    use = a['available']
    require(use.ndim == 2 and use.dtype == bool, 'Availability must be a boolean matrix')
    for key in ('baseline', 'candidate', 'control'):
        require(a[key].shape == use.shape and np.issubdtype(a[key].dtype, np.floating)
                and not np.isinf(a[key]).any(), 'Raw type/shape/infinity')
        require(np.all(np.isfinite(a[key])[use]), 'Available requires baseline and both arms')
        if key != 'baseline':
            require(np.array_equal(np.isfinite(a[key]), use), 'Candidate/control support differs')
    return a


def ranks(row, mask):
    out = np.zeros(len(row), dtype=float)
    ids = np.flatnonzero(mask)
    for j in ids:
        less = sum(row[k] < row[j] for k in ids)
        equal = sum(row[k] == row[j] for k in ids)
        out[j] = (less + (equal + 1.) / 2.) / len(ids) - .5
    return out


def verify_features(panel, meta, calendar, values, audit):
    got = arrays(values); wanted, expected = independent_features(panel, meta, calendar)
    require(np.array_equal(got['available'], wanted['available']), 'Independent support differs')
    for key in ('baseline', 'candidate', 'control'):
        close(got[key], wanted[key], 'Independent ' + key)
        for a, b in zip(got[key], wanted[key]):
            require(np.array_equal(ranks(a, np.isfinite(a)), ranks(b, np.isfinite(b))),
                    'Independent exact rank differs: ' + key)
    record_check(audit, expected)
    result = dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
        checked_cells=int(wanted['available'].size), available_cells=expected['available_cells'],
        unavailable_cells=expected['unavailable_cells'], independent_scalar_oracle=True,
        independent_scalar_rebuild=True, exact_common_availability=True, no_future_or_outcome_inputs=True,
        audit_checked=True, breakpoint_search_checked=True, fixed_midpoint_control_checked=True,
        formation_and_volatility_dates_checked=True, endpoint_fallback_checked=True,
        numerical_constant_fallback_checked=True,
        raw_rank_parity_checked=True, predictive_model_fits=0, computes_returns=False,
        portfolio_returns_calculated=False)
    for key in ('constant_mean_estimates', 'split_mean_estimates', 'candidate_breaks', 'control_breaks'):
        result[key] = expected[key]
    return result


def independent_deltas(values, original, dose=DOSE):
    require(type(dose) in (int, float) and dose == DOSE, 'Fixed half-core dose required')
    a = arrays(values); n, width = a['available'].shape
    core = np.asarray(original['core'], dtype=float)
    calm = np.asarray(original['state_calm10'])
    weights = np.asarray(original['core_weight'], dtype=float)
    require(core.shape == (n, width) and np.isfinite(core).all(), 'Core shape/finite')
    require(calm.shape == (n,) and np.isin(calm, [0, 1]).all(), 'Calm shape/binary')
    require(weights.shape == (n,) and np.isfinite(weights).all(), 'Core weight shape/finite')
    result = {arm: np.zeros((n, width)) for arm in RAW}
    for i in range(n):
        if calm[i]: continue
        require(weights[i] == .52, 'Noncalm weight must remain .52')
        mask = np.isfinite(a['baseline'][i])
        require(np.array_equal(core[i], .52 * ranks(a['baseline'][i], mask)), 'Original noncalm core rank parity')
        if i == 0 or not np.array_equal(mask, a['available'][i]): continue
        for arm, key in RAW.items():
            result[arm][i] = dose * (weights[i] * ranks(a[key][i], mask) - core[i])
    return result


def verify_deltas(values, deltas, original, dose=DOSE):
    require(set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, original, dose)
    for arm, target in expected.items():
        # The inputs/ranks are exact; only the single frozen half-slot expression remains.
        require(np.array_equal(np.asarray(deltas[arm]), target), 'Half-core delta differs: ' + arm)
    a = arrays(values)
    ready = np.all(np.isfinite(a['baseline']) == a['available'], axis=1)
    return dict(passed=True, checked_events=a['available'].shape[0], dose=DOSE,
        full_candidate_universe=True, common_available_only=True, initial_delta_zero=True,
        no_eligibility_or_model_filter=True, noncalm_core_parity_checked=True,
        baseline_rank_domain_checked=True, whole_row_readiness_checked=True, calm_rows_neutral=True,
        fixed_half_core_replacement=True, ranking_before_eligibility=True,
        ready_rows=int(ready.sum()), changed_cells={arm: int(np.count_nonzero(v)) for arm, v in expected.items()})
