"""Independent monotone/affine path-quality and half-core audit.

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
ARMS = ('incumbent_local67', 'monotone_path_core', 'affine_path_control')
RAW = {'monotone_path_core': 'candidate', 'affine_path_control': 'control'}
BLEND = .5
QUALITY_TOLERANCE = 1e-12
KKT_TOLERANCE = 1e-10
COUNT_FIELDS = ('valid_price_observations', 'formation_log_observations',
    'simple_return_observations', 'numerator_evaluations', 'risk_evaluations',
    'descriptive_pava_fits', 'descriptive_affine_fits', 'ranked_observations')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'original_numerator_checked',
    'current63_sample_volatility_checked', 'centered_log_path_checked',
    'monotone_projection_checked', 'pava_kkt_checked', 'constrained_affine_checked',
    'quality_branches_checked', 'positive_momentum_attenuation_checked',
    'common_rank_domain_checked', 'original_pct_ranks_checked', 'full295_price_support_checked',
    'official_session_windows_checked', 'monthend_queries_checked', 'exact_common_availability',
    'audit_checked', 'no_future_or_outcome_inputs', 'descriptive_fit_counts_checked',
    'no_predictive_fits_checked')
DELTA_FLAGS = ('passed', 'half_core_formula_checked', 'original_pct_ranks_preserved',
    'first_partial_unchanged', 'calm_rows_unchanged', 'missing_preserves_core',
    'core_weights_checked', 'no_second_rank', 'independent_delta_rebuild')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('monotone212_frozen_independent_accounts', FROZEN_HELPER)
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
                numerator=None, simple_returns=[], return_mean=None, volatility=None,
                formation_logs=[], log_mean=None, centered_logs=[], centered_mean=None, sst=None,
                blocks=[], fitted_iso=[], fitted_line=[], ols_slope=None, ols_intercept=None,
                sse_iso=None, sse_line=None, q_iso=None, q_line=None,
                q_iso_unclipped=None, q_line_unclipped=None,
                iso_quality_branch=None, line_quality_branch=None, normalized_momentum=None,
                raw_candidate=None, raw_control=None, attenuation_applied=False,
                descriptive_pava_fits=0, descriptive_affine_fits=0)


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from valid finite positive prices')
    return value


def total(items):
    try:
        return finite(math.fsum(items))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite arithmetic sum') from exc


def squared(x):
    return finite(x*x)


def sample_risk(returns):
    require(len(returns) == 63, 'Exactly63 simple returns required')
    average = finite(total(returns)/63.)
    variance = finite(total(squared(finite(r-average)) for r in returns)/62.)
    return average, finite(math.sqrt(variance))


def certify_projection(z, fitted, blocks):
    """Independent convex KKT certificate; its tolerance never changes ranks.

    For increasing fitted values, cumulative (data-fit) are nonnegative dual
    multipliers, have zero final sum, and vanish at strictly increasing edges.
    The strictly convex squared-error objective makes these conditions sufficient.
    """
    require(len(z) == len(fitted) and len(z)>0 and blocks, 'Projection dimensions')
    scale = max(1., total(abs(v) for v in z), total(abs(v) for v in fitted))
    tolerance = KKT_TOLERANCE*scale
    require(all(math.isfinite(v) for v in fitted) and
            all(a <= b for a,b in zip(fitted,fitted[1:])), 'Projection not nondecreasing')
    cursor = 0
    for block in blocks:
        lo,hi,count = block['start_index'],block['end_index'],block['count']
        require(lo==cursor and hi>=lo and count==hi-lo+1, 'Invalid projection partition')
        expected_mean = total(z[lo:hi+1])/count
        require(block['mean']==expected_mean and
                all(v==expected_mean for v in fitted[lo:hi+1]), 'Projection block mean differs')
        cursor = hi+1
    require(cursor==len(z), 'Incomplete projection partition')
    residuals = [finite(a-b) for a,b in zip(z,fitted)]
    prefixes = [total(residuals[:i+1]) for i in range(len(z))]
    require(abs(prefixes[-1])<=tolerance, 'Projection residual total not zero')
    require(all(v>=-tolerance for v in prefixes[:-1]), 'Projection KKT prefix negative')
    for i in range(len(z)-1):
        if fitted[i] < fitted[i+1]:
            require(abs(prefixes[i])<=tolerance, 'Projection complementary slackness failed')
    return dict(tolerance=tolerance,scale=scale)


def isotonic_oracle(z):
    """Whole-partition adjacent-violation scans, not producer's stack algorithm."""
    require(z and all(math.isfinite(v) for v in z), 'Finite nonempty isotonic input')
    spans = [(j,j) for j in range(len(z))]
    # Re-scan the partition after each merge. Exact equals are coalesced too.
    while True:
        means = [total(z[a:b+1])/(b-a+1) for a,b in spans]
        violation = next((k for k in range(len(spans)-1) if means[k]>=means[k+1]), None)
        if violation is None:
            break
        spans[violation:violation+2] = [(spans[violation][0],spans[violation+1][1])]
    blocks, fitted = [], []
    for lo,hi in spans:
        mean = finite(total(z[lo:hi+1])/(hi-lo+1))
        blocks.append(dict(start_index=lo,end_index=hi,count=hi-lo+1,mean=mean))
        fitted.extend([mean]*(hi-lo+1))
    certify_projection(z,fitted,blocks)
    return blocks,fitted


def quality(sse, sst, forced_branch=None):
    require(sst>0 and sse>=0, 'Invalid path sums of squares')
    if forced_branch is not None:
        return 0.,0.,forced_branch
    if sse==0:
        return 1.,1.,'zero_sse'
    raw = finite(1.-finite(sse/sst))
    require(-QUALITY_TOLERANCE<=raw<=1.+QUALITY_TOLERANCE, 'Path quality outside roundoff bound')
    return min(1.,max(0.,raw)),raw,'residual_ratio'


def path_fits(z, zmean, sst):
    blocks, fitted_iso = isotonic_oracle(z)
    coordinates = [j-(len(z)-1)/2. for j in range(len(z))]
    beta_num = total(finite(t*finite(v-zmean)) for t,v in zip(coordinates,z))
    beta_den = total(squared(t) for t in coordinates)
    beta = max(0.,finite(beta_num/beta_den))
    fitted_line = [finite(zmean+finite(beta*t)) for t in coordinates]
    sse_iso = total(squared(finite(v-y)) for v,y in zip(z,fitted_iso))
    sse_line = total(squared(finite(v-y)) for v,y in zip(z,fitted_line))
    qi, ui, bi = quality(sse_iso,sst,'single_block' if len(blocks)==1 else None)
    ql, ul, bl = quality(sse_line,sst,'zero_slope' if beta==0 else None)
    return dict(blocks=blocks,fitted_iso=fitted_iso,fitted_line=fitted_line,
                ols_slope=beta,ols_intercept=zmean,sse_iso=sse_iso,sse_line=sse_line,
                q_iso=qi,q_line=ql,q_iso_unclipped=ui,q_line_unclipped=ul,
                iso_quality_branch=bi,line_quality_branch=bl,
                descriptive_pava_fits=1,descriptive_affine_fits=1)


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
    numerator = finite(finite(p[252] / p[0]) - 1.)
    returns = [finite(finite(p[j] / p[j-1]) - 1.) for j in range(232, 295)]
    mean, volatility = sample_risk(returns)
    logs = [finite(math.log(x)) for x in p[:253]]
    log_mean = finite(total(logs)/253.)
    centered = [finite(x-log_mean) for x in logs]
    centered_mean = finite(total(centered)/253.)
    sst = total(squared(finite(x-centered_mean)) for x in centered)
    result = empty('zero_volatility' if volatility<=0 else 'zero_formation_variance')
    result.update(source_available=True, numerator=numerator,
        start_price=p[0], formation_end_price=p[252], risk_start_price=p[231], end_price=p[294],
        simple_returns=returns,return_mean=mean,volatility=volatility,formation_logs=logs,
        log_mean=log_mean,centered_logs=centered,centered_mean=centered_mean,sst=sst)
    if volatility<=0. or sst<=0.:
        return result
    fitted = path_fits(centered,centered_mean,sst)
    normalized = finite(numerator/volatility)
    attenuate = normalized>0.
    candidate = finite(normalized*fitted['q_iso']) if attenuate else normalized
    control = finite(normalized*fitted['q_line']) if attenuate else normalized
    result.update(available=True, reason='available', raw_candidate=candidate, raw_control=control,
                  normalized_momentum=normalized,attenuation_applied=attenuate,**fitted)
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
    floats = ('candidate', 'control', 'raw_candidate', 'raw_control', 'numerator',
              'volatility', 'q_iso', 'q_line')
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
            if cell['available']:
                for key in floats[2:]:
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
            formation_price_start=str(dates[0].date()) if dates else None,
            formation_price_end=str(dates[252].date()) if dates else None,
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
        formation_price_count=253,skip_sessions=42,current_return_count=63,ddof=1,
        quality_roundoff_tolerance=QUALITY_TOLERANCE,source_available_cells=sum(sources),
        source_available_by_event=sources, available_cells=available, available_by_event=counts,
        unavailable_cells=checked-available, reason_counts=reasons, valid_price_observations=295*sum(sources),
        formation_log_observations=253*sum(sources),simple_return_observations=63*sum(sources),
        numerator_evaluations=sum(sources),risk_evaluations=sum(sources),
        descriptive_pava_fits=available,descriptive_affine_fits=available,ranked_observations=2*available,
        predictive_model_fits=0,new_model_fits=0,new_predictive_model_fits=0,
        common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, event_history=history)
    return values, audit


def feature_arrays(values):
    keys = {'candidate','control','available','raw_candidate','raw_control',
            'numerator','volatility','q_iso','q_line'}
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
                *COUNT_FIELDS)
    return dict(**{key: True for key in FEATURE_FLAGS}, **{key: expected[key] for key in counters},
        predictive_model_fits=0,new_model_fits=0,new_predictive_model_fits=0,
        computes_returns=False,producer_modules_imported=False,
        numerical_comparison='exact canonical binary64 features, ranks and complete audit',
        numerical_tolerance=0.,kkt_relative_tolerance=KKT_TOLERANCE,
        kkt_scale_definition='max(1,fsum(abs(centered_logs)),fsum(abs(fitted_iso)))',
        kkt_certificate_only=True,
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
    require(all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))
                for x in rawcore[:n].flat) and all(isinstance(x, numbers.Real) and
                not isinstance(x, (bool, np.bool_)) for x in rawweight[:n].flat),
            'Original core/weights must be numeric nonboolean')
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
    return dict(**{key: True for key in DELTA_FLAGS},checked_events=n,blend=BLEND,max_absolute_delta=.26,
        changed_cells={arm:int(np.count_nonzero(value)) for arm,value in wanted.items()},
        target_available_cells={arm:int(np.isfinite(np.asarray(targets[arm])[:n]).sum()) for arm in RAW})
