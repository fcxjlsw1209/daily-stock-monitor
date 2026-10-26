"""Separate historical and current axes for the frozen226 third-slot error risk.

Pure caller inputs only. No source refits, labels, orders or returns are generated.
Statistical estimates are counted per supported query, including repeated annual
cohorts: no cache or annual approximation changes the common-history definition.
"""
from datetime import date
from pathlib import Path
import math
import numbers
import numpy as np

ARMS = ('incumbent_local67', 'joint_early_history_error_variance',
        'diagonal_early_history_error_variance')
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
INITIAL = '2016-08-17'
MIN_MONTHS, CEILING, GAP = 18, 5, .02
ARRAY_KEYS = ('candidate_risk', 'control_risk', 'candidate_effect', 'control_effect', 'ready')
COUNT_FIELDS = ('query_months', 'initial_fallback_queries', 'domain_fallback_queries',
    'history_fallback_queries', 'supported_queries', 'boundary_candidates',
    'attempted_history_months', 'complete_history_months', 'incomplete_history_months',
    'joined_label_rows', 'invalid_label_rows', 'missing_security_rows',
    'unsupported_source_rows', 'skipped_initial_months', 'error_observations',
    'security_mean_estimates', 'security_variance_estimates',
    'joint_variance_estimates', 'diagonal_variance_estimates',
    'parameter_estimates', 'cached_forecast_rows')


FROZEN_MODEL = Path(__file__).resolve().with_name('monthly_joint_error226_model.py')


def dependency_files():
    return (Path(__file__).resolve(), FROZEN_MODEL)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value):
    _require(isinstance(value, str), 'Dates must be ISO strings')
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid ISO day') from exc
    _require(parsed.isoformat() == value, 'Noncanonical ISO day')
    return value


def _month(value):
    return int(value[:4])*12 + int(value[5:7])


def _metadata(meta, width=None, initial_query_exception=False, history_axis=False):
    _require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'ends', 'symbols')),
             'Explicit dates, ends and security identities required')
    days, ends, names = (meta[k] for k in ('dates', 'ends', 'symbols'))
    _require(all(isinstance(v, list) for v in (days, ends, names))
             and len(days) == len(ends) == len(names), 'Monthly metadata axes differ')
    result = []
    for i, (day, end, symbols) in enumerate(zip(days, ends, names)):
        day, end = _day(day), _day(end)
        partial = initial_query_exception and (history_axis or i == 0) and day == INITIAL and end == '2016-08-31'
        if initial_query_exception and day == INITIAL:
            _require(partial, 'The initial partial date must retain its exact original end and position')
            if history_axis and i+1 < len(days):
                _require(days[i+1] == '2016-08-31', 'Initial partial must be followed by the formal August signal when supplied')
        _require(end > day and (_month(end) == _month(day)+1 or partial),
                 'Labels must end in next natural month, except initial partial')
        _require(i == 0 or day > days[i-1], 'Signal dates must be unique and increasing')
        initial_pair = initial_query_exception and i > 0 and (history_axis or i == 1) and days[i-1] == INITIAL and day == '2016-08-31'
        _require(i == 0 or _month(day) > _month(days[i-1]) or initial_pair,
                 'Duplicate signal month')
        _require(isinstance(symbols, list) and (width is None or len(symbols) == width)
                 and all(isinstance(s, str) and s and s == s.strip() for s in symbols)
                 and len(set(symbols)) == len(symbols), 'Exact unique security IDs and aligned width required')
        result.append((day, end, symbols))
    return result


def _boolean(value, label):
    _require(isinstance(value, (bool, np.bool_)), 'Exact boolean required: '+label)
    return bool(value)


def _number(value, label):
    _require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
             'Numeric nonboolean value required: '+label)
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid binary64 '+label) from exc


def _finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite '+label)
    return value


def _sum(values, label):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite '+label) from exc
    return _finite(result, label)


def _inputs(train, training_meta, query_meta, history_meta, source, historical_use):
    _require(isinstance(train, dict) and isinstance(source, dict), 'Caller array dictionaries required')
    required = {'offsets', 'dates', 'ends', 'train_plain', 'common_good'} | {'target_'+k for k in TARGETS}
    _require(required <= set(train), 'Original cached training fields missing')
    _require({'raw_local_linear', 'available'} <= set(source), 'Cached Local67 source fields missing')
    historical_use = np.asarray(historical_use)
    _require(historical_use.ndim == 2 and historical_use.shape[1] > 0,
             'Explicit historical use must be a matrix with positive width')
    h, width = historical_use.shape
    queries = _metadata(query_meta, width, initial_query_exception=True)
    forecast_history = _metadata(history_meta, width, initial_query_exception=True, history_axis=True)
    _require(len(forecast_history) == h, 'History metadata differs from historical use axis')
    history = _metadata(training_meta)
    offsets = [0]
    for _, _, symbols in history:
        offsets.append(offsets[-1]+len(symbols))
    provided = np.asarray(train['offsets'])
    _require(provided.shape == (len(offsets),) and provided.dtype.kind in 'iu'
             and np.array_equal(provided, np.asarray(offsets)), 'Training offsets do not match explicit identities')
    total = offsets[-1]
    dates, ends = np.asarray(train['dates']), np.asarray(train['ends'])
    _require(dates.shape == ends.shape == (total,), 'Training date axes differ')
    index = {}
    for i, (day, end, names) in enumerate(history):
        for j, symbol in enumerate(names, offsets[i]):
            _require(str(dates[j]) == day and str(ends[j]) == end, 'Training date/offset identity differs')
            key = (day, symbol, end)
            _require(key not in index, 'Duplicate training triple identity')
            index[key] = j
    data = {k: np.asarray(train[k]) for k in required if k not in ('offsets', 'dates', 'ends')}
    _require(data['train_plain'].shape == (total, 10), 'Original ten-feature training matrix required')
    _require(all(v.shape == (total,) for k, v in data.items() if k != 'train_plain'), 'Training vectors differ')
    q = h*width
    available = np.asarray(source['available'])
    _require(available.shape == (q,), 'Source availability axis differs')
    available = np.array([_boolean(v, 'source available') for v in available], dtype=bool)
    use = np.array([_boolean(v, 'historical U') for v in historical_use.reshape(-1)], dtype=bool)
    raw = np.asarray(source['raw_local_linear'])
    _require(raw.shape == (q,), 'Source Local67 forecast axis differs')
    raw = np.array([_number(v, 'raw_local_linear') for v in raw], dtype=float)
    _require(np.array_equal(np.isfinite(raw), available) and not np.isinf(raw).any(),
             'Source Local67 finite mask differs')
    return queries, forecast_history, index, data, raw, available, use, width


def _divide(total, denominator, label):
    value = _finite(total/denominator, label)
    if total != 0. and value == 0.:
        raise FloatingPointError('Nonzero '+label+' underflow')
    return value


def _square(value):
    answer = _finite(value*value, 'square')
    if value != 0. and answer == 0.:
        raise FloatingPointError('Nonzero square underflow')
    return answer


def measure(errors):
    """Canonical scalar reductions, with columns ordered fixed top2 then J."""
    matrix = np.asarray(errors)
    _require(matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[1] >= 3,
             'At least two common observations and three error columns required')
    n, width = matrix.shape
    columns = [[_finite(_number(matrix[i,j], 'error'), 'error') for i in range(n)]
               for j in range(width)]
    means = [_divide(_sum(column, 'error sum'), n, 'error mean') for column in columns]
    centered = [[_finite(v-means[j], 'centered error') for v in column]
                for j, column in enumerate(columns)]
    squares = [_sum((_square(v) for v in column), 'column sum squares') for column in centered]
    variances = [_divide(ss, n-1, 'sample variance') for ss in squares]
    full, diagonal = [], []
    for j in range(2, width):
        aggregate = [_divide(_sum((centered[0][i], centered[1][i], centered[j][i]),
                                  'trio centered sum'), 3, 'trio mean') for i in range(n)]
        full.append(_divide(_sum((_square(v) for v in aggregate), 'joint sum squares'),
                            n-1, 'joint variance'))
        diagonal.append(_divide(_sum((squares[0], squares[1], squares[j]),
                                     'diagonal sum squares'), 9*(n-1), 'diagonal variance'))
    return dict(means=means, sum_squares=squares, variances=variances,
                candidate=full, control=diagonal)


def effect_ranks(risks):
    values = [_finite(_number(v, 'risk'), 'risk') for v in risks]
    _require(values and all(v >= 0. for v in values), 'Nonnegative risks required')
    size = len(values)
    if size == 1 or all(v == values[0] for v in values):
        return [0.]*size
    order = sorted(range(size), key=lambda j: -values[j])
    result = [0.]*size
    left = 0
    while left < size:
        right = left+1
        while right < size and values[order[right]] == values[order[left]]:
            right += 1
        value = (((left+1+right)/2)-1)/(size-1)-.5
        for pos in range(left, right):
            result[order[pos]] = value
        left = right
    return result


def _domain(pre, eligible, symbols):
    order = sorted((j for j, flag in enumerate(eligible) if flag),
                   key=lambda j: (-pre[j], symbols[j]))
    top = order[:2]
    boundary = [] if len(order) < 3 else [j for j in order[2:CEILING]
                if _finite(pre[order[2]]-pre[j], 'boundary score gap') <= GAP]
    return order, top, boundary, top+boundary


def build(train, training_meta, query_meta, history_meta, history_source,
          historical_use, pre_score, eligible, *, current_available):
    """Build risks/effects only; all historical source identities are explicit.

    Current score, eligibility and current_available are (n,width); history use
    is (h,width), and history_source raw/mask vectors are flat h*width.
    current_available audits current cached forecast rows only. It does not
    filter the current boundary domain or the shared historical error cohort.
    """
    queries, forecast_history, index, data, raw, available, use, width = _inputs(
        train, training_meta, query_meta, history_meta, history_source, historical_use)
    n = len(queries)
    current = np.asarray(current_available)
    _require(current.shape == (n,width) and current.dtype.kind == 'b',
             'Current availability must be an exact boolean matrix on the current query axis')
    supplied_pre, supplied_eligible = np.asarray(pre_score), np.asarray(eligible)
    _require(supplied_pre.shape == supplied_eligible.shape == (n,width),
             'Current pre-score and original eligibility axes differ')
    pre = np.array([_finite(_number(v, 'pre-score'), 'pre-score')
                    for v in supplied_pre.reshape(-1)], dtype=float).reshape(n,width)
    mask = np.array([_boolean(v, 'current eligibility') for v in supplied_eligible.reshape(-1)],
                    dtype=bool).reshape(n,width)
    arrays = {key: np.full((n,width), np.nan) for key in ('candidate_risk','control_risk')}
    arrays.update({key: np.zeros((n,width)) for key in ('candidate_effect','control_effect')})
    arrays['ready'] = np.zeros(n, dtype=bool)
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    events = []
    for i, (day, end, symbols) in enumerate(queries):
        year = int(day[:4]); cutoff, start = f'{year-1:04d}-12-31', f'{year-6:04d}-12-31'
        order, top, boundary, B = _domain(pre[i], mask[i], symbols)
        c = dict.fromkeys(COUNT_FIELDS, 0)
        c.update(query_months=1, boundary_candidates=len(boundary),
                 cached_forecast_rows=int(current[i].sum()))
        event = dict(query_index=i, date=day, end=end, year=year, cutoff=cutoff,
            lookback_exclusive_start=start, symbols=list(symbols), eligible_order=order,
            ordered_top2=top, J=boundary, B=B, status=None, H=[], latest_label_end=None,
            history=[], security_statistics=[], trios=[], counts=c)
        if day == INITIAL:
            event['status'] = 'initial_partial'; c['initial_fallback_queries'] = 1
        elif len(order) < 3 or len(boundary) < 2:
            event['status'] = 'insufficient_candidates'; c['domain_fallback_queries'] = 1
        else:
            complete_errors = []
            for hi, (hday, hend, historical_symbols) in enumerate(forecast_history):
                if not (hday > start and hend <= cutoff):
                    continue
                if hday == INITIAL:
                    c['skipped_initial_months'] += 1
                    continue
                c['attempted_history_months'] += 1
                month = dict(date=hday, end=hend, complete=False, rows=[])
                lookup = {symbol: j for j, symbol in enumerate(historical_symbols)}
                for current_index in B:
                    symbol = symbols[current_index]
                    row = dict(symbol=symbol, source_row_id=None, training_row_id=None,
                               reason=None, prediction=None, target=None, error=None)
                    if symbol not in lookup:
                        row['reason'] = 'missing_security'; c['missing_security_rows'] += 1
                    else:
                        sid = hi*width+lookup[symbol]; row['source_row_id'] = sid
                        if not use[sid]:
                            row['reason'] = 'historical_use_false'; c['unsupported_source_rows'] += 1
                        elif not available[sid]:
                            row['reason'] = 'source_unavailable'; c['unsupported_source_rows'] += 1
                        else:
                            key = (hday, symbol, hend)
                            _require(key in index, 'Missing mature exact (signal, security, end) label: '+repr(key))
                            tid = index[key]; row['training_row_id'] = tid
                            c['joined_label_rows'] += 1
                            if not _boolean(data['common_good'][tid], 'common_good'):
                                row['reason'] = 'common_good_false'; c['invalid_label_rows'] += 1
                            else:
                                features = [_number(v, 'original training feature') for v in data['train_plain'][tid]]
                                targets = [_number(data['target_'+key][tid], 'original training target') for key in TARGETS]
                                if not all(math.isfinite(v) for v in features+targets):
                                    row['reason'] = 'nonfinite_training'; c['invalid_label_rows'] += 1
                                else:
                                    prediction, target = float(raw[sid]), targets[1]
                                    error = _finite(prediction-target, 'historical prediction error')
                                    row.update(reason='valid', prediction=prediction, target=target, error=error)
                    month['rows'].append(row)
                month['complete'] = all(row['reason']=='valid' for row in month['rows'])
                if month['complete']:
                    c['complete_history_months'] += 1
                    event['H'].append(hday); event['latest_label_end'] = hend
                    complete_errors.append([row['error'] for row in month['rows']])
                else:
                    c['incomplete_history_months'] += 1
                event['history'].append(month)
            c['error_observations'] = len(event['H'])*len(B)
            if len(event['H']) < MIN_MONTHS:
                event['status'] = 'insufficient_history'; c['history_fallback_queries'] = 1
            else:
                result = measure(complete_errors)
                candidate_effect, control_effect = (effect_ranks(result[key]) for key in ('candidate','control'))
                event['status'] = 'ready'; c['supported_queries'] = 1; arrays['ready'][i] = True
                for j, current_index in enumerate(B):
                    event['security_statistics'].append(dict(index=current_index, symbol=symbols[current_index],
                        error_mean=result['means'][j], error_sum_squares=result['sum_squares'][j],
                        sample_variance=result['variances'][j]))
                for j, current_index in enumerate(boundary):
                    indices = top+[current_index]
                    event['trios'].append(dict(third_index=current_index, symbol=symbols[current_index],
                        indices=indices, symbols=[symbols[k] for k in indices],
                        candidate_risk=result['candidate'][j], control_risk=result['control'][j],
                        candidate_effect=candidate_effect[j], control_effect=control_effect[j]))
                    arrays['candidate_risk'][i,current_index] = result['candidate'][j]
                    arrays['control_risk'][i,current_index] = result['control'][j]
                    arrays['candidate_effect'][i,current_index] = candidate_effect[j]
                    arrays['control_effect'][i,current_index] = control_effect[j]
                c['security_mean_estimates'] = c['security_variance_estimates'] = len(B)
                c['joint_variance_estimates'] = c['diagonal_variance_estimates'] = len(boundary)
                c['parameter_estimates'] = 2*len(B)+2*len(boundary)
        for key in COUNT_FIELDS:
            counts[key] += c[key]
        events.append(event)
    audit = dict(method='joint_asof_error_boundary_variance', arms=list(ARMS),
        query_months=n, history_months=len(forecast_history), separate_history_axis=True,
        symbols_per_event=width, min_months=MIN_MONTHS,
        lookback_calendar_years=5, ceiling=CEILING, gap=GAP,
        historical_errors='cached raw_local_linear minus unchanged target_log_excess',
        source_forecasts_refitted=False, labels_recomputed=False, local_available_filtered=False,
        historical_use_explicit=True, common_complete_history=True, counts=counts, event_history=events,
        source_model_refits=0, new_ridge_fits=0, new_tree_fits=0,
        new_statistical_estimation_queries=counts['supported_queries'],
        parameter_estimates=counts['parameter_estimates'])
    return dict(arrays=arrays, audit=audit)
