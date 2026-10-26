"""Annual exact-ID versus pooled calibration of cached local forecast increments.

Pure caller arrays only: no source refitting, labels, ranks, orders or returns.
Pooled and own supported convex coefficients are counted calibration fits.
"""
from datetime import date
from pathlib import Path
import math
import numbers
import numpy as np

ARMS = ('incumbent_local67', 'own_increment_calibration', 'pooled_increment_control')
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
MIN_MONTHS, MIN_ROWS = 18, 2
INITIAL = '2016-08-17'
COUNT_FIELDS = ('annual_cohorts', 'pooled_fits', 'cold_start_years',
    'zero_denominator_years', 'clipped_pooled_fits', 'own_history_groups',
    'own_fits', 'cold_start_groups', 'zero_denominator_groups', 'clipped_own_fits',
    'parameter_estimates', 'joined_label_rows', 'invalid_label_rows',
    'small_month_rows', 'skipped_initial_rows', 'calibration_months',
    'calibration_rows', 'own_history_rows', 'cached_forecast_rows')


def dependency_files():
    return (Path(__file__).resolve(),)


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


def _metadata(meta, width=None, initial_query_exception=False):
    _require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'ends', 'symbols')),
             'Explicit dates, ends and security identities required')
    days, ends, names = (meta[k] for k in ('dates', 'ends', 'symbols'))
    _require(all(isinstance(v, list) for v in (days, ends, names))
             and len(days) == len(ends) == len(names), 'Monthly metadata axes differ')
    result = []
    for i, (day, end, symbols) in enumerate(zip(days, ends, names)):
        day, end = _day(day), _day(end)
        partial = initial_query_exception and i == 0 and day == INITIAL and end == '2016-08-31'
        _require(end > day and (_month(end) == _month(day)+1 or partial),
                 'Labels must end in next natural month, except initial partial')
        _require(i == 0 or day > days[i-1], 'Signal dates must be unique and increasing')
        initial_pair = initial_query_exception and i == 1 and days[0] == INITIAL and day == '2016-08-31'
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


def _inputs(train, training_meta, query_meta, source, historical_use):
    _require(isinstance(train, dict) and isinstance(source, dict), 'Caller array dictionaries required')
    required = {'offsets', 'dates', 'ends', 'train_plain', 'common_good'} | {'target_'+k for k in TARGETS}
    _require(required <= set(train), 'Original cached training fields missing')
    _require({'raw_local_linear', 'raw_source_log', 'available'} <= set(source), 'Cached Local67 source fields missing')
    historical_use = np.asarray(historical_use)
    _require(historical_use.ndim == 2 and historical_use.shape[1] > 0,
             'Explicit historical original U must be a matrix with positive width')
    n, width = historical_use.shape
    queries = _metadata(query_meta, width, initial_query_exception=True)
    _require(len(queries) == n, 'Query metadata differs from historical U axis')
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
    q = n*width
    available = np.asarray(source['available'])
    _require(available.shape == (q,), 'Source availability axis differs')
    available = np.array([_boolean(v, 'source available') for v in available], dtype=bool)
    use = np.array([_boolean(v, 'historical U') for v in historical_use.reshape(-1)], dtype=bool)
    raw = np.asarray(source['raw_local_linear'])
    _require(raw.shape == (q,), 'Source Local67 forecast axis differs')
    raw = np.array([_number(v, 'raw_local_linear') for v in raw], dtype=float)
    _require(np.array_equal(np.isfinite(raw), available) and not np.isinf(raw).any(),
             'Source Local67 finite mask differs')
    pooled = np.asarray(source['raw_source_log'])
    _require(pooled.shape == (q,), 'Source pooled forecast axis differs')
    pooled = np.array([_number(v, 'raw_source_log') for v in pooled], dtype=float)
    _require(np.array_equal(np.isfinite(pooled), available) and not np.isinf(pooled).any(),
             'Source pooled finite mask differs')
    return queries, index, data, raw, pooled, available, use, width


def product(a, b):
    a = _finite(_number(a, 'product operand'), 'product operand')
    b = _finite(_number(b, 'product operand'), 'product operand')
    value = _finite(a*b, 'product')
    if a != 0. and b != 0. and value == 0.:
        raise FloatingPointError('Nonzero product underflow')
    return value


def mean(total, count):
    value = _finite(total/count, 'monthly mean')
    if total != 0. and value == 0.:
        raise FloatingPointError('Nonzero monthly mean underflow')
    return value


def coefficient(numerator, denominator, months):
    numerator = _finite(_number(numerator, 'numerator'), 'numerator')
    denominator = _finite(_number(denominator, 'denominator'), 'denominator')
    _require(type(months) is int and months >= 0 and denominator >= 0., 'Invalid calibration support')
    if months < MIN_MONTHS:
        return None, False, 'cold_start'
    if denominator == 0.:
        return None, False, 'zero_denominator'
    clipped = numerator < 0. or numerator > denominator
    if numerator <= 0.:
        return 0., clipped, 'calibrated'
    if numerator >= denominator:
        return 1., clipped, 'calibrated'
    value = _finite(numerator/denominator, 'interior convex coefficient')
    if value == 0.:
        raise FloatingPointError('Interior positive coefficient underflow')
    return value, clipped, 'calibrated'


def blend(prediction, pooled, alpha):
    prediction = _finite(_number(prediction, 'prediction'), 'prediction')
    pooled = _finite(_number(pooled, 'pooled prediction'), 'pooled prediction')
    alpha = _finite(_number(alpha, 'alpha'), 'alpha')
    _require(0. <= alpha <= 1., 'Convex alpha required')
    if alpha == 1. or prediction == pooled:
        return prediction
    if alpha == 0.:
        return pooled
    difference = _finite(prediction-pooled, 'current increment')
    return _finite(pooled+product(alpha, difference), 'calibrated prediction')


def build_forecasts(train, training_meta, query_meta, source, historical_use):
    queries, index, data, raw, pooled, available, use, width = _inputs(
        train, training_meta, query_meta, source, historical_use)
    names = [symbol for _, _, symbols in queries for symbol in symbols]
    out = {key: raw.copy() for key in ('candidate', 'control')}
    counts = {key: 0 for key in COUNT_FIELDS}
    counts['cached_forecast_rows'] = int(available.sum())
    annual = []
    for year in sorted({int(day[:4]) for day, _, _ in queries}):
        cutoff, start = f'{year-1:04d}-12-31', f'{year-6:04d}-12-31'
        current = [i*width+j for i, (day, _, _) in enumerate(queries)
                   if int(day[:4]) == year for j in range(width) if available[i*width+j]]
        securities = sorted({names[j] for j in current})
        row = dict(year=year, cutoff=cutoff, lookback_exclusive_start=start,
            prediction_rows=len(current), queried_security_count=len(securities),
            joined_label_rows=0, invalid_label_rows=0, small_month_rows=0,
            skipped_initial_rows=0, calibration_months=0, calibration_rows=0,
            latest_label_end=None, status=None, numerator=0., denominator=0.,
            alpha=None, clipped=False, own_fits=0, cold_start_groups=0,
            zero_denominator_groups=0, clipped_own_fits=0, parameter_estimates=0,
            months=[], groups=[])
        history = []
        for i, (day, end, symbols) in enumerate(queries):
            if not (day > start and end <= cutoff):
                continue
            chosen = [i*width+j for j in range(width) if use[i*width+j] and available[i*width+j]]
            if not chosen:
                continue
            if day == INITIAL:
                row['skipped_initial_rows'] += len(chosen)
                continue
            month = dict(date=day, end=end, joined_rows=0, invalid_rows=0, valid_rows=0,
                retained=False, rows=[], numerator_sum=None, denominator_sum=None,
                numerator_mean=None, denominator_mean=None)
            for source_id in chosen:
                symbol = names[source_id]
                key = day, symbol, end
                _require(key in index, 'Missing mature exact (signal, security, end) label: '+repr(key))
                tid = index[key]; month['joined_rows'] += 1
                if not _boolean(data['common_good'][tid], 'common_good'):
                    month['invalid_rows'] += 1
                    continue
                features = [_number(v, 'train_plain') for v in data['train_plain'][tid]]
                targets = {k: _number(data['target_'+k][tid], 'target_'+k) for k in TARGETS}
                if not all(math.isfinite(v) for v in features+list(targets.values())):
                    month['invalid_rows'] += 1
                    continue
                month['rows'].append(dict(source_row_id=source_id, training_row_id=tid,
                    symbol=symbol, date=day, end=end, prediction=float(raw[source_id]),
                    pooled_prediction=float(pooled[source_id]), target=targets['log_excess'],
                    disagreement=None, residual_target=None, product=None, square=None))
            size = len(month['rows']); month['valid_rows'] = size
            month['retained'] = size >= MIN_ROWS
            if month['retained']:
                for item in month['rows']:
                    d = _finite(item['prediction']-item['pooled_prediction'], 'historical increment')
                    z = _finite(item['target']-item['pooled_prediction'], 'historical residual target')
                    item.update(disagreement=d, residual_target=z, product=product(d,z), square=product(d,d))
                numerator = _sum([h['product'] for h in month['rows']], 'monthly numerator')
                denominator = _sum([h['square'] for h in month['rows']], 'monthly denominator')
                month.update(numerator_sum=numerator, denominator_sum=denominator,
                    numerator_mean=mean(numerator,size), denominator_mean=mean(denominator,size))
                history.extend(month['rows'])
            else:
                row['small_month_rows'] += size
            row['joined_label_rows'] += month['joined_rows']
            row['invalid_label_rows'] += month['invalid_rows']
            row['months'].append(month)
        retained = [m for m in row['months'] if m['retained']]
        row['calibration_months'], row['calibration_rows'] = len(retained), len(history)
        row['latest_label_end'] = max((h['end'] for h in history), default=None)
        row['numerator'] = _sum([m['numerator_mean'] for m in retained], 'pooled numerator')
        row['denominator'] = _sum([m['denominator_mean'] for m in retained], 'pooled denominator')
        alpha, clipped, status = coefficient(row['numerator'],row['denominator'],len(retained))
        row.update(alpha=alpha,clipped=clipped,status=status)
        pool_ready = status == 'calibrated'
        counts['annual_cohorts'] += 1
        counts[{'cold_start':'cold_start_years','zero_denominator':'zero_denominator_years',
                'calibrated':'pooled_fits'}[status]] += 1
        counts['clipped_pooled_fits'] += int(clipped)
        row['parameter_estimates'] = int(pool_ready)
        for symbol in securities:
            query_ids = [j for j in current if names[j] == symbol]
            own = [h for h in history if h['symbol'] == symbol]
            months = len({_month(h['date']) for h in own})
            _require(months == len(own), 'Duplicate own-security signal month')
            numerator = _sum([h['product'] for h in own], 'own numerator')
            denominator = _sum([h['square'] for h in own], 'own denominator')
            group = dict(symbol=symbol,query_row_ids=query_ids,
                query_dates=[queries[j//width][0] for j in query_ids],
                history_source_row_ids=[h['source_row_id'] for h in own],
                own_rows=len(own),own_months=months,
                latest_label_end=max((h['end'] for h in own),default=None),
                status='pooled_not_ready',numerator=numerator,denominator=denominator,
                alpha=None,applied_alpha=1.,clipped=False,fallback='original')
            if pool_ready:
                own_alpha, own_clipped, own_status = coefficient(numerator,denominator,months)
                ready = own_status == 'calibrated'
                group.update(status=own_status,alpha=own_alpha,clipped=own_clipped,
                    applied_alpha=own_alpha if ready else alpha,fallback='none' if ready else 'pooled')
            if group['status'] == 'calibrated':
                row['own_fits'] += 1; row['parameter_estimates'] += 1
                row['clipped_own_fits'] += int(group['clipped'])
            elif group['status'] == 'zero_denominator':
                row['zero_denominator_groups'] += 1
            else:
                row['cold_start_groups'] += 1
            if pool_ready:
                for j in query_ids:
                    out['candidate'][j] = blend(raw[j],pooled[j],group['applied_alpha'])
                    out['control'][j] = blend(raw[j],pooled[j],alpha)
            counts['own_history_rows'] += len(own)
            row['groups'].append(group)
        counts['own_history_groups'] += len(securities)
        for key in ('own_fits','cold_start_groups','zero_denominator_groups','clipped_own_fits',
                    'parameter_estimates','joined_label_rows','invalid_label_rows','small_month_rows',
                    'skipped_initial_rows','calibration_months','calibration_rows'):
            counts[key] += row[key]
        annual.append(row)
    return dict(raw=out, available=available.copy(), audit=dict(
        method='annual_own_increment_convex_calibration',arms=list(ARMS),query_months=len(queries),
        query_rows=len(queries)*width,symbols_per_event=width,min_months=MIN_MONTHS,
        min_rows_per_month=MIN_ROWS,lookback_calendar_years=5,source_forecasts_refitted=False,
        labels_recomputed=False,local_available_filtered=False,historical_use_explicit=True,
        current_availability_unchanged=True,pool_uses_all_historical_ids=True,
        monthly_equal_pool_weights=True,annual=annual,counts=counts,
        new_model_fits=counts['parameter_estimates'],predictive_model_fits=counts['parameter_estimates'],
        new_ridge_fits=0,new_tree_fits=0))


