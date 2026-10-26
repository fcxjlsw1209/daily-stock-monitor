"""Own-security disagreement-conditioned calibration of immutable as-of forecasts.

Pure caller-array calibration. No loading, source-model refits, label creation,
ranks, orders or returns. Three mean parameters use the same own-security mature history.
"""
from datetime import date
from pathlib import Path
import math
import numbers

import numpy as np

ARMS = ('incumbent_local67', 'disagreement_conditional', 'disagreement_unconditional')
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
MIN_MONTHS = 18
GROUP_PSEUDOCOUNT = 18
INITIAL = '2016-08-17'
COUNT_FIELDS = ('annual_cohorts', 'own_history_groups',
    'supported_year_security_groups', 'cold_start_groups', 'parameter_estimates',
    'joined_label_rows', 'invalid_label_rows', 'skipped_initial_rows',
    'error_observations', 'cached_forecast_rows')


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


def calibration(errors, disagreements):
    """Signed unconditional mean and two empirically shrunk conditional means.

    No support test occurs here: the caller enforces the 18-month threshold.
    Group membership uses only the sign of local minus pooled prediction.
    An empty conditional group copies the unconditional mean exactly.
    """
    values = [_finite(_number(v, 'error'), 'error') for v in errors]
    differences = [_finite(_number(v, 'disagreement'), 'disagreement') for v in disagreements]
    _require(bool(values) and len(values) == len(differences), 'Aligned nonempty error/disagreement pairs required')
    total = _sum(values, 'unconditional error sum')
    mean = _finite(total/len(values), 'unconditional mean')
    conditional = {}
    for name, positive in (('positive', True), ('nonpositive', False)):
        part = [e for e, d in zip(values, differences) if (d > 0.) == positive]
        subtotal = _sum(part, name+' error sum')
        if not part:
            estimate = mean
        else:
            prior = _finite(GROUP_PSEUDOCOUNT*mean, 'empirical prior numerator')
            numerator = _sum([subtotal, prior], name+' shrunk numerator')
            estimate = _finite(numerator/(len(part)+GROUP_PSEUDOCOUNT), name+' shrunk mean')
        conditional[name] = dict(rows=len(part), error_sum=subtotal, mean=estimate,
                                 penalty=max(0., estimate), empty_group_copy=not bool(part))
    return dict(error_sum=total, unconditional_mean=mean,
                unconditional_penalty=max(0., mean), conditional=conditional)


def corrected(prediction, penalty):
    """Zero penalty returns the original binary64, including signed zero."""
    prediction = _finite(_number(prediction, 'prediction'), 'prediction')
    penalty = _finite(_number(penalty, 'penalty'), 'penalty')
    _require(penalty >= 0., 'Penalty must be nonnegative')
    if penalty == 0.:
        return prediction
    return _finite(prediction-penalty, 'corrected forecast')


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


def build_forecasts(train, training_meta, query_meta, source, historical_use):
    """Estimate three annual own-security mean parameters, preserving source availability.

    Historical U is supplied explicitly by the host as forecast_available AND
    eligible. It is intersected with cached source availability only when
    collecting errors; no history/local mask filters current predictions.
    Labels are inspected only after their annual maturity and own-ID checks.
    """
    queries, index, data, raw, pooled, available, use, width = _inputs(
        train, training_meta, query_meta, source, historical_use)
    n = len(queries)
    names = [symbol for _, _, symbols in queries for symbol in symbols]
    out = {key: raw.copy() for key in ('candidate', 'control')}
    counts = {key: 0 for key in COUNT_FIELDS}
    counts['cached_forecast_rows'] = int(available.sum())
    annual = []
    years = sorted({int(day[:4]) for day, _, _ in queries})
    for year in years:
        cutoff, start = f'{year-1:04d}-12-31', f'{year-6:04d}-12-31'
        current = [i*width+j for i, (day, _, _) in enumerate(queries) if int(day[:4]) == year
                   for j in range(width) if available[i*width+j]]
        securities = sorted({names[j] for j in current})
        row = dict(year=year, cutoff=cutoff, lookback_exclusive_start=start,
            prediction_rows=len(current), queried_security_count=len(securities),
            joined_label_rows=0, invalid_label_rows=0, skipped_initial_rows=0, error_observations=0,
            supported_groups=0, cold_start_groups=0, parameter_estimates=0, latest_label_end=None, groups=[])
        counts['annual_cohorts'] += 1
        for symbol in securities:
            query_ids = [j for j in current if names[j] == symbol]
            group = dict(symbol=symbol, query_row_ids=query_ids,
                query_dates=[queries[j//width][0] for j in query_ids],
                joined_label_rows=0, invalid_label_rows=0, skipped_initial_rows=0, history=[],
                own_rows=0, own_months=0, latest_label_end=None, status='cold_start',
                error_sum=None, unconditional_mean=None, unconditional_penalty=0.,
                conditional={key: dict(rows=0, error_sum=None, mean=None, penalty=0.,
                                       empty_group_copy=False) for key in ('positive', 'nonpositive')},
                query_details=[])
            for i, (day, end, symbols) in enumerate(queries):
                if not (day > start and end <= cutoff) or symbol not in symbols:
                    continue
                source_id = i*width + symbols.index(symbol)
                if not (use[source_id] and available[source_id]):
                    continue
                if day == INITIAL:
                    group['skipped_initial_rows'] += 1
                    continue
                key = (day, symbol, end)
                _require(key in index, 'Missing mature exact (signal, security, end) label: '+repr(key))
                training_id = index[key]
                group['joined_label_rows'] += 1
                if not _boolean(data['common_good'][training_id], 'common_good'):
                    group['invalid_label_rows'] += 1
                    continue
                features = [_number(v, 'train_plain') for v in data['train_plain'][training_id]]
                targets = {k: _number(data['target_'+k][training_id], 'target_'+k) for k in TARGETS}
                if not all(math.isfinite(v) for v in features+list(targets.values())):
                    group['invalid_label_rows'] += 1
                    continue
                prediction, target = float(raw[source_id]), targets['log_excess']
                error = _finite(prediction-target, 'signed historical forecast error')
                pooled_prediction = float(pooled[source_id])
                disagreement = _finite(prediction-pooled_prediction, 'historical prediction disagreement')
                direction = 'positive' if disagreement > 0. else 'nonpositive'
                group['history'].append(dict(source_row_id=source_id, training_row_id=training_id,
                    date=day, end=end, prediction=prediction, target=target, error=error,
                    pooled_prediction=pooled_prediction, disagreement=disagreement, disagreement_group=direction))
            group['own_rows'] = len(group['history'])
            group['own_months'] = len({_month(h['date']) for h in group['history']})
            _require(group['own_rows'] == group['own_months'], 'Duplicate own-security signal month')
            if group['history']:
                group['latest_label_end'] = max(h['end'] for h in group['history'])
            for h in group['history']:
                group['conditional'][h['disagreement_group']]['rows'] += 1
            if group['own_months'] >= MIN_MONTHS:
                group.update(calibration([h['error'] for h in group['history']],
                                         [h['disagreement'] for h in group['history']]), status='calibrated')
                row['supported_groups'] += 1
                row['parameter_estimates'] += 3
            else:
                row['cold_start_groups'] += 1
            for j in query_ids:
                prediction, pooled_prediction = float(raw[j]), float(pooled[j])
                disagreement = _finite(prediction-pooled_prediction, 'current prediction disagreement')
                direction = 'positive' if disagreement > 0. else 'nonpositive'
                penalty = group['conditional'][direction]['penalty']
                unconditional_penalty = group['unconditional_penalty']
                out['candidate'][j] = corrected(prediction, penalty)
                out['control'][j] = corrected(prediction, unconditional_penalty)
                group['query_details'].append(dict(source_row_id=j, date=queries[j//width][0],
                    prediction=prediction, pooled_prediction=pooled_prediction,
                    disagreement=disagreement, disagreement_group=direction,
                    conditional_penalty=penalty, unconditional_penalty=unconditional_penalty))
            for key in ('joined_label_rows', 'invalid_label_rows', 'skipped_initial_rows'):
                row[key] += group[key]
            row['error_observations'] += group['own_rows']
            if group['latest_label_end'] is not None:
                row['latest_label_end'] = max(row['latest_label_end'] or group['latest_label_end'], group['latest_label_end'])
            row['groups'].append(group)
        counts['own_history_groups'] += len(securities)
        counts['supported_year_security_groups'] += row['supported_groups']
        for key in ('cold_start_groups', 'parameter_estimates', 'joined_label_rows', 'invalid_label_rows',
                    'skipped_initial_rows', 'error_observations'):
            counts[key] += row[key]
        annual.append(row)
    return dict(raw=out, available=available.copy(), audit=dict(
        method='annual_own_asof_disagreement_conditional_mean', arms=list(ARMS), query_months=n, query_rows=n*width,
        symbols_per_event=width, min_months=MIN_MONTHS, lookback_calendar_years=5,
        group_pseudocount=GROUP_PSEUDOCOUNT, disagreement_observations=counts['error_observations'],
        historical_errors='cached raw_local_linear minus unchanged target_log_excess',
        historical_disagreements='cached raw_local_linear minus cached raw_source_log; positive iff > 0',
        source_forecasts_refitted=False, labels_recomputed=False, local_available_filtered=False,
        historical_use_explicit=True, current_availability_unchanged=True, annual=annual, counts=counts,
        new_model_fits=counts['parameter_estimates'], predictive_model_fits=counts['parameter_estimates'],
        new_ridge_fits=0, new_tree_fits=0))
