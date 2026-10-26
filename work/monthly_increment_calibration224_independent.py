"""Independent annual own-increment convex calibration oracle.

Only immutable caller-supplied cached forecasts and mature original labels are
consumed. No producer, strategy or predictive source fitter is imported.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'own_increment_calibration', 'pooled_increment_control')
NEW_RAW = {'own_increment_calibration': 'candidate', 'pooled_increment_control': 'control'}
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
MODEL_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked',
    'cached_forecasts_reused', 'local_and_pooled_source_checked',
    'annual_prior_december_cutoffs_checked', 'five_year_mature_window_checked',
    'four_target_common_cohort_checked', 'historical_original_universe_checked',
    'two_rows_per_month_common_cohort_checked', 'pooled_equal_month_weights_checked',
    'own_security_normal_equations_checked', 'bounded_alpha_checked',
    'pooled_not_ready_copy_checked', 'own_pooled_fallback_checked',
    'endpoint_and_equal_prediction_copy_checked', 'no_local_history_filter',
    'availability_unchanged', 'original_targets_unchanged', 'no_future_or_outcome_inputs',
    'source_forecasts_not_refitted', 'audit_checked', 'parameter_estimates_counted')
RANK_FLAGS = ('passed', 'full_original_universe', 'includes_eligible_etfs',
    'no_local_history_filter', 'replacement_not_additive_local_slot',
    'paired_whole_row_readiness', 'initial_partial_unchanged', 'exact_rank_rebuild')
COUNT_FIELDS = ('annual_cohorts', 'pooled_fits', 'cold_start_years',
    'zero_denominator_years', 'clipped_pooled_fits', 'own_history_groups', 'own_fits',
    'cold_start_groups', 'zero_denominator_groups', 'clipped_own_fits',
    'parameter_estimates', 'joined_label_rows', 'invalid_label_rows', 'small_month_rows',
    'skipped_initial_rows', 'calibration_months', 'calibration_rows',
    'own_history_rows', 'cached_forecast_rows')
MODEL_COUNT_FIELDS = COUNT_FIELDS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('increment224_independent_accounting', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected, label, atol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaN mask')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    use = np.isfinite(expected)
    require(np.all(np.abs(actual[use] - expected[use]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, e) in enumerate(zip(actual, expected)):
            record_check(a, e, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label)
    return value


def fsum(values, label):
    try:
        return finite(math.fsum(values), label)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid ' + label) from exc


def real(value, label):
    require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
            label + ': nonboolean real numeric required')
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(label + ': binary64 conversion') from exc


def days(values, label):
    raw = np.asarray(values)
    require(raw.ndim == 1, label + ': date vector')
    out = []
    for value in raw:
        if isinstance(value, np.datetime64):
            require(not np.isnat(value) and value == value.astype('datetime64[D]'),
                    label + ': normalized date required')
            value = value.astype('datetime64[D]')
        text = str(value)
        require(len(text) == 10, label + ': exact calendar-day dates')
        try:
            parsed = date.fromisoformat(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(label + ': invalid date') from exc
        require(parsed.isoformat() == text, label + ': noncanonical date')
        out.append(text)
    return out


def metadata(meta, label, width=None):
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'ends', 'symbols')),
            label + ': metadata keys')
    require(isinstance(meta['dates'], list) and isinstance(meta['ends'], list), label + ': monthly date lists')
    ds, es = days(meta['dates'], label + ' dates'), days(meta['ends'], label + ' ends')
    rows = meta['symbols']
    require(isinstance(rows, list) and len(rows) == len(ds) == len(es), label + ': metadata axes')
    require(all(a < b for a, b in zip(ds, ds[1:])), label + ': strictly ascending signal dates')
    offsets, flat = [0], []
    for i, (d, end, row) in enumerate(zip(ds, es, rows)):
        partial = label == 'query' and i == 0 and d == '2016-08-17' and end == '2016-08-31'
        month = int(d[:4])*12 + int(d[5:7])
        end_month = int(end[:4])*12 + int(end[5:7])
        require(end > d and (partial or end_month == month+1), label + ': next-natural-month label end')
        if i:
            previous_month = int(ds[i-1][:4])*12 + int(ds[i-1][5:7])
            initial_pair = label == 'query' and i == 1 and ds[0] == '2016-08-17' and d == '2016-08-31'
            require(month > previous_month or initial_pair, label + ': duplicate signal month')
        require(isinstance(row, list) and (width is None or len(row) == width) and
                all(isinstance(x, str) and x and x == x.strip() for x in row) and
                len(set(row)) == len(row), label + ': exact unique security IDs')
        flat.extend((d, s, end) for s in row)
        offsets.append(len(flat))
    require(len(set(flat)) == len(flat), label + ': duplicate identity')
    return ds, es, offsets, flat


def validated_inputs(train, training_meta, query_meta, source, historical_use):
    require(isinstance(train, dict) and isinstance(source, dict), 'Caller dictionaries required')
    eligibility = np.asarray(historical_use)
    require(eligibility.ndim == 2 and eligibility.shape[1] >= 1 and
            all(isinstance(v, (bool, np.bool_)) for v in eligibility.flat), 'Exact boolean original historical U')
    n, width = eligibility.shape
    qd, qe, qo, query_ids = metadata(query_meta, 'query', width)
    require(len(qd) == n, 'Query eligibility axis')
    td, te, to, training_ids = metadata(training_meta, 'training')
    required = ('offsets', 'dates', 'ends', 'train_plain', 'common_good') + tuple('target_'+k for k in TARGETS)
    require(all(k in train for k in required), 'Original training keys missing')
    offsets = np.asarray(train['offsets'])
    require(offsets.dtype.kind in 'iu' and np.array_equal(offsets, to), 'Original offsets differ')
    require(days(train['dates'], 'training flat dates') == [t[0] for t in training_ids] and
            days(train['ends'], 'training flat ends') == [t[2] for t in training_ids], 'Original flat date identities')
    size, count = len(training_ids), n * width
    require(np.asarray(train['train_plain']).shape == (size, 10), 'Original plain10 axes')
    for key in ('common_good',) + tuple('target_'+k for k in TARGETS):
        require(np.asarray(train[key]).shape == (size,), 'Training vector axes: '+key)
    require(all(k in source for k in ('available', 'raw_local_linear', 'raw_source_log')),
            'Cached206 source keys missing')
    available = np.asarray(source['available'])
    require(available.shape == (count,) and all(isinstance(x, (bool, np.bool_)) for x in available),
            'Cached availability exact boolean axis')
    available = available.astype(bool, copy=True)
    raw = {}
    for key in ('raw_local_linear', 'raw_source_log'):
        values = np.asarray(source[key])
        require(values.shape == (count,), 'Cached raw axis: '+key)
        raw[key] = np.array([real(v, key) for v in values], dtype=np.float64)
        require(np.array_equal(np.isfinite(raw[key]), available) and
                np.array_equal(np.isnan(raw[key]), ~available), 'Cached availability/raw mismatch: '+key)
    return eligibility.astype(bool), available, raw, qd, qe, qo, query_ids, training_ids


def product(a, b, label):
    value = finite(a*b, label)
    if a != 0. and b != 0. and value == 0.:
        raise FloatingPointError('Underflow '+label)
    return value


def mean(total, n, label):
    require(n > 0, 'Positive observation count required')
    value = finite(total/n, label)
    if total != 0. and value == 0.:
        raise FloatingPointError('Underflow '+label)
    return value


def coefficient(numerator, denominator):
    numerator = finite(real(numerator, 'numerator'), 'numerator')
    denominator = finite(real(denominator, 'denominator'), 'denominator')
    require(denominator > 0., 'Positive calibration denominator required')
    clipped = numerator < 0. or numerator > denominator
    if numerator <= 0.:
        alpha = 0.
    elif numerator >= denominator:
        alpha = 1.
    else:
        alpha = finite(numerator/denominator, 'interior coefficient')
        if alpha == 0.:
            raise FloatingPointError('Underflow interior coefficient')
    return alpha, bool(clipped)


def blend(prediction, pooled_prediction, alpha):
    p = finite(real(prediction, 'prediction'), 'prediction')
    b = finite(real(pooled_prediction, 'pooled prediction'), 'pooled prediction')
    alpha = finite(real(alpha, 'coefficient'), 'coefficient')
    require(0. <= alpha <= 1., 'Convex coefficient required')
    if alpha == 1. or p == b:
        return p
    if alpha == 0.:
        return b
    difference = finite(p-b, 'current increment')
    return finite(b+product(alpha, difference, 'scaled current increment'), 'blended forecast')


def reconstruct(train, training_meta, query_meta, source, historical_use):
    use, available, raw, dates, ends, offsets, query_ids, training_ids = validated_inputs(
        train, training_meta, query_meta, source, historical_use)
    width = use.shape[1]
    lookup = {key:i for i,key in enumerate(training_ids)}
    predictions, pooled = raw['raw_local_linear'], raw['raw_source_log']
    answer = {key:np.full(len(query_ids), np.nan) for key in ('candidate','control')}
    counts = {key:0 for key in COUNT_FIELDS}
    counts['cached_forecast_rows'] = int(available.sum())
    annual = []
    for year in sorted({int(d[:4]) for d in dates}):
        cutoff = '%04d-12-31' % (year-1)
        start = '%04d-12-31' % (year-6)
        current = [j for j,(d,s,e) in enumerate(query_ids) if int(d[:4]) == year and available[j]]
        symbols = sorted({query_ids[j][1] for j in current})
        month_map = {}
        skipped = 0
        for j, (signal, symbol, end) in enumerate(query_ids):
            if not (signal > start and end <= cutoff and available[j] and use[j//width,j%width]):
                continue
            if signal == '2016-08-17':
                skipped += 1
                continue
            key = (signal,symbol,end)
            require(key in lookup, 'Missing mature exact training identity: '+repr(key))
            tid = lookup[key]
            if signal not in month_map:
                month_map[signal] = dict(date=signal,end=end,joined_rows=0,invalid_rows=0,
                    valid_rows=0,retained=False,rows=[],numerator_sum=None,denominator_sum=None,
                    numerator_mean=None,denominator_mean=None)
            month = month_map[signal]
            require(month['end'] == end, 'Mature month has mixed label ends')
            month['joined_rows'] += 1
            good = np.asarray(train['common_good'])[tid]
            require(isinstance(good,(bool,np.bool_)), 'Consumed common_good must be boolean')
            if not good:
                month['invalid_rows'] += 1
                continue
            x = [real(v,'mature training feature') for v in np.asarray(train['train_plain'])[tid]]
            targets = [real(np.asarray(train['target_'+name])[tid],'mature target') for name in TARGETS]
            if not all(math.isfinite(v) for v in x+targets):
                month['invalid_rows'] += 1
                continue
            month['rows'].append(dict(source_row_id=j,training_row_id=tid,symbol=symbol,
                date=signal,end=end,prediction=float(predictions[j]),pooled_prediction=float(pooled[j]),
                target=targets[1],disagreement=None,residual_target=None,product=None,square=None))
        months = list(month_map.values())
        retained = []
        for month in months:
            rows = month['rows']; month['valid_rows'] = len(rows)
            if len(rows) < 2:
                continue
            month['retained'] = True
            for row in rows:
                d = finite(row['prediction']-row['pooled_prediction'],'historical increment')
                z = finite(row['target']-row['pooled_prediction'],'historical residual target')
                row.update(disagreement=d,residual_target=z,
                           product=product(d,z,'increment product'),square=product(d,d,'increment square'))
            month['numerator_sum'] = fsum([r['product'] for r in rows],'monthly numerator')
            month['denominator_sum'] = fsum([r['square'] for r in rows],'monthly denominator')
            month['numerator_mean'] = mean(month['numerator_sum'],len(rows),'monthly numerator mean')
            month['denominator_mean'] = mean(month['denominator_sum'],len(rows),'monthly denominator mean')
            retained.append(month)
        numerator = fsum([m['numerator_mean'] for m in retained],'pooled numerator')
        denominator = fsum([m['denominator_mean'] for m in retained],'pooled denominator')
        alpha = None; clipped = False
        if len(retained) < 18:
            status = 'cold_start'
        elif denominator == 0.:
            status = 'zero_denominator'
        else:
            status = 'calibrated'; alpha,clipped = coefficient(numerator,denominator)
        history = [row for month in retained for row in month['rows']]
        groups = []
        for symbol in symbols:
            query_rows = [j for j in current if query_ids[j][1] == symbol]
            own = [row for row in history if row['symbol'] == symbol]
            own_months = len({r['date'][:7] for r in own})
            require(own_months == len(own), 'Duplicate own signal month')
            own_n = fsum([r['product'] for r in own],'own numerator')
            own_d = fsum([r['square'] for r in own],'own denominator')
            own_alpha = None; own_clipped = False
            if status != 'calibrated':
                own_status = 'pooled_not_ready'; applied_alpha = 1.; fallback = 'original'
            elif own_months < 18:
                own_status = 'cold_start'; applied_alpha = alpha; fallback = 'pooled'
            elif own_d == 0.:
                own_status = 'zero_denominator'; applied_alpha = alpha; fallback = 'pooled'
            else:
                own_status = 'calibrated'; own_alpha,own_clipped = coefficient(own_n,own_d)
                applied_alpha = own_alpha; fallback = 'none'
            for j in query_rows:
                p,b = float(predictions[j]),float(pooled[j])
                if status != 'calibrated':
                    answer['candidate'][j] = p; answer['control'][j] = p
                else:
                    answer['candidate'][j] = blend(p,b,applied_alpha)
                    answer['control'][j] = blend(p,b,alpha)
            groups.append(dict(symbol=symbol,query_row_ids=query_rows,
                query_dates=[query_ids[j][0] for j in query_rows],
                history_source_row_ids=[row['source_row_id'] for row in own],own_rows=len(own),
                own_months=own_months,latest_label_end=max((r['end'] for r in own),default=None),
                status=own_status,numerator=own_n,denominator=own_d,alpha=own_alpha,
                applied_alpha=applied_alpha,clipped=own_clipped,fallback=fallback))
        own_fits = sum(g['status']=='calibrated' for g in groups)
        record = dict(year=year,cutoff=cutoff,lookback_exclusive_start=start,
            prediction_rows=len(current),queried_security_count=len(symbols),
            joined_label_rows=sum(m['joined_rows'] for m in months),
            invalid_label_rows=sum(m['invalid_rows'] for m in months),
            small_month_rows=sum(m['valid_rows'] for m in months if not m['retained']),
            skipped_initial_rows=skipped,calibration_months=len(retained),calibration_rows=len(history),
            latest_label_end=max((row['end'] for row in history),default=None),status=status,
            numerator=numerator,denominator=denominator,alpha=alpha,clipped=clipped,
            own_fits=own_fits,cold_start_groups=sum(g['status'] in ('pooled_not_ready','cold_start') for g in groups),
            zero_denominator_groups=sum(g['status']=='zero_denominator' for g in groups),
            clipped_own_fits=sum(g['clipped'] for g in groups),
            parameter_estimates=int(status=='calibrated')+own_fits,months=months,groups=groups)
        annual.append(record)
        counts['pooled_fits'] += int(status=='calibrated')
        counts['cold_start_years'] += int(status=='cold_start')
        counts['zero_denominator_years'] += int(status=='zero_denominator')
        counts['clipped_pooled_fits'] += int(clipped)
        counts['own_history_groups'] += len(groups)
        counts['own_history_rows'] += sum(g['own_rows'] for g in groups)
        for key in ('own_fits','cold_start_groups','zero_denominator_groups','clipped_own_fits',
                    'parameter_estimates','joined_label_rows','invalid_label_rows','small_month_rows',
                    'skipped_initial_rows','calibration_months','calibration_rows'):
            counts[key] += record[key]
    counts['annual_cohorts'] = len(annual)
    fits = counts['parameter_estimates']
    audit = dict(method='annual_own_increment_convex_calibration',arms=list(ARMS),
        query_months=len(dates),query_rows=len(query_ids),symbols_per_event=width,
        min_months=18,min_rows_per_month=2,lookback_calendar_years=5,
        source_forecasts_refitted=False,labels_recomputed=False,local_available_filtered=False,
        historical_use_explicit=True,current_availability_unchanged=True,
        pool_uses_all_historical_ids=True,monthly_equal_pool_weights=True,annual=annual,counts=counts,
        new_model_fits=fits,predictive_model_fits=fits,new_ridge_fits=0,new_tree_fits=0)
    return dict(raw=answer,available=available.copy(),audit=audit)


def check_forecasts(train, training_meta, query_meta, source, historical_use, forecast):
    expected = reconstruct(train,training_meta,query_meta,source,historical_use)
    require(isinstance(forecast,dict) and set(forecast)=={'raw','available','audit'},'Forecast schema')
    require(isinstance(forecast['raw'],dict) and set(forecast['raw'])=={'candidate','control'},'Forecast arms')
    for key in ('candidate','control'):
        values = np.asarray(forecast['raw'][key])
        require(values.dtype==np.dtype('float64'),'Raw dtype: '+key)
        close(values,expected['raw'][key],'raw/'+key)
        zero = expected['raw'][key] == 0.
        require(np.array_equal(np.signbit(values[zero]),np.signbit(expected['raw'][key][zero])),
                'Zero prediction copy sign differs: '+key)
    mask = np.asarray(forecast['available'])
    require(mask.dtype==np.dtype(bool) and np.array_equal(mask,expected['available']),'Original availability changed')
    record_check(forecast['audit'],expected['audit'])
    return dict(**{key:True for key in MODEL_FLAGS},counts=expected['audit']['counts'],
        annual_cohorts=len(expected['audit']['annual']),query_rows=expected['audit']['query_rows'],
        new_model_fits=expected['audit']['new_model_fits'],predictive_model_fits=expected['audit']['predictive_model_fits'],
        new_ridge_fits=0,new_tree_fits=0,atol=0.,computes_returns=False,producer_modules_imported=False)


def centered_rank(v):
    """Stable average rank / population minus 0.5, without pandas/production rank."""
    v = np.asarray(v, float)
    require(v.ndim == 1 and np.isfinite(v).all() and len(v), 'Invalid rank population')
    order = np.argsort(v, kind='stable')
    answer = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]:
            j += 1
        answer[order[i:j]] = ((i + 1 + j) / 2.) / len(v) - .5
        i = j
    return answer


def rank_oracle(a, forecast):
    shape=np.asarray(a['eligible']).shape
    require(len(shape)==2, 'Original eligibility shape')
    n,width=shape
    def matrix(key,boolean=False):
        value=np.asarray(a[key])
        require(value.shape==shape,'Original matrix shape: '+key)
        if boolean:require(value.dtype==np.dtype(bool),'Boolean matrix: '+key)
        return value
    original=matrix('rank_log_excess')
    require(np.isfinite(original).all() and (np.abs(original)<=.5+1e-12).all(),'Invalid original ranks')
    use=matrix('forecast_available',True)&matrix('eligible',True)
    source=matrix('raw_source_log')
    require(np.isfinite(source[use]).all(),'Original source unavailable on U')
    old_available=matrix('model_available',True)
    old_values={k:matrix('raw_'+k) for k in ('local_constant','local_linear')}
    require(all(not np.isinf(v).any() for v in old_values.values()), 'Infinite original local forecast')
    old=original.copy()
    for i,members in enumerate(use):
        if members.any() and old_available[i,members].all() and all(np.isfinite(v[i,members]).all() for v in old_values.values()):
            v=old_values['local_linear'][i,members]
            if not np.array_equal(v,source[i,members]):old[i,members]=centered_rank(v)
    available=np.asarray(forecast['available'])
    require(available.dtype==np.dtype(bool) and available.size==n*width,'Forecast bool availability')
    available=available.reshape(shape)
    candidate={arm:np.asarray(forecast['raw'][key],float) for arm,key in NEW_RAW.items()}
    require(all(v.size==n*width and not np.isinf(v).any() for v in candidate.values()),'Invalid paired forecasts')
    candidate={k:v.reshape(shape) for k,v in candidate.items()}
    ready=np.zeros(n,bool);ranks={arm:old.copy() for arm in NEW_RAW}
    for i,members in enumerate(use):
        ready[i]=bool(members.any() and available[i,members].all() and
                      all(np.isfinite(v[i,members]).all() for v in candidate.values()))
        if ready[i]:
            for arm,v in candidate.items():
                ranks[arm][i,members]=(original[i,members] if np.array_equal(v[i,members],source[i,members])
                                       else centered_rank(v[i,members]))
    weak,calm=np.asarray(a['state_weak_narrow']),np.asarray(a['state_calm10'])
    require(weak.shape==calm.shape==(n,) and weak.dtype==calm.dtype==np.dtype(bool),'Boolean state axes')
    extension=matrix('extension_available',True)
    coefficient=.25*(.05*weak[:,None]+.01*(calm&~weak)[:,None]*extension)
    deltas={}
    active=use&ready[:,None]
    for arm,rank in ranks.items():
        delta=np.zeros(shape);delta[active]=(coefficient*(rank-old))[active]
        if n:delta[0]=0.
        require(np.isfinite(delta).all(),'Nonfinite slot delta')
        deltas[arm]=delta
    return ranks,deltas,{arm:ready.copy() for arm in NEW_RAW}


def verify_rank_slot(a, forecast, ranks, deltas):
    require(set(ranks)==set(deltas)==set(NEW_RAW),'Candidate slot names differ')
    rr,dd,ready=rank_oracle(a,forecast)
    for arm in NEW_RAW:
        close(ranks[arm],rr[arm],arm+' full-U rank',atol=0.)
        close(deltas[arm],dd[arm],arm+' replacement delta',atol=0.)
    return dict(**{key:True for key in RANK_FLAGS},
        ready_rows={k:int(v.sum()) for k,v in ready.items()},
        changed_cells={k:int(np.count_nonzero(v)) for k,v in dd.items()})


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            observed = wanted['metrics']['annual_returns']
            require(isinstance(observed, list) and len(observed) == len(annual), 'Annual return count')
            for actual, expected in zip(observed, annual):
                require(set(actual) == set(expected) and actual['year'] == expected['year'] and
                        actual['partial'] == expected['partial'], 'Annual return identity')
                close(actual['return_'], expected['return_'], 'Annual return', atol=1e-12)
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
