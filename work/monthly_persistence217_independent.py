"""Independent previous-formal-month forecast-direction confirmation oracle.

No producer, strategy, raw-price loader or new regression fitter is imported.
Callers pin the206 vintage cache; this module preserves it and independently
reconstructs exact monthly/identity joins, support precedence and scalar direction branches.
Canonical binary64 arithmetic is compared exactly, including ranks and masks.
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
ARMS = ('incumbent_local67', 'persistence_confirmed', 'persistence_half')
NEW_RAW = {'persistence_confirmed': 'confirmed', 'persistence_half': 'half'}
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
MODEL_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked',
    'previous_formal_calendar_month_checked', 'initial_partial_excluded',
    'no_missing_month_bridge', 'cached_forecasts_reused',
    'local_linear_and_pooled_sources_checked', 'current_and_previous_original_universe_checked',
    'cross_year_asof_pairing_checked', 'direction_confirmation_checked', 'half_correction_checked',
    'unsupported_and_zero_exact_copy', 'reason_precedence_checked', 'availability_unchanged',
    'no_local_history_filter', 'no_labels_or_future_outcomes_inputs',
    'source_forecasts_not_refitted', 'audit_checked', 'zero_fits_and_parameters_checked')

RANK_FLAGS = ('passed', 'full_original_universe', 'includes_eligible_etfs',
    'no_local_history_filter', 'replacement_not_additive_local_slot',
    'paired_whole_row_readiness', 'initial_partial_unchanged', 'exact_rank_rebuild')
MODEL_COUNT_FIELDS = ('query_rows', 'cached_forecast_rows', 'supported_rows',
    'unsupported_rows', 'current_zero_rows', 'confirmed_rows', 'unconfirmed_rows',
    'cross_year_supported_rows')
COUNT_FIELDS = MODEL_COUNT_FIELDS
ZERO_FIELDS = ('new_model_fits', 'predictive_model_fits', 'new_ridge_fits',
               'new_tree_fits', 'new_parameters')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('persistence217_independent_accounting', ACCOUNTING)
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




def validated_inputs(query_meta, source, original_use):
    require(isinstance(query_meta, dict) and isinstance(source, dict), 'Explicit input dictionaries required')
    require(all(k in query_meta for k in ('dates', 'ends', 'symbols')), 'Query identities missing')
    use = np.asarray(original_use)
    require(use.ndim == 2 and use.shape[1] > 0 and
            all(isinstance(v, (bool, np.bool_)) for v in use.flat), 'Original U exact boolean matrix')
    n, width = use.shape
    require(all(isinstance(query_meta[k], list) for k in ('dates','ends','symbols')), 'Monthly metadata lists')
    require(all(isinstance(d, str) for d in query_meta['dates']+query_meta['ends']), 'Exact ISO date strings')
    dates, ends = days(query_meta['dates'], 'query dates'), days(query_meta['ends'], 'query ends')
    symbols = query_meta['symbols']
    require(len(dates) == len(ends) == len(symbols) == n, 'Query metadata axes')
    formal, identities = {}, []
    previous_date = None
    for i, (day, end, names) in enumerate(zip(dates, ends, symbols)):
        month = int(day[:4])*12+int(day[5:7])
        end_month = int(end[:4])*12+int(end[5:7])
        initial = day == '2016-08-17'
        require(not initial or (i == 0 and end == '2016-08-31'), 'Initial partial identity')
        require(end > day and (initial or end_month == month+1), 'Next-natural-month end required')
        require(previous_date is None or day > previous_date, 'Strictly increasing query dates')
        require(isinstance(names, list) and len(names) == width and
                all(isinstance(v,str) and v and v == v.strip() for v in names) and
                len(set(names)) == width, 'Exact unique security identities')
        if not initial:
            require(month not in formal, 'Duplicate formal calendar month')
            formal[month] = i
        identities.extend((i,j,day,name,month,initial) for j,name in enumerate(names))
        previous_date = day
    count = n*width
    require(all(k in source for k in ('available','raw_local_linear','raw_source_log')), 'Cached206 fields missing')
    available = np.asarray(source['available'])
    require(available.shape == (count,) and
            all(isinstance(v,(bool,np.bool_)) for v in available), 'Exact source available boolean axis')
    available = available.astype(bool,copy=True)
    values = {}
    for key in ('raw_local_linear','raw_source_log'):
        value = np.asarray(source[key])
        require(value.shape == (count,), 'Cached raw source axis '+key)
        values[key] = np.array([real(v,key) for v in value], dtype=np.float64)
        require(not np.isinf(values[key]).any() and
                np.array_equal(np.isfinite(values[key]), available) and
                np.array_equal(np.isnan(values[key]), ~available), 'Cached raw/source availability '+key)
    return use.astype(bool,copy=True), available, values, dates, symbols, formal, identities


def reconstruct(query_meta, source, original_use):
    use, available, values, dates, symbols, formal, identities = validated_inputs(query_meta, source, original_use)
    p, b = values['raw_local_linear'], values['raw_source_log']
    n, width = use.shape
    out = {key:p.copy() for key in ('confirmed','half')}
    counts = {key:0 for key in MODEL_COUNT_FIELDS}
    counts.update(query_rows=n*width,cached_forecast_rows=int(available.sum()))
    records = []
    for row, (i,j,day,symbol,month,initial) in enumerate(identities):
        if not available[row]:
            continue
        previous_row, reason = None, None
        if not use[i,j]:
            reason = 'current_outside_universe'
        elif month-1 not in formal:
            reason = 'missing_previous_formal_month'
        else:
            prev_i = formal[month-1]
            if symbol not in symbols[prev_i]:
                reason = 'missing_previous_identity'
            else:
                previous_row = prev_i*width+symbols[prev_i].index(symbol)
                if not available[previous_row]:
                    reason = 'previous_unavailable'
                elif not use.flat[previous_row]:
                    reason = 'previous_outside_universe'
        current_difference = previous_difference = None
        cross_year = False
        if reason is not None:
            status = 'unsupported'
            counts['unsupported_rows'] += 1
        else:
            counts['supported_rows'] += 1
            cross_year = dates[previous_row//width][:4] != day[:4]
            counts['cross_year_supported_rows'] += int(cross_year)
            current_difference = finite(float(p[row])-float(b[row]), 'current difference')
            if current_difference == 0.:
                status = 'current_zero'
            else:
                previous_difference = finite(float(p[previous_row])-float(b[previous_row]), 'previous difference')
                confirmed = ((current_difference > 0. and previous_difference > 0.) or
                             (current_difference < 0. and previous_difference < 0.))
                status = 'confirmed' if confirmed else 'unconfirmed'
                out['confirmed'][row] = float(p[row]) if confirmed else float(b[row])
                out['half'][row] = finite(float(b[row]) + .5*current_difference, 'half correction')
            counts[status+'_rows'] += 1
            reason = status
        records.append(dict(row=row,date=day,symbol=symbol,prior_row=previous_row,status=status,
            reason=reason,current_difference=current_difference,previous_difference=previous_difference,
            cross_year=cross_year))
    audit = dict(method='previous_formal_month_direction_confirmation',counts=counts,records=records,
                 **{key:0 for key in ZERO_FIELDS})
    return dict(raw=out,available=available.copy(),audit=audit)


def check_forecasts(query_meta, source, original_use, forecast):
    expected = reconstruct(query_meta,source,original_use)
    require(isinstance(forecast,dict) and set(forecast)=={'raw','available','audit'},'Forecast schema')
    require(isinstance(forecast['raw'],dict) and set(forecast['raw'])=={'confirmed','half'},'Forecast arms')
    for key in ('confirmed','half'):
        value = np.asarray(forecast['raw'][key])
        require(value.dtype==np.dtype('float64'),'Raw binary64 dtype '+key)
        close(value,expected['raw'][key],'raw/'+key)
        zeros = expected['raw'][key] == 0.
        require(np.array_equal(np.signbit(value[zeros]),np.signbit(expected['raw'][key][zeros])),
                'Copied zero sign differs '+key)
    mask = np.asarray(forecast['available'])
    require(mask.dtype==np.dtype(bool) and np.array_equal(mask,expected['available']), 'Current availability changed')
    record_check(forecast['audit'],expected['audit'])
    return dict(**{key:True for key in MODEL_FLAGS},counts=expected['audit']['counts'],
        query_rows=expected['audit']['counts']['query_rows'],atol=0.,
        producer_modules_imported=False,computes_returns=False,**{key:0 for key in ZERO_FIELDS})


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
