"""Independent annual as-of risk reliability and funded auxiliary-slot audit.

Only synthetic callers or a registered host supply arrays. No producer is
imported. Frozen independent accounting is hash-checked only when requested.
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
ARMS = ('incumbent_local67', 'risk_forecast_reliability', 'previous_month_reliability_control')
RAW = {ARMS[1]:'candidate', ARMS[2]:'control'}
BUDGET = .01
MAX_DELTA = .01000000000001
VALUE_KEYS = ('available','candidate','control','raw_candidate','raw_control')
COUNT_FIELDS = ('checked_cells','available_cells','unavailable_cells','year_security_groups',
    'supported_groups','cold_start_groups','historical_month_pairs','source_available_pairs',
    'available_pairs','valid_price_observations','simple_return_observations',
    'second_moment_estimates','error_evaluations','calibration_estimates','ranked_observations')
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','annual_prior_december_cutoffs_checked',
    'five_year_sixty_calendar_months_checked','official_month_windows_checked','current63_forecast_checked',
    'same_realized_outcome_checked','preceding_month_control_checked','noncentered_second_moments_checked',
    'symmetric_relative_squared_errors_checked','common_pair_support_checked','same_column_history_checked',
    'no_historical_membership_filter','eighteen_month_support_checked','annual_group_reuse_checked',
    'common_rank_domain_checked','original_pct_ranks_checked','exact_common_availability','audit_checked',
    'calibration_estimates_counted','no_ridge_or_tree_refits','no_current_or_future_outcome_inputs',
    'monthend_queries_checked')
DELTA_FLAGS = ('passed','funded_aux_formula_checked','original_pct_ranks_preserved',
    'first_partial_unchanged','all_states_preserved','missing_preserves_score','core_weights_checked',
    'no_second_rank','independent_delta_rebuild')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('reliability223_frozen_independent_accounts', FROZEN_HELPER)
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



MEASURES = ('forecast_second_moment','prior_second_moment','realized_second_moment',
            'candidate_error','control_error')


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from finite source prices')
    return value


def total(values):
    try:
        return finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Scalar sum overflow') from exc


def positive_divide(numerator, denominator):
    result = finite(numerator / denominator)
    if numerator != 0. and result == 0.:
        raise FloatingPointError('Nonzero mean or quotient underflow')
    return result


def square(value):
    result = finite(value * value)
    if value != 0. and result == 0.:
        raise FloatingPointError('Nonzero squared value underflow')
    return result


def source_prices(values, label):
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Real) for x in values):
        return None, 'nonnumeric_'+label+'_price'
    try:
        prices = [float(x) for x in values]
    except (TypeError,ValueError,OverflowError):
        return None, 'nonnumeric_'+label+'_price'
    if any(not math.isfinite(x) or x <= 0. for x in prices):
        return None, 'invalid_'+label+'_price'
    return prices, None


def second_moment(prices):
    """Noncentered second moment; each simple return receives equal weight."""
    returns = [finite(b/a-1.) for a,b in zip(prices,prices[1:])]
    return positive_divide(total([square(r) for r in returns]), float(len(returns)))


def error_score(realized, forecast):
    difference = finite(realized-forecast)
    denominator = finite(realized+forecast)
    require(denominator > 0., 'Positive error denominator required')
    normalized = positive_divide(difference,denominator)
    return square(normalized)


def empty(reason):
    return dict(source_available=False,available=False,reason=reason,
                **{key:None for key in MEASURES})


def measure_oracle(forecast_prices, prior_prices, realized_prices):
    require(np.shape(forecast_prices)==(64,), 'Exactly64 forecast prices required')
    require(np.ndim(prior_prices)==1 and len(prior_prices)>=2, 'Complete prior price vector required')
    require(np.ndim(realized_prices)==1 and len(realized_prices)>=2, 'Complete realized price vector required')
    sources = []
    for raw,label in ((forecast_prices,'forecast'),(prior_prices,'prior'),(realized_prices,'realized')):
        prices,reason = source_prices(raw,label)
        if reason:
            return empty(reason)
        sources.append(prices)
    try:
        f,p,r = [second_moment(prices) for prices in sources]
        out = empty('zero_pair_denominator')
        out.update(source_available=True,forecast_second_moment=f,prior_second_moment=p,
                   realized_second_moment=r)
        df,dp = finite(r+f),finite(r+p)
        if df == 0. or dp == 0.:
            return out
        out.update(available=True,reason='available',candidate_error=error_score(r,f),
                   control_error=error_score(r,p))
        return out
    except (OverflowError,ValueError,ZeroDivisionError) as exc:
        raise FloatingPointError('Invalid arithmetic from finite positive source prices') from exc


def calendar_windows(sessions, month, month_positions=None):
    """Explicit natural-month support and both official boundary witnesses."""
    period = pd.Period(month,freq='M')
    if month_positions is None:
        month_positions = {}
        for k,day in enumerate(sessions):
            month_positions.setdefault(day.to_period('M'),[]).append(k)
    def anchored(p):
        indices = month_positions.get(p,[])
        require(bool(indices), 'Missing official calendar month '+str(p))
        first,last = indices[0],indices[-1]
        require(first>0 and last+1<len(sessions), 'Calendar needs preceding/following witnesses')
        require(sessions[first-1].to_period('M')==p-1 and
                sessions[last+1].to_period('M')==p+1, 'Calendar month-boundary gap')
        return sessions[first-1:last+1],first-1,last
    current,anchor,last = anchored(period)
    prior,_,prior_last = anchored(period-1)
    require(prior_last==anchor and anchor>=63, 'Previous month anchor or63-return warmup missing')
    forecast = sessions[anchor-63:anchor+1]
    return forecast,prior,current


def original_pct_ranks(raw, available):
    result = np.full(np.shape(raw),np.nan)
    for i in range(len(result)):
        indices = [j for j,good in enumerate(available[i]) if good]
        for j in indices:
            less = sum(raw[i,k] < raw[i,j] for k in indices)
            equal = sum(raw[i,k] == raw[i,j] for k in indices)
            result[i,j] = (less+(equal+1)/2.)/len(indices)-.5
    return result


def independent_features(panel,meta,calendar,limit=None):
    sessions,queries,width = schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0<=limit<=len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    shape = (len(queries),width)
    values = {key:np.full(shape,np.nan) for key in VALUE_KEYS if key!='available'}
    values['available'] = np.zeros(shape,bool)
    frame = panel['adj']
    unique = sorted({(day.year,symbol) for day,names,_ in queries for symbol in names})
    groups,reasons = [],{}
    counts = {key:0 for key in COUNT_FIELDS}
    windows = {}
    month_positions = {}
    for k,day in enumerate(sessions):
        month_positions.setdefault(day.to_period('M'),[]).append(k)
    for year,symbol in unique:
        group_id = str(year)+':'+symbol
        cutoff = str(year-1)+'-12-31'
        pairs,group_reasons = [],{}
        start = pd.Period(str(year-5)+'-01',freq='M')
        for offset in range(60):
            month = str(start+offset)
            if month not in windows:
                windows[month] = calendar_windows(sessions,month,month_positions)
            forecast,prior,realized = windows[month]
            require(realized[-1] <= iso(cutoff) and forecast[-1]<realized[-1], 'Historical cutoff violated')
            missing = ('missing_symbol' if symbol not in frame.columns else
                'missing_forecast_session' if any(d not in frame.index for d in forecast) else
                'missing_prior_session' if any(d not in frame.index for d in prior) else
                'missing_realized_session' if any(d not in frame.index for d in realized) else None)
            measured = empty(missing) if missing else measure_oracle(
                [frame.at[d,symbol] for d in forecast], [frame.at[d,symbol] for d in prior],
                [frame.at[d,symbol] for d in realized])
            dates = dict(month=month,forecast_date=str(forecast[-1].date()),
                forecast_price_start=str(forecast[0].date()),forecast_price_end=str(forecast[-1].date()),
                prior_price_start=str(prior[0].date()),prior_price_end=str(prior[-1].date()),
                realized_price_start=str(realized[0].date()),realized_price_end=str(realized[-1].date()),
                prior_return_count=len(prior)-1,realized_return_count=len(realized)-1)
            pairs.append(dict(**dates,**measured))
            reason = measured['reason']
            group_reasons[reason] = group_reasons.get(reason,0)+1
            reasons[reason] = reasons.get(reason,0)+1
            if measured['source_available']:
                counts['source_available_pairs'] += 1
                counts['valid_price_observations'] += len(forecast)+len(prior)+len(realized)
                counts['simple_return_observations'] += len(forecast)+len(prior)+len(realized)-3
                counts['second_moment_estimates'] += 3
            if measured['available']:
                counts['available_pairs'] += 1
                counts['error_evaluations'] += 2
        usable = [pair for pair in pairs if pair['available']]
        supported = len(usable)>=18
        qc = qp = None
        if supported:
            qc = -positive_divide(total([pair['candidate_error'] for pair in usable]),float(len(usable)))
            qp = -positive_divide(total([pair['control_error'] for pair in usable]),float(len(usable)))
            counts['supported_groups'] += 1
            counts['calibration_estimates'] += 2
        else:
            counts['cold_start_groups'] += 1
        groups.append(dict(group_id=group_id,year=year,symbol=symbol,cutoff=cutoff,
            history_start_month=str(start),history_end_month=str(start+59),
            source_available_pairs=sum(pair['source_available'] for pair in pairs),
            available_pairs=len(usable),status='supported' if supported else 'cold_start',
            raw_candidate=qc,raw_control=qp,pair_reason_counts=group_reasons,pairs=pairs))
    by_id = {group['group_id']:group for group in groups}
    events = []
    for i,(day,names,_) in enumerate(queries):
        cells = []
        for j,symbol in enumerate(names):
            gid = str(day.year)+':'+symbol
            group = by_id[gid]
            supported = group['status']=='supported'
            values['available'][i,j] = supported
            if supported:
                values['raw_candidate'][i,j] = group['raw_candidate']
                values['raw_control'][i,j] = group['raw_control']
            cells.append(dict(symbol=symbol,group_id=gid,available=supported,
                raw_candidate=group['raw_candidate'],raw_control=group['raw_control'],candidate=None,control=None))
        events.append(dict(date=str(day.date()),symbols=names,group_ids=[cell['group_id'] for cell in cells],
            available=[cell['available'] for cell in cells],available_count=sum(cell['available'] for cell in cells),cells=cells))
    for raw,rank in (('raw_candidate','candidate'),('raw_control','control')):
        values[rank] = original_pct_ranks(values[raw],values['available'])
    for i,event in enumerate(events):
        for j,cell in enumerate(event['cells']):
            if cell['available']:
                cell['candidate'] = float(values['candidate'][i,j])
                cell['control'] = float(values['control'][i,j])
    counts.update(checked_cells=int(values['available'].size),available_cells=int(values['available'].sum()),
        unavailable_cells=int((~values['available']).sum()),year_security_groups=len(groups),
        historical_month_pairs=60*len(groups),ranked_observations=2*int(values['available'].sum()))
    audit = dict(schema_version=1,method='annual_asof_risk_forecast_reliability',events=len(queries),
        symbols_per_event=width,history_months=60,min_months=18,forecast_returns=63,
        second_moment_centered=False,calendar_months_equal_weight=True,history_membership_filtered=False,
        pair_reason_counts=reasons,available_by_event=[event['available_count'] for event in events],
        **counts,new_model_fits=counts['calibration_estimates'],predictive_model_fits=counts['calibration_estimates'],
        new_ridge_fits=0,new_tree_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,
        portfolio_returns_calculated=False,group_history=groups,event_history=events)
    return values,audit


def feature_arrays(values):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Feature keys differ')
    a = {key:np.asarray(value) for key,value in values.items()}
    mask = a['available']
    require(mask.ndim==2 and mask.dtype==bool,'Boolean availability required')
    for key in set(VALUE_KEYS)-{'available'}:
        value = a[key]
        require(value.shape==mask.shape and np.issubdtype(value.dtype,np.floating),'Float feature matrix: '+key)
        require(not np.isinf(value).any() and np.array_equal(np.isfinite(value),mask),'Common mask: '+key)
    for raw,rank in (('raw_candidate','candidate'),('raw_control','control')):
        require(np.all(a[raw][mask]>=-1.) and np.all(a[raw][mask]<=0.),'Error mean bounds')
        require(np.array_equal(a[rank],original_pct_ranks(a[raw],mask),equal_nan=True),'Original percentile ranks: '+rank)
    return a


def verify_features(panel,meta,calendar,values,audit,limit=None):
    actual = feature_arrays(values)
    wanted,expected = independent_features(panel,meta,calendar,limit)
    for key in wanted:
        require(np.array_equal(actual[key],wanted[key],equal_nan=True),'Independent feature differs: '+key)
        if key!='available':
            used = np.isfinite(wanted[key])
            require(np.array_equal(np.signbit(actual[key][used]),np.signbit(wanted[key][used])), 'Independent sign differs: '+key)
    record_check(audit,expected)
    keys = ('events','symbols_per_event','available_by_event',*COUNT_FIELDS,
            'new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits')
    return dict(**{key:True for key in FEATURE_FLAGS},**{key:expected[key] for key in keys},
        computes_returns=False,producer_modules_imported=False,numerical_tolerance=0.,
        numerical_comparison='exact scalar annual sixty-month paired second-moment errors, pair-counter ranks and full audit',
        outcome_adapter_sha256=FROZEN_HELPER_SHA)


review_features = verify_features


def original_support(original,n,width):
    require(isinstance(original,dict) and all(key in original for key in
        ('baseline','core','core_weight','state_calm10')),'Original support/state fields required')
    rawb,rawc = np.asarray(original['baseline']),np.asarray(original['core'])
    require(rawb.ndim==2 and rawb.shape[1]==width and len(rawb)>=n and rawc.shape==rawb.shape,'Original score axes')
    full = len(rawb)
    state = np.asarray(original['state_calm10'])
    require(state.shape==(full,) and state.dtype==bool,'Original calm boolean axis')
    weights = np.asarray(original['core_weight'])
    if weights.shape==(full,):weights=weights[:,None]
    require(weights.shape in ((full,1),(full,width)),'Original core weight shape')
    for source in (rawb[:n],rawc[:n],weights[:n]):
        require(all(isinstance(x,numbers.Real) and not isinstance(x,(bool,np.bool_)) for x in source.flat),
                'Original scores must be numeric nonboolean')
    b,c = np.asarray(rawb[:n],float),np.asarray(rawc[:n],float)
    cw = np.broadcast_to(np.asarray(weights[:n],float),(n,width))
    require(np.isfinite(b).all() and np.isfinite(c).all() and np.isfinite(cw).all(),'Finite original support inputs')
    require(np.array_equal(cw,np.broadcast_to(np.where(state[:n],.5,.52)[:,None],cw.shape)), 'Original core weights')
    support = (b-c)/(1.-cw)
    require(np.isfinite(support).all() and np.all(np.abs(support)<=.5+1e-12),'Original support bounds')
    return support,full


def independent_deltas(values,original,ranks,budget=BUDGET):
    require(type(budget) in (float,int) and budget==BUDGET,'Fixed funded .01 budget required')
    a = feature_arrays(values)
    n,width = a['available'].shape
    support,full = original_support(original,n,width)
    require(isinstance(ranks,dict) and set(ranks)==set(RAW),'Rank arms differ')
    result = {}
    for arm,key in RAW.items():
        target = np.asarray(ranks[arm])
        require(target.shape in ((n,width),(full,width)) and np.issubdtype(target.dtype,np.floating),'Floating target axes')
        require(not np.isinf(target).any(),'Infinite target rejected')
        if len(target)>n:require(np.isnan(target[n:]).all(),'Unused target suffix must remain NaN')
        expected = a[key].copy()
        if n:expected[0]=np.nan
        require(np.array_equal(target[:n],expected,equal_nan=True),'Rank prefix or initial partial differs')
        delta = np.zeros((n,width),float)
        for i in range(n):
            for j in range(width):
                if math.isfinite(float(expected[i,j])):
                    delta[i,j] = budget*(float(expected[i,j])-float(support[i,j]))
        require(np.isfinite(delta).all() and np.all(np.abs(delta)<=MAX_DELTA),'Funded auxiliary delta bound')
        result[arm] = delta
    return result


def verify_deltas(values,arrays,ranks,deltas,budget=BUDGET):
    require(isinstance(deltas,dict) and set(deltas)==set(RAW),'Delta arms differ')
    expected = independent_deltas(values,arrays,ranks,budget)
    for arm,wanted in expected.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape==wanted.shape and np.issubdtype(actual.dtype,np.floating) and
            np.isfinite(actual).all() and np.array_equal(actual,wanted),'Exact funded delta differs: '+arm)
    n = np.shape(values['available'])[0]
    return dict(**{key:True for key in DELTA_FLAGS},checked_events=n,budget=BUDGET,max_absolute_delta=MAX_DELTA,
        changed_cells={arm:int(np.count_nonzero(value)) for arm,value in expected.items()},
        target_available_cells={arm:int(np.isfinite(np.asarray(ranks[arm])[:n]).sum()) for arm in RAW})
