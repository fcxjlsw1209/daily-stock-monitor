"""Independent numerical preparation, rank-slot and execution audit for round213.

No production model, strategy or replay engine is imported. Callers supply all
arrays/exports after enforcing their registered input hashes and phase gates.
Ridge systems are solved as augmented least squares rather than normal equations.
The existing independent scalar/accounting auditor is SHA-bound below.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
RAW_KEYS = ('source_log', 'local_linear', 'local_constant', 'candidate', 'control')
ARMS = ('incumbent_local67', 'same_asset_peer_transfer', 'all_asset_peer_control')
NEW_RAW = {'same_asset_peer_transfer': 'candidate', 'all_asset_peer_control': 'control'}
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('identity206_independent_accounting', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': nan mask')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    use = np.isfinite(expected)
    require(np.all(np.abs(actual[use] - expected[use]) <= atol), label + ': values')


def same(actual, expected, label):
    require(np.asarray(actual).shape == np.asarray(expected).shape and
            np.array_equal(actual, expected), label)


def record_check(actual, expected, label):
    """Check every independently produced field, allowing descriptive extra fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list length')
        for i, (a, e) in enumerate(zip(actual, expected)):
            record_check(a, e, label + '/' + str(i))
    elif isinstance(expected, float):
        close(actual, expected, label)
    else:
        require(actual == expected, label + ': differs')


PREPARE_FLAGS = ('passed', 'independent_augmented_least_squares',
    'annual_prior_december_cutoffs_checked', 'five_year_mature_cohort_checked',
    'four_target_common_cohort_checked', 'availability_unchanged',
    'own_history_threshold_checked', 'matched_month_identity_checked',
    'query_security_excluded', 'peer_kernel_checked', 'own_month_mass_preserved',
    'common_shrinkage_penalty_checked', 'no_future_or_outcome_inputs', 'audit_checked',
    'fixed_asset_types_checked', 'same_type_peer_domain_checked',
    'whole_query_fallback_checked', 'cached_all_peer_control_exact',
    'cached_source_and_local_reused', 'no_other_source_targets_or_local_refits')
MODEL_FLAGS = PREPARE_FLAGS
RANK_FLAGS = ('passed', 'full_original_universe', 'includes_eligible_etfs',
    'no_local_history_filter', 'replacement_not_additive_local_slot',
    'paired_whole_row_readiness', 'initial_partial_unchanged', 'exact_rank_rebuild')
COUNT_FIELDS = ('annual_pooled_log_refits', 'cached_forecast_rows',
    'own_weight_evaluations', 'all_peer_query_evaluations',
    'all_peer_month_evaluations', 'all_peer_row_observations',
    'same_type_supported_queries', 'same_type_fallback_queries',
    'same_type_month_evaluations', 'same_type_row_observations',
    'actual_regression_solves')


def finite_scalar(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + name)
    return value


def gaussian(row, query):
    require(len(row) == len(query) == 10, 'Ten local rank features required')
    distance = math.fsum((float(a)-float(b))**2 for a,b in zip(row,query)) / 10.
    return finite_scalar(math.exp(-distance/(2.*.5**2)), 'peer feature weight')


def matched_peer_months(dates, symbols, X, residual, own_indices, query, own_weights, row_ids):
    """Scalar peer selection; original own months and mass are never expanded."""
    months = np.asarray(dates,dtype='datetime64[M]')
    symbols, X, residual = np.asarray(symbols), np.asarray(X), np.asarray(residual)
    require(len(dates)==len(symbols)==len(X)==len(residual)==len(row_ids), 'Peer cohort shape')
    require(len(own_indices)==len(own_weights) and len(own_indices)>0, 'Own month rows required')
    names = {str(symbols[k]) for k in own_indices}
    require(len(names)==1, 'Own rows contain multiple identities')
    owner=next(iter(names)); details=[]; means=[]
    for k,weight in zip(own_indices,own_weights):
        peers=[j for j in range(len(months)) if months[j]==months[k] and str(symbols[j])!=owner]
        require(peers, 'Missing same-month non-self peer')
        u=[gaussian(X[j],query) for j in peers]
        mass=finite_scalar(math.fsum(u),'peer weight mass')
        require(mass>0, 'Nonpositive peer weight mass')
        weighted=finite_scalar(math.fsum(v*float(residual[j]) for v,j in zip(u,peers)), 'peer weighted residual')
        mean=finite_scalar(weighted/mass,'peer mean'); means.append(mean)
        details.append(dict(month=str(months[k]),own_row_id=int(row_ids[k]),own_weight=float(weight),
            own_residual=float(residual[k]),peer_row_ids=[int(row_ids[j]) for j in peers],
            peer_count=len(peers),peer_weight_mass=mass,peer_weighted_residual=weighted,peer_mean=mean))
    numerator=finite_scalar(math.fsum(float(w)*v for w,v in zip(own_weights,means)), 'peer numerator')
    return numerator,details


def least_squares(design, response, weights, penalties):
    """All penalties positive; solve a full-rank augmented system with SVD."""
    A = np.vstack((np.sqrt(weights)[:, None] * design, np.diag(np.sqrt(penalties))))
    b = np.concatenate((np.sqrt(weights) * response, np.zeros(design.shape[1])))
    require(np.isfinite(A).all() and np.isfinite(b).all(), 'Nonfinite ridge system')
    coefficient, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    require(rank == design.shape[1] and np.isfinite(coefficient).all(), 'Invalid ridge solution')
    return coefficient


def fixed_types(etfs):
    require(isinstance(etfs, (list, tuple, set, frozenset)), 'ETF identities must be an explicit collection')
    require(all(isinstance(s, (str, np.str_)) and s and s == s.strip() for s in etfs),
            'ETF identities must be nonempty exact strings')
    require(len(set(etfs)) == len(etfs), 'Duplicate ETF identities')
    return frozenset(str(s) for s in etfs)


def cached_vectors(cached, q):
    expected = {'raw_'+k for k in RAW_KEYS} | {'available', 'local_available'}
    require(isinstance(cached, dict) and set(cached) == expected, 'Exactly seven cached206 fields required')
    answer = {}
    for k in expected:
        value = np.asarray(cached[k])
        require(value.shape == (q,), 'Cached physical query prefix axis: '+k)
        if k in ('available', 'local_available'):
            require(value.dtype == np.dtype(bool), 'Cached boolean mask required: '+k)
        else:
            require(value.dtype.kind == 'f' and value.dtype.itemsize == 8,
                    'Cached binary64 vector required: '+k)
            require(not np.isinf(value).any(), 'Infinite cached vector: '+k)
        answer[k] = value.copy()
    require(not np.any(answer['local_available'] & ~answer['available']), 'Cached local exceeds pooled support')
    for k in RAW_KEYS:
        require(np.array_equal(np.isfinite(answer['raw_'+k]), answer['available']) and
                np.array_equal(np.isnan(answer['raw_'+k]), ~answer['available']), 'Cached raw support differs: '+k)
    close(answer['raw_control'], answer['raw_local_constant'], 'Cached206 own-control identity', atol=0.)
    return answer


def month_rows(months):
    """Insertion-order index, confined to one already mature annual cohort."""
    result={}
    for i,month in enumerate(months):result.setdefault(str(month),[]).append(i)
    return result


def asset_peer_months(dates, symbols, X, residual, own_indices, query, own_weights, row_ids, etfs,
                      *, grouped_months=None):
    """Scalar two-domain summaries; cache indices and weights without changing summation order."""
    types=fixed_types(etfs);months=np.asarray(dates,dtype='datetime64[M]');symbols=np.asarray(symbols)
    require(len(dates)==len(symbols)==len(X)==len(residual)==len(row_ids),'Peer cohort shape')
    require(len(own_indices)>0 and len(own_indices)==len(own_weights), 'Own months required')
    names={str(symbols[k]) for k in own_indices};require(len(names)==1, 'Mixed own identities')
    owner=next(iter(names));is_etf=owner in types
    groups=month_rows(months) if grouped_months is None else grouped_months
    all_details=[];domains=[];weights_by_row={};means=[]
    for k,w in zip(own_indices,own_weights):
        peers=[j for j in groups[str(months[k])] if str(symbols[j])!=owner]
        require(peers,'Missing same-month non-self peer')
        weights=[gaussian(X[j],query) for j in peers]
        for j,weight in zip(peers,weights):weights_by_row[j]=weight
        mass=finite_scalar(math.fsum(weights),'peer weight mass');require(mass>0,'Nonpositive peer weight mass')
        numerator=finite_scalar(math.fsum(weight*float(residual[j]) for weight,j in zip(weights,peers)), 'peer weighted residual')
        mean=finite_scalar(numerator/mass,'peer mean');means.append(mean)
        same_peers=[j for j in peers if (str(symbols[j]) in types)==is_etf];domains.append(same_peers)
        all_details.append(dict(month=str(months[k]),own_row_id=int(row_ids[k]),own_weight=float(w),
            own_residual=float(residual[k]),peer_row_ids=[int(row_ids[j]) for j in peers],peer_count=len(peers),
            peer_weight_mass=mass,peer_weighted_residual=numerator,peer_mean=mean,
            same_peer_row_ids=[int(row_ids[j]) for j in same_peers],same_peer_count=len(same_peers),
            same_peer_weight_mass=None,same_peer_weighted_residual=None,same_peer_mean=None))
    all_num=finite_scalar(math.fsum(float(w)*mean for w,mean in zip(own_weights,means)), 'all-peer numerator')
    missing=[str(months[k]) for k,peers in zip(own_indices,domains) if not peers]
    if missing:return all_num,all_details,None,missing
    means=[]
    for record,peers in zip(all_details,domains):
        weights=[weights_by_row[j] for j in peers]
        mass=finite_scalar(math.fsum(weights),'same-type peer mass');require(mass>0,'Nonpositive same-type peer mass')
        numerator=finite_scalar(math.fsum(weight*float(residual[j]) for weight,j in zip(weights,peers)), 'same-type peer residual')
        mean=finite_scalar(numerator/mass,'same-type peer mean');means.append(mean)
        record.update(same_peer_weight_mass=mass,same_peer_weighted_residual=numerator,same_peer_mean=mean)
    number=finite_scalar(math.fsum(float(w)*mean for w,mean in zip(own_weights,means)), 'same-type numerator')
    return all_num,all_details,number,[]


def reconstruct(dates, ends, symbols, X, targets, common_good, event_dates,
                event_symbols, event_X, event_good, *, etfs, cached_reference,
                min_months=18, min_own_months=18):
    """One independent SVD pooled log Ridge per usable year; cached own-linear is not refitted."""
    require(type(min_months) is int and min_months>0 and type(min_own_months) is int
            and min_own_months>0, 'Positive integer minimums required')
    dates,ends,event_dates=[np.asarray(v,dtype='datetime64[D]') for v in (dates,ends,event_dates)]
    require(all(v.ndim==1 and not np.isnat(v).any() for v in (dates,ends,event_dates)), 'Invalid dates')
    n,q=len(dates),len(event_dates)
    require(ends.shape==dates.shape and (ends>dates).all() and
        np.all(ends.astype('datetime64[M]')==dates.astype('datetime64[M]')+1), 'Invalid label ends')
    X,event_X=np.asarray(X),np.asarray(event_X)
    require(X.shape==(n,10) and event_X.shape==(q,10), 'Ten features required')
    symbols,event_symbols=np.asarray(symbols),np.asarray(event_symbols)
    require(symbols.shape==(n,) and event_symbols.shape==(q,) and
        all(isinstance(v,(str,np.str_)) and v for v in list(symbols)+list(event_symbols)), 'Invalid identities')
    common_good,event_good=np.asarray(common_good),np.asarray(event_good)
    require(common_good.shape==(n,) and event_good.shape==(q,) and
            all(isinstance(v,(bool,np.bool_)) for v in event_good), 'Exact bool masks required')
    require(set(targets)==set(TARGETS), 'All four original targets required')
    ys={k:np.asarray(v) for k,v in targets.items()}
    require(all(y.shape==(n,) for y in ys.values()), 'Target shape differs')
    # Interpret numbers only within the queried annual maturity envelope. Row
    # identities remain unchanged, while future object poison is never cast.
    years=event_dates.astype('datetime64[Y]').astype(int)+1970
    consumed=np.zeros(n,bool)
    if q:
        first=np.datetime64('%04d-12-31'%(int(years.min())-6))
        last=np.datetime64('%04d-12-31'%(int(years.max())-1))
        consumed=(dates>first)&(ends<=last)
    require(all(isinstance(v,(bool,np.bool_)) for v in common_good[consumed]),
            'Consumed common-good mask must be boolean')
    numeric=np.full((n,10),np.nan);numeric[consumed]=np.asarray(X[consumed],float);X=numeric
    for key,y in list(ys.items()):
        numeric=np.full(n,np.nan);numeric[consumed]=np.asarray(y[consumed],float);ys[key]=numeric
    good=np.zeros(n,bool);good[consumed]=common_good[consumed];common_good=good
    event_X=np.asarray(event_X,float);event_good=event_good.astype(bool)
    usable=common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():usable &= np.isfinite(y)
    months=dates.astype('datetime64[M]')
    for month in np.unique(months):
        members=months==month
        if np.count_nonzero(usable & members)<2:usable[members]=False
    years=event_dates.astype('datetime64[Y]').astype(int)+1970
    available,local_available=np.zeros(q,bool),np.zeros(q,bool)
    raw={k:np.full(q,np.nan) for k in RAW_KEYS}
    types=fixed_types(etfs);cache=cached_vectors(cached_reference,q)
    raw={k:cache['raw_'+k].copy() for k in RAW_KEYS[:3]}
    raw.update(candidate=cache['raw_candidate'].copy(),control=cache['raw_candidate'].copy())
    annual,details=[],[];counts={k:0 for k in COUNT_FIELDS}
    counts['cached_forecast_rows']=int(cache['available'].sum())
    for year in np.unique(years):
        cutoff=np.datetime64('%04d-12-31'%(year-1));start=np.datetime64('%04d-12-31'%(year-6))
        ids=np.flatnonzero(usable & (dates>start) & (ends<=cutoff))
        nm=len(np.unique(months[ids]));requests=np.flatnonzero((years==year)&event_good&np.isfinite(event_X).all(axis=1))
        pred=requests if nm>=min_months else np.array([],int)
        row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),
            training_rows=len(ids),training_months=nm,prediction_rows=len(pred),training_row_ids=ids.tolist(),
            first_training_signal=str(dates[ids].min()) if len(ids) else None,
            latest_label_end=str(ends[ids].max()) if len(ids) else None,
            status='fit' if nm>=min_months else 'cold_start',weight_sum=None)
        if len(pred):
            require(np.abs(X[ids]).max()<=.5 and np.abs(event_X[pred]).max()<=.5,'Actual features outside rank units')
            identities=[(str(months[k]),str(symbols[k])) for k in ids]
            require(len(identities)==len(set(identities)), 'Duplicate security/signal month')
            _,inv,num=np.unique(months[ids],return_inverse=True,return_counts=True)
            age=int(cutoff.astype('datetime64[M]').astype(int))-months[ids].astype(int)
            w=(1./num[inv].astype(float))*np.exp2(-age/12.);w*=len(w)/w.sum()
            row['weight_sum']=float(w.sum())
            mean=np.average(X[ids],axis=0,weights=w)
            scale=np.sqrt(np.average((X[ids]-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
            intercept=float(np.average(ys['log_excess'][ids],weights=w));z=(X[ids]-mean)/scale
            coef=least_squares(z,ys['log_excess'][ids]-intercept,w,np.full(10,10.))
            residual=ys['log_excess'][ids]-(np.sum(z*coef,axis=1)+intercept)
            source=np.sum(((event_X[pred]-mean)/scale)*coef,axis=1)+intercept
            require(np.isfinite(residual).all() and np.isfinite(source).all(), 'Nonfinite pooled fit')
            close(cache['raw_source_log'][pred],source,'Cached pooled log replay')
            available[pred]=True;counts['annual_pooled_log_refits']+=1
            grouped_months=month_rows(months[ids])
            for j in pred:
                own=np.flatnonzero(symbols[ids]==event_symbols[j]);own_ids=ids[own]
                own_months=len(np.unique(months[own_ids]))
                detail=dict(event_row=int(j),date=str(event_dates[j]),symbol=str(event_symbols[j]),
                    year=int(year),cutoff=str(cutoff),asset_type='etf' if str(event_symbols[j]) in types else 'stock',
                    own_rows=len(own),own_months=own_months,own_row_ids=own_ids.tolist(),
                    own_dates=dates[own_ids].astype(str).tolist(),own_label_ends=ends[own_ids].astype(str).tolist(),
                    latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                    status='insufficient_own_history',weight_mass=0.,effective_samples=0.,own_weights=[],own_residuals=[],
                    own_numerator=0.,correction_denominator=None,constant_correction=0.,
                    all_peer_numerator=0.,all_peer_correction=0.,
                    same_peer_numerator=None,same_peer_correction=None,missing_same_type_months=[],peer_months=[])
                if own_months>=min_own_months:
                    ages=int(cutoff.astype('datetime64[M]').astype(int))-months[own_ids].astype(int)
                    lw=np.array([float(2.**(-float(age)/12.))*gaussian(X[k],event_X[j]) for k,age in zip(own_ids,ages)])
                    mass=finite_scalar(math.fsum(float(v) for v in lw),'own mass');den=finite_scalar(mass+12.,'denominator')
                    effective=finite_scalar(mass*mass/math.fsum(float(v)*float(v) for v in lw),'effective samples') if mass else 0.
                    own_num=finite_scalar(math.fsum(float(wi)*float(residual[k]) for wi,k in zip(lw,own)),'own numerator')
                    constant=finite_scalar(own_num/den,'own constant')
                    close(cache['raw_local_constant'][j],cache['raw_source_log'][j]+constant,'Cached own-constant replay')
                    all_num,all_details,same_num,missing=asset_peer_months(dates[ids],symbols[ids],X[ids],residual,
                        own,event_X[j],lw,ids,types,grouped_months=grouped_months)
                    all_corr=finite_scalar(all_num/den,'all-peer correction')
                    close(cache['raw_candidate'][j],cache['raw_source_log'][j]+all_corr,'Cached all-peer replay')
                    counts['own_weight_evaluations']+=1;counts['all_peer_query_evaluations']+=1
                    counts['all_peer_month_evaluations']+=len(all_details)
                    counts['all_peer_row_observations']+=sum(p['peer_count'] for p in all_details)
                    local_available[j]=mass>0.
                    same_corr=None
                    if missing:
                        status='same_type_missing_month';counts['same_type_fallback_queries']+=1
                    else:
                        status='same_type_supported';counts['same_type_supported_queries']+=1
                        same_corr=finite_scalar(same_num/den,'same-type correction')
                        raw['candidate'][j]=cache['raw_source_log'][j] if same_corr==0. else finite_scalar(cache['raw_source_log'][j]+same_corr,'same forecast')
                        counts['same_type_month_evaluations']+=len(all_details)
                        counts['same_type_row_observations']+=sum(p['same_peer_count'] for p in all_details)
                    detail.update(status=status,weight_mass=mass,effective_samples=effective,own_weights=lw.tolist(),
                        own_residuals=residual[own].tolist(),own_numerator=own_num,correction_denominator=den,
                        constant_correction=constant,all_peer_numerator=all_num,all_peer_correction=all_corr,
                        peer_months=all_details,same_peer_numerator=same_num,same_peer_correction=same_corr,
                        missing_same_type_months=missing)
                else:
                    for key in ('raw_local_linear','raw_local_constant','raw_candidate'):
                        close(cache[key][j],cache['raw_source_log'][j],'Cached own-history pooled fallback '+key,atol=0.)
                details.append(detail)
        annual.append(row)
    counts['actual_regression_solves']=counts['annual_pooled_log_refits']
    same(available,cache['available'],'Cached pooled availability independently checked')
    same(local_available,cache['local_available'],'Cached local availability independently checked')
    return dict(raw=raw,available=available,local_available=local_available,audit=dict(
        arms=list(ARMS[1:]),raw_keys=list(RAW_KEYS),feature_count=10,target='unchanged cached centered log_excess',
        alpha=10.,half_life_months=12.,min_months=min_months,min_own_months=min_own_months,
        bandwidth=.5,intercept_penalty=12.,etfs=list(etfs),counts=counts,annual=annual,local=details,cached_reference_used=True,
        source_four_target_refits=0,local_linear_refits=0,
        correction_accounting='One pooled log Ridge refit per predicted annual cohort; own constant, all-peer and supported same-type corrections counted separately; cached four-target/local-linear fits are not repeated; prefix/oracle repeats excluded.'))


def check_prepare(train, meta, a, forecast, cached_reference):
    m,tm=meta['monthly'],meta['training']
    symbols=np.array([s for row in tm['symbols'] for s in row]);event_symbols=np.array([s for row in m['symbols'] for s in row])
    for key,metadata,source in [('dates',tm,'dates'),('ends',tm,'ends'),('dates',m,'event_dates')]:
        expanded=np.array([d for d,row in zip(metadata[key],metadata['symbols']) for _ in row],dtype='datetime64[D]')
        same(np.asarray(train[source],dtype='datetime64[D]'),expanded,'Metadata axis '+source)
    eligible=np.asarray(a['eligible']);require(eligible.dtype==np.dtype(bool) and eligible.ndim==2 and
        eligible.shape[0]==len(m['dates']) and len(m['symbols'])==len(m['dates']) and
        all(len(row)==eligible.shape[1] for row in m['symbols']),'Eligibility axis')
    expected=reconstruct(train['dates'],train['ends'],symbols,train['train_plain'],
        {k:train['target_'+k] for k in TARGETS},train['common_good'],train['event_dates'],
        event_symbols,train['event_plain'],eligible.ravel()&np.isfinite(train['event_plain']).all(axis=1),
        etfs=m['etfs'],cached_reference=cached_reference)
    require(set(forecast['raw'])==set(RAW_KEYS),'Raw forecast keys changed')
    for key in RAW_KEYS:close(forecast['raw'][key],expected['raw'][key],'Raw '+key)
    for key in ('available','local_available'):
        require(np.asarray(forecast[key]).dtype==np.dtype(bool),'Forecast mask dtype')
        same(forecast[key],expected[key],'Forecast '+key)
    same(forecast['available'],a['model_available'].ravel(),'Original pooled availability changed')
    same(forecast['local_available'],a['local_available'].ravel(),'Original local availability changed')
    for key in RAW_KEYS[:3]:
        close(forecast['raw'][key],cached_reference['raw_'+key],'Exact cached '+key,atol=0.)
        close(forecast['raw'][key],a['raw_'+key].ravel(),'Original '+key+' copy',atol=0.)
    close(forecast['raw']['control'],cached_reference['raw_candidate'],'Exact cached all-peer control',atol=0.)
    for row in expected['audit']['local']:
        if row['status'] in ('insufficient_own_history','same_type_missing_month'):
            j=row['event_row'];same(forecast['raw']['candidate'][j],forecast['raw']['control'][j],'Whole-query exact raw fallback')
    record_check(forecast['audit'],expected['audit'],'Model audit')
    return dict(**{k:True for k in PREPARE_FLAGS},atol=1e-12,annual_cohorts=len(expected['audit']['annual']),
        local_queries=len(expected['audit']['local']),counts=expected['audit']['counts'],
        independent_pooled_fits=expected['audit']['counts']['annual_pooled_log_refits'],
        residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False,
        cached_reference_used=True,source_four_target_refits=0,local_linear_refits=0,
        correction_accounting=expected['audit']['correction_accounting'])


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
            record_check(wanted['metrics']['annual_returns'], annual, 'Annual returns')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
