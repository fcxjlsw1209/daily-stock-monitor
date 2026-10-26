"""Independent numerical preparation, rank-slot and execution audit for round206.

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
ARMS = ('incumbent_local67', 'peer_residual_transfer', 'own_residual_control')
NEW_RAW = {'peer_residual_transfer': 'candidate', 'own_residual_control': 'control'}
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
    'four_target_common_cohort_checked', 'original_source_and_local_parity',
    'availability_unchanged', 'own_history_threshold_checked', 'matched_month_identity_checked',
    'query_security_excluded', 'peer_kernel_checked', 'own_month_mass_preserved',
    'common_shrinkage_penalty_checked', 'no_future_or_outcome_inputs', 'audit_checked')
RANK_FLAGS = ('passed', 'full_original_universe', 'includes_eligible_etfs',
    'no_local_history_filter', 'replacement_not_additive_local_slot',
    'paired_whole_row_readiness', 'initial_partial_unchanged', 'exact_rank_rebuild')
COUNT_FIELDS = ('source_target_fits', 'reference_pooled_log_refits',
    'reference_local_evaluations', 'reference_local_solves', 'transfer_pooled_log_refits',
    'own_constant_evaluations', 'peer_query_evaluations', 'peer_month_evaluations',
    'peer_row_observations', 'actual_regression_solves')


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


def local_fit(X, query, scale, residual, weights):
    require(X.shape == (len(weights), 10) and query.shape == scale.shape == (10,)
            and residual.shape == weights.shape, 'Invalid local shapes')
    require(all(np.isfinite(v).all() for v in (X, query, scale, residual, weights))
            and (scale > 0).all() and (weights >= 0).all(), 'Invalid local inputs')
    mass = float(weights.sum())
    effective = 0. if mass == 0 else float(mass * mass / np.dot(weights, weights))
    status = 'zero_weight' if mass == 0 else 'zero_residual' if not np.any(residual) else 'fit'
    if status != 'fit':
        return 0., 0., dict(weight_mass=mass, effective_samples=effective, status=status)
    design = np.column_stack((np.ones(len(X)), (X - query) / scale))
    c = least_squares(design, residual, weights, np.array([12.] + [10.] * 10))[0]
    constant = float(np.dot(weights, residual) / (mass + 12.))
    return constant, float(c), dict(weight_mass=mass, effective_samples=effective, status=status)


def reconstruct(dates, ends, symbols, X, targets, common_good, event_dates,
                event_symbols, event_X, event_good, *, min_months=18, min_own_months=18):
    """Independent annual fit and scalar matched-month transfer; no producer import."""
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
    annual,source_annual,details=[],[],[]
    counts={key:0 for key in COUNT_FIELDS}
    for year in np.unique(years):
        cutoff=np.datetime64('%04d-12-31'%(year-1))
        start=np.datetime64('%04d-12-31'%(year-6))
        ids=np.flatnonzero(usable & (dates>start) & (ends<=cutoff))
        nm=len(np.unique(months[ids]))
        requests=np.flatnonzero((years==year)&event_good&np.isfinite(event_X).all(axis=1))
        pred=requests if nm>=min_months else np.array([],int)
        first=str(dates[ids].min()) if len(ids) else None
        last=str(ends[ids].max()) if len(ids) else None
        status='fit' if nm>=min_months else 'cold_start'
        source_row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),
            training_rows=len(ids),training_months=nm,prediction_rows=len(requests),
            latest_label_end=last,first_training_signal=first,
            latest_training_signal=str(dates[ids].max()) if len(ids) else None,status=status)
        annual_row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),
            training_rows=len(ids),training_months=nm,prediction_rows=len(pred),training_row_ids=ids.tolist(),
            first_training_signal=first,latest_label_end=last,status=status,weight_sum=None)
        if len(pred):
            require(np.abs(X[ids]).max()<=.5 and np.abs(event_X[pred]).max()<=.5,'Actual features outside rank units')
            identities=[(str(months[k]),str(symbols[k])) for k in ids]
            require(len(identities)==len(set(identities)), 'Duplicate security/signal month')
            _,inv,num=np.unique(months[ids],return_inverse=True,return_counts=True)
            age=int(cutoff.astype('datetime64[M]').astype(int))-months[ids].astype(int)
            w=1./num[inv].astype(float)
            w*=np.exp2(-age/12.)
            w*=len(w)/w.sum()
            annual_row['weight_sum']=source_row['weight_sum']=float(w.sum())
            mean=np.average(X[ids],axis=0,weights=w)
            scale=np.sqrt(np.average((X[ids]-mean)**2,axis=0,weights=w))
            scale=np.where(scale>1e-12,scale,1.)
            intercept=float(np.average(ys['log_excess'][ids],weights=w))
            z=(X[ids]-mean)/scale
            coef=least_squares(z,ys['log_excess'][ids]-intercept,w,np.full(10,10.))
            residual=ys['log_excess'][ids]-(np.sum(z*coef,axis=1)+intercept)
            source=np.sum(((event_X[pred]-mean)/scale)*coef,axis=1)+intercept
            require(np.isfinite(residual).all() and np.isfinite(source).all(), 'Nonfinite pooled fit')
            for key in RAW_KEYS:raw[key][pred]=source
            available[pred]=True
            counts['source_target_fits']+=4
            counts['reference_pooled_log_refits']+=1
            counts['transfer_pooled_log_refits']+=1
            for j in pred:
                own=np.flatnonzero(symbols[ids]==event_symbols[j]); own_ids=ids[own]
                own_months=len(np.unique(months[own_ids]))
                detail=dict(event_row=int(j),date=str(event_dates[j]),symbol=str(event_symbols[j]),
                    year=int(year),cutoff=str(cutoff),own_rows=len(own),own_months=own_months,
                    own_row_ids=own_ids.tolist(),own_dates=dates[own_ids].astype(str).tolist(),
                    own_label_ends=ends[own_ids].astype(str).tolist(),
                    latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                    status='insufficient_own_history',weight_mass=0.,effective_samples=0.,
                    own_weights=[],own_residuals=[],own_numerator=0.,peer_numerator=0.,
                    correction_denominator=None,constant_correction=0.,linear_correction=0.,
                    peer_correction=0.,peer_months=[])
                if own_months>=min_own_months:
                    ages=int(cutoff.astype('datetime64[M]').astype(int))-months[own_ids].astype(int)
                    lw=np.array([float(2.**(-float(age)/12.))*gaussian(X[k],event_X[j])
                                 for k,age in zip(own_ids,ages)])
                    constant,linear,info=local_fit(X[own_ids],event_X[j],scale,residual[own],lw)
                    counts['reference_local_evaluations']+=1
                    counts['reference_local_solves']+=int(info['status']=='fit')
                    mass=finite_scalar(math.fsum(float(x) for x in lw),'own weight mass')
                    den=finite_scalar(mass+12.,'correction denominator')
                    own_num=finite_scalar(math.fsum(float(wi)*float(residual[k]) for wi,k in zip(lw,own)), 'own numerator')
                    peer_num,peer_months=matched_peer_months(dates[ids],symbols[ids],X[ids],residual,
                                                          own,event_X[j],lw,ids)
                    peer=finite_scalar(peer_num/den,'peer correction')
                    counts['own_constant_evaluations']+=1
                    counts['peer_query_evaluations']+=1
                    counts['peer_month_evaluations']+=len(peer_months)
                    counts['peer_row_observations']+=sum(row['peer_count'] for row in peer_months)
                    detail.update(status='fit',weight_mass=mass,effective_samples=info['effective_samples'],
                        own_weights=lw.tolist(),own_residuals=residual[own].tolist(),own_numerator=own_num,
                        peer_numerator=peer_num,correction_denominator=den,constant_correction=constant,
                        linear_correction=linear,peer_correction=peer,peer_months=peer_months)
                    local_available[j]=mass>0.
                    for key,corr in (('local_constant',constant),('control',constant),('local_linear',linear),('candidate',peer)):
                        if corr!=0.:raw[key][j]=finite_scalar(raw[key][j]+corr,'corrected forecast')
                details.append(detail)
        annual.append(annual_row);source_annual.append(source_row)
    counts['actual_regression_solves']=sum(counts[k] for k in ('source_target_fits',
        'reference_pooled_log_refits','reference_local_solves','transfer_pooled_log_refits'))
    require(all(np.array_equal(np.isfinite(v),available) for v in raw.values()),'Reconstructed availability differs')
    source_audit=dict(alpha=10.,half_life_months=12.,min_months=min_months,lookback_years=5,
        targets=list(TARGETS),shared_rows_and_weights=True,training_only_standardization=True,
        weighting='equal_month_total_then_age_decay_normalized_to_row_count',predictions_are_raw=True,
        label_ranks_used=False,annual=source_annual)
    return dict(raw=raw,available=available,local_available=local_available,audit=dict(
        arms=list(ARMS[1:]),raw_keys=list(RAW_KEYS),feature_count=10,target='unchanged cached centered log_excess',
        alpha=10.,half_life_months=12.,min_months=min_months,min_own_months=min_own_months,
        bandwidth=.5,intercept_penalty=12.,counts=counts,source=source_audit,annual=annual,local=details))


def check_prepare(train, meta, a, forecast):
    m, tm = meta['monthly'], meta['training']
    symbols = np.array([s for row in tm['symbols'] for s in row])
    event_symbols = np.array([s for row in m['symbols'] for s in row])
    for key, metadata, source in [('dates', tm, 'dates'), ('ends', tm, 'ends'), ('dates', m, 'event_dates')]:
        expanded = np.array([d for d, row in zip(metadata[key], metadata['symbols']) for _ in row], dtype='datetime64[D]')
        same(np.asarray(train[source], dtype='datetime64[D]'), expanded, 'Metadata axis ' + source)
    eligible=np.asarray(a['eligible'])
    require(eligible.dtype == np.dtype(bool) and eligible.ndim==2 and
            eligible.shape[0]==len(m['dates']) and len(m['symbols'])==len(m['dates']) and
            all(len(row)==eligible.shape[1] for row in m['symbols']), 'Eligibility axis')
    expected = reconstruct(train['dates'], train['ends'], symbols, train['train_plain'],
        {k: train['target_' + k] for k in TARGETS}, train['common_good'], train['event_dates'],
        event_symbols, train['event_plain'], a['eligible'].ravel() & np.isfinite(train['event_plain']).all(axis=1))
    require(set(forecast['raw']) == set(RAW_KEYS), 'Raw forecast keys changed')
    for key in RAW_KEYS:
        close(forecast['raw'][key], expected['raw'][key], 'Raw ' + key)
    for key in ('available', 'local_available'):
        require(np.asarray(forecast[key]).dtype == np.dtype(bool), 'Forecast mask dtype')
        same(forecast[key], expected[key], 'Forecast ' + key)
    same(forecast['available'], a['model_available'].ravel(), 'Original pooled availability changed')
    same(forecast['local_available'], a['local_available'].ravel(), 'Original local availability changed')
    for key in ('source_log', 'local_linear', 'local_constant'):
        close(forecast['raw'][key], a['raw_' + key].ravel(), 'Original ' + key + ' parity')
    require(np.array_equal(forecast['raw']['control'],forecast['raw']['local_constant'],equal_nan=True),
            'Control must copy original own-constant forecast exactly')
    fallback=forecast['available'] & ~forecast['local_available']
    for key in ('candidate','control'):
        same(forecast['raw'][key][fallback],forecast['raw']['source_log'][fallback], 'Exact no-own-history source fallback')
    record_check(forecast['audit'], expected['audit'], 'Model audit')
    return dict(**{key:True for key in PREPARE_FLAGS},atol=1e-12,
        annual_cohorts=len(expected['audit']['annual']),local_queries=len(expected['audit']['local']),
        counts=expected['audit']['counts'], independent_pooled_fits=expected['audit']['counts']['transfer_pooled_log_refits'],
        residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False)


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
