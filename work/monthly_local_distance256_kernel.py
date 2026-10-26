"""Pure caller-supplied diagonal-distance/local-linear contrast; no loader or replay."""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'outputs/monthly_local_linear70/model.py'
MODEL_SHA='6d9075e5ce08edd0587d2b2ea0d98635620141070f1af08771b6a69435467a8a'
BRIDGE_PATH=ROOT/'work/monthly_calendar_clock248_v2_kernel.py'
BRIDGE_SHA='b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb'

def _module(name,path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

model=_module('distance256_original_local_model',MODEL_PATH,MODEL_SHA)
bridge=_module('distance256_original_cache_bridge',BRIDGE_PATH,BRIDGE_SHA)
require=bridge.require
TARGETS=tuple(model.TARGETS)
ARMS=('incumbent_local67','diag_mass_local67')
RAW_KEYS=('source_log','local_constant','local_linear')
RANK_KEYS=('old_pool','old_local','new_pool','new_local')
INITIAL_PARTIAL='2016-08-17'
POLICY=dict(budget=.05,ceiling=5,gap=.02)
BUNDLE_KEYS=('schema','n','width','training_row_ids','control','candidate','label_values','label_preservation','control_gate','audit')
LABEL_KEYS=('schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed')
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','control_target_fits','control_log_refits','control_local_evaluations','control_local_solves','candidate_pooled_fits','candidate_log_refits','candidate_local_evaluations','candidate_local_solves','candidate_kernel_evaluations','candidate_equal_scale_evaluations','candidate_changed_weight_evaluations','actual_regression_solves')
GEOMETRY_KEYS=('source_scale','coefficients','old_distance','distance','original_weights','unscaled_weights','weights','equal_scales','original_mass','unscaled_mass','mass_multiplier','matched_mass','effective_samples','original_effective_samples','mass_absolute_error','maximum_absolute_weight_change','weights_exact_original')
CANDIDATE_ANNUAL_KEYS=('year','cutoff','training_row_ids','query_row_ids','source_scale')
CANDIDATE_LOCAL_KEYS=('event_row','date','symbol','year','cutoff','own_rows','own_months','latest_label_end','status','weight_mass','effective_samples','constant_correction','linear_correction','own_row_ids','geometry')
CANDIDATE_AUDIT_KEYS=('schema','feature_count','target','pooled_copied','local_constant_copied','residuals','local_linear_only','annual','local')
MASS_RTOL=16*np.finfo(float).eps
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells','changed_delta_cells','changed_holdings_events','incumbent_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','changed_delta_indices','delta','incumbent_choices','candidate_choices','episode_budget')

def dependency_files():
    return tuple(dict.fromkeys((MODEL_PATH,BRIDGE_PATH)+tuple(model.dependency_files())+tuple(bridge.dependency_files())))

def _days(values,name):
    a=np.asarray(values)
    require(a.ndim==1,name+' vector')
    out=[]
    for x in a:
        if isinstance(x,np.datetime64):
            require(not np.isnat(x),name+' NaT');s=str(x.astype('datetime64[D]'))
            require(x==np.datetime64(s),name+' whole days')
        else:
            require(isinstance(x,(str,np.str_)),name+' ISO date');s=str(x)
        require(date.fromisoformat(s).isoformat()==s,name+' canonical ISO');out.append(s)
    return np.asarray(out,dtype='datetime64[D]')

def _symbols(values,n,name):
    a=np.asarray(values);require(a.shape==(n,) and all(isinstance(s,(str,np.str_)) and s and str(s).strip()==s for s in a),name+' identities')
    return a.astype(str,copy=True)

def _bool(values,n,name):
    a=np.asarray(values);require(a.shape==(n,) and a.dtype==np.dtype(bool),name+' exact bool');return a.copy()

def _inputs(train,meta,arrays,n=None):
    require(type(train) is dict and type(meta) is dict and set(('monthly','training'))<=set(meta),'Combined metadata')
    m,tm=meta['monthly'],meta['training'];shape=np.shape(arrays['baseline'])
    require(len(shape)==2 and shape[1]>0,'Original monthly shape');total,w=shape
    n=total if n is None else n;require(type(n) is int and 0<=n<=total,'Monthly prefix')
    require(all(isinstance(m.get(k),list) and len(m[k])==total for k in ('dates','ends','symbols')),'Monthly axes')
    qdates=_days(m['dates'][:n],'query dates');qends=_days(m['ends'][:n],'query ends')
    require(np.all(qends>qdates) and (n<2 or np.all(qdates[1:]>qdates[:-1])),'Query chronology')
    require(all(isinstance(s,list) and len(s)==w and len(set(s))==w for s in m['symbols'][:n]),'Query symbols')
    event_symbols=_symbols([s for pool in m['symbols'][:n] for s in pool],n*w,'Query symbols')
    require(isinstance(tm.get('dates'),list) and isinstance(tm.get('ends'),list) and isinstance(tm.get('symbols'),list) and len(tm['dates'])==len(tm['ends'])==len(tm['symbols']),'Training metadata axes')
    td=_days(tm['dates'],'metadata signals');te=_days(tm['ends'],'metadata ends')
    widths=np.asarray([len(s) for s in tm['symbols']],dtype=int)
    require((widths>0).all() and all(len(s)==len(set(s)) for s in tm['symbols']),'Training identity widths')
    dates=_days(train['dates'],'training dates');ends=_days(train['ends'],'training ends');rows=len(dates)
    require(np.array_equal(dates,np.repeat(td,widths)) and np.array_equal(ends,np.repeat(te,widths)),'Training metadata dates exact')
    require(np.array_equal(np.asarray(train['offsets']),np.r_[0,np.cumsum(widths)]),'Training offsets exact')
    require(np.all(ends>dates) and np.all(ends.astype('datetime64[M]')==dates.astype('datetime64[M]')+1),'Next-month label ends')
    require(np.asarray(train['train_plain']).shape==(rows,10) and np.asarray(train['event_plain']).shape==(total*w,10),'Original plain10 shapes')
    require(np.asarray(train['event_dates']).shape==(total*w,),'Query date shape')
    event_dates=_days(np.asarray(train['event_dates'])[:n*w],'event dates')
    require(np.array_equal(event_dates,np.repeat(qdates,w)),'Query date parity')
    elig=np.asarray(arrays['eligible']);require(elig.shape==shape and elig.dtype==np.dtype(bool),'Original eligible bool')
    # Numeric training bodies beyond this prefix's latest annual cutoff are not touched.
    cutoff=np.datetime64('%04d-12-31'%(int(str(qdates[-1])[:4])-1)) if n else None
    keep=ends<=cutoff if n else np.zeros(rows,bool);ids=np.flatnonzero(keep)
    all_symbols=np.asarray([s for pool in tm['symbols'] for s in pool]);symbols=_symbols(all_symbols[keep],len(ids),'Consumed training symbols')
    X=np.asarray(np.asarray(train['train_plain'])[keep],dtype=float).copy()
    good=_bool(np.asarray(train['common_good'])[keep],len(ids),'Consumed common_good')
    targets={}
    for key in TARGETS:
        a=np.asarray(train['target_'+key]);require(a.shape==(rows,),'Training target axes '+key)
        targets[key]=np.asarray(a[keep],dtype=float).copy()
    query=np.asarray(np.asarray(train['event_plain'])[:n*w],dtype=float).copy()
    egood=elig[:n].reshape(-1)&np.isfinite(query).all(axis=1)
    args=(dates[keep].copy(),ends[keep].copy(),symbols,X,targets,good,event_dates,event_symbols,query,egood)
    return args,ids,n,w

def model_inputs(train,meta,arrays,n=None):return _inputs(train,meta,arrays,n)[0]

def _forecast_layout(f,n,w,limit=None):
    consumed=n if limit is None else limit
    require(type(consumed) is int and 0<=consumed<=n,'Forecast consumed prefix')
    q=consumed*w
    require(type(f) is dict and set(f)=={'raw','available','local_available','audit'} and set(f['raw'])==set(RAW_KEYS),'Forecast schema')
    for k in ('available','local_available'):
        a=np.asarray(f[k]);require(a.shape==(n*w,) and a.dtype==np.dtype(bool),k+' exact bool layout')
    require(not np.any(f['local_available'][:q]&~f['available'][:q]),'Local exceeds pooled support')
    for k in RAW_KEYS:
        x=np.asarray(f['raw'][k]);require(x.shape==(n*w,) and x.dtype==np.dtype(float),'Raw layout '+k)
        require(not np.isinf(x[:q]).any() and np.array_equal(np.isfinite(x[:q]),f['available'][:q]),'Raw support '+k)

def _rank(values,use,fallback):
    out=fallback.copy()
    for i in range(len(out)):
        if use[i].any():out[i,use[i]]=pd.Series(values[i]).where(use[i]).rank(pct=True).fillna(.5).to_numpy()[use[i]]-.5
    return out

def _old_ranks(arrays,n,w):
    shape=np.shape(arrays['baseline']);good=bridge._bool(arrays['eligible'],shape,n,'eligible')
    fa=bridge._bool(arrays['forecast_available'],shape,n,'forecast_available');use=good&fa
    pool=bridge._matrix(arrays['rank_log_excess'],shape,n,'pooled ranks');require((np.abs(pool)<=.5+1e-12).all(),'Original rank bounds')
    raw=bridge._matrix(arrays['raw_source_log'],shape,n,'pooled raw',False)
    require(np.isfinite(raw[use]).all(),'Original pooled support finite')
    np.testing.assert_array_equal(_rank(raw,use,pool),pool)
    ma=bridge._bool(arrays['model_available'],shape,n,'model_available')
    lr=bridge._matrix(arrays['raw_local_linear'],shape,n,'local raw',False)
    cr=bridge._matrix(arrays['raw_local_constant'],shape,n,'constant raw',False)
    require(not np.isinf(lr).any() and not np.isinf(cr).any(),'Infinite original local prediction')
    ready=use.any(axis=1)&np.all(~use|(ma&np.isfinite(lr)&np.isfinite(cr)),axis=1)
    local=pool.copy()
    for i in range(n):
        if ready[i] and not np.array_equal(lr[i,use[i]],raw[i,use[i]]):local[i]=_rank(lr[i:i+1],use[i:i+1],pool[i:i+1])[0]
    if 'rank_local_linear' in arrays:np.testing.assert_array_equal(local,np.asarray(arrays['rank_local_linear'])[:n])
    if 'paired_month_ready' in arrays:np.testing.assert_array_equal(ready,np.asarray(arrays['paired_month_ready'])[:n])
    return pool,local,use,ready

def control_parity(arrays,forecast,n=None):
    total,w=np.shape(arrays['baseline']);n=total if n is None else n;_forecast_layout(forecast,n,w)
    for k,original in (('available','model_available'),('local_available','local_available')):
        np.testing.assert_array_equal(forecast[k],np.asarray(arrays[original])[:n].reshape(-1))
    np.testing.assert_array_equal(forecast['available'],np.asarray(arrays['forecast_available'])[:n].reshape(-1))
    for k in RAW_KEYS:np.testing.assert_array_equal(forecast['raw'][k],np.asarray(arrays['raw_'+k])[:n].reshape(-1))
    if 'raw_log_excess' in arrays:np.testing.assert_allclose(forecast['raw']['source_log'],np.asarray(arrays['raw_log_excess'])[:n].reshape(-1),rtol=0,atol=1e-12,equal_nan=True)
    _old_ranks(arrays,n,w)
    return dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False)

def _fit_counts(f):
    src=f['audit']['source']['annual'];annual=f['audit']['annual'];local=f['audit']['local']
    return (4*sum(x['status']=='fit' and x['prediction_rows']>0 for x in src),sum(x['status']=='fit' and x['prediction_rows']>0 for x in annual),len(local),sum(x['status']=='fit' for x in local))

def _control_forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
                     event_symbols, event_X, event_good, *, min_months=model.MIN_MONTHS,
                     min_own_months=model.MIN_OWN_MONTHS):
    """Return source_log/local_constant/local_linear raw vectors and audit.

    `available` is independently reproduced original pooled availability. Both
    local outputs start as exact copies of its raw predictions. `local_available`
    only audits sufficient own history/positive local mass; it MUST NOT filter
    event eligibility or cross-sectional ranking population. Labels are neither
    rebuilt nor recentered. Local residuals are in-sample in the already mature
    annual cohort, not historical OOF residuals. Production defaults are fixed;
    the two minimum-month arguments support small synthetic tests only.
    """
    model._minimum(min_months, 'min_months')
    model._minimum(min_own_months, 'min_own_months')
    dates, ends = model.numeric._days(dates, 'dates'), model.numeric._days(ends, 'ends')
    event_dates = model.numeric._days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or (ends <= dates).any() or np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    symbols, event_symbols = model._symbols(symbols, n, 'symbols'), model._symbols(event_symbols, q, 'event_symbols')
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    if X.shape != (n, 10) or event_X.shape != (q, 10):
        raise ValueError('Original ten-feature matrices are required')
    common_good = model.original._bool(common_good, n, 'common_good')
    event_good = model.original._bool(event_good, q, 'event_good')
    if set(targets) != set(TARGETS):
        raise ValueError('All four original target arrays are required')
    ys = {key: np.asarray(targets[key], float) for key in TARGETS}
    if any(y.shape != (n,) for y in ys.values()):
        raise ValueError('Original targets must have exact training row length')
    usable = common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        usable &= np.isfinite(y)
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        selected = months == month
        if (usable & selected).sum() < 2:
            usable[selected] = False
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    annual, local_audit, contexts = [], [], []
    with threadpool_limits(limits=1):
        ref = model.original.annual_forecasts(dates, ends, X, ys, common_good, event_dates,
                                        event_X, event_good, min_months=min_months)
        raw = {key: ref['log_excess'].copy() for key in ('source_log',)+model.MODES}
        available = ref['available'].copy()
        local_available = np.zeros(q, bool)
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = model.numeric._years_before(cutoff)
            train = usable & (dates > start) & (ends <= cutoff)
            predict = (years == year) & available
            ids = np.flatnonzero(train)
            count = model.numeric._month_count(dates[train])
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                          training_rows=len(ids), training_months=count, prediction_rows=int(predict.sum()),
                          first_training_signal=str(dates[train].min()) if len(ids) else None,
                          latest_label_end=str(ends[train].max()) if len(ids) else None,
                          status='fit' if count >= min_months else 'cold_start')
            if predict.any():
                if count < min_months:
                    raise AssertionError('Independent pooled availability disagrees with source cohort')
                # Only actual mature/predicted rows are rank-validated. Poisoned
                # unused/future rows cannot make an earlier year fail.
                model._ranks('actual pooled training ranks', X[train])
                model._ranks('actual query ranks', event_X[predict])
                identities = list(zip(months[train].astype(str), symbols[train]))
                if len(set(identities)) != len(identities):
                    raise ValueError('Each actual training security must have one row per signal month')
                weights = model.numeric.month_balanced_weights(dates[train], cutoff, model.HALF_LIFE_MONTHS)
                pooled = model._fit_pooled(X[train], ys['log_excess'][train], weights)
                replay = model.numeric._predict(pooled, event_X[predict])
                np.testing.assert_array_equal(replay, raw['source_log'][predict])
                with np.errstate(over='raise', invalid='raise'):
                    residual = ys['log_excess'][train]-model.numeric._predict(pooled, X[train])
                model._finite('mature pooled residuals', residual)
                record['weight_sum'] = float(weights.sum())
                by_symbol = {s: np.flatnonzero(symbols[train] == s) for s in set(symbols[train])}
                contexts.append(dict(year=int(year),cutoff=cutoff,ids=ids.copy(),pooled=pooled,residual=residual.copy(),by_symbol=by_symbol,predict=np.flatnonzero(predict)))
                for j in np.flatnonzero(predict):
                    own = by_symbol.get(event_symbols[j], np.empty(0, dtype=int))
                    own_ids = ids[own]
                    own_months = model.numeric._month_count(dates[own_ids])
                    detail = dict(event_row=int(j), date=str(event_dates[j]), symbol=event_symbols[j],
                                  year=int(year), cutoff=str(cutoff), own_rows=len(own), own_months=own_months,
                                  latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                                  status='insufficient_own_history', weight_mass=0., effective_samples=0.,
                                  constant_correction=0., linear_correction=0.)
                    if own_months >= min_own_months:
                        w = model.local_weights(dates[own_ids], cutoff, X[own_ids], event_X[j])
                        constant, linear, info = model.local_corrections(X[own_ids], event_X[j], pooled[1], residual[own], w)
                        detail.update(info, constant_correction=constant, linear_correction=linear)
                        local_available[j] = info['weight_mass'] > 0.
                        for mode, correction in zip(model.MODES, (constant, linear)):
                            if correction != 0.:
                                with np.errstate(over='raise', invalid='raise'):
                                    raw[mode][j] += correction
                    local_audit.append(detail)
            annual.append(record)
    for key, values in raw.items():
        if not np.array_equal(np.isfinite(values), available):
            raise ValueError(key+' changed source prediction availability or became nonfinite')
    result=dict(raw=raw, available=available, local_available=local_available, audit=dict(
        modes=list(model.MODES), feature_count=10, target='unchanged cached centered log_excess',
        alpha=model.ALPHA, half_life_months=model.HALF_LIFE_MONTHS, min_months=int(min_months),
        min_own_months=int(min_own_months), bandwidth=model.BANDWIDTH,
        intercept_penalty=model.INTERCEPT_PENALTY, slope_penalty=model.SLOPE_PENALTY,
        same_local_rows_residuals_and_weights=True, local_weights_normalized=False,
        local_recency_clock='annual prior December cutoff calendar-month age',
        kernel='mean squared difference in original centered percentile plain10 units',
        linear_design='query-centered original X differences divided by pooled annual training scales',
        residuals='in-sample residuals within already matured annual training cohort, not OOF',
        source_control='independent original four-target annual model and exact log fit replay',
        no_own_history='exact source raw copy; never remove a query or alter eligibility',
        source=ref['audit'], annual=annual, local=local_audit))

    return result,contexts

def diagonal_weights(dates,cutoff,X,query,scale):
    """Canonical fixed metric; return weights and complete per-row evidence."""
    X=np.asarray(X,float);query=np.asarray(query,float);scale=np.asarray(scale,float)
    require(X.ndim==2 and X.shape[1]==10 and len(X)>0 and query.shape==(10,) and scale.shape==(10,),'Kernel axes')
    require(np.isfinite(scale).all() and (scale>0.).all(),'Positive finite original effective scales')
    original=model.local_weights(dates,cutoff,X,query)
    require(np.isfinite(original).all() and (original>0.).all(),'Positive finite original kernel support')
    dates=model.numeric._days(dates,'own dates');cutoff=np.datetime64(cutoff,'D')
    ages=cutoff.astype('datetime64[M]').astype(int)-dates.astype('datetime64[M]').astype(int)
    equal=bool(np.all(scale==scale[0]))
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        squared=(X-query)**2;old_distance=np.mean(squared,axis=1);old_mass=float(original.sum())
        if equal:
            coefficients=np.full(10,.1);distance=old_distance.copy();u=original.copy();weights=original.copy();u_mass=old_mass;multiplier=1.
        else:
            inverse=1./(scale*scale);coefficients=inverse/inverse.sum()
            require(np.isfinite(coefficients).all() and (coefficients>0.).all(),'Positive finite metric coefficients')
            distance=np.sum(squared*coefficients,axis=1)
            u=np.exp2(-ages/model.HALF_LIFE_MONTHS)*np.exp(-distance/(2*model.BANDWIDTH**2))
            u_mass=float(u.sum());require(np.isfinite(u_mass) and u_mass>0.,'Positive finite unscaled mass')
            multiplier=old_mass/u_mass;weights=u*multiplier
        mass=float(weights.sum());effective=float(mass**2/np.einsum('n,n->',weights,weights,optimize=False))
        original_effective=float(old_mass**2/np.einsum('n,n->',original,original,optimize=False))
    require(np.isfinite(distance).all() and (distance>=0.).all() and np.isfinite(u).all() and (u>0.).all(),'Finite candidate geometry')
    require(np.isfinite(weights).all() and (weights>0.).all() and np.isfinite(multiplier),'Positive finite candidate kernel support')
    error=abs(mass-old_mass);require(error<=MASS_RTOL*old_mass,'Mass matching tolerance exceeded')
    detail=dict(source_scale=scale.tolist(),coefficients=coefficients.tolist(),old_distance=old_distance.tolist(),distance=distance.tolist(),original_weights=original.tolist(),unscaled_weights=u.tolist(),weights=weights.tolist(),equal_scales=equal,original_mass=old_mass,unscaled_mass=u_mass,mass_multiplier=multiplier,matched_mass=mass,effective_samples=effective,original_effective_samples=original_effective,mass_absolute_error=error,maximum_absolute_weight_change=float(np.max(np.abs(weights-original))),weights_exact_original=weights.tobytes()==original.tobytes())
    return weights,detail


def _linear_only(X,query,scale,residual,weights):
    """Same original penalized local-linear equations; no constant-model fit."""
    mass=float(weights.sum())
    if mass==0. or not np.any(residual!=0.):return 0.,dict(weight_mass=mass,effective_samples=0. if mass==0. else float(mass**2/np.einsum('n,n->',weights,weights,optimize=False)),status='zero_weight' if mass==0. else 'zero_residual')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        d=(X-query)/scale;Z=np.column_stack((np.ones(len(X)),d))
        lhs=np.einsum('ni,n,nj->ij',Z,weights,Z,optimize=False)
        lhs+=np.diag([model.INTERCEPT_PENALTY]+[model.SLOPE_PENALTY]*10)
        rhs=np.einsum('ni,n,n->i',Z,weights,residual,optimize=False)
        effective=float(mass**2/np.einsum('n,n->',weights,weights,optimize=False))
    for name,values in (('candidate design',d),('candidate lhs',lhs),('candidate rhs',rhs),('candidate effective samples',effective)):model._finite(name,values)
    theta=np.linalg.solve(lhs,rhs);model._finite('candidate local coefficients',theta)
    return float(theta[0]),dict(weight_mass=mass,effective_samples=effective,status='fit')


def candidate_forecasts(args,control,contexts,*,min_own_months=18):
    dates,ends,symbols,X,targets,good,event_dates,event_symbols,event_X,event_good=args
    raw={key:control['raw'][key].copy() for key in RAW_KEYS};raw['local_linear']=raw['source_log'].copy()
    local_available=np.zeros(len(event_dates),bool);annual=[];local=[]
    original_details={x['event_row']:x for x in control['audit']['local']}
    with threadpool_limits(limits=1):
        for ctx in contexts:
            year,cutoff,ids,pooled,residual,by_symbol,predict=(ctx[k] for k in ('year','cutoff','ids','pooled','residual','by_symbol','predict'))
            annual.append(dict(year=year,cutoff=str(cutoff),training_row_ids=ids.tolist(),query_row_ids=predict.tolist(),source_scale=pooled[1].tolist()))
            for j in predict:
                own=by_symbol.get(event_symbols[j],np.empty(0,dtype=int));own_ids=ids[own]
                detail=deepcopy(original_details[int(j)])
                detail.update(status='insufficient_own_history',weight_mass=0.,effective_samples=0.,linear_correction=0.,own_row_ids=own_ids.tolist(),geometry=None)
                if detail['own_months']>=min_own_months:
                    weights,geometry=diagonal_weights(dates[own_ids],cutoff,X[own_ids],event_X[j],pooled[1])
                    linear,info=_linear_only(X[own_ids],event_X[j],pooled[1],residual[own],weights)
                    detail.update(info,linear_correction=linear,geometry=geometry);local_available[j]=info['weight_mass']>0.
                    if linear!=0.:
                        with np.errstate(over='raise',invalid='raise'):raw['local_linear'][j]+=linear
                local.append(detail)
    for key,v in raw.items():require(np.array_equal(np.isfinite(v),control['available']),'Candidate raw support changed '+key)
    np.testing.assert_array_equal(local_available,control['local_available'])
    return dict(raw=raw,available=control['available'].copy(),local_available=local_available,audit=dict(schema='monthly_local_distance256_candidate_audit_v1',feature_count=10,target='unchanged cached centered log_excess',pooled_copied=True,local_constant_copied=True,residuals='same mature in-sample pooled residuals as control',local_linear_only=True,annual=annual,local=local))


def _target_digest(values):return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def fit_models(train,meta,arrays,n=None,*,control_validator=None,min_months=18,min_own_months=18):
    args,ids,n,w=_inputs(train,meta,arrays,n);snapshot=bridge._snapshot(arrays)
    dates,ends,symbols,X,targets,good,event_dates,event_symbols,event_X,event_good=args
    identity=list(zip(dates.astype('datetime64[M]').astype(str),symbols.tolist()))
    require(len(identity)==len(set(identity)),'Duplicate security-month')
    require(np.isfinite(targets['log_excess'][good]).all(),'Nonfinite valid original log target')
    before={key:_target_digest(targets[key]) for key in TARGETS};good_before=_target_digest(good)
    control,contexts=_control_forecasts(*args,min_months=min_months,min_own_months=min_own_months)
    parity=control_parity(arrays,control,n);receipt=parity
    if control_validator is not None:
        require(callable(control_validator),'Control validator callable');receipt=control_validator(deepcopy(control))
        require(type(receipt) is dict and receipt.get('passed') is True,'Independent original model gate failed')
    candidate=candidate_forecasts(args,control,contexts,min_own_months=min_own_months)
    _forecast_layout(candidate,n,w)
    for key in ('source_log','local_constant'):require(candidate['raw'][key].tobytes()==control['raw'][key].tobytes(),'Preserved candidate forecast changed '+key)
    require(before=={key:_target_digest(targets[key]) for key in TARGETS} and good_before==_target_digest(good),'Original labels/common_good changed')
    lp=dict(schema='monthly_local_distance256_labels_preserved_v1',rows=len(ids),valid_rows=int(good.sum()),target_keys=list(TARGETS),target_sha256=before,common_good_sha256=good_before,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)
    counts=dict.fromkeys(MODEL_COUNT_FIELDS,0);counts.update(training_rows=len(ids),valid_training_rows=int(good.sum()))
    for suffix,value in zip(('target_fits','log_refits','local_evaluations','local_solves'),_fit_counts(control)):counts['control_'+suffix]=int(value)
    details=candidate['audit']['local'];geometries=[x['geometry'] for x in details if x['geometry'] is not None]
    counts.update(candidate_local_evaluations=len(details),candidate_local_solves=sum(x['status']=='fit' for x in details),candidate_kernel_evaluations=len(geometries),candidate_equal_scale_evaluations=sum(x['equal_scales'] for x in geometries),candidate_changed_weight_evaluations=sum(not x['weights_exact_original'] for x in geometries))
    counts['actual_regression_solves']=counts['control_target_fits']+counts['control_log_refits']+counts['control_local_solves']+counts['candidate_local_solves']
    require(snapshot==bridge._snapshot(arrays),'Original arrays changed')
    return dict(schema='monthly_local_distance256_models_v1',n=n,width=w,training_row_ids=ids,control=control,candidate=candidate,label_values=targets['log_excess'].copy(),label_preservation=lp,control_gate=deepcopy(receipt),audit=dict(schema='monthly_local_distance256_model_audit_v1',counts=counts,original_cache_parity=parity,original_row_order_preserved=True,original_common_good_preserved=True,all_targets_preserved=True,candidate_pooled_copied=True,candidate_local_constant_copied=True,candidate_local_linear_only=True,equal_scale_original_weights_exact=True,mass_relative_tolerance=float(MASS_RTOL),residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False))

def rank_and_delta(arrays,meta,bundle,n=None):
    total,w=np.shape(arrays['baseline']);n=total if n is None else n
    require(type(n) is int and 0<=n<=total and type(bundle) is dict and bundle.get('schema')=='monthly_local_distance256_models_v1' and type(bundle.get('n')) is int and n<=bundle['n'] and bundle.get('width')==w,'Model prefix axes')
    require(type(meta) is dict and isinstance(meta.get('dates'),list) and len(meta['dates'])==total,'Rank metadata')
    old_pool,old_local,use,old_ready=_old_ranks(arrays,n,w)
    f=bundle['candidate'];_forecast_layout(f,bundle['n'],w,limit=n)
    available=f['available'][:n*w].reshape(n,w);pool=f['raw']['source_log'][:n*w].reshape(n,w);local=f['raw']['local_linear'][:n*w].reshape(n,w)
    require(np.array_equal(available,np.asarray(arrays['model_available'])[:n]),'Candidate pooled support changed')
    oldraw=np.asarray(arrays['raw_source_log'])[:n].reshape(n,w)
    np.testing.assert_array_equal(pool,oldraw)
    np.testing.assert_array_equal(f['raw']['local_constant'][:n*w],np.asarray(arrays['raw_local_constant'])[:n].reshape(-1))
    ready=use.any(axis=1)&np.all(~use|(available&np.isfinite(pool)&np.isfinite(local)),axis=1)
    new_pool=old_pool.copy();new_local=old_local.copy()
    for i in range(n):
        if ready[i]:
            # Pooled ranks are an exact preserved copy, not a new model/rerank.
            new_local[i]=new_pool[i] if np.array_equal(local[i,use[i]],pool[i,use[i]]) else _rank(local[i:i+1],use[i:i+1],new_pool[i:i+1])[0]
    weak=bridge._bool(arrays['state_weak_narrow'],(total,),n,'weak')
    formal=np.asarray([x!=INITIAL_PARTIAL for x in meta['dates'][:n]],dtype=bool)
    active=use&ready[:,None]&old_ready[:,None]&weak[:,None]&formal[:,None];delta=np.zeros((n,w))
    with np.errstate(over='raise',invalid='raise'):
        delta[active]=.0125*(new_local[active]-old_local[active])
    same=(new_pool==old_pool)&(new_local==old_local);delta[same]=0.
    require(np.isfinite(delta).all() and (np.abs(delta)<=.0125+1e-12).all(),'Bounded weak replacement')
    ranks=dict(old_pool=old_pool,old_local=old_local,new_pool=new_pool,new_local=new_local)
    audit=dict(schema='monthly_local_distance256_rank_delta_v1',events=n,ranking_cells=int(use.sum()),ready_rows=ready.tolist(),active_indices=[np.flatnonzero(x).tolist() for x in active],changed_cells=int(np.count_nonzero(delta)),
        original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,pooled_ranks_preserved=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return ranks,delta,audit

def decisions(study,arrays,meta,bundle,n=None,*,incumbent_validator=None,expected_incumbent_orders=None):
    total,w=np.shape(arrays['baseline']);n=total if n is None else n;before=bridge._snapshot(arrays)
    b=bridge._cache(study,arrays,meta,n)
    source=dict(baseline=b['base'],core=b['core'],core_weight=b['cw'],eligible=b['good'],state_calm10=b['calm'],state_weak_narrow=b['weak'])
    orders={};scores={};pres={};choices={};receipt=None
    for i,arm in enumerate(ARMS):
        if i==0:pre=b['pre'].copy()
        else:
            ranks,delta,ra=rank_and_delta(arrays,meta,bundle,n)
            pre=b['pre'].copy();changed=delta!=0.;pre[changed]+=delta[changed]
        chosen,post=study.ep.select_boundary(pre.copy(),b['good'].copy(),deepcopy(b['meta']['symbols']),b['effects'].copy(),{k:v.copy() for k,v in source.items()},POLICY.copy())
        post=bridge._matrix(post,(n,w),n,'Final scores');choices[arm]=[[int(j) for j in row] for row in chosen]
        orders[arm]=study.rp.make_orders(choices[arm],post,b['meta']);scores[arm]=post;pres[arm]=pre
        if i==0:
            if expected_incumbent_orders is not None:require(orders[arm]==expected_incumbent_orders,'Frozen incumbent orders changed')
            if incumbent_validator is not None:
                require(callable(incumbent_validator),'Incumbent validator callable');receipt=incumbent_validator(deepcopy(orders[arm]),post.copy(),pre.copy())
                require(type(receipt) is dict and receipt.get('passed') is True,'Independent incumbent gate failed')
    use=b['good']&np.asarray(arrays['forecast_available'])[:n];events=[]
    for i,d in enumerate(meta['dates'][:n]):
        events.append(dict(query_index=i,date=d,initial_partial=d==INITIAL_PARTIAL,weak=bool(b['weak'][i]),ranking_indices=np.flatnonzero(use[i]).tolist(),active_indices=ra['active_indices'][i],changed_delta_indices=np.flatnonzero(delta[i]!=0.).tolist(),delta=delta[i].tolist(),incumbent_choices=choices[ARMS[0]][i],candidate_choices=choices[ARMS[1]][i],episode_budget=float(b['episode'][i])))
    counts=dict.fromkeys(COUNT_FIELDS,0);partial=sum(e['initial_partial'] for e in events)
    counts.update(events=n,symbols_per_event=w,initial_partial_events=partial,formal_events=n-partial,original_eligible_cells=int(b['good'].sum()),ranking_cells=int(use.sum()),weak_applicable_cells=sum(len(x) for x in ra['active_indices']),changed_delta_cells=int(np.count_nonzero(delta)),changed_holdings_events=sum(set(a)!=set(c) for a,c in zip(choices[ARMS[0]],choices[ARMS[1]])),incumbent_episode_calls=1,candidate_episode_calls=1)
    require(before==bridge._snapshot(arrays),'Original cache modified')
    return orders,scores,pres,dict(schema='monthly_local_distance256_decisions_v1',counts=counts,events=events,rank_delta=ra,incumbent_gate=deepcopy(receipt),original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)
