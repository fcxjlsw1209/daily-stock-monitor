"""Independent256 marginal-distance, fixed-mass local-linear proof.

No256 producer is imported. Only supplied synthetic/registered inputs are used.
The control and candidate share independently fitted mature pooled parameters;
the candidate never solves another pooled model.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work/monthly_asset_target255_independent.py'
PREVIOUS_SHA='01c0750c54594657857668b5fb59f22a02ef0f5f7b166ecc09a17326d4b9a9e2'
def require(ok,message):
    if not ok:raise ValueError(message)
require(hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()==PREVIOUS_SHA,'Frozen independent255 changed')
_spec=importlib.util.spec_from_file_location('distance256_frozen_independent255',PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
numeric,bridge,accounts=previous.numeric,previous.bridge,previous.accounts
exact_array,record_check=previous.exact_array,previous.record_check
continuous,check_record=previous.continuous,previous.check_record
model_inputs,original_ranks,_rank=previous.model_inputs,previous.original_ranks,previous._rank
original_base,accounting=previous.original_base,previous.accounting
ARMS=('incumbent_local67','diag_mass_local67')
TARGETS,RAW_KEYS=previous.TARGETS,previous.RAW_KEYS
COUNT_FIELDS,EVENT_KEYS=previous.COUNT_FIELDS,previous.EVENT_KEYS
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','control_target_fits','control_log_refits',
 'control_local_evaluations','control_local_solves','candidate_pooled_fits','candidate_log_refits',
 'candidate_local_evaluations','candidate_local_solves','candidate_kernel_evaluations',
 'candidate_equal_scale_evaluations','candidate_changed_weight_evaluations','actual_regression_solves')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
PATHS,ACCOUNTING_SHA=previous.PATHS,previous.ACCOUNTING_SHA
CONTROL_FLAGS=previous.CONTROL_FLAGS
INCUMBENT_FLAGS=previous.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
MODEL_FLAGS=CONTROL_FLAGS+('original_labels_unchanged','original_pooled_and_constant_copied',
 'same_mature_own_rows_checked','effective_pooled_scales_checked','inverse_variance_coefficients_checked',
 'fixed_original_local_mass_checked','exact_equal_scale_branch_checked','local_design_penalties_unchanged',
 'candidate_pooled_solves_zero','independent_augmented_least_squares','full_model_audit_checked',
 'actual_fit_counts_checked','no_future_training_values')
SELECTION_FLAGS=tuple(k for k in previous.SELECTION_FLAGS if k!='weak_only_five_percent_slot_checked')+('weak_only_local_linear_delta_checked','original_pooled_rank_unchanged')

def dependency_files():
    return tuple(sorted(set(previous.dependency_files())|{PREVIOUS,Path(__file__).resolve()}))

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    return previous.verify_incumbent(arrays,meta,orders,scores,pre_scores,n)

def control_oracle(inp,targets,*,min_months=18,min_own_months=18):
    require(type(min_months) is int and min_months>0 and type(min_own_months) is int and min_own_months>0,'Positive synthetic minima')
    d,end,s,X,g=(inp[k] for k in ('dates','ends','symbols','X','good'))
    qd,qs,Q,qgood=(inp[k] for k in ('query_dates','query_symbols','query_X','query_good'))
    usable=g & np.isfinite(X).all(axis=1)
    for y in targets.values():usable &= np.isfinite(y)
    months=d.astype('datetime64[M]')
    for month in np.unique(months):
        where=months==month
        if (usable&where).sum()<2:usable[where]=False
    years=qd.astype('datetime64[Y]').astype(int)+1970
    raw={k:np.full(len(qd),np.nan) for k in RAW_KEYS};available=np.zeros(len(qd),bool);local_available=available.copy()
    source_annual=[];annual=[];details=[];contexts=[]
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff=np.datetime64('%04d-12-31'%(int(year)-1));start=np.datetime64('%04d-12-31'%(int(year)-6))
            ids=np.flatnonzero(usable&(d>start)&(end<=cutoff));nm=len(np.unique(months[ids]))
            requests=np.flatnonzero((years==year)&qgood);pred=requests if nm>=min_months else np.array([],int)
            source_row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),training_rows=len(ids),training_months=nm,
              prediction_rows=len(requests),latest_label_end=str(end[ids].max()) if len(ids) else None,
              first_training_signal=str(d[ids].min()) if len(ids) else None,latest_training_signal=str(d[ids].max()) if len(ids) else None,
              status='fit' if nm>=min_months else 'cold_start')
            row={k:source_row[k] for k in ('year','cutoff','lookback_exclusive_start','training_rows','training_months')}
            row.update(prediction_rows=len(pred),first_training_signal=source_row['first_training_signal'],latest_label_end=source_row['latest_label_end'],status=source_row['status'])
            if len(pred):
                require(np.isfinite(X[ids]).all() and (np.abs(X[ids])<=.5).all() and (np.abs(Q[pred])<=.5).all(),'Original bounded rank features')
                require(len({(str(months[k]),s[k]) for k in ids})==len(ids),'Unique mature security months')
                unique,inv,counts=np.unique(months[ids],return_inverse=True,return_counts=True)
                age=int(cutoff.astype('datetime64[M]').astype(int))-unique.astype(int)
                w=(1./counts[inv])*np.exp2(-age[inv]/12.);w=w*(len(w)/w.sum())
                mu=np.average(X[ids],axis=0,weights=w);scale=np.sqrt(np.average((X[ids]-mu)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
                y=targets['log_excess'][ids];intercept=float(np.average(y,weights=w));Z=(X[ids]-mu)/scale
                coef=numeric.least_squares(Z,y-intercept,w,np.full(10,10.))
                source=np.einsum('ni,i->n',(Q[pred]-mu)/scale,coef,optimize=False)+intercept
                residual=y-(np.einsum('ni,i->n',Z,coef,optimize=False)+intercept)
                require(np.isfinite(source).all() and np.isfinite(residual).all(),'Finite independent pooled fit')
                contexts.append(dict(year=int(year),cutoff=cutoff,ids=ids.copy(),prediction_rows=pred.copy(),scale=scale.copy(),residual=residual.copy(),pooled_weights=w.copy(),mu=mu.copy(),coefficient=coef.copy(),intercept=intercept))
                for key in RAW_KEYS:raw[key][pred]=source
                available[pred]=True;row['weight_sum']=source_row['weight_sum']=float(w.sum())
                for j in pred:
                    own=np.flatnonzero(s[ids]==qs[j]);oi=ids[own];on=len(np.unique(months[oi]))
                    info=dict(event_row=int(j),date=str(qd[j]),symbol=str(qs[j]),year=int(year),cutoff=str(cutoff),own_rows=len(own),own_months=on,
                      latest_label_end=str(end[oi].max()) if len(oi) else None,status='insufficient_own_history',weight_mass=0.,effective_samples=0.,constant_correction=0.,linear_correction=0.)
                    if on>=min_own_months:
                        ages=int(cutoff.astype('datetime64[M]').astype(int))-months[oi].astype(int)
                        dist=np.mean((X[oi]-Q[j])**2,axis=1);lw=np.exp2(-ages/12.)*np.exp(-dist/(2*.5**2))
                        c,l,fit=numeric.local_fit(X[oi],Q[j],scale,residual[own],lw)
                        info.update(fit,constant_correction=c,linear_correction=l);local_available[j]=fit['weight_mass']>0
                        for key,delta in (('local_constant',c),('local_linear',l)):
                            if delta!=0:raw[key][j]+=delta
                    details.append(info)
            annual.append(row);source_annual.append(source_row)
    audit=dict(modes=['local_constant','local_linear'],feature_count=10,target='unchanged cached centered log_excess',alpha=10.,half_life_months=12.,
      min_months=min_months,min_own_months=min_own_months,bandwidth=.5,intercept_penalty=12.,slope_penalty=10.,same_local_rows_residuals_and_weights=True,
      local_weights_normalized=False,local_recency_clock='annual prior December cutoff calendar-month age',kernel='mean squared difference in original centered percentile plain10 units',
      linear_design='query-centered original X differences divided by pooled annual training scales',residuals='in-sample residuals within already matured annual training cohort, not OOF',
      source_control='independent original four-target annual model and exact log fit replay',no_own_history='exact source raw copy; never remove a query or alter eligibility',
      source=dict(alpha=10.,half_life_months=12.,min_months=min_months,lookback_years=5,targets=list(TARGETS),shared_rows_and_weights=True,
        training_only_standardization=True,weighting='equal_month_total_then_age_decay_normalized_to_row_count',predictions_are_raw=True,label_ranks_used=False,annual=source_annual),annual=annual,local=details)
    return dict(raw=raw,available=available,local_available=local_available,audit=audit),contexts


def verify_control(train,meta,arrays,control,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);expected,_=control_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    return verify_control_result(inp,arrays,control,expected)

def verify_control_result(inp,arrays,control,expected):
    """Validate an already computed oracle result without a second solve."""
    previous.check_forecast(control,expected,'Original model')
    n=inp['n'];shape=(n,inp['width'])
    for key in RAW_KEYS:exact_array(np.asarray(control['raw'][key]).reshape(shape),np.asarray(arrays['raw_'+key])[:n],'Original cache/'+key)
    for key in ('available','local_available'):
        source='model_available' if key=='available' else key
        exact_array(np.asarray(control[key]).reshape(shape),np.asarray(arrays[source])[:n],'Original cache/'+key)
    exact_array(np.asarray(control['available']).reshape(shape),np.asarray(arrays['forecast_available'])[:n],'Original forecast availability')
    if 'raw_log_excess' in arrays:
        a=np.asarray(control['raw']['source_log']).reshape(shape);e=np.asarray(arrays['raw_log_excess'])[:n]
        require(np.allclose(a,e,atol=1e-12,rtol=0,equal_nan=True),'Historical pooled source1e-12')
    old,local,U=original_ranks(arrays,n)
    for i,use in enumerate(U):
        if not use.any():continue
        exact_array(_rank(expected['raw']['source_log'].reshape(shape)[i,use]),old[i,use],'Independent pooled ranks')
        own=expected['raw']['local_linear'].reshape(shape)[i,use]
        exact_array(_rank(own),local[i,use],'Independent local ranks')
    return dict(**{k:True for k in CONTROL_FLAGS},checked_events=n,full_control_supplied=True,continuous_atol=CONTINUOUS_ATOL,
      continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,candidate_labels_computed=False,candidate_models_computed=False,
      episode_calls=0,producer_equivalent_fit_counts=producer_fit_counts(control),
      independent_solve_counts=independent_solve_counts(expected),producer_modules_imported=False,computes_returns=False)

def independent_solve_counts(model):
    pooled=sum(row['status']=='fit' and row['prediction_rows']>0 for row in model['audit']['annual'])
    local=sum(row['status']=='fit' for row in model['audit']['local'])
    return dict(pooled_log_solves=pooled,local_linear_solves=local,total_regression_solves=pooled+local)

def producer_fit_counts(model):
    own=independent_solve_counts(model);years=own['pooled_log_solves']
    return dict(target_fits=4*years,log_refits=years,local_evaluations=len(model['audit']['local']),
      local_solves=own['local_linear_solves'],actual_regression_solves=5*years+own['local_linear_solves'])

def geometry_oracle(dates,cutoff,X,query,scale):
    """Same histories/recency, changed diagonal geometry, fixed original mass."""
    d=previous.days(dates,'own dates');cutoff=np.datetime64(cutoff,'D')
    X,q,s=[np.asarray(v,dtype=np.float64) for v in (X,query,scale)]
    require(X.shape==(len(d),10) and q.shape==s.shape==(10,),'Geometry axes')
    require(not np.isnat(cutoff) and np.all(d<=cutoff),'Mature geometry clock')
    require(all(np.isfinite(v).all() for v in (X,q,s)) and (s>0).all(),'Finite positive effective scales')
    require((np.abs(X)<=.5).all() and (np.abs(q)<=.5).all(),'Original rank feature units')
    ages=int(cutoff.astype('datetime64[M]').astype(int))-d.astype('datetime64[M]').astype(int)
    with np.errstate(over='raise',divide='raise',invalid='raise',under='ignore'):
        square=(X-q)**2;old_distance=np.mean(square,axis=1)
        recency=np.exp2(-ages/12.);original=recency*np.exp(-old_distance/(2*.5**2))
        equal=bool(np.all(s==s[0]))
        inverse=1./(s*s);coefficients=np.full(10,.1) if equal else inverse/inverse.sum()
        distance=old_distance.copy() if equal else np.sum(square*coefficients,axis=1)
        unscaled=original.copy() if equal else recency*np.exp(-distance/(2*.5**2))
        original_mass=float(original.sum());unscaled_mass=float(unscaled.sum())
        require(original_mass>0 and unscaled_mass>0,'Positive own-history mass')
        multiplier=1. if equal else original_mass/unscaled_mass
        weights=original.copy() if equal else unscaled*multiplier
        matched=float(weights.sum())
        effective=float(matched*matched/np.dot(weights,weights))
        old_effective=float(original_mass*original_mass/np.dot(original,original))
    require(all(np.isfinite(v).all() for v in (coefficients,distance,unscaled,weights)),'Finite geometry')
    require((weights>=0).all() and matched>0,'Valid matched mass')
    require(abs(matched-original_mass)<=16*np.finfo(float).eps*original_mass,'Mass conservation')
    return dict(source_scale=s.tolist(),coefficients=coefficients.tolist(),old_distance=old_distance.tolist(),
      distance=distance.tolist(),original_weights=original.tolist(),unscaled_weights=unscaled.tolist(),weights=weights.tolist(),
      equal_scales=equal,original_mass=original_mass,unscaled_mass=unscaled_mass,mass_multiplier=multiplier,
      matched_mass=matched,effective_samples=effective,original_effective_samples=old_effective,
      mass_absolute_error=abs(matched-original_mass),maximum_absolute_weight_change=float(np.max(np.abs(weights-original))),
      weights_exact_original=weights.tobytes()==original.tobytes())

def label_preservation(inp):
    return dict(schema='monthly_local_distance256_labels_preserved_v1',rows=len(inp['dates']),valid_rows=int(inp['good'].sum()),
      target_keys=list(TARGETS),target_sha256={k:hashlib.sha256(inp['targets'][k].tobytes()).hexdigest() for k in TARGETS},
      common_good_sha256=hashlib.sha256(inp['good'].tobytes()).hexdigest(),labels_unchanged=True,common_good_unchanged=True,
      original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)

def verify_supplied_geometry(inp,row):
    """Internal binary64 identities are exact, unlike cross-solver comparisons."""
    supplied=row['geometry']
    if supplied is None:return
    ids=np.asarray(row['own_row_ids'],dtype=int);j=row['event_row']
    vectors=('source_scale','coefficients','old_distance','distance','original_weights','unscaled_weights','weights')
    scalars=('original_mass','unscaled_mass','mass_multiplier','matched_mass','effective_samples',
      'original_effective_samples','mass_absolute_error','maximum_absolute_weight_change')
    for name in vectors:
        require(type(supplied[name]) is list and all(type(v) is float and np.isfinite(v) for v in supplied[name]),'Saved geometry float vector/'+name)
    for name in scalars:require(type(supplied[name]) is float and np.isfinite(supplied[name]),'Saved geometry finite scalar/'+name)
    rebuilt=geometry_oracle(inp['dates'][ids],np.datetime64(row['cutoff']),inp['X'][ids],inp['query_X'][j],supplied['source_scale'])
    for name in vectors:exact_array(np.asarray(supplied[name]),np.asarray(rebuilt[name]),'Saved geometry exact algebra/'+name)
    for name in ('equal_scales','weights_exact_original','original_mass','unscaled_mass','mass_multiplier','matched_mass',
      'mass_absolute_error','maximum_absolute_weight_change'):
        record_check(supplied[name],rebuilt[name],'Saved geometry exact identity/'+name)
    weights=np.asarray(supplied['weights']);original=np.asarray(supplied['original_weights'])
    require((weights>0).all() and (original>0).all(),'Positive saved own weights')
    # Enforce the contractual mass bound against the supplied vector itself.
    require(abs(float(weights.sum())-float(original.sum()))<=16*np.finfo(float).eps*float(original.sum()),'Saved mass16eps')

def candidate_oracle(inp,control,contexts,*,min_own_months=18):
    """Only a new local-linear SVD. Pooled systems are carried from control."""
    raw={k:v.copy() for k,v in control['raw'].items()}
    raw['local_linear']=control['raw']['source_log'].copy()
    available=control['available'].copy();local_available=control['local_available'].copy()
    dates,ends,X,symbols=inp['dates'],inp['ends'],inp['X'],inp['symbols']
    months=dates.astype('datetime64[M]');Q=inp['query_X'];qd=inp['query_dates'];qs=inp['query_symbols']
    previous_local={int(r['event_row']):r for r in control['audit']['local']}
    annual=[];local=[]
    with threadpool_limits(limits=1):
        for ctx in contexts:
            ids,pred,scale=ctx['ids'],ctx['prediction_rows'],ctx['scale'];cutoff=ctx['cutoff']
            annual.append(dict(year=ctx['year'],cutoff=str(cutoff),training_row_ids=ids.tolist(),query_row_ids=pred.tolist(),source_scale=scale.tolist()))
            for j in pred:
                own=np.flatnonzero(symbols[ids]==qs[j]);oi=ids[own];on=len(np.unique(months[oi]))
                detail=dict(event_row=int(j),date=str(qd[j]),symbol=str(qs[j]),year=ctx['year'],cutoff=str(cutoff),
                  own_rows=len(own),own_months=on,latest_label_end=str(ends[oi].max()) if len(oi) else None,
                  status='insufficient_own_history',weight_mass=0.,effective_samples=0.,
                  constant_correction=float(previous_local[int(j)]['constant_correction']),linear_correction=0.,own_row_ids=oi.tolist(),geometry=None)
                if on>=min_own_months:
                    geometry=geometry_oracle(dates[oi],cutoff,X[oi],Q[j],scale)
                    weights=np.asarray(geometry['weights']);residual=ctx['residual'][own]
                    mass=float(weights.sum());effective=float(mass*mass/np.dot(weights,weights))
                    status='zero_weight' if mass==0 else 'zero_residual' if not np.any(residual) else 'fit'
                    linear=0.
                    if status=='fit':
                        design=np.column_stack((np.ones(len(oi)),(X[oi]-Q[j])/scale))
                        linear=float(numeric.least_squares(design,residual,weights,np.array([12.]+[10.]*10))[0])
                    detail.update(status=status,weight_mass=mass,effective_samples=effective,linear_correction=linear,geometry=geometry)
                    if linear!=0.:raw['local_linear'][j]+=linear
                local.append(detail)
    require(all(np.array_equal(np.isfinite(v),available) for v in raw.values()),'Candidate full availability')
    au=dict(schema='monthly_local_distance256_candidate_audit_v1',feature_count=10,target='unchanged cached centered log_excess',
      pooled_copied=True,local_constant_copied=True,residuals='same mature in-sample pooled residuals as control',
      local_linear_only=True,annual=annual,local=local)
    return dict(raw=raw,available=available,local_available=local_available,audit=au)

def model_counts(inp,control,candidate):
    original=producer_fit_counts(control);rows=candidate['audit']['local'];geometry=[r['geometry'] for r in rows if r['geometry'] is not None]
    c={k:0 for k in MODEL_COUNT_FIELDS}
    c.update(training_rows=len(inp['dates']),valid_training_rows=int(inp['good'].sum()))
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+key]=original[key]
    c.update(candidate_local_evaluations=len(rows),candidate_local_solves=sum(r['status']=='fit' for r in rows),
      candidate_kernel_evaluations=len(geometry),candidate_equal_scale_evaluations=sum(g['equal_scales'] for g in geometry),
      candidate_changed_weight_evaluations=sum(not g['weights_exact_original'] for g in geometry))
    c['actual_regression_solves']=original['actual_regression_solves']+c['candidate_local_solves']
    return c

def model_audit(inp,control,candidate):
    return dict(schema='monthly_local_distance256_model_audit_v1',counts=model_counts(inp,control,candidate),
      original_cache_parity=dict(passed=True,events=inp['n'],original_raw_exact=True,original_availability_exact=True,
        original_ranks_exact=True,candidate_models_computed=False),original_row_order_preserved=True,
      original_common_good_preserved=True,all_targets_preserved=True,candidate_pooled_copied=True,candidate_local_constant_copied=True,
      candidate_local_linear_only=True,equal_scale_original_weights_exact=True,residuals_are_mature_in_sample=True,
      mass_relative_tolerance=float(16*np.finfo(float).eps),residuals_are_oof=False,computes_returns=False)

def verify_models(train,meta,arrays,bundle,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);n=inp['n'];width=inp['width']
    keys={'schema','n','width','training_row_ids','control','candidate','label_values','label_preservation','control_gate','audit'}
    require(type(bundle) is dict and set(bundle)==keys,'Full256 model bundle')
    require(bundle['schema']=='monthly_local_distance256_models_v1' and type(bundle['n']) is int and bundle['n']==n
      and type(bundle['width']) is int and bundle['width']==width,'Model axes')
    exact_array(bundle['training_row_ids'],inp['training_row_ids'],'Consumed training row IDs')
    independent_control,contexts=control_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    proof=verify_control_result(inp,arrays,bundle['control'],independent_control)
    exact_array(bundle['label_values'],inp['targets']['log_excess'],'Unchanged cached log labels')
    record_check(bundle['label_preservation'],label_preservation(inp),'All original targets and good bytes')
    # Supplied pooled/constant bytes have already passed the independent control
    # proof; reuse those exact output bytes while residuals/scales come from SVD.
    candidate=candidate_oracle(inp,bundle['control'],contexts,min_own_months=min_own_months)
    previous.check_forecast(bundle['candidate'],candidate,'Independent local-linear candidate')
    for key in ('source_log','local_constant'):
        exact_array(bundle['candidate']['raw'][key],bundle['control']['raw'][key],'Copied original raw/'+key)
    for key in ('available','local_available'):
        exact_array(bundle['candidate'][key],bundle['control'][key],'Unchanged availability/'+key)
    for row in bundle['candidate']['audit']['local']:
        verify_supplied_geometry(inp,row)
        g=row['geometry'];j=row['event_row']
        if g is not None and g['weights_exact_original']:
            exact_array(np.asarray(bundle['candidate']['raw']['local_linear'])[j:j+1],
              np.asarray(bundle['control']['raw']['local_linear'])[j:j+1],'Exact original-weight local forecast')
    r,d,_,_,_=rank_oracle(arrays,meta['monthly'],bundle,n)
    rr,dd,_,_,_=rank_oracle(arrays,meta['monthly'],dict(bundle,candidate=candidate),n)
    for key in r:exact_array(r[key],rr[key],'Independent exact rank/'+key)
    exact_array(d,dd,'Independent exact local delta')
    wanted=model_audit(inp,bundle['control'],candidate);record_check(bundle['audit'],wanted,'Full model audit')
    gate=bundle['control_gate'];require(type(gate) is dict and gate.get('passed') is True,'Pre-candidate control gate')
    expected=wanted['original_cache_parity'] if set(gate)==set(wanted['original_cache_parity']) else proof
    record_check(gate,expected,'Saved original model gate')
    original_solves=proof['independent_solve_counts'];local_solves=sum(r['status']=='fit' for r in candidate['audit']['local'])
    candidate_solves=dict(pooled_log_solves=0,local_linear_solves=local_solves,total_regression_solves=local_solves)
    solves=dict(control=original_solves,candidate=candidate_solves,total_regression_solves=original_solves['total_regression_solves']+local_solves)
    return dict(**{k:True for k in MODEL_FLAGS},checked_events=n,counts=wanted['counts'],producer_equivalent_counts=wanted['counts'],
      independent_solve_counts=solves,full_bundle_supplied=True,full_labels_supplied=True,full_control_supplied=True,full_candidate_supplied=True,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
      independent_numeric_system='augmented_least_squares_SVD',producer_modules_imported=False,computes_returns=False)


def rank_oracle(arrays,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    old,local,U=original_ranks(arrays,n);width=old.shape[1]
    require(type(bundle) is dict and type(bundle.get('n')) is int and bundle['n']>=n and bundle.get('width')==width,'Forecast bundle prefix')
    candidate=bundle['candidate'];size=bundle['n']*width
    require(set(candidate['raw'])==set(RAW_KEYS),'Candidate raw keys')
    raw={k:np.asarray(candidate['raw'][k]) for k in RAW_KEYS}
    require(all(v.shape==(size,) and v.dtype==np.dtype('float64') for v in raw.values()),'Candidate raw axes')
    available=np.asarray(candidate['available']);require(available.shape==(size,) and available.dtype==np.dtype(bool),'Candidate bool availability')
    raw={k:v[:n*width].reshape(n,width) for k,v in raw.items()};available=available[:n*width].reshape(n,width)
    newpool=old.copy();newlocal=local.copy();ready=np.zeros(n,bool);delta=np.zeros_like(old);active=[]
    weak=np.asarray(arrays['state_weak_narrow']);require(weak.dtype==np.dtype(bool) and weak.shape[0]>=n,'Original weak bool')
    for i,use in enumerate(U):
        ids=np.flatnonzero(use)
        if len(ids):
            require(available[i,use].all() and all(np.isfinite(raw[k][i,use]).all() for k in RAW_KEYS),'Whole original rank domain available')
            ready[i]=True
            exact_array(raw['source_log'][i,use],np.asarray(arrays['raw_source_log'])[:n][i,use],'Candidate pooled copied')
            exact_array(raw['local_constant'][i,use],np.asarray(arrays['raw_local_constant'])[:n][i,use],'Candidate constant copied')
            newlocal[i,use]=newpool[i,use] if np.array_equal(raw['local_linear'][i,use],raw['source_log'][i,use]) else _rank(raw['local_linear'][i,use])
        isactive=bool(weak[i]) and str(meta['dates'][i])!='2016-08-17'
        active.append(ids.tolist() if isactive else [])
        if isactive and len(ids):
            delta[i,use]=.0125*(newlocal[i,use]-local[i,use])
    require(np.isfinite(delta).all(),'Finite weak slot delta')
    return dict(old_pool=old,old_local=local,new_pool=newpool,new_local=newlocal),delta,U,ready,active

def rank_and_delta(arrays,meta,bundle,n=None):
    ranks,delta,U,ready,active=rank_oracle(arrays,meta,bundle,n)
    return ranks,delta,dict(schema='monthly_local_distance256_rank_delta_v1',events=U.shape[0],ranking_cells=int(U.sum()),
      ready_rows=ready.tolist(),active_indices=active,changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,
      no_group_ranking=True,no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True,pooled_ranks_preserved=True)

def selection_oracle(original,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    ranks,delta,U,ready,active=rank_oracle(original,meta,bundle,n)
    pres={arm:base['P'].copy() for arm in ARMS};changed=delta!=0.;pres[ARMS[1]][changed]+=delta[changed]
    scores={};choices={}
    for arm in ARMS:scores[arm],choices[arm]=accounts.episode(pres[arm],base)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=base['width'],
      original_eligible_cells=int(base['E'].sum()),ranking_cells=int(U.sum()),incumbent_episode_calls=1,candidate_episode_calls=1)
    events=[]
    for i in range(n):
        partial=str(meta['dates'][i])=='2016-08-17'
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.min(remaining)>=.38-1e-12,'Original remaining support budget')
        counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(not partial)
        counts['weak_applicable_cells']+=len(active[i]);counts['changed_delta_cells']+=int(np.count_nonzero(delta[i]))
        counts['changed_holdings_events']+=int(set(choices[ARMS[0]][i])!=set(choices[ARMS[1]][i]))
        events.append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,weak=bool(base['weak'][i]),
          ranking_indices=np.flatnonzero(U[i]).tolist(),active_indices=active[i],changed_delta_indices=np.flatnonzero(delta[i]!=0).tolist(),
          delta=delta[i].tolist(),incumbent_choices=choices[ARMS[0]][i],candidate_choices=choices[ARMS[1]][i],episode_budget=float(base['budget'][i])))
    ra=dict(schema='monthly_local_distance256_rank_delta_v1',events=n,ranking_cells=int(U.sum()),ready_rows=ready.tolist(),
      active_indices=active,changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,no_group_ranking=True,
      no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True,pooled_ranks_preserved=True)
    return scores,choices,pres,dict(schema='monthly_local_distance256_decisions_v1',counts=counts,events=events,rank_delta=ra,
      incumbent_gate=None,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,
      single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)

def orders_from_choices(scores,choices,meta):
    return {a:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[a][i,j]) for j in ids])
      for i,ids in enumerate(choices[a])] for a in ARMS}

def verify_selection(arrays,meta,bundle,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Independent256 ordered choices')
    for actual,target,label in ((scores,expected,'Full final score'),(pre_scores,pres,'Full pre score')):
        require(type(actual) is dict and set(actual)==set(ARMS),label+' arms')
        for a in ARMS:exact_array(actual[a],target[a],label+'/'+a)
    require(type(audit) is dict and 'incumbent_gate' in audit,'Saved incumbent gate field')
    if audit['incumbent_gate'] is not None:
        receipt=verify_incumbent(arrays,meta,orders[ARMS[0]],scores[ARMS[0]],pre_scores[ARMS[0]],n)
        record_check(audit['incumbent_gate'],receipt,'Complete saved incumbent gate');wanted['incumbent_gate']=receipt
    record_check(audit,wanted,'Full decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},checked_events=n,ordered_choices_checked=2*n,counts=wanted['counts'],
      full_bundle_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
      numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Two256 accounting arms')
    mapping=dict(zip(ARMS,accounts.ARMS));rename=lambda d:{mapping[k]:v for k,v in d.items()}
    for arm in ARMS:
        for name,_,fee in PATHS:
            if fee==0.:
                require(float(results[arm][name]['metrics']['fees_paid'])==0. and float(results[arm][name]['slices']['full122']['fees_paid'])==0.,'ZERO aggregate fees exact0')
    report=accounts.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={a:report['paths'][mapping[a]] for a in ARMS})
