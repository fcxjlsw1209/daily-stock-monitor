"""Pure registered-input correlation-distance contrast; no loader or replay."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work/monthly_local_distance256_kernel.py'
PREVIOUS_SHA='97ada679902942f31e4881354fcf95b1fccd2119f77c09760904b8c328364e30'
if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!=PREVIOUS_SHA:raise ValueError('Frozen256 changed')
spec=importlib.util.spec_from_file_location('correlation257_frozen256',PREVIOUS)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
model,bridge,require=previous.model,previous.bridge,previous.require
model_inputs=previous.model_inputs
TARGETS,RAW_KEYS,RANK_KEYS=previous.TARGETS,previous.RAW_KEYS,previous.RANK_KEYS
INITIAL_PARTIAL,POLICY,MASS_RTOL=previous.INITIAL_PARTIAL,previous.POLICY,previous.MASS_RTOL
ARMS=('incumbent_local67','diag_mass_local67','corr_mass_local67')
BUNDLE_KEYS=('schema','n','width','training_row_ids','control','diagonal','candidate','label_values','label_preservation','control_gate','diagnostic_gate','audit')
LABEL_KEYS=previous.LABEL_KEYS
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','control_target_fits','control_log_refits','control_local_evaluations','control_local_solves','diag_pooled_fits','diag_log_refits','diag_local_evaluations','diag_local_solves','diag_kernel_evaluations','diag_equal_scale_evaluations','diag_changed_weight_evaluations','candidate_pooled_fits','candidate_log_refits','candidate_local_evaluations','candidate_local_solves','candidate_kernel_evaluations','candidate_identity_correlation_evaluations','candidate_changed_weight_evaluations','correlation_matrix_builds','actual_regression_solves')
MOMENT_KEYS=('source_mean','raw_scale','source_scale','floored','training_weights','covariance','correlation','shrinkage','cholesky','transform','trace','metric','identity_correlation')
GEOMETRY_KEYS=('source_scale','identity_correlation','old_distance','distance','original_weights','unscaled_weights','weights','original_mass','unscaled_mass','mass_multiplier','matched_mass','effective_samples','original_effective_samples','mass_absolute_error','maximum_absolute_weight_change','weights_exact_original','diagonal_geometry')
CANDIDATE_ANNUAL_KEYS=('year','cutoff','training_row_ids','query_row_ids','source_scale','moments')
CANDIDATE_LOCAL_KEYS=previous.CANDIDATE_LOCAL_KEYS
CANDIDATE_AUDIT_KEYS=previous.CANDIDATE_AUDIT_KEYS
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells','diag_changed_delta_cells','diag_changed_holdings_events','candidate_changed_delta_cells','candidate_changed_holdings_events','incumbent_episode_calls','diag_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','diag_changed_delta_indices','diag_delta','changed_delta_indices','delta','incumbent_choices','diag_choices','candidate_choices','episode_budget')


def dependency_files():return tuple(dict.fromkeys((PREVIOUS,)+previous.dependency_files()))


def _matmul(left,right):
    # This pinned BLAS can set spurious divide/invalid flags on finite matmul.
    # Preserve canonical @ arithmetic; reject actual nonfinite outputs directly.
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        result=left@right
    require(np.isfinite(result).all(),'Nonfinite matrix product')
    return result


def correlation_metric(X,weights,source_mean=None,source_scale=None):
    """Canonical original annual moments, identity-padded half correlation."""
    X=np.asarray(X,float);weights=np.asarray(weights,float)
    require(X.ndim==2 and X.shape[1]==10 and len(X)>0 and weights.shape==(len(X),),'Moment axes')
    model._ranks('Annual original ranks',X)
    require(np.isfinite(weights).all() and (weights>=0).all() and np.isfinite(weights.sum()) and weights.sum()>0,'Annual weights')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        mean=np.average(X,axis=0,weights=weights);centered=X-mean
        raw_scale=np.sqrt(np.average(centered*centered,axis=0,weights=weights))
        floored=raw_scale<=1e-12;scale=np.where(raw_scale>1e-12,raw_scale,1.)
        if source_mean is not None:np.testing.assert_array_equal(mean,np.asarray(source_mean))
        if source_scale is not None:np.testing.assert_array_equal(scale,np.asarray(source_scale))
        covariance=_matmul((centered*weights[:,None]).T,centered)/weights.sum()
        correlation=np.eye(10)
        for j in range(10):
            for k in range(j+1,10):
                if not floored[j] and not floored[k]:
                    correlation[j,k]=covariance[j,k]/(scale[j]*scale[k]);correlation[k,j]=correlation[j,k]
        require(np.array_equal(correlation,correlation.T) and np.array_equal(np.diag(correlation),np.ones(10)),'Exact symmetric unit-diagonal correlation')
        require(np.isfinite(correlation).all(),'Finite correlation')
        shrinkage=(correlation+np.eye(10))/2.
        cholesky=np.linalg.cholesky(shrinkage)
        np.testing.assert_allclose(_matmul(cholesky,cholesky.T),shrinkage,atol=1e-12,rtol=1e-12)
        transform=np.linalg.solve(cholesky,np.diag(1./scale))
        trace=float(np.sum(transform*transform));require(np.isfinite(trace) and trace>0.,'Positive finite metric trace')
        metric=_matmul(transform.T,transform)/trace
    for name,val in (('mean',mean),('raw scale',raw_scale),('scale',scale),('covariance',covariance),('Cholesky',cholesky),('transform',transform),('metric',metric)):model._finite('Correlation '+name,val)
    require((scale>0.).all(),'Positive effective scales')
    return dict(source_mean=mean.tolist(),raw_scale=raw_scale.tolist(),source_scale=scale.tolist(),floored=floored.tolist(),training_weights=weights.tolist(),covariance=covariance.tolist(),correlation=correlation.tolist(),shrinkage=shrinkage.tolist(),cholesky=cholesky.tolist(),transform=transform.tolist(),trace=trace,metric=metric.tolist(),identity_correlation=bool(np.array_equal(correlation,np.eye(10))))


def correlation_weights(dates,cutoff,X,query,moments):
    """Exact identity branch or fixed Cholesky metric, with original total mass."""
    require(type(moments) is dict and set(moments)==set(MOMENT_KEYS),'Moment evidence schema')
    X=np.asarray(X,float);query=np.asarray(query,float);scale=np.asarray(moments['source_scale'],float)
    require(X.ndim==2 and X.shape[1]==10 and len(X)>0 and query.shape==(10,) and scale.shape==(10,),'Kernel axes')
    require(np.isfinite(scale).all() and (scale>0.).all(),'Positive source scale')
    R=np.asarray(moments['correlation'],float);B=np.asarray(moments['transform'],float);trace=moments['trace']
    require(R.shape==(10,10) and B.shape==(10,10) and np.isfinite(R).all() and np.isfinite(B).all() and np.isfinite(trace) and trace>0,'Finite metric evidence')
    require(np.array_equal(R,R.T) and np.array_equal(np.diag(R),np.ones(10)),'Symmetric unit correlation')
    identity=bool(np.array_equal(R,np.eye(10)));require(type(moments['identity_correlation']) is bool and moments['identity_correlation']==identity,'Exact identity flag')
    original=model.local_weights(dates,cutoff,X,query);require((original>0).all() and np.isfinite(original).all(),'Original positive support')
    old_distance=np.mean((X-query)**2,axis=1);old_mass=float(original.sum());diagonal_geometry=None
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        if identity:
            weights,diagonal_geometry=previous.diagonal_weights(dates,cutoff,X,query,scale)
            distance=np.asarray(diagonal_geometry['distance']);u=np.asarray(diagonal_geometry['unscaled_weights'])
            unscaled_mass=diagonal_geometry['unscaled_mass'];multiplier=diagonal_geometry['mass_multiplier']
        else:
            distance=np.sum(_matmul(X-query,B.T)**2,axis=1)/trace
            dates=model.numeric._days(dates,'own dates');cutoff=np.datetime64(cutoff,'D')
            ages=cutoff.astype('datetime64[M]').astype(int)-dates.astype('datetime64[M]').astype(int)
            u=np.exp2(-ages/model.HALF_LIFE_MONTHS)*np.exp(-distance/(2*model.BANDWIDTH**2))
            unscaled_mass=float(u.sum());require(np.isfinite(unscaled_mass) and unscaled_mass>0,'Unscaled mass')
            multiplier=old_mass/unscaled_mass;weights=u*multiplier
        mass=float(weights.sum());effective=float(mass**2/np.einsum('n,n->',weights,weights,optimize=False))
        old_effective=float(old_mass**2/np.einsum('n,n->',original,original,optimize=False))
    require(np.isfinite(distance).all() and (distance>=0).all() and np.isfinite(u).all() and (u>0).all(),'Positive finite correlation geometry')
    require(np.isfinite(weights).all() and (weights>0).all() and np.isfinite(multiplier),'Positive finite correlation weights')
    error=abs(mass-old_mass);require(error<=MASS_RTOL*old_mass,'Mass matching tolerance exceeded')
    return weights,dict(source_scale=scale.tolist(),identity_correlation=identity,old_distance=old_distance.tolist(),distance=distance.tolist(),original_weights=original.tolist(),unscaled_weights=u.tolist(),weights=weights.tolist(),original_mass=old_mass,unscaled_mass=unscaled_mass,mass_multiplier=multiplier,matched_mass=mass,effective_samples=effective,original_effective_samples=old_effective,mass_absolute_error=error,maximum_absolute_weight_change=float(np.max(np.abs(weights-original))),weights_exact_original=weights.tobytes()==original.tobytes(),diagonal_geometry=diagonal_geometry)


def candidate_forecasts(args,control,contexts,*,min_own_months=18):
    dates,ends,symbols,X,targets,good,event_dates,event_symbols,event_X,event_good=args
    raw={key:control['raw'][key].copy() for key in RAW_KEYS};raw['local_linear']=raw['source_log'].copy()
    available=np.zeros(len(event_dates),bool);annual=[];local=[]
    details={x['event_row']:x for x in control['audit']['local']}
    with threadpool_limits(limits=1):
        for ctx in contexts:
            year,cutoff,ids,pooled,residual,by_symbol,predict=(ctx[k] for k in ('year','cutoff','ids','pooled','residual','by_symbol','predict'))
            training_weights=model.numeric.month_balanced_weights(dates[ids],cutoff,model.HALF_LIFE_MONTHS)
            moments=correlation_metric(X[ids],training_weights,pooled[0],pooled[1])
            annual.append(dict(year=year,cutoff=str(cutoff),training_row_ids=ids.tolist(),query_row_ids=predict.tolist(),source_scale=pooled[1].tolist(),moments=moments))
            for j in predict:
                own=by_symbol.get(event_symbols[j],np.empty(0,dtype=int));own_ids=ids[own]
                detail=deepcopy(details[int(j)]);detail.update(status='insufficient_own_history',weight_mass=0.,effective_samples=0.,linear_correction=0.,own_row_ids=own_ids.tolist(),geometry=None)
                if detail['own_months']>=min_own_months:
                    weights,geometry=correlation_weights(dates[own_ids],cutoff,X[own_ids],event_X[j],moments)
                    linear,info=previous._linear_only(X[own_ids],event_X[j],pooled[1],residual[own],weights)
                    detail.update(info,linear_correction=linear,geometry=geometry);available[j]=info['weight_mass']>0.
                    if linear!=0.:
                        with np.errstate(over='raise',invalid='raise'):raw['local_linear'][j]+=linear
                local.append(detail)
    for key,value in raw.items():require(np.array_equal(np.isfinite(value),control['available']),'Candidate support changed '+key)
    np.testing.assert_array_equal(available,control['local_available'])
    return dict(raw=raw,available=control['available'].copy(),local_available=available,audit=dict(schema='monthly_correlation_distance257_candidate_audit_v1',feature_count=10,target='unchanged cached centered log_excess',pooled_copied=True,local_constant_copied=True,residuals='same mature in-sample pooled residuals as control',local_linear_only=True,annual=annual,local=local))


def fit_models(train,meta,arrays,n=None,*,control_validator=None,diagnostic_validator=None,min_months=18,min_own_months=18):
    args,ids,n,w=previous._inputs(train,meta,arrays,n);snapshot=bridge._snapshot(arrays)
    dates,ends,symbols,X,targets,good,*_=args
    identity=list(zip(dates.astype('datetime64[M]').astype(str),symbols.tolist()))
    require(len(identity)==len(set(identity)),'Duplicate security-month')
    require(np.isfinite(targets['log_excess'][good]).all(),'Nonfinite valid original log target')
    target_hash={k:previous._target_digest(targets[k]) for k in TARGETS};good_hash=previous._target_digest(good)
    control,contexts=previous._control_forecasts(*args,min_months=min_months,min_own_months=min_own_months)
    parity=previous.control_parity(arrays,control,n);control_gate=parity
    if control_validator is not None:
        require(callable(control_validator),'Control validator callable');control_gate=control_validator(deepcopy(control))
        require(type(control_gate) is dict and control_gate.get('passed') is True,'Independent original model gate failed')
    diagonal=previous.candidate_forecasts(args,control,contexts,min_own_months=min_own_months)
    diagnostic_gate=dict(passed=True,events=n,frozen_diagonal_implementation=True,correlation_models_computed=False)
    if diagnostic_validator is not None:
        require(callable(diagnostic_validator),'Diagnostic validator callable');diagnostic_gate=diagnostic_validator(deepcopy(diagonal))
        require(type(diagnostic_gate) is dict and diagnostic_gate.get('passed') is True,'Frozen diagonal model gate failed')
    candidate=candidate_forecasts(args,control,contexts,min_own_months=min_own_months)
    for forecast in (diagonal,candidate):
        previous._forecast_layout(forecast,n,w)
        for key in ('source_log','local_constant'):require(forecast['raw'][key].tobytes()==control['raw'][key].tobytes(),'Preserved forecast changed '+key)
    require(target_hash=={k:previous._target_digest(targets[k]) for k in TARGETS} and good_hash==previous._target_digest(good),'Original labels/common_good changed')
    labels=dict(schema='monthly_correlation_distance257_labels_preserved_v1',rows=len(ids),valid_rows=int(good.sum()),target_keys=list(TARGETS),target_sha256=target_hash,common_good_sha256=good_hash,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)
    counts=dict.fromkeys(MODEL_COUNT_FIELDS,0);counts.update(training_rows=len(ids),valid_training_rows=int(good.sum()))
    for suffix,value in zip(('target_fits','log_refits','local_evaluations','local_solves'),previous._fit_counts(control)):counts['control_'+suffix]=int(value)
    for role,forecast,flag in (('diag',diagonal,'equal_scales'),('candidate',candidate,'identity_correlation')):
        rows=forecast['audit']['local'];geometries=[r['geometry'] for r in rows if r['geometry'] is not None]
        counts[role+'_local_evaluations']=len(rows);counts[role+'_local_solves']=sum(r['status']=='fit' for r in rows)
        counts[role+'_kernel_evaluations']=len(geometries);counts[role+'_changed_weight_evaluations']=sum(not g['weights_exact_original'] for g in geometries)
        counts[role+('_equal_scale_evaluations' if role=='diag' else '_identity_correlation_evaluations')]=sum(g[flag] for g in geometries)
    counts['correlation_matrix_builds']=len(candidate['audit']['annual'])
    counts['actual_regression_solves']=sum(counts[k] for k in ('control_target_fits','control_log_refits','control_local_solves','diag_local_solves','candidate_local_solves'))
    require(snapshot==bridge._snapshot(arrays),'Original arrays changed')
    audit=dict(schema='monthly_correlation_distance257_model_audit_v1',counts=counts,original_cache_parity=parity,original_row_order_preserved=True,original_common_good_preserved=True,all_targets_preserved=True,candidate_pooled_copied=True,candidate_local_constant_copied=True,candidate_local_linear_only=True,diagonal_forecast_reproduced=True,correlation_floor_coordinates_retained=True,identity_correlation_reuses_diagonal=True,mass_relative_tolerance=float(MASS_RTOL),residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False)
    return dict(schema='monthly_correlation_distance257_models_v1',n=n,width=w,training_row_ids=ids,control=control,diagonal=diagonal,candidate=candidate,label_values=targets['log_excess'].copy(),label_preservation=labels,control_gate=deepcopy(control_gate),diagnostic_gate=deepcopy(diagnostic_gate),audit=audit)


def rank_and_delta(arrays,meta,bundle,n=None,*,arm=ARMS[2]):
    require(type(bundle) is dict and bundle.get('schema')=='monthly_correlation_distance257_models_v1' and arm in ARMS[1:],'257 model/arm')
    # The frozen rank implementation validates full shape but consumes only n rows.
    view=dict(schema='monthly_local_distance256_models_v1',n=bundle['n'],width=bundle['width'],candidate=bundle['diagonal' if arm==ARMS[1] else 'candidate'])
    ranks,delta,audit=previous.rank_and_delta(arrays,meta,view,n)
    audit['schema']='monthly_correlation_distance257_rank_delta_v1'
    return ranks,delta,audit


def decisions(study,arrays,meta,bundle,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,diagnostic_validator=None,expected_diagnostic_orders=None):
    total,w=np.shape(arrays['baseline']);n=total if n is None else n;before=bridge._snapshot(arrays)
    b=bridge._cache(study,arrays,meta,n)
    source=dict(baseline=b['base'],core=b['core'],core_weight=b['cw'],eligible=b['good'],state_calm10=b['calm'],state_weak_narrow=b['weak'])
    orders={};scores={};pres={};choices={};rankaudits={};deltas={};inc_gate=None;diag_gate=None
    for i,arm in enumerate(ARMS):
        pre=b['pre'].copy()
        if i:
            ranks,delta,ra=rank_and_delta(arrays,meta,bundle,n,arm=arm);rankaudits[arm]=ra;deltas[arm]=delta
            changed=delta!=0.;pre[changed]+=delta[changed]
        chosen,post=study.ep.select_boundary(pre.copy(),b['good'].copy(),deepcopy(b['meta']['symbols']),b['effects'].copy(),{k:v.copy() for k,v in source.items()},POLICY.copy())
        post=bridge._matrix(post,(n,w),n,'Final scores');choices[arm]=[[int(j) for j in row] for row in chosen]
        orders[arm]=study.rp.make_orders(choices[arm],post,b['meta']);scores[arm]=post;pres[arm]=pre
        if i<2:
            expected=expected_incumbent_orders if i==0 else expected_diagnostic_orders
            callback=incumbent_validator if i==0 else diagnostic_validator
            if expected is not None:require(orders[arm]==expected,'Frozen '+('incumbent' if i==0 else 'diagonal')+' orders changed')
            if callback is not None:
                require(callable(callback),'Decision validator callable');receipt=callback(deepcopy(orders[arm]),post.copy(),pre.copy())
                require(type(receipt) is dict and receipt.get('passed') is True,('Independent incumbent' if i==0 else 'Frozen diagonal')+' gate failed')
                if i==0:inc_gate=receipt
                else:diag_gate=receipt
    use=b['good']&np.asarray(arrays['forecast_available'])[:n];events=[];dra=rankaudits[ARMS[1]];cra=rankaudits[ARMS[2]]
    require(dra['active_indices']==cra['active_indices'],'Common experimental application domain')
    for i,d in enumerate(meta['dates'][:n]):
        dd,cd=deltas[ARMS[1]][i],deltas[ARMS[2]][i]
        events.append(dict(query_index=i,date=d,initial_partial=d==INITIAL_PARTIAL,weak=bool(b['weak'][i]),ranking_indices=np.flatnonzero(use[i]).tolist(),active_indices=cra['active_indices'][i],diag_changed_delta_indices=np.flatnonzero(dd!=0.).tolist(),diag_delta=dd.tolist(),changed_delta_indices=np.flatnonzero(cd!=0.).tolist(),delta=cd.tolist(),incumbent_choices=choices[ARMS[0]][i],diag_choices=choices[ARMS[1]][i],candidate_choices=choices[ARMS[2]][i],episode_budget=float(b['episode'][i])))
    counts=dict.fromkeys(COUNT_FIELDS,0);partial=sum(e['initial_partial'] for e in events)
    counts.update(events=n,symbols_per_event=w,initial_partial_events=partial,formal_events=n-partial,original_eligible_cells=int(b['good'].sum()),ranking_cells=int(use.sum()),weak_applicable_cells=sum(len(x) for x in cra['active_indices']),incumbent_episode_calls=1,diag_episode_calls=1,candidate_episode_calls=1)
    for arm,role in zip(ARMS[1:],('diag','candidate')):
        counts[role+'_changed_delta_cells']=int(np.count_nonzero(deltas[arm]));counts[role+'_changed_holdings_events']=sum(set(a)!=set(c) for a,c in zip(choices[ARMS[0]],choices[arm]))
    require(before==bridge._snapshot(arrays),'Original cache modified')
    return orders,scores,pres,dict(schema='monthly_correlation_distance257_decisions_v1',counts=counts,events=events,rank_delta=rankaudits,incumbent_gate=deepcopy(inc_gate),diagnostic_gate=deepcopy(diag_gate),original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False,diagonal_decisions_reproduced=True)
