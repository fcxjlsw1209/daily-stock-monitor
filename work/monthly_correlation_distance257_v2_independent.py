"""Independent257 validator-only v2: canonical saved ESS uses float pow.

No257 producer import or actual-source loader. Independent moments and SPD
quadratic forms are cross-checked against canonical supplied matrix identities.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work/monthly_local_distance256_independent.py'
PREVIOUS_SHA='2ac297ec53ce8f9fd4f90a1868e86c26f523442eeada5e4f5bb4d8d65c4b40a9'
def require(ok,message):
    if not ok:raise ValueError(message)
require(hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()==PREVIOUS_SHA,'Frozen independent256 changed')
_spec=importlib.util.spec_from_file_location('corr257_independent256',PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
numeric,bridge,accounts=previous.numeric,previous.bridge,previous.accounts
exact_array,record_check=previous.exact_array,previous.record_check
continuous,check_record=previous.continuous,previous.check_record
model_inputs,original_ranks,control_oracle=previous.model_inputs,previous.original_ranks,previous.control_oracle
original_base,accounting=previous.original_base,previous.accounting
verify_control,verify_incumbent=previous.verify_control,previous.verify_incumbent
TARGETS,RAW_KEYS=previous.TARGETS,previous.RAW_KEYS
PATHS,ACCOUNTING_SHA=previous.PATHS,previous.ACCOUNTING_SHA
ARMS=('incumbent_local67','diag_mass_local67','corr_mass_local67')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
MASS_RTOL=16*np.finfo(float).eps
SPD_ATOL=SPD_RTOL=1e-12
def _matmul(left,right):
    # The pinned BLAS may leave spurious floating status on valid dot outputs.
    # Isolate only this operation, then reject every nonfinite result.
    with np.errstate(all='ignore'):result=left@right
    require(np.isfinite(result).all(),'Finite canonical matrix product')
    return result
CONTROL_FLAGS=previous.CONTROL_FLAGS
INCUMBENT_FLAGS=previous.INCUMBENT_FLAGS
DIAGNOSTIC_MODEL_FLAGS=CONTROL_FLAGS+('frozen_diagonal_geometry_checked','same_original_labels_checked','diagonal_local_only_checked','full_diagonal_audit_checked')
DIAGNOSTIC_SELECTION_FLAGS=('passed','independent_original_pre_checked','original_eligibility_unchanged','original_core_unchanged','frozen_diagonal_rank_checked','exact_diagonal_scores_and_choices_checked','no_correlation_selection')
MODEL_FLAGS=previous.MODEL_FLAGS+('weighted_moments_from_original_inputs_checked','weighted_covariance_checked','floor_unit_diagonal_and_zero_cross_checked',
 'shrinkage_spd_and_cholesky_checked','correlation_metric_checked','identity_correlation_reuses_diagonal_checked','independent_matrix_build_counts_checked','full_diagonal_supplied')
SELECTION_FLAGS=previous.SELECTION_FLAGS+('frozen_diagonal_diagnostic_checked','three_arms_single_episode_checked')
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','control_target_fits','control_log_refits','control_local_evaluations','control_local_solves',
 'diag_pooled_fits','diag_log_refits','diag_local_evaluations','diag_local_solves','diag_kernel_evaluations','diag_equal_scale_evaluations','diag_changed_weight_evaluations',
 'candidate_pooled_fits','candidate_log_refits','candidate_local_evaluations','candidate_local_solves','candidate_kernel_evaluations',
 'candidate_identity_correlation_evaluations','candidate_changed_weight_evaluations','correlation_matrix_builds','actual_regression_solves')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells',
 'diag_changed_delta_cells','diag_changed_holdings_events','candidate_changed_delta_cells','candidate_changed_holdings_events',
 'incumbent_episode_calls','diag_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','diag_changed_delta_indices','diag_delta',
 'changed_delta_indices','delta','incumbent_choices','diag_choices','candidate_choices','episode_budget')

def dependency_files():return tuple(sorted(set(previous.dependency_files())|{PREVIOUS,Path(__file__).resolve()}))

def _linear_solves(model):return sum(row['status']=='fit' for row in model['audit']['local'])
def _local_counts(model):
    count=_linear_solves(model)
    return dict(pooled_log_solves=0,local_linear_solves=count,total_regression_solves=count)

def _counts(control,diagonal,candidate=None):
    result=dict(control=previous.independent_solve_counts(control),diagonal=_local_counts(diagonal))
    if candidate is not None:result['candidate']=_local_counts(candidate)
    result['total_regression_solves']=sum(v['total_regression_solves'] for v in result.values())
    return result

DIAGNOSTIC_MODEL_COUNT_FIELDS=previous.MODEL_COUNT_FIELDS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS

def moments_oracle(X,weights):
    """Original-input moments, with a second coordinate/eigen derivation."""
    X,w=np.asarray(X,dtype=np.float64),np.asarray(weights,dtype=np.float64)
    require(X.ndim==2 and X.shape[1]==10 and len(X)>0 and w.shape==(len(X),),'Moment axes')
    require(np.isfinite(X).all() and (np.abs(X)<=.5).all() and np.isfinite(w).all() and (w>0).all(),'Moment input units')
    with threadpool_limits(limits=1),np.errstate(over='raise',divide='raise',invalid='raise'):
        mean=np.average(X,axis=0,weights=w);centered=X-mean
        raw=np.sqrt(np.average(centered*centered,axis=0,weights=w));floor=raw<=1e-12
        scale=np.where(raw>1e-12,raw,1.)
        covariance=_matmul((centered*w[:,None]).T,centered)/w.sum()
        alternative=np.array([[np.dot(w,centered[:,j]*centered[:,k])/w.sum() for k in range(10)] for j in range(10)])
        require(np.allclose(covariance,alternative,atol=CONTINUOUS_ATOL,rtol=CONTINUOUS_RTOL),'Independent coordinate covariance')
        R=np.eye(10)
        for j in range(10):
            for k in range(j+1,10):
                if not floor[j] and not floor[k]:R[j,k]=R[k,j]=covariance[j,k]/(scale[j]*scale[k])
        require(np.isfinite(R).all() and np.array_equal(R,R.T) and np.array_equal(np.diag(R),np.ones(10)),'Exact correlation symmetry and diagonal')
        A=(R+np.eye(10))/2.;L=np.linalg.cholesky(A)
        require(np.allclose(_matmul(L,L.T),A,atol=SPD_ATOL,rtol=SPD_RTOL),'Cholesky residual')
        transform=np.linalg.solve(L,np.diag(1./scale));trace=float(np.sum(transform*transform))
        require(np.isfinite(trace) and trace>0,'Positive transform trace')
        metric=_matmul(transform.T,transform)/trace
        # This separate SPD eigensystem is a proof, never the canonical distance.
        eig,vectors=np.linalg.eigh(A);require((eig>0).all(),'Independent SPD spectrum')
        inv=_matmul(vectors*(1./eig),vectors.T)
        independent=inv/scale[:,None]/scale[None,:];independent/=np.trace(independent)
        require(np.allclose(metric,independent,atol=CONTINUOUS_ATOL,rtol=CONTINUOUS_RTOL),'Independent inverse quadratic metric')
    return dict(source_mean=mean.tolist(),raw_scale=raw.tolist(),source_scale=scale.tolist(),floored=floor.tolist(),
      training_weights=w.tolist(),covariance=covariance.tolist(),correlation=R.tolist(),shrinkage=A.tolist(),
      cholesky=L.tolist(),transform=transform.tolist(),trace=trace,metric=metric.tolist(),identity_correlation=bool(np.array_equal(R,np.eye(10))))

def geometry_oracle(dates,cutoff,X,query,moments):
    d=previous.previous.days(dates,'own dates');cutoff=np.datetime64(cutoff,'D')
    X,q,s=[np.asarray(x,dtype=np.float64) for x in (X,query,moments['source_scale'])]
    require(X.shape==(len(d),10) and q.shape==s.shape==(10,) and len(d)>0,'Correlation geometry axes')
    require(not np.isnat(cutoff) and np.all(d<=cutoff),'Mature geometry clock')
    require(all(np.isfinite(v).all() for v in (X,q,s)) and (s>0).all() and (np.abs(X)<=.5).all() and (np.abs(q)<=.5).all(),'Correlation geometry units')
    R=np.asarray(moments['correlation']);identity=bool(np.array_equal(R,np.eye(10)))
    require(type(moments['identity_correlation']) is bool and moments['identity_correlation']==identity,'Exact R identity branch')
    diagonal=previous.geometry_oracle(d,cutoff,X,q,s) if identity else None
    if identity:
        original=np.asarray(diagonal['original_weights']);weights=np.asarray(diagonal['weights'])
        mass,oldmass=float(weights.sum()),float(original.sum())
        neff=float(mass**2/np.einsum('n,n->',weights,weights,optimize=False))
        oldeff=float(oldmass**2/np.einsum('n,n->',original,original,optimize=False))
        diagonal['effective_samples']=neff;diagonal['original_effective_samples']=oldeff
        take=lambda name:copy.deepcopy(diagonal[name])
        return dict(source_scale=s.tolist(),identity_correlation=True,old_distance=take('old_distance'),distance=take('distance'),
          original_weights=take('original_weights'),unscaled_weights=take('unscaled_weights'),weights=take('weights'),
          original_mass=take('original_mass'),unscaled_mass=take('unscaled_mass'),mass_multiplier=take('mass_multiplier'),
          matched_mass=take('matched_mass'),effective_samples=neff,original_effective_samples=oldeff,
          mass_absolute_error=take('mass_absolute_error'),maximum_absolute_weight_change=take('maximum_absolute_weight_change'),
          weights_exact_original=take('weights_exact_original'),diagonal_geometry=diagonal)
    B=np.asarray(moments['transform'],dtype=float);T=moments['trace']
    require(B.shape==(10,10) and np.isfinite(B).all() and np.isfinite(T) and T>0,'Finite transform')
    ages=int(cutoff.astype('datetime64[M]').astype(int))-d.astype('datetime64[M]').astype(int)
    with threadpool_limits(limits=1),np.errstate(over='raise',divide='raise',invalid='raise',under='ignore'):
        diff=X-q;old_distance=np.mean(diff*diff,axis=1)
        original=np.exp2(-ages/12.)*np.exp(-old_distance/(2*.5**2))
        distance=np.sum(_matmul(diff,B.T)**2,axis=1)/T
        alt=np.einsum('ni,ij,nj->n',diff,np.asarray(moments['metric']),diff,optimize=False)
        require(np.allclose(distance,alt,atol=CONTINUOUS_ATOL,rtol=CONTINUOUS_RTOL),'Independent quadratic distance')
        unscaled=np.exp2(-ages/12.)*np.exp(-distance/(2*.5**2))
        oldmass=float(original.sum());umass=float(unscaled.sum())
        require(oldmass>0 and umass>0,'Positive original and new mass')
        multiplier=oldmass/umass;weights=unscaled*multiplier;mass=float(weights.sum())
        neff=float(mass**2/np.einsum('n,n->',weights,weights,optimize=False));oldeff=float(oldmass**2/np.einsum('n,n->',original,original,optimize=False))
        require(np.isclose(neff,mass*mass/np.dot(weights,weights),atol=CONTINUOUS_ATOL,rtol=CONTINUOUS_RTOL),'Independent effective sample size')
    require(all(np.isfinite(v).all() for v in (distance,original,unscaled,weights)) and (weights>0).all() and (original>0).all(),'Finite positive own weights')
    require(abs(mass-oldmass)<=MASS_RTOL*oldmass,'Mass16eps')
    return dict(source_scale=s.tolist(),identity_correlation=False,old_distance=old_distance.tolist(),distance=distance.tolist(),
      original_weights=original.tolist(),unscaled_weights=unscaled.tolist(),weights=weights.tolist(),original_mass=oldmass,
      unscaled_mass=umass,mass_multiplier=multiplier,matched_mass=mass,effective_samples=neff,original_effective_samples=oldeff,
      mass_absolute_error=abs(mass-oldmass),maximum_absolute_weight_change=float(np.max(np.abs(weights-original))),
      weights_exact_original=weights.tobytes()==original.tobytes(),diagonal_geometry=None)

def candidate_oracle(inp,control,contexts,*,min_own_months=18):
    raw={k:v.copy() for k,v in control['raw'].items()};raw['local_linear']=control['raw']['source_log'].copy()
    dates,ends,X,symbols=(inp[k] for k in ('dates','ends','X','symbols'))
    Q,qd,qs=(inp[k] for k in ('query_X','query_dates','query_symbols'));months=dates.astype('datetime64[M]')
    original={r['event_row']:r for r in control['audit']['local']};annual=[];rows=[]
    with threadpool_limits(limits=1):
        for ctx in contexts:
            ids,pred,s,cutoff=ctx['ids'],ctx['prediction_rows'],ctx['scale'],ctx['cutoff']
            moments=moments_oracle(X[ids],ctx['pooled_weights'])
            exact_array(np.asarray(moments['source_mean']),ctx['mu'],'Original mean exact')
            exact_array(np.asarray(moments['source_scale']),s,'Original effective scale exact')
            annual.append(dict(year=ctx['year'],cutoff=str(cutoff),training_row_ids=ids.tolist(),query_row_ids=pred.tolist(),source_scale=s.tolist(),moments=moments))
            for j in pred:
                own=np.flatnonzero(symbols[ids]==qs[j]);oi=ids[own];on=len(np.unique(months[oi]))
                row=dict(event_row=int(j),date=str(qd[j]),symbol=str(qs[j]),year=ctx['year'],cutoff=str(cutoff),own_rows=len(oi),own_months=on,
                  latest_label_end=str(ends[oi].max()) if len(oi) else None,status='insufficient_own_history',weight_mass=0.,effective_samples=0.,
                  constant_correction=float(original[int(j)]['constant_correction']),linear_correction=0.,own_row_ids=oi.tolist(),geometry=None)
                if on>=min_own_months:
                    g=geometry_oracle(dates[oi],cutoff,X[oi],Q[j],moments);w=np.asarray(g['weights']);y=ctx['residual'][own]
                    mass=float(w.sum());neff=float(mass*mass/np.dot(w,w));linear=0.
                    status='zero_residual' if not np.any(y) else 'fit'
                    if status=='fit':
                        design=np.column_stack((np.ones(len(oi)),(X[oi]-Q[j])/s))
                        linear=float(numeric.least_squares(design,y,w,np.array([12.]+[10.]*10))[0])
                    row.update(status=status,weight_mass=mass,effective_samples=neff,linear_correction=linear,geometry=g)
                    if linear!=0.:raw['local_linear'][j]+=linear
                rows.append(row)
    return dict(raw=raw,available=control['available'].copy(),local_available=control['local_available'].copy(),audit=dict(
      schema='monthly_correlation_distance257_candidate_audit_v1',feature_count=10,target='unchanged cached centered log_excess',
      pooled_copied=True,local_constant_copied=True,residuals='same mature in-sample pooled residuals as control',local_linear_only=True,annual=annual,local=rows))

def verify_supplied_moments(inp,annual,context,wanted=None):
    require(annual['training_row_ids']==context['ids'].tolist() and annual['query_row_ids']==context['prediction_rows'].tolist(),'Original annual cohort IDs')
    if wanted is None:wanted=moments_oracle(inp['X'][context['ids']],context['pooled_weights'])
    # This exact test certifies the declared binary64 construction from source,
    # separately from the independent covariance and SVD tolerances.
    record_check(annual['moments'],wanted,'Supplied moments exact original-input algebra')
    exact_array(np.asarray(annual['source_scale']),context['scale'],'Saved annual scales exact')

def verify_supplied_geometry(inp,row,moments):
    g=row['geometry']
    if g is None:return
    ids=np.asarray(row['own_row_ids'],dtype=int);j=row['event_row']
    rebuilt=geometry_oracle(inp['dates'][ids],np.datetime64(row['cutoff']),inp['X'][ids],inp['query_X'][j],moments)
    record_check(g,rebuilt,'Supplied correlation geometry exact algebra')
    weights,original=np.asarray(g['weights']),np.asarray(g['original_weights'])
    require((weights>0).all() and (original>0).all() and abs(float(weights.sum())-float(original.sum()))<=MASS_RTOL*float(original.sum()),'Supplied mass16eps')

def label_preservation(inp):
    result=previous.label_preservation(inp);result['schema']='monthly_correlation_distance257_labels_preserved_v1';return result

def model_counts(inp,control,diagonal,candidate):
    original=previous.producer_fit_counts(control);diag=previous.model_counts(inp,control,diagonal)
    c={k:0 for k in MODEL_COUNT_FIELDS};c.update(training_rows=len(inp['dates']),valid_training_rows=int(inp['good'].sum()))
    for k in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+k]=original[k]
    for k in ('pooled_fits','log_refits','local_evaluations','local_solves','kernel_evaluations','equal_scale_evaluations','changed_weight_evaluations'):c['diag_'+k]=diag['candidate_'+k]
    local=candidate['audit']['local'];gs=[r['geometry'] for r in local if r['geometry'] is not None]
    c.update(candidate_local_evaluations=len(local),candidate_local_solves=_linear_solves(candidate),candidate_kernel_evaluations=len(gs),
      candidate_identity_correlation_evaluations=sum(g['identity_correlation'] for g in gs),candidate_changed_weight_evaluations=sum(not g['weights_exact_original'] for g in gs),
      correlation_matrix_builds=len(candidate['audit']['annual']))
    c['actual_regression_solves']=original['actual_regression_solves']+c['diag_local_solves']+c['candidate_local_solves'];return c

def model_audit(inp,control,diagonal,candidate):
    out=previous.model_audit(inp,control,diagonal);out['schema']='monthly_correlation_distance257_model_audit_v1'
    out['counts']=model_counts(inp,control,diagonal,candidate);del out['equal_scale_original_weights_exact']
    out.update(diagonal_forecast_reproduced=True,correlation_floor_coordinates_retained=True,identity_correlation_reuses_diagonal=True);return out

def _checked_diagonal(inp,arrays,control,contexts,diagonal,*,min_own_months=18):
    expected=previous.candidate_oracle(inp,control,contexts,min_own_months=min_own_months)
    previous.previous.check_forecast(diagonal,expected,'Frozen diagonal independent forecast')
    _copies(diagonal,control)
    for row in diagonal['audit']['local']:
        previous.verify_supplied_geometry(inp,row)
        if row['geometry'] is not None and row['geometry']['weights_exact_original']:
            j=row['event_row'];exact_array(diagonal['raw']['local_linear'][j:j+1],control['raw']['local_linear'][j:j+1],'Diagonal exact original weights')
    return expected

def _copies(candidate,control):
    for k in ('source_log','local_constant'):exact_array(candidate['raw'][k],control['raw'][k],'Copied original '+k)
    for k in ('available','local_available'):exact_array(candidate[k],control[k],'Copied original '+k)

def _diagnostic_receipt(inp,control,diagonal):
    counts=previous.model_counts(inp,control,diagonal)
    return dict(**{k:True for k in DIAGNOSTIC_MODEL_FLAGS},checked_events=inp['n'],counts=counts,producer_equivalent_counts=counts,
      independent_solve_counts=_counts(control,diagonal),independent_matrix_builds=0,full_diagonal_supplied=True,full_source_inputs_supplied=True,
      correlation_moments_computed=False,correlation_models_computed=False,episode_calls=0,continuous_atol=CONTINUOUS_ATOL,
      continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,producer_modules_imported=False,computes_returns=False)

def verify_diagnostic_models(train,meta,arrays,diagonal,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);own,contexts=control_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    control=copy.deepcopy(own)
    # Independently prove control/ranks, then retain authoritative frozen output
    # bytes for pooled and constant copying while using SVD residuals.
    for k in RAW_KEYS:control['raw'][k]=np.asarray(arrays['raw_'+k])[:inp['n']].reshape(-1).copy()
    previous.verify_control_result(inp,arrays,control,own)
    expected=_checked_diagonal(inp,arrays,control,contexts,diagonal,min_own_months=min_own_months)
    b=dict(n=inp['n'],width=inp['width'],candidate=diagonal);e=dict(b,candidate=expected)
    r,d,*_=previous.rank_oracle(arrays,meta['monthly'],b,inp['n']);rr,dd,*_=previous.rank_oracle(arrays,meta['monthly'],e,inp['n'])
    for k in r:exact_array(r[k],rr[k],'Diagnostic independent rank/'+k)
    exact_array(d,dd,'Diagnostic independent delta')
    return _diagnostic_receipt(inp,own,expected)

def verify_models(train,meta,arrays,bundle,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);n=inp['n']
    keys={'schema','n','width','training_row_ids','control','diagonal','candidate','label_values','label_preservation','control_gate','diagnostic_gate','audit'}
    require(type(bundle) is dict and set(bundle)==keys and bundle['schema']=='monthly_correlation_distance257_models_v1','Full257 bundle')
    require(type(bundle['n']) is int and bundle['n']==n and type(bundle['width']) is int and bundle['width']==inp['width'],'Full model axes')
    exact_array(bundle['training_row_ids'],inp['training_row_ids'],'Original training row IDs')
    own,contexts=control_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    proof=previous.verify_control_result(inp,arrays,bundle['control'],own)
    diagonal=_checked_diagonal(inp,arrays,bundle['control'],contexts,bundle['diagonal'],min_own_months=min_own_months)
    candidate=candidate_oracle(inp,bundle['control'],contexts,min_own_months=min_own_months)
    previous.previous.check_forecast(bundle['candidate'],candidate,'Independent correlation candidate')
    _copies(bundle['candidate'],bundle['control'])
    by_year={ctx['year']:ctx for ctx in contexts};saved={r['year']:r for r in bundle['candidate']['audit']['annual']}
    expected_year={r['year']:r for r in candidate['audit']['annual']}
    for row in saved.values():verify_supplied_moments(inp,row,by_year[row['year']],expected_year[row['year']]['moments'])
    diaglocal={r['event_row']:r for r in bundle['diagonal']['audit']['local']}
    for row in bundle['candidate']['audit']['local']:
        verify_supplied_geometry(inp,row,saved[row['year']]['moments'])
        g=row['geometry'];j=row['event_row']
        if g is not None and g['identity_correlation']:
            record_check(g['diagonal_geometry'],diaglocal[j]['geometry'],'Identity exact diagonal geometry')
            exact_array(bundle['candidate']['raw']['local_linear'][j:j+1],bundle['diagonal']['raw']['local_linear'][j:j+1],'Identity exact diagonal forecast')
        if g is not None and g['weights_exact_original']:
            exact_array(bundle['candidate']['raw']['local_linear'][j:j+1],bundle['control']['raw']['local_linear'][j:j+1],'Original-weight exact local forecast')
    exact_array(bundle['label_values'],inp['targets']['log_excess'],'Original log labels unchanged')
    record_check(bundle['label_preservation'],label_preservation(inp),'Original four labels bytes')
    for arm,expected in ((ARMS[1],diagonal),(ARMS[2],candidate)):
        r,d,*_=rank_oracle(arrays,meta['monthly'],bundle,n,arm=arm)
        alternate=dict(bundle,**{('diagonal' if arm==ARMS[1] else 'candidate'):expected})
        rr,dd,*_=rank_oracle(arrays,meta['monthly'],alternate,n,arm=arm)
        for k in r:exact_array(r[k],rr[k],'Independent exact rank/'+arm+'/'+k)
        exact_array(d,dd,'Independent exact delta/'+arm)
    wanted=model_audit(inp,bundle['control'],diagonal,candidate);record_check(bundle['audit'],wanted,'Full257 model audit')
    gate=bundle['control_gate'];default=wanted['original_cache_parity']
    record_check(gate,default if set(gate)==set(default) else proof,'Original model gate')
    default_diag=dict(passed=True,events=n,frozen_diagonal_implementation=True,correlation_models_computed=False)
    gate=bundle['diagnostic_gate'];diagproof=_diagnostic_receipt(inp,own,diagonal)
    record_check(gate,default_diag if set(gate)==set(default_diag) else diagproof,'Frozen diagonal model gate')
    return dict(**{k:True for k in MODEL_FLAGS},checked_events=n,counts=wanted['counts'],producer_equivalent_counts=wanted['counts'],
      independent_solve_counts=_counts(own,diagonal,candidate),independent_matrix_builds=len(contexts),full_bundle_supplied=True,full_labels_supplied=True,
      full_control_supplied=True,full_candidate_supplied=True,continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
      independent_numeric_system='augmented_least_squares_SVD',producer_modules_imported=False,computes_returns=False)

def rank_oracle(arrays,meta,bundle,n=None,*,arm=ARMS[2]):
    require(arm in ARMS[1:],'Forecast arm')
    # Only the requested forecast body is accessed, never future/corr fields
    # during the diagnostic gate.
    virtual=dict(n=bundle['n'],width=bundle['width'],candidate=bundle['diagonal' if arm==ARMS[1] else 'candidate'])
    return previous.rank_oracle(arrays,meta,virtual,n)

def rank_and_delta(arrays,meta,bundle,n=None,*,arm=ARMS[2]):
    ranks,delta,U,ready,active=rank_oracle(arrays,meta,bundle,n,arm=arm)
    au=dict(schema='monthly_correlation_distance257_rank_delta_v1',events=U.shape[0],ranking_cells=int(U.sum()),ready_rows=ready.tolist(),
      active_indices=active,changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,no_group_ranking=True,
      no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True,pooled_ranks_preserved=True)
    return ranks,delta,au

def _order_list(scores,choices,meta):
    return [dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[i,j]) for j in ids]) for i,ids in enumerate(choices)]

def orders_from_choices(scores,choices,meta):
    return {a:_order_list(scores[a],choices[a],meta) for a in ARMS}

def verify_diagnostic_selection(arrays,meta,bundle,orders,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    _,delta,*_=rank_oracle(arrays,meta,bundle,n,arm=ARMS[1])
    pre=base['P'].copy();changed=delta!=0.;pre[changed]+=delta[changed]
    final,choices=accounts.episode(pre,base)
    exact_array(pre_scores,pre,'Diagnostic exact pre');exact_array(scores,final,'Diagnostic exact final')
    record_check(orders,_order_list(final,choices,meta),'Diagnostic exact ordered choices')
    return dict(**{k:True for k in DIAGNOSTIC_SELECTION_FLAGS},checked_events=n,ordered_choices_checked=n,
      full_diagonal_scores_supplied=True,full_diagonal_pre_scores_supplied=True,numerical_tolerance=0.,
      correlation_selection_computed=False,episode_calls=1,producer_modules_imported=False,computes_returns=False)

def selection_oracle(original,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    _,old,U,ready,active=rank_oracle(original,meta,bundle,n,arm=ARMS[1])
    _,new,Unew,readynew,activenew=rank_oracle(original,meta,bundle,n,arm=ARMS[2])
    exact_array(U,Unew,'Common original rank domain');record_check(active,activenew,'Common weak domain')
    pres={a:base['P'].copy() for a in ARMS};scores={};choices={}
    for a,delta in ((ARMS[1],old),(ARMS[2],new)):
        change=delta!=0.;pres[a][change]+=delta[change]
    for a in ARMS:scores[a],choices[a]=accounts.episode(pres[a],base)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=base['width'],original_eligible_cells=int(base['E'].sum()),
      ranking_cells=int(U.sum()),incumbent_episode_calls=1,diag_episode_calls=1,candidate_episode_calls=1)
    events=[]
    for i in range(n):
        partial=str(meta['dates'][i])=='2016-08-17'
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.min(remaining)>=.38-1e-12,'Original remaining support budget')
        counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(not partial);counts['weak_applicable_cells']+=len(active[i])
        for label,a,delta in (('diag',ARMS[1],old),('candidate',ARMS[2],new)):
            counts[label+'_changed_delta_cells']+=int(np.count_nonzero(delta[i]))
            counts[label+'_changed_holdings_events']+=int(set(choices[ARMS[0]][i])!=set(choices[a][i]))
        events.append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,weak=bool(base['weak'][i]),ranking_indices=np.flatnonzero(U[i]).tolist(),
          active_indices=active[i],diag_changed_delta_indices=np.flatnonzero(old[i]!=0).tolist(),diag_delta=old[i].tolist(),
          changed_delta_indices=np.flatnonzero(new[i]!=0).tolist(),delta=new[i].tolist(),incumbent_choices=choices[ARMS[0]][i],
          diag_choices=choices[ARMS[1]][i],candidate_choices=choices[ARMS[2]][i],episode_budget=float(base['budget'][i])))
    ra={a:rank_and_delta(original,meta,bundle,n,arm=a)[2] for a in ARMS[1:]}
    return scores,choices,pres,dict(schema='monthly_correlation_distance257_decisions_v1',counts=counts,events=events,rank_delta=ra,
      incumbent_gate=None,diagnostic_gate=None,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,
      single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False,diagonal_decisions_reproduced=True)

def verify_selection(arrays,meta,bundle,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n;expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Independent three-arm ordered choices')
    for actual,target,label in ((scores,expected,'Final scores'),(pre_scores,pres,'Pre scores')):
        require(type(actual) is dict and set(actual)==set(ARMS),'Full '+label+' arms')
        for a in ARMS:exact_array(actual[a],target[a],label+'/'+a)
    require(type(audit) is dict and 'incumbent_gate' in audit and 'diagnostic_gate' in audit,'Saved decision gates')
    if audit['incumbent_gate'] is not None:
        proof=verify_incumbent(arrays,meta,orders[ARMS[0]],scores[ARMS[0]],pre_scores[ARMS[0]],n)
        record_check(audit['incumbent_gate'],proof,'Full incumbent gate');wanted['incumbent_gate']=proof
    if audit['diagnostic_gate'] is not None:
        proof=verify_diagnostic_selection(arrays,meta,bundle,orders[ARMS[1]],scores[ARMS[1]],pre_scores[ARMS[1]],n)
        record_check(audit['diagnostic_gate'],proof,'Full diagnostic gate');wanted['diagnostic_gate']=proof
    record_check(audit,wanted,'Full257 decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},checked_events=n,ordered_choices_checked=3*n,counts=wanted['counts'],
      full_bundle_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
      numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three257 accounting arms')
    mapping=dict(zip(ARMS,bridge.ARMS));rename=lambda d:{mapping[k]:v for k,v in d.items()}
    for a in ARMS:
        for name,_,fee in PATHS:
            if fee==0.:
                require(float(results[a][name]['metrics']['fees_paid'])==0. and float(results[a][name]['slices']['full122']['fees_paid'])==0.,'ZERO aggregate fees exact0')
    report=bridge.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={a:report['paths'][mapping[a]] for a in ARMS})

