"""Independent259 annual identity partial pooling; no new producer imports.

Canonical normal equations prove exact forecasts/ranks; separate augmented SVD
and block identities prove the fit. Both kinds of actual solve are reported.
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
_spec=importlib.util.spec_from_file_location('joint259_independent256',PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
numeric,bridge,accounts=previous.numeric,previous.bridge,previous.accounts
exact_array,record_check=previous.exact_array,previous.record_check
continuous,check_record=previous.continuous,previous.check_record
model_inputs,original_ranks,control_oracle=previous.model_inputs,previous.original_ranks,previous.control_oracle
original_base,accounting=previous.original_base,previous.accounting
TARGETS,RAW_KEYS=previous.TARGETS,previous.RAW_KEYS
PATHS,ACCOUNTING_SHA=previous.PATHS,previous.ACCOUNTING_SHA
ARMS=('incumbent_local67','sequential_identity','joint_identity')
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','control_target_fits','control_log_refits','control_local_evaluations',
 'control_local_solves','annual_design_builds','qualified_security_columns','qualified_query_rows','cold_query_rows',
 'no_qualified_columns_years','no_qualified_query_years','zero_centered_design_years','diagnostic_pooled_fits',
 'diagnostic_dummy_solves','candidate_pooled_fits','candidate_joint_solves','actual_new_model_solves','actual_regression_solves')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells',
 'weak_applicable_cells','calm_applicable_cells','diagnostic_changed_delta_cells','candidate_changed_delta_cells',
 'diagnostic_changed_holdings_events','candidate_changed_holdings_events','candidate_vs_diagnostic_changed_holdings_events',
 'incumbent_episode_calls','diagnostic_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','calm','ranking_indices','active_indices',
 'diagnostic_changed_delta_indices','candidate_changed_delta_indices','diagnostic_delta','candidate_delta',
 'incumbent_choices','diagnostic_choices','candidate_choices','episode_budget')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
CONTROL_FLAGS=previous.CONTROL_FLAGS
INCUMBENT_FLAGS=previous.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
MODEL_FLAGS=CONTROL_FLAGS+('original_labels_unchanged','original_pooled_and_constant_copied',
 'training_only_identity_columns_checked','unqualified_training_rows_retained','weighted_dummy_centering_checked',
 'original_effective_scaler_checked','canonical_normal_equations_checked','independent_augmented_least_squares',
 'joint_block_identities_checked','cold_query_pooled_copy_checked','full_model_audit_checked','actual_fit_counts_checked',
 'no_future_training_values')
SELECTION_FLAGS=tuple(k for k in previous.SELECTION_FLAGS if k not in
 ('weak_only_local_linear_delta_checked','calm_nonweak_slot_unchanged'))+(
 'full_original_local_quarter_checked','new_rank_minus_old_local_checked','three_arms_single_episode_checked')

def dependency_files():return tuple(sorted(set(previous.dependency_files())|{PREVIOUS,Path(__file__).resolve()}))

def verify_control_models(train,meta,arrays,control,n=None,*,min_months=18,min_own_months=18):
    proof=previous.verify_control(train,meta,arrays,control,n,min_months=min_months,min_own_months=min_own_months)
    return dict(proof,independent_canonical_solve_counts=dict(pooled_log_solves=0,sequential_dummy_solves=0,
      joint_block_solves=0,total_regression_solves=0),independent_design_builds=0)
verify_control=verify_control_models
verify_incumbent_selection=previous.verify_incumbent
verify_incumbent=verify_incumbent_selection

def finite(value,name):
    require(np.isfinite(value).all(),'Finite '+name)
    return value

def normal_solve(design,response,weights,penalties):
    """The prospective canonical path, independently written from its contract."""
    with threadpool_limits(limits=1),np.errstate(over='raise',divide='raise',invalid='raise'):
        lhs=np.einsum('ni,n,nj->ij',design,weights,design,optimize=False)
        lhs+=np.diag(penalties)
        rhs=np.einsum('ni,n,n->i',design,weights,response,optimize=False)
        finite(lhs,'normal matrix');finite(rhs,'normal RHS')
        solution=np.linalg.solve(lhs,rhs)
    return finite(solution,'normal solution'),lhs,rhs

def identity_design(inp,context,min_own_months=18):
    """Full annual rows, lexical qualified IDs, exact original weighted units."""
    require(type(min_own_months) is int and min_own_months>0,'Positive own-month minimum')
    ids=context['ids'];X=inp['X'][ids];symbols=inp['symbols'][ids]
    months=inp['dates'][ids].astype('datetime64[M]')
    require(len(set(zip(symbols.tolist(),months.astype(str).tolist())))==len(ids),'Unique identity/month')
    unique,inverse,count=np.unique(months,return_inverse=True,return_counts=True)
    age=int(context['cutoff'].astype('datetime64[M]').astype(int))-unique.astype(int)
    w=1./count[inverse].astype(float);w*=np.exp2(-age[inverse]/12.);w=w*(len(w)/w.sum())
    finite(w,'original weights');require((w>0).all(),'Positive original weights')
    mu=np.average(X,axis=0,weights=w);centered=X-mu
    scale=np.sqrt(np.average(centered*centered,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    Z=centered/scale;y=inp['targets']['log_excess'][ids];a=float(np.average(y,weights=w))
    firm_months={str(s):int(len(np.unique(months[symbols==s]))) for s in np.unique(symbols)}
    firms=sorted(s for s,k in firm_months.items() if k>=min_own_months)
    H=np.zeros((len(ids),len(firms)),dtype=np.float64)
    for j,s in enumerate(firms):H[:,j]=symbols==s
    p=np.einsum('i,ij->j',w,H,optimize=False)/np.sum(w)
    Hc=H-p;pred=context['prediction_rows'];qs=inp['query_symbols'][pred]
    qualified=np.array([str(s) in firms for s in qs],dtype=bool)
    qids=pred[qualified];Hq=np.zeros((len(qids),len(firms)),dtype=np.float64)
    for i,j in enumerate(qids):Hq[i,firms.index(str(inp['query_symbols'][j]))]=1.
    Hcq=Hq-p;Zq=(inp['query_X'][qids]-mu)/scale
    require(all(np.isfinite(v).all() for v in (mu,scale,Z,y,Hc,Hcq,Zq)),'Finite annual design')
    beta,lhs,rhs=normal_solve(Z,y-a,w,np.full(10,10.))
    continuous(mu,context['mu'],'Independent original mean');continuous(scale,context['scale'],'Independent original scale')
    continuous(beta,context['coefficient'],'Independent SVD pooled beta')
    reason='no_qualified_columns' if not firms else 'no_qualified_query' if not len(qids) else 'zero_centered_design' if not np.any(Hc!=0.) else 'fit'
    return dict(ids=ids,pred=pred,qids=qids,firms=firms,firm_months=firm_months,w=w,mu=mu,scale=scale,
      a=a,Z=Z,H=H,Hc=Hc,p=p,y=y,Zq=Zq,Hcq=Hcq,beta0=beta,pooled_gram=lhs,pooled_rhs=rhs,status=reason)

def fit_identity(design,*,verify_svd=True):
    """Return both fits; no producer functions and no source/data loading."""
    d=design;beta=d['beta0'];J=len(d['firms']);zero=np.zeros(J)
    out=dict(sequential_beta=beta.copy(),sequential_delta=zero.copy(),joint_beta=beta.copy(),joint_delta=zero.copy(),
      sequential_gram=None,sequential_rhs=None,joint_gram=None,joint_rhs=None)
    if d['status']!='fit':return out
    Z,H,w,y=d['Z'],d['Hc'],d['w'],d['y']-d['a']
    residual=y-np.einsum('ij,j->i',Z,beta,optimize=False)
    seq,D,dr=normal_solve(H,residual,w,np.full(J,12.))
    B=np.column_stack((Z,H));pen=np.r_[np.full(10,10.),np.full(J,12.)]
    joint,G,r=normal_solve(B,y,w,pen)
    out.update(sequential_delta=seq,joint_beta=joint[:10],joint_delta=joint[10:],
      sequential_gram=D,sequential_rhs=dr,joint_gram=G,joint_rhs=r)
    if verify_svd:
        # Independent sqrt(W) augmented systems, rather than normal equations.
        svdseq=numeric.least_squares(H,residual,w,np.full(J,12.))
        svdjoint=numeric.least_squares(B,y,w,pen)
        continuous(seq,svdseq,'Sequential augmented SVD')
        continuous(joint,svdjoint,'Joint augmented SVD')
        cross=np.einsum('ni,n,nj->ij',Z,w,H,optimize=False)
        c=np.einsum('ni,n,n->i',Z,w,y,optimize=False)
        rH=np.einsum('ni,n,n->i',H,w,y,optimize=False)
        # Coordinate dot products are a second derivation of the block identities.
        independent_cross=np.array([[np.dot(w*Z[:,j],H[:,k]) for k in range(J)] for j in range(10)])
        continuous(cross,independent_cross,'Independent cross block')
        continuous(G[:10,:10],d['pooled_gram'],'Shared feature block')
        continuous(G[10:,10:],D,'Shared identity block')
        continuous(G[:10,10:],cross,'Joint coupling block')
        continuous(np.einsum('ij,j->i',d['pooled_gram'],joint[:10],optimize=False)+
          np.einsum('ij,j->i',cross,joint[10:],optimize=False),c,'Joint feature normal identity')
        continuous(np.einsum('ji,j->i',cross,joint[:10],optimize=False)+
          np.einsum('ij,j->i',D,joint[10:],optimize=False),rH,'Joint dummy normal identity')
        continuous(np.einsum('ni,n->i',H,w,optimize=False),np.zeros(J),'Weighted dummy centering')
    return out

def label_preservation(inp):
    result=previous.label_preservation(inp)
    result['schema']='monthly_joint_identity259_labels_preserved_v1'
    return result

def identity_forecasts(inp,control,contexts,*,min_own_months=18,verify_svd=True):
    """Both forecast roles share independently built annual designs."""
    result={role:dict(raw={k:v.copy() for k,v in control['raw'].items()},
      available=control['available'].copy(),local_available=control['local_available'].copy()) for role in ('diagnostic','candidate')}
    for value in result.values():value['raw']['local_linear']=control['raw']['source_log'].copy()
    annual={role:[] for role in result}
    digest=lambda a:hashlib.sha256(np.asarray(a,dtype=np.float64).tobytes()).hexdigest()
    with threadpool_limits(limits=1):
        for ctx in contexts:
            d=identity_design(inp,ctx,min_own_months);fit=fit_identity(d,verify_svd=verify_svd)
            pool=np.einsum('ni,i->n',(inp['query_X'][d['pred']]-d['mu'])/d['scale'],d['beta0'],optimize=False)+d['a']
            exact_array(pool,np.asarray(control['raw']['source_log'])[d['pred']],'Canonical original pooled forecast')
            common=dict(year=int(ctx['year']),cutoff=str(ctx['cutoff']),training_row_ids=d['ids'].tolist(),query_row_ids=d['pred'].tolist(),
              qualified_symbols=d['firms'],qualified_query_row_ids=d['qids'].tolist(),
              unqualified_query_row_ids=[int(j) for j in d['pred'] if j not in set(d['qids'])],own_month_counts=d['firm_months'],
              weight_sum=float(np.sum(d['w'])),weights=d['w'].tolist(),source_mean=d['mu'].tolist(),source_scale=d['scale'].tolist(),
              source_intercept=d['a'],source_beta=d['beta0'].tolist(),dummy_mean=d['p'].tolist(),raw_dummy_sha256=digest(d['H']),
              centered_dummy_sha256=digest(d['Hc']),standardized_design_sha256=digest(d['Z']),
              weighted_dummy_sum=np.einsum('i,ij->j',d['w'],d['Hc'],optimize=False).tolist(),
              feature_dummy_cross=np.einsum('ni,n,nj->ij',d['Z'],d['w'],d['Hc'],optimize=False).tolist(),
              centered_dummy_zero=bool(not np.any(d['Hc']!=0.)),status=d['status'])
            for role,tag in (('diagnostic','sequential'),('candidate','joint')):
                beta,delta=fit[tag+'_beta'],fit[tag+'_delta'];gram,rhs=fit[tag+'_gram'],fit[tag+'_rhs']
                penalty=[12.]*len(d['firms']) if role=='diagnostic' else [10.]*10+[12.]*len(d['firms'])
                annual[role].append(dict(common,global_beta=beta.tolist(),dummy_delta=delta.tolist(),
                  gram=None if gram is None else gram.tolist(),rhs=None if rhs is None else rhs.tolist(),penalty=penalty,
                  fit_performed=d['status']=='fit'))
                if d['status']=='fit':
                    prediction=(d['a']+np.einsum('ij,j->i',d['Zq'],beta,optimize=False))+np.einsum('ij,j->i',d['Hcq'],delta,optimize=False)
                    result[role]['raw']['local_linear'][d['qids']]=finite(prediction,'identity forecast')
    for role,value in result.items():
        value['audit']=dict(schema='monthly_joint_identity259_forecast_audit_v1',role=role,feature_count=10,
          target='unchanged cached centered log_excess',source_log_copied=True,local_constant_copied=True,
          availability_preserved=True,local_support_preserved=True,security_intercepts_only=True,global_penalty=10.,dummy_penalty=12.,
          source_global_parameters_fixed=role=='diagnostic',common_weights=True,annual=annual[role])
    return result['diagnostic'],result['candidate']

def model_counts(inp,control,diagnostic,candidate):
    old=previous.producer_fit_counts(control);rows=candidate['audit']['annual']
    count={k:0 for k in MODEL_COUNT_FIELDS}
    count.update(training_rows=len(inp['dates']),valid_training_rows=int(inp['good'].sum()))
    for key in ('target_fits','log_refits','local_evaluations','local_solves'):count['control_'+key]=old[key]
    count.update(annual_design_builds=len(rows),qualified_security_columns=sum(len(r['qualified_symbols']) for r in rows),
      qualified_query_rows=sum(len(r['qualified_query_row_ids']) for r in rows),cold_query_rows=sum(len(r['unqualified_query_row_ids']) for r in rows),
      no_qualified_columns_years=sum(r['status']=='no_qualified_columns' for r in rows),
      no_qualified_query_years=sum(r['status']=='no_qualified_query' for r in rows),
      zero_centered_design_years=sum(r['status']=='zero_centered_design' for r in rows),
      diagnostic_dummy_solves=sum(r['fit_performed'] for r in diagnostic['audit']['annual']),
      candidate_joint_solves=sum(r['fit_performed'] for r in rows))
    count['actual_new_model_solves']=count['diagnostic_dummy_solves']+count['candidate_joint_solves']
    count['actual_regression_solves']=old['actual_regression_solves']+count['actual_new_model_solves']
    return count

def model_audit(inp,control,diagnostic,candidate):
    return dict(schema='monthly_joint_identity259_model_audit_v1',counts=model_counts(inp,control,diagnostic,candidate),
      original_cache_parity=dict(passed=True,events=inp['n'],original_raw_exact=True,original_availability_exact=True,
        original_ranks_exact=True,candidate_models_computed=False),original_row_order_preserved=True,
      original_common_good_preserved=True,all_targets_preserved=True,diagnostic_source_log_copied=True,
      candidate_source_log_copied=True,diagnostic_local_constant_copied=True,candidate_local_constant_copied=True,
      availability_preserved=True,local_support_preserved=True,common_design_weights_penalties=True,
      security_intercepts_only=True,candidate_global_coefficients_refit=True,computes_returns=False)

def verify_models(train,meta,arrays,bundle,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);n=inp['n'];width=inp['width']
    keys={'schema','n','width','training_row_ids','control','diagnostic','candidate','label_values','label_preservation','control_gate','audit'}
    require(type(bundle) is dict and set(bundle)==keys,'Full259 bundle')
    require(bundle['schema']=='monthly_joint_identity259_models_v1' and type(bundle['n']) is int and bundle['n']==n
      and type(bundle['width']) is int and bundle['width']==width,'Model bundle axes')
    exact_array(bundle['training_row_ids'],inp['training_row_ids'],'Original consumed row identities')
    expected,contexts=control_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    proof=previous.verify_control_result(inp,arrays,bundle['control'],expected)
    proof=dict(proof,independent_canonical_solve_counts=dict(pooled_log_solves=0,sequential_dummy_solves=0,joint_block_solves=0,total_regression_solves=0),independent_design_builds=0)
    exact_array(bundle['label_values'],inp['targets']['log_excess'],'Preserved original log label bytes')
    record_check(bundle['label_preservation'],label_preservation(inp),'Original label preservation')
    diagnostic,candidate=identity_forecasts(inp,bundle['control'],contexts,min_own_months=min_own_months)
    for role,wanted in (('diagnostic',diagnostic),('candidate',candidate)):
        actual=bundle[role]
        require(type(actual) is dict and set(actual)==set(wanted),'Full forecast/'+role)
        require(type(actual['raw']) is dict and set(actual['raw'])==set(RAW_KEYS),'Raw keys/'+role)
        for key in RAW_KEYS:exact_array(actual['raw'][key],wanted['raw'][key],'Canonical full raw/'+role+'/'+key)
        for key in ('available','local_available'):exact_array(actual[key],wanted[key],'Original support/'+role+'/'+key)
        record_check(actual['audit'],wanted['audit'],'Canonical design/normal equations/'+role)
    audit=model_audit(inp,bundle['control'],diagnostic,candidate)
    record_check(bundle['audit'],audit,'Complete model audit')
    gate=bundle['control_gate'];require(type(gate) is dict and gate.get('passed') is True,'Passed early original control')
    record_check(gate,audit['original_cache_parity'] if set(gate)==set(audit['original_cache_parity']) else proof,'Full saved control gate')
    for arm in ARMS[1:]:
        r,d,_=rank_and_delta(arrays,meta['monthly'],bundle,n,arm=arm)
        expected_bundle=dict(bundle,diagnostic=diagnostic,candidate=candidate)
        rr,dd,_=rank_and_delta(arrays,meta['monthly'],expected_bundle,n,arm=arm)
        for key in r:exact_array(r[key],rr[key],'Exact complete ranks/'+arm+'/'+key)
        exact_array(d,dd,'Exact score delta/'+arm)
    c=audit['counts'];ns,nj=c['diagnostic_dummy_solves'],c['candidate_joint_solves'];old=proof['independent_solve_counts']
    svd=dict(control=old,sequential=dict(dummy_solves=ns,total_regression_solves=ns),joint=dict(block_solves=nj,total_regression_solves=nj),
      total_regression_solves=old['total_regression_solves']+ns+nj)
    canonical=dict(pooled_log_solves=len(contexts),sequential_dummy_solves=ns,joint_block_solves=nj,total_regression_solves=len(contexts)+ns+nj)
    return dict(**{k:True for k in MODEL_FLAGS},checked_events=n,counts=c,producer_equivalent_counts=c,independent_solve_counts=svd,
      independent_canonical_solve_counts=canonical,independent_design_builds=len(contexts),full_bundle_supplied=True,
      full_labels_supplied=True,full_control_supplied=True,full_diagnostic_supplied=True,full_candidate_supplied=True,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
      independent_numeric_system='augmented_least_squares_SVD_and_canonical_normal_equations',producer_modules_imported=False,computes_returns=False)

def rank_oracle(arrays,meta,bundle,n=None,*,arm=ARMS[2]):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']) and arm in ARMS[1:],'Decision prefix and arm')
    old,local,U=original_ranks(arrays,n);width=old.shape[1]
    require(type(bundle) is dict and type(bundle.get('n')) is int and bundle['n']>=n and bundle.get('width')==width,'Model prefix layout')
    role='diagnostic' if arm==ARMS[1] else 'candidate';forecast=bundle[role];size=bundle['n']*width
    require(set(forecast['raw'])==set(RAW_KEYS),'Full forecast raw fields')
    raw={k:np.asarray(forecast['raw'][k]) for k in RAW_KEYS}
    require(all(v.shape==(size,) and v.dtype==np.dtype('float64') for v in raw.values()),'Forecast float64 axes')
    available=np.asarray(forecast['available']);require(available.shape==(size,) and available.dtype==np.dtype(bool),'Availability bool axis')
    raw={k:v[:n*width].reshape(n,width) for k,v in raw.items()};available=available[:n*width].reshape(n,width)
    weak=np.asarray(arrays['state_weak_narrow']);calm=np.asarray(arrays['state_calm10']);ext=np.asarray(arrays['extension_available'])
    require(weak.dtype==calm.dtype==ext.dtype==np.dtype(bool) and weak.shape[0]>=n and calm.shape[0]>=n and ext.shape[0]>=n
      and ext.shape[1]==width,'Original state masks')
    coefficient=.25*(.05*weak[:n,None]+.01*(calm[:n]&~weak[:n])[:,None]*ext[:n])
    newpool=old.copy();newlocal=local.copy();delta=np.zeros_like(old);ready=np.zeros(n,bool);active=[]
    for i,use in enumerate(U):
        if use.any():
            require(available[i,use].all() and all(np.isfinite(v[i,use]).all() for v in raw.values()),'Complete original U forecasts')
            exact_array(raw['source_log'][i,use],np.asarray(arrays['raw_source_log'])[i,use],'Copied pooled raw')
            exact_array(raw['local_constant'][i,use],np.asarray(arrays['raw_local_constant'])[i,use],'Copied constant raw')
            ready[i]=True
            newlocal[i,use]=old[i,use] if np.array_equal(raw['local_linear'][i,use],raw['source_log'][i,use]) else previous._rank(raw['local_linear'][i,use])
        mask=use&(coefficient[i]!=0.)
        if str(meta['dates'][i])=='2016-08-17':mask[:]=False
        ids=np.flatnonzero(mask);active.append(ids.tolist())
        delta[i,mask]=coefficient[i,mask]*(newlocal[i,mask]-local[i,mask])
    finite(delta,'local-quarter delta')
    return dict(old_pool=old,old_local=local,new_pool=newpool,new_local=newlocal),delta,U,ready,active,coefficient

def rank_and_delta(arrays,meta,bundle,n=None,*,arm=ARMS[2]):
    ranks,delta,U,ready,active,c=rank_oracle(arrays,meta,bundle,n,arm=arm)
    return ranks,delta,dict(schema='monthly_joint_identity259_rank_delta_v1',arm=arm,events=len(U),ranking_cells=int(U.sum()),
      ready_rows=ready.tolist(),active_indices=active,changed_cells=int(np.count_nonzero(delta)),coefficient=c.tolist(),
      original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,pooled_ranks_preserved=True,
      full_original_local_quarter=True,partial_exact=True,calm_quarter_included=True,no_post_rerank=True)

def selection_oracle(arrays,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n)
    results={a:rank_and_delta(arrays,meta,bundle,n,arm=a) for a in ARMS[1:]}
    pres={a:base['P'].copy() for a in ARMS};scores={};choices={}
    for arm in ARMS[1:]:
        delta=results[arm][1];changed=delta!=0.;pres[arm][changed]+=delta[changed]
    for arm in ARMS:scores[arm],choices[arm]=accounts.episode(pres[arm],base)
    U=original_ranks(arrays,n)[2];c={k:0 for k in COUNT_FIELDS};c.update(events=n,symbols_per_event=base['width'],
      original_eligible_cells=int(base['E'].sum()),ranking_cells=int(U.sum()),incumbent_episode_calls=1,diagnostic_episode_calls=1,candidate_episode_calls=1)
    events=[]
    for i in range(n):
        partial=str(meta['dates'][i])=='2016-08-17';active=results[ARMS[2]][2]['active_indices'][i]
        ds,dj=results[ARMS[1]][1][i],results[ARMS[2]][1][i]
        c['initial_partial_events']+=int(partial);c['formal_events']+=int(not partial)
        c['weak_applicable_cells']+=len(active) if base['weak'][i] else 0
        c['calm_applicable_cells']+=len(active) if base['calm'][i] and not base['weak'][i] else 0
        c['diagnostic_changed_delta_cells']+=int(np.count_nonzero(ds));c['candidate_changed_delta_cells']+=int(np.count_nonzero(dj))
        c['diagnostic_changed_holdings_events']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[0]][i]))
        c['candidate_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[0]][i]))
        c['candidate_vs_diagnostic_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[1]][i]))
        events.append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,weak=bool(base['weak'][i]),calm=bool(base['calm'][i]),
          ranking_indices=np.flatnonzero(U[i]).tolist(),active_indices=active,diagnostic_changed_delta_indices=np.flatnonzero(ds).tolist(),
          candidate_changed_delta_indices=np.flatnonzero(dj).tolist(),diagnostic_delta=ds.tolist(),candidate_delta=dj.tolist(),
          incumbent_choices=choices[ARMS[0]][i],diagnostic_choices=choices[ARMS[1]][i],candidate_choices=choices[ARMS[2]][i],episode_budget=float(base['budget'][i])))
    audit=dict(schema='monthly_joint_identity259_decisions_v1',counts=c,events=events,rank_delta={a:results[a][2] for a in ARMS[1:]},
      incumbent_gate=None,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,
      single_original_episode_per_arm=True,full_original_local_quarter=True,partial_exact=True,computes_returns=False)
    return scores,choices,pres,audit

def orders_from_choices(scores,choices,meta):
    return {a:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[a][i,j]) for j in ids])
      for i,ids in enumerate(choices[a])] for a in ARMS}

def verify_selection(arrays,meta,bundle,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Exact three-arm ordered choices')
    for actual,target,label in ((scores,expected,'Final score'),(pre_scores,pres,'Pre score')):
        require(type(actual) is dict and set(actual)==set(ARMS),'Complete '+label+' arms')
        for arm in ARMS:exact_array(actual[arm],target[arm],label+'/'+arm)
    require(type(audit) is dict and 'incumbent_gate' in audit,'Full saved decision audit')
    if audit['incumbent_gate'] is not None:
        proof=verify_incumbent_selection(arrays,meta,orders[ARMS[0]],scores[ARMS[0]],pre_scores[ARMS[0]],n)
        record_check(audit['incumbent_gate'],proof,'Complete original decision gate');wanted['incumbent_gate']=proof
    record_check(audit,wanted,'Complete259 decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},checked_events=n,ordered_choices_checked=3*n,counts=wanted['counts'],
      full_bundle_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,
      numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three259 accounting arms')
    mapping=dict(zip(ARMS,bridge.ARMS));rename=lambda d:{mapping[k]:v for k,v in d.items()}
    for arm in ARMS:
        for path,_,fee in PATHS:
            if fee==0.:
                require(float(results[arm][path]['metrics']['fees_paid'])==0. and float(results[arm][path]['slices']['full122']['fees_paid'])==0.,'ZERO aggregate fees exact0')
    report=bridge.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={a:report['paths'][mapping[a]] for a in ARMS})
