"""Caller-supplied 266 paired trees; no data loaders, returns or registration.

The two callbacks are mandatory in a future actual host. Omitted callbacks are
explicitly marked synthetic. The host must independently verify the returned
full bundle before deploying either experimental decision stream.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parents[1]
LEGACY_PATH=ROOT/'work/monthly_weak_tree261_kernel.py'
LEGACY_SHA='36595e322ca41138fd71721d0ff27507b42554602806b8b7ff7cf39748d3641f'
if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()!=LEGACY_SHA:
    raise ValueError('Changed frozen261 producer')
_spec=importlib.util.spec_from_file_location('monotone266_frozen261',LEGACY_PATH)
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
base,model,bridge,require=legacy.base,legacy.model,legacy.bridge,legacy.require
TARGETS=legacy.TARGETS
INITIAL_PARTIAL=legacy.INITIAL_PARTIAL
POLICY=dict(legacy.POLICY)
ARMS=('incumbent_local67','unconstrained_weak_tree261_control','monotone_raw_weak_tree_candidate')
ROLES=('calibration_ridge','unconstrained_control','monotone_candidate')
TREE_ROLES=ROLES[1:]
TREE_PARAMETERS=deepcopy(legacy.TREE_PARAMETERS)
BUNDLE_KEYS=('schema','n','width','training_row_ids','calibration_ridge','unconstrained_control','monotone_candidate','label_values','label_preservation','calibration_gate','control_gate','audit')
FORECAST_KEYS=legacy.FORECAST_KEYS
LABEL_KEYS=legacy.LABEL_KEYS
MODEL_COUNT_FIELDS=('events','training_rows','valid_training_rows','ordinary_years','supported_years','cold_years','no_prediction_years','matched_ridge_fits','prediction_rows','unconstrained_tree_fits','unconstrained_boosting_iterations','unconstrained_trees','unconstrained_tree_nodes','unconstrained_manual_tree_traversals','monotone_tree_fits','monotone_boosting_iterations','monotone_trees','monotone_tree_nodes','monotone_manual_tree_traversals','original_other_target_fits','original_local_fits','actual_model_fits')
RIDGE_ANNUAL_KEYS=legacy.RIDGE_ANNUAL_KEYS
TREE_ANNUAL_KEYS=legacy.TREE_ANNUAL_KEYS
MODEL_AUDIT_KEYS=legacy.MODEL_AUDIT_KEYS
TREE_KEYS=legacy.TREE_KEYS+('role','shape_certificate')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells','control_changed_delta_cells','candidate_changed_delta_cells','control_changed_holdings_events','candidate_changed_holdings_events','candidate_control_changed_holdings_events','incumbent_episode_calls','control_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','control_changed_delta_indices','candidate_changed_delta_indices','control_delta','candidate_delta','incumbent_choices','control_choices','candidate_choices','episode_budget')
AUDIT_KEYS=legacy.AUDIT_KEYS
RANK_AUDIT_KEYS=('schema','events','ranking_cells','active_indices','control_changed_cells','candidate_changed_cells','original_ranking_population','no_group_ranking','no_own_history_filter','local_basis_preserved','weak_only','partial_exact','calm_slot_preserved','no_post_rerank')
_axes=legacy._axes
model_inputs=legacy.model_inputs
fit_ridge=legacy.fit_ridge
_pack,_unpack,_digest=legacy._pack,legacy._unpack,legacy._digest
_source_snapshot=legacy._source_snapshot
_episode=legacy._episode

def _schema(s):return 'monthly_monotone_tree266_'+s+'_v1'
def dependency_files():return tuple(dict.fromkeys((LEGACY_PATH,)+tuple(legacy.dependency_files())))

def parameters(role):
    require(role in TREE_ROLES,'Known tree role')
    p=deepcopy(TREE_PARAMETERS)
    if role=='monotone_candidate':p['monotonic_cst']=[1,0,0,0,0,0,0,0,0,0]
    return p

def _parameters_check(value,role):
    expected=parameters(role)
    require(type(value) is dict and set(value)==set(expected),'Exact21 parameter names')
    for key,wanted in expected.items():
        got=value[key]
        require(type(got) is type(wanted),'Native parameter type:'+key)
        if isinstance(wanted,list):
            require(len(got)==10 and all(type(x) is int for x in got),'Ten native integer constraints')
        require(got==wanted,'Fixed parameter:'+key)

def _shape_certificate(tree):
    role=tree['role'];required=role=='monotone_candidate';records=[]
    if required:
        for record in tree['trees']:
            nodes=record['nodes']
            def leaves(start):
                todo=[start];seen=set();out=[]
                while todo:
                    j=todo.pop();require(type(j) is int and 0<=j<len(nodes) and j not in seen,'Shape child topology');seen.add(j)
                    node=nodes[j]
                    if node['is_leaf']:out.append(j)
                    else:todo.extend([node['right'],node['left']])
                return sorted(out)
            for j,node in enumerate(nodes):
                if node['is_leaf'] or node['feature_idx']!=0:continue
                left,right=leaves(node['left']),leaves(node['right'])
                require(left and right,'Nonempty constrained descendants')
                lv=max(float.fromhex(nodes[i]['value']) for i in left)
                rv=min(float.fromhex(nodes[i]['value']) for i in right)
                require(np.isfinite(lv) and np.isfinite(rv) and lv<=rv,'Feature0 descendant-leaf monotonicity')
                records.append(dict(iteration=record['iteration'],node_index=j,left_leaf_indices=left,right_leaf_indices=right,left_max=lv.hex(),right_min=rv.hex()))
    return dict(schema=_schema('shape_certificate'),role=role,feature_index=0,required=required,constrained_nodes=records,passed=True)

def serialize_tree(estimator,role):
    _parameters_check(estimator.get_params(deep=False),role)
    require(estimator.is_categorical_ is None and estimator._preprocessor is None and estimator._bin_mapper.known_categories is None,'Numeric-only tree features')
    records=[]
    for i,group in enumerate(estimator._predictors):
        require(len(group)==1,'One regression tree per iteration')
        p=group[0];require(p.nodes.dtype==PREDICTOR_RECORD_DTYPE,'Pinned node dtype')
        rows=[]
        for node in p.nodes:
            rows.append({k:float(node[k]).hex() if p.nodes.dtype[k].kind=='f' else int(node[k]) for k in p.nodes.dtype.names})
        records.append(dict(iteration=i,class_index=0,nodes=rows,raw_left_cat_bitsets=_pack(p.raw_left_cat_bitsets),binned_left_cat_bitsets=_pack(p.binned_left_cat_bitsets)))
    out=dict(schema=_schema('serialized_tree'),parameters=parameters(role),n_features=int(estimator.n_features_in_),n_iter=int(estimator.n_iter_),n_trees_per_iteration=int(estimator.n_trees_per_iteration_),baseline=_pack(estimator._baseline_prediction),node_dtype=[[k,PREDICTOR_RECORD_DTYPE[k].str] for k in PREDICTOR_RECORD_DTYPE.names],trees=records,bin_thresholds=[_pack(x) for x in estimator._bin_mapper.bin_thresholds_],n_bins_non_missing=[int(x) for x in estimator._bin_mapper.n_bins_non_missing_],missing_bin=int(estimator._bin_mapper.missing_values_bin_idx_),is_categorical=None,known_categories=None,preprocessor=None)
    out.update(role=role,shape_certificate=None)
    out['shape_certificate']=_shape_certificate(out)
    validate_tree(out,role);return out

def validate_tree(tree,role=None):
    require(type(tree) is dict and set(tree)==set(TREE_KEYS) and tree['schema']==_schema('serialized_tree'),'Serialized model schema')
    wanted=tree['role'] if role is None else role
    require(tree['role']==wanted and wanted in TREE_ROLES,'Exact tree role')
    _parameters_check(tree['parameters'],wanted)
    require(all(type(tree[k]) is int for k in ('n_features','n_iter','n_trees_per_iteration')) and tree['n_features']==10 and tree['n_iter']==100 and tree['n_trees_per_iteration']==1,'Fixed fitted estimator shape')
    require(tree['node_dtype']==[[k,PREDICTOR_RECORD_DTYPE[k].str] for k in PREDICTOR_RECORD_DTYPE.names],'Exact node dtype declaration')
    baseline=_unpack(tree['baseline'],'f');require(baseline.shape==(1,1) and baseline.dtype==np.dtype(float) and np.isfinite(baseline).all(),'Finite baseline')
    require(all(tree[k] is None for k in ('is_categorical','known_categories','preprocessor')) and tree['missing_bin']==255,'Numerical source binning')
    require(len(tree['bin_thresholds'])==len(tree['n_bins_non_missing'])==10,'Ten bin maps')
    for p,nb in zip(tree['bin_thresholds'],tree['n_bins_non_missing']):
        a=_unpack(p,'f');require(a.ndim==1 and np.isfinite(a).all() and np.all(a[1:]>=a[:-1]) and type(nb) is int and 1<=nb<=255 and len(a)==nb-1,'Numeric thresholds')
    require(len(tree['trees'])==100,'Fixed100 iterations')
    for i,r in enumerate(tree['trees']):
        require(set(r)=={'iteration','class_index','nodes','raw_left_cat_bitsets','binned_left_cat_bitsets'} and r['iteration']==i and r['class_index']==0,'Predictor iteration order')
        for key in ('raw_left_cat_bitsets','binned_left_cat_bitsets'):
            a=_unpack(r[key],'u');require(a.shape==(0,8) and a.dtype==np.dtype('uint32'),'No categorical bitsets')
        nodes=r['nodes'];require(0<len(nodes)<=7,'Depth2 node count');seen=set();stack=[(0,0)]
        while stack:
            j,depth=stack.pop();require(type(j) is int and 0<=j<len(nodes) and j not in seen,'Acyclic unique reachable children');seen.add(j);node=nodes[j]
            require(set(node)==set(PREDICTOR_RECORD_DTYPE.names),'Complete node fields')
            for key in PREDICTOR_RECORD_DTYPE.names:
                if PREDICTOR_RECORD_DTYPE[key].kind=='f':
                    require(type(node[key]) is str and float.fromhex(node[key]).hex()==node[key],'Canonical node float')
                    v=float.fromhex(node[key]);permitted_infinity=key=='num_threshold' and node['is_leaf']==0 and v==float('inf')
                    require(np.isfinite(v) or permitted_infinity,'Finite node float or explicit active positive infinity threshold')
                else:require(type(node[key]) is int and node[key]>=0,'Native nonnegative structural integer')
            require(node['depth']==depth<=2 and node['is_leaf'] in (0,1) and node['missing_go_to_left'] in (0,1) and node['is_categorical']==0,'Node depth/flags')
            if not node['is_leaf']:
                require(node['feature_idx']<10 and node['bin_threshold']<255 and depth<2,'Numeric split bounds')
                stack.extend([(node['right'],depth+1),(node['left'],depth+1)])
        require(len(seen)==len(nodes),'No unreachable nodes')
    expected=_shape_certificate(tree)
    require(_fingerprint(tree['shape_certificate'])==_fingerprint(expected),'Exact recomputed shape certificate')
    return True

def manual_predict(tree,X,role=None):
    validate_tree(tree,role);X=np.asarray(X,float);require(X.ndim==2 and X.shape[1]==10 and np.isfinite(X).all(),'Finite manual query')
    baseline=_unpack(tree['baseline'])[0,0];out=np.zeros(len(X),float);out+=baseline
    for r in tree['trees']:
        for i,x in enumerate(X):
            j=0
            while not r['nodes'][j]['is_leaf']:
                node=r['nodes'][j];j=node['left'] if x[node['feature_idx']]<=float.fromhex(node['num_threshold']) else node['right']
            out[i]+=float.fromhex(r['nodes'][j]['value'])
    require(np.isfinite(out).all(),'Finite manual prediction');return out


def legacy_unconstrained_tree(tree):
    """Exact old261 structural projection; no reference loading or fitting."""
    validate_tree(tree,'unconstrained_control')
    out={k:deepcopy(tree[k]) for k in legacy.TREE_KEYS}
    out['schema']='monthly_weak_tree261_serialized_tree_v1'
    legacy.validate_tree(out)
    return out

def _fit_tree(X,y,weights,role):
    expected=parameters(role);estimator=HistGradientBoostingRegressor(**expected)
    _parameters_check(estimator.get_params(deep=False),role)
    return estimator.fit(X,y,sample_weight=weights)

def _runtime():
    info=threadpool_info()
    require(all(type(x.get('num_threads')) is int and x['num_threads']==1 for x in info),'Effective native thread limit1')
    return deepcopy(info)

def _fingerprint(value):
    h=hashlib.sha256()
    def walk(v):
        h.update(type(v).__name__.encode()+b':')
        if isinstance(v,np.ndarray):h.update(v.dtype.str.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for key in sorted(v):walk(key);walk(v[key])
        elif isinstance(v,(list,tuple)):
            for item in v:walk(item)
        else:h.update(repr(v).encode())
    walk(value);return h.hexdigest()

def _array_snapshot(arrays,n):return _fingerprint({k:np.asarray(v)[:n] for k,v in arrays.items()})

def _gate(callback,*args):
    if callback is None:return dict(passed=True,synthetic_no_callback=True)
    require(callable(callback),'Callable gate');before=_fingerprint(args);receipt=callback(*args)
    require(before==_fingerprint(args),'Gate mutated supplied view')
    require(type(receipt) is dict and receipt.get('passed') is True,'Independent gate failed')
    return deepcopy(receipt)

def _model_gate(callback,bundle,train,metadata,arrays,n,source_snapshot,snapshot):
    before=_fingerprint(bundle);view=deepcopy(bundle)
    receipt=_gate(callback,view)
    require(before==_fingerprint(bundle),'Gate mutated producer bundle')
    require(snapshot==_array_snapshot(arrays,n),'Original arrays changed by model gate')
    args,ids,_,_=model_inputs(train,metadata,arrays,n)
    require(source_snapshot==_source_snapshot(args,ids,metadata,n),'Consumed original source changed by model gate')
    return receipt

def fit_models(train,combined_meta,arrays,n=None,*,calibration_validator=None,control_validator=None):
    n,w,qd=_axes(arrays,combined_meta['monthly'],n);snap=_array_snapshot(arrays,n)
    args,ids,_,_=model_inputs(train,combined_meta,arrays,n)
    dates,ends,symbols,X,targets,good,event_dates,event_symbols,Q,event_good=args
    source_snapshot=_source_snapshot(args,ids,combined_meta,n)
    before={k:_digest(v) for k,v in targets.items()};good_before=_digest(good)
    old,_,use,_=base._old_ranks(arrays,n,w);raw=np.asarray(arrays['raw_source_log'])[:n].reshape(-1).copy()
    available=np.asarray(arrays['forecast_available'])[:n].reshape(-1).copy()
    formal=np.repeat(qd!=np.datetime64(INITIAL_PARTIAL),w);u=use.reshape(-1)&formal
    years=sorted({int(str(d)[:4]) for d in qd if str(d)!=INITIAL_PARTIAL})
    usable=good&np.isfinite(X).all(axis=1)
    for y in targets.values():usable&=np.isfinite(y)
    months=dates.astype('datetime64[M]')
    for month in np.unique(months):
        selected=months==month
        if np.count_nonzero(usable&selected)<2:usable[selected]=False
    expected=available.copy();expected[formal]=False;annual=[];contexts=[]
    counts=dict.fromkeys(MODEL_COUNT_FIELDS,0);counts.update(events=n,training_rows=len(ids),valid_training_rows=int(usable.sum()),ordinary_years=len(years))
    with threadpool_limits(limits=1):
        for year in years:
            cutoff=np.datetime64('%04d-12-31'%(year-1));lower=model.numeric._years_before(cutoff)
            mask=usable&(dates>lower)&(ends<=cutoff);ti=np.flatnonzero(mask);nm=model.numeric._month_count(dates[mask])
            inyear=np.asarray([int(str(d)[:4])==year for d in event_dates])&formal
            if nm>=18:expected[inyear&event_good]=True
            qi=np.flatnonzero(inyear&u)
            status='cold_start' if nm<18 else 'no_prediction_rows' if not len(qi) else 'fit'
            counts['cold_years' if nm<18 else 'supported_years']+=1;counts['no_prediction_years']+=int(status=='no_prediction_rows')
            r=dict(year=year,cutoff=str(cutoff),lower_bound=str(lower),training_row_ids=ids[ti].tolist(),query_row_ids=qi.tolist(),training_months=nm,status=status,weights=[],mean=None,scale=None,beta=None,intercept=None,gram=None,rhs=None,raw_predictions=[])
            if status=='fit':
                model._ranks('Mature training ranks',X[ti]);model._ranks('Ordinary query ranks',Q[qi]);require(len(set(zip(months[ti].astype(str),symbols[ti])))==len(ti),'Unique security month')
                weights=model.numeric.month_balanced_weights(dates[ti],cutoff,12.)
                pooled,gram,rhs=fit_ridge(X[ti],targets['log_excess'][ti],weights)
                predictions=model.numeric._predict(pooled,Q[qi]);np.testing.assert_array_equal(predictions,raw[qi])
                raw[qi]=predictions
                r.update(weights=weights.tolist(),mean=pooled[0].tolist(),scale=pooled[1].tolist(),beta=pooled[2].tolist(),intercept=pooled[3],gram=gram.tolist(),rhs=rhs.tolist(),raw_predictions=predictions.tolist())
                contexts.append((r,ti,qi,weights));counts['matched_ridge_fits']+=1;counts['prediction_rows']+=len(qi)
            annual.append(r)
    np.testing.assert_array_equal(expected,available)
    rank=base._rank(raw.reshape(n,w),use,old);np.testing.assert_array_equal(rank,old)
    parity=dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False)
    control=dict(raw=raw,available=available,rank=rank,audit=dict(schema=_schema('ridge_audit'),annual=annual))
    lp=dict(schema=_schema('labels_preserved'),rows=len(ids),valid_rows=int(good.sum()),target_keys=list(TARGETS),target_sha256=before,common_good_sha256=good_before,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)
    counts['actual_model_fits']=counts['matched_ridge_fits']
    audit=dict(schema=_schema('model_audit'),counts=counts,original_cache_parity=parity,original_row_order_preserved=True,original_common_good_preserved=True,all_targets_preserved=True,local_models_reused=True,other_targets_not_fitted=True,weak_only=True,computes_returns=False,execution_runtime=[])
    bundle=dict(schema=_schema('models'),n=n,width=w,training_row_ids=ids,calibration_ridge=control,unconstrained_control=None,monotone_candidate=None,label_values=targets['log_excess'].copy(),label_preservation=lp,calibration_gate=None,control_gate=None,audit=audit)
    bundle['calibration_gate']=_model_gate(calibration_validator,bundle,train,combined_meta,arrays,n,source_snapshot,snap)
    def build(role):
        forecast=dict(raw=raw.copy(),available=available.copy(),rank=old.copy(),audit=dict(schema=_schema('tree_audit'),annual=[]))
        byyear={r['year']:(ti,qi,W) for r,ti,qi,W in contexts}
        prefix='unconstrained' if role=='unconstrained_control' else 'monotone'
        with threadpool_limits(limits=1):
            for r in annual:
                tr=dict(year=r['year'],cutoff=r['cutoff'],training_row_ids=r['training_row_ids'].copy(),query_row_ids=r['query_row_ids'].copy(),status=r['status'],tree=None,raw_predictions=[],manual_predictions=[])
                if r['status']=='fit':
                    ti,qi,W=byyear[r['year']];fit_runtime=_runtime();estimator=_fit_tree(X[ti],targets['log_excess'][ti],W,role)
                    predict_runtime=_runtime();predictions=np.asarray(estimator.predict(Q[qi]),float)
                    require(predictions.shape==(len(qi),) and np.isfinite(predictions).all(),'Complete tree U predictions')
                    serialized=serialize_tree(estimator,role);manual=manual_predict(serialized,Q[qi],role)
                    require(predictions.tobytes()==manual.tobytes(),'Exact manual tree predictions')
                    audit['execution_runtime'].append(dict(role=role,year=r['year'],fit_threadpools=fit_runtime,predict_threadpools=predict_runtime))
                    forecast['raw'][qi]=predictions;tr.update(tree=serialized,raw_predictions=predictions.tolist(),manual_predictions=manual.tolist())
                    counts[prefix+'_tree_fits']+=1;counts[prefix+'_boosting_iterations']+=serialized['n_iter'];counts[prefix+'_trees']+=len(serialized['trees'])
                    counts[prefix+'_tree_nodes']+=sum(len(x['nodes']) for x in serialized['trees']);counts[prefix+'_manual_tree_traversals']+=len(qi)*len(serialized['trees'])
                forecast['audit']['annual'].append(tr)
        for i in range(n):
            if str(qd[i])!=INITIAL_PARTIAL and use[i].any():forecast['rank'][i]=base._rank(forecast['raw'].reshape(n,w)[i:i+1],use[i:i+1],old[i:i+1])[0]
        counts['actual_model_fits']=counts['matched_ridge_fits']+counts['unconstrained_tree_fits']+counts['monotone_tree_fits']
        return forecast
    bundle['unconstrained_control']=build('unconstrained_control')
    bundle['control_gate']=_model_gate(control_validator,bundle,train,combined_meta,arrays,n,source_snapshot,snap)
    bundle['monotone_candidate']=build('monotone_candidate')
    require(counts['matched_ridge_fits']==counts['unconstrained_tree_fits']==counts['monotone_tree_fits'],'Paired actual annual fits')
    require(before=={k:_digest(v) for k,v in targets.items()} and good_before==_digest(good),'Training unchanged')
    after_args,after_ids,_,_=model_inputs(train,combined_meta,arrays,n)
    require(source_snapshot==_source_snapshot(after_args,after_ids,combined_meta,n) and snap==_array_snapshot(arrays,n),'Original source unchanged')
    return bundle


def _forecast_rank(arrays,meta,bundle,role,n):
    w=bundle['width'];size=bundle['n'];old,_,use,_=base._old_ranks(arrays,n,w);f=bundle[role]
    require(type(f) is dict and set(f)==set(FORECAST_KEYS),'Complete forecast role:'+role)
    raw=np.asarray(f['raw']);av=np.asarray(f['available']);rank=np.asarray(f['rank'])
    require(raw.shape==(size*w,) and raw.dtype==np.dtype(float) and av.shape==raw.shape and av.dtype==np.dtype(bool) and rank.shape==(size,w) and rank.dtype==np.dtype(float),'Forecast axes:'+role)
    require(av[:n*w].tobytes()==np.asarray(arrays['forecast_available'])[:n].reshape(-1).tobytes(),'Original availability:'+role)
    values=raw[:n*w].reshape(n,w);original=np.asarray(arrays['raw_source_log'])[:n]
    expected=old.copy();outside=~use.copy()
    for i,d in enumerate(meta['dates'][:n]):
        if d==INITIAL_PARTIAL:outside[i]=True;continue
        require(np.isfinite(values[i,use[i]]).all(),'Full original U finite forecasts')
        if use[i].any():expected[i]=base._rank(values[i:i+1],use[i:i+1],old[i:i+1])[0]
    require(values[outside].tobytes()==original[outside].tobytes(),'Unconsumed forecast exact source copy')
    require(rank[:n].tobytes()==expected.tobytes(),'Exact forecast ranks:'+role)
    if role=='calibration_ridge':
        require(values.tobytes()==original.tobytes() and expected.tobytes()==old.tobytes(),'Original calibrated forecasts/ranks')
    return expected

def _bundle_scope(arrays,meta,bundle,n):
    n,w,dates=_axes(arrays,meta,n)
    require(type(bundle) is dict and set(bundle)==set(BUNDLE_KEYS) and bundle['schema']==_schema('models') and type(bundle['n']) is int and n<=bundle['n'] and bundle['width']==w,'Model prefix layout')
    return n,w,dates

def rank_and_delta(arrays,meta,bundle,n=None):
    n,w,dates=_bundle_scope(arrays,meta,bundle,n)
    old=_forecast_rank(arrays,meta,bundle,'calibration_ridge',n);_,_,use,_=base._old_ranks(arrays,n,w)
    weak=bridge._bool(arrays['state_weak_narrow'],(np.shape(arrays['baseline'])[0],),n,'Weak state')
    active=use&weak[:,None]&(dates!=np.datetime64(INITIAL_PARTIAL))[:,None]
    ranks={'old_pool':old};deltas={}
    for short,role in zip(('control','candidate'),TREE_ROLES):
        rank=_forecast_rank(arrays,meta,bundle,role,n);delta=np.zeros((n,w));delta[active]=.0375*(rank[active]-old[active])
        require(np.isfinite(delta).all(),'Finite exact weak delta');ranks[short]=rank;deltas[short]=delta
    audit=dict(schema=_schema('rank_delta'),events=n,ranking_cells=int(use.sum()),active_indices=[np.flatnonzero(row).tolist() for row in active],control_changed_cells=int(np.count_nonzero(deltas['control'])),candidate_changed_cells=int(np.count_nonzero(deltas['candidate'])),original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,local_basis_preserved=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return ranks,deltas,audit

def original_decisions(study,arrays,meta,n=None):
    return legacy.original_decisions(study,arrays,meta,n)

def _one_decision(study,arrays,meta,bundle,role,n=None):
    n,w,dates=_bundle_scope(arrays,meta,bundle,n);before=_array_snapshot(arrays,n);b=bridge._cache(study,arrays,meta,n)
    old=_forecast_rank(arrays,meta,bundle,'calibration_ridge',n);rank=_forecast_rank(arrays,meta,bundle,role,n)
    use=b['good']&np.asarray(arrays['forecast_available'])[:n]
    active=use&b['weak'][:,None]&(dates!=np.datetime64(INITIAL_PARTIAL))[:,None]
    delta=np.zeros((n,w));delta[active]=.0375*(rank[active]-old[active]);pre=b['pre'].copy();changed=delta!=0.;pre[changed]+=delta[changed]
    orders,scores,_=_episode(study,b,pre)
    require(before==_array_snapshot(arrays,n),'Source changed by single-role selector')
    return orders,scores,pre

def calibration_decisions(study,arrays,meta,bundle,n=None):
    return _one_decision(study,arrays,meta,bundle,'calibration_ridge',n)

def control_decisions(study,arrays,meta,bundle,n=None):
    require(bundle.get('unconstrained_control') is not None,'Full unconstrained control supplied')
    return _one_decision(study,arrays,meta,bundle,'unconstrained_control',n)

def decisions(study,arrays,meta,bundle,n=None,*,incumbent_validator=None,expected_incumbent_orders=None):
    n,w,_=_axes(arrays,meta,n);before=_array_snapshot(arrays,n);b=bridge._cache(study,arrays,meta,n)
    inc,score,choices0=_episode(study,b,b['pre'])
    if expected_incumbent_orders is not None:require(inc==expected_incumbent_orders,'Frozen incumbent orders')
    gate=_gate(incumbent_validator,deepcopy(inc),score.copy(),b['pre'].copy())
    require(before==_array_snapshot(arrays,n),'Original arrays changed by incumbent gate')
    _,deltas,ra=rank_and_delta(arrays,meta,bundle,n)
    orders={ARMS[0]:inc};scores={ARMS[0]:score};pres={ARMS[0]:b['pre'].copy()};choices={ARMS[0]:choices0}
    for arm,short in zip(ARMS[1:],('control','candidate')):
        pre=b['pre'].copy();delta=deltas[short];changed=delta!=0.;pre[changed]+=delta[changed]
        o,f,c=_episode(study,b,pre);orders[arm]=o;scores[arm]=f;pres[arm]=pre;choices[arm]=c
    use=b['good']&np.asarray(arrays['forecast_available'])[:n];events=[]
    for i,d in enumerate(meta['dates'][:n]):
        events.append(dict(query_index=i,date=d,initial_partial=d==INITIAL_PARTIAL,weak=bool(b['weak'][i]),ranking_indices=np.flatnonzero(use[i]).tolist(),active_indices=ra['active_indices'][i],control_changed_delta_indices=np.flatnonzero(deltas['control'][i]!=0.).tolist(),candidate_changed_delta_indices=np.flatnonzero(deltas['candidate'][i]!=0.).tolist(),control_delta=deltas['control'][i].tolist(),candidate_delta=deltas['candidate'][i].tolist(),incumbent_choices=choices[ARMS[0]][i],control_choices=choices[ARMS[1]][i],candidate_choices=choices[ARMS[2]][i],episode_budget=float(b['episode'][i])))
    c=dict.fromkeys(COUNT_FIELDS,0);partial=sum(r['initial_partial'] for r in events)
    c.update(events=n,symbols_per_event=w,initial_partial_events=partial,formal_events=n-partial,original_eligible_cells=int(b['good'].sum()),ranking_cells=int(use.sum()),weak_applicable_cells=sum(len(x) for x in ra['active_indices']),control_changed_delta_cells=int(np.count_nonzero(deltas['control'])),candidate_changed_delta_cells=int(np.count_nonzero(deltas['candidate'])),control_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices0,choices[ARMS[1]])),candidate_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices0,choices[ARMS[2]])),candidate_control_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices[ARMS[1]],choices[ARMS[2]])),incumbent_episode_calls=1,control_episode_calls=1,candidate_episode_calls=1)
    require(before==_array_snapshot(arrays,n),'Original arrays unchanged')
    audit=dict(schema=_schema('decisions'),counts=c,events=events,rank_delta=ra,incumbent_gate=gate,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)
    return orders,scores,pres,audit
