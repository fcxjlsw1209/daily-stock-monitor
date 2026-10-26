"""Caller-supplied matched pooled Ridge versus fixed GBDT; no loaders/replay."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'work/monthly_local_distance256_kernel.py'
BASE_SHA='97ada679902942f31e4881354fcf95b1fccd2119f77c09760904b8c328364e30'
RUNTIME_PATH=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
RUNTIME_SHA='ad796b9d48f1a6dd910dfcf8d66c650359d3c2dd8851a3cd042be935079cc59d'
CONTEXT_PATH=ROOT/'work/monthly_weak_tree261_context.json'
CONTEXT_SHA='e60db38b1d0fe73fa2453cf17691b007bbb89c03fef29b4711e4e6595dfdd37c'
CONTRACT_PATH=ROOT/'work/monthly_weak_tree261_IMPLEMENTATION_CONTRACT.json'
CONTRACT_SHA='c955aba02bb469c0b35e0499ae80f1b229d638306f72f7d1d0bfde36d9f5c4d6'
for _p,_h in ((BASE_PATH,BASE_SHA),(RUNTIME_PATH,RUNTIME_SHA),(CONTEXT_PATH,CONTEXT_SHA),(CONTRACT_PATH,CONTRACT_SHA)):
    if hashlib.sha256(_p.read_bytes()).hexdigest()!=_h:raise ValueError('Changed dependency: '+str(_p))
_spec=importlib.util.spec_from_file_location('tree261_frozen_source',BASE_PATH)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
model,bridge,require=base.model,base.bridge,base.require
require(sklearn.__version__=='1.6.1','Pinned sklearn1.6.1')
TARGETS=base.TARGETS
INITIAL_PARTIAL=base.INITIAL_PARTIAL
POLICY=dict(base.POLICY)
ARMS=('incumbent_local67','matched_pooled_ridge','weak_pooled_tree')
TREE_PARAMETERS=dict(loss='squared_error',quantile=None,learning_rate=.05,max_iter=100,max_leaf_nodes=31,max_depth=2,min_samples_leaf=50,l2_regularization=1.,max_features=1.,max_bins=255,categorical_features='from_dtype',monotonic_cst=None,interaction_cst=None,warm_start=False,early_stopping=False,scoring='loss',validation_fraction=.1,n_iter_no_change=10,tol=1e-7,verbose=0,random_state=1729)
BUNDLE_KEYS=('schema','n','width','training_row_ids','control','candidate','label_values','label_preservation','control_gate','audit')
FORECAST_KEYS=('raw','available','rank','audit')
LABEL_KEYS=base.LABEL_KEYS
MODEL_COUNT_FIELDS=('events','training_rows','valid_training_rows','ordinary_years','supported_years','cold_years','no_prediction_years','matched_ridge_fits','tree_fits','prediction_rows','boosting_iterations','trees','tree_nodes','manual_tree_traversals','original_other_target_fits','original_local_fits','actual_model_fits')
RIDGE_ANNUAL_KEYS=('year','cutoff','lower_bound','training_row_ids','query_row_ids','training_months','status','weights','mean','scale','beta','intercept','gram','rhs','raw_predictions')
TREE_ANNUAL_KEYS=('year','cutoff','training_row_ids','query_row_ids','status','tree','raw_predictions','manual_predictions')
MODEL_AUDIT_KEYS=('schema','counts','original_cache_parity','original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only','computes_returns','execution_runtime')
TREE_KEYS=('schema','parameters','n_features','n_iter','n_trees_per_iteration','baseline','node_dtype','trees','bin_thresholds','n_bins_non_missing','missing_bin','is_categorical','known_categories','preprocessor')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells','changed_delta_cells','matched_changed_holdings_events','tree_changed_holdings_events','tree_vs_matched_changed_holdings_events','incumbent_episode_calls','matched_episode_calls','tree_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','changed_delta_indices','delta','incumbent_choices','matched_choices','tree_choices','episode_budget')
AUDIT_KEYS=('schema','counts','events','rank_delta','incumbent_gate','original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact','computes_returns')
RANK_AUDIT_KEYS=('schema','events','ranking_cells','active_indices','changed_cells','original_ranking_population','no_group_ranking','no_own_history_filter','local_basis_preserved','weak_only','partial_exact','calm_slot_preserved','no_post_rerank')


def _schema(s):return 'monthly_weak_tree261_'+s+'_v1'
def dependency_files():return tuple(dict.fromkeys((BASE_PATH,RUNTIME_PATH,CONTEXT_PATH,CONTRACT_PATH)+tuple(base.dependency_files())))
def _digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def _gate(callback,*args):
    if callback is None:return dict(passed=True,synthetic_no_callback=True)
    require(callable(callback),'Callable gate');receipt=callback(*args)
    require(type(receipt) is dict and receipt.get('passed') is True,'Independent gate failed')
    return deepcopy(receipt)


def _axes(arrays,meta,n):
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0,'Original axes')
    total,w=shape;n=total if n is None else n
    require(type(n) is int and 0<=n<=total,'Query prefix')
    require(all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','ends','symbols')),'Monthly metadata headers')
    dates=base._days(meta['dates'][:n],'Consumed queries')
    require(n<2 or np.all(dates[1:]>dates[:-1]),'Ordered queries')
    return n,w,dates


def model_inputs(train,combined_meta,arrays,n=None):
    """Crop whole mature months before invoking the frozen input validator."""
    n,w,qd=_axes(arrays,combined_meta['monthly'],n)
    ordinary=qd[qd!=np.datetime64(INITIAL_PARTIAL)]
    if not len(ordinary):
        empty=np.empty(0,dtype='datetime64[D]');z=np.empty((0,10),float)
        return (empty,empty.copy(),np.empty(0,dtype=str),z,{k:np.empty(0) for k in TARGETS},np.empty(0,bool),np.repeat(qd,w),np.asarray([s for row in combined_meta['monthly']['symbols'][:n] for s in row],dtype=str),np.zeros((n*w,10)),np.zeros(n*w,bool)),np.empty(0,int),n,w
    tm=combined_meta['training'];td=base._days(tm['dates'],'Training signal headers');te=base._days(tm['ends'],'Training maturity headers')
    require(len(td)==len(te)==len(tm['symbols']),'Training metadata headers')
    offsets=np.asarray(train['offsets']);require(offsets.shape==(len(td)+1,) and offsets.dtype.kind in 'iu' and offsets[0]==0 and np.all(np.diff(offsets)>0),'Offset headers')
    cutoff=np.datetime64('%04d-12-31'%(int(str(ordinary[-1])[:4])-1))
    mask=te<=cutoff;require(not np.any(mask[1:]&~mask[:-1]),'Mature whole-month prefix')
    count=int(mask.sum());stop=int(offsets[count]);rows=int(offsets[-1])
    sizes={'dates':(rows,),'ends':(rows,),'train_plain':(rows,10),'common_good':(rows,),'event_dates':(np.shape(arrays['baseline'])[0]*w,),'event_plain':(np.shape(arrays['baseline'])[0]*w,10)}
    sizes.update({'target_'+k:(rows,) for k in TARGETS})
    for k,shape in sizes.items():require(np.shape(train[k])==shape,'Source static shape '+k)
    t={k:np.asarray(train[k])[:(n*w if k.startswith('event_') else stop)].copy() for k in sizes}
    t['offsets']=offsets[:count+1].copy()
    m={key:deepcopy(combined_meta['monthly'][key][:n]) for key in ('dates','ends','symbols')};m['etfs']=deepcopy(combined_meta['monthly']['etfs'])
    mt={key:deepcopy(tm[key][:count]) for key in ('dates','ends','symbols')}
    a={k:np.asarray(v)[:n].copy() for k,v in arrays.items()}
    args,ids,_,_=base._inputs(t,dict(monthly=m,training=mt),a,n)
    return args,ids,n,w


def fit_ridge(X,y,weights):
    """Original operation order, with a hard solve failure (no lstsq rescue)."""
    for a in (X,y,weights):require(np.isfinite(a).all(),'Finite Ridge source')
    require(np.all(weights>0),'Positive original row weights')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        mean=np.average(X,axis=0,weights=weights);centered=X-mean
        scale=np.sqrt(np.average(centered*centered,axis=0,weights=weights));scale=np.where(scale>1e-12,scale,1.)
        z=centered/scale;intercept=float(np.average(y,weights=weights))
        gram=np.einsum('ni,n,nj->ij',z,weights,z,optimize=False);gram+=10.*np.eye(10)
        rhs=np.einsum('ni,n,n->i',z,weights,y-intercept,optimize=False)
    require(np.isfinite(gram).all() and np.isfinite(rhs).all(),'Finite Ridge equations')
    beta=np.linalg.solve(gram,rhs);require(np.isfinite(beta).all(),'Finite Ridge coefficients')
    return (mean,scale,beta,intercept),gram,rhs


def _pack(a):
    a=np.asarray(a);require(a.dtype.kind in 'fiu','Supported numeric structural type')
    def conv(v):return float(v).hex() if a.dtype.kind=='f' else int(v)
    vals=np.asarray([conv(v) for v in a.reshape(-1)],object).reshape(a.shape).tolist()
    return dict(dtype=a.dtype.str,shape=list(a.shape),values=vals)


def _unpack(p,kind=None):
    require(type(p) is dict and set(p)=={'dtype','shape','values'},'Typed array record')
    dtype=np.dtype(p['dtype']);require(dtype.kind in 'fiu' and (kind is None or dtype.kind==kind),'Typed numeric kind')
    shape=p['shape'];require(isinstance(shape,list) and all(type(v) is int and v>=0 for v in shape),'Typed shape')
    obj=np.asarray(p['values'],object);require(obj.size==int(np.prod(shape,dtype=int)),'Typed array size')
    if obj.size:require(obj.shape==tuple(shape),'Typed array shape')
    if dtype.kind=='f':
        require(all(type(v) is str and float.fromhex(v).hex()==v for v in obj.reshape(-1)),'Canonical hexadecimal floats')
        out=np.asarray([float.fromhex(v) for v in obj.reshape(-1)],dtype=dtype).reshape(shape)
    else:
        require(all(type(v) is int for v in obj.reshape(-1)),'Native structural integers');out=np.asarray(obj.tolist(),dtype=dtype).reshape(shape)
    require(_pack(out)==p,'Typed structural roundtrip exact');return out


def serialize_tree(estimator):
    require(estimator.get_params(deep=False)==TREE_PARAMETERS,'All21 fixed parameters')
    require(estimator.is_categorical_ is None and estimator._preprocessor is None and estimator._bin_mapper.known_categories is None,'Numeric-only tree features')
    records=[]
    for i,group in enumerate(estimator._predictors):
        require(len(group)==1,'One regression tree per iteration')
        p=group[0];require(p.nodes.dtype==PREDICTOR_RECORD_DTYPE,'Pinned node dtype')
        rows=[]
        for node in p.nodes:
            rows.append({k:float(node[k]).hex() if p.nodes.dtype[k].kind=='f' else int(node[k]) for k in p.nodes.dtype.names})
        records.append(dict(iteration=i,class_index=0,nodes=rows,raw_left_cat_bitsets=_pack(p.raw_left_cat_bitsets),binned_left_cat_bitsets=_pack(p.binned_left_cat_bitsets)))
    out=dict(schema=_schema('serialized_tree'),parameters=dict(TREE_PARAMETERS),n_features=int(estimator.n_features_in_),n_iter=int(estimator.n_iter_),n_trees_per_iteration=int(estimator.n_trees_per_iteration_),baseline=_pack(estimator._baseline_prediction),node_dtype=[[k,PREDICTOR_RECORD_DTYPE[k].str] for k in PREDICTOR_RECORD_DTYPE.names],trees=records,bin_thresholds=[_pack(x) for x in estimator._bin_mapper.bin_thresholds_],n_bins_non_missing=[int(x) for x in estimator._bin_mapper.n_bins_non_missing_],missing_bin=int(estimator._bin_mapper.missing_values_bin_idx_),is_categorical=None,known_categories=None,preprocessor=None)
    validate_tree(out);return out


def validate_tree(tree):
    require(type(tree) is dict and set(tree)==set(TREE_KEYS) and tree['schema']==_schema('serialized_tree'),'Serialized model schema')
    require(tree['parameters']==TREE_PARAMETERS and tree['n_features']==10 and tree['n_iter']==100 and tree['n_trees_per_iteration']==1,'Fixed fitted estimator shape')
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
    return True


def manual_predict(tree,X):
    validate_tree(tree);X=np.asarray(X,float);require(X.ndim==2 and X.shape[1]==10 and np.isfinite(X).all(),'Finite manual query')
    baseline=_unpack(tree['baseline'])[0,0];out=np.zeros(len(X),float);out+=baseline
    for r in tree['trees']:
        for i,x in enumerate(X):
            j=0
            while not r['nodes'][j]['is_leaf']:
                node=r['nodes'][j];j=node['left'] if x[node['feature_idx']]<=float.fromhex(node['num_threshold']) else node['right']
            out[i]+=float.fromhex(r['nodes'][j]['value'])
    require(np.isfinite(out).all(),'Finite manual prediction');return out


def _fit_tree(X,y,weights):
    estimator=HistGradientBoostingRegressor(**TREE_PARAMETERS)
    require(estimator.get_params(deep=False)==TREE_PARAMETERS,'Complete constructor fixed')
    return estimator.fit(X,y,sample_weight=weights)


def _source_snapshot(args,ids,meta,n):
    def bits(a):
        a=np.asarray(a);return (a.dtype.str,a.shape,a.tobytes())
    entries=[{key:bits(value) for key,value in x.items()} if isinstance(x,dict) else bits(x) for x in args]
    monthly={key:deepcopy(meta['monthly'][key][:n]) for key in ('dates','ends','symbols')}
    monthly['etfs']=deepcopy(meta['monthly']['etfs'])
    return entries,bits(ids),monthly


def _runtime():
    info=threadpool_info()
    require(all(type(x.get('num_threads')) is int and x['num_threads']==1 for x in info),'Effective native thread limit1')
    return deepcopy(info)


def fit_models(train,combined_meta,arrays,n=None,*,control_validator=None):
    snap=bridge._snapshot(arrays);n,w,qd=_axes(arrays,combined_meta['monthly'],n)
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
    bundle=dict(schema=_schema('models'),n=n,width=w,training_row_ids=ids,control=control,candidate=None,label_values=targets['log_excess'].copy(),label_preservation=lp,control_gate=None,audit=audit)
    gate=_gate(control_validator,deepcopy(bundle))
    require(snap==bridge._snapshot(arrays),'Original arrays changed by control gate')
    after_args,after_ids,_,_=model_inputs(train,combined_meta,arrays,n)
    require(source_snapshot==_source_snapshot(after_args,after_ids,combined_meta,n),'Consumed original source changed by control gate')
    candidate_raw=raw.copy();tree_records=[];context_by_year={r['year']:(ti,qi,W) for r,ti,qi,W in contexts}
    with threadpool_limits(limits=1):
        for r in annual:
            tr=dict(year=r['year'],cutoff=r['cutoff'],training_row_ids=r['training_row_ids'].copy(),query_row_ids=r['query_row_ids'].copy(),status=r['status'],tree=None,raw_predictions=[],manual_predictions=[])
            if r['status']=='fit':
                ti,qi,W=context_by_year[r['year']];fit_runtime=_runtime();estimator=_fit_tree(X[ti],targets['log_excess'][ti],W)
                predict_runtime=_runtime();predictions=np.asarray(estimator.predict(Q[qi]),float);require(predictions.shape==(len(qi),) and np.isfinite(predictions).all(),'Complete tree U predictions')
                tree=serialize_tree(estimator);manual=manual_predict(tree,Q[qi]);np.testing.assert_array_equal(predictions,manual)
                audit['execution_runtime'].append(dict(year=r['year'],fit_threadpools=fit_runtime,predict_threadpools=predict_runtime))
                candidate_raw[qi]=predictions;tr.update(tree=tree,raw_predictions=predictions.tolist(),manual_predictions=manual.tolist())
                counts['tree_fits']+=1;counts['boosting_iterations']+=tree['n_iter'];counts['trees']+=len(tree['trees']);counts['tree_nodes']+=sum(len(x['nodes']) for x in tree['trees']);counts['manual_tree_traversals']+=len(qi)*len(tree['trees'])
            tree_records.append(tr)
    candidate_rank=old.copy()
    for i in range(n):
        if str(qd[i])!=INITIAL_PARTIAL and use[i].any():candidate_rank[i]=base._rank(candidate_raw.reshape(n,w)[i:i+1],use[i:i+1],old[i:i+1])[0]
    bundle['candidate']=dict(raw=candidate_raw,available=available.copy(),rank=candidate_rank,audit=dict(schema=_schema('tree_audit'),annual=tree_records))
    bundle['control_gate']=gate;counts['actual_model_fits']+=counts['tree_fits']
    require(counts['matched_ridge_fits']==counts['tree_fits'],'Paired annual fitting')
    require(before=={k:_digest(v) for k,v in targets.items()} and good_before==_digest(good),'Training unchanged')
    require(snap==bridge._snapshot(arrays),'Original arrays unchanged');return bundle


def rank_and_delta(arrays,meta,bundle,n=None):
    n,w,dates=_axes(arrays,meta,n)
    require(type(bundle) is dict and set(bundle)==set(BUNDLE_KEYS) and bundle['schema']==_schema('models') and type(bundle['n']) is int and n<=bundle['n'] and bundle['width']==w,'Model prefix layout')
    old,_,use,_=base._old_ranks(arrays,n,w);tree_rank=old.copy()
    for role in ('control','candidate'):
        f=bundle[role];require(type(f) is dict and set(f)==set(FORECAST_KEYS),'Forecast role')
        raw=np.asarray(f['raw']);avail=np.asarray(f['available']);ranks=np.asarray(f['rank'])
        require(raw.shape==(bundle['n']*w,) and raw.dtype==np.dtype(float) and avail.shape==raw.shape and avail.dtype==np.dtype(bool) and ranks.shape==(bundle['n'],w) and ranks.dtype==np.dtype(float),'Forecast shape/dtype')
        np.testing.assert_array_equal(avail[:n*w],np.asarray(arrays['forecast_available'])[:n].reshape(-1))
        values=raw[:n*w].reshape(n,w);require(np.isfinite(values[use]).all(),'Complete original U raw')
        expected=old.copy()
        for i,d in enumerate(dates):
            if str(d)!=INITIAL_PARTIAL:expected[i]=base._rank(values[i:i+1],use[i:i+1],old[i:i+1])[0]
        np.testing.assert_array_equal(expected,ranks[:n])
        if role=='control':
            np.testing.assert_array_equal(values,np.asarray(arrays['raw_source_log'])[:n]);np.testing.assert_array_equal(expected,old)
        else:tree_rank=expected
    weak=bridge._bool(arrays['state_weak_narrow'],(np.shape(arrays['baseline'])[0],),n,'Weak state')
    active=use&weak[:,None]&(dates!=np.datetime64(INITIAL_PARTIAL))[:,None]
    delta=np.zeros((n,w));delta[active]=.0375*(tree_rank[active]-old[active]);require(np.isfinite(delta).all(),'Finite exact weak delta')
    audit=dict(schema=_schema('rank_delta'),events=n,ranking_cells=int(use.sum()),active_indices=[np.flatnonzero(row).tolist() for row in active],changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,local_basis_preserved=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return dict(old_pool=old,tree=tree_rank),delta,audit


def _episode(study,b,pre):
    source=dict(baseline=b['base'],core=b['core'],core_weight=b['cw'],eligible=b['good'],state_calm10=b['calm'],state_weak_narrow=b['weak'])
    choices,post=study.ep.select_boundary(pre.copy(),b['good'].copy(),deepcopy(b['meta']['symbols']),b['effects'].copy(),{k:v.copy() for k,v in source.items()},dict(POLICY))
    post=bridge._matrix(post,pre.shape,len(pre),'Final score');choices=[[int(j) for j in row] for row in choices]
    return study.rp.make_orders(choices,post,b['meta']),post,choices


def original_decisions(study,arrays,meta,n=None):
    n,_,_=_axes(arrays,meta,n);before=bridge._snapshot(arrays);b=bridge._cache(study,arrays,meta,n)
    orders,scores,_=_episode(study,b,b['pre']);require(before==bridge._snapshot(arrays),'Original cache unchanged')
    return orders,scores,b['pre'].copy()


def matched_decisions(study,arrays,meta,control_view,n=None):
    """Control-only calibration helper; one original selector, no tree access."""
    n,w,dates=_axes(arrays,meta,n);f=control_view['control'];size=control_view['n']
    require(type(size) is int and n<=size and control_view['width']==w,'Control-only prefix')
    raw=np.asarray(f['raw']);av=np.asarray(f['available']);rank=np.asarray(f['rank'])
    require(raw.shape==(size*w,) and raw.dtype==np.dtype(float) and av.shape==raw.shape and av.dtype==np.dtype(bool) and rank.shape==(size,w) and rank.dtype==np.dtype(float),'Control-only forecast layout')
    old,_,use,_=base._old_ranks(arrays,n,w)
    np.testing.assert_array_equal(raw[:n*w].reshape(n,w),np.asarray(arrays['raw_source_log'])[:n])
    np.testing.assert_array_equal(av[:n*w].reshape(n,w),np.asarray(arrays['forecast_available'])[:n])
    expected=old.copy()
    for i,d in enumerate(dates):
        if str(d)!=INITIAL_PARTIAL:expected[i]=base._rank(raw[:n*w].reshape(n,w)[i:i+1],use[i:i+1],old[i:i+1])[0]
    np.testing.assert_array_equal(expected,old);np.testing.assert_array_equal(rank[:n],old)
    return original_decisions(study,arrays,meta,n)


def decisions(study,arrays,meta,bundle,n=None,*,incumbent_validator=None,expected_incumbent_orders=None):
    n,w,_=_axes(arrays,meta,n);before=bridge._snapshot(arrays);b=bridge._cache(study,arrays,meta,n)
    inc,score,choices0=_episode(study,b,b['pre'])
    if expected_incumbent_orders is not None:require(inc==expected_incumbent_orders,'Frozen incumbent orders')
    gate=_gate(incumbent_validator,deepcopy(inc),score.copy(),b['pre'].copy())
    require(before==bridge._snapshot(arrays),'Original arrays changed by incumbent gate')
    _,delta,ra=rank_and_delta(arrays,meta,bundle,n)
    orders={ARMS[0]:inc};scores={ARMS[0]:score};pres={ARMS[0]:b['pre'].copy()};choices={ARMS[0]:choices0}
    for arm in ARMS[1:]:
        pre=b['pre'].copy()
        if arm==ARMS[2]:changed=delta!=0.;pre[changed]+=delta[changed]
        o,f,c=_episode(study,b,pre);orders[arm]=o;scores[arm]=f;pres[arm]=pre;choices[arm]=c
    require(orders[ARMS[1]]==inc and scores[ARMS[1]].tobytes()==score.tobytes() and pres[ARMS[1]].tobytes()==pres[ARMS[0]].tobytes(),'Matched Ridge exact strategy identity')
    use=b['good']&np.asarray(arrays['forecast_available'])[:n];events=[]
    for i,d in enumerate(meta['dates'][:n]):
        events.append(dict(query_index=i,date=d,initial_partial=d==INITIAL_PARTIAL,weak=bool(b['weak'][i]),ranking_indices=np.flatnonzero(use[i]).tolist(),active_indices=ra['active_indices'][i],changed_delta_indices=np.flatnonzero(delta[i]!=0.).tolist(),delta=delta[i].tolist(),incumbent_choices=choices[ARMS[0]][i],matched_choices=choices[ARMS[1]][i],tree_choices=choices[ARMS[2]][i],episode_budget=float(b['episode'][i])))
    c=dict.fromkeys(COUNT_FIELDS,0);partial=sum(r['initial_partial'] for r in events)
    c.update(events=n,symbols_per_event=w,initial_partial_events=partial,formal_events=n-partial,original_eligible_cells=int(b['good'].sum()),ranking_cells=int(use.sum()),weak_applicable_cells=sum(len(x) for x in ra['active_indices']),changed_delta_cells=int(np.count_nonzero(delta)),matched_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices0,choices[ARMS[1]])),tree_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices0,choices[ARMS[2]])),tree_vs_matched_changed_holdings_events=sum(set(a)!=set(z) for a,z in zip(choices[ARMS[1]],choices[ARMS[2]])),incumbent_episode_calls=1,matched_episode_calls=1,tree_episode_calls=1)
    require(before==bridge._snapshot(arrays),'Original arrays unchanged')
    audit=dict(schema=_schema('decisions'),counts=c,events=events,rank_delta=ra,incumbent_gate=gate,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)
    return orders,scores,pres,audit
