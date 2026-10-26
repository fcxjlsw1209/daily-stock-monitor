"""Independent261 cohort/Ridge/selection and shared-trainer tree reproduction.

No new producer or host is imported. Ridge has separate canonical and augmented
SVD paths. Tree refitting shares pinned sklearn (NOT an independent training
algorithm); serialized nodes and predictions are checked by separate traversal.
Only caller-supplied inputs are used; there is no actual-cache loader or runner.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work/monthly_local_distance256_independent.py'
PREVIOUS_SHA='2ac297ec53ce8f9fd4f90a1868e86c26f523442eeada5e4f5bb4d8d65c4b40a9'
SOURCE=ROOT/'work/monthly_leader_utility260_independent.py'
SOURCE_SHA='ec3b122a08f44e0ad3110940403ce483f5f734cd517fba2606edc9708e824a98'
RUNTIME=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
RUNTIME_SHA='ad796b9d48f1a6dd910dfcf8d66c650359d3c2dd8851a3cd042be935079cc59d'
def require(ok,message):
    if not ok:raise ValueError(message)
def _load(path,digest,name):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Frozen dependency '+path.name)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
previous=_load(PREVIOUS,PREVIOUS_SHA,'weak261_independent256')
source=_load(SOURCE,SOURCE_SHA,'weak261_independent260')
require(sklearn.__version__=='1.6.1','Pinned sklearn1.6.1')
require(hashlib.sha256(RUNTIME.read_bytes()).hexdigest()==RUNTIME_SHA,'Pinned numerical manifest')
exact_array,record_check,continuous=previous.exact_array,previous.record_check,previous.continuous
numeric,accounts,bridge=previous.numeric,previous.accounts,previous.bridge
original_base,accounting=previous.original_base,previous.accounting
PATHS,ACCOUNTING_SHA=previous.PATHS,previous.ACCOUNTING_SHA
TARGETS=previous.TARGETS
ARMS=('incumbent_local67','matched_pooled_ridge','weak_pooled_tree')
INITIAL='2016-08-17'
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
TREE_PARAMETERS=dict(loss='squared_error',quantile=None,learning_rate=.05,max_iter=100,max_leaf_nodes=31,max_depth=2,
 min_samples_leaf=50,l2_regularization=1.,max_features=1.,max_bins=255,categorical_features='from_dtype',
 monotonic_cst=None,interaction_cst=None,warm_start=False,early_stopping=False,scoring='loss',validation_fraction=.1,
 n_iter_no_change=10,tol=1e-7,verbose=0,random_state=1729)
BUNDLE_KEYS=('schema','n','width','training_row_ids','control','candidate','label_values','label_preservation','control_gate','audit')
MODEL_COUNT_FIELDS=('events','training_rows','valid_training_rows','ordinary_years','supported_years','cold_years','no_prediction_years',
 'matched_ridge_fits','tree_fits','prediction_rows','boosting_iterations','trees','tree_nodes','manual_tree_traversals',
 'original_other_target_fits','original_local_fits','actual_model_fits')
RIDGE_ANNUAL_KEYS=('year','cutoff','lower_bound','training_row_ids','query_row_ids','training_months','status','weights','mean','scale','beta','intercept','gram','rhs','raw_predictions')
TREE_ANNUAL_KEYS=('year','cutoff','training_row_ids','query_row_ids','status','tree','raw_predictions','manual_predictions')
TREE_KEYS=('schema','parameters','n_features','n_iter','n_trees_per_iteration','baseline','node_dtype','trees','bin_thresholds','n_bins_non_missing','missing_bin','is_categorical','known_categories','preprocessor')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells',
 'changed_delta_cells','matched_changed_holdings_events','tree_changed_holdings_events','tree_vs_matched_changed_holdings_events',
 'incumbent_episode_calls','matched_episode_calls','tree_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','changed_delta_indices','delta','incumbent_choices','matched_choices','tree_choices','episode_budget')
CONTROL_FLAGS=('original_training_identity_checked','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked',
 'all_four_target_common_cohort_checked','canonical_matched_ridge_checked','independent_augmented_svd','original_model_cache_checked',
 'original_query_rank_domain_checked','full_control_bundle_checked','no_future_training_values')
MODEL_FLAGS=CONTROL_FLAGS+('original_labels_unchanged','full_model_audit_checked','actual_fit_counts_checked','full_candidate_supplied',
 'all_tree_parameters_checked','independent_tree_input_pipeline','same_library_tree_refit_checked','serialized_nodes_bins_exact',
 'manual_tree_predictions_exact','local_and_other_models_not_refitted','nonweak_supported_years_included')
INCUMBENT_FLAGS=source.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
SELECTION_FLAGS=('independent_original_pre_checked','original_eligibility_unchanged','original_core_support_preserved',
 'original_query_rank_domain_checked','weak_pooled_three_quarter_checked','local_quarter_unchanged','partial_nonweak_exact',
 'three_arms_single_episode_checked','no_final_global_rerank','exact_scores_and_choices_checked','full_decision_audit_checked')
NUMERIC_SYSTEM='independent canonical Ridge plus augmented SVD; shared pinned sklearn refit and independent manual tree traversal'

def dependency_files():
    manifest=json.loads(RUNTIME.read_text())
    return tuple(sorted(set(previous.dependency_files())|set(source.dependency_files())|{PREVIOUS,SOURCE,RUNTIME,Path(__file__).resolve()}|{ROOT/p for p in manifest['input_sha256']}))

def _shape_slice(value,shape,ids,name,dtype=float):
    a=np.asarray(value);require(a.shape==shape,name+' shape')
    b=a[ids]
    if dtype is bool:require(b.dtype==np.dtype(bool),name+' Boolean')
    elif b.dtype!=np.dtype('float64'):b=source._floats(b,name)
    return np.asarray(b,dtype=dtype).copy()

def model_inputs(train,meta,arrays,n=None):
    require(type(meta) is dict and 'monthly' in meta and 'training' in meta,'Combined metadata')
    m,tm=meta['monthly'],meta['training'];n=len(m['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(m['dates']),'Prefix count')
    E=np.asarray(arrays['eligible']);require(E.ndim==2 and E.dtype==np.dtype(bool) and len(E)>=n and E.shape[1]>0,'Eligibility shape')
    w=E.shape[1];qd=source._days(m['dates'][:n],'Query days');require(np.all(qd[1:]>qd[:-1]),'Increasing query dates')
    qs=[]
    for row in m['symbols'][:n]:
        names=source._ids(row,w,'Query names');require(len(set(names))==w,'Unique query inventory');qs.extend(names)
    formal=qd!=np.datetime64(INITIAL);flat_formal=np.repeat(formal,w)
    # Partial-only bypass does not even retrieve training or query numeric bodies.
    empty=dict(n=n,width=w,training_row_ids=np.empty(0,np.int64),dates=np.empty(0,'datetime64[D]'),ends=np.empty(0,'datetime64[D]'),
      symbols=np.empty(0,str),X=np.empty((0,10)),good=np.empty(0,bool),targets={k:np.empty(0) for k in TARGETS},
      query_dates=np.repeat(qd,w),query_symbols=np.asarray(qs,str),query_X=np.zeros((n*w,10)),query_good=np.zeros(n*w,bool),formal=formal)
    if not formal.any():return empty
    td=source._days(train['dates'],'Training dates');te=source._days(train['ends'],'Training ends')
    require(td.shape==te.shape and np.all(te>td) and np.all(te.astype('datetime64[M]')==td.astype('datetime64[M]')+1),'Monthly label clocks')
    offsets=np.asarray(train['offsets']);require(offsets.ndim==1 and offsets.dtype.kind in 'iu' and len(offsets)==len(tm['dates'])+1,'Training offsets')
    require(len(offsets)>0 and offsets[0]==0 and offsets[-1]==len(td) and np.all(offsets[1:]>offsets[:-1]),'Complete month spans')
    require(len(tm['ends'])==len(tm['symbols'])==len(tm['dates']),'Training metadata axes')
    last_year=int(qd[formal][-1].astype('datetime64[Y]').astype(int))+1970;cutoff=np.datetime64('%04d-12-31'%(last_year-1))
    ids=np.flatnonzero(te<=cutoff);names=[]
    for k,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        a,b=int(a),int(b)
        selected=te[a:b]<=cutoff
        if not selected.any():continue
        require(selected.all(),'Whole mature month')
        ss=source._ids(tm['symbols'][k],b-a,'Training names');require(len(set(ss))==len(ss),'Unique month securities')
        require(np.all(td[a:b]==np.datetime64(tm['dates'][k],'D')) and np.all(te[a:b]==np.datetime64(tm['ends'][k],'D')),'Training identity')
        names.extend(ss)
    X=_shape_slice(train['train_plain'],(len(td),10),ids,'Train plain10')
    g=_shape_slice(train['common_good'],(len(td),),ids,'Common good',bool)
    targets={k:_shape_slice(train['target_'+k],(len(td),),ids,'Target '+k) for k in TARGETS}
    qe=source._days(np.asarray(train['event_dates'])[:n*w],'Cached query dates');require(np.array_equal(qe,np.repeat(qd,w)),'Query date identity')
    q0=np.asarray(train['event_plain']);require(q0.ndim==2 and q0.shape[0]>=n*w and q0.shape[1]==10,'Query feature shape')
    Q=np.zeros((n*w,10));Q[flat_formal]=source._floats(q0[:n*w][flat_formal],'Ordinary query plain10')
    good=E[:n].reshape(-1)&np.isfinite(Q).all(axis=1)&flat_formal
    return dict(n=n,width=w,training_row_ids=ids.astype(np.int64),dates=td[ids],ends=te[ids],symbols=np.asarray(names,str),X=X,good=g,
      targets=targets,query_dates=qe,query_symbols=np.asarray(qs,str),query_X=Q,query_good=good,formal=formal)

def usable_rows(inp):
    use=inp['good']&np.isfinite(inp['X']).all(axis=1)
    for y in inp['targets'].values():use &= np.isfinite(y)
    months=inp['dates'].astype('datetime64[M]')
    for month in np.unique(months):
        ix=months==month
        if int(np.count_nonzero(use&ix))<2:use[ix]=False
    return use

def month_weights(dates,cutoff):
    unique,inverse,counts=np.unique(dates.astype('datetime64[M]'),return_inverse=True,return_counts=True)
    w=1./counts[inverse].astype(float);age=cutoff.astype('datetime64[M]').astype(int)-unique.astype(int)
    w*=np.exp2(-age[inverse]/12.);total=w.sum();require(np.isfinite(total) and total>0,'Positive weight mass')
    return w*(len(w)/total)

def ridge_fit(X,y,Q,w,*,svd=True):
    require(np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(Q).all() and np.isfinite(w).all() and (w>0).all(),'Finite Ridge inputs')
    with np.errstate(over='raise',invalid='raise',divide='raise'),threadpool_limits(limits=1):
        mean=np.average(X,axis=0,weights=w);centered=X-mean
        scale=np.sqrt(np.average(centered*centered,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
        Z=centered/scale;a=float(np.average(y,weights=w))
        gram=np.einsum('ni,n,nj->ij',Z,w,Z,optimize=False);gram+=10.*np.eye(10)
        rhs=np.einsum('ni,n,n->i',Z,w,y-a,optimize=False);beta=np.linalg.solve(gram,rhs)
        pred=np.einsum('ni,i->n',(Q-mean)/scale,beta,optimize=False)+a
        if svd:
            independent=numeric.least_squares(Z,y-a,w,np.full(10,10.))
            continuous(independent,beta,'SVD/canonical beta')
            continuous(np.einsum('ni,i->n',(Q-mean)/scale,independent,optimize=False)+a,pred,'SVD/canonical predictions')
    require(all(np.isfinite(v).all() for v in (mean,scale,beta,gram,rhs,pred)) and np.isfinite(a),'Finite Ridge outputs')
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=a,gram=gram.tolist(),rhs=rhs.tolist(),raw_predictions=pred.tolist()),pred

def label_preservation(inp):
    out=previous.label_preservation(inp);out['schema']='monthly_weak_tree261_labels_preserved_v1';return out

def _pool(arrays,n):
    old,local,U=previous.original_ranks(arrays,n)
    return old,U

def ridge_oracle(inp,arrays,*,svd=True):
    n,w=inp['n'],inp['width'];old,U=_pool(arrays,n)
    raw=np.asarray(arrays['raw_source_log'])[:n].reshape(-1).copy();available=np.asarray(arrays['forecast_available'])[:n].reshape(-1).copy()
    require(raw.shape==(n*w,) and raw.dtype==np.dtype('float64') and available.dtype==np.dtype(bool),'Original forecast axes')
    # All outside-U raw/support and the initial partial are exact cache copies.
    formal=np.repeat(inp['formal'],w)
    use=usable_rows(inp);months=inp['dates'].astype('datetime64[M]');years=inp['query_dates'].astype('datetime64[Y]').astype(int)+1970
    annual=[];contexts=[]
    for year in np.unique(years[formal]):
        cutoff=np.datetime64('%04d-12-31'%(year-1));lower=np.datetime64('%04d-12-31'%(year-6))
        ids=np.flatnonzero(use&(inp['dates']>lower)&(inp['ends']<=cutoff));nm=len(np.unique(months[ids]))
        yearly=(years==year)&formal
        expected=inp['query_good'][yearly] if nm>=18 else np.zeros(int(yearly.sum()),bool)
        exact_array(available[yearly],expected,'Original cohort-derived availability')
        qids=np.flatnonzero(yearly&U.reshape(-1));status='cold_start' if nm<18 else ('fit' if len(qids) else 'no_prediction_rows')
        row=dict(year=int(year),cutoff=str(cutoff),lower_bound=str(lower),training_row_ids=inp['training_row_ids'][ids].tolist(),query_row_ids=qids.tolist(),training_months=nm,status=status,
          weights=[],mean=None,scale=None,beta=None,intercept=None,gram=None,rhs=None,raw_predictions=[])
        if status=='fit':
            require(inp['query_good'][qids].all(),'Original U complete finite query support')
            X,y,Q=inp['X'][ids],inp['targets']['log_excess'][ids],inp['query_X'][qids]
            require((np.abs(X)<=.5).all() and (np.abs(Q)<=.5).all(),'Bounded plain10 ranks')
            require(len({(str(months[i]),str(inp['symbols'][i])) for i in ids})==len(ids),'Unique security/month cohort')
            weights=month_weights(inp['dates'][ids],cutoff);mom,pred=ridge_fit(X,y,Q,weights,svd=svd);row.update(mom,weights=weights.tolist())
            raw[qids]=pred;available[qids]=True;contexts.append(dict(ids=ids,qids=qids,X=X,y=y,Q=Q,weights=weights,row=row))
        annual.append(row)
    ranked=old.copy()
    for i,mask in enumerate(U):
        if inp['formal'][i] and mask.any():
            require(available.reshape(n,w)[i,mask].all(),'Original U requires fitted support')
            ranked[i,mask]=previous._rank(raw.reshape(n,w)[i,mask])
    return dict(raw=raw,available=available,rank=ranked,audit=dict(schema='monthly_weak_tree261_ridge_audit_v1',annual=annual)),contexts

def _typed(a):
    a=np.asarray(a)
    values=np.vectorize(lambda x:float(x).hex(),otypes=[object])(a).tolist() if a.dtype.kind=='f' and a.size else a.tolist()
    return dict(dtype=a.dtype.str,shape=list(a.shape),values=values)

def serialize_tree(model):
    require(model._preprocessor is None and model.is_categorical_ is None,'Numerical tree only')
    trees=[]
    for iteration,predictors in enumerate(model._predictors):
        require(len(predictors)==1,'One squared-error tree per iteration')
        predictor=predictors[0];nodes=[]
        for node in predictor.nodes:
            nodes.append({name:float(node[name]).hex() if predictor.nodes.dtype[name].kind=='f' else int(node[name]) for name in predictor.nodes.dtype.names})
        trees.append(dict(iteration=iteration,class_index=0,nodes=nodes,raw_left_cat_bitsets=_typed(predictor.raw_left_cat_bitsets),binned_left_cat_bitsets=_typed(predictor.binned_left_cat_bitsets)))
    mapper=model._bin_mapper
    return dict(schema='monthly_weak_tree261_serialized_tree_v1',parameters=copy.deepcopy(model.get_params(deep=False)),n_features=int(model.n_features_in_),n_iter=int(model.n_iter_),
      n_trees_per_iteration=int(model.n_trees_per_iteration_),baseline=_typed(model._baseline_prediction),node_dtype=[[k,PREDICTOR_RECORD_DTYPE[k].str] for k in PREDICTOR_RECORD_DTYPE.names],
      trees=trees,bin_thresholds=[_typed(x) for x in mapper.bin_thresholds_],n_bins_non_missing=mapper.n_bins_non_missing_.tolist(),missing_bin=int(mapper.missing_values_bin_idx_),
      is_categorical=None,known_categories=None,preprocessor=None)

def _hex(value,label,*,finite=True):
    require(type(value) is str,label+' hexadecimal string')
    try:f=float.fromhex(value)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError(label+' hexadecimal value') from exc
    require(value==f.hex(),label+' canonical hexadecimal')
    if finite:require(np.isfinite(f),label+' finite')
    return f

def manual_predict(tree,Q):
    """No sklearn prediction or tree predictor call; exact serialized traversal."""
    require(type(tree) is dict and set(tree)==set(TREE_KEYS),'Serialized tree schema')
    record_check(tree['parameters'],TREE_PARAMETERS,'Tree parameters')
    require(tree['schema']=='monthly_weak_tree261_serialized_tree_v1' and tree['n_features']==10 and tree['n_iter']==100 and tree['n_trees_per_iteration']==1,'Tree dimensions')
    require(tree['is_categorical'] is tree['known_categories'] is tree['preprocessor'] is None,'No categorical preprocessing')
    record_check(tree['node_dtype'],[[k,PREDICTOR_RECORD_DTYPE[k].str] for k in PREDICTOR_RECORD_DTYPE.names],'Node types')
    require(type(tree['baseline']) is dict and tree['baseline'].get('dtype')=='<f8' and tree['baseline'].get('shape')==[1,1],'Baseline binary64 shape')
    baseline=_hex(tree['baseline']['values'][0][0],'Baseline');Q=np.asarray(Q)
    require(Q.ndim==2 and Q.shape[1]==10 and Q.dtype==np.dtype('float64') and np.isfinite(Q).all(),'Finite raw tree queries')
    require(len(tree['trees'])==100,'Exact 100 trees');out=np.zeros(len(Q));out+=baseline
    for iteration,tr in enumerate(tree['trees']):
        require(type(tr) is dict and set(tr)=={'iteration','class_index','nodes','raw_left_cat_bitsets','binned_left_cat_bitsets'} and type(tr['iteration']) is int and tr['iteration']==iteration and tr['class_index']==0,'Tree ordering')
        nodes=tr['nodes'];require(type(nodes) is list and 1<=len(nodes)<=7,'Depth2 tree size')
        reached=set();stack=[(0,0)]
        while stack:
            j,depth=stack.pop();require(type(j) is int and 0<=j<len(nodes) and j not in reached,'Acyclic rooted tree');reached.add(j);node=nodes[j]
            require(type(node) is dict and set(node)==set(PREDICTOR_RECORD_DTYPE.names),'Complete typed node')
            for key in PREDICTOR_RECORD_DTYPE.names:
                dtype=PREDICTOR_RECORD_DTYPE[key]
                if dtype.kind=='f':_hex(node[key],'Node/'+key,finite=key!='num_threshold')
                else:require(type(node[key]) is int and np.iinfo(dtype).min<=node[key]<=np.iinfo(dtype).max,'Native node integer/'+key)
            require(node['depth']==depth and depth<=2 and node['is_leaf'] in (0,1) and node['is_categorical']==0,'Tree depth/type')
            require(node['missing_go_to_left'] in (0,1),'Missing direction')
            if node['is_leaf']:_hex(node['num_threshold'],'Unused leaf threshold')
            if not node['is_leaf']:
                require(depth<2 and 0<=node['feature_idx']<10,'Internal split feature')
                threshold=_hex(node['num_threshold'],'Active numeric threshold',finite=False)
                require(np.isfinite(threshold) or threshold==np.inf,'Only positive-infinity missing-split sentinel')
                stack.extend([(node['right'],depth+1),(node['left'],depth+1)])
        require(len(reached)==len(nodes),'All serialized nodes reachable')
        for row,x in enumerate(Q):
            j=0
            while not nodes[j]['is_leaf']:
                node=nodes[j];j=node['left'] if x[node['feature_idx']]<=float.fromhex(node['num_threshold']) else node['right']
            out[row]+=float.fromhex(nodes[j]['value'])
    require(np.isfinite(out).all(),'Finite manual predictions');return out

def tree_oracle(inp,control,contexts):
    candidate=dict(raw=control['raw'].copy(),available=control['available'].copy(),rank=control['rank'].copy(),audit=dict(schema='monthly_weak_tree261_tree_audit_v1',annual=[]))
    byyear={ctx['row']['year']:ctx for ctx in contexts};runtime=[]
    for old in control['audit']['annual']:
        row={key:copy.deepcopy(old[key]) for key in ('year','cutoff','training_row_ids','query_row_ids','status')};row.update(tree=None,raw_predictions=[],manual_predictions=[])
        if old['status']=='fit':
            ctx=byyear[old['year']]
            with threadpool_limits(limits=1):
                model=HistGradientBoostingRegressor(**TREE_PARAMETERS)
                fit_pools=threadpool_info();require(all(v['num_threads']==1 for v in fit_pools),'Fit native threads1')
                model.fit(ctx['X'],ctx['y'],sample_weight=ctx['weights'])
                predict_pools=threadpool_info();require(all(v['num_threads']==1 for v in predict_pools),'Predict native threads1')
                raw=model.predict(ctx['Q'])
            runtime.append(dict(year=old['year'],fit_threadpools=fit_pools,predict_threadpools=predict_pools))
            serialized=serialize_tree(model);manual=manual_predict(serialized,ctx['Q']);exact_array(manual,raw,'Shared fit/manual raw prediction')
            candidate['raw'][ctx['qids']]=raw;row.update(tree=serialized,raw_predictions=raw.tolist(),manual_predictions=manual.tolist())
        candidate['audit']['annual'].append(row)
    return candidate,runtime

def forecast_ranks(forecast,inp,arrays):
    old,U=_pool(arrays,inp['n']);result=old.copy();n,w=old.shape;raw=forecast['raw'].reshape(n,w);avail=forecast['available'].reshape(n,w)
    for i,mask in enumerate(U):
        if inp['formal'][i] and mask.any():
            require(avail[i,mask].all() and np.isfinite(raw[i,mask]).all(),'Full original U forecast domain')
            result[i,mask]=previous._rank(raw[i,mask])
    forecast['rank']=result

def parity(inp,arrays,control):
    n,w=inp['n'],inp['width'];old,U=_pool(arrays,n);formal=np.repeat(inp['formal'],w)
    expected=np.asarray(arrays['raw_source_log'])[:n].reshape(-1)
    exact_array(control['raw'],expected,'Original canonical pooled forecasts')
    exact_array(control['available'],np.asarray(arrays['forecast_available'])[:n].reshape(-1),'Original forecast availability')
    exact_array(control['rank'],old,'Original matched pooled ranks')
    return dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False)

def model_counts(inp,control,candidate):
    c={key:0 for key in MODEL_COUNT_FIELDS};rows=control['audit']['annual']
    c.update(events=inp['n'],training_rows=len(inp['dates']),valid_training_rows=int(usable_rows(inp).sum()),ordinary_years=len(rows),
      supported_years=sum(r['status']!='cold_start' for r in rows),cold_years=sum(r['status']=='cold_start' for r in rows),
      no_prediction_years=sum(r['status']=='no_prediction_rows' for r in rows),matched_ridge_fits=sum(r['status']=='fit' for r in rows),
      prediction_rows=sum(len(r['query_row_ids']) for r in rows if r['status']=='fit'))
    if candidate is not None:
        for row in candidate['audit']['annual']:
            if row['status']!='fit':continue
            tr=row['tree'];c['tree_fits']+=1;c['boosting_iterations']+=tr['n_iter'];c['trees']+=len(tr['trees'])
            c['tree_nodes']+=sum(len(t['nodes']) for t in tr['trees']);c['manual_tree_traversals']+=len(tr['trees'])*len(row['query_row_ids'])
    c['actual_model_fits']=c['matched_ridge_fits']+c['tree_fits'];return c

def model_audit(inp,arrays,control,candidate,runtime=None):
    return dict(schema='monthly_weak_tree261_model_audit_v1',counts=model_counts(inp,control,candidate),original_cache_parity=parity(inp,arrays,control),
      original_row_order_preserved=True,original_common_good_preserved=True,all_targets_preserved=True,local_models_reused=True,
      other_targets_not_fitted=True,weak_only=True,computes_returns=False,execution_runtime=[] if runtime is None else runtime)

def model_oracle(train,meta,arrays,n=None,*,candidate=True):
    inp=model_inputs(train,meta,arrays,n);control,contexts=ridge_oracle(inp,arrays)
    # Original cache parity completes before the first tree fit.
    pa=parity(inp,arrays,control)
    tree,runtime=tree_oracle(inp,control,contexts) if candidate else (None,[])
    if tree is not None:forecast_ranks(tree,inp,arrays)
    bundle=dict(schema='monthly_weak_tree261_models_v1',n=inp['n'],width=inp['width'],training_row_ids=inp['training_row_ids'].copy(),
      control=control,candidate=tree,label_values=inp['targets']['log_excess'].copy(),label_preservation=label_preservation(inp),
      control_gate=pa if candidate else None,audit=model_audit(inp,arrays,control,tree,runtime))
    return bundle,inp,contexts

def _receipt(n,flags,**kw):
    out=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,
      independent_canonical_solves=0,independent_svd_solves=0,independent_tree_refits=0,manual_tree_traversals=0,
      original_episode_calls=0,original_other_target_fits=0,original_local_fits=0)
    out.update({k:True for k in flags});out.update(kw);return out

def _model_receipt(expected,control_only):
    n=expected['n'];c=expected['audit']['counts'];fits=c['matched_ridge_fits']
    return _receipt(n,CONTROL_FLAGS if control_only else MODEL_FLAGS,counts=copy.deepcopy(c),producer_equivalent_counts=copy.deepcopy(c),
      independent_canonical_solves=fits,independent_svd_solves=fits,independent_tree_refits=c['tree_fits'],manual_tree_traversals=c['manual_tree_traversals'],
      continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,full_bundle_supplied=True,full_control_supplied=True,
      full_candidate_supplied=not control_only,full_label_preservation_supplied=True,full_model_audit_supplied=True,
      candidate_models_computed=not control_only,shared_tree_trainer=True,independent_tree_training_algorithm=False,independent_numeric_system=NUMERIC_SYSTEM)

def _compare_bundle(actual,expected,label,gate=True):
    require(type(actual) is dict and set(actual)==set(BUNDLE_KEYS),label+' exact bundle keys')
    wanted=dict(expected)
    if gate:wanted['control_gate']=actual['control_gate']
    actual=copy.deepcopy(actual);wanted=copy.deepcopy(wanted)
    for obj in (actual,wanted):
        rows=obj['audit']['execution_runtime']
        require(type(rows) is list,label+' runtime list')
        for row in rows:
            require(set(row)=={'year','fit_threadpools','predict_threadpools'} and type(row['year']) is int,label+' runtime fields')
            for key in ('fit_threadpools','predict_threadpools'):
                pools=row[key];require(type(pools) is list and pools and all(type(v) is dict and type(v.get('num_threads')) is int and v['num_threads']==1 and type(v.get('filepath')) is str for v in pools),label+' single-thread runtime')
                require(len({v['filepath'] for v in pools})==len(pools),label+' unique native library')
                row[key]=sorted(pools,key=lambda v:v['filepath'])
    source.check_tree(actual,wanted,label)

def verify_control_models(train,meta,arrays,control_view,n=None):
    expected,_,_=model_oracle(train,meta,arrays,n,candidate=False)
    _compare_bundle(control_view,expected,'Control-only proof',gate=False)
    return _model_receipt(expected,True)
verify_control=verify_control_models

def verify_models(train,meta,arrays,bundle,n=None):
    expected,inp,contexts=model_oracle(train,meta,arrays,n)
    gate=bundle.get('control_gate');require(type(gate) is dict and gate.get('passed') is True,'Control gate passed')
    if gate!=expected['control_gate']:
        controlview=dict(expected,candidate=None,control_gate=None,audit=model_audit(inp,arrays,expected['control'],None))
        record_check(gate,_model_receipt(controlview,True),'Control proof scope')
    _compare_bundle(bundle,expected,'Full independent model')
    # Traverse supplied trees too, rather than trusting copied manual predictions.
    byyear={ctx['row']['year']:ctx for ctx in contexts}
    extra=0
    for row in bundle['candidate']['audit']['annual']:
        if row['status']=='fit':
            pred=manual_predict(row['tree'],byyear[row['year']]['Q']);exact_array(pred,np.asarray(row['raw_predictions']),'Supplied tree raw')
            extra+=len(pred)*100
    result=_model_receipt(expected,False);result['manual_tree_traversals']+=extra;return result

def verify_incumbent_selection(arrays,meta,orders,scores,pre_scores,n=None):
    result=source.verify_incumbent(arrays,meta,orders,scores,pre_scores,n)
    for k in ('old_model_solves','conditional_ridge_solves'):result.pop(k,None)
    result.update(independent_tree_refits=0,manual_tree_traversals=0,original_other_target_fits=0,original_local_fits=0)
    return result
verify_incumbent=verify_incumbent_selection

def _inc_gate(n):
    gate=source._inc_receipt(n)
    for k in ('old_model_solves','conditional_ridge_solves'):gate.pop(k,None)
    gate.update(independent_tree_refits=0,manual_tree_traversals=0,original_other_target_fits=0,original_local_fits=0);return gate

def rank_and_delta(arrays,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n;old,U=_pool(arrays,n);w=old.shape[1]
    require(type(bundle) is dict and type(bundle.get('n')) is int and bundle['n']>=n and bundle['width']==w,'Prefix forecast bundle')
    size=bundle['n']*w;candidate=bundle['candidate'];control=bundle['control']
    for f in (control,candidate):
        require(type(f) is dict and set(f)=={'raw','available','rank','audit'},'Forecast keys')
        for key,shape,dtype in (('raw',(size,),float),('available',(size,),bool),('rank',(bundle['n'],w),float)):
            v=np.asarray(f[key]);require(v.shape==shape and v.dtype==np.dtype(dtype),'Forecast array '+key)
    cached_raw=np.asarray(arrays['raw_source_log'])[:n].reshape(-1);cached_av=np.asarray(arrays['forecast_available'])[:n].reshape(-1)
    exact_array(np.asarray(control['raw'])[:n*w],cached_raw,'Consumed control raw source')
    for role,f in (('matched',control),('tree',candidate)):
        exact_array(np.asarray(f['available'])[:n*w],cached_av,'Consumed '+role+' availability')
    outside=~U.copy()
    for i in range(n):
        if meta['dates'][i]==INITIAL:outside[i]=True
    exact_array(np.asarray(candidate['raw'])[:n*w][outside.reshape(-1)],cached_raw[outside.reshape(-1)],'Unconsumed tree raw source copy')
    tree=old.copy();delta=np.zeros_like(old);active=[];weak=np.asarray(arrays['state_weak_narrow'])[:n]
    require(weak.dtype==np.dtype(bool) and weak.shape==(n,),'Weak state bool')
    for i,mask in enumerate(U):
        partial=meta['dates'][i]==INITIAL;ids=np.flatnonzero(mask);active.append(ids.tolist() if weak[i] and not partial else [])
        if not partial and len(ids):
            raw=candidate['raw'][i*w:(i+1)*w][mask];av=candidate['available'][i*w:(i+1)*w][mask]
            require(av.all() and np.isfinite(raw).all(),'Original full U tree prediction')
            tree[i,mask]=previous._rank(raw)
        exact_array(np.asarray(control['rank'])[i],old[i],'Matched rank prefix')
        exact_array(np.asarray(candidate['rank'])[i],tree[i],'Tree rank prefix')
        if active[-1]:delta[i,mask]=.0375*(tree[i,mask]-old[i,mask])
    audit=dict(schema='monthly_weak_tree261_rank_delta_v1',events=n,ranking_cells=int(U.sum()),active_indices=active,changed_cells=int(np.count_nonzero(delta)),
      original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,local_basis_preserved=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return dict(old_pool=old,tree=tree),delta,audit

def selection_oracle(arrays,meta,bundle,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n);ranks,delta,ra=rank_and_delta(arrays,meta,bundle,n)
    _,U=_pool(arrays,n);pres={arm:base['P'].copy() for arm in ARMS};mask=delta!=0;pres[ARMS[2]][mask]+=delta[mask]
    scores={};choices={}
    for arm in ARMS:scores[arm],choices[arm]=accounts.episode(pres[arm],base)
    c={k:0 for k in COUNT_FIELDS};c.update(events=n,symbols_per_event=base['width'],original_eligible_cells=int(base['E'].sum()),ranking_cells=int(U.sum()),
      incumbent_episode_calls=1,matched_episode_calls=1,tree_episode_calls=1)
    events=[]
    for i in range(n):
        partial=meta['dates'][i]==INITIAL;active=ra['active_indices'][i]
        c['initial_partial_events']+=int(partial);c['formal_events']+=int(not partial);c['weak_applicable_cells']+=len(active);c['changed_delta_cells']+=int(np.count_nonzero(delta[i]))
        c['matched_changed_holdings_events']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[0]][i]))
        c['tree_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[0]][i]))
        c['tree_vs_matched_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[1]][i]))
        events.append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,weak=bool(base['weak'][i]),ranking_indices=np.flatnonzero(U[i]).tolist(),
          active_indices=active,changed_delta_indices=np.flatnonzero(delta[i]).tolist(),delta=delta[i].tolist(),incumbent_choices=choices[ARMS[0]][i],matched_choices=choices[ARMS[1]][i],tree_choices=choices[ARMS[2]][i],episode_budget=float(base['budget'][i])))
    return scores,choices,pres,dict(schema='monthly_weak_tree261_decisions_v1',counts=c,events=events,rank_delta=ra,incumbent_gate=dict(passed=True,synthetic_no_callback=True),
      original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,
      calm_slot_preserved=True,partial_exact=True,computes_returns=False)

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids]) for i,ids in enumerate(choices[arm])] for arm in ARMS}

def verify_selection(arrays,meta,bundle,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(audit) is dict and 'incumbent_gate' in audit,'Full decision audit')
    gate=audit['incumbent_gate'];require(type(gate) is dict and gate.get('passed') is True,'Passed original decision gate')
    if gate!=dict(passed=True,synthetic_no_callback=True):record_check(gate,_inc_gate(n),'Incumbent receipt')
    expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,n);wanted['incumbent_gate']=copy.deepcopy(gate)
    record_check(orders,orders_from_choices(expected,choices,meta),'Exact ordered choices')
    for actual,target,label in ((scores,expected,'Scores'),(pre_scores,pres,'Pres')):
        require(set(actual)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(actual[arm],target[arm],label+'/'+arm)
    record_check(audit,wanted,'Exact complete decisions')
    return _receipt(n,SELECTION_FLAGS,counts=wanted['counts'],ordered_choices_checked=3*n,full_bundle_supplied=True,full_scores_supplied=True,
      full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,original_episode_calls=3)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three261 accounting arms')
    rename=lambda d:{old:d[new] for new,old in zip(ARMS,source.ARMS)}
    report=source.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={new:report['paths'][old] for new,old in zip(ARMS,source.ARMS)})
