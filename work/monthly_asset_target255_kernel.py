"""Pure caller-supplied label/model/weak-slot contrast; no data loader or replay."""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'outputs/monthly_local_linear70/model.py'
MODEL_SHA='6d9075e5ce08edd0587d2b2ea0d98635620141070f1af08771b6a69435467a8a'
BRIDGE_PATH=ROOT/'work/monthly_calendar_clock248_v2_kernel.py'
BRIDGE_SHA='b580b2a2389201f48d81cd78ddecb641699fdbd919c79202746665789b5a6bcb'

def _module(name,path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

model=_module('asset255_original_local_model',MODEL_PATH,MODEL_SHA)
bridge=_module('asset255_original_cache_bridge',BRIDGE_PATH,BRIDGE_SHA)
require=bridge.require
TARGETS=tuple(model.TARGETS)
ARMS=('incumbent_local67','asset_centered_local67')
RAW_KEYS=('source_log','local_constant','local_linear')
RANK_KEYS=('old_pool','old_local','new_pool','new_local')
INITIAL_PARTIAL='2016-08-17'
POLICY=dict(budget=.05,ceiling=5,gap=.02)
BUNDLE_KEYS=('schema','n','width','training_row_ids','control','candidate','label_values','label_audit','control_gate','audit')
LABEL_COUNT_FIELDS=('rows','valid_rows','invalid_rows','signal_dates','nonempty_groups','empty_groups','singleton_groups','changed_rows')
LABEL_GROUP_KEYS=('date','asset_type','row_ids','mean','singleton')
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','signal_dates','nonempty_groups','singleton_groups','changed_label_rows','control_target_fits','control_log_refits','control_local_evaluations','control_local_solves','candidate_target_fits','candidate_log_refits','candidate_local_evaluations','candidate_local_solves','actual_regression_solves')
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

def recenter_labels(dates,symbols,cached_y,common_good,etfs):
    dates=_days(dates,'training dates');n=len(dates);symbols=_symbols(symbols,n,'training symbols')
    good=_bool(common_good,n,'common_good');y=np.asarray(cached_y)
    require(y.shape==(n,) and y.dtype==np.dtype('float64'),'Saved binary64 label vector')
    require(isinstance(etfs,(list,tuple)) and len(set(etfs))==len(etfs) and all(type(s) is str and s for s in etfs),'Frozen ETF IDs')
    identities=list(zip(dates.astype('datetime64[M]').astype(str),symbols.tolist()))
    require(len(identities)==len(set(identities)),'Duplicate security-month')
    require(np.isfinite(y[good]).all(),'Nonfinite valid cached target')
    out=y.copy();is_etf=np.isin(symbols,etfs);groups=[];counts=dict.fromkeys(LABEL_COUNT_FIELDS,0)
    counts.update(rows=n,valid_rows=int(good.sum()),invalid_rows=int((~good).sum()),signal_dates=len(np.unique(dates)))
    for d in np.unique(dates):
        for name,group in (('stock',~is_etf),('ETF',is_etf)):
            ids=np.flatnonzero((dates==d)&group&good);mean=None
            if len(ids):
                with np.errstate(over='raise',invalid='raise',divide='raise'):
                    mean=float(np.mean(y[ids],dtype=np.float64))
                    if len(ids)==1:out[ids]=0.
                    else:out[ids]=y[ids]-mean
                require(np.isfinite(mean) and np.isfinite(out[ids]).all(),'Nonfinite recentered target')
                counts['nonempty_groups']+=1;counts['singleton_groups']+=int(len(ids)==1)
            else:counts['empty_groups']+=1
            groups.append(dict(date=str(d),asset_type=name,row_ids=ids.tolist(),mean=mean,singleton=len(ids)==1))
    counts['changed_rows']=int(np.count_nonzero(out[good]!=y[good]))
    require(out[~good].tobytes()==y[~good].tobytes(),'Invalid target bytes changed')
    return out,dict(schema='monthly_asset_target255_labels_v1',counts=counts,groups=groups,
        mean_arithmetic='numpy.mean float64 in original row order',input_already_centered=True,
        raw_target_reconstructed=False,invalid_rows_unchanged=True,common_good_unchanged=True,
        no_new_row_filter=True,no_reclip=True)

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

def fit_models(train,meta,arrays,n=None,*,control_validator=None,min_months=18,min_own_months=18):
    args,ids,n,w=_inputs(train,meta,arrays,n);snapshot=bridge._snapshot(arrays)
    # Validate original label identities/validity before any original model call; no new label is formed.
    dates,ends,symbols,X,targets,good,event_dates,event_symbols,event_X,event_good=args
    identity=list(zip(dates.astype('datetime64[M]').astype(str),symbols.tolist()))
    require(len(identity)==len(set(identity)),'Duplicate security-month')
    require(np.isfinite(targets['log_excess'][good]).all(),'Nonfinite valid original log target')
    control=model.annual_forecasts(*args,min_months=min_months,min_own_months=min_own_months)
    parity=control_parity(arrays,control,n)
    receipt=parity
    if control_validator is not None:
        require(callable(control_validator),'Control validator callable');receipt=control_validator(deepcopy(control))
        require(type(receipt) is dict and receipt.get('passed') is True,'Independent original model gate failed')
    new_y,la=recenter_labels(dates,symbols,targets['log_excess'],good,meta['monthly']['etfs'])
    modified=list(args);modified[4]={k:v.copy() for k,v in targets.items()};modified[4]['log_excess']=new_y.copy()
    candidate=model.annual_forecasts(*modified,min_months=min_months,min_own_months=min_own_months)
    _forecast_layout(candidate,n,w)
    np.testing.assert_array_equal(candidate['available'],control['available']);np.testing.assert_array_equal(candidate['local_available'],control['local_available'])
    counts=dict.fromkeys(MODEL_COUNT_FIELDS,0)
    counts.update(training_rows=len(ids),valid_training_rows=int(good.sum()),signal_dates=la['counts']['signal_dates'],nonempty_groups=la['counts']['nonempty_groups'],singleton_groups=la['counts']['singleton_groups'],changed_label_rows=la['counts']['changed_rows'])
    for prefix,f in (('control',control),('candidate',candidate)):
        for suffix,value in zip(('target_fits','log_refits','local_evaluations','local_solves'),_fit_counts(f)):counts[prefix+'_'+suffix]=int(value)
    counts['actual_regression_solves']=sum(counts[p+'_'+s] for p in ('control','candidate') for s in ('target_fits','log_refits','local_solves'))
    require(snapshot==bridge._snapshot(arrays),'Original arrays changed')
    return dict(schema='monthly_asset_target255_models_v1',n=n,width=w,training_row_ids=ids,control=control,candidate=candidate,label_values=new_y,label_audit=la,control_gate=deepcopy(receipt),audit=dict(schema='monthly_asset_target255_model_audit_v1',counts=counts,original_cache_parity=parity,
        original_row_order_preserved=True,original_common_good_preserved=True,all_other_targets_preserved=True,new_labels_feed_pooled_and_local=True,
        inherited_model_audit_target_text='Frozen helper wording unchanged cached centered log_excess describes its argument; candidate argument IS recentered. Wrapper label_audit is authoritative.',
        residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False))

def rank_and_delta(arrays,meta,bundle,n=None):
    total,w=np.shape(arrays['baseline']);n=total if n is None else n
    require(type(n) is int and 0<=n<=total and type(bundle) is dict and bundle.get('schema')=='monthly_asset_target255_models_v1' and type(bundle.get('n')) is int and n<=bundle['n'] and bundle.get('width')==w,'Model prefix axes')
    require(type(meta) is dict and isinstance(meta.get('dates'),list) and len(meta['dates'])==total,'Rank metadata')
    old_pool,old_local,use,old_ready=_old_ranks(arrays,n,w)
    f=bundle['candidate'];_forecast_layout(f,bundle['n'],w,limit=n)
    available=f['available'][:n*w].reshape(n,w);pool=f['raw']['source_log'][:n*w].reshape(n,w);local=f['raw']['local_linear'][:n*w].reshape(n,w)
    require(np.array_equal(available,np.asarray(arrays['model_available'])[:n]),'Candidate pooled support changed')
    ready=use.any(axis=1)&np.all(~use|(available&np.isfinite(pool)&np.isfinite(local)),axis=1)
    new_pool=old_pool.copy();new_local=old_local.copy()
    for i in range(n):
        if ready[i]:
            new_pool[i]=_rank(pool[i:i+1],use[i:i+1],old_pool[i:i+1])[0]
            new_local[i]=new_pool[i] if np.array_equal(local[i,use[i]],pool[i,use[i]]) else _rank(local[i:i+1],use[i:i+1],new_pool[i:i+1])[0]
    weak=bridge._bool(arrays['state_weak_narrow'],(total,),n,'weak')
    formal=np.asarray([x!=INITIAL_PARTIAL for x in meta['dates'][:n]],dtype=bool)
    active=use&ready[:,None]&old_ready[:,None]&weak[:,None]&formal[:,None];delta=np.zeros((n,w))
    with np.errstate(over='raise',invalid='raise'):
        delta[active]=.05*(new_pool[active]-old_pool[active])+.0125*((new_local[active]-new_pool[active])-(old_local[active]-old_pool[active]))
    same=(new_pool==old_pool)&(new_local==old_local);delta[same]=0.
    require(np.isfinite(delta).all() and (np.abs(delta)<=.05+1e-12).all(),'Bounded weak replacement')
    ranks=dict(old_pool=old_pool,old_local=old_local,new_pool=new_pool,new_local=new_local)
    audit=dict(schema='monthly_asset_target255_rank_delta_v1',events=n,ranking_cells=int(use.sum()),ready_rows=ready.tolist(),active_indices=[np.flatnonzero(x).tolist() for x in active],changed_cells=int(np.count_nonzero(delta)),
        original_ranking_population=True,no_group_ranking=True,no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
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
    return orders,scores,pres,dict(schema='monthly_asset_target255_decisions_v1',counts=counts,events=events,rank_delta=ra,incumbent_gate=deepcopy(receipt),original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)
