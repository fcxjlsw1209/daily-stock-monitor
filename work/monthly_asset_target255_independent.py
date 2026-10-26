"""Independent255 label centering, augmented-LS model and scalar selector audit.

No255 producer/kernel/host is imported. Inputs are supplied by the registered
caller; this module contains no actual-source loader or strategy runner.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'work/monthly_identity_transfer206_independent.py'
MODEL_SHA='a321b2cd07eb2b19c08f7747842e18a47438d781bed9b474c8ef321ec9f5611e'
BRIDGE=ROOT/'work/monthly_calendar_memory254_independent.py'
BRIDGE_SHA='9b5eab143f3d299cf43b2a0eddaebeef532efb634a53fa737e4930c8eed774cc'
ACCOUNTS=ROOT/'work/monthly_calendar_clock248_v2_independent.py'
ACCOUNTS_SHA='c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'

def require(ok,message):
    if not ok:raise ValueError(message)
def load(path,digest,name):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Frozen independent dependency changed')
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
numeric=load(MODEL,MODEL_SHA,'asset255_independent_numeric')
bridge=load(BRIDGE,BRIDGE_SHA,'asset255_independent_bridge')
accounts=load(ACCOUNTS,ACCOUNTS_SHA,'asset255_independent_accounts')
exact_array,record_check=bridge.exact_array,bridge.record_check
ARMS=('incumbent_local67','asset_centered_local67')
TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
RAW_KEYS=('source_log','local_constant','local_linear')
LABEL_COUNT_FIELDS=('rows','valid_rows','invalid_rows','signal_dates','nonempty_groups','empty_groups','singleton_groups','changed_rows')
MODEL_COUNT_FIELDS=('training_rows','valid_training_rows','signal_dates','nonempty_groups','singleton_groups','changed_label_rows',
 'control_target_fits','control_log_refits','control_local_evaluations','control_local_solves',
 'candidate_target_fits','candidate_log_refits','candidate_local_evaluations','candidate_local_solves','actual_regression_solves')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells',
 'weak_applicable_cells','changed_delta_cells','changed_holdings_events','incumbent_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','weak','ranking_indices','active_indices','changed_delta_indices','delta',
 'incumbent_choices','candidate_choices','episode_budget')
CONTROL_FLAGS=('passed','original_training_identity_checked','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked',
 'original_pooled_local_models_checked','original_model_cache_checked','availability_unchanged','original_query_rank_domain_checked')
MODEL_FLAGS=CONTROL_FLAGS+('stock_etf_group_identity_checked','saved_binary64_label_centering_checked','invalid_labels_unchanged',
 'singleton_rows_retained','both_pooled_and_local_targets_changed','independent_augmented_least_squares',
 'full_model_audit_checked','actual_fit_counts_checked','no_future_training_values')
INCUMBENT_FLAGS=bridge.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_eligibility_unchanged','original_core_unchanged',
 'original_support_and_budget_checked','original_query_rank_domain_checked','pooled_local_common_readiness_checked',
 'weak_only_five_percent_slot_checked','calm_nonweak_slot_unchanged','named_initial_partial_unchanged',
 'exact_zero_delta_preserved','single_episode_per_arm_checked','no_final_global_rerank','two_names_and_cash_checked',
 'exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
PATHS,ACCOUNTING_SHA=accounts.PATHS,accounts.ACCOUNTING_SHA

def dependency_files():
    return tuple(sorted(set(bridge.dependency_files())|set(accounts.dependency_files())|{MODEL,BRIDGE,ACCOUNTS,Path(__file__).resolve()}))
def accounting():return accounts.accounting()
def original_base(original,meta,n):return accounts.original_base(original,meta,n)

def continuous(actual,expected,label):
    a,e=np.asarray(actual),np.asarray(expected)
    require(a.shape==e.shape,label+' shape')
    require(np.array_equal(np.isnan(a),np.isnan(e)) and np.array_equal(np.isfinite(a),np.isfinite(e)),label+' masks')
    require(not np.isinf(a).any() and not np.isinf(e).any(),label+' infinity')
    use=np.isfinite(e);require(np.all(np.abs(a[use]-e[use])<=CONTINUOUS_ATOL+CONTINUOUS_RTOL*np.abs(e[use])),label+' values')

def check_record(actual,expected,label):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual)==set(expected),label+' keys')
        for k,v in expected.items():check_record(actual[k],v,label+'/'+k)
    elif type(expected) is list:
        require(type(actual) is list and len(actual)==len(expected),label+' list')
        for i,(a,e) in enumerate(zip(actual,expected)):check_record(a,e,label+'/'+str(i))
    elif type(expected) is float:continuous(actual,expected,label)
    elif isinstance(expected,str):require(isinstance(actual,(str,np.str_)) and actual==expected,label+' string')
    else:require(type(actual)==type(expected) and actual==expected,label+' discrete')

def days(value,name):
    a=np.asarray(value)
    try:out=np.asarray(a,dtype='datetime64[D]')
    except (TypeError,ValueError) as exc:raise ValueError('Invalid '+name) from exc
    require(out.ndim==1 and not np.isnat(out).any(),'Valid '+name)
    return out

def identifiers(value,n,name):
    a=np.asarray(value)
    require(a.shape==(n,) and all(isinstance(v,(str,np.str_)) and v and str(v).strip()==v for v in a),'Explicit '+name)
    return a.astype(str)

def recenter_labels(dates,symbols,cached_y,common_good,etfs):
    d=days(dates,'label dates');s=identifiers(symbols,len(d),'training securities');y=np.asarray(cached_y);g=np.asarray(common_good)
    require(y.shape==(len(d),) and y.dtype==np.dtype('float64'),'Saved binary64 log labels')
    require(g.shape==y.shape and g.dtype==np.dtype(bool),'Exact Boolean common_good')
    require(isinstance(etfs,(list,tuple)) and len(set(etfs))==len(etfs) and all(type(v) is str and v and v.strip()==v for v in etfs),'Fixed ETF list')
    require(len({(str(day.astype('datetime64[M]')),symbol) for day,symbol in zip(d,s)})==len(d),'Duplicate security-month')
    require(np.isfinite(y[g]).all(),'Nonfinite valid cached label')
    out=y.copy();groups=[];counts={k:0 for k in LABEL_COUNT_FIELDS}
    counts.update(rows=len(d),valid_rows=int(g.sum()),invalid_rows=int((~g).sum()),signal_dates=len(np.unique(d)))
    etf=np.array([v in set(etfs) for v in s],bool)
    for day in np.unique(d):
        for kind,mask in (('stock',~etf),('ETF',etf)):
            ids=np.flatnonzero((d==day)&g&mask);mean=None
            if len(ids):
                mean=float(np.mean(y[ids],dtype=np.float64));require(np.isfinite(mean),'Finite group mean')
                out[ids]=0. if len(ids)==1 else y[ids]-mean
                require(np.isfinite(out[ids]).all(),'Finite centered labels')
            counts['nonempty_groups']+=int(bool(len(ids)));counts['empty_groups']+=int(not len(ids));counts['singleton_groups']+=int(len(ids)==1)
            groups.append(dict(date=str(day),asset_type=kind,row_ids=ids.tolist(),mean=mean,singleton=len(ids)==1))
    require(out[~g].tobytes()==y[~g].tobytes(),'Invalid label bytes unchanged')
    counts['changed_rows']=int(np.count_nonzero(out[g]!=y[g]))
    return out,dict(schema='monthly_asset_target255_labels_v1',counts=counts,groups=groups,
      mean_arithmetic='numpy.mean float64 in original row order',input_already_centered=True,
      raw_target_reconstructed=False,invalid_rows_unchanged=True,common_good_unchanged=True,no_new_row_filter=True,no_reclip=True)

def model_inputs(train,meta,arrays,n=None):
    require(type(meta) is dict and 'monthly' in meta and 'training' in meta,'Matching training/monthly metadata')
    m,tm=meta['monthly'],meta['training'];n=len(m['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(m['dates']),'Model query prefix')
    E=np.asarray(arrays['eligible']);require(E.ndim==2 and E.dtype==np.dtype(bool) and E.shape[0]>=n,'Original Boolean eligibility')
    width=E.shape[1];require(width>0,'Positive query width')
    qd=days(m['dates'][:n],'query dates');require(all(a<b for a,b in zip(qd,qd[1:])),'Increasing queries')
    qnames=[]
    for ss in m['symbols'][:n]:
        require(isinstance(ss,(list,tuple)) and len(ss)==width and len(set(ss))==width,'Unique query row identities')
        qnames.extend(identifiers(ss,width,'query symbols'))
    d=days(train['dates'],'training dates');ends=days(train['ends'],'training ends')
    require(d.shape==ends.shape and np.all(ends>d) and np.all(ends.astype('datetime64[M]')==d.astype('datetime64[M]')+1),'Next-month labels')
    offsets=np.asarray(train['offsets']);require(offsets.ndim==1 and offsets.dtype.kind in 'iu' and len(offsets)==len(tm['dates'])+1,'Training offsets')
    require(offsets[0]==0 and offsets[-1]==len(d) and np.all(np.diff(offsets)>0),'Complete training offsets')
    require(len(tm['ends'])==len(tm['symbols'])==len(tm['dates']),'Training metadata axes')
    keep=np.zeros(len(d),bool)
    if n:
        year=int(qd[-1].astype('datetime64[Y]').astype(int))+1970
        keep=ends<=np.datetime64('%04d-12-31'%(year-1))
    ids=np.flatnonzero(keep);names=np.empty(len(d),dtype=object)
    for i in range(len(offsets)-1):
        a,b=int(offsets[i]),int(offsets[i+1])
        if not keep[a:b].any():continue
        require(keep[a:b].all(),'No partial training cohort maturity')
        ss=identifiers(tm['symbols'][i],b-a,'training symbols');require(len(set(ss))==len(ss),'Duplicate training security')
        require(np.all(d[a:b]==np.datetime64(tm['dates'][i],'D')) and np.all(ends[a:b]==np.datetime64(tm['ends'][i],'D')),'Training metadata identity')
        names[a:b]=ss
    X0=np.asarray(train['train_plain']);require(X0.shape==(len(d),10),'Ten original train features')
    good0=np.asarray(train['common_good']);require(good0.shape==(len(d),) and good0.dtype==np.dtype(bool),'Original bool good')
    ys={}
    for key in TARGETS:
        a=np.asarray(train['target_'+key]);require(a.shape==(len(d),),'Target row axis')
        ys[key]=np.asarray(a[ids],dtype=np.float64)
    query_dates=days(np.asarray(train['event_dates'])[:n*width],'cached query dates')
    require(np.array_equal(query_dates,np.repeat(qd,width)),'Exact query date rows')
    qX0=np.asarray(train['event_plain']);require(qX0.ndim==2 and qX0.shape[0]>=n*width and qX0.shape[1]==10,'Query features')
    qX=np.asarray(qX0[:n*width],float)
    return dict(n=n,width=width,training_row_ids=ids.astype(np.int64),dates=d[ids],ends=ends[ids],symbols=np.asarray(names[ids],str),
      X=np.asarray(X0[ids],float),targets=ys,good=good0[ids].copy(),query_dates=query_dates,query_symbols=np.asarray(qnames,str),
      query_X=qX,query_good=E[:n].reshape(-1)&np.isfinite(qX).all(axis=1),etfs=m['etfs'])

def annual_oracle(inp,targets,*,min_months=18,min_own_months=18):
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
    source_annual=[];annual=[];details=[]
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
    return dict(raw=raw,available=available,local_available=local_available,audit=audit)

def check_forecast(actual,expected,label):
    require(type(actual) is dict and set(actual)=={'raw','available','local_available','audit'},label+' fields')
    require(set(actual['raw'])==set(RAW_KEYS),label+' raw keys')
    for k in RAW_KEYS:
        require(np.asarray(actual['raw'][k]).dtype==np.dtype('float64'),label+'/'+k+' binary64')
        continuous(actual['raw'][k],expected['raw'][k],label+'/'+k)
    for k in ('available','local_available'):exact_array(actual[k],expected[k],label+'/'+k)
    check_record(actual['audit'],expected['audit'],label+'/audit')

def _rank(values):
    v=np.asarray(values,float);require(v.ndim==1 and np.isfinite(v).all(),'Finite rank domain')
    return np.array([(sum(v<x)+(sum(v==x)+1)/2.)/len(v)-.5 for x in v],dtype=float)

def original_ranks(arrays,n):
    E=np.asarray(arrays['eligible'])[:n];shape=E.shape
    old=np.asarray(arrays['rank_log_excess'])[:n];fore=np.asarray(arrays['forecast_available'])[:n]
    require(E.dtype==fore.dtype==np.dtype(bool) and old.shape==fore.shape==shape,'Original rank axes')
    require(np.isfinite(old).all() and (np.abs(old)<=.5+1e-12).all(),'Original rank bounds')
    U=E&fore;local=old.copy();source=np.asarray(arrays['raw_source_log'])[:n]
    raws={k:np.asarray(arrays['raw_'+k])[:n] for k in ('local_constant','local_linear')}
    available=np.asarray(arrays['model_available'])[:n];require(available.shape==shape and available.dtype==np.dtype(bool),'Original model available')
    for i,use in enumerate(U):
        if use.any():
            require(np.isfinite(source[i,use]).all(),'Original source domain')
            exact_array(_rank(source[i,use]),old[i,use],'Original pooled cache ranks')
        if use.any() and available[i,use].all() and all(np.isfinite(v[i,use]).all() for v in raws.values()):
            values=raws['local_linear'][i,use]
            if not np.array_equal(values,source[i,use]):local[i,use]=_rank(values)
    if 'rank_local_linear' in arrays:exact_array(np.asarray(arrays['rank_local_linear'])[:n],local,'Original local cache ranks')
    return old.copy(),local,U

def verify_control(train,meta,arrays,control,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);expected=annual_oracle(inp,inp['targets'],min_months=min_months,min_own_months=min_own_months)
    check_forecast(control,expected,'Original model')
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

def verify_models(train,meta,arrays,bundle,n=None,*,min_months=18,min_own_months=18):
    inp=model_inputs(train,meta,arrays,n);n=inp['n'];width=inp['width']
    require(type(bundle) is dict and set(bundle)=={'schema','n','width','training_row_ids','control','candidate','label_values','label_audit','control_gate','audit'},'Full model bundle keys')
    require(bundle['schema']=='monthly_asset_target255_models_v1' and type(bundle['n']) is int and bundle['n']==n and type(bundle['width']) is int and bundle['width']==width,'Model bundle axes')
    exact_array(bundle['training_row_ids'],inp['training_row_ids'],'Original training row IDs')
    proof=verify_control(train,meta,arrays,bundle['control'],n,min_months=min_months,min_own_months=min_own_months)
    y,la=recenter_labels(inp['dates'],inp['symbols'],inp['targets']['log_excess'],inp['good'],inp['etfs'])
    exact_array(bundle['label_values'],y,'All centered label bytes');record_check(bundle['label_audit'],la,'Complete label audit')
    targets=dict(inp['targets'],log_excess=y)
    candidate=annual_oracle(inp,targets,min_months=min_months,min_own_months=min_own_months)
    check_forecast(bundle['candidate'],candidate,'Recentered pooled and local model')
    exact_array(bundle['candidate']['available'],bundle['control']['available'],'Common model availability')
    exact_array(bundle['candidate']['local_available'],bundle['control']['local_available'],'Common local history availability')
    supplied_ranks,delta,_,_,_=rank_oracle(arrays,meta['monthly'],bundle,n)
    rebuilt_ranks,rebuilt_delta,_,_,_=rank_oracle(arrays,meta['monthly'],dict(bundle,candidate=candidate),n)
    for k in supplied_ranks:exact_array(supplied_ranks[k],rebuilt_ranks[k],'Independent exact rank/'+k)
    exact_array(delta,rebuilt_delta,'Independent exact delta')
    counts=model_counts(inp,la,bundle['control'],candidate)
    wanted=dict(schema='monthly_asset_target255_model_audit_v1',counts=counts,
      original_cache_parity=dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False),
      original_row_order_preserved=True,original_common_good_preserved=True,all_other_targets_preserved=True,new_labels_feed_pooled_and_local=True,
      inherited_model_audit_target_text='Frozen helper wording unchanged cached centered log_excess describes its argument; candidate argument IS recentered. Wrapper label_audit is authoritative.',
      residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False)
    record_check(bundle['audit'],wanted,'Complete model and fit audit')
    gate=bundle['control_gate'];require(type(gate) is dict and gate.get('passed') is True,'Pre-candidate original model gate')
    gate_expected=wanted['original_cache_parity'] if set(gate)==set(wanted['original_cache_parity']) else proof
    record_check(gate,gate_expected,'Complete saved original model gate')
    control_solves=proof['independent_solve_counts'];candidate_solves=independent_solve_counts(candidate)
    solves=dict(control=control_solves,candidate=candidate_solves,total_regression_solves=control_solves['total_regression_solves']+candidate_solves['total_regression_solves'])
    return dict(**{k:True for k in MODEL_FLAGS},checked_events=n,counts=counts,producer_equivalent_counts=counts,
      independent_solve_counts=solves,full_bundle_supplied=True,
      full_labels_supplied=True,full_control_supplied=True,full_candidate_supplied=True,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
      independent_numeric_system='augmented_least_squares_SVD',producer_modules_imported=False,computes_returns=False)

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    return accounts.verify_incumbent(arrays,meta,orders,scores,pre_scores,n)

def model_counts(inp,label_audit,control,candidate):
    c={k:0 for k in MODEL_COUNT_FIELDS};lc=label_audit['counts']
    c.update(training_rows=lc['rows'],valid_training_rows=lc['valid_rows'],signal_dates=lc['signal_dates'],
      nonempty_groups=lc['nonempty_groups'],singleton_groups=lc['singleton_groups'],changed_label_rows=lc['changed_rows'])
    for name,model in (('control',control),('candidate',candidate)):
        years=sum(row['status']=='fit' and row['prediction_rows']>0 for row in model['audit']['annual'])
        c[name+'_target_fits']=4*years;c[name+'_log_refits']=years
        c[name+'_local_evaluations']=len(model['audit']['local'])
        c[name+'_local_solves']=sum(row['status']=='fit' for row in model['audit']['local'])
        c['actual_regression_solves']+=5*years+c[name+'_local_solves']
    return c

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
            newpool[i,use]=_rank(raw['source_log'][i,use])
            newlocal[i,use]=newpool[i,use] if np.array_equal(raw['local_linear'][i,use],raw['source_log'][i,use]) else _rank(raw['local_linear'][i,use])
        isactive=bool(weak[i]) and str(meta['dates'][i])!='2016-08-17'
        active.append(ids.tolist() if isactive else [])
        if isactive and len(ids):
            delta[i,use]=.05*(newpool[i,use]-old[i,use])+.0125*((newlocal[i,use]-newpool[i,use])-(local[i,use]-old[i,use]))
    require(np.isfinite(delta).all(),'Finite weak slot delta')
    return dict(old_pool=old,old_local=local,new_pool=newpool,new_local=newlocal),delta,U,ready,active

def rank_and_delta(arrays,meta,bundle,n=None):
    ranks,delta,U,ready,active=rank_oracle(arrays,meta,bundle,n)
    return ranks,delta,dict(schema='monthly_asset_target255_rank_delta_v1',events=U.shape[0],ranking_cells=int(U.sum()),
      ready_rows=ready.tolist(),active_indices=active,changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,
      no_group_ranking=True,no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)

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
    ra=dict(schema='monthly_asset_target255_rank_delta_v1',events=n,ranking_cells=int(U.sum()),ready_rows=ready.tolist(),
      active_indices=active,changed_cells=int(np.count_nonzero(delta)),original_ranking_population=True,no_group_ranking=True,
      no_own_history_filter=True,weak_only=True,partial_exact=True,calm_slot_preserved=True,no_post_rerank=True)
    return scores,choices,pres,dict(schema='monthly_asset_target255_decisions_v1',counts=counts,events=events,rank_delta=ra,
      incumbent_gate=None,original_arrays_unchanged=True,original_eligibility_unchanged=True,original_core_support_preserved=True,
      single_original_episode_per_arm=True,calm_slot_preserved=True,partial_exact=True,computes_returns=False)

def orders_from_choices(scores,choices,meta):
    return {a:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[a][i,j]) for j in ids])
      for i,ids in enumerate(choices[a])] for a in ARMS}

def verify_selection(arrays,meta,bundle,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Independent255 ordered choices')
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
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Two255 accounting arms')
    mapping=dict(zip(ARMS,accounts.ARMS));rename=lambda d:{mapping[k]:v for k,v in d.items()}
    for arm in ARMS:
        for name,_,fee in PATHS:
            if fee==0.:
                require(float(results[arm][name]['metrics']['fees_paid'])==0. and float(results[arm][name]['slices']['full122']['fees_paid'])==0.,'ZERO aggregate fees exact0')
    report=accounts.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={a:report['paths'][mapping[a]] for a in ARMS})
