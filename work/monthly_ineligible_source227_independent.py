"""Independent source-only extension audit; no producer imports or portfolio math.

Label arithmetic and calendar/cohort control flow are rebuilt here. Forecasts
use independently reconstructed annual/query control flow with explicitly shared,
SHA-bound original numerical primitives; this is not an independent linear solver.
"""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
RAW_KEYS = ('source_log','local_constant','local_linear')
LIMITS = dict(simple_excess=1.,log_excess=1.,vol_scaled=5.,loss_utility=1.)
LOCAL = ROOT/'outputs/monthly_local_linear70/model.py'
PINS = {'outputs/monthly_local_linear70/model.py':'6d9075e5ce08edd0587d2b2ea0d98635620141070f1af08771b6a69435467a8a',
 'outputs/monthly_target_definition70/model.py':'e0a67648920f06bb7516d5914907ee237cac2fe4a3c5316940d54f21e7de9832',
 'outputs/monthly_architecture_restart/residual_model.py':'ea294b9a0a1e9493799f1014dcec464effa93dce70f44fb5de53a604cd832886'}
FORECAST_COUNT_FIELDS = ('query_rows','original_available_rows','expansion_requested_rows','added_available_rows','available_rows',
 'old_source_target_fits','new_source_target_fits','old_pooled_log_refits','new_pooled_log_refits',
 'old_local_evaluations','new_local_evaluations','old_local_solves','new_local_solves','actual_regression_solves')
FORECAST_FLAGS = ('passed','independent_annual_control_flow','shared_frozen_primitives_disclosed','old_forecasts_exact',
 'old_masks_exact','old_forecasts_copied','same_training_cohort','four_targets_unchanged','annual_prior_december_cutoffs',
 'five_year_mature_window','minimum_18_months','same_security_local_history','local_history_not_query_filter',
 'separate_expansion_replay','warmup_mask_enforced','no_query_rerank','fit_counts_checked','audit_checked')
LABEL_FLAGS = ('passed','independent_label_rebuild','exact_original_identity','original_centering_cohort_unchanged',
 'fixed_four_clips_checked','old_targets_exact_copy','old_common_mask_preserved','original_reconstruction_checked',
 'only_previously_ineligible_added','finite_plain10_checked','full_295_history_checked','volatility63_checked',
 'full_official_future_path_checked','official_endpoints_checked','no_added_training_rows','audit_checked')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen():
    for name,digest in PINS.items(): require(sha(ROOT/name)==digest,'Frozen primitive changed: '+name)
    spec=importlib.util.spec_from_file_location('source227_independent_frozen_local',LOCAL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple([Path(__file__).resolve(),*(ROOT/p for p in PINS)])


def close(actual,expected,label,atol=0.):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape,label+': shape')
    require(not np.isinf(a).any() and not np.isinf(b).any(),label+': infinity')
    require(np.array_equal(np.isnan(a),np.isnan(b)),label+': NaN mask')
    use=np.isfinite(b);require(np.all(np.abs(a[use]-b[use])<=atol),label+': values')
    if atol==0: require(np.array_equal(np.signbit(a[use]),np.signbit(b[use])),label+': signed zero')


def record_check(actual,expected,label='audit'):
    if isinstance(actual,np.generic):actual=actual.item()
    if isinstance(expected,np.generic):expected=expected.item()
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': keys')
        for k in expected: record_check(actual[k],expected[k],label+'/'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    else:require(type(actual) is type(expected) and actual==expected,label+': exact value/type')


def days(values,name):
    a=np.asarray(values);require(a.ndim==1,name+': vector')
    if not len(a): return np.array([],dtype='datetime64[D]')
    require(a.dtype.kind not in 'biufc',name+': nonnumeric dates')
    d=pd.DatetimeIndex(a);require(not d.hasnans and d.tz is None and d.equals(d.normalize()),name+': midnight dates')
    return d.to_numpy(dtype='datetime64[D]').copy()


def boolean(values,shape,name):
    a=np.asarray(values);require(a.shape==shape and a.dtype.kind=='b',name+': exact bool shape');return a.copy()


def symbols(values,n,name):
    a=np.asarray(values);require(a.shape==(n,) and all(isinstance(s,(str,np.str_)) and s and s==s.strip() for s in a),name+': exact IDs')
    return a.astype(str,copy=True)


def numeric(values,shape,name):
    a=np.asarray(values);require(a.shape==shape and a.dtype.kind in 'fiu',name+': numeric shape');return a.astype(float,copy=True)


def model_inputs(args):
    require(len(args)==10,'Original ten model arguments')
    d,e,s,X,ys,g,qd,qs,Q,qg=args
    d,e,qd=days(d,'training dates'),days(e,'training ends'),days(qd,'query dates')
    n,q=len(d),len(qd)
    require(e.shape==d.shape and np.all(e>d) and np.all(e.astype('datetime64[M]')==d.astype('datetime64[M]')+1),'Mature next-month ends')
    s,qs=symbols(s,n,'training symbols'),symbols(qs,q,'query symbols')
    require(len(set(zip(d.astype(str),s)))==n,'Duplicate training identity')
    require(len(set(zip(qd.astype(str),qs)))==q,'Duplicate query identity')
    X,Q=numeric(X,(n,10),'train plain10'),numeric(Q,(q,10),'query plain10')
    g,qg=boolean(g,(n,),'common_good'),boolean(qg,(q,),'event_good')
    require(isinstance(ys,dict) and set(ys)==set(TARGETS),'Four original targets')
    ys={k:numeric(ys[k],(n,),'target '+k) for k in TARGETS}
    for month in np.unique(d.astype('datetime64[M]')):
        ids=d.astype('datetime64[M]')==month
        require(len(np.unique(d[ids]))==len(np.unique(e[ids]))==1,'One signal/end per training month')
    return d,e,s,X,ys,g,qd,qs,Q,qg


def replay_oracle(*args):
    """Annual fits, then independently issue one-row numerical predictions."""
    d,e,s,X,ys,g,qd,qs,Q,qg=model_inputs(args);n,q=len(d),len(qd)
    lib=frozen();num=lib.numeric
    months=d.astype('datetime64[M]');years=qd.astype('datetime64[Y]').astype(int)+1970
    good=g & np.isfinite(X).all(axis=1)
    for y in ys.values():good &= np.isfinite(y)
    for month in np.unique(months):
        group=months==month
        if np.count_nonzero(good & group)<2:good[group]=False
    queries=qg & np.isfinite(Q).all(axis=1)
    raw={k:np.full(q,np.nan) for k in RAW_KEYS};available=np.zeros(q,bool);local_available=np.zeros(q,bool)
    source_annual=[];annual=[];details=[]
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff=np.datetime64('%04d-12-31'%(year-1),'D');start=np.datetime64('%04d-12-31'%(year-6),'D')
            ids=np.flatnonzero(good & (d>start) & (e<=cutoff));selected=np.flatnonzero((years==year)&queries)
            count=len(np.unique(months[ids]));status='fit' if count>=18 else 'cold_start'
            base=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),training_rows=len(ids),training_months=count,
                prediction_rows=len(selected),latest_label_end=str(e[ids].max()) if len(ids) else None,
                first_training_signal=str(d[ids].min()) if len(ids) else None,
                latest_training_signal=str(d[ids].max()) if len(ids) else None,status=status)
            if count>=18 and len(selected):
                require(np.all(np.abs(X[ids])<=.5) and np.all(np.abs(Q[selected])<=.5),'Used original percentile bounds')
                # Same fixed numerical primitive, but no production annual/query controller.
                unique,inverse,counts=np.unique(months[ids],return_inverse=True,return_counts=True)
                weights=1./counts[inverse].astype(float)
                ages=cutoff.astype('datetime64[M]').astype(int)-unique.astype(int)
                weights*=np.exp2(-ages[inverse]/12.)
                weights*=len(weights)/weights.sum()
                models={k:num._fit_ridge(X[ids],ys[k][ids],weights,10.) for k in TARGETS}
                for j in selected:
                    raw['source_log'][j]=num._predict(models['log_excess'],Q[j:j+1])[0]
                available[selected]=True;base['weight_sum']=float(weights.sum())
            source_annual.append(base)
            a={k:v for k,v in base.items() if k!='latest_training_signal'}
            a['prediction_rows']=int(np.count_nonzero((years==year)&available))
            if a['prediction_rows']:
                fit=num._fit_ridge(X[ids],ys['log_excess'][ids],weights,10.)
                resid=ys['log_excess'][ids]-num._predict(fit,X[ids])
                require(np.isfinite(resid).all(),'Finite mature residuals')
                for j in np.flatnonzero((years==year)&available):
                    source=float(num._predict(fit,Q[j:j+1])[0]);require(source==raw['source_log'][j],'Pooled log replay')
                    raw['local_constant'][j]=source;raw['local_linear'][j]=source
                    own=np.flatnonzero(s[ids]==qs[j]);oid=ids[own];nc=len(np.unique(months[oid]))
                    detail=dict(event_row=int(j),date=str(qd[j]),symbol=str(qs[j]),year=int(year),cutoff=str(cutoff),own_rows=len(oid),own_months=nc,
                        latest_label_end=str(e[oid].max()) if len(oid) else None,status='insufficient_own_history',weight_mass=0.,effective_samples=0.,constant_correction=0.,linear_correction=0.)
                    if nc>=18:
                        w=lib.local_weights(d[oid],cutoff,X[oid],Q[j])
                        c,l,info=lib.local_corrections(X[oid],Q[j],fit[1],resid[own],w)
                        detail.update(info,constant_correction=c,linear_correction=l)
                        local_available[j]=info['weight_mass']>0.
                        if c!=0.:raw['local_constant'][j]+=c
                        if l!=0.:raw['local_linear'][j]+=l
                    details.append(detail)
            annual.append(a)
    source_audit=dict(alpha=10.,half_life_months=12.,min_months=18,lookback_years=5,targets=list(TARGETS),shared_rows_and_weights=True,
        training_only_standardization=True,weighting='equal_month_total_then_age_decay_normalized_to_row_count',predictions_are_raw=True,label_ranks_used=False,annual=source_annual)
    audit=dict(modes=['local_constant','local_linear'],feature_count=10,target='unchanged cached centered log_excess',alpha=10.,half_life_months=12.,min_months=18,
        min_own_months=18,bandwidth=.5,intercept_penalty=12.,slope_penalty=10.,same_local_rows_residuals_and_weights=True,local_weights_normalized=False,
        local_recency_clock='annual prior December cutoff calendar-month age',kernel='mean squared difference in original centered percentile plain10 units',
        linear_design='query-centered original X differences divided by pooled annual training scales',residuals='in-sample residuals within already matured annual training cohort, not OOF',
        source_control='independent original four-target annual model and exact log fit replay',no_own_history='exact source raw copy; never remove a query or alter eligibility',
        source=source_audit,annual=annual,local=details)
    return dict(raw=raw,available=available,local_available=local_available,audit=audit)


def forecasts_oracle(*args,cached_source,expansion_allowed):
    args=list(model_inputs(args));q=len(args[6]);allowed=boolean(expansion_allowed,(q,),'expansion_allowed')
    old=replay_oracle(*args)
    ca=boolean(cached_source['available'],(q,),'cached available');cl=boolean(cached_source['local_available'],(q,),'cached local_available')
    require(np.array_equal(old['available'],ca) and np.array_equal(old['local_available'],cl),'Original mask exact parity')
    for key in RAW_KEYS:close(old['raw'][key],cached_source['raw_'+key],'Original '+key)
    newmask=allowed & np.isfinite(args[8]).all(axis=1) & ~args[9]
    newargs=list(args);newargs[9]=newmask
    new=replay_oracle(*newargs)
    require(not np.any(new['available']&ca),'Separate query domain')
    raw={k:np.asarray(cached_source['raw_'+k],float).copy() for k in RAW_KEYS}
    for k in RAW_KEYS:raw[k][new['available']]=new['raw'][k][new['available']]
    avail=ca | new['available'];local=cl | new['local_available']
    counts=dict(query_rows=q,original_available_rows=int(ca.sum()),expansion_requested_rows=int(newmask.sum()),
        added_available_rows=int(new['available'].sum()),available_rows=int(avail.sum()))
    for name,record in (('old',old),('new',new)):
        a=record['audit'];counts[name+'_source_target_fits']=4*sum(r['status']=='fit' and r['prediction_rows']>0 for r in a['source']['annual'])
        counts[name+'_pooled_log_refits']=sum(r['prediction_rows']>0 for r in a['annual'])
        counts[name+'_local_evaluations']=len(a['local']);counts[name+'_local_solves']=sum(r['status']=='fit' for r in a['local'])
    counts['actual_regression_solves']=sum(counts[k] for k in ('old_source_target_fits','new_source_target_fits','old_pooled_log_refits','new_pooled_log_refits','old_local_solves','new_local_solves'))
    audit=dict(method='same_query_ineligible_prediction_extension',query_rows=q,feature_count=10,raw_keys=list(RAW_KEYS),old_parity_exact=True,
        original_forecasts_copied=True,new_queries_separate_replay=True,original_training_unchanged=True,features_recomputed=False,reranked=False,
        expansion_allowed=allowed.tolist(),new_query_mask=newmask.tolist(),old_replay=old['audit'],new_replay=new['audit'],counts=counts)
    return dict(raw=raw,available=avail,local_available=local,audit=audit)


def check_forecasts(*args,cached_source,expansion_allowed,result):
    forecast=result
    expected=forecasts_oracle(*args,cached_source=cached_source,expansion_allowed=expansion_allowed)
    require(set(forecast)==set(expected),'Forecast return schema')
    for k in RAW_KEYS:close(forecast['raw'][k],expected['raw'][k],'Extended '+k)
    for k in ('available','local_available'):require(np.array_equal(forecast[k],expected[k]),k+' exact')
    record_check(forecast['audit'],expected['audit'])
    return dict(**{k:True for k in FORECAST_FLAGS},counts=expected['audit']['counts'],query_rows=len(expected['available']),
        numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,shared_numerical_primitives=True)

LABEL_COUNT_FIELDS = ('training_rows','original_good_rows','added_rows','extended_good_rows','original_center_months','original_strict_exceptions','new_candidate_rows')


def frame(adj):
    require(isinstance(adj,pd.DataFrame) and isinstance(adj.index,pd.DatetimeIndex),'Adjusted price frame')
    axis=days(adj.index,'price index');require(adj.columns.is_unique and (not len(axis) or np.all(axis[1:]>axis[:-1])),'Unique sorted price axes')
    return axis


def training_axes(train,meta):
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','ends','symbols')),'Training metadata')
    md,me=days(meta['dates'],'metadata dates'),days(meta['ends'],'metadata ends')
    require(md.shape==me.shape and len(md)==len(meta['symbols']),'Metadata dimensions')
    require(not len(md) or np.all(md[1:]>md[:-1]),'Ordered unique signals')
    require(np.all(me>md) and np.all(me.astype('datetime64[M]')==md.astype('datetime64[M]')+1),'Next natural month')
    widths=[];names=[]
    for pool in meta['symbols']:
        s=symbols(pool,len(pool),'metadata symbols');require(len(s)==len(set(s)),'Unique monthly IDs');widths.append(len(s));names.extend(s)
    off=np.r_[0,np.cumsum(widths)].astype(int)
    require(np.asarray(train['offsets']).dtype.kind in 'iu' and np.array_equal(train['offsets'],off),'Original exact offsets')
    d,e=days(train['dates'],'training dates'),days(train['ends'],'training ends')
    require(np.array_equal(d,np.repeat(md,widths)) and np.array_equal(e,np.repeat(me,widths)),'Original exact date/end identity')
    n=len(d);X=numeric(train['train_plain'],(n,10),'original plain10')
    eligible=boolean(train['train_eligible'],(n,),'training eligible');good=boolean(train['common_good'],(n,),'common good')
    cached=boolean(train['train_good'],(n,),'original train_good')
    require(not np.any(good & (~eligible | ~cached | ~np.isfinite(X).all(axis=1))),'Original common subset')
    targets={k:numeric(train['target_'+k],(n,),'original target '+k) for k in TARGETS}
    for k in TARGETS:
        require(not np.isinf(targets[k]).any() and np.array_equal(np.isfinite(targets[k]),good),'Original target/common mask')
    vol=numeric(train['vol63'],(n,),'original vol63');require(not np.isinf(vol).any(),'Original vol infinity')
    return md,me,off,d,e,np.asarray(names,str),X,eligible,good,targets,vol


def schedule(calendar,md,me):
    axis=days(calendar,'official calendar')
    require(not len(axis) or np.all(axis[1:]>axis[:-1]),'Sorted unique official sessions')
    positions={v:i for i,v in enumerate(axis)}
    for d,e in zip(md,me):
        require(d in positions and e in positions,'Missing official signal/end')
        for v in (d,e):
            i=positions[v]
            require(i+1<len(axis) and axis[i+1].astype('datetime64[M]')==v.astype('datetime64[M]')+1,'Official month-end next-session witness')
    return axis,positions


def prices(adj,symbol,index):
    if symbol not in adj.columns:return None
    try: result=adj.loc[index,symbol].to_numpy(dtype=float)
    except (KeyError,ValueError,TypeError,OverflowError):return None
    return result if np.isfinite(result).all() and np.all(result>0) else None


def raw_labels(history,asset_endpoints,spy_endpoints):
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        r=history[1:]/history[:-1]-1.
        volatility=float(np.std(r,ddof=1))
        own=asset_endpoints[1]/asset_endpoints[0]
        bench=spy_endpoints[1]/spy_endpoints[0]
        ar=own-1.;br=bench-1.;excess=ar-br
        if not np.isfinite(r).all() or not np.isfinite(volatility):raise FloatingPointError('Nonfinite volatility')
        if volatility==0:return volatility,None
        raw=dict(simple_excess=float(excess),log_excess=float(np.log(own)-np.log(bench)),
            vol_scaled=float(excess/(volatility*np.sqrt(21.))),loss_utility=float(excess-.5*max(0.,-ar)))
    if not all(np.isfinite(v) for v in raw.values()):raise FloatingPointError('Nonfinite derived raw label')
    return volatility,raw


def original_measure(adj,axis,date,end,symbol):
    p=int(np.searchsorted(axis,date))
    require(p<len(axis) and axis[p]==date and p>=63,'Original old row requires 64 frame prices')
    hist=prices(adj,symbol,pd.DatetimeIndex(axis[p-63:p+1]));endpoints=pd.DatetimeIndex([date,end])
    asset=prices(adj,symbol,endpoints);spy=prices(adj,'SPY',endpoints)
    require(hist is not None and asset is not None and spy is not None,'Original admitted source unavailable')
    v,raw=raw_labels(hist,asset,spy);require(raw is not None,'Original admitted zero vol')
    return v,raw


def strict_measure(adj,price_axis,official,pos,date,end,symbol,finite_features):
    if not finite_features:return False,'nonfinite_features',None,None
    p=pos[date]
    if p<294:return False,'insufficient_history',None,None
    if symbol not in adj.columns:return False,'missing_symbol',None,None
    h=official[p-294:p+1];future=official[p:pos[end]+1]
    if not np.isin(np.r_[h,future],price_axis).all():return False,'missing_official_price_session',None,None
    hist=prices(adj,symbol,pd.DatetimeIndex(h))
    if hist is None:return False,'invalid_history_price',None,None
    asset=prices(adj,symbol,pd.DatetimeIndex(future))
    if asset is None:return False,'invalid_future_asset_price',None,None
    if 'SPY' not in adj.columns:return False,'missing_spy',None,None
    spy=prices(adj,'SPY',pd.DatetimeIndex(future))
    if spy is None:return False,'invalid_future_spy_price',None,None
    v,raw=raw_labels(hist[-64:],asset[[0,-1]],spy[[0,-1]])
    if v==0:return False,'zero_volatility',v,None
    return True,'valid',v,raw


def labels_oracle(adj,train,training_meta,calendar):
    axis=frame(adj)
    md,me,offset,d,e,names,X,eligible,oldgood,oldtargets,vol=training_axes(train,training_meta)
    official,pos=schedule(calendar,md,me);n=len(d)
    targets={k:v.copy() for k,v in oldtargets.items()};good=oldgood.copy();added=np.zeros(n,bool)
    records=[];month_records=[];exceptions=0;center_months=0
    for month,(date,end) in enumerate(zip(md,me)):
        ids=np.arange(offset[month],offset[month+1]);oldids=ids[oldgood[ids]]
        require(len(oldids)==0 or len(oldids)>=2,'Original singleton center is invalid')
        centers={k:None for k in TARGETS}
        if len(oldids)>=2:
            oldraw={k:[] for k in TARGETS}
            for row in oldids:
                v,raw=original_measure(adj,axis,date,end,names[row]);close(v,vol[row],'Original vol63',1e-12)
                for k in TARGETS:oldraw[k].append(raw[k])
            for k in TARGETS:
                clipped=np.clip(np.asarray(oldraw[k],float),-LIMITS[k],LIMITS[k]);centers[k]=float(clipped.mean())
                close(clipped-centers[k],oldtargets[k][oldids],'Original centered '+k,1e-12)
            center_months+=1
        for row in ids:
            raw=None;strict=None;exception=False;reason='original_eligible_not_admitted'
            if oldgood[row] or not eligible[row]:
                strict,sr,v,raw=strict_measure(adj,axis,official,pos,date,end,names[row],bool(np.isfinite(X[row]).all()))
                if oldgood[row]:
                    exception=not strict;exceptions+=int(exception)
                    reason='original_retained' if strict else 'original_retained_exception:'+sr
                else:
                    reason=sr
                    if strict:
                        vol[row]=v
                        if centers[TARGETS[0]] is None:reason='fewer_than_two_original_center_rows'
                        else:
                            for k in TARGETS:targets[k][row]=np.clip(raw[k],-LIMITS[k],LIMITS[k])-centers[k]
                            added[row]=True;good[row]=True;reason='added'
            records.append(dict(row=int(row),symbol=str(names[row]),date=str(date),end=str(end),original_eligible=bool(eligible[row]),
                original_good=bool(oldgood[row]),added=bool(added[row]),strict_source_valid=strict,reason=reason,
                original_strict_exception=exception,vol63=float(vol[row]) if np.isfinite(vol[row]) else None,raw_targets=raw))
        month_records.append(dict(date=str(date),end=str(end),rows=ids.tolist(),original_center_row_ids=oldids.tolist(),centers=centers,
            original_good_rows=int(len(oldids)),added_rows=int(added[ids].sum())))
    reasons={}
    for row in records:reasons[row['reason']]=reasons.get(row['reason'],0)+1
    counts=dict(training_rows=n,original_good_rows=int(oldgood.sum()),added_rows=int(added.sum()),extended_good_rows=int(good.sum()),
        original_center_months=center_months,original_strict_exceptions=exceptions,new_candidate_rows=int((~eligible).sum()))
    audit=dict(method='same_query_ineligible_label_extension',label_parity_atol=1e-12,label_parity_rtol=0.,original_targets_copied=True,
        original_common_good_preserved=True,original_centers_only=True,added_rows_never_train=True,full_official_paths_required_for_added=True,
        original_strict_exceptions_retained=True,counts=counts,reason_counts=reasons,months=month_records,rows=records)
    return dict(targets=targets,good=good,added=added,vol63=vol,audit=audit)


def check_labels(adj,train,training_meta,calendar,result):
    expected=labels_oracle(adj,train,training_meta,calendar)
    require(set(result)==set(expected),'Label result schema')
    for k in TARGETS:close(result['targets'][k],expected['targets'][k],'Extended label '+k)
    for k in ('good','added'):require(np.array_equal(result[k],expected[k]),k+' exact')
    close(result['vol63'],expected['vol63'],'vol63 exact')
    record_check(result['audit'],expected['audit'])
    return dict(**{k:True for k in LABEL_FLAGS},counts=expected['audit']['counts'],numerical_tolerance=0.,
        original_reconstruction_atol=1e-12,producer_modules_imported=False,computes_returns=False,computes_labels=True)
