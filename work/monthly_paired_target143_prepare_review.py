"""Independent round143 preparation oracle; no real-data work on import.

No imports of the new producer, target adapter or model. Label schedules, paired
labels, annual Ridge, same-security corrections and ranking are reconstructed.
The old frozen independent episode/price verifier is reused, not the strategy.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import monthly_price_delay140_prepare_review as prior

old=prior.old
ROOT=Path(__file__).resolve().parents[1]
STUDY='outputs/monthly_paired_target143'
LOCAL=old.LOCAL
TARGET='outputs/monthly_target_definition70'
TRAIN_SOURCE='outputs/monthly_earnings_factor_research'
ARCH='outputs/monthly_architecture_restart'
NAMES=('incumbent_local67','paired_close_target','paired_open_target')
ARMS=('close','open')
INPUTS=('proposal','meta','policy','original_orders','prices','protected_manifest','train_inputs','training_meta')
EXTRA=('paired_raw_close','paired_raw_open','paired_pooled_close','paired_pooled_open','paired_available','paired_local_available')
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','LABELS.npz','LABEL_AUDIT.json','MODEL_AUDIT.json','IDENTITY_AUDIT.json','CALENDAR.json','RANK_AUDIT.json')
CHECKS=('original27_arrays_byte_identical','training_identity_and_plain10_verified','raw_prices_independently_verified','calendar_overlay_independently_verified','paired_labels_independently_reconstructed','shared_maturity_and_cohort_verified','annual_forecasts_independently_reconstructed','all_three_score_matrices_and_ordered_choices_independently_verified','all_signal_prefixes_verified','whole_month_incumbent_fallback_verified','one_episode_and_top2_lock_verified')
CACHED_KEYS=('dates','ends','event_dates','offsets','train_plain','event_plain','train_good','train_eligible','eligible')
ATOL=1e-12
PROPOSAL_SHA='d1b93c8ae18e11998e011a3b91a76af761d79be0ada8f3435258b3365cfe5256'
require=old.require
compare=prior.compare


def source_files(root=None):
    root=ROOT if root is None else Path(root).resolve()
    import threadpoolctl
    return sorted(set([root/'work/monthly_paired_target143_prepare_review.py',root/'work/test_monthly_paired_target143_prepare_review.py',Path(threadpoolctl.__file__).resolve()]+prior.source_files(root)))


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def day_vector(values,name):
    axis=pd.DatetimeIndex(values)
    require(not axis.hasnans and axis.tz is None and axis.equals(axis.normalize()),'Canonical dates '+name)
    return axis.to_numpy(dtype='datetime64[D]')


def bool_vector(values,n,name):
    out=np.asarray(values);require(out.shape==(n,) and out.dtype.kind=='b','Boolean vector '+name)
    return out.copy()


def schedule_oracle(sessions,dates):
    calendar=prior.canonical_axis(sessions);days=day_vector(dates,'signals');n=len(days)
    require(len(calendar)>0,'Nonempty official calendar')
    end=np.empty(n,dtype='datetime64[D]');entry=end.copy();exit_=end.copy()
    for d in np.unique(days):
        date=pd.Timestamp(d);month=date.to_period('M');current=calendar[calendar.to_period('M')==month]
        following=calendar[calendar.to_period('M')==month+1]
        require(len(current)>0 and date==current[-1] and date in calendar,'Training signal must be official month end')
        require(len(following)>0,'Full following calendar month required')
        e=following[-1];apos=calendar.searchsorted(date,side='right');bpos=calendar.searchsorted(e,side='right')
        require(apos<len(calendar) and bpos<len(calendar),'Exit following open required')
        a,b=calendar[apos],calendar[bpos]
        require(a.to_period('M')==month+1 and b.to_period('M')==month+2,'Calendar boundary coverage')
        mask=days==d;end[mask]=e.to_datetime64();entry[mask]=a.to_datetime64();exit_[mask]=b.to_datetime64()
    return dict(dates=days,calendar_ends=end,entry_sessions=entry,exit_sessions=exit_,available_sessions=exit_.copy())


def labels_oracle(close,opening,sessions,dates,symbols,common_good,eligible,finite_features):
    schedule=schedule_oracle(sessions,dates);dates=schedule['dates'];n=len(dates)
    for frame in (close,opening):
        require(isinstance(frame,pd.DataFrame),'Price frame');prior.canonical_axis(frame.index)
        require(frame.columns.is_unique and 'SPY' in frame.columns,'Unique symbols and SPY')
    require(close.index.equals(opening.index) and close.columns.equals(opening.columns),'Matched close/open axes')
    symbols=np.asarray(symbols);require(symbols.shape==(n,) and all(isinstance(s,(str,np.str_)) and s for s in symbols),'Explicit symbols')
    months=dates.astype('datetime64[M]');require(len(set(zip(months.astype(str),symbols)))==n,'Unique security/month')
    base=bool_vector(common_good,n,'common')&bool_vector(eligible,n,'eligible')&bool_vector(finite_features,n,'features')
    raw={k:np.full(n,np.nan)for k in ARMS};target={k:np.full(n,np.nan)for k in ARMS};endpoint=np.zeros(n,bool);good=np.zeros(n,bool);audit=[]
    def read_pair(frame,points,symbol):
        if symbol not in frame.columns:return None
        values=np.asarray([frame.at[pd.Timestamp(day),symbol]for day in points],dtype=float)
        return values if np.isfinite(values).all() and (values>0).all() else None
    for month in np.unique(months):
        rows=np.flatnonzero(months==month);first=rows[0];candidates=rows[base[rows]]
        points=[dates[first],schedule['calendar_ends'][first],schedule['entry_sessions'][first],schedule['exit_sessions'][first]]
        reason='fewer_than_two_paired_rows'
        if not len(candidates):reason='no_originally_usable_rows'
        elif not all(pd.Timestamp(d)in close.index for d in points):reason='missing_price_session'
        else:
            benchmarks=[read_pair(close,points[:2],'SPY'),read_pair(opening,points[2:],'SPY')]
            if any(x is None for x in benchmarks):reason='invalid_benchmark_endpoint'
            else:
                for j in candidates:
                    stocks=[read_pair(close,points[:2],symbols[j]),read_pair(opening,points[2:],symbols[j])]
                    if any(x is None for x in stocks):continue
                    for key,pair,bench in zip(ARMS,stocks,benchmarks):
                        # Four logarithms, no unstable ratio or future path filter.
                        raw[key][j]=(np.log(pair[1])-np.log(pair[0]))-(np.log(bench[1])-np.log(bench[0]))
                        require(np.isfinite(raw[key][j]),'Finite raw log target')
                    endpoint[j]=True
        chosen=rows[endpoint[rows]];clips={k:0 for k in ARMS}
        if len(chosen)>=2:
            good[chosen]=True;reason='complete'
            for key in ARMS:
                clips[key]=int(np.count_nonzero(np.abs(raw[key][chosen])>1.))
                values=np.clip(raw[key][chosen],-1.,1.);target[key][chosen]=values-values.mean()
        audit.append(dict(month=str(month),signal_date=str(dates[first]),calendar_end=str(schedule['calendar_ends'][first]),entry_session=str(schedule['entry_sessions'][first]),exit_session=str(schedule['exit_sessions'][first]),available_session=str(schedule['available_sessions'][first]),input_rows=int(len(rows)),base_rows=int(len(candidates)),endpoint_rows=int(len(chosen)),paired_rows=int(good[rows].sum()),reason=reason,clip_counts={'paired_'+k:v for k,v in clips.items()}))
    arrays=dict(schedule,symbols=symbols.copy(),good=good,base_good=base,endpoint_good=endpoint)
    arrays.update({'target_'+k:v for k,v in target.items()})
    report=dict(stage='preparatory_paired_labels_only',arms=['paired_close','paired_open'],min_month_rows=2,clip_limit=1.,clipping='clip raw values then center on identical paired monthly rows',maturity='both arms available only at exit_session opening',inherited_full_path_mask=True,future_query_eligibility_used=False,calendar_provenance_verified_here=False,months=audit)
    return dict(arrays=arrays,raw=raw,audit=report)


def weights_oracle(dates,cutoff):
    months,inverse,counts=np.unique(dates.astype('datetime64[M]'),return_inverse=True,return_counts=True)
    ages=cutoff.astype('datetime64[M]').astype(int)-months.astype(int)
    values=(1./counts[inverse].astype(float))*np.exp2(-ages[inverse]/12.)
    return values*(len(values)/values.sum())


def fit_oracle(x,y,w):
    """Independent augmented least-squares Ridge, same weighted standardizer."""
    mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    z=(x-mean)/scale;intercept=float(np.average(y,weights=w))
    matrix=np.vstack([z*np.sqrt(w[:,None]),np.eye(10)*np.sqrt(10.)]);rhs=np.r_[(y-intercept)*np.sqrt(w),np.zeros(10)]
    beta=np.linalg.lstsq(matrix,rhs,rcond=None)[0]
    return mean,scale,beta,intercept


def predict_oracle(fit,x):return np.sum(((x-fit[0])/fit[1])*fit[2],axis=1)+fit[3]


def model_oracle(labels,x,event_dates,symbols,event_x,event_good,*,min_months=18,min_own_months=18,fit_cache=None):
    require(type(min_months)is int and min_months>0 and type(min_own_months)is int and min_own_months>0,'Positive model minima')
    d=day_vector(labels['dates'],'training');ends=day_vector(labels['calendar_ends'],'ends');availdays=day_vector(labels['available_sessions'],'availability')
    ed=day_vector(event_dates,'events');n,q=len(d),len(ed);months=d.astype('datetime64[M]')
    require(ends.shape==availdays.shape==(n,) and np.all(ends>d) and np.all(availdays>ends) and np.all(ends.astype('datetime64[M]')==months+1) and np.all(availdays.astype('datetime64[M]')==months+2),'Shared maturity anchors')
    x=np.asarray(x,float);event_x=np.asarray(event_x,float);require(x.shape==(n,10) and event_x.shape==(q,10),'Original ten columns')
    ts=np.asarray(labels['symbols']);symbols=np.asarray(symbols);require(ts.shape==(n,) and symbols.shape==(q,),'Identity axes')
    good=bool_vector(labels['good'],n,'training');event_good=bool_vector(event_good,q,'event')
    targets={k:np.asarray(labels['target_'+k],float)for k in ARMS};require(all(v.shape==(n,)for v in targets.values()),'Target axes')
    usable=good&np.isfinite(x).all(axis=1)
    for target in targets.values():usable&=np.isfinite(target)
    for m in np.unique(months):
        if np.count_nonzero(usable&(months==m))<2:usable[months==m]=False
    events=event_good&np.isfinite(event_x).all(axis=1);years=ed.astype('datetime64[Y]').astype(int)+1970
    raw={k:np.full(q,np.nan)for k in ARMS};pooled={k:np.full(q,np.nan)for k in ARMS};available=np.zeros(q,bool);local_available=np.zeros(q,bool);annual=[];local=[];fits=0
    for year in np.unique(years):
        cutoff=np.datetime64('%04d-12-31'%(year-1),'D');start=np.datetime64('%04d-12-31'%(year-6),'D')
        train=usable&(d>start)&(availdays<=cutoff);query=(years==year)&events;ids=np.flatnonzero(train);count=len(np.unique(months[train]))
        record=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),training_rows=len(ids),training_months=count,prediction_rows=int(query.sum()),first_training_signal=str(d[train].min())if len(ids)else None,latest_calendar_end=str(ends[train].max())if len(ids)else None,latest_available_session=str(availdays[train].max())if len(ids)else None,training_indices_sha256=digest(ids),status='fit'if count>=min_months and query.any()else 'cold_start'if count<min_months else 'no_queries')
        if count>=min_months and query.any():
            require(len(set(zip(months[train].astype(str),ts[train])))==len(ids),'Unique actual training identity')
            require((np.abs(x[train])<=.5).all() and (np.abs(event_x[query])<=.5).all(),'Actual centered percentile ranks')
            w=weights_oracle(d[train],cutoff);key=(int(year),digest(ids),digest(x[train]),digest(targets['close'][train]),digest(targets['open'][train]))
            if fit_cache is not None and key in fit_cache:models=fit_cache[key]
            else:
                models={k:fit_oracle(x[train],targets[k][train],w)for k in ARMS}
                if fit_cache is not None:fit_cache[key]=models
            fits+=2;record.update(weights_sha256=digest(w),weight_sum=float(w.sum()),scaler_mean_sha256=digest(models['close'][0]),scaler_scale_sha256=digest(models['close'][1]))
            residual={k:targets[k][train]-predict_oracle(models[k],x[train])for k in ARMS}
            available[query]=True
            for k in ARMS:pooled[k][query]=predict_oracle(models[k],event_x[query]);raw[k][query]=pooled[k][query]
            for j in np.flatnonzero(query):
                own=np.flatnonzero(ts[train]==symbols[j]);ownids=ids[own];owncount=len(np.unique(months[ownids]));corrections={k:0. for k in ARMS}
                detail=dict(event_row=int(j),date=str(ed[j]),symbol=str(symbols[j]),year=int(year),cutoff=str(cutoff),own_rows=len(own),own_months=owncount,latest_available_session=str(availdays[ownids].max())if len(own)else None,status='insufficient_own_history',corrections=corrections)
                if owncount>=min_own_months:
                    age=cutoff.astype('datetime64[M]').astype(int)-months[ownids].astype(int)
                    localw=np.exp2(-age/12.)*np.exp(-np.mean((x[ownids]-event_x[j])**2,axis=1)/(2*.5**2))
                    mass=float(localw.sum());local_available[j]=mass>0
                    for k in ARMS:
                        if mass>0 and np.any(residual[k][own]!=0.):
                            z=np.column_stack([np.ones(len(own)),(x[ownids]-event_x[j])/models[k][1]])
                            lhs=np.vstack([z*np.sqrt(localw[:,None]),np.diag(np.sqrt([12.]+[10.]*10))]);rhs=np.r_[residual[k][own]*np.sqrt(localw),np.zeros(11)]
                            corrections[k]=float(np.linalg.lstsq(lhs,rhs,rcond=None)[0][0]);raw[k][j]+=corrections[k]
                    detail.update(status='fit'if mass>0 else 'zero_weight',weight_mass=mass,weights_sha256=digest(localw))
                local.append(detail)
        annual.append(record)
    for k in ARMS:require(np.array_equal(np.isfinite(raw[k]),available) and np.array_equal(np.isfinite(pooled[k]),available),'Finite output exactly available')
    return dict(raw=raw,pooled=pooled,available=available,local_available=local_available,audit=dict(targets=list(ARMS),new_pooled_model_fits=fits,annual=annual,local=local,alpha=10.,half_life_months=12.,min_months=min_months,min_own_months=min_own_months,bandwidth=.5,intercept_penalty=12.,slope_penalty=10.,shared_rows_weights_scaler=True,local_weights_normalized=False,residuals='in-sample within mature annual cohort, not OOF',maturity='shared next opening after next-calendar-month end; not calendar_end',query_eligibility='only supplied original signal-time mask and finite features',exact_calendar_validation='caller must validate with frozen official session calendar'))


def rank_oracle(v):
    return np.array([(sum(y<x for y in v)+(sum(y==x for y in v)+1)/2)/len(v)-.5 for x in v])


def scores_oracle(a,meta,n=None):
    h=old.original_helper();x=h.original_prefix(a,meta,n);n=x['n'];shape=x['core'].shape
    values={k:np.asarray(a['paired_raw_'+k])[:n]for k in ARMS};available=np.asarray(a['paired_available'])[:n]
    require(available.shape==shape and available.dtype.kind=='b','Model availability')
    require(all(v.shape==shape and v.dtype.kind in 'fiu' and not np.isinf(v).any()for v in values.values()),'Raw forecast schema')
    use=np.asarray(a['eligible'])[:n]&np.asarray(a['forecast_available'])[:n];original=np.asarray(a['rank_log_excess'])[:n];ready=np.zeros(n,bool)
    ranks={k:original.copy()for k in ARMS}
    for i in range(n):
        ready[i]=bool(use[i].any()and available[i,use[i]].all()and all(np.isfinite(v[i,use[i]]).all()for v in values.values()))
        if ready[i]:
            for k in ARMS:ranks[k][i,use[i]]=rank_oracle(values[k][i,use[i]])
    c,s,records=h.episode(x['local'],x);scores={NAMES[0]:s};choices={NAMES[0]:c}
    for name,k in zip(NAMES[1:],ARMS):
        pre=x['scope'].copy();g=.25*(.05*x['weak'][:,None]+.01*(x['calm']&~x['weak'])[:,None]*np.asarray(a['extension_available'])[:n]);active=use&ready[:,None]
        delta=np.zeros(shape);delta[active]=(g*(ranks[k]-original))[active];pre[delta!=0.]+=delta[delta!=0.]
        require((np.abs(delta)<=.0125+1e-12).all(),'Unchanged small slot')
        cc,ss,details=h.episode(pre,x)
        for i in range(n):
            if not ready[i]:cc[i]=c[i].copy();ss[i]=s[i]
            else:require(set(details[i]['top2'])<=set(details[i]['selected'])if len(details[i]['selected'])>=2 else True,'Top two locked')
        scores[name]=ss;choices[name]=cc
    report=dict(months=[dict(signal_index=i,date=meta['dates'][i],original_population=int(use[i].sum()),paired_ready=bool(ready[i]),fallback_to_local_incumbent=not bool(ready[i]))for i in range(n)])
    return scores,choices,report



def independent_forecast_decisions(a,meta,forecast):
    """Compare discrete ranks and orders, not just near-equal raw forecasts.

    A sub-tolerance raw difference can reverse a near tie. Independent raw
    forecasts therefore drive a second complete rank/score/episode path and
    must produce exactly identical ranks and ordered decisions on original U.
    """
    shape=np.asarray(a['baseline']).shape
    independent=dict(a)
    for group in ('raw','pooled'):
        for arm in ARMS:
            independent['paired_'+group+'_'+arm]=np.asarray(forecast[group][arm]).reshape(shape).copy()
    for key in ('available','local_available'):
        independent['paired_'+key]=np.asarray(forecast[key]).reshape(shape).copy()
    producer_scores,producer_choices,producer_audit=scores_oracle(a,meta)
    scores,choices,audit=scores_oracle(independent,meta)
    compare(producer_audit,audit,'Independent ranking readiness')
    use=np.asarray(a['eligible'])&np.asarray(a['forecast_available'])
    for i,month in enumerate(audit['months']):
        if not month['paired_ready']:continue
        for arm in ARMS:
            actual=rank_oracle(np.asarray(a['paired_raw_'+arm])[i,use[i]])
            expected=rank_oracle(independent['paired_raw_'+arm][i,use[i]])
            require(np.array_equal(actual,expected),'Independent forecast discrete rank mismatch')
    for name in NAMES:
        require(producer_choices[name]==choices[name],'Independent forecast ordered choices mismatch')
        np.testing.assert_allclose(producer_scores[name],scores[name],rtol=0,atol=ATOL)
    return independent,scores,choices,audit


def identity_oracle(meta,training,cached,source,arch):
    require(training['training']==arch['training'] and training['monthly']==arch['monthly']==meta,'Historical metadata identity')
    for k in CACHED_KEYS:
        require(cached[k].dtype==source[k].dtype and cached[k].shape==source[k].shape and cached[k].tobytes()==source[k].tobytes(),'Original training bytes '+k)
    tm=training['training'];widths=np.array([len(s)for s in tm['symbols']],int);offsets=np.r_[0,np.cumsum(widths)]
    require(len(tm['dates'])==len(tm['ends'])==len(widths),'Training month axes')
    require(all(len(s)==len(set(s))and all(isinstance(v,str)and v for v in s)for s in tm['symbols']),'Unique original pools')
    np.testing.assert_array_equal(cached['offsets'],offsets)
    for k in ('dates','ends'):np.testing.assert_array_equal(np.asarray(cached[k],dtype='datetime64[D]'),np.repeat(np.array(tm[k],dtype='datetime64[D]'),widths))
    np.testing.assert_array_equal(day_vector(cached['event_dates'],'eventdates'),np.repeat(np.array(meta['dates'],dtype='datetime64[D]'),49))
    require(cached['train_plain'].shape==(len(np.concatenate(tm['symbols'])),10)and cached['event_plain'].shape==(len(meta['dates'])*49,10),'Original ten-feature axes')
    for k in ('train_eligible','train_good'):bool_vector(cached[k],offsets[-1],k)
    return dict(training_months=len(offsets)-1,rows=int(offsets[-1]),eligible_rows=int(cached['train_eligible'].sum()),exact_all9_source_arrays=True,historical_metadata_equal_architecture=True,explicit_symbols_offsets_dates_ends=True,no_price_or_label_reconstruction=True)


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha)is not None,'Explicit PLAN SHA')
    r=old.Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==143 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)and plan.get('primary_challenger')==NAMES[2],'Round143 policy')
    require(r.path(plan['output_directory'])==p,'Output directory');require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Runtime')
    require(not any((p/k).exists()for k in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','FROZEN.json','RESULTS.json')),'Before outcome stages')
    pins=r.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named input schema');inputs={}
    for k,ref in plan['inputs'].items():
        path=r.path(ref['path']);require(r.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input '+k);inputs[k]=path
    require(r.sha(inputs['proposal'])==PROPOSAL_SHA,'Frozen proposal')
    require(all(type(plan.get(k))is int and plan[k]==0 for k in ('actual_model_fits_before_registration','actual_feature_calculations_before_registration','actual_returns_before_registration')),'Prospective registration')
    require(plan['tolerances']['feature_atol']==ATOL and plan['tolerances']['feature_rtol']==0 and plan['tolerances']['scores_atol']==ATOL and plan['tolerances']['scores_rtol']==0,'Frozen tolerances')
    sources={str(f):r.sha(f)for f in source_files(root)};require(r.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent source closure')
    oldplan=r.json(LOCAL+'/PLAN.json');raw=r.verify(plan['raw_source_hashes']);require(raw==r.verify(oldplan['raw_source_hashes'])and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Raw source identity')
    oldcache=r.verify(r.json(LOCAL+'/CACHE_HASHES.json'),r.path(LOCAL));require(all(pins.get(k)==v for k,v in oldcache.items()),'Original cache closure')
    for key,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):require(inputs[key]==r.path(LOCAL+'/'+file),'Original input '+key)
    require(inputs['prices']==r.path(old.PRICES) and inputs['train_inputs']==r.path(TARGET+'/TRAIN_INPUTS.npz')and inputs['training_meta']==r.path(TRAIN_SOURCE+'/META.json'),'Original training/price wiring')
    cache=r.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Cache output schema');r.verify(cache,p)
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta']),'Metadata equality');require(len(meta['dates'])==122 and meta['train_months']==88 and meta['early_months']==53 and all(len(s)==49 for s in meta['symbols']),'122x49 monthly events')
    original=r.npz(LOCAL+'/DATA.npz');a=r.npz(p/'DATA.npz');require(len(original)==27 and set(a)==set(original)|set(EXTRA),'Preserve original27 array schema')
    for k,v in original.items():require(a[k].shape==v.shape and a[k].dtype==v.dtype and a[k].tobytes()==v.tobytes(),'Original bytes '+k)
    h=old.original_helper();require(all(r.json(inputs['policy']).get(k)==v for k,v in h.POLICY.items()),'Original episode policy')
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Raw loader code');spec=importlib.util.spec_from_file_location('paired143_independent_raw',r.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(old.LOADER))]=old.LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Consumed raw source closure');r.verify(used)
    price=r.npz(inputs['prices']);require(set(price)=={'dates','symbols','close','opening'},'Price cache schema');parity=old.price_parity(panel,price);compare(r.json(p/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    px=pd.DataFrame(price['close'],index=pd.DatetimeIndex(price['dates']),columns=price['symbols']);op=pd.DataFrame(price['opening'],index=px.index,columns=px.columns)
    sessions=old.official_sessions(r,'2035-12-31');cal=r.json(p/'CALENDAR.json');require(cal['start']=='2008-01-01'and cal['end']=='2035-12-31'and cal['provenance']==plan['calendar'],'Calendar provenance')
    require(cal['sessions']==[str(d.date())for d in sessions],'All official sessions and exceptions')
    calendar_hashes=r.verify(plan['calendar']['source_hashes']);require(all(pins.get(k)==v for k,v in calendar_hashes.items()),'Bound calendar source closure')
    require(plan['calendar']==oldplan['calendar'],'Original calendar provenance equality')
    training=r.json(inputs['training_meta']);cached=r.npz(inputs['train_inputs']);source=r.npz(TRAIN_SOURCE+'/DATA.npz');arch=r.json(ARCH+'/META.json')
    ident=identity_oracle(meta,training,cached,source,arch);compare(r.json(p/'IDENTITY_AUDIT.json'),ident,'IDENTITY_AUDIT');np.testing.assert_array_equal(cached['eligible'],a['eligible'])
    symbols=np.array([s for row in training['training']['symbols']for s in row]);event_symbols=np.array([s for row in meta['symbols']for s in row])
    expected=labels_oracle(px,op,sessions,cached['dates'],symbols,cached['common_good'],cached['train_eligible'],np.isfinite(cached['train_plain']).all(axis=1))
    saved_labels=r.npz(p/'LABELS.npz');require(set(saved_labels)==set(expected['arrays']),'Label array schema')
    for k,v in expected['arrays'].items():
        require(saved_labels[k].shape==v.shape and saved_labels[k].dtype==v.dtype,'Label schema '+k)
        if k.startswith('target_'):np.testing.assert_allclose(saved_labels[k],v,rtol=0,atol=ATOL,equal_nan=True)
        else:np.testing.assert_array_equal(saved_labels[k],v)
    compare(r.json(p/'LABEL_AUDIT.json'),expected['audit'],'LABEL_AUDIT')
    eg=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1);fit_cache={}
    with threadpool_limits(limits=1):forecast=model_oracle(expected['arrays'],cached['train_plain'],cached['event_dates'],event_symbols,cached['event_plain'],eg,fit_cache=fit_cache)
    compare(r.json(p/'MODEL_AUDIT.json'),forecast['audit'],'MODEL_AUDIT')
    error=0.
    for name in EXTRA:
        if name in ('paired_available','paired_local_available'):v=forecast[name[len('paired_'):]].reshape(122,49)
        else:
            _,group,arm=name.split('_');v=forecast[group][arm].reshape(122,49)
        require(a[name].shape==v.shape and a[name].dtype==v.dtype,'Forecast schema '+name)
        if v.dtype.kind=='b':np.testing.assert_array_equal(a[name],v)
        else:
            np.testing.assert_allclose(a[name],v,rtol=0,atol=ATOL,equal_nan=True);finite=np.isfinite(v)
            if finite.any():error=max(error,float(np.max(np.abs(a[name][finite]-v[finite]))))
    independent_arrays,scores,choices,rank_audit=independent_forecast_decisions(a,meta,forecast);compare(r.json(p/'RANK_AUDIT.json'),rank_audit,'RANK_AUDIT');saved_scores=r.npz(p/'SCORES.npz');orders=r.json(p/'PRETRAIN_ORDERS.json');require(set(saved_scores)==set(orders)==set(NAMES),'Three arms')
    for name in NAMES:
        wanted=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j]for j in choice],scores=[float(scores[name][i,j])for j in choice])for i,(day,choice)in enumerate(zip(meta['dates'],choices[name]))]
        compare(orders[name],wanted,'ORDERS/'+name);np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=ATOL)
    h.assert_exact_tree(orders[NAMES[0]],r.json(inputs['original_orders']));require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Baseline score bytes')
    # Every score prefix is rebuilt. Forecast prefixes use only matured rows and
    # truncate future query/price data; pooled fits are memoized by full inputs.
    for n in range(123):
        ss,cc,_=scores_oracle(independent_arrays,meta,n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'All score/order prefixes')
        if n==0:continue
        cutoff=pd.Timestamp(meta['dates'][n-1]);labels_prefix=labels_oracle(px.loc[:cutoff],op.loc[:cutoff],sessions,cached['dates'],symbols,cached['common_good'],cached['train_eligible'],np.isfinite(cached['train_plain']).all(axis=1))
        with threadpool_limits(limits=1):ff=model_oracle(labels_prefix['arrays'],cached['train_plain'],cached['event_dates'][:n*49],event_symbols[:n*49],cached['event_plain'][:n*49],eg[:n*49],fit_cache=fit_cache)
        for k in ARMS:
            np.testing.assert_allclose(ff['raw'][k],forecast['raw'][k][:n*49],rtol=0,atol=ATOL,equal_nan=True)
            np.testing.assert_allclose(ff['pooled'][k],forecast['pooled'][k][:n*49],rtol=0,atol=ATOL,equal_nan=True)
        np.testing.assert_array_equal(ff['available'],forecast['available'][:n*49]);np.testing.assert_array_equal(ff['local_available'],forecast['local_available'][:n*49])
    missing=dict(independent_arrays,paired_available=np.zeros((122,49),bool));ss,cc,_=scores_oracle(missing,meta)
    for name in NAMES[1:]:require(cc[name]==choices[NAMES[0]]and ss[name].tobytes()==scores[NAMES[0]].tobytes(),'Exact whole-month incumbent fallback')
    required={str(p/k):r.sha(p/k)for k in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for k,v in list(r.seen.items()):require(r.sha(k)==v and allowed.get(k)==v,'Unbound consumed artifact '+k)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),producer_modules_imported=False,returns_or_nav_computed=False,orders_checked=sum(len(v)for v in orders.values()),months_checked=len(meta['dates']),candidate_cells_checked=int(a['baseline'].size),training_rows_checked=len(symbols),forecast_arrays_verified=len(EXTRA),prefixes_checked=len(meta['dates'])+1,new_model_fits=forecast['audit']['new_pooled_model_fits'],independent_numerical_solver='augmented weighted least squares; no producer imports',independent_forecast_ranks_and_orders_verified=True,max_forecast_error=error,price_parity=parity,limitations=['Historical sample reused, not pristine out of sample.','Inherited proxy universe and adjusted-price vintage limitations remain.','This verifies preparation, not economic efficacy or returns.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha',required=True);p.add_argument('--prepare-complete',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    require(args.prepare_complete,'Run only after registered producer completion and root authorization')
    result=audit(args.root,args.plan or args.root/STUDY/'PLAN.json',args.plan_sha);out=args.output or args.root/STUDY/'INDEPENDENT_PREPARE.json'
    with out.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__=='__main__':main()
