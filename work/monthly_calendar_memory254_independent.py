"""Independent254 month-end memory source, maturity, estimation and selector proof.

Only frozen independent selection/accounting helpers are imported. No254
producer, source loader, cached market data, or strategy execution is imported.
"""
import copy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BRIDGE=ROOT/'work/monthly_leader_bridge253_independent.py'
BRIDGE_SHA='6a594103bdeadeec381a66faca3ae0764549cbe7ca5f3de6df8f1cf7b8cfcd92'
EP=ROOT/'outputs/monthly_daily_episodes/episodes.py'
EP_SHA='8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a'
def require(ok,message):
    if not ok:raise ValueError(message)
require(hashlib.sha256(BRIDGE.read_bytes()).hexdigest()==BRIDGE_SHA,'Frozen independent bridge changed')
_spec=importlib.util.spec_from_file_location('calendar254_independent_bridge',BRIDGE)
bridge=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(bridge)
exact_array,record_check=bridge.exact_array,bridge.record_check
ARMS=('incumbent_local67','calendar_memory_local67','fixed21_memory_control')
FRAME_KEYS=('adj','close','high','low','volume','open')
DATA_KEYS=('event_symbol','event_dates','event_anchor_index','event_signal','event_entry','event_exit21',
 'event_exit_calendar','event_common_maturity','event_valid21','event_valid_calendar','event_common_valid',
 'event_excess21','event_excess_calendar','event_utility21','event_utility_calendar','query_dates','query_symbols',
 'query_shape','query_market','query_signal','query_vol')
SOURCE_COUNT_FIELDS=('queries','symbols_per_query','securities','anchors','valid21_anchors','valid_calendar_anchors',
 'common_valid_anchors','common_invalid_anchors','initial_partial_queries','formal_queries','query_cells',
 'finite_query_cells','positive_signal_query_cells','new_model_fits','predictive_model_fits')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells',
 'query_signal_cells','query_finite_cells','calendar_nonzero_effect_cells','fixed21_nonzero_effect_cells',
 'calendar_evidence_cells','fixed21_evidence_cells','candidate_changed_holdings_events','control_changed_holdings_events',
 'candidate_vs_control_changed_holdings_events','original_episode_calls','candidate_episode_calls','control_episode_calls',
 'anchors','mature_common_anchors','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','original_eligible_indices','pre_order','boundary_candidates',
 'episode_budget','incumbent_choices','candidate_choices','control_choices')
SOURCE_FLAGS=('passed','official_monthend_anchors_checked','no_anchor_lockout_checked','original_causal_features_checked',
 'query_features_checked','next_adjusted_open_checked','fixed21_exit_checked','next_calendar_month_exit_checked',
 'full_official_paths_checked','separate_path_validity_checked','maximum_common_maturity_checked',
 'source_counts_checked','no_future_numeric_source_consumption')
INCUMBENT_FLAGS=bridge.INCUMBENT_GATE_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_eligibility_unchanged','original_core_unchanged',
 'original_support_and_budget_checked','common_mature_history_checked','same_security_signal_only_checked',
 'five_year_history_checked','decay_prior_neff_and_one_se_checked','shared_sqrt21_normalizer_checked',
 'exact_effects_checked','single_episode_per_arm_checked','top_two_locked','boundary_rank3_to5_checked',
 'no_final_global_rerank','named_initial_partial_unchanged','two_names_and_cash_checked',
 'exact_scores_and_choices_checked','full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked')
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
PATHS,ACCOUNTING_SHA=bridge.PATHS,bridge.ACCOUNTING_SHA

def dependency_files():
    require(hashlib.sha256(EP.read_bytes()).hexdigest()==EP_SHA,'Canonical episode source changed')
    return tuple(sorted(set(bridge.dependency_files())|{BRIDGE,EP,Path(__file__).resolve()}))
def accounting():return bridge.accounting()
def original_base(original,meta,n):return bridge.original_base(original,meta,n)

def _day(value):
    try:d=pd.Timestamp(value)
    except Exception as exc:raise ValueError('Invalid calendar date') from exc
    require(not pd.isna(d) and d.tz is None and d==d.normalize(),'Unambiguous daily date')
    return d

def _queries(meta,n=None):
    require(type(meta) is dict and all(k in meta for k in ('dates','symbols')),'Metadata fields')
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']) and len(meta['symbols'])>=n,'Query prefix')
    days=[_day(v) for v in meta['dates'][:n]];names=[]
    require(all(a<b for a,b in zip(days,days[1:])),'Increasing query dates')
    require(all(d!=pd.Timestamp('2016-08-17') or i==0 for i,d in enumerate(days)),'Named initial partial only first')
    for ss in meta['symbols'][:n]:
        require(isinstance(ss,(list,tuple)) and len(ss)==49 and len(set(ss))==49,'Exactly49 unique query names')
        require(all(type(s) is str and s and s.strip()==s and len(s)<=64 for s in ss),'Canonical symbols')
        names.append(list(ss))
    return n,days,names

def _empty_data(n):
    out={}
    for k in DATA_KEYS:
        if k.startswith('query_'):continue
        dtype='U64' if k=='event_symbol' else 'datetime64[D]' if k in ('event_dates','event_entry','event_exit21','event_exit_calendar','event_common_maturity') else bool if 'valid' in k else np.int64 if k in ('event_anchor_index','event_signal') else np.float64
        out[k]=np.empty(0,dtype=dtype)
    out.update(query_dates=np.empty(n,dtype='datetime64[D]'),query_symbols=np.empty((n,49),dtype='U64'),
      query_shape=np.empty((n,49,8),dtype=np.float64),query_market=np.empty((n,3),dtype=np.float64),
      query_signal=np.empty((n,49),dtype=np.int64),query_vol=np.empty((n,49),dtype=np.float64))
    return out

def _source_frames(panel,calendar,days,names):
    raw=calendar['sessions'] if type(calendar) is dict else calendar
    require(isinstance(raw,(list,tuple,pd.DatetimeIndex,np.ndarray)),'Official calendar sequence')
    dates=[_day(x) for x in raw]
    require(all(x<y for x,y in zip(dates,dates[1:])),'Unique ordered official sessions')
    require(type(panel) is dict and all(k in panel for k in FRAME_KEYS),'All six OHLCV frames required')
    frame=panel['adj'];require(isinstance(frame,pd.DataFrame),'Price frame')
    index=pd.DatetimeIndex([_day(d) for d in frame.index]);cols=list(frame.columns)
    require(index.is_unique and index.is_monotonic_increasing and len(set(cols))==len(cols),'Unique source axes')
    require(all(type(c) is str and c for c in cols),'Source symbol axes')
    for k in FRAME_KEYS:
        f=panel[k];require(isinstance(f,pd.DataFrame) and f.index.equals(frame.index) and f.columns.equals(frame.columns),'Identical six-frame axes')
    if not days:return {},dates
    known=set(dates);require(all(d in known for d in index),'Source rows on official calendar')
    require(all(d in known and d in index for d in days),'Query official row present')
    for d in days:
        t=dates.index(d);require(t+1<len(dates),'Query next-session witness')
        require(d==pd.Timestamp('2016-08-17') or dates[t+1].to_period('M')!=d.to_period('M'),'Formal month-end query')
    needed=sorted(set(s for ss in names for s in ss)|{'SPY','QQQ'})
    require(set(needed)<=set(cols),'Required source columns present')
    past=index<=days[-1];out={}
    for k in FRAME_KEYS:
        source=panel[k].loc[past,needed];values=np.empty(source.shape,dtype=np.float64)
        for ij,v in np.ndenumerate(source.to_numpy(dtype=object)):
            if v is None or v is pd.NA:values[ij]=np.nan;continue
            require(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)),'OHLCV real nonboolean source')
            try:values[ij]=float(v)
            except (TypeError,ValueError,OverflowError) as exc:raise ValueError('OHLCV float conversion') from exc
        # Mark absent official sessions as missing, never compress a rolling clock.
        grid=pd.DatetimeIndex([d for d in dates if index[0]<=d<=days[-1]])
        out[k]=pd.DataFrame(values,index=index[past],columns=needed).reindex(grid)
    return out,dates

def causal_features(panel):
    """Independent spelling of the frozen pandas historical feature equations."""
    p,c,h,l,v=(panel[k] for k in ('adj','close','high','low','volume'))
    ret=p.pct_change(fill_method=None);vol=ret.rolling(63).std();span=(h-l).replace(0,np.nan)
    tr=pd.DataFrame(np.maximum.reduce([span.to_numpy(),(h-c.shift()).abs().to_numpy(),(l-c.shift()).abs().to_numpy()]),index=p.index,columns=p.columns)
    atr=tr.rolling(21).mean()/c;momentum=p.shift(42)/p.shift(252)-1
    location=(2*c-h-l)/span;money=(location*v).rolling(21).sum()/v.rolling(21).sum().replace(0,np.nan)
    logs=np.log(p/p.shift());efficiency=np.log(p.shift(42)/p.shift(294))/logs.abs().rolling(252).sum().shift(42).replace(0,np.nan)
    signed=(np.sign(ret)*v).rolling(63).sum()/v.rolling(63).sum().replace(0,np.nan)
    shape=[np.tanh(momentum),np.tanh(momentum/(vol*np.sqrt(252)).replace(0,np.nan)),(vol*np.sqrt(252)).clip(0,1),
      np.tanh(p.pct_change(5,fill_method=None)/atr/3),location.clip(-1,1),money.clip(-1,1),efficiency.clip(-1,1),signed.clip(-1,1)]
    ma=p.SPY.rolling(200).mean()
    market=pd.DataFrame(dict(trend=(2*(p.SPY>ma).astype(float)-1).where(ma.notna()),
      volatility=(vol.SPY*np.sqrt(252)/.30).clip(0,2)-1,
      growth=np.tanh((p.QQQ.pct_change(126,fill_method=None)-p.SPY.pct_change(126,fill_method=None))/.20)))
    weak=p.pct_change(5,fill_method=None)<(-.5*atr);lowclose=(c-l)/span<=.25
    known=atr.notna()&p.shift(5).notna()&span.notna()
    signal=(weak.astype(int)+2*lowclose.astype(int)).where(known,-1)
    good=p.notna().rolling(295).sum().eq(295)&momentum.gt(0)
    for f in shape:good &= np.isfinite(f)
    good &= market.notna().all(axis=1).to_numpy()[:,None]
    return shape,market,signal,good,vol

def path_label(panel,sessions,symbol,entry,exit_day,positions=None):
    """Official inclusive path; absent rows invalidate instead of compressing time."""
    positions={d:i for i,d in enumerate(sessions)} if positions is None else positions
    required=sessions[positions[entry]:positions[exit_day]+1]
    p=panel['adj']
    if not all(d in p.index for d in required):return np.nan,np.nan
    asset=p.loc[required,symbol].to_numpy(dtype=float);market=p.loc[required,'SPY'].to_numpy(dtype=float)
    with np.errstate(all='ignore'):
        op=p.at[entry,symbol]*panel['open'].at[entry,symbol]/panel['close'].at[entry,symbol]
        mop=p.at[entry,'SPY']*panel['open'].at[entry,'SPY']/panel['close'].at[entry,'SPY']
    if not (np.isfinite(asset).all() and np.isfinite(market).all() and (asset>0).all() and (market>0).all()
      and np.isfinite(op) and op>0 and np.isfinite(mop) and mop>0):return np.nan,np.nan
    with np.errstate(all='ignore'):
        returns=asset/op-1
        excess=returns[-1]-(market[-1]/mop-1)
        adverse=max(0.,-float(returns.min()))
        return float(np.clip(excess,-1,1)),float(np.clip(excess-.5*adverse,-1,1))

def independent_source(panel,calendar,meta,n=None):
    n,days,names=_queries(meta,n);p,sessions=_source_frames(panel,calendar,days,names)
    data=_empty_data(n)
    if not n:return data,source_audit(data,meta,n)
    with np.errstate(all='ignore'):shape,market,signal,good,vol=causal_features(p)
    events={k:[] for k in DATA_KEYS if k.startswith('event_')};positions={d:i for i,d in enumerate(sessions)}
    month_days={};months=[d.to_period('M') for d in sessions]
    for day,month in zip(sessions,months):month_days.setdefault(month,[]).append(day)
    monthends={sessions[i] for i in range(len(sessions)-1) if months[i]!=months[i+1]}
    for symbol in sorted(set(s for ss in names for s in ss)):
        for day in p['adj'].index:
            i=positions[day]
            require(i+1<len(sessions),'Anchor calendar witness')
            if day not in monthends or not bool(good.at[day,symbol]) or signal.at[day,symbol]<=0:continue
            require(i+21<len(sessions),'Fixed21 official exit exists')
            month=months[i]+1;nextdays=month_days.get(month,[])
            require(nextdays and nextdays[-1] in monthends,'Complete next-month calendar exit')
            entry=sessions[i+1];end21=sessions[i+21];endcal=nextdays[-1]
            x21,u21=path_label(p,sessions,symbol,entry,end21,positions);xc,uc=path_label(p,sessions,symbol,entry,endcal,positions)
            v21=bool(np.isfinite(x21) and np.isfinite(u21));vc=bool(np.isfinite(xc) and np.isfinite(uc))
            row=(symbol,day,i,int(signal.at[day,symbol]),entry,end21,endcal,max(end21,endcal),v21,vc,v21 and vc,x21,xc,u21,uc)
            for k,vv in zip(events,row):events[k].append(vv)
    for k,vals in events.items():data[k]=np.asarray(vals,dtype=data[k].dtype)
    for i,(day,ss) in enumerate(zip(days,names)):
        data['query_dates'][i]=day.to_datetime64();data['query_symbols'][i]=ss
        data['query_shape'][i]=np.column_stack([f.loc[day,ss] for f in shape]);data['query_market'][i]=market.loc[day]
        data['query_signal'][i]=signal.loc[day,ss].to_numpy(dtype=np.int64);data['query_vol'][i]=vol.loc[day,ss]
    return data,source_audit(data,meta,n)

def source_audit(data,meta,n):
    n,days,names=_queries(meta,n);finite=query_finite(data)
    partial=sum(d==pd.Timestamp('2016-08-17') for d in days);a=len(data['event_dates'])
    counts=dict(zip(SOURCE_COUNT_FIELDS,(n,49,len(set(s for ss in names for s in ss)),a,
      int(data['event_valid21'].sum()),int(data['event_valid_calendar'].sum()),int(data['event_common_valid'].sum()),
      a-int(data['event_common_valid'].sum()),partial,n-partial,n*49,int(finite.sum()),int((data['query_signal']>0).sum()),0,0)))
    return dict(schema='monthly_calendar_memory254_features_v1',counts=counts,
      anchor_rule='official_monthends_original_good_positive_signal_no_lockout',common_maturity_rule='maximum_of_actual_exits',
      numeric_cutoff=str(days[-1].date()) if n else None,candidate_selection_computed=False,portfolio_returns_computed=False)

def query_finite(data):
    return np.isfinite(data['query_shape']).all(axis=2)&np.isfinite(data['query_market']).all(axis=1)[:,None]&np.isfinite(data['query_vol'])

def verify_source(panel,calendar,meta,data,audit,n=None):
    n=len(meta['dates']) if n is None else n;expected,wanted=independent_source(panel,calendar,meta,n)
    require(type(data) is dict and set(data)==set(DATA_KEYS),'All21 source arrays')
    for k in DATA_KEYS:exact_array(data[k],expected[k],'Independent source/'+k)
    record_check(audit,wanted,'Complete source audit')
    return dict(**{k:True for k in SOURCE_FLAGS},checked_events=n,source_audit=wanted,counts=wanted['counts'],
      full_source_audit_supplied=True,episode_calls=0,memory_effects_computed=False,candidate_selection_computed=False,
      producer_modules_imported=False,computes_returns=False,continuous_atol=CONTINUOUS_ATOL,
      continuous_rtol=CONTINUOUS_RTOL,exact_discrete=True,numerical_tolerance=0.)

def data_prefix(data,meta,n=None):
    n,days,names=_queries(meta,n)
    require(type(data) is dict and set(data)==set(DATA_KEYS),'Complete source cache schema')
    dates=data['event_dates'];symbols=data['event_symbol']
    require(isinstance(dates,np.ndarray) and dates.ndim==1 and dates.dtype==np.dtype('datetime64[D]'),'Event date dtype')
    count=len(dates);require(isinstance(symbols,np.ndarray) and symbols.shape==(count,) and symbols.dtype==np.dtype('U64'),'Event symbols')
    union=set(s for ss in names for s in ss)
    chosen=np.flatnonzero((dates<=np.datetime64(days[-1],'D'))&np.isin(symbols,list(union))) if n else np.empty(0,dtype=int)
    out={}
    for k in DATA_KEYS:
        a=data[k];require(isinstance(a,np.ndarray),k+' ndarray')
        if k.startswith('event_'):require(a.shape==(count,),k+' length');out[k]=a[chosen].copy()
        else:require(len(a)>=n,k+' prefix');out[k]=a[:n].copy()
    for k in ('event_entry','event_exit21','event_exit_calendar','event_common_maturity'):
        require(out[k].dtype==np.dtype('datetime64[D]') and not np.isnat(out[k]).any(),k+' daily dates')
    for k in ('event_valid21','event_valid_calendar','event_common_valid'):require(out[k].dtype==np.dtype(bool),k+' bool')
    for k in ('event_signal','event_anchor_index','query_signal'):require(out[k].dtype==np.dtype('int64'),k+' int64')
    require(out['query_shape'].shape==(n,49,8) and out['query_market'].shape==(n,3) and out['query_vol'].shape==(n,49),'Query shapes')
    require(out['query_symbols'].shape==(n,49) and out['query_symbols'].dtype==np.dtype('U64'),'Query names shape')
    require(out['query_signal'].shape==(n,49),'Query signals shape')
    for k in ('query_shape','query_market','query_vol'):require(out[k].dtype==np.dtype('float64'),k+' float64')
    exact_array(out['query_dates'],np.asarray(days,dtype='datetime64[D]'),'Cached query clock')
    exact_array(out['query_symbols'],np.asarray(names,dtype='U64').reshape(n,49),'Cached query identity')
    pairs=list(zip(out['event_symbol'].tolist(),out['event_dates'].tolist()))
    require(pairs==sorted(set(pairs)),'Sorted unique own anchors')
    require(((out['event_signal']>=1)&(out['event_signal']<=3)).all(),'Positive event signals')
    require(((out['query_signal']>=-1)&(out['query_signal']<=3)).all(),'Query signal range')
    require((out['event_entry']>out['event_dates']).all() and (out['event_exit21']>=out['event_entry']).all()
      and (out['event_exit_calendar']>=out['event_entry']).all(),'Ordered entry/exit clocks')
    exact_array(out['event_common_maturity'],np.maximum(out['event_exit21'],out['event_exit_calendar']),'Common maturity maximum')
    exact_array(out['event_common_valid'],out['event_valid21']&out['event_valid_calendar'],'Common validity AND')
    return out

def estimate(data,symbol,day,signal,eligible,vol,path):
    """Independent conditional level estimate; maturity filtering precedes labels."""
    none=dict(observations=0,effective_samples=0.,own_weight=0.,latest_label_end=None,raw_effect=0.,standard_error=0.,shrunk_effect=0.)
    if signal<=0 or not eligible:return 0.,none
    cutoff=np.datetime64(day,'D');earliest=cutoff-np.timedelta64(1827,'D')
    mask=(data['event_symbol']==symbol)&(data['event_dates']>=earliest)&(data['event_dates']<cutoff)&(data['event_common_maturity']<=cutoff)&data['event_common_valid']
    ids=np.flatnonzero(mask)
    if not len(ids):return 0.,none
    # Crucially no conversion or isfinite over future/unmatured label values.
    vals=[]
    for v in data['event_utility'+path][ids]:
        require(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)),'Mature utility real')
        x=float(v);require(math.isfinite(x) and -1<=x<=1,'Mature utility bounded');vals.append(x)
    y=np.asarray(vals,dtype=float)
    age=(cutoff-data['event_dates'][ids]).astype('timedelta64[D]').astype(float)/30.4375
    weights=2**(-age/24.)*(data['event_signal'][ids]==signal)
    used=weights>0;observations=int(used.sum());mass=float(weights.sum())
    if mass<=0:mean=variance=neff=0.
    else:
        mean=float(weights@y/mass);variance=float(weights@(y-mean)**2/mass);neff=mass**2/float(weights@weights)
    info=dict(none,observations=observations,effective_samples=neff,
      latest_label_end=str(data['event_common_maturity'][ids[used]].max()) if observations else None)
    if neff<3:return 0.,info
    se=np.sqrt(variance/neff);confidence=mass/(mass+6.)
    shrunk=np.sign(mean)*max(abs(mean)-se,0.)*confidence
    value=.5*np.tanh(shrunk/max(vol*np.sqrt(21),.01))
    require(np.isfinite(value),'Finite memory estimate')
    info.update(own_weight=confidence,raw_effect=float(mean),standard_error=float(se),shrunk_effect=float(shrunk))
    return float(value),info

def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    return bridge.verify_incumbent(original,meta,orders,scores,pre_scores,n)

def selection_oracle(original,meta,data,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n);require(base['width']==49,'49-name original cache')
    d=data_prefix(data,meta,n);finite=query_finite(d)
    effects={ARMS[0]:base['effects'].copy(),ARMS[1]:np.zeros((n,49)),ARMS[2]:np.zeros((n,49))}
    evidence={arm:0 for arm in ARMS[1:]}
    for i,day in enumerate(base['days']):
        if day==pd.Timestamp('2016-08-17'):
            for arm in ARMS[1:]:effects[arm][i]=base['effects'][i]
            continue
        for j,symbol in enumerate(base['names'][i]):
            for arm,path in ((ARMS[1],'_calendar'),(ARMS[2],'21')):
                value,info=estimate(d,symbol,day,int(d['query_signal'][i,j]),bool(base['E'][i,j] and finite[i,j]),d['query_vol'][i,j],path)
                effects[arm][i,j]=value
                evidence[arm]+=int(info['effective_samples']>=3)
    scores={};choices={};pres={a:base['P'].copy() for a in ARMS};preorders=boundaries=None
    for arm in ARMS:
        state=dict(base,effects=effects[arm]);scores[arm],choices[arm],po,bc=bridge.restricted_episode(base['P'],state,base['E'])
        if preorders is None:preorders,boundaries=po,bc
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=49,
      original_episode_calls=1,candidate_episode_calls=1,control_episode_calls=1,anchors=len(d['event_dates']),
      mature_common_anchors=int(((d['event_common_maturity']<=d['query_dates'][-1])&d['event_common_valid']).sum()) if n else 0,
      calendar_evidence_cells=evidence[ARMS[1]],fixed21_evidence_cells=evidence[ARMS[2]])
    events=[]
    for i,day in enumerate(base['days']):
        partial=day==pd.Timestamp('2016-08-17');counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(not partial)
        counts['original_eligible_cells']+=int(base['E'][i].sum())
        if not partial:
            counts['query_signal_cells']+=int((d['query_signal'][i]>0).sum());counts['query_finite_cells']+=int(finite[i].sum())
            counts['calendar_nonzero_effect_cells']+=int(np.count_nonzero(effects[ARMS[1]][i]));counts['fixed21_nonzero_effect_cells']+=int(np.count_nonzero(effects[ARMS[2]][i]))
        for key,left,right in (('candidate_changed_holdings_events',ARMS[1],ARMS[0]),('control_changed_holdings_events',ARMS[2],ARMS[0]),('candidate_vs_control_changed_holdings_events',ARMS[1],ARMS[2])):
            counts[key]+=int(set(choices[left][i])!=set(choices[right][i]))
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.isfinite(remaining).all() and (remaining>=.38-1e-12).all(),'Original budget including calm extension')
        events.append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,original_eligible_indices=np.flatnonzero(base['E'][i]).tolist(),
          pre_order=preorders[i],boundary_candidates=boundaries[i],episode_budget=float(base['budget'][i]),incumbent_choices=choices[ARMS[0]][i],
          candidate_choices=choices[ARMS[1]][i],control_choices=choices[ARMS[2]][i]))
    return scores,choices,pres,effects,dict(schema='monthly_calendar_memory254_decisions_v1',counts=counts,events=events)

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],
      scores=[float(scores[arm][i,j]) for j in ids]) for i,ids in enumerate(choices[arm])] for arm in ARMS}

def verify_selection(original,meta,data,orders,*,scores,pre_scores,effects,audit,n=None):
    n=len(meta['dates']) if n is None else n
    sc,ch,pr,ef,expected=selection_oracle(original,meta,data,n)
    record_check(orders,orders_from_choices(sc,ch,meta),'All ordered254 selections')
    for actual,wanted,label in ((scores,sc,'Full scores'),(pre_scores,pr,'Original pre'),(effects,ef,'Memory effects')):
        require(type(actual) is dict and set(actual)==set(ARMS),label+' three arms')
        for arm in ARMS:exact_array(actual[arm],wanted[arm],label+'/'+arm)
    record_check(audit,expected,'Complete254 decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},checked_events=n,ordered_choices_checked=3*n,counts=expected['counts'],
      full_scores_supplied=True,full_pre_scores_supplied=True,full_effects_supplied=True,full_audit_supplied=True,
      producer_modules_imported=False,computes_returns=False,numerical_tolerance=0.)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three254 accounting arms')
    mapping=dict(zip(ARMS,bridge.ARMS));rename=lambda d:{mapping[k]:v for k,v in d.items()}
    result=bridge.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(result,paths={arm:result['paths'][mapping[arm]] for arm in ARMS})
