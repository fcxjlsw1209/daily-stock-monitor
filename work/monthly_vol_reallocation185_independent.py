"""Independent round185 source, allocation and twelve-path accounting oracle.

No production strategy/kernel import, feature entrypoint or file mutation.
Full-history rolling is the pinned original algorithm; percentile and allocation
arithmetic are rebuilt with scalar comparisons independent of production.
"""
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
LAMBDAS = dict(vol_reallocate_0025=.025, vol_reallocate_0050=.05, vol_reallocate_0100=.10)
ARMS = ('incumbent_local67',) + tuple(LAMBDAS)
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.), ('next_open_cost', 'open', .0005))


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING)==ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec=importlib.util.spec_from_file_location('reallocation185_accounting',ACCOUNTING)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(sha(OUTCOME_HELPER)==OUTCOME_SHA,'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING)==ACCOUNTING_SHA,'Frozen independent accounting changed')
    spec=importlib.util.spec_from_file_location('reallocation185_isolated_outcome',OUTCOME_HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.ARMS==('incumbent_local67','price_activity_distribution','price_only_distribution'),'Unexpected old helper')
    module.ARMS=ARMS
    result=module.audit_outcomes(px,opening,orders,exports,results,meta)
    # The frozen helper loops over ARMS but its final 9/45 counters are static.
    # Derive counts only from the fully audited result map, including arm four.
    require(set(result['paths'])==set(ARMS),'Missing audited arm')
    for paths in result['paths'].values():
        require(set(paths)=={p[0] for p in PATHS},'Missing audited path')
        for record in paths.values():
            require(record['passed'] is True and set(record['slices'])==
                    {'full122','first88','early53','recent35','later34'},'Missing audited slice')
    result['checked_paths']=sum(len(paths) for paths in result['paths'].values())
    result['checked_slices']=sum(len(record['slices']) for paths in result['paths'].values() for record in paths.values())
    require((result['checked_paths'],result['checked_slices'])==(12,60),'Wrong actual outcome coverage')
    result['outcome_helper_sha256']=OUTCOME_SHA
    return result


def close(actual,expected,label,atol=1e-12):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    require(a.shape==b.shape,label+': shape')
    require(np.array_equal(np.isnan(a),np.isnan(b)) and not np.isinf(a).any() and not np.isinf(b).any(),label+': finite mask')
    valid=np.isfinite(b)
    require(np.all(np.abs(a[valid]-b[valid])<=atol),label+': values')


def record_check(actual,expected,label='audit'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict),label+': mapping')
        for key,value in expected.items():
            require(key in actual,label+': missing '+key);record_check(actual[key],value,label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    elif isinstance(expected,float):
        require(type(actual) in (int,float),label+': scalar');close(actual,expected,label)
    else:require(type(actual) is type(expected) and actual==expected,label+': differs')


def iso(text):
    require(isinstance(text,str),'ISO date string required')
    day=date.fromisoformat(text);require(day.isoformat()==text,'Strict ISO date required')
    return pd.Timestamp(day)


def schema(panel,meta,calendar):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Panel keys')
    frame=panel['adj']
    require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),'Adjusted frame required')
    idx=frame.index
    require(len(idx)>0 and not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and
            idx.tz is None and idx.equals(idx.normalize()),'Source date axis')
    require(frame.columns.is_unique and 'SPY' in frame and all(isinstance(s,str) and s and s.strip()==s for s in frame),'Source identities/SPY')
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','symbols','etfs')),'Metadata fields')
    require(isinstance(meta['etfs'],(list,tuple,set,frozenset)) and len(meta['etfs'])==len(set(meta['etfs'])) and
            all(isinstance(s,str) and s and s.strip()==s for s in meta['etfs']),'ETF identities')
    require(len(meta['dates'])==len(meta['symbols']),'Event axis')
    days=[iso(d) for d in calendar]
    require(days and all(d.weekday()<5 for d in days) and all(a<b for a,b in zip(days,days[1:])),'Calendar sessions/order')
    lookup={d:i for i,d in enumerate(days)};events=[];width=0
    for text,names in zip(meta['dates'],meta['symbols']):
        day=iso(text)
        require(day in lookup and (not events or day>events[-1][0]),'Event session/order')
        require(isinstance(names,(list,tuple)) and names and len(names)==len(set(names)) and
                all(isinstance(s,str) and s and s.strip()==s for s in names),'Event identities')
        require(not events or len(names)==width,'Event widths')
        width=len(names);events.append((day,list(names),lookup[day]))
    return days,events,width


def matrix(value,shape,name):
    a=np.asarray(value)
    require(a.shape==shape and np.issubdtype(a.dtype,np.number) and a.dtype.kind!='c',name+': numeric matrix')
    require(np.isfinite(a).all(),name+': nonfinite')
    return a


def state(value,rows,name):
    a=np.asarray(value);require(a.shape==(rows,) and a.dtype==bool,name+': boolean state')
    return a


def broadcast(value,shape,name):
    a=np.asarray(value)
    require(np.issubdtype(a.dtype,np.number) and a.dtype.kind!='c',name+': numeric')
    if a.shape==(shape[0],):a=a[:,None]
    require(a.shape in ((shape[0],1),shape),name+': coefficient shape')
    out=np.broadcast_to(a,shape);require(np.isfinite(out).all(),name+': finite')
    return out


def prefix_matrix(value,shape,n,name):
    a=np.asarray(value)
    require(a.shape==shape and not np.iscomplexobj(a),name+': numeric matrix shape')
    try:out=np.asarray(a[:n],float)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError(name+': numeric prefix') from exc
    require(np.isfinite(out).all(),name+': nonfinite prefix')
    return out


def prefix_weight(value,shape,n,name):
    a=np.asarray(value)
    require(not np.iscomplexobj(a),name+': numeric coefficient')
    if a.shape==(shape[0],):a=a[:,None]
    require(a.shape in ((shape[0],1),shape),name+': coefficient shape')
    try:out=np.broadcast_to(np.asarray(a[:n],float),(n,shape[1]))
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError(name+': numeric prefix') from exc
    require(np.isfinite(out).all(),name+': nonfinite prefix')
    return out


def inputs(original,source,rows,width,limit=None):
    shape=(rows,width)
    n=rows if limit is None else limit
    require(type(n) is int and 0<=n<=rows,'Original array prefix')
    require(isinstance(original,dict) and all(k in original for k in ('baseline','core','core_weight','state_calm10')),'Original arrays')
    require(isinstance(source,dict) and {'vol','vol_weight','state_highvol'}<=set(source)<=
            {'vol','vol_weight','state_highvol','core','core_weight'},'Source quality fields')
    baseline=prefix_matrix(original['baseline'],shape,n,'baseline');core=prefix_matrix(original['core'],shape,n,'core')
    cw=prefix_weight(original['core_weight'],shape,n,'core_weight');calm=state(original['state_calm10'],rows,'calm')[:n]
    require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,width))),'Original core coefficient/state')
    vol=prefix_matrix(source['vol'],shape,n,'vol');require(np.all(np.abs(vol)<=.5+1e-12),'Vol rank bounds')
    high=state(source['state_highvol'],rows,'highvol')[:n];vw=prefix_weight(source['vol_weight'],shape,n,'vol_weight')
    expected=np.broadcast_to(np.where(calm,.2,.219375*(1-np.where(high,.1,0.)/.48))[:,None],(n,width))
    require(np.array_equal(vw,expected),'Original volatility coefficient/state')
    require(np.all(np.abs(core/cw)<=.5+1e-12),'Original core bounds')
    support=(baseline-core)/(1.-cw)
    require(np.isfinite(support).all() and np.all(np.abs(support)<=.5+1e-12),'Original support bounds')
    if 'core' in source:require(np.array_equal(prefix_matrix(source['core'],shape,n,'source core'),core),'Quality original core parity')
    if 'core_weight' in source:require(np.array_equal(prefix_weight(source['core_weight'],shape,n,'source core_weight'),cw),'Quality original coefficient parity')
    return baseline,core,cw,vol,vw,calm,high,support


def ranks(raw):
    ids=[j for j,x in enumerate(raw) if math.isfinite(float(x))];out=np.zeros(len(raw))
    for j in ids:
        below=sum(raw[k]<raw[j] for k in ids);equal=sum(raw[k]==raw[j] for k in ids)
        out[j]=(below+(equal+1.)/2.)/len(ids)-.5
    return out


def raw_source(panel,events,sessions):
    last=events[-1][0];needed={'SPY'}|{s for _,ss,_ in events for s in ss}
    frame=panel['adj'].loc[:last,[s for s in panel['adj'] if s in needed]].copy(deep=True)
    require(len(frame)>0 and all(d.weekday()<5 for d in frame.index),'Source prefix weekday/history')
    official=pd.DatetimeIndex(sessions);known=frame.index[frame.index>=official[0]]
    require(known.isin(official).all(),'Source date not in official calendar')
    early=frame.index[frame.index<official[0]]
    axis=early.append(official[(official>=max(frame.index[0],official[0]))&(official<=last)])
    try:frame=frame.reindex(axis).astype(float)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Nonnumeric original prices') from exc
    a=frame.to_numpy();require(not np.isinf(a).any() and not np.any(np.isfinite(a)&(a<=0.)),'Nonpositive or infinite prices')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):daily=frame.pct_change(fill_method=None)
    require(not np.isinf(daily.to_numpy()).any(),'Simple-return arithmetic overflow')
    vol=daily.rolling(126).std(ddof=1);spy=daily['SPY'].rolling(63).std(ddof=1)
    complete=daily.notna().rolling(126).sum()==126
    spy_complete=daily['SPY'].notna().rolling(63).sum()==63
    require(np.isfinite(vol.to_numpy()[complete.to_numpy()]).all(),'Complete volatility calculation nonfinite')
    require(np.isfinite(spy.to_numpy()[spy_complete.to_numpy()]).all(),'Complete SPY volatility calculation nonfinite')
    return vol,spy*np.sqrt(252)<.10,spy>spy.rolling(252).median(),daily


def scalar_unit(b,c,k,v):
    ans=(1.-float(k))*float(v)-(float(b)-float(c))
    require(math.isfinite(ans),'Nonfinite budget reallocation')
    return ans


def independent_features(panel,meta,calendar,original,source,limit=None):
    sessions,events,width=schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0<=limit<=len(events),'Invalid prefix')
    b,c,k,v,vw,calm,high,support=inputs(original,source,len(events),width,limit=limit)
    events=events if limit is None else events[:limit];n=len(events)
    values=dict(vol_rank=np.zeros((n,width)),unit_delta=np.zeros((n,width)))
    rows=[];finite_count=0
    raw_vol,raw_calm,raw_high,daily=raw_source(panel,events,sessions) if n else (None,None,None,None)
    for i,(day,names,pos) in enumerate(events):
        require(bool(raw_calm.at[day])==bool(calm[i]) and bool(raw_high.at[day])==bool(high[i]),'Original market state parity')
        raw=raw_vol.loc[day].reindex(names).to_numpy(float);domain=np.isfinite(raw);rebuilt=ranks(raw)
        # Independently validate each canonical pandas statistic from scalar
        # returns, retaining original rolling bits for exact percentile ties.
        end=daily.index.get_loc(day)
        for j,name in enumerate(names):
            window=daily[name].iloc[max(0,end-125):end+1].to_numpy(float) if name in daily else np.array([])
            complete=len(window)==126 and all(math.isfinite(float(x)) for x in window)
            require(bool(domain[j])==complete,'Original rolling support differs')
            if complete:
                mean=math.fsum(float(x) for x in window)/126.
                sigma=math.sqrt(math.fsum((float(x)-mean)**2 for x in window)/125.)
                close(raw[j],sigma,'Independent scalar rolling volatility',atol=1e-12+1e-12*abs(sigma))
        require(np.array_equal(rebuilt,v[i]),'Original volatility rank parity')
        values['vol_rank'][i]=rebuilt
        for j in range(width):values['unit_delta'][i,j]=scalar_unit(b[i,j],c[i,j],k[i,j],rebuilt[j])
        finite_count+=int(domain.sum())
        rows.append(dict(date=str(day.date()),symbols=names,session_position=pos,calm=bool(calm[i]),highvol=bool(high[i]),
            raw_vol=[float(x) if math.isfinite(float(x)) else None for x in raw],vol_domain=domain.tolist(),
            vol_rank=rebuilt.tolist(),unit_delta=values['unit_delta'][i].tolist(),core_weight=k[i].tolist(),
            vol_weight=vw[i].tolist(),original_support=support[i].tolist(),nonvol_donor_budget=(1.-k[i]-vw[i]).tolist()))
    detail=dict(events=n,symbols_per_event=width,checked_cells=n*width,source_vol_finite_cells=finite_count,
        source_vol_missing_neutral_cells=n*width-finite_count,volatility_rank_reconstructions=n,
        predictive_model_fits=0,event_history=rows,no_new_availability_filter=True,
        ranking_eligibility_filtered=False,portfolio_returns_calculated=False)
    return values,detail


def arrays(values,limit=None):
    require(isinstance(values,dict) and set(values)=={'vol_rank','unit_delta'},'Feature keys')
    require(np.asarray(values['vol_rank']).ndim==2,'Feature matrix')
    shape=np.asarray(values['vol_rank']).shape
    n=shape[0] if limit is None else limit
    require(type(n) is int and 0<=n<=shape[0],'Feature prefix')
    out={k:prefix_matrix(v,shape,n,k) for k,v in values.items()}
    require(np.all(np.abs(out['vol_rank'])<=.5+1e-12),'Vol rank bounds')
    return out


def verify_features(panel,meta,calendar,original,source,values,audit):
    got=arrays(values);expected,detail=independent_features(panel,meta,calendar,original,source)
    require(np.array_equal(got['vol_rank'],expected['vol_rank']),'Rebuilt vol ranks exact parity')
    close(got['unit_delta'],expected['unit_delta'],'Unit delta',atol=2e-17)
    record_check(audit,detail)
    count_keys=('events','symbols_per_event','checked_cells','source_vol_finite_cells','source_vol_missing_neutral_cells','volatility_rank_reconstructions')
    return dict(passed=True,**{k:detail[k] for k in count_keys},independent_scalar_rebuild=True,
        original_raw_vol_checked=True,original_vol_rank_parity_checked=True,original_market_states_checked=True,
        original_missing_neutral_checked=True,original_scalar_vol_statistics_checked=True,
        support_budget_reallocation_checked=True,no_new_availability_filter=True,
        no_future_or_outcome_inputs=True,audit_checked=True,predictive_model_fits=0,computes_returns=False)


def independent_deltas(values,original,source,limit=None):
    original_shape=np.asarray(original['baseline']).shape
    require(len(original_shape)==2,'Original baseline matrix')
    total,width=original_shape
    rawshape=np.asarray(values['vol_rank']).shape
    require(len(rawshape)==2,'Feature matrix')
    n=total if limit is None else limit
    require(type(n) is int and 0<=n<=total and rawshape[0] in (n,total) and rawshape[1]==width,'Invalid delta prefix')
    a=arrays(values,limit=n)
    b,c,k,v,vw,calm,high,support=inputs(original,source,total,width,limit=n)
    require(np.array_equal(a['vol_rank'][:n],v[:n]),'Original V changed')
    output={arm:np.zeros((n,width)) for arm in LAMBDAS}
    for i in range(n):
        for j in range(width):
            unit=scalar_unit(b[i,j],c[i,j],k[i,j],v[i,j])
            require(a['unit_delta'][i,j]==unit,'Supplied unit formula changed')
            if i:
                for arm,dose in LAMBDAS.items():output[arm][i,j]=dose*unit
    detail=dict(events=n,width=width,lambdas=LAMBDAS.copy(),changed_cells={k:int(np.count_nonzero(x)) for k,x in output.items()},
        max_abs_delta={k:float(np.max(np.abs(x))) if x.size else 0. for k,x in output.items()},
        initial_delta_zero=True,no_new_availability_filter=True,episode_applied=False,
        original_core_unchanged=True,original_model_corrections_unchanged=True,original_support_unchanged=True)
    return output,detail


def verify_deltas(values,original,source,deltas,audit):
    require(isinstance(deltas,dict) and set(deltas)==set(LAMBDAS),'Three delta arms required')
    expected,detail=independent_deltas(values,original,source)
    for arm in LAMBDAS:
        got=matrix(deltas[arm],expected[arm].shape,arm)
        require(np.array_equal(got,expected[arm]),'Exact independent dose formula '+arm)
    record_check(audit,detail)
    return dict(passed=True,events=detail['events'],symbols_per_event=detail['width'],
        checked_cells=detail['events']*detail['width'],independent_scalar_rebuild=True,three_fixed_doses_checked=True,
        support_budget_reallocation_checked=True,initial_delta_zero=True,no_new_availability_filter=True,
        original_support_unchanged=True,no_extra_rank_or_model_filter=True,audit_checked=True,lambdas=LAMBDAS.copy(),
        episode_applications=0,predictive_model_fits=0,changed_cells=detail['changed_cells'],audit=detail)
