"""Pure original-core reconstruction and one calendar-clock full-core contrast.

Callers provide all data. No loaders, fitting, registration or performance replay.
Original control gates precede every calendar evaluation. The full historical
pandas rolling context is retained; no reset64 window or source imputation.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ADAPTER_PATH=ROOT/'outputs/monthly_core_timing70_research/core_adapter.py'
ADAPTER_SHA='b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a'
if hashlib.sha256(ADAPTER_PATH.read_bytes()).hexdigest()!=ADAPTER_SHA:raise ValueError('Changed frozen core adapter')
_spec=importlib.util.spec_from_file_location('calendar248_core_adapter',ADAPTER_PATH)
core_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(core_adapter)
ARMS=('incumbent_local67','calendar_mixed_core')
INITIAL_PARTIAL='2016-08-17'
C_SOURCE_ATOL=1e-12
C_SOURCE_RTOL=0.
POLICY=dict(budget=.05,ceiling=5,gap=.02)
EVENT_KEYS=('query_index','date','initial_partial','calm','core_weight','symbols','branch','panel_first_session',
 'query_position','next_official_session','trading_price_start','trading_price_end','risk_price_start','risk_price_end',
 'calendar_start','calendar_end','calendar_lag_sessions','u0_indices','calendar_missing_indices','core_source_max_abs_error','cells')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error',
 'sigma63','calendar_source_available','calendar_reason','calendar_start_price','calendar_end_price','calendar_raw','calendar_rank','delta')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','calm_formal_events','noncalm_formal_events',
 'original_checked_cells','original_finite_cells','calendar_required_cells','calendar_available_cells','calendar_missing_cells',
 'available_events','empty_domain_events','missing_calendar_events','candidate_changed_core_cells','changed_holdings_events',
 'original_episode_calls','new_model_fits','predictive_model_fits')
AUDIT_EVENT_KEYS=('query_index','date','branch','u0_indices','candidate_delta','extension_active_indices','episode_budget',
 'original_remaining_support','incumbent_choices','candidate_choices')
BRANCHES=('initial_partial','empty_domain','missing_calendar','available')


def dependency_files():
    return (ADAPTER_PATH,core_adapter.PATH_ADAPTER_PATH,ROOT/'outputs/monthly_local_linear70/study.py',
      ROOT/'outputs/monthly_model_scope70/study.py',ROOT/'outputs/monthly_daily_episodes/episodes.py',
      ROOT/'outputs/monthly_return_priority/return_priority.py',ROOT/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
      ROOT/'outputs/monthly_stability_research/stability.py',ROOT/'outputs/monthly_guidance_tweaks/tweaks.py',
      ROOT/'outputs/monthly_quality_research/explore.py',ROOT/'outputs/monthly_stability_monitor/base_strategy.py',
      ROOT/'outputs/monthly_stability_monitor/strategy.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def _iso(x,label):
    require(type(x) is str,label+' ISO string')
    try:d=date.fromisoformat(x)
    except (ValueError,TypeError) as exc:raise ValueError(label+' ISO date') from exc
    require(d.isoformat()==x,label+' canonical ISO');return x


def _number(x,label,finite=True):
    require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),label+' real nonboolean')
    try:v=float(x)
    except (OverflowError,ValueError,TypeError) as exc:raise ValueError(label+' binary64') from exc
    if finite:require(math.isfinite(v),label+' finite')
    return v


def _matrix(value,shape,n,label,finite=True):
    a=np.asarray(value);require(a.shape==shape,label+' shape')
    z=np.empty(a[:n].shape,float)
    for k in np.ndindex(z.shape):z[k]=_number(a[:n][k],label,finite)
    return z


def _bool(value,shape,n,label):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind=='b',label+' boolean shape');return a[:n].copy()


def _snapshot(arrays):return {k:(np.asarray(v).shape,np.asarray(v).dtype.str,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _cache(study,arrays,meta,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Input mappings')
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Prefix/width')
    total,w=shape
    require(all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','ends','symbols')),'Metadata axes')
    require(isinstance(meta.get('etfs'),(list,tuple)) and all(type(s) is str and s for s in meta['etfs']) and len(set(meta['etfs']))==len(meta['etfs']),'ETF identities')
    prev=None
    for i in range(n):
        d=_iso(meta['dates'][i],'query');end=_iso(meta['ends'][i],'end');names=meta['symbols'][i]
        require(end>d and (prev is None or d>prev),'Chronology');prev=d
        require(d!=INITIAL_PARTIAL or i==0,'Named partial first')
        require(isinstance(names,list) and len(names)==w and len(set(names))==w and all(type(s) is str and s and s.strip()==s for s in names),'Symbols')
    base=_matrix(arrays['baseline'],shape,n,'baseline');core=_matrix(arrays['core'],shape,n,'core');effects=_matrix(arrays['effects'],shape,n,'effects')
    good=_bool(arrays['eligible'],shape,n,'eligible');calm=_bool(arrays['state_calm10'],(total,),n,'calm');weak=_bool(arrays['state_weak_narrow'],(total,),n,'weak')
    raw=np.asarray(arrays['core_weight']);require(raw.shape in ((total,),(total,1),shape),'Weight shape')
    cw=_matrix(raw,raw.shape,n,'weight');cw=cw[:,None] if cw.ndim==1 else cw;cw=np.broadcast_to(cw,(n,w)).copy()
    require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,w))),'Original .50/.52 weights')
    with np.errstate(over='raise',invalid='raise',divide='raise'):support=(base-core)/(1.-cw)
    require(np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all(),'Original support')
    extension=_bool(arrays['extension_available'],shape,n,'extension');forecast=_bool(arrays['forecast_available'],shape,n,'forecast')
    logrank=_matrix(arrays['rank_log_excess'],shape,n,'logrank',False)
    ext=calm[:,None]&~weak[:,None]&extension&forecast&good&np.isfinite(logrank)
    episode=np.minimum(.05,np.maximum(0.,.10-(.025*calm+.05*weak)))
    remaining=(1.-cw)-(.025*calm[:,None]+.05*weak[:,None]+.01*ext)-episode[:,None]
    require(np.isfinite(remaining).all() and (remaining>=.38-1e-12).all(),'Original support budget')
    aa={}
    for k,v in arrays.items():
        a=np.asarray(v);require(a.ndim>=1 and a.shape[0]==total,'Monthly cache axis '+k);aa[k]=a[:n].copy()
    aa['core_weight']=cw.copy();mm=dict(meta,dates=meta['dates'][:n],ends=meta['ends'][:n],symbols=meta['symbols'][:n])
    pre=_matrix(study.modified_pre_score(study.configs()[4],aa,n),(n,w),n,'Local67 pre')
    return dict(width=w,base=base,core=core,cw=cw,effects=effects,good=good,calm=calm,weak=weak,ext=ext,episode=episode,remaining=remaining,pre=pre,meta=mm)


def _calendar(calendar):
    require(isinstance(calendar,(list,tuple)),'Calendar list')
    out=[]
    for s in calendar:
        d=_iso(s,'session');require(date.fromisoformat(d).weekday()<5 and (not out or d>out[-1]),'Ordered unique weekday sessions');out.append(d)
    return out


def _anchors(day,sessions,lookup):
    t=lookup[day];require(t+1<len(sessions),'Next official witness')
    require(sessions[t+1][:7]!=day[:7],'Formal month end')
    y,m=map(int,day[:7].split('-'));m0=y*12+m-1;out={}
    for lag in (2,12,14):
        k=m0-lag;month=f'{k//12:04d}-{k%12+1:02d}';ds=[s for s in sessions if s[:7]==month]
        require(bool(ds),'Complete calendar anchor month');anchor=ds[-1];j=lookup[anchor]
        require(j+1<len(sessions) and sessions[j+1][:7]!=month,'Anchor month end witness');out['M-'+str(lag)]=anchor
    return out


def _clock(i,meta,b,sessions,lookup,first):
    day=meta['dates'][i];require(day in lookup,'Query official session');t=lookup[day]
    require(t>=294 and t+1<len(sessions),'Complete trading clock/witness')
    partial=day==INITIAL_PARTIAL;anchors={} if partial else _anchors(day,sessions,lookup)
    start=None if partial else anchors['M-12' if b['calm'][i] else 'M-14'];end=None if partial else anchors['M-2']
    return dict(query_index=i,date=day,initial_partial=partial,calm=bool(b['calm'][i]),core_weight=float(b['cw'][i,0]),symbols=list(meta['symbols'][i]),
      branch='initial_partial' if partial else 'empty_domain',panel_first_session=first,query_position=t,next_official_session=sessions[t+1],
      trading_price_start=sessions[t-(252 if b['calm'][i] else 294)],trading_price_end=sessions[t-42],risk_price_start=sessions[t-63],risk_price_end=day,
      calendar_start=start,calendar_end=end,calendar_lag_sessions=None if partial else t-lookup[end],u0_indices=[],calendar_missing_indices=[],core_source_max_abs_error=None,cells=[])


def _setup(panel,meta,calendar,review,b,n):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Original adjusted panel only');adj=panel['adj']
    require(isinstance(adj,pd.DataFrame) and isinstance(adj.index,pd.DatetimeIndex),'Price frame/date index')
    require(adj.index.tz is None and adj.index.is_unique and adj.index.is_monotonic_increasing and not adj.index.hasnans and all(t==t.normalize() for t in adj.index),'Price axis')
    require(adj.columns.is_unique and all(type(s) is str and s for s in adj.columns),'Price columns')
    require(isinstance(review,dict) and review.get('schema')=='calendar_clock248_metadata_source_review_v1','Source review schema')
    first=_iso(review.get('panel_first_session'),'Original panel origin');sessions=_calendar(calendar);lookup={d:i for i,d in enumerate(sessions)}
    events=[_clock(i,meta,b,sessions,lookup,first) for i in range(n)]
    bounds=review.get('listing_bounds');identities=review.get('identity_cells');require(isinstance(bounds,list) and isinstance(identities,list),'Listing metadata')
    bysymbol={}
    for x in bounds:
        require(isinstance(x,dict) and all(type(x.get(k)) is str and x[k] for k in ('symbol','issuer_key','not_before','available_from','evidence_id','kind')),'Listing bound fields')
        _iso(x['not_before'],'not before');_iso(x['available_from'],'available from');require(x['symbol'] not in bysymbol,'Unique listing bound');bysymbol[x['symbol']]=x
    seen={}
    for x in identities:
        require(isinstance(x,dict) and type(x.get('query_index')) is int and x['query_index']>=0,'Identity query')
        i=x['query_index']
        if i>=n:continue
        j=x.get('column_index');require(type(j) is int and 0<=j<b['width'] and (i,j) not in seen,'Identity column/duplicate');seen[i,j]=x
    expected={(i,j) for i in range(n) for j,s in enumerate(meta['symbols'][i]) if s in bysymbol}
    require(set(seen)==expected,'Complete consumed listing identities')
    for (i,j),x in seen.items():
        e=events[i];bound=bysymbol[e['symbols'][j]];t=e['query_position'];anchors={} if e['initial_partial'] else _anchors(e['date'],sessions,lookup)
        old=dict(t_minus_42=sessions[t-42],t_minus_252=sessions[t-252],t_minus_294=sessions[t-294],risk63_price_start=sessions[t-63],risk63_price_end=e['date'],risk63_prices=64,risk63_return_intervals=63)
        want=dict(date=e['date'],symbol=e['symbols'][j],issuer_key=bound['issuer_key'],available_from=bound['available_from'],not_before=bound['not_before'],bound_visible=bound['available_from']<=e['date'],calendar_anchors=anchors,old_clock_on_official_axis=old)
        require(all(x.get(k)==v for k,v in want.items()),'Listing identity/clock mismatch')
    formal=[e for e in events if not e['initial_partial']]
    if not formal:return adj,events,bysymbol,None
    end=formal[-1]['date'];past=adj.loc[:end]
    require(len(past)>0 and str(past.index[0].date())==first,'Full original rolling history origin')
    names=sorted(set(s for e in formal for s in e['symbols']));require(set(names)<=set(past.columns),'Original source missing symbol')
    for e in formal:
        day=pd.Timestamp(e['date']);require(day in past.index,'Original source missing query')
        p=past.index.get_loc(day);t=e['query_position']
        require(p>=294 and past.index[p-294:p+1].strftime('%Y-%m-%d').tolist()==sessions[t-294:t+1],'Original consumed clock mismatch')
    # No new positive or finite filter on original historical values. Missing and
    # infinity follow the frozen pandas source semantics; malformed types fail.
    px=past[names].copy()
    for s in names:
        values=[]
        for x in px[s]:
            if x is None or x is pd.NA or (isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) and pd.isna(x)):values.append(np.nan)
            else:values.append(_number(x,'Original historical price',False))
        px[s]=np.asarray(values,float)
    with np.errstate(all='ignore'):
        sigma=px.pct_change(fill_method=None).rolling(63).std()
        raw210=px.shift(42)/px.shift(252)-1.
        raw252=(px.shift(42)/px.shift(294)-1.)/sigma.replace(0,np.nan)
    return adj,events,bysymbol,(sigma,raw210,raw252)


def _rank(values):return pd.Series(values,dtype=float).replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).to_numpy()-.5


def _shown(v):return float(v) if np.isfinite(v) else None


def _original(events,frames,b,bounds):
    if frames is None:return
    sigma,raw210,raw252=frames
    # Every source/control gate passes before any new-calendar endpoint read.
    for e in events:
        if e['initial_partial']:continue
        i=e['query_index'];names=e['symbols'];day=e['date'];raw=(raw210 if e['calm'] else raw252).loc[day,names].replace([np.inf,-np.inf],np.nan).to_numpy();ranks=_rank(raw)
        rebuilt=e['core_weight']*ranks;errors=np.abs(rebuilt-b['core'][i]);require((errors<=C_SOURCE_ATOL).all(),'Original weighted-core source mismatch')
        e['core_source_max_abs_error']=float(errors.max());e['u0_indices']=np.flatnonzero(np.isfinite(raw)).tolist()
        for j,symbol in enumerate(names):
            good=bool(np.isfinite(raw[j]));bound=bounds.get(symbol)
            if good and bound and bound['available_from']<=day:require(e['trading_price_start']>=bound['not_before'],'Original pre-inception core contradiction')
            e['cells'].append(dict(column_index=j,symbol=symbol,original_finite=good,original_raw=_shown(raw[j]),original_rank=float(ranks[j]),
              frozen_core=float(b['core'][i,j]),rebuilt_core=float(rebuilt[j]),core_abs_error=float(errors[j]),sigma63=_shown(sigma.at[pd.Timestamp(day),symbol]),
              calendar_source_available=False,calendar_reason='outside_U0',calendar_start_price=None,calendar_end_price=None,calendar_raw=None,calendar_rank=None,delta=0.))


def _endpoint(adj,symbol,day,bound,query):
    if symbol not in adj.columns:return None,'missing_symbol'
    if pd.Timestamp(day) not in adj.index:return None,'missing_row'
    x=adj.at[pd.Timestamp(day),symbol]
    if x is None or x is pd.NA or (isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) and pd.isna(x)):return None,'missing_nan'
    v=_number(x,'Required calendar price');require(v>0,'Positive calendar price')
    if bound and bound['available_from']<=query:require(day>=bound['not_before'],'Calendar pre-inception endpoint contradiction')
    return v,'available'


def _calendar_values(adj,events,bounds):
    priority={'missing_symbol':0,'missing_row':1,'missing_nan':2}
    for e in events:
        if e['initial_partial']:continue
        for j in e['u0_indices']:
            c=e['cells'][j];bound=bounds.get(c['symbol']);start,rs=_endpoint(adj,c['symbol'],e['calendar_start'],bound,e['date']);end,re=_endpoint(adj,c['symbol'],e['calendar_end'],bound,e['date'])
            c['calendar_start_price']=start;c['calendar_end_price']=end
            reasons=[r for r in (rs,re) if r!='available']
            if reasons:c['calendar_reason']=min(reasons,key=priority.get);e['calendar_missing_indices'].append(j);continue
            try:
                raw=(float(end)/float(start))-1.
                if not e['calm']:raw=raw/c['sigma63']
            except (OverflowError,ZeroDivisionError,TypeError) as exc:raise ArithmeticError('Calendar derived arithmetic') from exc
            if not math.isfinite(raw):raise ArithmeticError('Nonfinite calendar raw')
            c['calendar_source_available']=True;c['calendar_reason']='available';c['calendar_raw']=raw
        if not e['u0_indices']:e['branch']='empty_domain'
        elif e['calendar_missing_indices']:e['branch']='missing_calendar'
        else:
            e['branch']='available';rank=_rank([c['calendar_raw'] if c['original_finite'] else np.nan for c in e['cells']])
            for j in e['u0_indices']:
                c=e['cells'][j];c['calendar_rank']=float(rank[j]);c['delta']=e['core_weight']*c['calendar_rank']-c['frozen_core']
                require(math.isfinite(c['delta']),'Finite calendar delta')


def _cached(features,meta,b,n):
    require(isinstance(features,list) and len(features)>=n,'Feature prefix list')
    for i,e in enumerate(features[:n]):
        require(isinstance(e,dict) and tuple(e)==EVENT_KEYS,'Feature event schema')
        require(e['query_index']==i and e['date']==meta['dates'][i] and e['symbols']==meta['symbols'][i],'Feature identities')
        require(type(e['calm']) is bool and e['calm']==bool(b['calm'][i]) and e['core_weight']==float(b['cw'][i,0]),'Feature state/core weight')
        partial=meta['dates'][i]==INITIAL_PARTIAL;require(e['initial_partial'] is partial,'Partial identity');require(e['branch'] in BRANCHES,'Branch')
        _iso(e['panel_first_session'],'Feature origin');_iso(e['next_official_session'],'Next session')
        for k in ('trading_price_start','trading_price_end','risk_price_start','risk_price_end'):_iso(e[k],k)
        require(e['risk_price_end']==e['date'] and e['next_official_session']>e['date'],'Feature clock order')
        if partial:
            require(e['branch']=='initial_partial' and e['cells']==[] and e['u0_indices']==[] and e['calendar_missing_indices']==[] and e['core_source_max_abs_error'] is None,'Partial no numeric feature')
            require(all(e[k] is None for k in ('calendar_start','calendar_end','calendar_lag_sessions')),'Partial no calendar targets');continue
        require(e['next_official_session'][:7]!=e['date'][:7],'Formal feature witness')
        y,m=map(int,e['date'][:7].split('-'));idx=y*12+m-1
        for k,lag in (('calendar_start',12 if e['calm'] else 14),('calendar_end',2)):
            _iso(e[k],k);z=idx-lag;require(e[k][:7]==f'{z//12:04d}-{z%12+1:02d}', 'Calendar anchor month')
        require(type(e['query_position']) is int and e['query_position']>=294 and type(e['calendar_lag_sessions']) is int and e['calendar_lag_sessions']>0,'Feature clock positions')
        require(isinstance(e['cells'],list) and len(e['cells'])==b['width'],'Feature cell axis')
        raw=[];missing=[];available=[];errs=[]
        for j,c in enumerate(e['cells']):
            require(isinstance(c,dict) and tuple(c)==CELL_KEYS and c['column_index']==j and c['symbol']==e['symbols'][j],'Cell schema/identity')
            require(type(c['original_finite']) is bool and type(c['calendar_source_available']) is bool,'Cell masks')
            r=np.nan if c['original_raw'] is None else _number(c['original_raw'],'Raw trade');raw.append(r)
            require(c['original_finite']==bool(np.isfinite(r)),'Original finite mask')
            require(c['frozen_core']==float(b['core'][i,j]),'Frozen core source')
            for k in ('original_rank','rebuilt_core','core_abs_error','delta'):_number(c[k],k)
            if c['sigma63'] is not None:_number(c['sigma63'],'sigma63')
            if not c['original_finite']:
                require(c['calendar_source_available'] is False and c['calendar_reason']=='outside_U0' and all(c[k] is None for k in ('calendar_start_price','calendar_end_price','calendar_raw','calendar_rank')) and c['delta']==0.,'No domain expansion')
            elif c['calendar_source_available']:
                available.append(j);require(c['calendar_reason']=='available','Available reason')
                start=_number(c['calendar_start_price'],'start');end=_number(c['calendar_end_price'],'end');require(start>0 and end>0,'Positive cached endpoints')
                z=end/start-1.
                if not e['calm']:
                    sd=_number(c['sigma63'],'sigma');require(sd>0,'Positive source sigma');z=z/sd
                require(math.isfinite(z) and c['calendar_raw']==z,'Cached calendar formula')
            else:
                missing.append(j);require(c['calendar_reason'] in ('missing_symbol','missing_row','missing_nan') and c['calendar_raw'] is None,'Missing source reason')
                require(c['calendar_start_price'] is None or _number(c['calendar_start_price'],'partial endpoint')>0,'Partial endpoint')
                require(c['calendar_end_price'] is None or _number(c['calendar_end_price'],'partial endpoint')>0,'Partial endpoint')
        ranks=_rank(raw);u0=np.flatnonzero(np.isfinite(raw)).tolist()
        require(e['u0_indices']==u0 and e['calendar_missing_indices']==missing,'Exact source domains')
        expected='empty_domain' if not u0 else ('missing_calendar' if missing else 'available');require(e['branch']==expected,'Common fallback')
        cr=_rank([c['calendar_raw'] if c['original_finite'] else np.nan for c in e['cells']]) if expected=='available' else None
        for j,c in enumerate(e['cells']):
            rebuilt=e['core_weight']*float(ranks[j]);err=abs(rebuilt-c['frozen_core']);errs.append(err)
            require(c['original_rank']==float(ranks[j]) and c['rebuilt_core']==rebuilt and c['core_abs_error']==err and err<=C_SOURCE_ATOL,'Original ranks/core source parity')
            target=float(cr[j]) if expected=='available' and j in u0 else None
            delta=e['core_weight']*target-c['frozen_core'] if target is not None else 0.
            require(c['calendar_rank']==target and c['delta']==delta,'Exact target/delta')
        require(e['core_source_max_abs_error']==max(errs),'Source maximum error')
    return features[:n]


def _select(study,meta,b,features,n,*,incumbent_validator=None,expected_incumbent_orders=None):
    require(incumbent_validator is None or callable(incumbent_validator),'Incumbent validator callable')
    require(expected_incumbent_orders is None or (isinstance(expected_incumbent_orders,list) and len(expected_incumbent_orders)==n),'Expected incumbent orders prefix')
    fs=_cached(features,meta,b,n);w=b['width'];target=np.full((n,w),np.nan);delta=np.zeros((n,w))
    for i,e in enumerate(fs):
        for c in e['cells']:
            j=c['column_index'];delta[i,j]=c['delta']
            if c['calendar_rank'] is not None:target[i,j]=c['calendar_rank']
    source=dict(baseline=b['base'],core=b['core'],core_weight=b['cw'],eligible=b['good'],state_calm10=b['calm'],state_weak_narrow=b['weak'])
    orders={};scores={};pres={};choices={};calls=[]
    for k,arm in enumerate(ARMS):
        d=np.zeros((n,w)) if k==0 else delta;pre=b['pre'].copy();changed=d!=0.
        with np.errstate(over='raise',invalid='raise'):pre[changed]+=d[changed]
        require(np.isfinite(pre).all(),'Finite candidate pre');arm_calls=[]
        def boundary(score,eligible,names,effects,original,policy):
            require(not arm_calls,'One episode per arm');require(np.array_equal(score,pre) and np.array_equal(eligible,b['good']),'Expected pre/original E')
            require(names==meta['symbols'][:n] and np.array_equal(effects,b['effects']),'Original names/effects')
            for key,v in source.items():
                if key!='eligible':require(np.array_equal(original[key],v),'Original episode support '+key)
            require(policy==POLICY,'Original episode policy');arm_calls.append(True);calls.append(arm)
            return study.ep.select_boundary(score,eligible,names,effects,original,policy)
        chosen,post=core_adapter.score_and_choose('incumbent66',float(k),np.full((n,w),np.nan) if k==0 else target,source,b['meta'],n,
          pre_score=b['pre'],effects=b['effects'],episode_policy=POLICY.copy(),boundary_selector=boundary)
        require(len(arm_calls)==1,'Exactly one episode');post=_matrix(post,(n,w),n,'Final scores');require(isinstance(chosen,list) and len(chosen)==n,'Choices axis')
        choices[arm]=[[int(j) for j in row] for row in chosen];scores[arm]=post;pres[arm]=pre;orders[arm]=study.rp.make_orders(chosen,post,b['meta'])
        if k==0:
            if expected_incumbent_orders is not None:require(orders[arm]==expected_incumbent_orders,'Frozen incumbent ordered choices mismatch')
            if incumbent_validator is not None:
                receipt=incumbent_validator(deepcopy(orders[arm]),post.copy(),pre.copy())
                require(isinstance(receipt,dict) and receipt.get('passed') is True,'Independent incumbent gate failed')
    for i,e in enumerate(fs):
        if e['branch']!='available':
            require(orders[ARMS[0]][i]==orders[ARMS[1]][i] and np.array_equal(pres[ARMS[0]][i],pres[ARMS[1]][i]) and np.array_equal(scores[ARMS[0]][i],scores[ARMS[1]][i]),'Exact whole-query fallback')
    cells=[c for e in fs for c in e['cells']];formal=[e for e in fs if not e['initial_partial']]
    counts=dict(events=n,symbols_per_event=w,initial_partial_events=n-len(formal),formal_events=len(formal),calm_formal_events=sum(e['calm'] for e in formal),noncalm_formal_events=sum(not e['calm'] for e in formal),
      original_checked_cells=len(cells),original_finite_cells=sum(c['original_finite'] for c in cells),calendar_required_cells=sum(c['original_finite'] for c in cells),calendar_available_cells=sum(c['calendar_source_available'] for c in cells),
      calendar_missing_cells=sum(len(e['calendar_missing_indices']) for e in fs),available_events=sum(e['branch']=='available' for e in fs),empty_domain_events=sum(e['branch']=='empty_domain' for e in fs),missing_calendar_events=sum(e['branch']=='missing_calendar' for e in fs),
      candidate_changed_core_cells=int(np.count_nonzero(delta)),changed_holdings_events=sum(set(choices[ARMS[0]][i])!=set(choices[ARMS[1]][i]) for i in range(n)),original_episode_calls=len(calls),new_model_fits=0,predictive_model_fits=0)
    require(tuple(counts)==COUNT_FIELDS,'Count schema');events=[]
    for i,e in enumerate(fs):
        z=dict(query_index=i,date=e['date'],branch=e['branch'],u0_indices=list(e['u0_indices']),candidate_delta=delta[i].tolist(),extension_active_indices=np.flatnonzero(b['ext'][i]).tolist(),episode_budget=float(b['episode'][i]),original_remaining_support=b['remaining'][i].tolist(),incumbent_choices=choices[ARMS[0]][i],candidate_choices=choices[ARMS[1]][i])
        require(tuple(z)==AUDIT_EVENT_KEYS,'Decision schema');events.append(z)
    return orders,scores,pres,dict(schema='monthly_calendar_clock248_decisions_v1',counts=counts,events=events)


def prepare(study,arrays,meta,panel,calendar,source_review,n,*,source_validator=None,incumbent_validator=None,expected_incumbent_orders=None):
    before=_snapshot(arrays);before_meta=deepcopy(meta);b=_cache(study,arrays,meta,n)
    adj,events,bounds,frames=_setup(panel,meta,calendar,source_review,b,n)
    _original(events,frames,b,bounds);_calendar_values(adj,events,bounds)
    require(all(tuple(e)==EVENT_KEYS and all(tuple(c)==CELL_KEYS for c in e['cells']) for e in events),'Prepared feature schema')
    if source_validator is not None:
        require(callable(source_validator),'Source validator callable')
        receipt=source_validator(deepcopy(events))
        require(isinstance(receipt,dict) and receipt.get('passed') is True,'Independent original-source gate failed')
    orders,scores,pres,audit=_select(study,meta,b,events,n,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders)
    require(before==_snapshot(arrays) and meta==before_meta,'Original input mutation')
    return events,orders,scores,pres,audit


def decisions(study,arrays,meta,features,n,*,incumbent_validator=None,expected_incumbent_orders=None):
    before=_snapshot(arrays);before_meta=deepcopy(meta);before_features=deepcopy(features)
    b=_cache(study,arrays,meta,n);out=_select(study,meta,b,features,n,incumbent_validator=incumbent_validator,expected_incumbent_orders=expected_incumbent_orders)
    require(before==_snapshot(arrays) and meta==before_meta and features==before_features,'Cached input mutation')
    return out
