"""Pure exact daily SPY-up participation and funded original-score integration.

No source loading, price-return replay, fitting, or cross-sectional ranking.
Security source support is inspected independently of benchmark support.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AXES_PATH=ROOT/'work/monthly_risk_consensus219_kernel.py'
AXES_SHA='7316b7cc081e7f8c25fc8cbeb51f43642223367609be1f77e4beb81929c1be77'
ADAPTER_PATH=ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
ADAPTER_SHA='5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b'


def _module(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen dependency: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


axes=_module(AXES_PATH,AXES_SHA,'up244_axes219')
adapter=_module(ADAPTER_PATH,ADAPTER_SHA,'up244_path_adapter')
ARMS=('incumbent_local67','up_participation','all_participation_control')
VALUE_KEYS=('available','candidate','control')
FEATURE_KEYS=VALUE_KEYS
COUNT_FIELDS=('events','symbols_per_event','checked_cells','source_available_cells','benchmark_available_events','paired_source_cells',
    'available_cells','unavailable_cells','security_price_observations','benchmark_price_observations','relative_comparisons',
    'conditional_comparisons','conditional_estimates','unconditional_estimates','new_model_fits','predictive_model_fits')
MEASURE_KEYS=('source_available','source_reason','start_price','end_price','benchmark_available','benchmark_reason',
    'benchmark_start_price','benchmark_end_price','up_count','down_count','flat_count','available','reason',
    'wins_up','ties_up','losses_up','wins_all','ties_all','losses_all','candidate','control')
CELL_KEYS=('column_index','symbol')+MEASURE_KEYS
EVENT_KEYS=('query_index','date','symbols','session_position','price_start','price_end','source_dates','missing_official_sessions',
    'benchmark_available','benchmark_reason','benchmark_start_price','benchmark_end_price','up_count','down_count','flat_count',
    'source_available_indices','available_indices','cells')
DECISION_COUNT_FIELDS=('events','symbols_per_event','active_events','available_active_cells','candidate_nonzero_delta_cells',
    'control_nonzero_delta_cells','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','active','available_indices','extension_active_indices',
    'episode_budget','candidate_delta','control_delta','original_remaining_support','candidate_remaining_support','control_remaining_support')
PRICE_COUNT=253
RETURN_COUNT=252
MIN_UP=40
INITIAL_PARTIAL='2016-08-17'
BUDGET=.025
POLICY=dict(budget=.05,ceiling=5,gap=.02)


def require(ok,message):
    if not ok:raise ValueError(message)


def dependency_files():
    return (AXES_PATH,ADAPTER_PATH,ROOT/'outputs/monthly_local_linear70/study.py',
        ROOT/'outputs/monthly_daily_episodes/episodes.py',ROOT/'outputs/monthly_return_priority/return_priority.py',
        ROOT/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json')


def _empty_source(reason):return dict(available=False,reason=reason,prices=None)


def _source(prices):
    require(np.shape(prices)==(PRICE_COUNT,),'Exactly253 adjusted prices required')
    raw=np.asarray(prices,dtype=object)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw):return _empty_source('nonnumeric_adjusted_price')
    try:out=[float(v) for v in raw]
    except (TypeError,ValueError,OverflowError):return _empty_source('nonnumeric_adjusted_price')
    if not all(math.isfinite(v) and v>0. for v in out):return _empty_source('invalid_adjusted_price')
    return dict(available=True,reason='available',prices=out)


def _market(source):
    if not source['available']:
        return dict(source,up=None,up_count=None,down_count=None,flat_count=None,gross=None)
    p=source['prices'];up=[p[j+1]>p[j] for j in range(RETURN_COUNT)]
    down=sum(p[j+1]<p[j] for j in range(RETURN_COUNT));count=sum(up)
    ratios=[v.as_integer_ratio() for v in p]
    gross=[(ratios[j+1][0]*ratios[j][1],ratios[j+1][1]*ratios[j][0]) for j in range(RETURN_COUNT)]
    return dict(source,up=up,up_count=count,down_count=down,flat_count=RETURN_COUNT-count-down,gross=gross)


def _measure(source,market):
    sv,mv=source['available'],market['available'];sp,mp=source['prices'],market['prices']
    result=dict(source_available=sv,source_reason=source['reason'],start_price=sp[0] if sv else None,end_price=sp[-1] if sv else None,
        benchmark_available=mv,benchmark_reason=market['reason'],benchmark_start_price=mp[0] if mv else None,
        benchmark_end_price=mp[-1] if mv else None,up_count=market['up_count'],down_count=market['down_count'],flat_count=market['flat_count'],
        available=False,reason=source['reason'] if not sv else 'benchmark_'+market['reason'],
        wins_up=None,ties_up=None,losses_up=None,wins_all=None,ties_all=None,losses_all=None,candidate=np.nan,control=np.nan)
    if sv and mv:
        rational=[v.as_integer_ratio() for v in sp];outcomes=[]
        for j,(mn,md) in enumerate(market['gross']):
            sn=rational[j+1][0]*rational[j][1];sd=rational[j+1][1]*rational[j][0]
            left,right=sn*md,mn*sd;outcomes.append(int(left>right)-int(left<right))
        up_outcomes=[v for v,u in zip(outcomes,market['up']) if u]
        wu,tu,lu=(up_outcomes.count(v) for v in (1,0,-1));wa,ta,la=(outcomes.count(v) for v in (1,0,-1))
        result.update(wins_up=wu,ties_up=tu,losses_up=lu,wins_all=wa,ties_all=ta,losses_all=la)
        n=market['up_count']
        if n<MIN_UP:result['reason']='insufficient_up_days'
        else:
            q=float(2*wu+tu-n)/float(2*n);c=float(2*wa+ta-RETURN_COUNT)/float(2*RETURN_COUNT)
            if n==RETURN_COUNT:c=q
            require(math.isfinite(q) and math.isfinite(c) and -.5<=q<=.5 and -.5<=c<=.5,'Finite bounded direct proportions')
            result.update(available=True,reason='available',candidate=q,control=c)
    require(tuple(result)==MEASURE_KEYS,'Measure schema');return result


def measure(security,market):
    """Full independent source checks, then exact integer cross products."""
    own=_source(security);benchmark=_market(_source(market));return _measure(own,benchmark)


def _inputs(panel,meta,calendar,limit):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Panel requires exactly adj');axes._frame(panel['adj'])
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','symbols','etfs')),'Feature metadata')
    require(not isinstance(meta['dates'],(str,bytes)) and not isinstance(meta['symbols'],(str,bytes)),'Metadata sequences')
    total=len(meta['dates']);require(len(meta['symbols'])==total,'Metadata date/symbol axes')
    n=total if limit is None else limit;require(type(n) is int and 0<=n<=total,'Valid prefix')
    require(not isinstance(calendar,(str,bytes)),'Explicit official calendar')
    sessions=axes._calendar(calendar) if len(calendar) else pd.DatetimeIndex([])
    consumed=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]);queries,width=axes._queries(consumed,sessions)
    if n==0 and total:
        require(not isinstance(meta['symbols'][0],(str,bytes)),'Width sequence');width=len(meta['symbols'][0])
    return panel['adj'],sessions,queries,width


def _extract(adj,dates,missing,symbol):
    if not len(dates):return _empty_source('insufficient_history')
    if symbol not in adj.columns:return _empty_source('missing_symbol')
    if missing:return _empty_source('missing_official_session')
    return _source(adj.loc[dates,symbol].to_numpy(copy=True))


def build_features(panel,meta,calendar,limit=None):
    adj,sessions,queries,width=_inputs(panel,meta,calendar,limit);shape=(len(queries),width)
    values={key:np.zeros(shape,bool) if key=='available' else np.full(shape,np.nan) for key in VALUE_KEYS}
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=len(queries),symbols_per_event=width,checked_cells=len(queries)*width)
    history=[];reasons=Counter()
    for i,(day,symbols,t) in enumerate(queries):
        dates=sessions[t-294:t-41] if t>=294 else pd.DatetimeIndex([])
        isos=dates.strftime('%Y-%m-%d').tolist();missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
        market=_market(_extract(adj,dates,missing,'SPY'));cells=[];source_indices=[];available_indices=[]
        if market['available']:counts['benchmark_available_events']+=1
        for j,symbol in enumerate(symbols):
            # Never short-circuit this inspection because market/count support failed.
            result=_measure(_extract(adj,dates,missing,symbol),market)
            for key in VALUE_KEYS:values[key][i,j]=result[key]
            cell=dict(column_index=j,symbol=symbol,**result)
            for key in ('candidate','control'):
                if not result['available']:cell[key]=None
            require(tuple(cell)==CELL_KEYS,'Cell schema');cells.append(cell);reasons[result['reason']]+=1
            if result['source_available']:source_indices.append(j)
            if result['available']:available_indices.append(j)
            if result['source_available'] and result['benchmark_available']:
                counts['paired_source_cells']+=1;counts['conditional_comparisons']+=market['up_count']
        counts['source_available_cells']+=len(source_indices);counts['available_cells']+=len(available_indices)
        event=dict(query_index=i,date=str(day.date()),symbols=symbols,session_position=t,price_start=isos[0] if isos else None,
            price_end=isos[-1] if isos else None,source_dates=isos,missing_official_sessions=missing,
            benchmark_available=market['available'],benchmark_reason=market['reason'],benchmark_start_price=market['prices'][0] if market['available'] else None,
            benchmark_end_price=market['prices'][-1] if market['available'] else None,up_count=market['up_count'],down_count=market['down_count'],
            flat_count=market['flat_count'],source_available_indices=source_indices,available_indices=available_indices,cells=cells)
        require(tuple(event)==EVENT_KEYS,'Event schema');history.append(event)
    count=counts['available_cells']
    counts.update(unavailable_cells=counts['checked_cells']-count,security_price_observations=PRICE_COUNT*counts['source_available_cells'],
        benchmark_price_observations=PRICE_COUNT*counts['benchmark_available_events'],relative_comparisons=RETURN_COUNT*counts['paired_source_cells'],
        conditional_estimates=count,unconditional_estimates=count)
    return values,dict(schema='monthly_up_participation244_features_v1',events=len(queries),counts=counts,reason_counts=dict(reasons),event_history=history)


def _snapshot(arrays):
    return {k:(np.asarray(v).shape,np.asarray(v).dtype.str,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _numeric(value,shape,n,label,finite=True):
    raw=np.asarray(value);require(raw.shape==shape,'Exact shape: '+label)
    require(all(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)) for v in raw[:n].flat),'Real nonboolean: '+label)
    try:out=raw[:n].astype(float,copy=True)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Unrepresentable '+label) from exc
    if finite:require(np.isfinite(out).all(),'Finite: '+label)
    return out


def _bool(value,shape,n,label):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind=='b','Boolean shape: '+label);return a[:n].copy()


def _features(values,shape,n):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Three exact feature matrices')
    mask=_bool(values['available'],shape,n,'availability');out={'available':mask}
    for key in ('candidate','control'):
        v=_numeric(values[key],shape,n,key,finite=False)
        require(np.isfinite(v[mask]).all() and np.isnan(v[~mask]).all(),'Common finite/NaN features')
        require((np.abs(v[mask])<=.5).all(),'Bounded proportion');out[key]=v
    return out


def decisions(study,arrays,meta,values,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Selection mappings')
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Selection prefix')
    total,width=shape
    require(all(isinstance(meta.get(k),list) and len(meta[k])==total for k in ('dates','ends','symbols')),'Selection metadata axes')
    previous=None
    for i in range(n):
        day=axes._day(meta['dates'][i],'Selection date');end=axes._day(meta['ends'][i],'Selection end');names=meta['symbols'][i]
        require(end>day and (previous is None or day>previous),'Selection chronology');previous=day
        require(str(day.date())!=INITIAL_PARTIAL or i==0,'Named partial must be first')
        require(isinstance(names,list) and len(names)==width and all(type(x) is str and x and x.strip()==x for x in names) and len(set(names))==width,'Unique symbols')
    before=_snapshot(arrays);before_values=_snapshot(values);before_meta=deepcopy(meta);f=_features(values,shape,n)
    base=_numeric(arrays['baseline'],shape,n,'baseline');core=_numeric(arrays['core'],shape,n,'weighted core');effect=_numeric(arrays['effects'],shape,n,'effects')
    good=_bool(arrays['eligible'],shape,n,'eligible');calm=_bool(arrays['state_calm10'],(total,),n,'calm');weak=_bool(arrays['state_weak_narrow'],(total,),n,'weak')
    rawcw=np.asarray(arrays['core_weight'])
    if rawcw.shape==(total,):cw=_numeric(rawcw,(total,),n,'core_weight')[:,None]
    else:
        require(rawcw.shape in ((total,1),shape),'Core weight shape');cw=_numeric(rawcw,rawcw.shape,n,'core_weight')
    cw=np.broadcast_to(cw,(n,width)).copy();require(np.array_equal(cw,np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,width))),'Original core weights')
    with np.errstate(over='raise',invalid='raise',divide='raise'):support=(base-core)/(1.-cw)
    require(np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all(),'Original bounded support')
    extension=_bool(arrays['extension_available'],shape,n,'extension');forecast=_bool(arrays['forecast_available'],shape,n,'forecast')
    logrank=_numeric(arrays['rank_log_excess'],shape,n,'logrank',finite=False)
    ext=calm[:,None]&~weak[:,None]&extension&forecast&good&np.isfinite(logrank)
    active=np.array([meta['dates'][i]!=INITIAL_PARTIAL for i in range(n)],bool);use=active[:,None]&f['available']
    episode=np.minimum(.05,np.maximum(0.,.10-(.025*calm+.05*weak)))
    remaining=(1.-cw)-(.025*calm[:,None]+.05*weak[:,None]+.01*ext)-episode[:,None]
    funded=remaining-BUDGET*use
    require(np.isfinite(funded).all() and (funded>=.355-1e-12).all(),'Extra extension and funded budget lower bound')
    pre=_numeric(study.modified_pre_score(study.configs()[4],arrays,n),shape,n,'original pre')
    episode_arrays=dict(baseline=base.copy(),core=core.copy(),core_weight=cw.copy(),eligible=good.copy(),state_calm10=calm.copy(),state_weak_narrow=weak.copy())
    scores={};pres={};orders={};deltas={};calls=[]
    for arm,key in zip(ARMS,(None,'candidate','control')):
        target=np.full((n,width),np.nan);delta=np.zeros((n,width))
        if key is not None:
            target[use]=f[key][use]
            with np.errstate(over='raise',invalid='raise'):delta[use]=BUDGET*(target[use]-support[use])
        arm_pre=pre.copy();changed=delta!=0.
        with np.errstate(over='raise',invalid='raise'):arm_pre[changed]+=delta[changed]
        require(np.isfinite(delta).all() and np.isfinite(arm_pre).all(),'Finite funded scores');arm_calls=[]
        def boundary(score,eligible,names,source_effects,original,policy):
            require(not arm_calls,'Exactly one episode per arm')
            require(np.array_equal(score,arm_pre) and np.array_equal(eligible,good),'Expected pre and original E')
            require(names==meta['symbols'][:n] and np.array_equal(source_effects,effect),'Original symbols/effects')
            for field,expected in (('baseline',base),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
                require(np.array_equal(original[field],expected),'Original episode support '+field)
            require(policy==POLICY,'Original episode policy');arm_calls.append(True);calls.append(arm)
            return study.ep.select_boundary(score,eligible,names,source_effects,original,policy)
        chosen,post=adapter.score_and_choose('incumbent66',0. if key is None else BUDGET,target,episode_arrays,meta,n,
            pre_score=pre,effects=effect,episode_policy=POLICY.copy(),boundary_selector=boundary)
        require(len(arm_calls)==1,'Exactly one original episode');post=_numeric(post,(n,width),n,'post score')
        require(isinstance(chosen,list) and len(chosen)==n,'Choice axes')
        scores[arm]=post.copy();pres[arm]=arm_pre.copy();orders[arm]=study.rp.make_orders(chosen,post,meta)
        if key is not None:deltas[key]=delta
    events=[]
    for i in range(n):
        event=dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]==INITIAL_PARTIAL,calm=bool(calm[i]),weak=bool(weak[i]),active=bool(active[i]),
            available_indices=np.flatnonzero(f['available'][i]).tolist(),extension_active_indices=np.flatnonzero(ext[i]).tolist(),episode_budget=float(episode[i]),
            candidate_delta=deltas['candidate'][i].tolist(),control_delta=deltas['control'][i].tolist(),original_remaining_support=remaining[i].tolist(),
            candidate_remaining_support=funded[i].tolist(),control_remaining_support=funded[i].tolist())
        require(tuple(event)==DECISION_EVENT_KEYS,'Decision event schema');events.append(event)
        if not active[i]:
            for arm in ARMS[1:]:require(orders[arm][i]==orders[ARMS[0]][i] and np.array_equal(scores[arm][i],scores[ARMS[0]][i]),'Initial exact incumbent')
    counts=dict(events=n,symbols_per_event=width,active_events=int(active.sum()),available_active_cells=int(use.sum()),
        candidate_nonzero_delta_cells=int(np.count_nonzero(deltas['candidate'])),control_nonzero_delta_cells=int(np.count_nonzero(deltas['control'])),
        original_episode_calls=len(calls),new_model_fits=0,predictive_model_fits=0)
    require(tuple(counts)==DECISION_COUNT_FIELDS,'Decision count schema')
    require(_snapshot(arrays)==before and _snapshot(values)==before_values and meta==before_meta,'Input mutation')
    return orders,scores,pres,dict(schema='monthly_up_participation244_decisions_v1',counts=counts,events=events)
