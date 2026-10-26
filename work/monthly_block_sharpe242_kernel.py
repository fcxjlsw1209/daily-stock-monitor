"""Pure paired block/daily Sharpe features and frozen Local67 core integration.

No price loading, source admission, model fitting, or portfolio-return replay.
Caller provenance gates precede any actual-data use. Numeric source support is
shared across arms; feature ranking precedes original trading eligibility.
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
ADAPTER_PATH=ROOT/'outputs/monthly_core_timing70_research/core_adapter.py'
ADAPTER_SHA='b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a'


def _module(path,expected,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Changed immutable pure dependency: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


axes=_module(AXES_PATH,AXES_SHA,'block242_axes219')
core_adapter=_module(ADAPTER_PATH,ADAPTER_SHA,'block242_original_core_adapter')
ARMS=('incumbent_local67','block21_sharpe','daily_sharpe_control')
VALUE_KEYS=('available','candidate','control','block_sharpe','daily_sharpe')
FEATURE_KEYS=VALUE_KEYS
COUNT_FIELDS=('events','symbols_per_event','checked_cells','source_available_cells','available_cells','unavailable_cells','valid_price_observations','daily_returns','block_returns','ranked_observations','zero_daily_volatility_cells','zero_block_volatility_cells','new_model_fits','predictive_model_fits')
MEASURE_KEYS=('source_available','available','reason','start_price','end_price','daily_return_sum','daily_mean','daily_centered_square_sum','daily_sample_vol','block_return_sum','block_mean','block_centered_square_sum','block_sample_vol','daily_sharpe','block_sharpe')
CELL_KEYS=('column_index','symbol')+MEASURE_KEYS+('candidate','control')
EVENT_KEYS=('query_index','date','symbols','session_position','price_start','price_end','source_dates','block_endpoint_dates','missing_official_sessions','source_available_indices','available_indices','cells')
DECISION_COUNT_FIELDS=('events','symbols_per_event','active_events','available_active_cells','candidate_nonzero_delta_cells','control_nonzero_delta_cells','original_episode_calls','new_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','active','available_indices','candidate_delta','control_delta')
INITIAL_PARTIAL='2016-08-17'
POLICY=dict(budget=.05,ceiling=5,gap=.02)
PRICE_COUNT=253
BLOCK_LENGTH=21
BLOCK_COUNT=12


def require(ok,message):
    if not ok:raise ValueError(message)


def dependency_files():
    return (AXES_PATH,ADAPTER_PATH,core_adapter.PATH_ADAPTER_PATH,
        ROOT/'outputs/monthly_local_linear70/study.py',
        ROOT/'outputs/monthly_daily_episodes/episodes.py',
        ROOT/'outputs/monthly_return_priority/return_priority.py',
        ROOT/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json')


def _finite(value):
    value=float(value)
    if not math.isfinite(value):raise FloatingPointError('Nonfinite derived arithmetic')
    return value


def empty_measure(reason):
    result=dict.fromkeys(MEASURE_KEYS,np.nan)
    result.update(source_available=False,available=False,reason=reason,start_price=None,end_price=None)
    return result


def _moments(returns):
    count=len(returns)
    total=_finite(math.fsum(returns));mean=_finite(total/float(count))
    if total!=0. and mean==0.:raise FloatingPointError('Mean underflow')
    deviations=[_finite(x-mean) for x in returns]
    squares=[_finite(d*d) for d in deviations]
    if any(d!=0. and q==0. for d,q in zip(deviations,squares)):
        raise FloatingPointError('Centered square underflow')
    square_sum=_finite(math.fsum(squares))
    variance=_finite(square_sum/float(count-1))
    if square_sum>0. and variance==0.:raise FloatingPointError('Variance underflow')
    sd=_finite(math.sqrt(variance))
    return total,mean,square_sum,sd


def _simple(end,start):
    ratio=_finite(end/start)
    if ratio==0.:raise FloatingPointError('Positive price ratio underflow')
    return _finite(ratio-1.)


def _sharpe(mean,sd):
    value=_finite(mean/sd)
    if mean!=0. and value==0.:raise FloatingPointError('Sharpe underflow')
    return value


def measure(prices):
    """Exactly253 prices; daily252 and disjoint block12 two-pass sample risks."""
    require(np.shape(prices)==(PRICE_COUNT,),'Exactly253 adjusted prices required')
    raw=np.asarray(prices,dtype=object)
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:p=[float(x) for x in raw]
    except (TypeError,ValueError,OverflowError):return empty_measure('nonnumeric_adjusted_price')
    if not all(math.isfinite(x) and x>0. for x in p):return empty_measure('invalid_adjusted_price')
    try:
        daily=[_simple(p[j+1],p[j]) for j in range(252)]
        blocks=[_simple(p[j+21],p[j]) for j in range(0,252,21)]
        dt,dm,ds,dv=_moments(daily);bt,bm,bs,bv=_moments(blocks)
        available=dv>0. and bv>0.
        if dv==0. and bv==0.:reason='zero_both_sample_volatility'
        elif dv==0.:reason='zero_daily_sample_volatility'
        elif bv==0.:reason='zero_block_sample_volatility'
        else:reason='available'
        daily_sharpe=_sharpe(dm,dv) if available else np.nan
        block_sharpe=_sharpe(bm,bv) if available else np.nan
    except (OverflowError,ZeroDivisionError) as exc:
        raise FloatingPointError('Arithmetic overflow from valid prices') from exc
    return dict(source_available=True,available=available,reason=reason,start_price=p[0],end_price=p[-1],
        daily_return_sum=dt,daily_mean=dm,daily_centered_square_sum=ds,daily_sample_vol=dv,
        block_return_sum=bt,block_mean=bm,block_centered_square_sum=bs,block_sample_vol=bv,
        daily_sharpe=daily_sharpe,block_sharpe=block_sharpe)


def percentile_rank(raw,available):
    raw=np.asarray(raw);available=np.asarray(available)
    require(raw.ndim==1 and raw.shape==available.shape and available.dtype.kind=='b','Rank axes')
    require(np.isfinite(raw[available]).all(),'Finite available raw ranks')
    out=np.full(raw.shape,np.nan);indices=np.flatnonzero(available).tolist();n=len(indices)
    for j in indices:
        less=sum(raw[k]<raw[j] for k in indices);equal=sum(raw[k]==raw[j] for k in indices)
        out[j]=(less+(equal+1)/2.0)/n-.5
    return out


def _inputs(panel,meta,calendar,limit):
    require(isinstance(panel,dict) and set(panel)=={'adj'},'Panel requires exactly adj')
    axes._frame(panel['adj'])
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','symbols','etfs')),'Feature metadata')
    require(not isinstance(meta['dates'],(str,bytes)) and not isinstance(meta['symbols'],(str,bytes)),'Metadata sequences')
    total=len(meta['dates']);require(len(meta['symbols'])==total,'Metadata date/symbol axes')
    n=total if limit is None else limit
    require(type(n) is int and 0<=n<=total,'Valid prefix limit')
    require(not isinstance(calendar,(str,bytes)),'Explicit official calendar')
    sessions=axes._calendar(calendar) if len(calendar) else pd.DatetimeIndex([])
    consumed=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
    queries,width=axes._queries(consumed,sessions)
    if n==0 and total:
        require(not isinstance(meta['symbols'][0],(str,bytes)),'Width sequence')
        width=len(meta['symbols'][0])
    return panel['adj'],sessions,queries,width


def _json_number(value):
    return None if isinstance(value,(float,np.floating)) and not math.isfinite(value) else value


def build_features(panel,meta,calendar,limit=None):
    adj,sessions,queries,width=_inputs(panel,meta,calendar,limit)
    shape=(len(queries),width)
    values={key:np.zeros(shape,bool) if key=='available' else np.full(shape,np.nan) for key in VALUE_KEYS}
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=len(queries),symbols_per_event=width,checked_cells=len(queries)*width)
    history=[];reasons=Counter()
    for i,(day,symbols,t) in enumerate(queries):
        dates=sessions[t-294:t-41] if t>=294 else pd.DatetimeIndex([])
        isos=dates.strftime('%Y-%m-%d').tolist();missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
        cells=[];source_indices=[]
        for j,symbol in enumerate(symbols):
            if not len(dates):result=empty_measure('insufficient_history')
            elif symbol not in adj.columns:result=empty_measure('missing_symbol')
            elif missing:result=empty_measure('missing_official_session')
            else:result=measure(adj.loc[dates,symbol].to_numpy(copy=True))
            values['available'][i,j]=result['available']
            for key in ('block_sharpe','daily_sharpe'):values[key][i,j]=result[key]
            cell=dict(column_index=j,symbol=symbol,**{k:_json_number(v) for k,v in result.items()},candidate=None,control=None)
            require(tuple(cell)==CELL_KEYS,'Cell schema');cells.append(cell);reasons[result['reason']]+=1
            if result['source_available']:
                source_indices.append(j)
                counts['zero_daily_volatility_cells']+=int(result['daily_sample_vol']==0.)
                counts['zero_block_volatility_cells']+=int(result['block_sample_vol']==0.)
        for target,rawkey in (('candidate','block_sharpe'),('control','daily_sharpe')):
            values[target][i]=percentile_rank(values[rawkey][i],values['available'][i])
        indices=np.flatnonzero(values['available'][i]).tolist()
        for j in indices:
            cells[j]['candidate']=float(values['candidate'][i,j]);cells[j]['control']=float(values['control'][i,j])
        counts['source_available_cells']+=len(source_indices);counts['available_cells']+=len(indices)
        event=dict(query_index=i,date=str(day.date()),symbols=symbols,session_position=t,
            price_start=isos[0] if isos else None,price_end=isos[-1] if isos else None,
            source_dates=isos,block_endpoint_dates=isos[::21],missing_official_sessions=missing,
            source_available_indices=source_indices,available_indices=indices,cells=cells)
        require(tuple(event)==EVENT_KEYS,'Event schema');history.append(event)
    s=counts['source_available_cells'];a=counts['available_cells']
    counts.update(unavailable_cells=counts['checked_cells']-a,valid_price_observations=253*s,
        daily_returns=252*s,block_returns=12*s,ranked_observations=2*a)
    return values,dict(schema='monthly_block_sharpe242_features_v1',counts=counts,reason_counts=dict(reasons),event_history=history)


def _snapshot(arrays):
    return {k:(np.asarray(v).shape,np.asarray(v).dtype.str,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _feature_values(values,shape,n):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Five exact feature matrices')
    mask=np.asarray(values['available']);require(mask.shape==shape and mask.dtype.kind=='b','Common boolean feature mask')
    mask=mask[:n].copy();out={'available':mask}
    for key in VALUE_KEYS[1:]:
        raw=np.asarray(values[key]);require(raw.shape==shape,'Full feature axes')
        require(all(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) for x in raw[:n].flat),'Real nonboolean feature values')
        a=raw[:n].astype(float,copy=True)
        require(np.isfinite(a[mask]).all() and np.isnan(a[~mask]).all(),'Exact common finite/NaN feature support')
        if key in ('candidate','control'):require(((a[mask]>=-.5)&(a[mask]<=.5)).all(),'Bounded core targets')
        out[key]=a
    return out


def decisions(study,arrays,meta,values,n):
    """Three arms, each one original episode; full source arrays never changed."""
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Selection mappings')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Selection prefix axes')
    require(all(isinstance(meta.get(k),list) and len(meta[k])==shape[0] for k in ('dates','ends','symbols')),'Selection metadata axes')
    for i in range(n):
        day=axes._day(meta['dates'][i],'Selection date');end=axes._day(meta['ends'][i],'Selection end')
        names=meta['symbols'][i]
        require(end>day and (i==0 or day>axes._day(meta['dates'][i-1],'Prior date')),'Selection chronology')
        require(str(day.date())!=INITIAL_PARTIAL or i==0,'Named partial must be first')
        require(isinstance(names,list) and len(names)==shape[1] and len(set(names))==shape[1] and all(type(x) is str and x for x in names),'Unique selection symbols')
    before=_snapshot(arrays);before_meta=deepcopy(meta);before_values=_snapshot(values)
    features=_feature_values(values,shape,n)
    calm=study.adapter._state(arrays,'state_calm10',shape[0],n)
    weak=study.adapter._state(arrays,'state_weak_narrow',shape[0],n)
    canonical=core_adapter._canonical_weights(arrays,n)
    expected=np.broadcast_to(np.where(calm,.50,.52)[:,None],(n,shape[1]))
    require(np.array_equal(canonical[:n],expected),'Exact original common core coefficient')
    _,_,baseline,core,cw,_,_,_=core_adapter.path_adapter._inputs('incumbent66',0.,np.full(shape,np.nan),dict(arrays,core_weight=canonical),n)
    eligible=study.adapter._matrix(arrays['eligible'],shape,n,'eligible',boolean=True)
    pre=np.asarray(study.modified_pre_score(study.configs()[4],arrays,n))
    require(pre.shape==shape and np.isfinite(pre[:n]).all(),'Finite original Local67 full pre shape')
    active=np.asarray([meta['dates'][i]!=INITIAL_PARTIAL and not calm[i] for i in range(n)],bool)
    scores={};pres={};orders={};deltas={};calls=[]
    for arm,key in zip(ARMS,(None,'candidate','control')):
        target=np.full(shape,np.nan);delta=np.zeros((n,shape[1]))
        if key is not None:
            use=active[:,None]&features['available'];target[:n][use]=features[key][use]
            delta[use]=.5*(cw[use]*features[key][use]-core[use])
        require(np.isfinite(delta).all(),'Finite core replacement delta')
        expected_pre=pre[:n].copy();nonzero=delta!=0.;expected_pre[nonzero]+=delta[nonzero]
        require(np.isfinite(expected_pre).all(),'Finite replaced pre score')
        arm_calls=[]
        def boundary(score,good,names,source_effects,original,policy):
            require(not arm_calls,'Exactly one episode per arm')
            require(np.array_equal(score,expected_pre) and np.array_equal(good,eligible),'Exact modified pre and unchanged E')
            require(names==meta['symbols'][:n] and np.array_equal(source_effects,np.asarray(arrays['effects'])[:n]),'Original symbols/effects')
            for field,expected_value in (('baseline',baseline),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
                require(np.array_equal(original[field],expected_value),'Original episode support: '+field)
            require(policy==POLICY,'Original episode policy');arm_calls.append(True);calls.append(arm)
            return study.ep.select_boundary(score,good,names,source_effects,original,policy)
        chosen,full=core_adapter.score_and_choose('incumbent66',0. if key is None else .5,target,arrays,meta,n,
            pre_score=pre,effects=arrays['effects'],episode_policy=POLICY.copy(),boundary_selector=boundary)
        require(len(arm_calls)==1 and np.shape(full)==(n,shape[1]) and np.isfinite(full).all(),'One finite full episode result')
        scores[arm]=full.copy();pres[arm]=expected_pre.copy();orders[arm]=study.rp.make_orders(chosen,full,meta)
        if key is not None:deltas[key]=delta
    events=[]
    for i in range(n):
        event=dict(query_index=i,date=meta['dates'][i],initial_partial=meta['dates'][i]==INITIAL_PARTIAL,
            calm=bool(calm[i]),active=bool(active[i]),available_indices=np.flatnonzero(features['available'][i]).tolist(),
            candidate_delta=deltas['candidate'][i].tolist(),control_delta=deltas['control'][i].tolist())
        require(tuple(event)==DECISION_EVENT_KEYS,'Decision event schema');events.append(event)
        if not active[i]:
            for arm in ARMS[1:]:
                require(orders[arm][i]==orders[ARMS[0]][i] and np.array_equal(scores[arm][i],scores[ARMS[0]][i]),'Inactive exact incumbent')
    counts=dict(events=n,symbols_per_event=shape[1],active_events=int(active.sum()),
        available_active_cells=int((active[:,None]&features['available']).sum()),
        candidate_nonzero_delta_cells=int(np.count_nonzero(deltas['candidate'])),
        control_nonzero_delta_cells=int(np.count_nonzero(deltas['control'])),original_episode_calls=len(calls),new_model_fits=0)
    require(tuple(counts)==DECISION_COUNT_FIELDS,'Decision count schema')
    require(_snapshot(arrays)==before and meta==before_meta and _snapshot(values)==before_values,'Input mutation')
    return orders,scores,pres,dict(schema='monthly_block_sharpe242_decisions_v1',counts=counts,events=events)
