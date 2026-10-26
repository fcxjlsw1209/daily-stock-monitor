"""Independent167 volume follow-through and score oracle. No new producer imports."""
from copy import deepcopy
from datetime import date, datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

NAMES = ('incumbent_local67','sector_volume_diffusion','sector_additive_control')
EXTRA = HANDS = ('delta_followthrough','delta_control')
INITIAL = '2016-08-17'
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'

def require(ok, message):
    if not ok: raise ValueError(message)

def independent_imports():
    require(not any(x in sys.modules for x in ('sector_diffusion167_kernel','sector_diffusion167_runner','sector_diffusion167_host')), 'New producer/kernel/host imported')

def old_helper():
    independent_imports()
    path=Path(__file__).resolve().parents[1]/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA, 'Old independent helper changed')
    spec=importlib.util.spec_from_file_location('sector_diffusion167_old_independent',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def ns(text):
    require(type(text) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})',text), 'Explicit timezone timestamp required')
    delta=datetime.fromisoformat(text.replace('Z','+00:00')).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000



def calendar_positions(calendar):
    require(type(calendar) is list and calendar and calendar==sorted(set(calendar)), 'Unique nonempty ordered calendar')
    require(all(type(x) is str and date.fromisoformat(x).isoformat()==x for x in calendar), 'Official date axis')
    require(all(date.fromisoformat(x).weekday()<5 for x in calendar),'Weekday official sessions')
    return {day:i for i,day in enumerate(calendar)}

def center_unit(values):
    """Scalar reductions make all near-constant checks dimensionless."""
    values=[float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):return None
    scale=max(abs(v) for v in values)
    if scale==0.:return None
    scaled=[v/scale for v in values];average=math.fsum(scaled)/len(scaled)
    centered=[v-average for v in scaled]
    norm=math.sqrt(math.fsum(v*v for v in centered))
    if norm<=1e-12*math.sqrt(math.fsum(v*v for v in scaled)):return None
    return np.asarray([v/norm for v in centered],dtype=float)


def dot(a,b):
    return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def correlation_clip(value):
    if not math.isfinite(value) or abs(value)>1.+64*np.finfo(float).eps:return None
    return min(1.,max(-1.,float(value)))


def basis_oracle(vectors):
    """SVD declares identification; QR independently supplies projection basis."""
    design=np.column_stack(vectors)
    singular=np.linalg.svd(design,full_matrices=False,compute_uv=False)
    if not np.isfinite(singular).all() or singular[-1]<=1e-12*singular[0]:return None
    basis,_=np.linalg.qr(design,mode='reduced')
    return basis


def residual_oracle(vector,basis):
    residual=vector-basis@(basis.T@vector)
    length=math.sqrt(dot(residual,residual))
    return None if length<=1e-12 or not math.isfinite(length) else residual/length


def partial_oracle(r,b,a,y,d):
    """Sector-volume interaction and matched additive sector-lag association."""
    ur,ub,ua,uy,ud=(center_unit(v) for v in (r,b,a,y,d))
    if any(v is None for v in (ur,ub,ua,uy,ud)):return None,None
    uq=center_unit(ub*ua)
    if uq is None:return None,None
    candidate=basis_oracle((ub,ua,ur,ud));control=basis_oracle((ua,ur,ud))
    if candidate is None or control is None:return None,None
    residuals=[residual_oracle(v,basis) for v,basis in
               ((uq,candidate),(uy,candidate),(ub,control),(uy,control))]
    if any(v is None for v in residuals):return None,None
    values=[correlation_clip(dot(residuals[i],residuals[i+1])) for i in (0,2)]
    return (None,None) if any(v is None for v in values) else tuple(values)


def paired_oracle(stock_adjusted,sector_adjusted,sector_volume):
    """Exactly212 closes for each security and231 sector volumes form210 pairs."""
    try:stock,sector,volume=(np.asarray(v,dtype=float) for v in (stock_adjusted,sector_adjusted,sector_volume))
    except (TypeError,ValueError,OverflowError):return None,None
    if stock.shape!=(212,) or sector.shape!=(212,) or volume.shape!=(231,):return None,None
    if any(not np.all(np.isfinite(v)&(v>0)) for v in (stock,sector,volume)):return None,None
    r=np.diff(np.log(stock));b=np.diff(np.log(sector));logs=np.log(volume)
    activity=[float(logs[i+21])-math.fsum(float(v) for v in logs[i:i+21])/21 for i in range(210)]
    return partial_oracle(r[:-1],b[:-1],activity,r[1:],b[1:])


def midranks(values):
    population=[v for v in values if v is not None]
    result=[]
    for value in values:
        if value is None:result.append(None);continue
        less=sum(x<value for x in population);equal=sum(x==value for x in population)
        result.append(float(Fraction(2*less+equal+1,2))/len(population)-.5)
    return result


def features_oracle(meta, calendar, training_meta, prices, n=None):
    """Map direct stocks only; ETFs and unknown sectors remain exactly neutral."""
    independent_imports()
    dates=meta['dates'];n=len(dates) if n is None else n
    require(type(n) is int and 0<=n<=len(dates) and dates==sorted(set(dates)) and dates and dates[0]==INITIAL,'Original date prefix')
    require(len(meta['symbols'])==len(dates),'Query axes')
    positions=calendar_positions(calendar)
    require(type(prices) is dict and set(prices)=={'adj','volume','benchmark_map'},'Exact adjusted-close/volume panel')
    for frame in (prices['adj'],prices['volume']):
        require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex) and
                not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing and frame.columns.is_unique and
                frame.index.tz is None and frame.index.equals(frame.index.normalize()) and
                all(type(symbol) is str and symbol for symbol in frame.columns),'Unique daily price axes')
        require(frame.index.equals(prices['adj'].index) and frame.columns.equals(prices['adj'].columns),'Exactly aligned source panel axes')
    require(type(prices['benchmark_map']) is dict,'Per-event benchmark mapping required')
    width=len(meta['symbols'][0]);require(width>0,'Nonempty query width')
    for day,names in zip(dates,meta['symbols']):
        require(day in positions and type(names) is list and len(names)==width and len(set(names))==width and
                all(type(name) is str and name for name in names),'Exact current pool schema')
        if day!=INITIAL:
            t=positions[day]
            require(t+1<len(calendar) and calendar[t+1][:7]!=day[:7],'Official month-end event required')
    arrays={key:np.zeros((n,width),dtype=np.float64) for key in EXTRA}
    history=[];total=0;direct_total=0
    for i,(day,names) in enumerate(zip(dates[:n],meta['symbols'][:n])):
        t=positions[day]
        start=calendar[t-253] if t>=253 else None;end=calendar[t-42] if t>=42 else None
        volume_start=calendar[t-273] if t>=273 else None;volume_end=calendar[t-43] if t>=43 else None
        pair_start=calendar[t-252] if t>=252 else None
        mapping=prices['benchmark_map'].get(day)
        require(type(mapping) is dict and set(mapping)==set(names),'Exact per-event benchmark keys')
        require(all(v is None or type(v) is str and v in {'XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY'} for v in mapping.values()),'Declared sector ETF identity')
        require(all(mapping[s] is None for s in names if s in meta.get('etfs',[])),'ETF must be neutral')
        sectors=[mapping[s] for s in names];direct=[v is not None for v in sectors]
        require(not any(s==b for s,b in zip(names,sectors)),'Sector ETF must be unmapped')
        if day!=INITIAL:direct_total+=sum(direct)
        values=[[None]*width,[None]*width]
        if t>=273:
            adjusted=prices['adj'].reindex(index=pd.to_datetime(calendar[t-253:t-41]),columns=list(dict.fromkeys(names+[b for b in sectors if b is not None])))
            volume=prices['volume'].reindex(index=pd.to_datetime(calendar[t-273:t-42]),columns=list(dict.fromkeys(b for b in sectors if b is not None)))
            for j,name in enumerate(names):
                if direct[j]:values[0][j],values[1][j]=paired_oracle(adjusted[name].to_numpy(),adjusted[sectors[j]].to_numpy(),volume[sectors[j]].to_numpy())
        ranks=[midranks(row) for row in values];initial=day==INITIAL
        support=[a is not None and c is not None and not initial for a,c in zip(*values)];total+=sum(support)
        for j,valid in enumerate(support):
            if valid:
                for k,key in enumerate(EXTRA):arrays[key][i,j]=.04*ranks[k][j]
        history.append(dict(date=day,symbols=list(names),window_start=start,window_end=end,
            volume_window_start=volume_start,volume_window_end=volume_end,pair_start=pair_start,pair_end=volume_end,
            eligible_direct_stock=direct,sector_symbols=sectors,common_support=support,raw_followthrough=values[0],raw_control=values[1],
            ranks_followthrough=ranks[0],ranks_control=ranks[1],initial_forced_neutral=initial))
    audit=dict(events=n,symbols_per_event=width,window_returns=210,skip_sessions=42,
        window_pairs=210,adjusted_window_prices=212,volume_window_values=231,activity_baseline_sessions=21,
        relative_degeneracy_epsilon=1e-12,common_support_cells=total,eligible_direct_stock_cells=direct_total,neutral_cells=n*width-total,
        trading_eligibility_modified=False,outcomes_used=False,event_history=history)
    return dict(arrays=arrays,audit=audit)


def poison_outside_windows(prices,meta,calendar,n):
    """Poison outside the exact union of independently required prefix windows."""
    positions=calendar_positions(calendar);kept={key:set() for key in ('adj','volume')}
    for day in meta['dates'][:n]:
        t=positions[day]
        if t>=273:
            kept['adj'].update(calendar[t-253:t-41]);kept['volume'].update(calendar[t-273:t-42])
    result={key:prices[key].copy() for key in ('adj','volume')}
    result['benchmark_map']={day:deepcopy(mapping) for day,mapping in prices['benchmark_map'].items() if day in meta['dates'][:n]}
    for key in ('adj','volume'):
        frame=result[key];frame.loc[~frame.index.isin(pd.to_datetime(sorted(kept[key])))]=np.nan
    return result


def state_values(audit):
    return deepcopy({key:[row[key] for row in audit['event_history']]
                     for key in ('raw_followthrough','raw_control','ranks_followthrough','ranks_control','common_support')})

def compare(actual,expected,label='root'):
    if isinstance(expected,dict):
        require(type(actual) is dict and set(actual)==set(expected),'Exact dictionary keys: '+label)
        for key in expected:compare(actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,list):
        require(type(actual) is list and len(actual)==len(expected),'Exact list size: '+label)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'/'+str(i))
    elif type(expected) is float:
        require(type(actual) in (float,int) and math.isfinite(actual) and abs(actual-expected)<=1e-12,'Finite numeric agreement: '+label)
    else:require(type(actual) is type(expected) and actual==expected,'Exact metadata agreement: '+label)

def verify_feature_output(expected,arrays,audit,saved_states):
    compare(audit,expected['audit'],'FEATURE_AUDIT')
    compare(saved_states,state_values(expected['audit']),'STATE_VALUES')
    for key in EXTRA:
        require(arrays[key].dtype==np.float64 and arrays[key].shape==expected['arrays'][key].shape,'Delta shape/dtype')
        require(np.isfinite(arrays[key]).all() and (np.abs(arrays[key])<=.02).all(),'Finite actual deltas within hard two-percent bound')
        np.testing.assert_allclose(arrays[key],expected['arrays'][key],atol=1e-12,rtol=0)
        neutral=~np.asarray([r['common_support'] for r in expected['audit']['event_history']],dtype=bool)
        require(np.all(arrays[key][neutral]==0),'Exact neutral delta positions')
    for i,row in enumerate(expected['audit']['event_history']):
        for key,rank in zip(EXTRA,('ranks_followthrough','ranks_control')):
            for j,value in enumerate(row[rank]):
                if value==0:require(arrays[key][i,j]==0,'Exact zero-rank delta')


def verify_original_arrays(original, prepared):
    require(len(original)==27 and set(prepared)==set(original)|set(EXTRA),'Original27 plus exact2 arrays')
    for key,value in original.items():
        other=prepared[key]
        require(value.shape==other.shape and value.dtype==other.dtype and value.tobytes()==other.tobytes(),'Original array changed: '+key)
    require(original['baseline'].shape==(122,49),'Original full array axis')


def score_oracle(arrays, meta, hands, n=None):
    helper=old_helper();state=helper.original_prefix(arrays,meta,n)
    shape,count=state['local'].shape,state['n']
    scores={};choices={};deltas={};episodes={}
    for index,name in enumerate(NAMES):
        pre=state['local'].copy();delta=np.zeros(shape)
        if index:
            hand=np.asarray(hands[HANDS[index-1]])[:count]
            require(hand.shape==shape and hand.dtype.kind=='f' and np.isfinite(hand).all() and ((hand>=-.02)&(hand<=.02)).all(),'Bounded final score delta')
            for i,day in enumerate(meta['dates'][:count]):
                if day==INITIAL:require(not hand[i].any(),'Neutral initial partial')
                require(all(hand[i,j]==0. for j,symbol in enumerate(meta['symbols'][i]) if symbol in meta['etfs']),'ETF score delta must be exactly neutral')
            delta=hand.copy();active=delta!=0.;pre[active]+=delta[active]
        selected,final,records=helper.episode(pre,state)
        scores[name]=final;choices[name]=selected;deltas[name]=delta;episodes[name]=records
    return dict(scores=scores,choices=choices,deltas=deltas,episodes=episodes)


def order_oracle(meta,result):
    return {name:[dict(date=meta['dates'][i],signal_date=meta['dates'][i],end=meta['ends'][i],
        symbols=[meta['symbols'][i][j] for j in chosen],scores=[float(result['scores'][name][i,j]) for j in chosen])
        for i,chosen in enumerate(result['choices'][name])] for name in NAMES}



def verify_orders(actual, expected):
    """Require exact ordered holdings and dates; compare scores at panel tolerance."""
    require(type(actual) is dict and set(actual)==set(expected)==set(NAMES),'Exact order arm schema')
    for name in NAMES:
        require(type(actual[name]) is list and len(actual[name])==len(expected[name]),'Exact order count')
        for observed,wanted in zip(actual[name],expected[name]):
            require(type(observed) is dict and set(observed)==set(wanted)=={'date','signal_date','end','symbols','scores'},'Exact order row schema')
            require(all(observed[key]==wanted[key] for key in ('date','signal_date','end','symbols')),'Exact ordered holdings and dates')
            scores=observed['scores']
            require(type(scores) is list and len(scores)==len(wanted['scores']) and
                    all(type(x) in (float,int) and np.isfinite(x) for x in scores),'Finite aligned order scores')
            np.testing.assert_allclose(scores,wanted['scores'],atol=1e-12,rtol=0)



