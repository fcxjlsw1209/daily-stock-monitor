"""Independent165 sector-recovery and score oracle. No new producer imports."""
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

NAMES = ('incumbent_local67','recovery_race','impact_control')
EXTRA = HANDS = ('delta_recovery','delta_control')
INITIAL = '2016-08-17'
SECTOR_ETFS = ('XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY')
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'

def require(ok, message):
    if not ok: raise ValueError(message)

def independent_imports():
    require(not any(x in sys.modules for x in ('recovery165_kernel','recovery165_runner','recovery165_host')), 'New producer/kernel/host imported')

def old_helper():
    independent_imports()
    path=Path(__file__).resolve().parents[1]/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA, 'Old independent helper changed')
    spec=importlib.util.spec_from_file_location('recovery165_old_independent',path)
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

def recovery_time(prices,start):
    """First future session at/above the close immediately before the decline."""
    threshold=float(prices[start-1])
    for delay in range(1,22):
        if prices[start+delay]>=threshold:return delay
    return 22


def positive_path(values):
    if np.ndim(values)!=1 or len(values)!=211:return None
    try:result=[float(v) for v in values]
    except (TypeError,ValueError,OverflowError):return None
    return result if all(math.isfinite(v) and v>0 for v in result) else None


def paired_oracle(stock,sector):
    """Scalar event scan over all 189 eligible starts; overlapping events remain."""
    stock,sector=positive_path(stock),positive_path(sector)
    if stock is None or sector is None:return None,None,None
    differences=[];controls=[]
    for start in range(1,190):
        if stock[start]<stock[start-1] and sector[start]<sector[start-1]:
            differences.append(recovery_time(sector,start)-recovery_time(stock,start))
            a=(stock[start]-stock[start-1])/stock[start-1]
            b=(sector[start]-sector[start-1])/sector[start-1]
            controls.append((a-b)/(abs(a)+abs(b)))
    count=len(differences)
    if count<20:return None,None,count
    return float(Fraction(sum(differences),21*count)),math.fsum(controls)/count,count


def midranks(values):
    population=[v for v in values if v is not None]
    result=[]
    for value in values:
        if value is None:result.append(None);continue
        less=sum(x<value for x in population);equal=sum(x==value for x in population)
        result.append(float(Fraction(2*less+equal+1,2))/len(population)-.5)
    return result


def features_oracle(meta, calendar, sector_mapping, prices, n=None):
    """Use current-date frozen own-sector mapping and only skip-42 price history."""
    independent_imports()
    dates=meta['dates'];n=len(dates) if n is None else n
    require(type(n) is int and 0<=n<=len(dates) and dates==sorted(set(dates)) and dates and dates[0]==INITIAL,'Original date prefix')
    require(len(meta['symbols'])==len(dates),'Query axes')
    positions=calendar_positions(calendar)
    require(isinstance(prices,pd.DataFrame) and isinstance(prices.index,pd.DatetimeIndex) and
            not prices.index.hasnans and prices.index.is_unique and prices.index.is_monotonic_increasing and prices.columns.is_unique and
            prices.index.tz is None and prices.index.equals(prices.index.normalize()) and
            all(type(s) is str and s for s in prices.columns),'Unique daily price axes')
    require(type(sector_mapping) is dict and set(sector_mapping)=={'etfs','by_date'},'Exact sector mapping schema')
    etfs=sector_mapping['etfs'];mapping=sector_mapping['by_date']
    require(type(etfs) is list and len(etfs)==len(set(etfs)) and all(type(v) is str and v for v in etfs),'Explicit unique ETF exclusion')
    require(type(mapping) is dict and set(mapping)==set(dates),'Exact mapping date axes')
    width=len(meta['symbols'][0]);require(width>0,'Nonempty query width')
    for day,names in zip(dates,meta['symbols']):
        require(day in positions and type(names) is list and len(names)==width and len(set(names))==width and
                all(type(name) is str and name for name in names),'Exact current pool schema')
        require(type(mapping[day]) is dict and set(mapping[day])==set(names) and
                all(v is None or (type(v) is str and v in SECTOR_ETFS) for v in mapping[day].values()),'Exact current pool own-sector mapping')
        require(all(mapping[day][name] is None for name in names if name in etfs or name in SECTOR_ETFS),'ETF candidates must have neutral mapping')
        if day!=INITIAL:
            t=positions[day]
            require(t+1<len(calendar) and calendar[t+1][:7]!=day[:7],'Official month-end event required')
    arrays={key:np.zeros((n,width),dtype=np.float64) for key in EXTRA}
    history=[];total=0
    for i,(day,names) in enumerate(zip(dates[:n],meta['symbols'][:n])):
        t=positions[day];start=calendar[t-252] if t>=252 else None;end=calendar[t-42] if t>=42 else None
        sectors=[mapping[day][name] for name in names]
        values=[[None]*width,[None]*width];counts=[None]*width
        reasons=['etf_not_applicable' if name in etfs or name in SECTOR_ETFS else
                 'missing_sector_mapping' if sector is None else 'insufficient_history'
                 for name,sector in zip(names,sectors)]
        if t>=252:
            block=prices.reindex(index=pd.to_datetime(calendar[t-252:t-41]))
            for j,(name,bench) in enumerate(zip(names,sectors)):
                if bench is None:continue
                stock=positive_path(block[name].to_numpy()) if name in block else None
                sector=positive_path(block[bench].to_numpy()) if bench in block else None
                if stock is None:reasons[j]='invalid_stock_path'
                elif sector is None:reasons[j]='invalid_sector_path'
                else:
                    values[0][j],values[1][j],counts[j]=paired_oracle(stock,sector)
                    reasons[j]='available' if counts[j]>=20 else 'insufficient_joint_down'
        ranks=[midranks(row) for row in values];initial=day==INITIAL
        support=[a is not None and c is not None and not initial for a,c in zip(*values)];total+=sum(support)
        for j,valid in enumerate(support):
            if valid:
                for k,key in enumerate(EXTRA):arrays[key][i,j]=.04*ranks[k][j]
        history.append(dict(date=day,symbols=list(names),window_start=start,window_end=end,
            sector_etfs=sectors,joint_down_counts=counts,source_status=reasons,common_support=support,
            raw_recovery=values[0],raw_control=values[1],ranks_recovery=ranks[0],ranks_control=ranks[1],
            initial_forced_neutral=initial))
    audit=dict(events=n,symbols_per_event=width,window_prices=211,window_returns=210,skip_sessions=42,
        start_count=189,recovery_horizon=21,censor_value=22,minimum_joint_down=20,
        common_support_cells=total,neutral_cells=n*width-total,trading_eligibility_modified=False,
        outcomes_used=False,event_history=history)
    return dict(arrays=arrays,audit=audit)


def state_values(audit):
    return deepcopy({key:[row[key] for row in audit['event_history']]
                     for key in ('raw_recovery','raw_control','ranks_recovery','ranks_control','joint_down_counts','sector_etfs','common_support')})

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
        for key,rank in zip(EXTRA,('ranks_recovery','ranks_control')):
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



