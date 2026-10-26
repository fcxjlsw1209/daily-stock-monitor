"""Independent160 supplied-data feature/score oracle; no producer imports or I/O phases."""
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import importlib.util
import io
from pathlib import Path
import re
import sys
import numpy as np

NAMES = ('incumbent_local67','vix_control','vvix_challenge')
EXTRA = HANDS = ('delta_vix','delta_vvix')
SOURCES = ('VIX','VVIX')
INITIAL = '2016-08-17'
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'


def require(ok, message):
    if not ok: raise ValueError(message)


def independent_imports():
    require(not any(x in sys.modules for x in ('vvix160_kernel','vvix160_runner','vvix160_host')), 'New producer/kernel/host imported')


def old_helper():
    independent_imports()
    path=Path(__file__).resolve().parents[1]/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA, 'Old independent helper changed')
    spec=importlib.util.spec_from_file_location('vvix160_old_independent',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def ns(text):
    require(type(text) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})',text), 'Explicit timezone timestamp required')
    delta=datetime.fromisoformat(text.replace('Z','+00:00')).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000


def csv_values(name, raw):
    """Preserve every source date/literal; numerical conversion occurs only in a consumed window."""
    require(name in SOURCES and type(raw) is bytes,'Raw named CSV required')
    reader=csv.reader(io.StringIO(raw.decode('utf-8'),newline=''),strict=True)
    header=['DATE','VVIX'] if name=='VVIX' else ['DATE','OPEN','HIGH','LOW','CLOSE']
    require(next(reader,None)==header,'Exact official CSV schema')
    result={}; previous=None
    for row in reader:
        require(len(row)==len(header) and re.fullmatch(r'\d{2}/\d{2}/\d{4}',row[0]),'CSV row schema')
        stamp=datetime.strptime(row[0],'%m/%d/%Y').date().isoformat()
        require(previous is None or previous<stamp,'Duplicate/out-of-order CSV date')
        result[stamp]=row[-1];previous=stamp
    require(result,'Nonempty source CSV')
    return result


def calendar_positions(calendar):
    require(calendar==sorted(set(calendar)), 'Unique ordered calendar')
    require(all(type(x) is str and date.fromisoformat(x).isoformat()==x and date.fromisoformat(x).weekday()<5 for x in calendar),'Official date axis')
    return {day:i for i,day in enumerate(calendar)}


def previous_window(calendar, signal):
    positions=calendar_positions(calendar)
    require(signal in positions,'Signal must be official session')
    index=positions[signal]
    require(index>=252,'Insufficient exact252 prior sessions')
    return calendar[index-252:index]


def midrank_state(literals):
    require(len(literals)==252,'Exactly252 observations')
    exact=[]
    for raw in literals:
        require(type(raw) is str,'Preserved CSV literal required')
        try: value=Decimal(raw)
        except InvalidOperation: raise ValueError('Invalid required source value')
        require(value.is_finite() and value>0,'Positive finite required source value')
        exact.append(value)
    floats=[float(x) for x in exact]
    require(all(np.isfinite(x) and x>0 for x in floats),'Float conversion must remain positive finite')
    last=exact[-1]; low=sum(x<last for x in exact); equal=sum(x==last for x in exact)
    require(low==sum(x<floats[-1] for x in floats) and equal==sum(x==floats[-1] for x in floats), 'Decimal/float order or tie collision')
    rank=Fraction(2*low+equal+1,2)
    state=Fraction(2,1)*(rank-1)/251-1
    require(-1<=state<=1,'Bounded centered midrank')
    return float(state),float(rank)


def features_oracle(meta, calendar, values, vol, n=None):
    dates=meta['dates']; n=len(dates) if n is None else n
    require(type(n) is int and 0<=n<=len(dates) and dates==sorted(set(dates)) and dates[0]==INITIAL,'Original date prefix')
    require(set(values)==set(SOURCES),'Exactly two raw source maps')
    positions=calendar_positions(calendar)
    raw=np.asarray(vol);require(raw.ndim==2 and raw.shape==(len(dates),49) and raw.dtype==np.float64,'Original float64 vol126 axis')
    consumed=raw[:n]
    require(np.isfinite(consumed).all() and ((consumed>=-.5)&(consumed<=.5)).all(),'Original vol rank bounds')
    states={s:np.zeros(n,dtype=np.float64) for s in SOURCES};common=np.zeros(n,bool);monthly=[]
    for i,day in enumerate(dates[:n]):
        require(day in positions,'Signal outside official calendar')
        row=dict(signal_index=i,date=day,initial=i==0,initial_partial=i==0,window_start=None,window_end=None,window_size=0)
        for source in SOURCES: row[source]=dict(passed=True,observations=0,latest_rank=None,m=0.0)
        if i:
            endpoint=positions[day]
            require(endpoint>=252,'Insufficient exact252 prior sessions')
            window=calendar[endpoint-252:endpoint]
            row.update(window_start=window[0],window_end=window[-1],window_size=252)
            for source in SOURCES:
                require(all(d in values[source] for d in window),'Missing exact required session; no stale fill')
                m,rank=midrank_state([values[source][d] for d in window])
                states[source][i]=m
                row[source]=dict(passed=True,observations=252,latest_rank=rank,m=m)
            common[i]=True
        monthly.append(row)
    arrays={key:-.02*states[source][:,None]*consumed for source,key in zip(SOURCES,HANDS)}
    if n:
        for value in arrays.values(): value[0]=0.
    return dict(arrays=arrays,states=states,common_available=common,audit=dict(monthly=monthly))


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
            require(hand.shape==shape and hand.dtype.kind=='f' and np.isfinite(hand).all() and ((hand>=-.01)&(hand<=.01)).all(),'Bounded final score delta')
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


def verify_feature_output(expected, arrays, audit, state_values):
    require(audit==expected['audit'],'Exact independent feature audit')
    require(state_values==dict(states={k:v.tolist() for k,v in expected['states'].items()},common_available=expected['common_available'].tolist()),'Exact independent state vectors')
    for key in EXTRA:
        require(arrays[key].dtype==np.float64 and arrays[key].shape==expected['arrays'][key].shape and
                arrays[key].tobytes()==expected['arrays'][key].tobytes(),'Exact independent score-delta bytes: '+key)
