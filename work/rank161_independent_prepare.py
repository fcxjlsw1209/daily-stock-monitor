"""Independent161 rank-history and score oracle. No new producer imports."""
from collections import Counter
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

NAMES = ('incumbent_local67','rank_residence_control','rank_trend_challenge')
EXTRA = HANDS = ('delta_residence','delta_rank_trend')
INITIAL = '2016-08-17'
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'

def require(ok, message):
    if not ok: raise ValueError(message)

def independent_imports():
    require(not any(x in sys.modules for x in ('rank161_kernel','rank161_runner','rank161_host')), 'New producer/kernel/host imported')

def old_helper():
    independent_imports()
    path=Path(__file__).resolve().parents[1]/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA, 'Old independent helper changed')
    spec=importlib.util.spec_from_file_location('rank161_old_independent',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def ns(text):
    require(type(text) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})',text), 'Explicit timezone timestamp required')
    delta=datetime.fromisoformat(text.replace('Z','+00:00')).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000



def calendar_positions(calendar):
    require(type(calendar) is list and calendar==sorted(set(calendar)), 'Unique ordered calendar')
    require(all(type(x) is str and date.fromisoformat(x).isoformat()==x for x in calendar), 'Official date axis')
    return {day:i for i,day in enumerate(calendar)}

def month_keys(day):
    d=date.fromisoformat(day); number=d.year*12+d.month-1
    return ['%04d-%02d'%(v//12,v%12+1) for v in range(number-5,number+1)]

def json_numbers(values):
    if isinstance(values, np.ndarray): values=values.tolist()
    if isinstance(values,list): return [json_numbers(v) for v in values]
    return float(values) if np.isfinite(values) else None

def snapshot_oracle(prices, day, names, calendar, positions):
    """Explicit scalar endpoint checks and pair-count midranks; no pandas rank."""
    t=positions[day]; required=calendar[max(0,t-294):t+1]
    date_set=set(prices.index.strftime('%Y-%m-%d'))
    missing=[d for d in required if d not in date_set]
    momentum=[]; valid=[]; warm=[]; positive=[]
    for symbol in names:
        value=math.nan; ok=False; complete=False
        if symbol in prices.columns:
            if t>=252 and calendar[t-252] in date_set and calendar[t-42] in date_set:
                start=prices.at[pd.Timestamp(calendar[t-252]),symbol]
                end=prices.at[pd.Timestamp(calendar[t-42]),symbol]
                if np.isfinite(start) and np.isfinite(end) and start>0 and end>0:
                    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                        value=float(end/start-1.)
                    ok=math.isfinite(value)
            if t>=294 and not missing:
                history=prices.loc[pd.to_datetime(required),symbol].to_numpy(dtype=float)
                complete=bool(np.isfinite(history).all() and (history>0).all())
        momentum.append(value if ok else math.nan);valid.append(ok);warm.append(complete)
        positive.append(bool(ok and value>0))
    population=[v for v,ok in zip(momentum,valid) if ok]; ranks=[]
    for v,ok in zip(momentum,valid):
        if not ok: ranks.append(math.nan);continue
        less=sum(x<v for x in population); equal=sum(x==v for x in population)
        # Match rank/N - .5 operation order, using an independently counted rank.
        rank=Fraction(2*less+equal+1,2)
        ranks.append(float(rank)/len(population)-.5)
    usable=[a and b for a,b in zip(warm,positive)]
    audit=dict(date=day,symbols=list(names),member_count=len(names),endpoint_valid_count=sum(valid),
               usable_count=sum(usable),momentum=json_numbers(momentum),centered_percentile_ranks=json_numbers(ranks),
               endpoint_valid=valid,complete_295_prices=warm,positive_momentum=positive,usable=usable,
               calendar_history_sufficient=t>=294,required_window_start=required[0],required_window_end=day,
               required_window_sessions=len(required),missing_price_sessions=missing)
    return dict(zip(names,ranks)),dict(zip(names,usable)),audit

def deltas_from_ranks(values):
    require(len(values)==6 and all(math.isfinite(v) and -.5<=v<=.5 for v in values), 'Six finite bounded ranks')
    average=math.fsum(values)/6
    deviation=math.sqrt(math.fsum((v-average)**2 for v in values)/6)
    residence=.02*min(.5,max(-.5,average-deviation))
    trend=.01*math.fsum((k-2.5)*v for k,v in enumerate(values))/4.5
    require(abs(trend)<=.01+1e-14,'Mathematical trend bound')
    return residence,min(.01,max(-.01,trend))

def features_oracle(meta, calendar, training_meta, prices, n=None):
    independent_imports()
    dates=meta['dates']; n=len(dates) if n is None else n
    require(type(n) is int and 0<=n<=len(dates) and dates==sorted(set(dates)) and dates[0]==INITIAL,'Original date prefix')
    require(len(meta['symbols'])==len(dates),'Query axes')
    positions=calendar_positions(calendar)
    require(isinstance(prices,pd.DataFrame) and isinstance(prices.index,pd.DatetimeIndex) and
            prices.index.is_unique and prices.index.is_monotonic_increasing and prices.columns.is_unique and
            prices.index.tz is None and prices.index.equals(prices.index.normalize()),'Unique daily price axes')
    require(len(training_meta['dates'])==len(training_meta['symbols']),'Historical axes')
    width=len(meta['symbols'][0]); require(width>0,'Nonempty query width')
    require(all(len(x)==width for x in meta['symbols']),'Consistent query width')
    snapshots={}
    # Complete structural validation is allowed; only used snapshots read numbers.
    for day,names in list(zip(training_meta['dates'],training_meta['symbols']))+list(zip(dates,meta['symbols'])):
        require(day in positions and type(names) is list and len(names)==len(set(names)) and
                all(type(s) is str and s for s in names),'Exact snapshot schema')
        if day in snapshots:
            require(set(snapshots[day])==set(names),'Conflicting same-date membership')
        else:snapshots[day]=list(names)
    monthly={}
    for day in sorted(snapshots):
        if day==INITIAL:continue
        month=day[:7]
        require(month not in monthly,'Multiple noninitial snapshots in calendar month')
        monthly[month]=day
    for day in monthly.values():
        t=positions[day]
        require(t+1<len(calendar) and calendar[t+1][:7]!=day[:7],'Noninitial snapshot must be an official month end')
    arrays={k:np.zeros((n,width),dtype=np.float64) for k in EXTRA}
    history=[];cache={};histogram=Counter();support_total=0
    for i,(day,names) in enumerate(zip(dates[:n],meta['symbols'][:n])):
        months=month_keys(day)
        selected=[monthly.get(m) for m in months]
        selected=[d if d is not None and d<=day else None for d in selected]
        if day==INITIAL:selected[-1]=day
        raw=np.full((6,width),np.nan); observed=np.full((6,width),np.nan);usable=np.zeros((6,width),bool)
        for k,snapshot in enumerate(selected):
            if snapshot is None:continue
            if snapshot not in cache:cache[snapshot]=snapshot_oracle(prices,snapshot,snapshots[snapshot],calendar,positions)
            ranks,good,_=cache[snapshot]
            for j,symbol in enumerate(names):
                if symbol in ranks:raw[k,j]=ranks[symbol]
                if good.get(symbol,False):observed[k,j]=ranks[symbol];usable[k,j]=True
        counts=usable.sum(axis=0);support=(counts==6)&(day!=INITIAL)
        histogram.update(int(x) for x in counts);support_total+=int(support.sum())
        for j in np.flatnonzero(support):
            arrays[EXTRA[0]][i,j],arrays[EXTRA[1]][i,j]=deltas_from_ranks(observed[:,j].tolist())
        history.append(dict(date=day,symbols=list(names),calendar_months=months,snapshots=selected,
            missing_snapshot_months=[m for m,d in zip(months,selected) if d is None],
            usable_observations=counts.astype(int).tolist(),common_support=support.tolist(),usable_mask=usable.tolist(),
            rank_trajectories=json_numbers(observed),raw_rank_trajectories=json_numbers(raw),
            initial_forced_neutral=day==INITIAL))
    audit=dict(events=n,symbols_per_event=width,months=6,minimum_observations=6,return_price_offsets=[-252,-42],
        required_complete_positive_prices=295,
        rank_basis='all original snapshot members with finite positive endpoints and finite momentum; average percentile minus 0.5',
        observation_gate='positive momentum and 295 exact official-session finite positive prices ending at the historical snapshot',
        residence_formula='0.02*clip(mean(r)-population_std(r),-0.5,0.5)',
        rank_trend_formula='0.01*sum((k-2.5)*r[k],k=0..5)/4.5; roundoff-clipped to [-0.01,0.01]',
        trajectory_order='oldest to newest; six calendar-month rows by current event symbol columns',
        partial_initialization='2016-08-17 query forced neutral; its snapshot never supplies later history',
        missing_policy='no month or price filling, no older substitutes; both additions exactly zero unless all six observations usable',
        common_support_cells=support_total,neutral_cells=n*width-support_total,
        observation_count_histogram={str(k):int(v) for k,v in sorted(histogram.items())},
        snapshots_computed=len(cache),snapshot_audits={d:cache[d][2] for d in sorted(cache)},event_history=history,
        trading_eligibility_modified=False,outcomes_used=False)
    return dict(arrays=arrays,audit=audit)

def state_values(audit):
    return deepcopy(dict(rank_trajectories=[r['rank_trajectories'] for r in audit['event_history']],
                common_support=[r['common_support'] for r in audit['event_history']]))

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
        require(np.isfinite(arrays[key]).all() and (np.abs(arrays[key])<=.01).all(),'Finite actual deltas within hard one-percent bound')
        np.testing.assert_allclose(arrays[key],expected['arrays'][key],atol=1e-12,rtol=0)
        neutral=~np.asarray([r['common_support'] for r in expected['audit']['event_history']],dtype=bool)
        require(np.all(arrays[key][neutral]==0),'Exact neutral delta positions')
    # Algebraically cancelling nonconstant paths may differ below tolerance
    # between independent summation orders. Constant paths must be exactly zero.
    for i,row in enumerate(expected['audit']['event_history']):
        for j,usable in enumerate(row['common_support']):
            if usable and len({r[j] for r in row['rank_trajectories']})==1:
                require(arrays[EXTRA[1]][i,j]==0,'Constant rank path must have exact zero trend')

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



