"""Same-formation compounded momentum divided by cubic/quadratic downside risk.

Pure source-to-feature calculation, no I/O, eligibility, predictive fits or
portfolio execution. Actual use requires the prospective host PLAN.
"""
from collections import Counter
import math
import numbers
import numpy as np
import pandas as pd
from monthly_risk_consensus219_kernel import _frame, _calendar, _queries

PRICE_COUNT, SKIP, WINDOW_RETURNS = 253, 42, 252
FEATURE_KEYS = ('available','candidate','control','momentum','cubic_risk',
                'quadratic_risk','raw_candidate','raw_control')
COUNT_FIELDS = ('checked_cells','source_available_cells','available_cells','unavailable_cells',
    'valid_price_observations','simple_returns','downside_moment_evaluations',
    'normalized_momentum_scores','ranked_observations')
FLOAT_FIELDS = ('momentum','cubic_risk','quadratic_risk','raw_candidate','raw_control')


def empty_measure(reason):
    return dict(source_available=False,available=False,reason=reason,
        start_price=None,end_price=None,momentum=np.nan,negative_day_count=None,
        downside_square_sum=None,downside_cube_sum=None,cubic_risk=np.nan,
        quadratic_risk=np.nan,raw_candidate=np.nan,raw_control=np.nan)


def finite(x):
    x=float(x)
    if not math.isfinite(x):raise FloatingPointError('Nonfinite calculation from finite source')
    return x


def measure(prices):
    """253 complete prices. Loss moments divide by all252 sessions, not down days.

    Deterministic binary64 ratios, explicit products, fsum, sqrt and np.cbrt.
    No negative simple return means common missing risk, never infinite score.
    Negative and zero cumulative momentum remain valid when risk is positive.
    """
    if np.shape(prices)!=(PRICE_COUNT,):raise ValueError('Exactly253 adjusted prices required')
    raw=np.asarray(prices,dtype=object)
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Real) for x in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:p=[float(x) for x in raw]
    except (TypeError,ValueError,OverflowError):return empty_measure('nonnumeric_adjusted_price')
    if not all(math.isfinite(x) and x>0. for x in p):return empty_measure('invalid_adjusted_price')
    try:
        momentum=finite(p[-1]/p[0]-1.)
        returns=[finite(p[i+1]/p[i]-1.) for i in range(WINDOW_RETURNS)]
        losses=[-x if x<0. else 0. for x in returns]
        n=sum(x>0. for x in losses)
        squares=[finite(x*x) for x in losses]
        cubes=[finite((x*x)*x) for x in losses]
        s2,s3=finite(math.fsum(squares)),finite(math.fsum(cubes))
        d2=finite(math.sqrt(s2/WINDOW_RETURNS))
        d3=finite(np.cbrt(s3/WINDOW_RETURNS))
        result=dict(source_available=True,available=False,reason='no_negative_returns',
            start_price=p[0],end_price=p[-1],momentum=momentum,negative_day_count=n,
            downside_square_sum=s2,downside_cube_sum=s3,cubic_risk=d3,quadratic_risk=d2,
            raw_candidate=np.nan,raw_control=np.nan)
        if n==0:
            if any(x!=0. for x in (s2,s3,d2,d3)):raise FloatingPointError('Loss-free source has nonzero downside')
            return result
        if min(s2,s3,d2,d3)<=0.:raise FloatingPointError('Nonzero downside underflowed')
        result.update(available=True,reason='available',
            raw_candidate=finite(momentum/d3),raw_control=finite(momentum/d2))
        return result
    except (OverflowError,ZeroDivisionError) as exc:
        raise FloatingPointError('Arithmetic failure from finite source') from exc


def percentile_rank(raw,available):
    raw,available=np.asarray(raw),np.asarray(available)
    if (raw.ndim!=1 or raw.shape!=available.shape or available.dtype!=bool or
        not np.isfinite(raw[available]).all()):raise ValueError('Finite shared rank domain required')
    result=np.full(raw.shape,np.nan)
    ids=np.flatnonzero(available)
    if len(ids):result[ids]=pd.Series(raw[ids]).rank(method='average',pct=True).to_numpy()-.5
    return result


def build_features(panel,meta,calendar,limit=None):
    if not isinstance(panel,dict) or set(panel)!={'adj'}:raise ValueError('Exactly adj panel required')
    adj=panel['adj'];_frame(adj)
    sessions=_calendar(calendar);queries,width=_queries(meta,sessions)
    if limit is not None and (type(limit) is not int or not 0<=limit<=len(queries)):
        raise ValueError('Integer event prefix required')
    queries=queries if limit is None else queries[:limit]
    shape=len(queries),width
    values={k:np.full(shape,np.nan) for k in FEATURE_KEYS if k!='available'}
    values['available']=np.zeros(shape,bool)
    history,reasons,source_counts,counts=[],Counter(),[],[]
    for i,(day,symbols,t) in enumerate(queries):
        iso=str(day.date())
        dates=sessions[t-294:t-42+1] if t>=294 else None
        missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells=[]
        for j,symbol in enumerate(symbols):
            if dates is None:row=empty_measure('insufficient_history')
            elif symbol not in adj.columns:row=empty_measure('missing_symbol')
            elif missing:row=empty_measure('missing_official_session')
            else:row=measure(adj.loc[dates,symbol].to_numpy(copy=True))
            values['available'][i,j]=row['available']
            for key in FLOAT_FIELDS:values[key][i,j]=row[key]
            cell=dict(row,symbol=symbol,date=iso,candidate=None,control=None)
            for key in FLOAT_FIELDS:
                if not math.isfinite(cell[key]):cell[key]=None
            cells.append(cell)
        for key in ('candidate','control'):
            values[key][i]=percentile_rank(values['raw_'+key][i],values['available'][i])
        for j,cell in enumerate(cells):
            if cell['available']:cell.update(candidate=float(values['candidate'][i,j]),control=float(values['control'][i,j]))
            reasons[cell['reason']]+=1
        source=sum(c['source_available'] for c in cells);a=int(values['available'][i].sum())
        source_counts.append(source);counts.append(a)
        history.append(dict(date=iso,symbols=symbols,session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            first_return_date=str(dates[1].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells],available=[c['available'] for c in cells],
            source_available_count=source,available_count=a,cells=cells))
    source_total,total=sum(source_counts),sum(counts)
    audit=dict(events=len(queries),symbols_per_event=width,checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT,skip_sessions=SKIP,window_returns=WINDOW_RETURNS,
        downside_threshold=0.,moment_denominator=WINDOW_RETURNS,moment_orders=[3,2],
        source_available_cells=source_total,source_available_by_event=source_counts,
        available_cells=total,available_by_event=counts,
        unavailable_cells=int(values['available'].size)-total,reason_counts=dict(reasons),
        valid_price_observations=PRICE_COUNT*source_total,simple_returns=WINDOW_RETURNS*source_total,
        downside_moment_evaluations=2*source_total,normalized_momentum_scores=2*total,ranked_observations=2*total,
        predictive_model_fits=0,new_model_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,
        portfolio_returns_calculated=False,event_history=history)
    return values,audit
