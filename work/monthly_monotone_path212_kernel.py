"""Descriptive monotone versus nonnegative affine path quality, pure source input.

Both targets normalize the same skip42 numerator by current63 sample risk.
PAVA and OLS describe the observed formation path; they do not fit forecasts.
Initial/calm targets are computed here; only the host suppresses their deltas.
"""
from collections import Counter
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PRICE_COUNT, FORMATION_COUNT, RETURN_COUNT, SKIP = 295, 253, 63, 42
QUALITY_TOLERANCE = 1e-12
VALUE_KEYS = ('candidate','control','raw_candidate','raw_control','numerator','volatility','q_iso','q_line','available')
COUNT_FIELDS = ('valid_price_observations','formation_log_observations','simple_return_observations',
    'numerator_evaluations','risk_evaluations','descriptive_pava_fits','descriptive_affine_fits','ranked_observations')


def dependency_files():
    return (Path(__file__).resolve(),Path(__file__).with_name('monthly_nested_trend199_kernel.py').resolve())


def _finite(value, label):
    if not math.isfinite(value): raise FloatingPointError('Nonfinite '+label+' from valid finite source')
    return value


def _sum(values, label):
    try: return _finite(math.fsum(values),label)
    except (OverflowError,ValueError) as exc: raise FloatingPointError('Nonfinite '+label) from exc


def _squares(values, label):
    return [_finite(x*x,label) for x in values]


def empty_measure(reason):
    return dict(source_available=False,available=False,reason=reason,
        start_price=None,formation_end_price=None,risk_start_price=None,end_price=None,
        numerator=None,simple_returns=[],return_mean=None,volatility=None,
        formation_logs=[],log_mean=None,centered_logs=[],centered_mean=None,sst=None,
        blocks=[],fitted_iso=[],fitted_line=[],ols_slope=None,ols_intercept=None,
        sse_iso=None,sse_line=None,q_iso=None,q_line=None,q_iso_unclipped=None,q_line_unclipped=None,
        iso_quality_branch=None,line_quality_branch=None,normalized_momentum=None,
        raw_candidate=np.nan,raw_control=np.nan,attenuation_applied=False,
        descriptive_pava_fits=0,descriptive_affine_fits=0)


def pava(values):
    """Equal-weight increasing projection; equal adjacent means form maximal blocks."""
    if np.ndim(values)!=1 or not len(values): raise ValueError('Nonempty scalar path required')
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) for v in values):
        raise ValueError('Nonboolean numeric path required')
    z=[float(v) for v in values]
    if not all(math.isfinite(v) for v in z): raise ValueError('Finite path required')
    blocks=[]
    for j,v in enumerate(z):
        blocks.append(dict(start_index=j,end_index=j,count=1,mean=v))
        while len(blocks)>1 and blocks[-2]['mean']>=blocks[-1]['mean']:
            right=blocks.pop();left=blocks.pop();start,end=left['start_index'],right['end_index'];count=end-start+1
            mean=_finite(_sum(z[start:end+1],'PAVA block sum')/count,'PAVA block mean')
            blocks.append(dict(start_index=start,end_index=end,count=count,mean=mean))
    fitted=[0.]*len(z)
    for block in blocks:
        for j in range(block['start_index'],block['end_index']+1): fitted[j]=block['mean']
    return fitted,blocks


def quality(sse,sst,zero_branch=None):
    """Only declared roundoff outside [0,1] is clipped; no denominator floor."""
    if not all(isinstance(v,numbers.Real) and not isinstance(v,(bool,np.bool_)) for v in (sse,sst)):
        raise ValueError('Numeric SSE/SST required')
    sse,sst=float(sse),float(sst)
    _finite(sse,'SSE');_finite(sst,'SST')
    if sse<0 or sst<=0:raise ValueError('Nonnegative SSE and strictly positive SST required')
    if zero_branch is not None:
        if zero_branch not in ('single_block','zero_slope'):raise ValueError('Unknown exact zero branch')
        return 0.,0.,zero_branch
    if sse==0.:return 1.,1.,'zero_sse'
    raw=_finite(1.-sse/sst,'Path quality')
    if not -QUALITY_TOLERANCE<=raw<=1.+QUALITY_TOLERANCE:
        raise FloatingPointError('Path quality exceeds declared roundoff bound')
    return min(1.,max(0.,raw)),raw,'residual_ratio'


def path_fits(z,zmean,sst):
    """Canonical descriptive fits for an already centered positive-SST path."""
    fitted,blocks=pava(z);n=len(z)
    if n<2 or sst<=0:raise ValueError('Positive path variation and two observations required')
    time=[float(j)-(n-1)/2. for j in range(n)]
    denominator=_sum(_squares(time,'Centered time square'),'OLS time denominator')
    covariance=_sum([_finite(t*_finite(v-zmean,'Centered log residual'),'OLS cross product') for t,v in zip(time,z)],'OLS numerator')
    beta=max(0.,_finite(covariance/denominator,'OLS slope'))
    line=[_finite(zmean+_finite(beta*t,'OLS fitted product'),'OLS fitted value') for t in time]
    sse_iso=_sum(_squares([_finite(v-f,'PAVA residual') for v,f in zip(z,fitted)],'PAVA residual square'),'PAVA SSE')
    sse_line=_sum(_squares([_finite(v-f,'OLS residual') for v,f in zip(z,line)],'OLS residual square'),'OLS SSE')
    qi,ui,bi=quality(sse_iso,sst,'single_block' if len(blocks)==1 else None)
    ql,ul,bl=quality(sse_line,sst,'zero_slope' if beta==0. else None)
    return dict(blocks=blocks,fitted_iso=fitted,fitted_line=line,ols_slope=beta,ols_intercept=zmean,
        sse_iso=sse_iso,sse_line=sse_line,q_iso=qi,q_line=ql,q_iso_unclipped=ui,q_line_unclipped=ul,
        iso_quality_branch=bi,line_quality_branch=bl,descriptive_pava_fits=1,descriptive_affine_fits=1)


def measure(prices):
    if np.shape(prices)!=(PRICE_COUNT,):raise ValueError('Exactly295 adjusted prices required')
    raw=np.asarray(prices,dtype=object)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) for v in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:p=[float(v) for v in raw]
    except (ValueError,TypeError,OverflowError):return empty_measure('nonnumeric_adjusted_price')
    if not all(math.isfinite(v) and v>0 for v in p):return empty_measure('invalid_adjusted_price')
    try:
        numerator=_finite(p[252]/p[0]-1.,'Long momentum numerator')
        returns=[_finite(p[j]/p[j-1]-1.,'Current simple return') for j in range(232,295)]
        mean=_finite(_sum(returns,'Return sum')/63.,'Return mean')
        variance=_finite(_sum(_squares([_finite(v-mean,'Return residual') for v in returns],'Return residual square'),'Return SSE')/62.,'Sample return variance')
        vol=_finite(math.sqrt(variance),'Sample volatility')
        logs=[_finite(math.log(v),'Formation log') for v in p[:253]]
        log_mean=_finite(_sum(logs,'Formation log sum')/253.,'Formation log mean')
        z=[_finite(v-log_mean,'Centered formation log') for v in logs]
        zmean=_finite(_sum(z,'Centered log sum')/253.,'Residual centered mean')
        sst=_sum(_squares([_finite(v-zmean,'Centered log residual') for v in z],'Centered log square'),'Formation SST')
        result=empty_measure('zero_volatility' if vol==0. else 'zero_formation_variance')
        result.update(source_available=True,start_price=p[0],formation_end_price=p[252],risk_start_price=p[231],
            end_price=p[294],numerator=numerator,simple_returns=returns,return_mean=mean,volatility=vol,
            formation_logs=logs,log_mean=log_mean,centered_logs=z,centered_mean=zmean,sst=sst)
        if vol==0. or sst==0.:return result
        fits=path_fits(z,zmean,sst)
        normalized=_finite(numerator/vol,'Risk-normalized momentum')
        q=_finite(normalized*fits['q_iso'],'Monotone adjusted momentum') if normalized>0. else normalized
        c=_finite(normalized*fits['q_line'],'Affine adjusted momentum') if normalized>0. else normalized
        result.update(fits);result.update(available=True,reason='available',normalized_momentum=normalized,
            raw_candidate=q,raw_control=c,attenuation_applied=normalized>0.)
        return result
    except OverflowError as exc:raise FloatingPointError('Nonfinite arithmetic from valid finite prices') from exc


def percentile_rank(raw,available):
    raw,available=np.asarray(raw),np.asarray(available)
    if raw.ndim!=1 or raw.shape!=available.shape or available.dtype!=bool or not np.isfinite(raw[available]).all():
        raise ValueError('Aligned finite raw values and boolean availability required')
    result=np.full(raw.shape,np.nan);ids=np.flatnonzero(available)
    if len(ids):result[ids]=pd.Series(raw[ids]).rank(method='average',pct=True).to_numpy()-.5
    return result


def build_features(panel,meta,calendar,limit=None):
    if not isinstance(panel,dict) or set(panel)!={'adj'}:raise ValueError('Panel requires exactly adj')
    adj=panel['adj'];_frame(adj);sessions=_calendar(calendar);queries,width=_queries(meta,sessions)
    if limit is not None and (type(limit) is not int or not 0<=limit<=len(queries)):raise ValueError('Invalid event prefix')
    queries=queries if limit is None else queries[:limit];shape=(len(queries),width)
    values={k:np.full(shape,np.nan) for k in VALUE_KEYS if k!='available'};values['available']=np.zeros(shape,bool)
    history=[];reasons=Counter();source_counts=[];counts=[]
    for i,(day,symbols,t) in enumerate(queries):
        iso=str(day.date());dates=sessions[t-294:t+1] if t>=294 else None
        missing=dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells=[]
        for j,symbol in enumerate(symbols):
            if dates is None:result=empty_measure('insufficient_history')
            elif symbol not in adj.columns:result=empty_measure('missing_symbol')
            elif missing:result=empty_measure('missing_official_session')
            else:result=measure(adj.loc[dates,symbol].to_numpy(copy=True))
            values['available'][i,j]=result['available']
            if result['available']:
                for key in ('raw_candidate','raw_control','numerator','volatility','q_iso','q_line'):values[key][i,j]=result[key]
            cell=dict(result,symbol=symbol,date=iso,candidate=None,control=None)
            if not result['available']:cell.update(raw_candidate=None,raw_control=None)
            cells.append(cell)
        for key in ('candidate','control'):values[key][i]=percentile_rank(values['raw_'+key][i],values['available'][i])
        for j,cell in enumerate(cells):
            if cell['available']:cell.update(candidate=float(values['candidate'][i,j]),control=float(values['control'][i,j]))
            reasons[cell['reason']]+=1
        source=sum(c['source_available'] for c in cells);available=int(values['available'][i].sum())
        source_counts.append(source);counts.append(available)
        def stamp(index):return str(dates[index].date()) if dates is not None else None
        history.append(dict(date=iso,symbols=symbols,session_position=t,price_start=stamp(0),price_end=stamp(294),
            numerator_start=stamp(0),numerator_end=stamp(252),formation_price_start=stamp(0),formation_price_end=stamp(252),
            risk_price_start=stamp(231),risk_price_end=stamp(294),missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells],available=[c['available'] for c in cells],
            source_available_count=source,available_count=available,cells=cells))
    s,a=sum(source_counts),sum(counts)
    return values,dict(events=len(queries),symbols_per_event=width,checked_cells=values['available'].size,
        price_count=295,formation_price_count=253,skip_sessions=42,current_return_count=63,ddof=1,
        quality_roundoff_tolerance=QUALITY_TOLERANCE,source_available_cells=s,source_available_by_event=source_counts,
        available_cells=a,available_by_event=counts,unavailable_cells=values['available'].size-a,
        valid_price_observations=295*s,formation_log_observations=253*s,simple_return_observations=63*s,
        numerator_evaluations=s,risk_evaluations=s,descriptive_pava_fits=a,descriptive_affine_fits=a,ranked_observations=2*a,
        new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,portfolio_returns_calculated=False,
        reason_counts=dict(reasons),event_history=history)
