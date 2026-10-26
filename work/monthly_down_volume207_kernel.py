"""Pure recent down-session stored-volume dispersion, without any file/model I/O.

Population CV squared uses each group's own maximum only for arithmetic scale.
The prospectively rejected normalized-HHI n-1 divisor is not applied. This
auxiliary current63 clock does not replace the incumbent skip42 momentum clock.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd

PRICE_COUNT=64
VOLUME_COUNT=63
MINIMUM_DOWN_DAYS=2
INITIAL_PARTIAL='2016-08-17'
COUNT_FIELDS=('valid_price_observations','valid_volume_observations','down_comparisons',
              'dispersion_estimates','raw_score_observations','ranked_score_observations')


def _empty(reason):
    return dict(source_available=False,available=False,reason=reason,
        candidate=float('nan'),control=float('nan'),prices=[],volumes=[],
        down_flags=[],down_indices=[],down_count=0,down_volumes=[],
        candidate_scale=None,control_scale=None,candidate_scaled=[],control_scaled=[],
        candidate_mean=None,control_mean=None,
        candidate_squared_deviation_sum=None,control_squared_deviation_sum=None,
        candidate_constant=False,control_constant=False,dispersion_estimates=0)


def _source(raw,label):
    values=np.asarray(raw,dtype=object)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) for v in values):
        return None,'nonnumeric_'+label
    try:
        values=np.asarray(values,dtype=np.float64)
    except (ValueError,TypeError,OverflowError):
        return None,'nonnumeric_'+label
    if not (np.isfinite(values)&(values>0)).all():
        return None,'invalid_'+label
    return values.tolist(),None


def _dispersion(volumes):
    """Canonical declared arithmetic; caller has validated positive finite values."""
    n=len(volumes)
    if n<2:
        raise ValueError('Dispersion requires at least two positive volumes')
    scale=max(volumes);constant=all(v==volumes[0] for v in volumes)
    scaled=[v/scale for v in volumes]
    mean=math.fsum(scaled)/n
    centered_sum=0. if constant else math.fsum((x-mean)**2 for x in scaled)
    denominator=n*mean*mean
    if (any(not math.isfinite(v) for v in scaled) or not math.isfinite(mean) or mean<=0
            or not math.isfinite(centered_sum) or centered_sum<0
            or not math.isfinite(denominator) or denominator<=0):
        raise FloatingPointError('Nonfinite or invalid scaled dispersion arithmetic')
    value=0. if constant else centered_sum/denominator
    if not math.isfinite(value) or value<0:
        raise FloatingPointError('Nonfinite or invalid population CV squared')
    return value,dict(scale=scale,scaled=scaled,mean=mean,
                      squared_deviation_sum=centered_sum,constant=constant)


def measure(prices,volume):
    """Return one exact64-price/63-volume paired statistic and source diagnostics."""
    if np.shape(prices)!=(PRICE_COUNT,) or np.shape(volume)!=(VOLUME_COUNT,):
        raise ValueError('Expected exactly64 prices and63 aligned stored volumes')
    p,reason=_source(prices,'adjusted_price')
    if reason:return _empty(reason)
    v,reason=_source(volume,'volume')
    if reason:return _empty(reason)
    flags=[p[k+1]<p[k] for k in range(VOLUME_COUNT)]
    indices=[k for k,down in enumerate(flags) if down]
    selected=[v[k] for k in indices]
    result=_empty('insufficient_down_days')
    result.update(source_available=True,prices=p,volumes=v,down_flags=flags,
                  down_indices=indices,down_count=len(indices),down_volumes=selected)
    if len(indices)<MINIMUM_DOWN_DAYS:
        return result
    candidate,ca=_dispersion(selected);control,co=_dispersion(v)
    result.update(available=True,reason='available',candidate=candidate,control=control,
                  dispersion_estimates=2)
    for name,details in (('candidate',ca),('control',co)):
        result.update({name+'_'+key:value for key,value in details.items()})
    return result


def centered_rank(raw, available):
    """Higher is preferred: average tied zero-based rank/(n-1)-0.5."""
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.zeros(raw.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) < 2:
        return result
    ranked = pd.Series(raw[indices]).rank(method='average').to_numpy()
    result[indices] = (ranked-1)/(len(indices)-1)-.5
    return result


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label+' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label+' must be a naive midnight date')
    return day


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Source requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Source sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                         for s in frame.columns)):
        raise ValueError('Source columns must be unique nonempty security IDs')


def _calendar(calendar):
    if isinstance(calendar, (str, bytes)):
        raise ValueError('Calendar requires an explicit sequence')
    dates = []
    for value in calendar:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()) or day.weekday() >= 5:
            raise ValueError('Calendar requires ISO weekday sessions')
        if dates and day <= dates[-1]:
            raise ValueError('Calendar must be unique and increasing')
        dates.append(day)
    if not dates:
        raise ValueError('Calendar cannot be empty')
    return pd.DatetimeIndex(dates)


def _queries(meta, sessions):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    if isinstance(meta['dates'], (str, bytes)) or isinstance(meta['symbols'], (str, bytes)):
        raise ValueError('Metadata dates and symbols require sequences')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF identities require an explicit sequence')
    etfs = list(meta['etfs'])
    if (any(not isinstance(s, str) or not s or s.strip() != s for s in etfs)
            or len(etfs) != len(set(etfs))):
        raise ValueError('ETF identities must be unique nonempty strings')
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        if not (not rows and str(day.date()) == INITIAL_PARTIAL):
            if t+1 >= len(sessions) or sessions[t+1].to_period('M') == day.to_period('M'):
                raise ValueError('Event must be a verified official calendar month-end')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
                or len(symbols) != len(set(symbols))):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, t))
    return rows, width or 0



def build_features(panel,meta,calendar,limit=None):
    """Use only requested official windows; rank the common full candidate domain."""
    if not isinstance(panel,dict) or set(panel)!={'adj','volume'}:
        raise ValueError('Panel requires exactly adj and volume')
    adj,volume=panel['adj'],panel['volume'];_frame(adj);_frame(volume)
    sessions=_calendar(calendar);queries,width=_queries(meta,sessions)
    if limit is not None and (type(limit) is not int or not 0<=limit<=len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries=queries if limit is None else queries[:limit]
    shape=len(queries),width
    values=dict(candidate=np.full(shape,np.nan),control=np.full(shape,np.nan),
        available=np.zeros(shape,dtype=bool),candidate_rank=np.zeros(shape),control_rank=np.zeros(shape))
    history,reasons=[],Counter();source_by_event=[];available_by_event=[]
    for i,(day,symbols,t) in enumerate(queries):
        iso=str(day.date());dates=sessions[t-63:t+1] if t>=63 else None
        vdates=dates[1:] if dates is not None else None
        missing_p=dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        missing_v=vdates.difference(volume.index).strftime('%Y-%m-%d').tolist() if vdates is not None else []
        cells=[]
        for j,symbol in enumerate(symbols):
            if dates is None:cell=_empty('insufficient_history')
            elif symbol not in adj.columns:cell=_empty('missing_price_symbol')
            elif symbol not in volume.columns:cell=_empty('missing_volume_symbol')
            elif missing_p:cell=_empty('missing_official_price_session')
            elif missing_v:cell=_empty('missing_official_volume_session')
            else:cell=measure(adj.loc[dates,symbol].to_numpy(copy=True),volume.loc[vdates,symbol].to_numpy(copy=True))
            for key in ('candidate','control','available'):values[key][i,j]=cell[key]
            reasons[cell['reason']]+=1
            cell=dict(cell,symbol=symbol,date=iso)
            if not cell['available']:cell['candidate']=cell['control']=None
            cells.append(cell)
        for arm in ('candidate','control'):
            values[arm+'_rank'][i]=centered_rank(values[arm][i],values['available'][i])
        for j,cell in enumerate(cells):
            cell.update(candidate_rank=float(values['candidate_rank'][i,j]),
                        control_rank=float(values['control_rank'][i,j]))
        source=[cell['source_available'] for cell in cells];available=values['available'][i].tolist()
        scount,acount=sum(source),sum(available);source_by_event.append(scount);available_by_event.append(acount)
        history.append(dict(date=iso,symbols=symbols,session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            volume_start=str(vdates[0].date()) if vdates is not None else None,
            volume_end=str(vdates[-1].date()) if vdates is not None else None,
            price_dates=dates.strftime('%Y-%m-%d').tolist() if dates is not None else [],
            volume_dates=vdates.strftime('%Y-%m-%d').tolist() if vdates is not None else [],
            missing_official_price_sessions=missing_p,missing_official_volume_sessions=missing_v,
            source_available=source,available=available,source_available_count=scount,
            available_count=acount,cells=cells))
    source=sum(source_by_event);available=sum(available_by_event);checked=int(values['available'].size)
    return values,dict(events=len(queries),symbols_per_event=width,checked_cells=checked,
        price_count=PRICE_COUNT,volume_count=VOLUME_COUNT,minimum_down_days=MINIMUM_DOWN_DAYS,
        source_available_cells=source,source_available_by_event=source_by_event,
        available_cells=available,available_by_event=available_by_event,unavailable_cells=checked-available,
        reason_counts=dict(reasons),valid_price_observations=PRICE_COUNT*source,
        valid_volume_observations=VOLUME_COUNT*source,down_comparisons=VOLUME_COUNT*source,
        dispersion_estimates=2*available,raw_score_observations=2*available,ranked_score_observations=2*available,
        predictive_model_fits=0,new_model_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,
        portfolio_returns_calculated=False,event_history=history)

