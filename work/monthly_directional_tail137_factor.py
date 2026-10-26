"""Pure cross-stock relative tail balance; no I/O, fitting or portfolio logic."""
import math
import numpy as np
import pandas as pd

RETURN_SESSIONS, PRICE_ROWS, STOCK_COUNT, ETF_COUNT = 21, 22, 32, 17


def _detail(reason):
    return dict(available=False, reason=reason, stock_returns=None, mean_return=None,
        centered_scale=None, positive_sum_squares_scaled=None, negative_sum_squares_scaled=None,
        positive_count=None, negative_count=None, zero_count=None, q=None)


def _finite(value, name):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite derived '+name)
    return value


def tail_balance(prices):
    """Return q and JSON-safe evidence for exactly22 official prices of32 stocks.

    Missing/nonpositive source prices disable the whole state. Unrepresentable
    arithmetic is fatal. Equal rounded endpoint returns give unavailable state;
    no absolute or relative epsilon reclassifies small nonzero dispersion.
    """
    source=np.asarray(prices)
    if source.shape != (PRICE_ROWS, STOCK_COUNT):
        raise ValueError('Exactly22 official price rows and32 stock columns required')
    if source.dtype.kind not in 'fiu':
        if source.dtype.kind!='O' or any(not isinstance(v,(int,float,np.integer,np.floating)) or
                isinstance(v,(bool,np.bool_)) for v in source.flat):
            raise ValueError('Prices must be numeric, not bool/string')
    values=np.asarray(source,dtype=np.float64)
    detail=_detail('invalid_price')
    if not (np.isfinite(values)&(values>0)).all():
        return float('nan'),detail
    r=[_finite(float(values[-1,j])/float(values[0,j])-1.,'endpoint return') for j in range(STOCK_COUNT)]
    differences=[_finite(v-r[0],'anchor difference') for v in r]
    # Divide before summing to avoid overflow of a representable mean.
    offset=_finite(math.fsum(v/STOCK_COUNT for v in differences),'mean offset')
    mean=_finite(r[0]+offset,'mean return')
    centered=[_finite(v-offset,'centered return') for v in differences]
    scale=max(abs(v) for v in centered)
    detail.update(stock_returns=r,mean_return=mean,centered_scale=scale,
                  positive_count=sum(v>0 for v in centered),negative_count=sum(v<0 for v in centered),
                  zero_count=sum(v==0 for v in centered))
    if scale==0:
        detail.update(reason='zero_dispersion',positive_sum_squares_scaled=0.,negative_sum_squares_scaled=0.)
        return float('nan'),detail
    y=[_finite(v/scale,'scaled centered return') for v in centered]
    up=_finite(math.fsum(v*v for v in y if v>0),'positive squared sum')
    down=_finite(math.fsum(v*v for v in y if v<0),'negative squared sum')
    total=_finite(up+down,'total squared sum')
    if total<=0:
        raise FloatingPointError('Nonpositive derived dispersion')
    q=_finite(up/total,'tail balance')
    if not 0<=q<=1:
        raise FloatingPointError('Tail balance outside unit interval')
    detail.update(available=True,reason='available',positive_sum_squares_scaled=up,
                  negative_sum_squares_scaled=down,q=q)
    return q,detail


def _sessions(values,name):
    index=pd.DatetimeIndex(values)
    if (index.tz is not None or index.hasnans or not index.is_unique or
            not index.is_monotonic_increasing or not index.equals(index.normalize())):
        raise ValueError(name+' must be unique sorted timezone-naive official session dates')
    return index


def build_features(prices,meta,sessions):
    """Read only each original32-stock/22-session window through its signal.

    Missing official rows are reindexed to NaN, never compressed or filled.
    Future rows and all ETF price values are outside the state calculation.
    """
    if not isinstance(prices,pd.DataFrame):
        raise ValueError('Explicit adjusted-price DataFrame required')
    axis=_sessions(prices.index,'prices'); official=_sessions(sessions,'sessions')
    dates=_sessions(meta['dates'],'signals')
    if not len(dates) or not len(official):
        raise ValueError('Nonempty signals and official sessions required')
    if (not prices.columns.is_unique or any(not isinstance(s,str) or not s for s in prices.columns)):
        raise ValueError('Unique named price columns required')
    etfs=meta['etfs']
    if len(etfs)!=ETF_COUNT or len(set(etfs))!=ETF_COUNT or any(not isinstance(s,str) or not s for s in etfs):
        raise ValueError('Exactly17 unique original ETFs required')
    universes=meta['symbols']
    if len(universes)!=len(dates):
        raise ValueError('Original signal and universe count mismatch')
    positions=official.get_indexer(dates)
    if (positions<0).any():
        raise ValueError('Every signal must be an official session')
    inspected=axis[(axis>=official[0])&(axis<=dates[-1])]
    if not inspected.isin(official).all():
        raise ValueError('Price axis contains a nonofficial historical session')
    q=np.full(len(dates),np.nan); months=[]
    for i,(day,names,pos) in enumerate(zip(dates,universes,positions)):
        if (len(names)!=49 or len(set(names))!=49 or not set(etfs)<=set(names) or
                any(not isinstance(s,str) or not s for s in names)):
            raise ValueError('Each original universe requires32 stocks and17 ETFs')
        stocks=[s for s in names if s not in set(etfs)]
        window=official[pos-RETURN_SESSIONS:pos+1] if pos>=RETURN_SESSIONS else official[:0]
        if len(window)!=PRICE_ROWS:
            detail=_detail('insufficient_history')
        elif not set(stocks)<=set(prices.columns):
            detail=_detail('missing_stock_columns')
        else:
            q[i],detail=tail_balance(prices.reindex(index=window,columns=stocks).to_numpy())
        months.append(dict(detail,signal_index=i,signal_date=str(day.date()),stock_symbols=stocks,
            window_dates=[str(x.date()) for x in window],window_start=str(window[0].date()) if len(window) else None,
            window_end=str(window[-1].date()) if len(window) else None))
    return dict(q=q,audit=dict(months=months,new_model_fits=0,return_sessions=RETURN_SESSIONS,
        price_rows=PRICE_ROWS,stock_count=STOCK_COUNT,etf_count=ETF_COUNT,no_eligibility_filter=True))
