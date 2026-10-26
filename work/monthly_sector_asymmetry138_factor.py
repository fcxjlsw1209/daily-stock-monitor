"""Pure conditional sector beta/correlation features; no I/O or portfolio work."""
import math
import numpy as np
import pandas as pd

MAPPING_SHA='ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
EXTRA=('raw_beta_gap','raw_correlation_gap','rank_beta_gap','rank_correlation_gap','common_available')


def _require(ok,message):
    if not ok: raise ValueError(message)


def _finite(value,label):
    if not math.isfinite(value): raise FloatingPointError('Nonfinite '+label)
    return value


def _numeric(values,shape):
    raw=np.asarray(values)
    _require(raw.shape==shape,'Exact numeric shape required')
    _require(all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in raw.flat),'Numeric nonboolean source required')
    return np.asarray(raw,dtype=np.float64)


def log_returns(prices):
    """Exact round60 frexp/NumPy log implementation; no direct price ratio."""
    p=_numeric(prices,(253,))
    _require((np.isfinite(p)&(p>0)).all(),'Complete positive finite prices required')
    mantissa,exponent=np.frexp(p)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        r=np.log(mantissa[1:]/mantissa[:-1])+np.diff(exponent)*np.log(2.)
    if not np.isfinite(r).all(): raise FloatingPointError('Nonfinite log return')
    return r


def _center(values):
    v=[float(x) for x in values]
    d=[_finite(x-v[0],'anchor difference') for x in v]
    anchor_scale=max(map(abs,d))
    if anchor_scale==0: return v[0],0.,[0.]*len(v)
    z=[x/anchor_scale for x in d]
    mid=math.fsum(z)/len(z)
    c=[x-mid for x in z]; spread=max(map(abs,c))
    _require(spread>0,'Nonconstant centered spread required')
    mean=_finite(v[0]+anchor_scale*mid,'group mean')
    scale=_finite(anchor_scale*spread,'group scale')
    if scale==0: raise FloatingPointError('Unrepresentable centered scale')
    return mean,scale,[x/spread for x in c]


def _scaled_ratio(value,numerator,denominator):
    if value==0: return 0.
    vm,ve=math.frexp(value); nm,ne=math.frexp(numerator);dm,de=math.frexp(denominator)
    try: result=math.ldexp(vm*nm/dm,ve+ne-de)
    except OverflowError as exc: raise FloatingPointError('Unrepresentable beta') from exc
    if result==0: raise FloatingPointError('Underflowed nonzero beta')
    return _finite(result,'beta')


def group_moments(stock,sector):
    y=_numeric(stock,(len(stock),));x=_numeric(sector,(len(stock),))
    _require(len(x)>=2 and np.isfinite(x).all() and np.isfinite(y).all(),'Finite group observations required')
    my,sy,cy=_center(y);mx,sx,cx=_center(x)
    yy=math.fsum(v*v for v in cy);xx=math.fsum(v*v for v in cx);xy=math.fsum(a*b for a,b in zip(cx,cy))
    out=dict(count=len(x),stock_mean=my,sector_mean=mx,stock_scale=sy,sector_scale=sx,
        stock_ss_scaled=yy,sector_ss_scaled=xx,cross_ss_scaled=xy,beta=None,rho=None)
    if sx==0 or sy==0: return out
    beta=_scaled_ratio(xy/xx,sy,sx)
    rho=_finite(xy/math.sqrt(xx*yy),'correlation')
    if abs(rho)>1+1e-12: raise FloatingPointError('Correlation outside roundoff bound')
    out.update(beta=beta,rho=max(-1.,min(1.,rho)))
    return out


def _detail(reason):
    return dict(available=False,reason=reason,up=None,down=None,zero_count=None,
        raw_beta_gap=None,raw_correlation_gap=None)


def conditional_gaps(stock_log_returns,sector_log_returns):
    y=_numeric(stock_log_returns,(252,));x=_numeric(sector_log_returns,(252,))
    _require(np.isfinite(x).all() and np.isfinite(y).all(),'Finite log returns required')
    up=x>0;down=x<0;out=_detail('insufficient_group');out['zero_count']=int((x==0).sum())
    if int(up.sum())<40 or int(down.sum())<40: return out
    out['up']=group_moments(y[up],x[up]);out['down']=group_moments(y[down],x[down])
    for name in ('up','down'):
        if out[name]['sector_scale']==0:out['reason']='zero_sector_variance';return out
        if out[name]['stock_scale']==0:out['reason']='zero_stock_variance';return out
    out.update(available=True,reason='available',
        raw_beta_gap=_finite(out['up']['beta']-out['down']['beta'],'beta gap'),
        raw_correlation_gap=_finite(out['up']['rho']-out['down']['rho'],'correlation gap'))
    return out


def _sessions(values,label):
    idx=pd.DatetimeIndex(values)
    _require(idx.tz is None and not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.equals(idx.normalize()),label+' must be sorted unique naive session dates')
    return idx


def build_features(panel,meta,sessions,benchmark_map):
    prices=panel['adj'];sectors=panel['sectors']
    _require(isinstance(prices,pd.DataFrame) and isinstance(sectors,pd.DataFrame),'Explicit source frames required')
    axis=_sessions(prices.index,'prices');official=_sessions(sessions,'official sessions');dates=_sessions(meta['dates'],'signals')
    _require(len(official)>0 and len(dates)>0 and len(meta['symbols'])==len(dates),'Nonempty original axes')
    _require(prices.columns.is_unique and sectors.index.is_unique and sectors.columns.is_unique,'Unique source axes')
    etfs=meta['etfs'];_require(len(etfs)==17 and len(set(etfs))==17,'Original17 ETFs')
    positions=official.get_indexer(dates);_require((positions>=0).all(),'Signals must be official sessions')
    inspected=axis[(axis>=official[0])&(axis<=dates[-1])]
    _require(inspected.isin(official).all(),'Nonofficial historical source price dates')
    arrays={k:np.full((len(dates),49),np.nan) for k in EXTRA[:-1]}
    arrays['common_available']=np.zeros((len(dates),49),dtype=bool);cells=[]
    for i,(day,names,pos) in enumerate(zip(dates,meta['symbols'],positions)):
        _require(len(names)==49 and len(set(names))==49 and set(etfs)<=set(names),'Original32 stocks plus17 ETFs required')
        mapping=benchmark_map(day,names,sectors,etfs)
        _require(isinstance(mapping,dict) and set(mapping)==set(names),'Exact benchmark map required')
        window=official[pos-294:pos-41] if pos>=294 else official[:0]
        start=str(window[0].date()) if len(window) else None;end=str(window[-1].date()) if len(window) else None
        for j,symbol in enumerate(names):
            benchmark=None if symbol in etfs else mapping[symbol]
            _require(benchmark is None or isinstance(benchmark,str) and bool(benchmark),'Named benchmark or missing')
            d=_detail('etf_not_applicable')
            if symbol not in etfs:
                if benchmark is None:d=_detail('missing_sector_mapping')
                elif len(window)!=253:d=_detail('insufficient_history')
                else:
                    x=prices.reindex(index=window,columns=[benchmark]).to_numpy()[:,0]
                    y=prices.reindex(index=window,columns=[symbol]).to_numpy()[:,0]
                    x=_numeric(x,(253,));y=_numeric(y,(253,))
                    if not (np.isfinite(x)&(x>0)).all():d=_detail('invalid_sector_path')
                    elif not (np.isfinite(y)&(y>0)).all():d=_detail('invalid_stock_path')
                    else:d=conditional_gaps(log_returns(y),log_returns(x))
            cell=dict(signal_index=i,pool_index=j,signal_date=str(day.date()),symbol=symbol,
                benchmark=benchmark,window_start=start,window_end=end,**d,
                rank_beta_gap=None,rank_correlation_gap=None)
            cells.append(cell)
            if d['available']:
                arrays['common_available'][i,j]=True
                for key in EXTRA[:2]:arrays[key][i,j]=d[key]
        for raw,rank in zip(EXTRA[:2],EXTRA[2:4]):
            values=pd.Series(arrays[raw][i]).rank(method='average',pct=True)-.5
            arrays[rank][i]=values.to_numpy()
            for j,value in enumerate(values):
                if math.isfinite(value):cells[i*49+j][rank]=float(value)
    audit=dict(original_arrays_unchanged=True,original_months=len(dates),original_width=49,new_model_fits=0,
        returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True,
        mapping_source_sha256=MAPPING_SHA,cells=cells)
    return dict(arrays=arrays,audit=audit)
