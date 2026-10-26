"""Pure reverse sensitivity: return original volatility budget to nonvol terms.

No new forecasts, eligibility, risk estimators or portfolio decisions. The caller
adds these deltas to the original pre-score and calls the original episode once.
"""
import numpy as np
import pandas as pd

LAMBDAS = {'aux_reallocate_0025': .025, 'aux_reallocate_0050': .05,
           'aux_reallocate_0100': .10}
TOL = 1e-12


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted prices require a DataFrame with DatetimeIndex')
    idx = frame.index
    if (idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing or idx.tz is not None
            or not idx.equals(idx.normalize())):
        raise ValueError('Price sessions must be unique increasing naive dates')
    if not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame):
        raise ValueError('Price columns must be unique security IDs')


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Expected a naive midnight date')
    return day


def _axes(meta, calendar, limit):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    dates = [_day(x) for x in calendar]
    if (not dates or any(not isinstance(x, str) or x != str(d.date()) or d.weekday() >= 5
                         for x, d in zip(calendar, dates))
            or any(a >= b for a, b in zip(dates, dates[1:]))):
        raise ValueError('Calendar must contain increasing unique ISO weekday sessions')
    sessions = pd.DatetimeIndex(dates); lookup = {d: i for i, d in enumerate(dates)}
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Event dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)) or len(set(meta['etfs'])) != len(meta['etfs']):
        raise ValueError('ETF identities require a unique sequence')
    if any(not isinstance(s, str) or not s or s.strip() != s for s in meta['etfs']):
        raise ValueError('Invalid ETF identity')
    rows, width = [], None
    for date, ss in zip(meta['dates'], meta['symbols']):
        day = _day(date)
        if day not in lookup or (rows and rows[-1][0] >= day):
            raise ValueError('Events must be increasing official sessions')
        if isinstance(ss, (str, bytes)):
            raise ValueError('Symbols require a sequence')
        ss = list(ss)
        if (not ss or len(ss) != len(set(ss)) or
                any(not isinstance(s, str) or not s or s.strip() != s for s in ss)):
            raise ValueError('Event security IDs must be unique')
        if width is not None and len(ss) != width:
            raise ValueError('Event width must be constant')
        width = len(ss); rows.append((day, ss, lookup[day]))
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(rows)):
        raise ValueError('limit must be an integer event prefix')
    return sessions, rows if limit is None else rows[:limit], (len(rows), width or 0)


def _matrix(value, shape, n, name):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(name+' must match full event matrix')
    try:
        return arr[:n].astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be numeric') from exc


def _state(value, rows, n, name):
    arr = np.asarray(value)
    if arr.shape != (rows,) or arr.dtype.kind != 'b':
        raise ValueError(name+' must be a full boolean event vector')
    return arr[:n]


def _weight(value, shape, n, name):
    arr = np.asarray(value)
    if arr.shape == (shape[0],):
        arr = arr[:, None]
    if arr.shape not in ((shape[0], 1), shape):
        raise ValueError(name+' shape must be n, n-by-1 or n-by-width')
    try:
        out = np.broadcast_to(arr[:n].astype(float, copy=False), (n, shape[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be numeric') from exc
    if not np.isfinite(out).all():
        raise ValueError(name+' must be finite')
    return out


def _sources(original, source, shape, n):
    required = {'vol', 'vol_weight', 'state_highvol'}
    if not isinstance(source, dict) or not required <= set(source) or set(source)-required-{'core','core_weight'}:
        raise ValueError('Source requires vol, vol_weight, state_highvol; optional core/core_weight')
    if not isinstance(original, dict) or not {'baseline','core','core_weight','state_calm10'} <= set(original):
        raise ValueError('Original baseline/core/core_weight/state_calm10 required')
    b = _matrix(original['baseline'], shape, n, 'baseline')
    core = _matrix(original['core'], shape, n, 'core')
    calm = _state(original['state_calm10'], shape[0], n, 'state_calm10')
    high = _state(source['state_highvol'], shape[0], n, 'state_highvol')
    cw = _weight(original['core_weight'], shape, n, 'core_weight')
    if not np.array_equal(cw, np.broadcast_to(np.where(calm,.5,.52)[:,None], cw.shape)):
        raise ValueError('Original core weight disagrees with calm state')
    v = _matrix(source['vol'], shape, n, 'vol')
    vw = _weight(source['vol_weight'], shape, n, 'vol_weight')
    expected = np.broadcast_to(np.where(calm,.20,.219375*(1.-np.where(high,.10,0.)/.48))[:,None], vw.shape)
    if not np.array_equal(vw, expected):
        raise ValueError('Original volatility coefficient disagrees with source states')
    if not (np.isfinite(b).all() and np.isfinite(core).all() and np.isfinite(v).all()):
        raise ValueError('Original score components must be finite, including neutral missing ranks')
    if 'core' in source and not np.array_equal(_matrix(source['core'],shape,n,'source core'),core):
        raise ValueError('Source and original core differ')
    if 'core_weight' in source and not np.array_equal(_weight(source['core_weight'],shape,n,'source cw'),cw):
        raise ValueError('Source and original core weights differ')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support = (b-core)/(1.-cw)
        unit = (b-core)-(1.-cw)*v
        donor = 1.-cw-vw
    if not (np.isfinite(support).all() and np.isfinite(unit).all()):
        raise FloatingPointError('Nonfinite support or reallocation')
    if ((np.abs(v)>.5+TOL).any() or (np.abs(core/cw)>.5+TOL).any()
            or (np.abs(support)>.5+TOL).any() or (donor<=0.).any()):
        raise ValueError('Original ranks or noncore budget outside registered bounds')
    return b,core,cw,v,vw,calm,high,support,unit,donor


def _rank(values):
    return pd.Series(values).replace([np.inf,-np.inf],np.nan).rank(method='average',pct=True).fillna(.5).to_numpy()-.5


def build_features(panel, meta, calendar, original, source, limit=None):
    if not isinstance(panel,dict) or set(panel)!={'adj'}:
        raise ValueError('Panel requires exactly adj')
    _frame(panel['adj'])
    sessions, queries, shape = _axes(meta,calendar,limit)
    n,width = len(queries),shape[1]
    b,core,cw,v,vw,calm,high,support,unit,donor = _sources(original,source,shape,n)
    history = []
    if queries:
        last = queries[-1][0]; prefix = panel['adj'].loc[:last]
        if prefix.empty or 'SPY' not in prefix:
            raise ValueError('Original source requires SPY history')
        needed = sorted({'SPY'}|{s for _,ss,_ in queries for s in ss})
        official = sessions[(sessions>=prefix.index[0])&(sessions<=last)]
        overlap = prefix.index[prefix.index>=sessions[0]]
        if len(overlap.difference(sessions)):
            raise ValueError('Source includes nonofficial covered sessions')
        prefix = prefix.reindex(index=prefix.index.union(official),columns=needed)
        try:
            px = prefix.astype(float)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError('Price prefix must be numeric') from exc
        x=px.to_numpy(); supplied=~np.isnan(x)
        if not (np.isfinite(x[supplied])&(x[supplied]>0.)).all():
            raise ValueError('Observed prices must be finite positive; NaN is missing')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            r=px.pct_change(fill_method=None)
        if np.isinf(r.to_numpy()).any():
            raise FloatingPointError('Simple-return arithmetic overflow')
        sigma126=r.rolling(126).std(ddof=1)
        spy63=r['SPY'].rolling(63).std(ddof=1)
        # A complete finite-return window cannot legitimately have a missing
        # deviation. Distinguish arithmetic overflow from source/warmup NaNs.
        bad126=(r.rolling(126).count()==126)&~np.isfinite(sigma126)
        bad63=(r['SPY'].rolling(63).count()==63)&~np.isfinite(spy63)
        if bad126.to_numpy().any() or bad63.to_numpy().any():
            raise FloatingPointError('Volatility arithmetic failed on complete finite returns')
        rebuilt_calm=spy63*np.sqrt(252.)<.10
        rebuilt_high=spy63>spy63.rolling(252).median()
        for i,(day,symbols,t) in enumerate(queries):
            if bool(rebuilt_calm.at[day])!=bool(calm[i]) or bool(rebuilt_high.at[day])!=bool(high[i]):
                raise ValueError('Original market state parity failed')
            raw=sigma126.loc[day,symbols].to_numpy()
            if np.isinf(raw).any():
                raise FloatingPointError('Nonfinite volatility estimate')
            ranked=_rank(raw)
            if not np.array_equal(ranked,v[i]):
                raise ValueError('Original full-domain volatility rank parity failed')
            history.append(dict(date=str(day.date()),symbols=symbols,session_position=t,
                calm=bool(calm[i]),highvol=bool(high[i]),
                raw_vol=[float(z) if np.isfinite(z) else None for z in raw],
                vol_domain=np.isfinite(raw).tolist(),vol_rank=v[i].tolist(),
                unit_delta=unit[i].tolist(),core_weight=cw[i].tolist(),vol_weight=vw[i].tolist(),
                original_support=support[i].tolist(),nonvol_budget=donor[i].tolist()))
    finite=sum(sum(row['vol_domain']) for row in history)
    return dict(vol_rank=v.copy(),unit_delta=unit.copy()),dict(
        events=n,symbols_per_event=width,checked_cells=n*width,
        source_vol_finite_cells=finite,source_vol_missing_neutral_cells=n*width-finite,
        volatility_rank_reconstructions=n,predictive_model_fits=0,event_history=history,
        direction='vol_to_nonvol',
        no_new_availability_filter=True,ranking_eligibility_filtered=False,
        portfolio_returns_calculated=False)


def scaled_delta(unit_delta, amount):
    """Only the three fixed doses and the zero-identity test/control are legal."""
    if type(amount) not in (float,int) or amount not in (0.,*LAMBDAS.values()):
        raise ValueError('Only zero and fixed registered sensitivity doses allowed')
    unit=np.asarray(unit_delta,dtype=float)
    if unit.ndim!=2 or not np.isfinite(unit).all():
        raise ValueError('Unit correction must be a finite matrix')
    out=np.zeros(unit.shape,dtype=float) if amount==0. else float(amount)*unit
    if len(out):out[0]=0.
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite scaled correction')
    return out


def build_deltas(values, original, source, limit=None):
    shape=np.asarray(original['baseline']).shape
    if len(shape)!=2:raise ValueError('Original baseline must be a matrix')
    if limit is not None and (type(limit) is not int or not 0<=limit<=shape[0]):
        raise ValueError('limit must be an integer event prefix')
    n=shape[0] if limit is None else limit
    b,core,cw,v,vw,calm,high,support,unit,donor=_sources(original,source,shape,n)
    if not isinstance(values,dict) or set(values)!={'vol_rank','unit_delta'}:
        raise ValueError('Expected exactly vol_rank/unit_delta')
    checked={}
    for name in values:
        arr=np.asarray(values[name])
        if arr.ndim!=2 or arr.shape[1]!=shape[1] or arr.shape[0] not in (n,shape[0]):
            raise ValueError(name+' must align with the requested prefix or full source')
        checked[name]=arr[:n].astype(float,copy=False)
    if not np.array_equal(checked['vol_rank'],v) or not np.array_equal(checked['unit_delta'],unit):
        raise ValueError('Stored volatility rank or unit reallocation does not match source')
    deltas={name:scaled_delta(unit,amount) for name,amount in LAMBDAS.items()}
    return deltas,dict(events=n,width=shape[1],lambdas=dict(LAMBDAS),direction='vol_to_nonvol',
        changed_cells={name:int(np.count_nonzero(x)) for name,x in deltas.items()},
        max_abs_delta={name:float(np.max(np.abs(x))) if x.size else 0. for name,x in deltas.items()},
        initial_delta_zero=True,no_new_availability_filter=True,episode_applied=False,
        original_core_unchanged=True,original_model_corrections_unchanged=True,
        original_support_unchanged=True)
