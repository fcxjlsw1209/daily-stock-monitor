"""Fixed CMF-state softening; no model, I/O, selection or portfolio replay."""
import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PANEL_KEYS = {'adj', 'close', 'high', 'low', 'volume'}

def market_gate(window):
    x = np.asarray(window, dtype=float)
    if x.shape != (252,) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('252 finite nonnegative canonical sigmas required')
    current = float(x[-1]); median = float(pd.Series(x).median())
    less, equal = int(np.count_nonzero(x < current)), int(np.count_nonzero(x == current))
    state = 1. if current > median else -1.
    gate = 0. if current == median else (2*less + equal)/252. - 1.
    if not -1. < gate < 1. or gate * state < 0:
        raise ValueError('Gate violates sign-preserving shrinkage')
    return dict(sigma=current, median=median, less=less, equal=equal,
                median_tie=current == median, state=state,
                gate_candidate=gate, gate_control=.5*state)


def cmf_ranks(raw):
    a = np.asarray(raw, dtype=float)
    if a.ndim != 1 or np.isinf(a).any():
        raise ValueError('One-dimensional finite-or-missing CMF values required')
    return pd.Series(a).rank(pct=True).fillna(.5).to_numpy() - .5


def build_features(panel, meta, calendar, limit=None):
    if not isinstance(panel, dict) or set(panel) != PANEL_KEYS:
        raise ValueError('Original adjusted prices and four signal OHLCV fields required')
    for frame in panel.values(): _frame(frame)
    adj = panel['adj']
    if 'SPY' not in adj:
        raise ValueError('Original SPY series required')
    if any(not frame.index.equals(adj.index) for frame in panel.values()):
        raise ValueError('Original source session axes differ')
    sessions = _calendar(calendar); queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('Invalid query prefix')
    queries = queries if limit is None else queries[:limit]
    n = len(queries)
    values = {k:np.empty(n, dtype=float) for k in ('state', 'gate_candidate', 'gate_control', 'coefficient', 'sigma', 'median')}
    values['calm'] = np.empty(n, dtype=bool)
    values['cmf_rank'] = np.empty((n, width), dtype=float)
    values['sigma_window'] = np.empty((n, 252), dtype=float)
    history=[]
    if n:
        # Physical prefix slicing precedes numerical conversion and rolling.
        stop=queries[-1][0]
        p={k:v.loc[:stop].copy() for k,v in panel.items()}
        spy=p['adj']['SPY'].astype(float)
        sigma=spy.pct_change(fill_method=None).rolling(63).std(ddof=1)
        c,h,l,v=(p[k].astype(float) for k in ('close','high','low','volume'))
        spread=h-l
        loc=((2*c-h-l)/spread.replace(0,np.nan)).where(spread!=0,0.)
        factor=(-(loc*v).rolling(63).sum()/v.rolling(63).sum()).replace([np.inf,-np.inf],np.nan)
        for i,(day,symbols,t) in enumerate(queries):
            if t<314: raise ValueError('Insufficient official SPY window')
            dates=sessions[t-314:t+1]
            if not p['adj'].loc[dates[0]:day].index.equals(dates):
                raise ValueError('SPY source must have exactly315 official sessions')
            price=spy.loc[dates].to_numpy()
            if not (np.isfinite(price)&(price>0)).all():
                raise ValueError('Missing or invalid SPY price; no fill permitted')
            if any(not set(symbols).issubset(p[k].columns) for k in PANEL_KEYS):
                raise ValueError('Original query security missing from source')
            window=sigma.loc[dates[-252:]].to_numpy(copy=True)
            state=market_gate(window)
            calm=bool(state['sigma']*np.sqrt(252)<.10)
            coefficient=.075 if calm else .075*(1-(.10 if state['state']>0 else 0.)/.48)
            raw=factor.loc[day,symbols].to_numpy()
            values['cmf_rank'][i]=cmf_ranks(raw)
            values['sigma_window'][i]=window
            for key in ('state','gate_candidate','gate_control','sigma','median'): values[key][i]=state[key]
            values['calm'][i]=calm; values['coefficient'][i]=coefficient
            history.append(dict(date=str(day.date()),symbols=symbols,
                spy_window_start=str(dates[0].date()),spy_window_end=str(day.date()),
                sigma_window_start=str(dates[-252].date()),session_position=t,
                **{k:state[k] for k in ('less','equal','median_tie','state','gate_candidate','gate_control')},
                coefficient=coefficient,calm=calm,cmf_available_count=int(np.isfinite(raw).sum())))
    count=sum(row['cmf_available_count'] for row in history)
    audit=dict(events=n,symbols_per_event=width,checked_cells=n*width,
        finite_cmf_cells=count,neutral_cmf_cells=n*width-count,
        spy_price_observations=315*n,volatility_rank_observations=252*n,
        new_model_fits=0,predictive_model_fits=0,portfolio_returns_calculated=False,
        event_history=history)
    return values,audit
