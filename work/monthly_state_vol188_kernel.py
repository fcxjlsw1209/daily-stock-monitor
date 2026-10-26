"""Pure own-volatility historical midpoint CDFs; no I/O, ranking or trading.

Canonical statistics use pinned pandas over the official-calendar prefix.
Current risk is compared with exactly the previous 756 session slots. The
conditional arm matches historical SPY calm states, not return outcomes.
"""
import math

import numpy as np
import pandas as pd

HISTORY = 756
OWN_WINDOW = 126
MARKET_WINDOW = 63
MIN_JOINT = 252
MIN_MATCHED = 63
CALM_THRESHOLD = .10
ANNUALIZATION = 252


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' requires a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + ' requires a naive midnight date')
    return day


def _panel(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    frame = panel['adj']
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted source requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Panel dates must be unique increasing naive midnight sessions')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                           for s in frame.columns)):
        raise ValueError('Panel symbols must be unique nonempty strings')


def _calendar(calendar):
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
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF IDs require an explicit sequence')
    etfs = list(meta['etfs'])
    if any(not isinstance(s, str) or not s or s.strip() != s for s in etfs) or len(set(etfs)) != len(etfs):
        raise ValueError('ETF IDs must be unique nonempty strings')
    positions = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in positions or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Candidate IDs require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
                or len(set(symbols)) != len(symbols)):
            raise ValueError('Candidate IDs must be unique nonempty strings')
        if width is not None and width != len(symbols):
            raise ValueError('Candidate width must remain constant')
        width = len(symbols)
        rows.append((day, symbols, positions[day]))
    return rows, width or 0


def _finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def _calm(sigma):
    if sigma is None:
        return None, None
    annual = sigma * math.sqrt(ANNUALIZATION)
    if not math.isfinite(annual):
        raise FloatingPointError('Finite SPY volatility produced nonfinite annualized risk')
    return bool(annual < CALM_THRESHOLD), annual


def _risk_prefix(frame, sessions, queries):
    """No numeric access outside the requested prefix or its queried symbols."""
    names = sorted({'SPY'} | {s for _, symbols, _ in queries for s in symbols})
    index = sessions[:queries[-1][2] + 1]
    present = [s for s in names if s in frame.columns]
    raw = frame.reindex(index=index, columns=present)
    try:
        numeric = raw.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Historical adjusted prices must be numeric or missing') from exc
    numeric[~np.isfinite(numeric) | (numeric <= 0.)] = np.nan
    prices = pd.DataFrame(numeric, index=index, columns=present).reindex(columns=names)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        returns = prices.pct_change(fill_method=None)
    rv = returns.to_numpy()
    if np.isinf(rv).any():
        raise FloatingPointError('Finite valid prices produced nonfinite simple returns')
    own = returns.rolling(OWN_WINDOW, min_periods=OWN_WINDOW).std(ddof=1)
    own_complete = returns.notna().rolling(OWN_WINDOW, min_periods=OWN_WINDOW).sum().eq(OWN_WINDOW)
    if ((~np.isfinite(own.to_numpy())) & own_complete.to_numpy()).any():
        raise FloatingPointError('Complete finite returns produced nonfinite own volatility')
    market_r = returns['SPY']
    market = market_r.rolling(MARKET_WINDOW, min_periods=MARKET_WINDOW).std(ddof=1)
    market_complete = market_r.notna().rolling(MARKET_WINDOW, min_periods=MARKET_WINDOW).sum().eq(MARKET_WINDOW)
    if ((~np.isfinite(market.to_numpy())) & market_complete.to_numpy()).any():
        raise FloatingPointError('Complete finite returns produced nonfinite market volatility')
    if (own.to_numpy()[np.isfinite(own.to_numpy())] < 0.).any() or (market.dropna() < 0.).any():
        raise FloatingPointError('Sample volatility cannot be negative')
    return own, market


def measure_reference(current_sigma, current_calm, reference_sigmas, reference_states):
    """Midpoint CDF on a full reference window, with a common support rule.

This helper consumes volatility statistics, not prices or model predictions.
Nonfinite historical statistics and unknown states are unavailable references.
"""
    if len(reference_sigmas) != HISTORY or len(reference_states) != HISTORY:
        raise ValueError('Reference requires exactly 756 official slots')
    if current_calm is not None and not isinstance(current_calm, (bool, np.bool_)):
        raise ValueError('Market state must be bool or None')
    sigma = _finite_or_none(current_sigma) if current_sigma is not None else None
    if sigma is not None and sigma < 0.:
        raise ValueError('Volatility cannot be negative')
    joint, matched = [], []
    for value, state in zip(reference_sigmas, reference_states):
        if state is not None and not isinstance(state, (bool, np.bool_)):
            raise ValueError('Historical market state must be bool or None')
        risk = _finite_or_none(value) if value is not None else None
        if risk is not None and risk < 0.:
            raise ValueError('Volatility cannot be negative')
        if risk is not None and state is not None:
            joint.append(risk)
            if current_calm is not None and bool(state) == bool(current_calm):
                matched.append(risk)
    out = dict(current_sigma=sigma, current_calm=None if current_calm is None else bool(current_calm),
               reference_count=len(joint), matched_count=len(matched),
               reference_below=None, reference_equal=None, matched_below=None, matched_equal=None,
               candidate=float('nan'), control=float('nan'), available=False, reason=None)
    if sigma is not None:
        out['reference_below'] = sum(x < sigma for x in joint)
        out['reference_equal'] = sum(x == sigma for x in joint)
        if current_calm is not None:
            out['matched_below'] = sum(x < sigma for x in matched)
            out['matched_equal'] = sum(x == sigma for x in matched)
    if current_calm is None:
        out['reason'] = 'current_market_state_unavailable'
    elif sigma is None:
        out['reason'] = 'current_volatility_unavailable'
    elif len(joint) < MIN_JOINT:
        out['reason'] = 'insufficient_joint_history'
    elif len(matched) < MIN_MATCHED:
        out['reason'] = 'insufficient_matched_history'
    else:
        out.update(available=True, reason='available',
                   candidate=(out['matched_below'] + .5*out['matched_equal'])/len(matched) - .5,
                   control=(out['reference_below'] + .5*out['reference_equal'])/len(joint) - .5)
        if not (-.5 <= out['candidate'] <= .5 and -.5 <= out['control'] <= .5):
            raise FloatingPointError('Midpoint CDF outside declared bounds')
    return out


def build_features(panel, meta, calendar, limit=None):
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is None:
        limit = len(queries)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= len(queries):
        raise ValueError('Limit must be an integer event prefix')
    limit = int(limit)
    values = dict(candidate=np.full((limit, width), np.nan), control=np.full((limit, width), np.nan),
                  available=np.zeros((limit, width), dtype=bool))
    history, reasons = [], {}
    reference_observations = matched_observations = 0
    frame = panel['adj']
    if limit:
        own, market = _risk_prefix(frame, sessions, queries[:limit])
        market_states = [_calm(_finite_or_none(x))[0] for x in market.to_numpy()]
    for i, (day, symbols, position) in enumerate(queries[:limit]):
        iso = str(day.date())
        sigma_market = _finite_or_none(market.iloc[position])
        current_calm, annual = _calm(sigma_market)
        has_history = position >= HISTORY
        ref_start = str(sessions[position-HISTORY].date()) if has_history else None
        ref_end = str(sessions[position-1].date()) if has_history else None
        support = sessions[max(0, position-HISTORY-OWN_WINDOW):position+1]
        cells = []
        for j, symbol in enumerate(symbols):
            sigma = _finite_or_none(own[symbol].iloc[position])
            if has_history:
                cell = measure_reference(sigma, current_calm,
                    own[symbol].iloc[position-HISTORY:position].to_numpy(),
                    market_states[position-HISTORY:position])
            else:
                cell = dict(current_sigma=sigma, current_calm=current_calm, reference_count=0,
                            matched_count=0, reference_below=None, reference_equal=None,
                            matched_below=None, matched_equal=None, candidate=float('nan'),
                            control=float('nan'), available=False, reason='insufficient_history')
            if has_history and symbol not in frame.columns:
                cell['reason'] = 'missing_symbol'
            for key in values:
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            reference_observations += cell['reference_count']
            matched_observations += cell['matched_count']
            cell = dict(cell, symbol=symbol, date=iso, reference_start=ref_start, reference_end=ref_end)
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=iso, symbols=symbols, session_position=position,
            current_calm=current_calm, spy_sigma63=sigma_market, spy_annualized_sigma=annual,
            reference_start=ref_start, reference_end=ref_end, reference_slot_count=HISTORY if has_history else 0,
            own_price_start=str(sessions[position-OWN_WINDOW].date()) if position >= OWN_WINDOW else None,
            own_price_end=iso,
            market_price_start=str(sessions[position-MARKET_WINDOW].date()) if position >= MARKET_WINDOW else None,
            market_price_end=iso, source_start=str(support[0].date()), source_end=iso,
            missing_official_sessions=[str(d.date()) for d in support if d not in frame.index], cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=limit, symbols_per_event=width, checked_cells=limit*width,
        history_slots=HISTORY, own_return_count=OWN_WINDOW, market_return_count=MARKET_WINDOW,
        minimum_joint=MIN_JOINT, minimum_matched=MIN_MATCHED, calm_threshold=CALM_THRESHOLD,
        annualization=ANNUALIZATION, ddof=1, available_cells=available,
        unavailable_cells=limit*width-available, reference_observations=reference_observations,
        matched_observations=matched_observations, reason_counts=reasons, predictive_model_fits=0,
        common_availability=True, portfolio_returns_calculated=False, ranking_performed=False,
        trading_eligibility_modified=False, event_history=history)
    return values, audit
