"""Historical upside-gain retention; pure descriptive source measurement.

The last 42 sessions are not consulted. Positive daily moves at indices
1..231 have 21 subsequent observations within the full 253-price window.
Adjacent mature labels overlap; neither their count nor their gain weights
are independent sample counts. This is not a gain-survival path estimator:
only the 21st subsequent close is compared with the event's two endpoints.
"""
import math

import numpy as np
import pandas as pd


PRICE_COUNT = 253
SKIP = 42
HORIZON = 21
LAST_ANCHOR = 231


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'),
        control=float('nan'), retention_share_estimates=0, positive_anchor_count=0,
        anchor_indices=[], positive_log_returns=[], retained_log_moves=[],
        capped_retained_log_moves=[], binary_recovered=[], positive_return_sum=None,
        retained_return_sum=None, binary_weighted_sum=None,
        retention_share=None, binary_retention_share=None)


def measure(adjusted):
    """Measure the registered log-level differences from 253 supplied prices.

All prices must be finite and positive. Once source validity is established,
any nonfinite arithmetic fails explicitly. No floor, epsilon denominator,
extra minimum-event count, clipping of the final shares, or fitted model is
used. Comparisons in the binary control use prices, including exact ties.
"""
    if np.shape(adjusted) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 253 adjusted session prices')
    try:
        prices = np.asarray(adjusted, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty('nonnumeric_adjusted_price')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        return _empty('invalid_adjusted_price')
    logs = [math.log(float(p)) for p in prices]
    if not all(math.isfinite(v) for v in logs):
        raise FloatingPointError('Positive finite prices produced nonfinite log levels')
    out = _empty('no_positive_anchors')
    out['retention_share_estimates'] = 2
    binary_terms = []
    for i in range(1, LAST_ANCHOR+1):
        initial = logs[i] - logs[i-1]
        if not math.isfinite(initial):
            raise FloatingPointError('Nonfinite initial log gain')
        if initial <= 0.:
            continue
        retained = logs[i+HORIZON] - logs[i-1]
        if not math.isfinite(retained):
            raise FloatingPointError('Nonfinite mature log gain')
        capped = min(initial, max(0., retained))
        full = bool(prices[i+HORIZON] >= prices[i])
        out['anchor_indices'].append(i)
        out['positive_log_returns'].append(initial)
        out['retained_log_moves'].append(retained)
        out['capped_retained_log_moves'].append(capped)
        out['binary_recovered'].append(full)
        binary_terms.append(initial if full else 0.)
    out['positive_anchor_count'] = len(out['anchor_indices'])
    try:
        total = math.fsum(out['positive_log_returns'])
        remaining = math.fsum(out['capped_retained_log_moves'])
        binary = math.fsum(binary_terms)
    except OverflowError as exc:
        raise FloatingPointError('Nonfinite aggregate log gains') from exc
    if not all(math.isfinite(v) for v in (total, remaining, binary)):
        raise FloatingPointError('Nonfinite aggregate log gains')
    out.update(positive_return_sum=total, retained_return_sum=remaining,
               binary_weighted_sum=binary)
    if not out['positive_anchor_count']:
        return out
    if not total > 0. or not 0. <= binary <= remaining <= total:
        raise FloatingPointError('Mature retained gains violate their bounds')
    q, c = remaining/total, binary/total
    if not (math.isfinite(q) and math.isfinite(c) and 0. <= c <= q <= 1.):
        raise FloatingPointError('Finite gains produced invalid retention shares')
    out.update(available=True, reason='available', retention_share=q,
               binary_retention_share=c, candidate=q-.5, control=c-.5)
    return out


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


def build_features(panel, meta, calendar, limit=None):
    """Measure each original candidate without a trading-eligibility filter."""
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is None:
        limit = len(queries)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= len(queries):
        raise ValueError('Limit must be an integer event prefix')
    values = dict(candidate=np.full((limit,width), np.nan),
                  control=np.full((limit,width), np.nan),
                  available=np.zeros((limit,width), dtype=bool))
    frame, history, reasons = panel['adj'], [], {}
    estimates = anchors = 0
    for row, (day, symbols, position) in enumerate(queries[:limit]):
        window = sessions[position-294:position-SKIP+1] if position >= 294 else None
        missing, windows = [], None
        if window is not None:
            missing = [str(d.date()) for d in window if d not in frame.index]
            windows = dict(
                formation=dict(price_start=str(window[0].date()),price_end=str(window[-1].date()),return_count=252),
                anchors=dict(first_date=str(window[1].date()),last_date=str(window[LAST_ANCHOR].date()),count=LAST_ANCHOR),
                maturities=dict(first_date=str(window[1+HORIZON].date()),last_date=str(window[-1].date()),horizon_sessions=HORIZON))
        cells = []
        for j, symbol in enumerate(symbols):
            if window is None:
                cell = _empty('insufficient_history')
            elif symbol not in frame.columns:
                cell = _empty('missing_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(frame.loc[window,symbol].to_numpy())
            for key in values:
                values[key][row,j] = cell[key]
            estimates += cell['retention_share_estimates']
            anchors += cell['positive_anchor_count']
            reasons[cell['reason']] = reasons.get(cell['reason'],0)+1
            cell = dict(cell,symbol=symbol,date=str(day.date()),windows=windows)
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=str(day.date()),symbols=symbols,session_position=position,
            price_start=str(window[0].date()) if window is not None else None,
            price_end=str(window[-1].date()) if window is not None else None,
            windows=windows,missing_official_sessions=missing,cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=int(limit),symbols_per_event=width,checked_cells=int(limit)*width,
        price_count=PRICE_COUNT,skip_sessions=SKIP,formation_return_count=252,
        anchor_count=LAST_ANCHOR,horizon_sessions=HORIZON,
        available_cells=available,unavailable_cells=int(limit)*width-available,
        reason_counts=reasons,retention_share_estimates=estimates,positive_anchor_count=anchors,
        predictive_model_fits=0,common_availability=True,portfolio_returns_calculated=False,
        ranking_performed=False,trading_eligibility_modified=False,event_history=history)
    return values,audit
