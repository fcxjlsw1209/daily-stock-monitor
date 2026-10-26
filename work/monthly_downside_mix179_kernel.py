"""Pure downside-share changes, with no I/O, fitting, ranking, or trading.

The full official-session window is t-294..t. Both arms preserve the caller's
eligibility and score architecture; this module neither ranks nor selects.
The old252 and current63 windows overlap by 21 returns; the references are
not independent. Downside share is a zero-referenced lower partial second
moment divided by the full uncentered second moment, not downside variance
around a sample mean, nor the standard deviation of negative-only days.
Recent prices are
used only for risk, never to impose a new momentum or pullback eligibility gate.
"""
import math

import numpy as np
import pandas as pd


PRICE_COUNT = 295
SKIP = 42
DENOMINATOR_FLOOR = 1e-24
WINDOWS = {'old252': (0, 253), 'current63': (231, 295)}


def _empty(reason):
    out = dict(available=False, reason=reason, candidate=float('nan'),
               control=float('nan'), downside_share_estimates=0)
    for name in WINDOWS:
        out['sum_squares_' + name] = None
        out['downside_sum_squares_' + name] = None
        out['share_' + name] = None
    return out


def measure(adjusted):
    """Use exactly 295 positive finite adjusted prices; missing source is neutral.

Once valid source values have been established, invalid numeric arithmetic is
an error, not missing evidence. Both uncentered sums of squares are computed
before applying their common strict >1e-24 floor. Up and zero-return days remain
in the denominator; no averaging over only the negative observations occurs.
"""
    if np.shape(adjusted) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 295 adjusted session prices')
    try:
        prices = np.asarray(adjusted, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty('nonnumeric_adjusted_price')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        return _empty('invalid_adjusted_price')
    out = _empty('downside_denominator_floor')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for name, (start, stop) in WINDOWS.items():
            p = prices[start:stop]
            ratio = p[1:] / p[:-1]
            if not (np.isfinite(ratio) & (ratio > 0.)).all():
                raise FloatingPointError('Valid prices produced an invalid simple-return ratio')
            returns = ratio - 1.
            squares = np.square(returns)
            if not np.isfinite(squares).all():
                raise FloatingPointError('Valid prices produced nonfinite squared returns')
            try:
                total = math.fsum(float(v) for v in squares)
                downside = math.fsum(float(v) for v in squares[returns < 0.])
            except OverflowError as exc:
                raise FloatingPointError('Valid returns produced a nonfinite sum of squares') from exc
            if not math.isfinite(total) or not math.isfinite(downside):
                raise FloatingPointError('Valid returns produced a nonfinite sum of squares')
            if not 0. <= downside <= total:
                raise FloatingPointError('Downside second moment outside total second moment')
            out['sum_squares_' + name] = total
            out['downside_sum_squares_' + name] = downside
    out['downside_share_estimates'] = 2
    if any(out['sum_squares_' + name] <= DENOMINATOR_FLOOR for name in WINDOWS):
        return out
    for name in WINDOWS:
        share = out['downside_sum_squares_' + name] / out['sum_squares_' + name]
        if not math.isfinite(share) or not 0. <= share <= 1.:
            raise FloatingPointError('Finite positive moments produced an invalid downside share')
        out['share_' + name] = share
    candidate = out['share_old252'] - out['share_current63']
    control = .5 - out['share_current63']
    if not (-1. <= candidate <= 1. and -.5 <= control <= .5):
        raise FloatingPointError('Downside contrast outside its mathematical bounds')
    out.update(available=True, reason='available', candidate=candidate, control=control)
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
    """Read only the requested event prefix on the supplied official calendar.

ETF identifiers receive the same statistic as stocks. There is no current
pullback requirement, cross-sectional rank, or eligibility modification.
"""
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is None:
        limit = len(queries)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= len(queries):
        raise ValueError('Limit must be an integer event prefix')
    values = dict(candidate=np.full((limit, width), np.nan),
                  control=np.full((limit, width), np.nan),
                  available=np.zeros((limit, width), dtype=bool))
    history, reasons, estimates = [], {}, 0
    frame = panel['adj']
    for i, (day, symbols, position) in enumerate(queries[:limit]):
        window = sessions[position-294:position+1] if position >= 294 else None
        windows, missing = None, []
        if window is not None:
            missing = [str(d.date()) for d in window if d not in frame.index]
            windows = {name: dict(price_start=str(window[start].date()),
                                  price_end=str(window[stop-1].date()),
                                  return_count=stop-start-1)
                       for name, (start, stop) in WINDOWS.items()}
        cells = []
        for j, symbol in enumerate(symbols):
            if window is None:
                cell = _empty('insufficient_history')
            elif symbol not in frame.columns:
                cell = _empty('missing_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(frame.loc[window, symbol].to_numpy())
            for key in values:
                values[key][i, j] = cell[key]
            estimates += cell['downside_share_estimates']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cell = dict(cell, symbol=symbol, date=str(day.date()), windows=windows)
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=symbols, session_position=position,
                            price_start=str(window[0].date()) if window is not None else None,
                            price_end=str(window[-1].date()) if window is not None else None,
                            windows=windows, missing_official_sessions=missing, cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=int(limit), symbols_per_event=width, checked_cells=int(limit)*width,
                 price_count=PRICE_COUNT, skip_sessions=SKIP, old_return_count=252,
                 current_return_count=63, denominator_floor=DENOMINATOR_FLOOR,
                 available_cells=available, unavailable_cells=int(limit)*width-available,
                 reason_counts=reasons, downside_share_estimates=estimates,
                 predictive_model_fits=0, common_availability=True,
                 portfolio_returns_calculated=False, ranking_performed=False,
                 trading_eligibility_modified=False, event_history=history)
    return values, audit
