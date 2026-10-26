"""Pure twelve-block sector-relative versus absolute sign recurrence.

All 253 official-session prices in each series must be finite and positive.
Only thirteen endpoints enter exact sign comparisons; no models or return replay.
"""
import math

import numpy as np
import pandas as pd

PRICE_COUNT = 253
SKIP = 42
RETURN_COUNT = 252
BLOCK_COUNT = 12
BLOCK_SESSIONS = 21
NEUTRAL_PSEUDO_BLOCKS = 4
DENOMINATOR = BLOCK_COUNT + NEUTRAL_PSEUDO_BLOCKS
INITIAL_PARTIAL = '2016-08-17'


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'), control=float('nan'),
        endpoint_indices=[], block_start_indices=[], block_end_indices=[],
        relative_signs=[], absolute_signs=[],
        relative_positive_count=0, relative_negative_count=0, relative_tie_count=0,
        absolute_positive_count=0, absolute_negative_count=0, absolute_tie_count=0,
        relative_signed_count=0, absolute_signed_count=0,
        relative_sign_observations=0, absolute_sign_observations=0)


def _relative_sign(stock_end, stock_start, benchmark_end, benchmark_start):
    """Exact comparison of ratios of the stored binary floats, with no overflow."""
    parts = []
    for value in (stock_end, stock_start, benchmark_end, benchmark_start):
        number = float(value)
        if not math.isfinite(number) or number <= 0.:
            raise ValueError('Endpoints must be finite positive prices')
        parts.append(number.as_integer_ratio())
    (sn, sd), (on, od), (bn, bd), (pn, pd_) = parts
    left = sn * pn * bd * od
    right = bn * on * sd * pd_
    return 1 if left > right else -1 if left < right else 0


def measure(stock, benchmark):
    """Shared full-window support; exact relative and absolute twelve-block signs."""
    if np.shape(stock) != (PRICE_COUNT,) or np.shape(benchmark) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 253 stock and benchmark prices')
    prices = []
    for name, source in (('stock', stock), ('benchmark', benchmark)):
        try:
            converted = np.asarray(source, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return _empty('nonnumeric_' + name + '_price')
        if not (np.isfinite(converted) & (converted > 0.)).all():
            return _empty('invalid_' + name + '_price')
        prices.append(converted)
    stock, benchmark = prices
    endpoints = list(range(0, PRICE_COUNT, BLOCK_SESSIONS))
    starts, ends = endpoints[:-1], endpoints[1:]
    relative, absolute = [], []
    for a, b in zip(starts, ends):
        relative.append(_relative_sign(stock[b], stock[a], benchmark[b], benchmark[a]))
        absolute.append(1 if stock[b] > stock[a] else -1 if stock[b] < stock[a] else 0)
    relative_sum, absolute_sum = sum(relative), sum(absolute)
    result = _empty('available')
    result.update(available=True, candidate=relative_sum / DENOMINATOR,
        control=absolute_sum / DENOMINATOR, endpoint_indices=endpoints,
        block_start_indices=starts, block_end_indices=ends,
        relative_signs=relative, absolute_signs=absolute,
        relative_positive_count=relative.count(1), relative_negative_count=relative.count(-1),
        relative_tie_count=relative.count(0), absolute_positive_count=absolute.count(1),
        absolute_negative_count=absolute.count(-1), absolute_tie_count=absolute.count(0),
        relative_signed_count=relative_sum, absolute_signed_count=absolute_sum,
        relative_sign_observations=BLOCK_COUNT, absolute_sign_observations=BLOCK_COUNT)
    return result


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + ' must be a naive midnight date')
    return day


def _frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(label + ' requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError(label + ' sessions must be unique increasing naive midnight dates')
    if not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame.columns):
        raise ValueError(label + ' columns must be unique nonempty security IDs')


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
        raise ValueError('ETF identities require an explicit sequence')
    etfs = list(meta['etfs'])
    if len(etfs) != len(set(etfs)) or any(not isinstance(s, str) or not s or s.strip() != s for s in etfs):
        raise ValueError('ETF identities must be unique nonempty strings')
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        if not (not rows and str(day.date()) == INITIAL_PARTIAL):
            if t + 1 >= len(sessions) or sessions[t + 1].to_period('M') == day.to_period('M'):
                raise ValueError('Noninitial event must be a verified official calendar month-end')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or len(symbols) != len(set(symbols))
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, t))
    return rows, width or 0, etfs


def build_features(panel, meta, calendar, benchmark_map, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'sectors'}:
        raise ValueError('Panel requires exactly adj and sectors')
    for key in panel:
        _frame(panel[key], key)
    if not callable(benchmark_map):
        raise TypeError('The frozen benchmark mapper must be callable')
    sessions = _calendar(calendar)
    queries, width, etfs = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool))
    history, reasons = [], {}
    adj, sectors = panel['adj'], panel['sectors']
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        mapping = benchmark_map(iso, symbols, sectors.loc[:day], etfs)
        if (not isinstance(mapping, dict) or set(mapping) != set(symbols)
                or any(b is not None and (not isinstance(b, str) or not b or b.strip() != b)
                       for b in mapping.values())):
            raise ValueError('Frozen mapping must contain exactly the event identities')
        if any(mapping[s] is not None for s in symbols if s in etfs):
            raise ValueError('Frozen mapping must keep candidate ETFs neutral')
        dates = sessions[t - RETURN_COUNT - SKIP:t - SKIP + 1] if t >= RETURN_COUNT + SKIP else None
        missing, windows = [], None
        if dates is not None:
            missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
            date = lambda index: str(dates[index].date())
            windows = dict(formation=dict(price_start=date(0), price_end=date(252), return_count=252),
                blocks=dict(first_start_date=date(0), last_end_date=date(252),
                            block_sessions=BLOCK_SESSIONS, block_count=BLOCK_COUNT))
        cells = []
        for j, symbol in enumerate(symbols):
            benchmark = mapping[symbol]
            if symbol in etfs:
                cell = _empty('etf_neutral')
            elif benchmark is None:
                cell = _empty('missing_benchmark')
            elif dates is None:
                cell = _empty('insufficient_history')
            elif symbol not in adj.columns:
                cell = _empty('missing_symbol')
            elif benchmark not in adj.columns:
                cell = _empty('missing_benchmark_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(adj.loc[dates, symbol].to_numpy(copy=True),
                               adj.loc[dates, benchmark].to_numpy(copy=True))
            for key in values:
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cell = dict(cell, symbol=symbol, benchmark=benchmark, date=iso, windows=windows)
            for index_key, date_key in (('endpoint_indices', 'endpoint_dates'),
                    ('block_start_indices', 'block_start_dates'), ('block_end_indices', 'block_end_dates')):
                cell[date_key] = [str(dates[a].date()) for a in cell[index_key]]
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, return_count=RETURN_COUNT,
        block_count=BLOCK_COUNT, block_sessions=BLOCK_SESSIONS, neutral_pseudo_blocks=NEUTRAL_PSEUDO_BLOCKS,
        denominator=DENOMINATOR, available_cells=available, unavailable_cells=int(values['available'].size) - available,
        reason_counts=reasons, relative_sign_observations=BLOCK_COUNT * available,
        absolute_sign_observations=BLOCK_COUNT * available, predictive_model_fits=0,
        common_availability=True, portfolio_returns_calculated=False, ranking_performed=False,
        trading_eligibility_modified=False, event_history=history)
