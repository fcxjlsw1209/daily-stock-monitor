"""Pure historical sector-correlation dispersion and own-sector quality.

No I/O, strategy selection, return replay, or predictive model fitting occurs
here. Every required price ends at least 42 official sessions before a query.
"""
import math

import numpy as np
import pandas as pd


SECTOR_ETFS = ('XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')
PAIRS = tuple((i, j) for i in range(9) for j in range(i + 1, 9))
SKIP = 42
MARKET_PRICE_COUNT = 1513
MARKET_RETURN_SESSIONS = 252
HISTORICAL_WINDOWS = 60
WINDOW_SPACING = 21
RELATIVE_PRICE_COUNT = 211
ANNUALIZATION = 252
STD_FLOOR = 1e-12
ROUNDOFF_TOLERANCE = 1e-12
MARKET_START_OFFSET = 1554
RELATIVE_START_OFFSET = 252


def _log_ratio(end, start):
    end, start = float(end), float(start)
    ratio = end / start
    if .5 <= ratio <= 2.:
        return math.log1p((end - start) / start)
    if math.isfinite(ratio) and ratio > 0.:
        return math.log(ratio)
    em, ee = math.frexp(end)
    sm, se = math.frexp(start)
    return math.log(em / sm) + (ee - se) * math.log(2.)


def _bounded(value, low, high, label):
    value = float(value)
    if not math.isfinite(value) or value < low - ROUNDOFF_TOLERANCE or value > high + ROUNDOFF_TOLERANCE:
        raise ArithmeticError(label + ' exceeded mathematical bounds')
    return min(high, max(low, value))


def _center(values):
    # Translate before summing: nearly constant returns retain their variation.
    base = float(values[0])
    shifted = np.asarray(values, dtype=float) - base
    offset = math.fsum(shifted) / len(shifted)
    centered = shifted - offset
    squares = math.fsum(float(x) * float(x) for x in centered)
    return base + offset, centered, math.sqrt(squares / (len(shifted) - 1)), squares


def _empty_market(reason):
    return dict(available=False, reason=reason, windows=[], dispersion_values=[],
                current_dispersion=None, historical_dispersion_mean=None,
                invalid_window_indices=[], correlation_window_estimates=0, pair_estimates=0)


def measure_market(adjusted):
    """Use 1513x9 prices, fixed SECTOR_ETFS order; k=0 is current D.

    Window k has inclusive price indices [1260-21*k, 1512-21*k]. All
    61 windows must pass. A bad window does not suppress the other diagnostics.
    """
    try:
        prices = np.asarray(adjusted, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty_market('nonnumeric_market_price')
    if prices.shape != (MARKET_PRICE_COUNT, len(SECTOR_ETFS)):
        raise ValueError('Expected exactly 1513 by 9 sector prices')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        return _empty_market('invalid_market_price')
    returns = np.array([[_log_ratio(prices[i + 1, j], prices[i, j])
                         for j in range(9)] for i in range(MARKET_PRICE_COUNT - 1)])
    windows, dispersions, invalid = [], [], []
    for k in range(HISTORICAL_WINDOWS + 1):
        a, b = 1260 - WINDOW_SPACING * k, 1512 - WINDOW_SPACING * k
        moments = [_center(returns[a:b, j]) for j in range(9)]
        means = [m[0] for m in moments]
        sds = [m[2] for m in moments]
        good = all(math.isfinite(s) and s > STD_FLOOR for s in sds)
        correlations, dispersion = [], None
        if good:
            for i, j in PAIRS:
                cross = math.fsum(float(x) * float(y) for x, y in zip(moments[i][1], moments[j][1]))
                # Separate square roots avoid multiplying tiny sums of squares.
                rho = (cross / math.sqrt(moments[i][3])) / math.sqrt(moments[j][3])
                correlations.append(_bounded(rho, -1., 1., 'Correlation'))
            dispersion = _bounded(1. - math.fsum(r * r for r in correlations) / len(PAIRS),
                                  0., 1., 'Sector dispersion')
        else:
            invalid.append(k)
        dispersions.append(dispersion)
        windows.append(dict(k=k, start_index=a, end_index=b, means=means,
                            standard_deviations=sds, pair_correlations=correlations,
                            dispersion=dispersion, available=good,
                            reason='available' if good else 'market_volatility_floor'))
    count = len(windows) - len(invalid)
    out = _empty_market('market_volatility_floor' if invalid else 'available')
    out.update(windows=windows, dispersion_values=dispersions,
               invalid_window_indices=invalid, correlation_window_estimates=count,
               pair_estimates=count * len(PAIRS))
    if not invalid:
        out.update(available=True, current_dispersion=dispersions[0],
                   historical_dispersion_mean=math.fsum(dispersions[1:]) / HISTORICAL_WINDOWS)
    return out


def _empty_relative(reason):
    return dict(available=False, reason=reason, relative_mean=None,
                relative_standard_deviation=None, relative_sharpe=None, quality=None)


def measure_relative(stock, benchmark):
    """Own-sector relative log returns from exactly 211 prices (210 returns)."""
    arrays = []
    for name, supplied in (('stock', stock), ('benchmark', benchmark)):
        try:
            array = np.asarray(supplied, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return _empty_relative('nonnumeric_' + name + '_price')
        if array.shape != (RELATIVE_PRICE_COUNT,):
            raise ValueError('Expected exactly 211 ' + name + ' prices')
        if not (np.isfinite(array) & (array > 0.)).all():
            return _empty_relative('invalid_' + name + '_price')
        arrays.append(array)
    stock, benchmark = arrays
    relative = np.array([_log_ratio(stock[i + 1], stock[i]) - _log_ratio(benchmark[i + 1], benchmark[i])
                         for i in range(RELATIVE_PRICE_COUNT - 1)])
    mean, _, sd, _ = _center(relative)
    out = _empty_relative('relative_volatility_floor')
    out.update(relative_mean=mean, relative_standard_deviation=sd)
    if not math.isfinite(sd) or sd <= STD_FLOOR:
        return out
    sharpe = math.sqrt(ANNUALIZATION) * mean / sd
    if not math.isfinite(sharpe):
        raise ArithmeticError('Finite relative returns produced nonfinite quality')
    out.update(available=True, reason='available', relative_sharpe=sharpe, quality=math.tanh(sharpe))
    return out


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
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame.columns)):
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
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or len(symbols) != len(set(symbols))
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, lookup[day]))
    return rows, width or 0, etfs


def build_features(panel, meta, calendar, benchmark_map, limit=None):
    """Measure only the requested event prefix, with a separately pinned mapper.

    Adj and sector-classification frames have independent axes. The original
    mapper receives classifications through the event date only. No candidate
    ETF receives Q; sector proxies XLC and XLRE remain legitimate for stocks.
    """
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
    history, reasons, window_count = [], {}, 0
    adj, sectors = panel['adj'], panel['sectors']
    for i, (day, symbols, t) in enumerate(queries):
        mapping = benchmark_map(str(day.date()), symbols, sectors.loc[:day], etfs)
        if (not isinstance(mapping, dict) or set(mapping) != set(symbols)
                or any(b is not None and (not isinstance(b, str) or not b or b.strip() != b)
                       for b in mapping.values())):
            raise ValueError('Frozen mapping must contain exactly the event identities')
        if any(mapping[s] is not None for s in symbols if s in etfs):
            raise ValueError('Frozen mapping must keep candidate ETFs neutral')
        enough = t >= MARKET_START_OFFSET
        market_dates = sessions[t - MARKET_START_OFFSET:t - SKIP + 1] if enough else None
        relative_dates = sessions[t - RELATIVE_START_OFFSET:t - SKIP + 1] if t >= RELATIVE_START_OFFSET else None
        missing_symbols = [s for s in SECTOR_ETFS if s not in adj.columns]
        missing_dates = ([] if not enough else
                         market_dates.difference(adj.index).strftime('%Y-%m-%d').tolist())
        if not enough:
            market = _empty_market('insufficient_history')
        elif missing_symbols:
            market = _empty_market('missing_market_symbol')
        elif missing_dates:
            market = _empty_market('missing_market_session')
        else:
            market = measure_market(adj.loc[market_dates, list(SECTOR_ETFS)].to_numpy(copy=True))
        for window in market['windows']:
            window['start_date'] = str(market_dates[window['start_index']].date())
            window['end_date'] = str(market_dates[window['end_index']].date())
        window_count += market['correlation_window_estimates']
        cells = []
        for j, symbol in enumerate(symbols):
            benchmark = mapping[symbol]
            if symbol in etfs:
                detail = _empty_relative('etf_neutral')
            elif benchmark is None:
                detail = _empty_relative('missing_benchmark')
            elif not market['available']:
                detail = _empty_relative(market['reason'])
            elif symbol not in adj.columns:
                detail = _empty_relative('missing_symbol')
            elif benchmark not in adj.columns:
                detail = _empty_relative('missing_benchmark_symbol')
            elif len(relative_dates.difference(adj.index)):
                detail = _empty_relative('missing_relative_session')
            else:
                detail = measure_relative(adj.loc[relative_dates, symbol].to_numpy(copy=True),
                                          adj.loc[relative_dates, benchmark].to_numpy(copy=True))
            good = detail['available']
            candidate = market['current_dispersion'] * detail['quality'] if good else None
            control = market['historical_dispersion_mean'] * detail['quality'] if good else None
            cell = dict(detail, symbol=symbol, benchmark=benchmark, candidate=candidate, control=control)
            values['candidate'][i, j] = candidate if good else np.nan
            values['control'][i, j] = control if good else np.nan
            values['available'][i, j] = good
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=list(symbols), session_position=t,
                            market_start=str(market_dates[0].date()) if enough else None,
                            market_end=str(market_dates[-1].date()) if enough else None,
                            relative_start=str(relative_dates[0].date()) if relative_dates is not None else None,
                            relative_end=str(relative_dates[-1].date()) if relative_dates is not None else None,
                            missing_market_sessions=missing_dates, missing_market_symbols=missing_symbols,
                            market=market, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    return values, dict(events=len(queries), symbols_per_event=width, sector_etfs=list(SECTOR_ETFS),
                        pair_indices=[list(p) for p in PAIRS], skip_sessions=SKIP,
                        market_price_count=MARKET_PRICE_COUNT, market_return_sessions=MARKET_RETURN_SESSIONS,
                        market_window_count=HISTORICAL_WINDOWS + 1, market_window_spacing=WINDOW_SPACING,
                        historical_window_count=HISTORICAL_WINDOWS, relative_price_count=RELATIVE_PRICE_COUNT,
                        relative_return_sessions=RELATIVE_PRICE_COUNT - 1, annualization_sessions=ANNUALIZATION,
                        standard_deviation_floor=STD_FLOOR, roundoff_tolerance=ROUNDOFF_TOLERANCE,
                        available_cells=available, unavailable_cells=int(values['available'].size) - available,
                        reason_counts=reasons, event_history=history,
                        correlation_window_estimates=window_count, pair_estimates=window_count * len(PAIRS),
                        relative_quality_estimates=available, predictive_model_fits=0, common_availability=True,
                        portfolio_returns_calculated=False, ranking_performed=False, trading_eligibility_modified=False)
