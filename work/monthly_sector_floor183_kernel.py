"""Own-sector relative returns conditional on sector up/down days.

Pure source measurement only. The caller supplies the separately pinned
original sector mapper. Flat sector days contribute to neither group nor the
pooled control. Every price is required and ends42 sessions before the query.
"""
import math

import numpy as np
import pandas as pd


PRICE_COUNT = 211
SKIP = 42
MINIMUM_GROUP_COUNT = 40


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'), control=float('nan'),
                up_count=0, down_count=0, flat_count=0, mean_up=None, mean_down=None,
                pooled_mean=None, conditional_mean_estimates=0, pooled_mean_estimates=0,
                excess_log_returns=[], sector_states=[])


def _finite_sum(values):
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite sector-relative sum') from exc
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite sector-relative sum')
    return value


def measure(stock, benchmark):
    """Compute min(two conditional means) and their frequency-weighted pool.

Up/down classification is exact on the supplied sector prices, not on rounded
log differences. There is no beta, variance normalization, price fill, count
shrinkage, or return floor. Insufficient groups retain source diagnostics but
neither arm is available and no conditional/pooled mean is estimated.
"""
    if np.shape(stock) != (PRICE_COUNT,) or np.shape(benchmark) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 211 stock and benchmark prices')
    prices = []
    for name, source in (('stock', stock), ('benchmark', benchmark)):
        try:
            converted = np.asarray(source, dtype=float)
        except (ValueError, TypeError, OverflowError):
            return _empty('nonnumeric_' + name + '_price')
        if not (np.isfinite(converted) & (converted > 0.)).all():
            return _empty('invalid_' + name + '_price')
        prices.append(converted)
    stock, benchmark = prices
    logs = [[math.log(float(price)) for price in array] for array in prices]
    if not all(math.isfinite(x) for row in logs for x in row):
        raise FloatingPointError('Positive prices produced nonfinite log levels')
    excess, states = [], []
    for i in range(1, PRICE_COUNT):
        value = (logs[0][i] - logs[0][i-1]) - (logs[1][i] - logs[1][i-1])
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite sector-relative log return')
        state = 1 if benchmark[i] > benchmark[i-1] else -1 if benchmark[i] < benchmark[i-1] else 0
        excess.append(value); states.append(state)
    up = [value for value, state in zip(excess, states) if state == 1]
    down = [value for value, state in zip(excess, states) if state == -1]
    out = _empty('insufficient_groups')
    out.update(up_count=len(up), down_count=len(down), flat_count=states.count(0),
               excess_log_returns=excess, sector_states=states)
    if min(len(up), len(down)) < MINIMUM_GROUP_COUNT:
        return out
    mean_up, mean_down = _finite_sum(up)/len(up), _finite_sum(down)/len(down)
    pooled = _finite_sum(value for value, state in zip(excess, states) if state)/(len(up)+len(down))
    if not all(math.isfinite(x) for x in (mean_up, mean_down, pooled)):
        raise FloatingPointError('Finite relative returns produced nonfinite means')
    out.update(available=True, reason='available', candidate=min(mean_up, mean_down),
               control=pooled, mean_up=mean_up, mean_down=mean_down, pooled_mean=pooled,
               conditional_mean_estimates=2, pooled_mean_estimates=1)
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
    """Use original query-date classifications, without later source values."""
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
        mapping = benchmark_map(str(day.date()), symbols, sectors.loc[:day], etfs)
        if (not isinstance(mapping, dict) or set(mapping) != set(symbols)
                or any(b is not None and (not isinstance(b, str) or not b or b.strip() != b)
                       for b in mapping.values())):
            raise ValueError('Frozen mapping must contain exactly the event identities')
        if any(mapping[s] is not None for s in symbols if s in etfs):
            raise ValueError('Frozen mapping must keep candidate ETFs neutral')
        dates = sessions[t-252:t-SKIP+1] if t >= 252 else None
        missing, windows = [], None
        if dates is not None:
            missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
            windows = dict(formation=dict(price_start=str(dates[0].date()), price_end=str(dates[-1].date()), return_count=210),
                           first_return_date=str(dates[1].date()), last_return_date=str(dates[-1].date()))
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
            reasons[cell['reason']] = reasons.get(cell['reason'], 0)+1
            cell = dict(cell, symbol=symbol, benchmark=benchmark, date=str(day.date()), windows=windows)
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=symbols, session_position=t,
                            price_start=str(dates[0].date()) if dates is not None else None,
                            price_end=str(dates[-1].date()) if dates is not None else None,
                            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
                        price_count=PRICE_COUNT, skip_sessions=SKIP, return_count=210,
                        minimum_group_count=MINIMUM_GROUP_COUNT, available_cells=available,
                        unavailable_cells=int(values['available'].size)-available, reason_counts=reasons,
                        conditional_mean_estimates=2*available, pooled_mean_estimates=available,
                        predictive_model_fits=0, common_availability=True, portfolio_returns_calculated=False,
                        ranking_performed=False, trading_eligibility_modified=False, event_history=history)
