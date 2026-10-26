"""Pure round233 exact signed-magnitude-rank own-sector persistence.

No data loading, predictive fit, selection or portfolio replay. Input calendar,
query and benchmark-mapping validation preserves the round192 contract locally;
the candidate mathematics has no dependency on any predecessor producer.
"""
from fractions import Fraction
import math
from numbers import Real

import numpy as np
import pandas as pd

PRICE_COUNT = 253
SKIP = 42
RETURN_COUNT = 252
BLOCK_COUNT = 12
BLOCK_SESSIONS = 21
NEUTRAL_PSEUDO_BLOCKS = 4
CANDIDATE_DENOMINATOR = 208
CONTROL_DENOMINATOR = 16
EXPECTED_TWICE_RANK_SUM = 156
INITIAL_PARTIAL = '2016-08-17'
COUNT_FIELDS = (
    'events', 'symbols_per_event', 'checked_cells', 'available_cells',
    'unavailable_cells', 'source_price_observations', 'relative_sign_observations',
    'magnitude_rank_observations', 'magnitude_pair_comparisons',
    'signed_rank_estimates', 'sign_balance_estimates',
    'constant_magnitude_copies', 'predictive_model_fits',
)


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'), control=float('nan'),
        endpoint_indices=[], block_start_indices=[], block_end_indices=[],
        relative_signs=[], exact_relative_z=[], twice_magnitude_ranks=[],
        relative_positive_count=0, relative_negative_count=0, relative_tie_count=0,
        relative_signed_count=0, candidate_numerator=0, twice_rank_sum=0,
        constant_magnitude_copy=False, relative_sign_observations=0,
        magnitude_rank_observations=0, magnitude_pair_comparisons=0)


def _positive_prices(source, name):
    values = []
    for value in np.asarray(source, dtype=object):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return None, 'nonnumeric_' + name + '_price'
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return None, 'invalid_' + name + '_price'
        if not math.isfinite(converted) or converted <= 0.:
            return None, 'invalid_' + name + '_price'
        values.append(converted)
    return values, None


def _relative_z(stock_start, stock_end, benchmark_start, benchmark_end):
    """Exact symmetric relative gain of already admitted binary64 endpoints."""
    gs = Fraction.from_float(stock_end) / Fraction.from_float(stock_start)
    gb = Fraction.from_float(benchmark_end) / Fraction.from_float(benchmark_start)
    return (gs - gb) / (gs + gb)


def _twice_ranks(magnitudes):
    """One relationship per unordered pair; exact average ties, including zero."""
    ranks = [2] * len(magnitudes)
    comparisons = 0
    for i in range(len(magnitudes)):
        for j in range(i + 1, len(magnitudes)):
            comparisons += 1
            if magnitudes[i] < magnitudes[j]:
                ranks[j] += 2
            elif magnitudes[i] > magnitudes[j]:
                ranks[i] += 2
            else:
                ranks[i] += 1
                ranks[j] += 1
    return ranks, comparisons


def measure(stock, benchmark):
    """Shared full253 support; exact signed ranks versus unweighted signs."""
    if np.shape(stock) != (PRICE_COUNT,) or np.shape(benchmark) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 253 stock and benchmark prices')
    admitted = []
    for name, source in (('stock', stock), ('benchmark', benchmark)):
        values, reason = _positive_prices(source, name)
        if reason is not None:
            return _empty(reason)
        admitted.append(values)
    stock, benchmark = admitted
    endpoints = list(range(0, PRICE_COUNT, BLOCK_SESSIONS))
    starts, ends = endpoints[:-1], endpoints[1:]
    z = [_relative_z(stock[a], stock[b], benchmark[a], benchmark[b])
         for a, b in zip(starts, ends)]
    signs = [(value > 0) - (value < 0) for value in z]
    magnitudes = [abs(value) for value in z]
    ranks, comparisons = _twice_ranks(magnitudes)
    signed_count = sum(signs)
    numerator = sum(sign * rank for sign, rank in zip(signs, ranks))
    rank_sum = sum(ranks)
    if rank_sum != EXPECTED_TWICE_RANK_SUM or comparisons != 66:
        raise ArithmeticError('Invalid exact magnitude-rank accounting')
    control = signed_count / CONTROL_DENOMINATOR
    constant = all(value == magnitudes[0] for value in magnitudes)
    candidate = control if constant else numerator / CANDIDATE_DENOMINATOR
    if not math.isfinite(candidate) or not math.isfinite(control):
        raise ArithmeticError('Derived signed-rank statistics must be finite')
    result = _empty('available')
    result.update(available=True, candidate=candidate, control=control,
        endpoint_indices=endpoints, block_start_indices=starts, block_end_indices=ends,
        relative_signs=signs, exact_relative_z=[str(value) for value in z],
        twice_magnitude_ranks=ranks, relative_positive_count=signs.count(1),
        relative_negative_count=signs.count(-1), relative_tie_count=signs.count(0),
        relative_signed_count=signed_count, candidate_numerator=numerator,
        twice_rank_sum=rank_sum, constant_magnitude_copy=constant,
        relative_sign_observations=BLOCK_COUNT, magnitude_rank_observations=BLOCK_COUNT,
        magnitude_pair_comparisons=comparisons)
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
    copies = sum(cell['constant_magnitude_copy'] for event in history for cell in event['cells'])
    return values, dict(schema='monthly_sector_signedrank233_features_v1',
        events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, return_count=RETURN_COUNT,
        block_count=BLOCK_COUNT, block_sessions=BLOCK_SESSIONS,
        neutral_pseudo_blocks=NEUTRAL_PSEUDO_BLOCKS,
        candidate_denominator=CANDIDATE_DENOMINATOR, control_denominator=CONTROL_DENOMINATOR,
        expected_twice_rank_sum=EXPECTED_TWICE_RANK_SUM,
        available_cells=available, unavailable_cells=int(values['available'].size) - available,
        source_price_observations=2 * PRICE_COUNT * available, reason_counts=reasons,
        relative_sign_observations=BLOCK_COUNT * available,
        magnitude_rank_observations=BLOCK_COUNT * available,
        magnitude_pair_comparisons=66 * available,
        signed_rank_estimates=available, sign_balance_estimates=available,
        constant_magnitude_copies=copies, predictive_model_fits=0,
        common_availability=True, portfolio_returns_calculated=False,
        cross_sectional_ranking_performed=False, within_cell_ranking_performed=True,
        trading_eligibility_modified=False, event_history=history)
