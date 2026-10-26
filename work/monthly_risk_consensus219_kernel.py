"""Pure same-window return-amplitude / arithmetic-Sharpe consensus.

No source I/O, eligibility filter, model fit, episode, order or return replay.
Initial query targets are computed normally; the host suppresses its core delta.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd

PRICE_COUNT = 253
SKIP = 42
WINDOW_RETURNS = 252
FEATURE_KEYS = ('available', 'candidate', 'control', 'momentum', 'mean_return',
                'sample_vol', 'sharpe', 'rank_momentum', 'rank_sharpe')
COUNT_FIELDS = ('checked_cells', 'source_available_cells', 'available_cells',
                'unavailable_cells', 'valid_price_observations', 'simple_returns',
                'ranked_observations', 'aggregated_scores')
INITIAL_PARTIAL = '2016-08-17'


def empty_measure(reason):
    return dict(source_available=False, available=False, reason=reason,
                start_price=None, end_price=None, simple_return_sum=None,
                mean_return=np.nan, centered_square_sum=None, sample_vol=np.nan,
                momentum=np.nan, sharpe=np.nan)


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from finite positive source')
    return value


def measure(prices):
    """253 positive prices; fsum mean and two-pass sample volatility.

    Zero volatility preserves source statistics but excludes both ranked arms.
    No tolerance, floor or clipping. Arithmetic overflow is a hard failure.
    """
    if np.shape(prices) != (PRICE_COUNT,):
        raise ValueError('Expected exactly253 adjusted prices')
    raw = np.asarray(prices, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:
        p = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return empty_measure('nonnumeric_adjusted_price')
    if not (np.isfinite(p) & (p > 0)).all():
        return empty_measure('invalid_adjusted_price')
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            momentum = _finite(p[-1]/p[0]-1.)
            returns = [_finite(p[j+1]/p[j]-1.) for j in range(WINDOW_RETURNS)]
        total = _finite(math.fsum(returns))
        mean = _finite(total/WINDOW_RETURNS)
        differences = [_finite(r-mean) for r in returns]
        squares = [_finite(difference*difference) for difference in differences]
        square_sum = _finite(math.fsum(squares))
        sample_vol = _finite(math.sqrt(_finite(square_sum/(WINDOW_RETURNS-1))))
        sharpe = _finite(mean/sample_vol) if sample_vol != 0. else np.nan
    except (OverflowError, ZeroDivisionError) as exc:
        raise FloatingPointError('Overflow arithmetic from finite positive source') from exc
    available = sample_vol != 0.
    return dict(source_available=True, available=available,
                reason='available' if available else 'zero_sample_volatility',
                start_price=float(p[0]), end_price=float(p[-1]), simple_return_sum=total,
                mean_return=mean, centered_square_sum=square_sum, sample_vol=sample_vol,
                momentum=momentum, sharpe=sharpe)


def percentile_rank(raw, available):
    """Average ascending percentile minus.5; no eligibility or second ranking."""
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices):
        result[indices] = pd.Series(raw[indices]).rank(method='average', pct=True).to_numpy()-.5
    return result


def aggregate(ranks):
    if np.shape(ranks) != (2,):
        raise ValueError('Exactly two component ranks required')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in ranks):
        raise ValueError('Numeric component ranks required')
    r = [float(x) for x in ranks]
    if not all(math.isfinite(x) and -.5 <= x <= .5 for x in r):
        raise ValueError('Finite bounded component ranks required')
    equal = r[0] == r[1]
    if equal:
        return r[0], r[0], True
    return min(r), math.fsum(r)/2, False


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label+' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label+' must be a naive midnight date')
    return day


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted prices require DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Price sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                         for s in frame.columns)):
        raise ValueError('Price columns must be unique nonempty security IDs')


def _calendar(calendar):
    if isinstance(calendar, (str, bytes)):
        raise ValueError('Calendar requires an explicit sequence')
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
    if isinstance(meta['dates'], (str, bytes)) or isinstance(meta['symbols'], (str, bytes)):
        raise ValueError('Metadata dates and symbols require sequences')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF identities require an explicit sequence')
    etfs = list(meta['etfs'])
    if (any(not isinstance(s, str) or not s or s.strip() != s for s in etfs)
            or len(etfs) != len(set(etfs))):
        raise ValueError('ETF identities must be unique nonempty strings')
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        if not (not rows and str(day.date()) == INITIAL_PARTIAL):
            if t+1 >= len(sessions) or sessions[t+1].to_period('M') == day.to_period('M'):
                raise ValueError('Event must be a verified official calendar month-end')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
                or len(symbols) != len(set(symbols))):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, t))
    return rows, width or 0


def build_features(panel, meta, calendar, limit=None):
    """Nine matrices: descriptive M/mean/vol survive source-valid zero risk."""
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    adj = panel['adj']; _frame(adj)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = {key: np.full(shape, np.nan) for key in FEATURE_KEYS if key != 'available'}
    values['available'] = np.zeros(shape, bool)
    history, reasons, counts, source_counts = [], Counter(), [], []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        dates = sessions[t-294:t-42+1] if t >= 294 else None
        missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells = []
        for j, symbol in enumerate(symbols):
            if dates is None:
                result = empty_measure('insufficient_history')
            elif symbol not in adj.columns:
                result = empty_measure('missing_symbol')
            elif missing:
                result = empty_measure('missing_official_session')
            else:
                result = measure(adj.loc[dates, symbol].to_numpy(copy=True))
            values['available'][i, j] = result['available']
            for key in ('momentum', 'mean_return', 'sample_vol', 'sharpe'):
                values[key][i, j] = result[key]
            cell = dict(result, symbol=symbol, date=iso, rank_momentum=None, rank_sharpe=None,
                        candidate=None, control=None, exact_equal_ranks=False)
            for key in ('momentum', 'mean_return', 'sample_vol', 'sharpe'):
                if not math.isfinite(cell[key]): cell[key] = None
            cells.append(cell)
        for key in ('momentum', 'sharpe'):
            values['rank_'+key][i] = percentile_rank(values[key][i], values['available'][i])
        for j, cell in enumerate(cells):
            if cell['available']:
                qm, qs = float(values['rank_momentum'][i, j]), float(values['rank_sharpe'][i, j])
                q, c, equal = aggregate([qm, qs])
                values['candidate'][i, j], values['control'][i, j] = q, c
                cell.update(rank_momentum=qm, rank_sharpe=qs,
                            candidate=q, control=c, exact_equal_ranks=equal)
            reasons[cell['reason']] += 1
        count = int(values['available'][i].sum()); counts.append(count)
        source_count = sum(c['source_available'] for c in cells); source_counts.append(source_count)
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            first_return_date=str(dates[1].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells], available=[c['available'] for c in cells],
            source_available_count=source_count, available_count=count, cells=cells))
    total, source_total = sum(counts), sum(source_counts)
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, window_returns=WINDOW_RETURNS, std_ddof=1,
        source_available_cells=source_total, source_available_by_event=list(source_counts),
        available_cells=total, available_by_event=list(counts),
        unavailable_cells=int(values['available'].size)-total, reason_counts=dict(reasons),
        valid_price_observations=253*source_total, simple_returns=252*source_total,
        ranked_observations=2*total, aggregated_scores=2*total,
        predictive_model_fits=0, new_model_fits=0, common_availability=True, ranking_performed=True,
        ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, event_history=history)
