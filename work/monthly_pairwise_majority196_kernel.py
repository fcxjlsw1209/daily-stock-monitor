"""Exact historical pairwise majority and balance; no model or portfolio I/O."""
from collections import Counter
from fractions import Fraction
import numbers

import numpy as np
import pandas as pd

PRICE_COUNT = 253
BLOCK_COUNT = 12
BLOCK_SESSIONS = 21
SKIP = 42
INITIAL_PARTIAL = '2016-08-17'


def source_prices(prices):
    """Validate every formation price, before retaining the13 block endpoints."""
    if np.shape(prices) != (PRICE_COUNT,):
        raise ValueError('Expected exactly253 adjusted prices')
    raw = np.asarray(prices, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return None, 'nonnumeric_adjusted_price'
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, 'nonnumeric_adjusted_price'
    if not (np.isfinite(numeric) & (numeric > 0)).all():
        return None, 'invalid_adjusted_price'
    return numeric, 'available'


def tournament(endpoint_rows):
    """13 binary64 endpoints per source-valid name; one comparison per unordered pair.

    No rounded gross-return division or floating multiplication enters a sign.
    Empty/singleton domains have no feature values, but retain the matrix axis.
    """
    source = np.asarray(endpoint_rows)
    if source.ndim != 2 or source.shape[1] != BLOCK_COUNT+1:
        raise ValueError('Endpoint matrix must have shape(n,13)')
    raw = np.asarray(endpoint_rows, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw.flat):
        raise ValueError('Endpoint matrix must be numeric')
    source = np.asarray(raw, dtype=np.float64)
    if not (np.isfinite(source) & (source > 0)).all():
        raise ValueError('Endpoints must be finite strictly positive')
    size = len(source)
    rational = [[Fraction.from_float(float(x)) for x in row] for row in source]
    matrix = [[0]*size for _ in range(size)]
    pairs = []
    for i in range(size):
        for j in range(i+1, size):
            signs = []
            for k in range(BLOCK_COUNT):
                left = rational[i][k+1] * rational[j][k]
                right = rational[j][k+1] * rational[i][k]
                signs.append(int(left > right)-int(left < right))
            margin = sum(signs)
            matrix[i][j], matrix[j][i] = margin, -margin
            pairs.append(dict(left_index=i, right_index=j, block_signs=signs,
                              margin=margin, majority_sign=int(margin > 0)-int(margin < 0)))
    candidate, control = np.full(size, np.nan), np.full(size, np.nan)
    majority_sums, margin_sums = [None]*size, [None]*size
    if size >= 2:
        for i, row in enumerate(matrix):
            majority_sums[i] = sum(int(m > 0)-int(m < 0) for m in row)
            margin_sums[i] = sum(row)
            candidate[i] = majority_sums[i]/(size-1)
            control[i] = margin_sums[i]/(BLOCK_COUNT*(size-1))
    return dict(candidate=candidate, control=control, pair_margin_matrix=matrix, pairs=pairs,
                majority_sums=majority_sums, margin_sums=margin_sums)


def centered_rank(raw, available):
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.zeros(raw.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) >= 2:
        ranked = pd.Series(raw[indices]).rank(method='average').to_numpy()
        result[indices] = (ranked-1)/(len(indices)-1)-.5
    return result


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
    """Five aligned arrays; source-domain facts are separately retained in audit."""
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    adj = panel['adj']
    _frame(adj)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons, domain_counts = [], Counter(), []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        dates = sessions[t-(PRICE_COUNT-1)-SKIP:t-SKIP+1] if t >= PRICE_COUNT-1+SKIP else None
        missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        endpoint_dates = [str(dates[k].date()) for k in range(0, PRICE_COUNT, BLOCK_SESSIONS)] if dates is not None else []
        cells, indices, endpoints = [], [], []
        for j, symbol in enumerate(symbols):
            prices = None
            if dates is None:
                reason = 'insufficient_history'
            elif symbol not in adj.columns:
                reason = 'missing_symbol'
            elif missing:
                reason = 'missing_official_session'
            else:
                prices, reason = source_prices(adj.loc[dates, symbol].to_numpy(copy=True))
            valid = prices is not None
            endpoint = prices[::BLOCK_SESSIONS].tolist() if valid else []
            if valid:
                indices.append(j)
                endpoints.append(endpoint)
            cells.append(dict(symbol=symbol, date=iso, source_available=valid, source_reason=reason,
                available=False, reason=reason, endpoint_prices=endpoint, opponent_count=0,
                majority_sum=None, margin_sum=None, candidate=None, control=None,
                candidate_rank=0., control_rank=0.))
        count = len(indices)
        domain_counts.append(count)
        result = tournament(np.asarray(endpoints, dtype=float).reshape(count, BLOCK_COUNT+1))
        for local, j in enumerate(indices):
            cell = cells[j]
            if count < 2:
                cell['reason'] = 'insufficient_comparison_domain'
                continue
            cell.update(available=True, reason='available', opponent_count=count-1,
                majority_sum=result['majority_sums'][local], margin_sum=result['margin_sums'][local],
                candidate=float(result['candidate'][local]), control=float(result['control'][local]))
            values['available'][i, j] = True
            values['candidate'][i, j] = cell['candidate']
            values['control'][i, j] = cell['control']
        for arm in ('candidate', 'control'):
            values[arm+'_rank'][i] = centered_rank(values[arm][i], values['available'][i])
        for j, cell in enumerate(cells):
            cell.update(candidate_rank=float(values['candidate_rank'][i, j]),
                        control_rank=float(values['control_rank'][i, j]))
            reasons[cell['reason']] += 1
        pairs = [dict(pair, left_index=indices[pair['left_index']], right_index=indices[pair['right_index']])
                 for pair in result['pairs']]
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            endpoint_dates=endpoint_dates, missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells], source_indices=indices,
            domain_count=count, source_available_count=count, available_count=count if count >= 2 else 0,
            pair_margin_matrix=result['pair_margin_matrix'], pairs=pairs, cells=cells))
    source_count, available = sum(domain_counts), int(values['available'].sum())
    pair_count = sum(n*(n-1)//2 for n in domain_counts)
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, block_count=BLOCK_COUNT, block_sessions=BLOCK_SESSIONS,
        source_available_cells=source_count, source_available_by_event=domain_counts,
        available_cells=available, unavailable_cells=int(values['available'].size)-available,
        reason_counts=dict(reasons), valid_price_observations=PRICE_COUNT*source_count,
        block_comparisons=BLOCK_COUNT*pair_count, pair_majority_decisions=pair_count,
        predictive_model_fits=0, new_model_fits=0, common_availability=True,
        ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False, event_history=history)
