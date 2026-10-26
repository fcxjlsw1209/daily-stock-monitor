"""Pure rank-volume weighted log-price slopes on fixed skip42 windows.

No file I/O, fitting of predictive models, portfolio selection or returns. The
host owns cross-sectional score ranks and the original single episode selector.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd

PRICE_COUNT = VOLUME_COUNT = 253
SPAN = 252
SKIP = 42
START_OFFSET = SPAN + SKIP
INITIAL_PARTIAL = '2016-08-17'
COUNTS = ('checked_cells', 'eligible_cells', 'attempted_cells', 'available_cells',
          'unavailable_cells', 'initial_partial_cells', 'ineligible_cells',
          'source_price_observations', 'source_volume_observations',
          'volume_rank_observations', 'descriptive_time_slope_estimates',
          'unweighted_slope_solves', 'weighted_slope_solves',
          'constant_volume_copies', 'constant_price_shortcuts',
          'constant_price_cells', 'constant_volume_cells',
          'cross_sectional_rank_observations')


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite derived slope arithmetic')
    return value


def _sum(values):
    try:
        return _finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid derived slope sum') from exc


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'),
                control=float('nan'), prices=[], volumes=[], volume_ranks=[],
                weights=[], log_prices_relative=[], constant_price=None,
                constant_volume=None, control_stats=None, candidate_stats=None,
                control_branch=None, candidate_branch=None)


def _source(values, kind):
    raw = np.asarray(values, dtype=object)
    if raw.shape != (PRICE_COUNT,):
        raise ValueError('Each source must contain exactly253 observations')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in raw):
        return None, 'nonnumeric_' + kind
    try:
        result = [float(v) for v in raw]
    except (OverflowError, ValueError, TypeError):
        return None, 'invalid_' + kind
    if any(not math.isfinite(v) or v <= 0. for v in result):
        return None, 'invalid_' + kind
    return result, None


def _ranks(values):
    """One-based average ranks, using exact stored numeric equality."""
    result = [0.] * len(values)
    indices = sorted(range(len(values)), key=values.__getitem__)
    first = 0
    while first < len(indices):
        last = first + 1
        while last < len(indices) and values[indices[last]] == values[indices[first]]:
            last += 1
        rank = (first + 1 + last) / 2.
        for j in indices[first:last]:
            result[j] = rank
        first = last
    return result


def _slope(x, y, weights):
    total = _sum(weights)
    if total <= 0.:
        raise FloatingPointError('Nonpositive derived weight total')
    xbar = _finite(_sum(_finite(w*a) for w, a in zip(weights, x)) / total)
    ybar = _finite(_sum(_finite(w*b) for w, b in zip(weights, y)) / total)
    try:
        numerator = _sum(_finite(w*(a-xbar)*(b-ybar)) for w, a, b in zip(weights, x, y))
        denominator = _sum(_finite(w*(a-xbar)**2) for w, a in zip(weights, x))
    except OverflowError as exc:
        raise FloatingPointError('Overflow in derived slope arithmetic') from exc
    if denominator <= 0.:
        raise FloatingPointError('Nonpositive derived slope denominator')
    slope = _finite(numerator/denominator)
    return dict(weight_sum=total, x_mean=xbar, y_mean=ybar,
                numerator=numerator, denominator=denominator, slope=slope)


def measure(prices, volumes):
    """Raw weighted and unweighted slopes, with the same complete source mask."""
    if np.shape(prices) != (PRICE_COUNT,) or np.shape(volumes) != (VOLUME_COUNT,):
        raise ValueError('Expected exactly253 prices and253 stored volumes')
    p, error = _source(prices, 'price')
    if error:
        return _empty(error)
    v, error = _source(volumes, 'volume')
    if error:
        return _empty(error)
    ranks = _ranks(v)
    weights = [_finite(.5 + (rank-1.)/SPAN) for rank in ranks]
    flat_price = all(value == p[0] for value in p)
    flat_volume = all(value == v[0] for value in v)
    result = dict(available=True, reason='available', prices=p, volumes=v,
                  volume_ranks=ranks, weights=weights, constant_price=flat_price,
                  constant_volume=flat_volume, control_stats=None, candidate_stats=None)
    if flat_price:
        result.update(log_prices_relative=[0.]*PRICE_COUNT, candidate=0., control=0.,
                      control_branch='constant_price_zero', candidate_branch='constant_price_zero')
        return result
    try:
        initial = _finite(math.log(p[0]))
        y = [_finite(_finite(math.log(value))-initial) for value in p]
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Invalid derived log-price arithmetic') from exc
    x = [i/SPAN for i in range(PRICE_COUNT)]
    control = _slope(x, y, [1.]*PRICE_COUNT)
    if flat_volume:
        candidate = dict(control)
        branch = 'constant_volume_copy'
    else:
        candidate = _slope(x, y, weights)
        branch = 'weighted_slope'
    result.update(log_prices_relative=y, candidate=candidate['slope'], control=control['slope'],
                  control_stats=control, candidate_stats=candidate,
                  control_branch='unweighted_slope', candidate_branch=branch)
    return result


def centered_rank(raw, available):
    """Exact-tie symmetric ranks on a supplied common available eligible set."""
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or raw.dtype.kind not in 'fi' or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite numeric raw values and boolean availability required')
    result = np.zeros(raw.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) < 2:
        return result
    observed = [float(raw[j]) for j in indices]
    if all(v == observed[0] for v in observed):
        return result
    result[indices] = [(rank-1.)/(len(indices)-1)-.5 for rank in _ranks(observed)]
    return result


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label+' requires a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label+' requires a naive midnight date')
    return day


def _frame(frame, name):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(name+' requires a DataFrame with DatetimeIndex')
    idx = frame.index
    if (idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing
            or idx.tz is not None or not idx.equals(idx.normalize())):
        raise ValueError(name+' sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                         for s in frame.columns)):
        raise ValueError(name+' symbols must be unique nonempty strings')


def _calendar(calendar):
    if isinstance(calendar, (str, bytes)):
        raise ValueError('Calendar requires a sequence')
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


def _queries(meta, sessions, eligible):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    for key in ('dates', 'symbols', 'etfs'):
        if isinstance(meta[key], (str, bytes)):
            raise ValueError('Metadata '+key+' requires a sequence')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    etfs = list(meta['etfs'])
    if (len(etfs) != len(set(etfs)) or any(not isinstance(s, str) or not s or s.strip() != s for s in etfs)):
        raise ValueError('ETF identities must be unique nonempty strings')
    if (eligible.ndim != 2 or eligible.dtype != bool or eligible.shape[0] != len(meta['dates'])):
        raise ValueError('Eligibility requires exact boolean event/candidate matrix')
    width = eligible.shape[1]
    lookup = {day: i for i, day in enumerate(sessions)}
    rows = []
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        if str(day.date()) != INITIAL_PARTIAL:
            if t+1 >= len(sessions) or sessions[t+1].to_period('M') == day.to_period('M'):
                raise ValueError('Event requires an official next-session month-end witness')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require a sequence')
        symbols = list(supplied)
        if (not symbols or len(symbols) != width or len(set(symbols)) != len(symbols)
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must match eligibility width and be unique nonempty strings')
        rows.append((day, symbols, t))
    return rows, width


def build_features(panel, meta, calendar, eligible, limit=None):
    """Read only supported eligible formation windows and return raw values/audit.

    Optional logical prefixes preserve metadata validation; physical prefixes
    may provide empty queries and an eligibility matrix of shape (0,K).
    """
    if not isinstance(panel, dict) or set(panel) != {'adj', 'volume'}:
        raise ValueError('Panel requires exactly adj and volume')
    for name in ('adj', 'volume'):
        _frame(panel[name], name)
    sessions = _calendar(calendar)
    eligible = np.asarray(eligible)
    queries, width = _queries(meta, sessions, eligible)
    if limit is None:
        limit = len(queries)
    if type(limit) is not int or not 0 <= limit <= len(queries):
        raise ValueError('Limit requires a valid integer event prefix')
    queries = queries[:limit]
    eligibility = eligible[:limit]
    values = dict(candidate=np.full((limit, width), np.nan), control=np.full((limit, width), np.nan),
                  available=np.zeros((limit, width), dtype=bool))
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(checked_cells=limit*width, eligible_cells=int(eligibility.sum()))
    reasons, history = Counter(), []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        dates = sessions[t-START_OFFSET:t-SKIP+1] if t >= START_OFFSET else None
        missing_p = [] if dates is None else dates.difference(panel['adj'].index).strftime('%Y-%m-%d').tolist()
        missing_v = [] if dates is None else dates.difference(panel['volume'].index).strftime('%Y-%m-%d').tolist()
        cells = []
        for j, symbol in enumerate(symbols):
            active = bool(eligibility[i, j])
            if iso == INITIAL_PARTIAL:
                cell = _empty('initial_partial'); counts['initial_partial_cells'] += 1
            elif not active:
                cell = _empty('ineligible'); counts['ineligible_cells'] += 1
            else:
                counts['attempted_cells'] += 1
                if dates is None:
                    cell = _empty('insufficient_history')
                elif symbol not in panel['adj'].columns:
                    cell = _empty('missing_price_symbol')
                elif symbol not in panel['volume'].columns:
                    cell = _empty('missing_volume_symbol')
                elif missing_p:
                    cell = _empty('missing_price_session')
                elif missing_v:
                    cell = _empty('missing_volume_session')
                else:
                    cell = measure(panel['adj'].loc[dates, symbol].to_numpy(copy=True),
                                   panel['volume'].loc[dates, symbol].to_numpy(copy=True))
            for key in values:
                values[key][i, j] = cell[key]
            reasons[cell['reason']] += 1
            if cell['available']:
                counts['available_cells'] += 1
                counts['constant_price_cells'] += int(cell['constant_price'])
                counts['constant_volume_cells'] += int(cell['constant_volume'])
                counts['constant_price_shortcuts'] += int(cell['constant_price'])
                counts['unweighted_slope_solves'] += int(not cell['constant_price'])
                counts['weighted_slope_solves'] += int(not cell['constant_price'] and not cell['constant_volume'])
                counts['constant_volume_copies'] += int(not cell['constant_price'] and cell['constant_volume'])
            else:
                cell['candidate'] = cell['control'] = None
            cell.update(symbol=symbol, date=iso, column=j, eligible=active)
            cells.append(cell)
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            source_start=None if dates is None else str(dates[0].date()),
            source_end=None if dates is None else str(dates[-1].date()),
            source_dates=[] if dates is None else dates.strftime('%Y-%m-%d').tolist(),
            missing_price_sessions=missing_p, missing_volume_sessions=missing_v, cells=cells))
    accepted = counts['available_cells']
    counts.update(unavailable_cells=counts['checked_cells']-accepted,
        source_price_observations=PRICE_COUNT*accepted, source_volume_observations=VOLUME_COUNT*accepted,
        volume_rank_observations=VOLUME_COUNT*accepted, descriptive_time_slope_estimates=2*accepted)
    return values, dict(schema='volume_slope232_features_v1', events=limit, symbols_per_event=width,
        **counts, window_price_count=PRICE_COUNT, window_volume_count=VOLUME_COUNT,
        span_sessions=SPAN, skip_sessions=SKIP, source_start_offset=-START_OFFSET,
        source_end_offset=-SKIP, reason_counts=dict(reasons), common_availability=True,
        feature_domain='original_eligible', within_volume_ranking_performed=True,
        cross_sectional_ranking_performed=False, trading_eligibility_modified=False,
        predictive_model_fits=0, new_model_fits=0, portfolio_returns_calculated=False,
        event_history=history)
