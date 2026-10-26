"""Pure descriptive drift-break statistics; no I/O, selection, or trading.

The original endpoint/volatility signal retains its wider data domain. New
signals need the entire old formation interval and never fill a missing price.
"""
import math

import numpy as np
import pandas as pd


SKIP = 42
RETURNS = 252
FORMATION_COUNT = 253
FORMATION_OFFSET = 294
CURRENT_COUNT = 64
MIN_SEGMENT = 63
SPLITS = tuple(range(63, 190))
MIDPOINT = 126
STD_FLOOR = 1e-12
SSE_TIE_RELATIVE = 1e-12
PENALTY = 2.0 * math.log(RETURNS)


def _empty(reason):
    return dict(baseline=float('nan'), candidate=float('nan'), control=float('nan'),
                baseline_available=False, baseline_reason=reason, available=False, reason=reason,
                recent_volatility=None, endpoint_growth=None, full_mean=None, sse0=None,
                formation_volatility=None, numerically_constant=False, split_tie_tolerance=None, splits=[],
                candidate_fit=None, control_fit=None, constant_mean_estimates=0,
                split_mean_estimates=0, candidate_breaks=0, control_breaks=0)


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


def _mean_sse(values):
    """Two-pass centered fsum, translated before summation for stable drift."""
    origin = values[0]
    shifted = [x - origin for x in values]
    offset = math.fsum(shifted) / len(shifted)
    return origin + offset, math.fsum((x - offset) ** 2 for x in shifted)


def _split(values, k):
    pre, left = _mean_sse(values[:k])
    post, right = _mean_sse(values[k:])
    return dict(k=k, pre_mean=pre, post_mean=post, sse=math.fsum((left, right)))


def _best(splits, sse0=0.):
    minimum = min(row['sse'] for row in splits)
    tolerance = SSE_TIE_RELATIVE*sse0
    return min((row for row in splits if row['sse'] <= minimum+tolerance), key=lambda row: row['k'])


def _fit(split, sse0, full_mean, endpoint_growth, numerically_constant):
    perfect = sse0 > 0. and split['sse'] == 0.
    statistic = (None if numerically_constant else 0. if sse0 == 0. else None if perfect else
                 RETURNS * (math.log(sse0) - math.log(split['sse'])))
    accepted = not numerically_constant and (perfect or statistic > PENALTY)
    mean = split['post_mean'] if accepted else full_mean
    if accepted:
        try:
            growth = math.expm1(RETURNS * mean)
        except OverflowError:
            growth = None
        if growth is not None and not math.isfinite(growth):
            growth = None
    else:
        growth = endpoint_growth
    return dict(split, statistic=statistic, perfect_fit=perfect, accepted=accepted,
                predicted_mean=mean, forecast_growth=growth)


def measure(formation, current):
    """Use 253 prices t-294..t-42 and 64 prices t-63..t, inclusive.

    Baseline exactly copies endpoint growth / current simple-return sample
    standard deviation. Its domain uses only the two formation endpoints.
    Sub-floor formation variation suppresses numerical pseudo-breaks.
    """
    formation = np.asarray(formation, dtype=object)
    current = np.asarray(current, dtype=object)
    if formation.shape != (FORMATION_COUNT,) or current.shape != (CURRENT_COUNT,):
        raise ValueError('Require exactly 253 formation prices and 64 current prices')
    result = _empty('nonnumeric_current_price')
    try:
        recent = np.asarray(current, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return result
    if not (np.isfinite(recent) & (recent > 0.)).all():
        return _empty('invalid_current_price')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        changes = recent[1:] / recent[:-1] - 1.
    if not np.isfinite(changes).all():
        return _empty('nonfinite_current_return')
    with np.errstate(over='ignore', invalid='ignore'):
        volatility = float(np.std(changes, ddof=1))
    if not math.isfinite(volatility) or volatility <= 0.:
        return _empty('zero_current_volatility')
    result = _empty('nonnumeric_endpoint_price')
    result['recent_volatility'] = volatility
    try:
        first, last = float(formation[0]), float(formation[-1])
    except (TypeError, ValueError, OverflowError):
        return result
    if not all(math.isfinite(p) and p > 0. for p in (first, last)):
        result.update(reason='invalid_endpoint_price', baseline_reason='invalid_endpoint_price')
        return result
    growth = last / first - 1.
    baseline = growth / volatility
    if not math.isfinite(growth) or not math.isfinite(baseline):
        result.update(reason='nonfinite_baseline', baseline_reason='nonfinite_baseline')
        return result
    result.update(baseline=baseline, baseline_available=True, baseline_reason='available',
                  endpoint_growth=growth)
    if volatility <= STD_FLOOR:
        result['reason'] = 'volatility_floor'
        return result
    try:
        old = np.asarray(formation, dtype=float)
    except (TypeError, ValueError, OverflowError):
        result['reason'] = 'nonnumeric_formation_price'
        return result
    if not (np.isfinite(old) & (old > 0.)).all():
        result['reason'] = 'invalid_formation_price'
        return result
    changes = [_log_ratio(b, a) for a, b in zip(old[:-1], old[1:])]
    if not all(math.isfinite(x) for x in changes):
        raise ArithmeticError('Finite positive prices produced a nonfinite log return')
    mean, sse0 = _mean_sse(changes)
    splits = [_split(changes, k) for k in SPLITS]
    formation_vol = math.sqrt(sse0 / (RETURNS - 1))
    constant = formation_vol <= STD_FLOOR
    candidate = _fit(splits[0] if constant else _best(splits, sse0), sse0, mean, growth, constant)
    control = _fit(splits[MIDPOINT - MIN_SEGMENT], sse0, mean, growth, constant)
    candidate['selection_rule'] = 'canonical_inactive_k63' if constant else 'minimum_sse_relative_tie_earliest'
    control['selection_rule'] = 'fixed_midpoint'
    result.update(full_mean=mean, sse0=sse0, formation_volatility=formation_vol,
                  numerically_constant=constant, split_tie_tolerance=SSE_TIE_RELATIVE*sse0,
                  splits=splits, candidate_fit=candidate,
                  control_fit=control, constant_mean_estimates=1,
                  split_mean_estimates=len(SPLITS), candidate_breaks=int(candidate['accepted']),
                  control_breaks=int(control['accepted']))
    forecasts = []
    for fit in (candidate, control):
        value = (baseline if not fit['accepted'] else
                 None if fit['forecast_growth'] is None else fit['forecast_growth'] / volatility)
        forecasts.append(value)
    if any(x is None or not math.isfinite(x) for x in forecasts):
        result['reason'] = 'nonfinite_forecast'
        return result
    result.update(candidate=forecasts[0], control=forecasts[1], available=True, reason='available')
    return result


def _day(value, name):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(name + ' must be a naive midnight date')
    return day


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
        raise ValueError('Calendar is empty')
    return pd.DatetimeIndex(dates)


def _queries(meta, sessions):
    if not isinstance(meta, dict) or not {'dates', 'symbols'} <= set(meta):
        raise ValueError('Metadata requires dates and symbols')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata event axes differ')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], None
    for date, names in zip(meta['dates'], meta['symbols']):
        day = _day(date, 'Event')
        if day not in lookup or (queries and day <= queries[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(names, (str, bytes)):
            raise TypeError('Names must be an explicit sequence')
        names = list(names)
        if (not names or len(set(names)) != len(names) or
                any(not isinstance(n, str) or not n or n.strip() != n for n in names)):
            raise ValueError('Names must be unique nonempty IDs')
        if width is not None and len(names) != width:
            raise ValueError('Event widths differ')
        width = len(names)
        queries.append((day, names, lookup[day]))
    return queries, width or 0


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adj must be a DataFrame with DatetimeIndex')
    index = frame.index
    if (index.hasnans or not index.is_unique or not index.is_monotonic_increasing or
            index.tz is not None or not index.equals(index.normalize())):
        raise ValueError('Adj dates must be unique increasing naive midnight sessions')
    if (not frame.columns.is_unique or
            any(not isinstance(n, str) or not n or n.strip() != n for n in frame.columns)):
        raise ValueError('Adj names must be unique nonempty IDs')


def build_features(panel, meta, calendar, limit=None):
    """Return raw signals, common new-support mask, and dated fit evidence.

    Slice before numeric conversion; only the specified prefix is measured.
    No global panel conversion can touch poisoned/future values.
    """
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    adj = panel['adj']
    _frame(adj)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be a valid integer prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = {k: np.full(shape, np.nan) for k in ('baseline', 'candidate', 'control')}
    values['available'] = np.zeros(shape, bool)
    history, reasons = [], {}
    totals = dict(constant_mean_estimates=0, split_mean_estimates=0, candidate_breaks=0, control_breaks=0)
    for i, (day, names, t) in enumerate(queries):
        enough = t >= FORMATION_OFFSET
        formation_dates = sessions[t-FORMATION_OFFSET:t-SKIP+1] if enough else None
        current_dates = sessions[t-CURRENT_COUNT+1:t+1] if t >= CURRENT_COUNT-1 else None
        missing_formation = ([] if not enough else
                             formation_dates.difference(adj.index).strftime('%Y-%m-%d').tolist())
        missing_current = ([] if current_dates is None else
                           current_dates.difference(adj.index).strftime('%Y-%m-%d').tolist())
        endpoints_missing = (False if not enough else
                             formation_dates[0] not in adj.index or formation_dates[-1] not in adj.index)
        cells = []
        for j, symbol in enumerate(names):
            if not enough:
                detail = _empty('insufficient_history')
            elif symbol not in adj.columns:
                detail = _empty('missing_symbol')
            elif missing_current:
                detail = _empty('missing_volatility_session')
            elif endpoints_missing:
                detail = _empty('missing_endpoint_session')
            else:
                detail = measure(adj[symbol].reindex(formation_dates).to_numpy(copy=True),
                                 adj.loc[current_dates, symbol].to_numpy(copy=True))
                if missing_formation and detail['baseline_available']:
                    detail['reason'] = 'missing_formation_session'
                for fit in detail['splits'] + [detail['candidate_fit'], detail['control_fit']]:
                    if fit is not None:
                        fit['break_after_date'] = str(formation_dates[fit['k']].date())
                        fit['post_first_return_date'] = str(formation_dates[fit['k']+1].date())
            for key in ('baseline', 'candidate', 'control'):
                values[key][i, j] = detail[key]
            values['available'][i, j] = detail['available']
            for key in totals:
                totals[key] += detail[key]
            reasons[detail['reason']] = reasons.get(detail['reason'], 0) + 1
            cell = dict(detail, symbol=symbol)
            for key in ('baseline', 'candidate', 'control'):
                if not math.isfinite(cell[key]):
                    cell[key] = None
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
                            formation_start=str(formation_dates[0].date()) if enough else None,
                            formation_end=str(formation_dates[-1].date()) if enough else None,
                            volatility_start=str(current_dates[0].date()) if current_dates is not None else None,
                            volatility_end=str(current_dates[-1].date()) if current_dates is not None else None,
                            missing_formation_sessions=missing_formation,
                            missing_volatility_sessions=missing_current, cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=len(queries), symbols_per_event=width, skip_sessions=SKIP,
                 formation_price_count=FORMATION_COUNT, formation_return_count=RETURNS,
                 current_price_count=CURRENT_COUNT, current_return_count=CURRENT_COUNT-1,
                 minimum_segment=MIN_SEGMENT, split_candidates=len(SPLITS), split_min=SPLITS[0],
                 split_max=SPLITS[-1], midpoint=MIDPOINT, penalty=PENALTY,
                 volatility_floor=STD_FLOOR, formation_standard_deviation_floor=STD_FLOOR,
                 split_sse_relative_tolerance=SSE_TIE_RELATIVE,
                 baseline_finite_cells=int(np.isfinite(values['baseline']).sum()),
                 available_cells=available, unavailable_cells=values['available'].size-available,
                 reason_counts=reasons, event_history=history, predictive_model_fits=0,
                 common_availability=True, portfolio_returns_calculated=False,
                 ranking_performed=False, trading_eligibility_modified=False, **totals)
    return values, audit
