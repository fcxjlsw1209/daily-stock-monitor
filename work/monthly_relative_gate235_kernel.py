"""Pure235 weak-state same-window SPY-relative entry qualification.

Only required endpoint prices are consumed. Original eligibility is a source
input, never reconstructed or mutated. No ranking, selection, fitting or replay.
"""
import math
from numbers import Real

import numpy as np
import pandas as pd

INITIAL_PARTIAL = '2016-08-17'
START_OFFSET, END_OFFSET, RETURN_COUNT = 252, 42, 210
VALUE_KEYS = ('original_eligible', 'candidate_eligible', 'required', 'momentum',
              'spy_momentum', 'threshold')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells',
    'original_eligible_cells', 'weak_events', 'active_events', 'required_cells',
    'source_price_observations', 'spy_price_observations', 'rejected_cells',
    'candidate_eligible_cells', 'new_model_fits', 'predictive_model_fits')


def _day(value, label):
    if not isinstance(value, str):
        raise ValueError(label + ' requires canonical ISO date strings')
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' is not a date') from exc
    if (pd.isna(day) or day.tzinfo is not None or day != day.normalize()
            or value != str(day.date())):
        raise ValueError(label + ' requires canonical naive ISO dates')
    return day


def _calendar(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('Calendar must be an official-session sequence')
    dates = []
    for value in values:
        day = _day(value, 'Calendar')
        if day.weekday() >= 5 or (dates and day <= dates[-1]):
            raise ValueError('Calendar must contain unique increasing weekdays')
        dates.append(day)
    return pd.DatetimeIndex(dates)


def _frame(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel must contain exactly adj')
    frame = panel['adj']
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('adj requires a DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Price dates must be unique increasing naive midnights')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s
            or s.strip() != s for s in frame.columns)):
        raise ValueError('Price columns must be unique nonempty security IDs')
    return frame


def _booleans(raw, n, label):
    # Slice first: a future object or numeric poison must not be consumed.
    prefix = raw[:n]
    if any(not isinstance(v, (bool, np.bool_)) for v in prefix.flat):
        raise ValueError(label + ' requires exact boolean values')
    return np.array(prefix, dtype=bool, copy=True)


def _inputs(arrays, meta, sessions, limit):
    if not isinstance(arrays, dict) or not {'eligible', 'state_weak_narrow'} <= set(arrays):
        raise ValueError('Original eligible and state_weak_narrow arrays required')
    eligible = np.asarray(arrays['eligible'])
    weak = np.asarray(arrays['state_weak_narrow'])
    if eligible.ndim != 2 or eligible.shape[1] <= 0:
        raise ValueError('Original eligible must have event and nonempty security axes')
    total, width = eligible.shape
    if weak.shape != (total,):
        raise ValueError('Original weak state must have exactly one event axis')
    if limit is not None and (type(limit) is not int or not 0 <= limit <= total):
        raise ValueError('limit must be an integer event prefix')
    n = total if limit is None else limit
    if (not isinstance(meta, dict) or any(not isinstance(meta.get(k), list)
            or len(meta[k]) != total for k in ('dates', 'symbols'))):
        raise ValueError('Metadata dates and symbols must match original event axis')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries = []
    for i in range(n):
        day = _day(meta['dates'][i], 'Event')
        if day not in lookup or (queries and day <= queries[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        partial = i == 0 and str(day.date()) == INITIAL_PARTIAL
        if not partial and (t + 1 == len(sessions)
                or sessions[t + 1].to_period('M') == day.to_period('M')):
            raise ValueError('Regular query requires a witnessed official month-end')
        names = meta['symbols'][i]
        if (not isinstance(names, list) or len(names) != width
                or any(not isinstance(s, str) or not s or s.strip() != s for s in names)
                or len(set(names)) != width):
            raise ValueError('Event symbols must be unique IDs matching original width')
        queries.append((day, list(names), t, partial))
    return _booleans(eligible, n, 'eligible'), _booleans(weak, n, 'state_weak_narrow'), queries, width


def _price(value, symbol, day):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('Required price must be real and nonboolean: ' + symbol + '/' + str(day.date()))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid required price: ' + symbol + '/' + str(day.date())) from exc
    if not math.isfinite(number) or number <= 0.:
        raise ValueError('Required price must be finite and positive: ' + symbol + '/' + str(day.date()))
    return number


def _momentum(frame, symbol, start, end):
    if symbol not in frame.columns:
        raise ValueError('Missing required security column: ' + symbol)
    first = _price(frame.at[start, symbol], symbol, start)
    last = _price(frame.at[end, symbol], symbol, end)
    try:
        result = last / first - 1.
    except (OverflowError, ZeroDivisionError) as exc:
        raise FloatingPointError('Nonfinite derived momentum: ' + symbol) from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite derived momentum: ' + symbol)
    return result


def build_features(panel, meta, calendar, arrays, limit=None):
    """Return original/new E, source-required mask and sparse endpoint evidence.

    Required securities are original-E members on regular weak rows only.
    A regular weak row always requires SPY, even with no eligible securities.
    Inactive, noneligible and initial-partial numeric evidence remains NaN.
    """
    frame = _frame(panel)
    sessions = _calendar(calendar)
    eligible, weak, queries, width = _inputs(arrays, meta, sessions, limit)
    n = len(queries)
    values = dict(original_eligible=eligible, candidate_eligible=eligible.copy(),
        required=np.zeros((n, width), dtype=bool), momentum=np.full((n, width), np.nan),
        spy_momentum=np.full(n, np.nan), threshold=np.full(n, np.nan))
    history, active_events = [], 0
    for i, (day, names, t, partial) in enumerate(queries):
        active = bool(weak[i]) and not partial
        branch = 'initial_partial' if partial else 'state_inactive' if not active else 'weak_relative_gate'
        original_ids = np.flatnonzero(eligible[i]).tolist()
        required_ids, rejected_ids, positions = [], [], []
        start_date = end_date = spy = threshold = None
        if active:
            if t < START_OFFSET:
                raise ValueError('Required formation has insufficient official history: ' + str(day.date()))
            positions = [t - START_OFFSET, t - END_OFFSET]
            start, end = sessions[positions[0]], sessions[positions[1]]
            if start not in frame.index or end not in frame.index:
                raise ValueError('Missing required official endpoint: ' + str(day.date()))
            start_date, end_date = str(start.date()), str(end.date())
            spy = _momentum(frame, 'SPY', start, end)
            threshold = max(0., spy)
            values['spy_momentum'][i], values['threshold'][i] = spy, threshold
            required_ids = list(original_ids)
            for j in required_ids:
                momentum = _momentum(frame, names[j], start, end)
                if momentum <= 0.:
                    raise ValueError('Original eligible/source positive-momentum mismatch: ' + names[j] + '/' + str(day.date()))
                values['required'][i, j] = True
                values['momentum'][i, j] = momentum
                if not momentum > threshold:
                    values['candidate_eligible'][i, j] = False
                    rejected_ids.append(j)
            active_events += 1
        history.append(dict(query_index=i, date=str(day.date()), symbols=names,
            session_position=t, weak_narrow=bool(weak[i]), initial_partial=partial,
            active=active, branch=branch, source_start=start_date, source_end=end_date,
            source_positions=positions, original_eligible_indices=original_ids,
            required_indices=required_ids, rejected_indices=rejected_ids,
            candidate_eligible_indices=np.flatnonzero(values['candidate_eligible'][i]).tolist(),
            spy_momentum=spy, threshold=threshold))
    original_count = int(eligible.sum())
    candidate_count = int(values['candidate_eligible'].sum())
    required_count = int(values['required'].sum())
    audit = dict(schema='monthly_relative_gate235_features_v1', events=n,
        symbols_per_event=width, checked_cells=n * width,
        original_eligible_cells=original_count, weak_events=int(weak.sum()),
        active_events=active_events, required_cells=required_count,
        source_price_observations=2 * required_count, spy_price_observations=2 * active_events,
        rejected_cells=original_count - candidate_count, candidate_eligible_cells=candidate_count,
        new_model_fits=0, predictive_model_fits=0, start_offset=START_OFFSET,
        end_offset=END_OFFSET, return_count=RETURN_COUNT, rankings_recomputed=False,
        source_missing_filled=False, portfolio_returns_calculated=False, event_history=history)
    return values, audit


build = build_features
