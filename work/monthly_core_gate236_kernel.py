"""Pure236 full core-horizon entry replacement on noncalm month ends.

Reconstruct original warm295/positive210 eligibility before replacing its sign
horizon with252. All source rows are original, never reindexed or filled. No
ranking, selection, model fitting or portfolio replay is performed.
"""
import math
from numbers import Real

import numpy as np
import pandas as pd

INITIAL_PARTIAL = '2016-08-17'
WARM_PRICE_COUNT = 295
OLD_START_OFFSET, NEW_START_OFFSET, END_OFFSET = 252, 294, 42
OLD_RETURN_COUNT, NEW_RETURN_COUNT = 210, 252
VALUE_KEYS = ('original_eligible','candidate_eligible','required','warm','momentum210','momentum252')
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','original_eligible_cells',
    'calm_events','active_events','required_cells','warm_cells','warm_observations',
    'source_price_observations','added_cells','rejected_cells','candidate_eligible_cells',
    'new_model_fits','predictive_model_fits')


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
    if not isinstance(arrays, dict) or not {'eligible', 'state_calm10'} <= set(arrays):
        raise ValueError('Original eligible and state_calm10 arrays required')
    eligible = np.asarray(arrays['eligible'])
    calm = np.asarray(arrays['state_calm10'])
    if eligible.ndim != 2 or eligible.shape[1] <= 0:
        raise ValueError('Original eligible must have event and nonempty security axes')
    total, width = eligible.shape
    if calm.shape != (total,):
        raise ValueError('Original calm state must have exactly one event axis')
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
    return _booleans(eligible, n, 'eligible'), _booleans(calm, n, 'state_calm10'), queries, width


def _current(value, symbol, day):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('Required current must be real and nonboolean: ' + symbol + '/' + str(day.date()))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid required current: ' + symbol + '/' + str(day.date())) from exc
    if not math.isfinite(result):
        raise ValueError('Required current must be finite: ' + symbol + '/' + str(day.date()))
    return result


def _positive(value, symbol, day):
    result = _current(value, symbol, day)
    if result <= 0.:
        raise ValueError('Required lag endpoint must be positive: ' + symbol + '/' + str(day.date()))
    return result


def _return(first, last, symbol):
    try:
        result = last / first - 1.
    except (OverflowError, ZeroDivisionError) as exc:
        raise FloatingPointError('Nonfinite derived momentum: ' + symbol) from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite derived momentum: ' + symbol)
    return result


def build_features(panel, meta, calendar, arrays, limit=None):
    """Return full replacement E and source evidence over all noncalm names.

    NaN anywhere in the exact295 source window is a legitimate warm failure.
    Only a fully nonmissing window consumes current/lag endpoint numerics;
    interior nonendpoint values retain the inherited notna-only semantics.
    """
    frame = _frame(panel)
    sessions = _calendar(calendar)
    eligible, calm, queries, width = _inputs(arrays, meta, sessions, limit)
    n = len(queries)
    values = dict(original_eligible=eligible, candidate_eligible=eligible.copy(),
        required=np.zeros((n, width), dtype=bool), warm=np.zeros((n, width), dtype=bool),
        momentum210=np.full((n, width), np.nan), momentum252=np.full((n, width), np.nan))
    history, active_events = [], 0
    for i, (day, names, t, partial) in enumerate(queries):
        active = not bool(calm[i]) and not partial
        branch = 'initial_partial' if partial else 'calm_bypass' if not active else 'core_horizon_gate'
        positions, source_dates = [], []
        if active:
            if t < NEW_START_OFFSET:
                raise ValueError('Required warm window has insufficient official history: ' + str(day.date()))
            official = sessions[t - NEW_START_OFFSET:t + 1]
            window = frame.loc[official[0]:day]
            if len(official) != WARM_PRICE_COUNT or not window.index.equals(official):
                raise ValueError('Original price window must equal all295 official sessions: ' + str(day.date()))
            # Equality above also makes these exactly the last295 original rows
            # through the query. No calendar reindexing or inferred price rows.
            if not frame.loc[:day].tail(WARM_PRICE_COUNT).index.equals(official):
                raise ValueError('Original295-row warm window differs from official sessions')
            for symbol in names:
                if symbol not in frame.columns:
                    raise ValueError('Missing required security column: ' + symbol)
            positions = [t - NEW_START_OFFSET, t - OLD_START_OFFSET, t - END_OFFSET, t]
            dates = [sessions[j] for j in positions]
            source_dates = [str(d.date()) for d in dates]
            values['required'][i] = True
            values['candidate_eligible'][i] = False
            for j, symbol in enumerate(names):
                not_missing = bool(window[symbol].notna().all())
                warm = False
                if not_missing:
                    current = _current(window.at[day, symbol], symbol, day)
                    warm = current > 0.
                if not warm:
                    if bool(eligible[i, j]):
                        raise ValueError('Original eligible/warm source mismatch: ' + symbol + '/' + str(day.date()))
                    continue
                first252 = _positive(window.at[dates[0], symbol], symbol, dates[0])
                first210 = _positive(window.at[dates[1], symbol], symbol, dates[1])
                end = _positive(window.at[dates[2], symbol], symbol, dates[2])
                momentum210 = _return(first210, end, symbol)
                momentum252 = _return(first252, end, symbol)
                if bool(eligible[i, j]) != (momentum210 > 0.):
                    raise ValueError('Original eligible/positive210 source mismatch: ' + symbol + '/' + str(day.date()))
                values['warm'][i, j] = True
                values['momentum210'][i, j] = momentum210
                values['momentum252'][i, j] = momentum252
                values['candidate_eligible'][i, j] = momentum252 > 0.
            active_events += 1
        old = eligible[i]
        new = values['candidate_eligible'][i]
        history.append(dict(query_index=i, date=str(day.date()), symbols=names,
            session_position=t, calm10=bool(calm[i]), initial_partial=partial, active=active,
            branch=branch, source_positions=positions, source_dates=source_dates,
            original_eligible_indices=np.flatnonzero(old).tolist(),
            candidate_eligible_indices=np.flatnonzero(new).tolist(),
            required_indices=np.flatnonzero(values['required'][i]).tolist(),
            warm_indices=np.flatnonzero(values['warm'][i]).tolist(),
            added_indices=np.flatnonzero(new & ~old).tolist(),
            rejected_indices=np.flatnonzero(old & ~new).tolist()))
    original_count = int(eligible.sum())
    required_count = int(values['required'].sum())
    warm_count = int(values['warm'].sum())
    audit = dict(schema='monthly_core_gate236_features_v1',events=n,symbols_per_event=width,
        checked_cells=n*width,original_eligible_cells=original_count,calm_events=int(calm.sum()),
        active_events=active_events,required_cells=required_count,warm_cells=warm_count,
        warm_observations=WARM_PRICE_COUNT*required_count,source_price_observations=4*warm_count,
        added_cells=int((values['candidate_eligible'] & ~eligible).sum()),
        rejected_cells=int((eligible & ~values['candidate_eligible']).sum()),
        candidate_eligible_cells=int(values['candidate_eligible'].sum()),new_model_fits=0,predictive_model_fits=0,
        warm_price_count=WARM_PRICE_COUNT,old_start_offset=OLD_START_OFFSET,new_start_offset=NEW_START_OFFSET,
        end_offset=END_OFFSET,old_return_count=OLD_RETURN_COUNT,new_return_count=NEW_RETURN_COUNT,
        rankings_recomputed=False,source_missing_filled=False,portfolio_returns_calculated=False,event_history=history)
    return values, audit


build = build_features
