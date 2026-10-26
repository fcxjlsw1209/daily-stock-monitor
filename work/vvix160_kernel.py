"""Pure, fixed round160 VIX/VVIX state and score-delta arithmetic.

There is no file I/O, source admission, price return, or selection here. The
caller must pin the source/calendar/META/original vol126 array and register the
experiment before real-data use. Values are official daily close observations
keyed by strict ISO session dates. Only the required historical values are read.
"""
from collections.abc import Mapping
from datetime import date
from numbers import Real

import numpy as np


INITIAL = '2016-08-17'
WINDOW = 252
WIDTH = 49
SOURCES = ('VIX', 'VVIX')
ARMS = ('incumbent', 'vix_control', 'vvix_challenge')
EXTRA = ('delta_vix', 'delta_vvix')
COEFFICIENT = -0.02


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value):
    require(isinstance(value, str), 'Strict ISO date string required')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError('Strict ISO date string required: ' + repr(value)) from error
    require(parsed.isoformat() == value, 'Strict ISO date string required: ' + value)
    return parsed


def _ordered_dates(values, label):
    require(isinstance(values, (list, tuple)), label + ' must be an ordered list or tuple')
    result = [_day(value) for value in values]
    require(all(a < b for a, b in zip(result, result[1:])),
            label + ' must be strictly ascending and unique')
    return result


def _limit(limit, length):
    if limit is None:
        return length
    require(type(limit) is int and 0 <= limit <= length,
            'limit must be an integer prefix length in [0, N]')
    return limit


def _observation(value, source, session):
    # Do not coerce strings, booleans or opaque objects into financial values.
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
            source + ' nonnumeric close at ' + session)
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(source + ' invalid close at ' + session) from error
    require(np.isfinite(result) and result > 0,
            source + ' close must be finite and positive at ' + session)
    return result


def _latest_midrank(values):
    """One-based ascending midrank; no percentile API or time reweighting."""
    latest = values[-1]
    lower = sum(value < latest for value in values)
    equal = sum(value == latest for value in values)
    rank = 1.0 + lower + (equal - 1) / 2.0
    # Algebraically 2*(rank-1)/251-1. Integer numerator gives exact zero for
    # an all-equal 252-session window, without a numerical tolerance branch.
    state = (2 * lower + equal - WINDOW) / (WINDOW - 1)
    require(-1.0 <= state <= 1.0, 'State outside fixed bounds')
    return rank, state


def build_states(meta_dates, calendar_sessions, values_by_source, limit=None):
    """Return two state vectors, a common mask and one audit row per prefix.

    ``meta_dates`` starts with the original initial partial month, 2016-08-17.
    The ordered calendar is supplied by the caller, never synthesized here.
    Each noninitial signal must itself be a session and consumes exactly
    calendar[t-252:t], ending immediately before the signal. Both sources must
    have all 252 finite positive numeric values; failure raises, not neutralizes.
    Source mappings may contain later or noncalendar data, which is not read.

    ``limit`` is a prefix length (including initial), 0..len(meta_dates). Static
    date axes are validated in full; financial values only for that prefix.
    Source numeric inputs are real Python/NumPy scalars, not decimal strings.
    """
    dates = _ordered_dates(meta_dates, 'META dates')
    calendar = _ordered_dates(calendar_sessions, 'Calendar sessions')
    require(bool(dates) and meta_dates[0] == INITIAL, 'Original initial partial date required')
    require(bool(calendar), 'Calendar sessions cannot be empty')
    require(all(day.weekday() < 5 for day in calendar), 'Weekend in supplied session calendar')
    require(isinstance(values_by_source, Mapping) and set(values_by_source) == set(SOURCES),
            'Exactly VIX and VVIX source mappings required')
    require(all(isinstance(values_by_source[name], Mapping) for name in SOURCES),
            'Each source must be a date-to-close mapping')
    count = _limit(limit, len(dates))
    positions = {day: i for i, day in enumerate(calendar)}
    require(all(day in positions for day in dates[:count]), 'Signal missing from calendar')
    states = {name: np.zeros(count, dtype=np.float64) for name in SOURCES}
    common = np.zeros(count, dtype=np.bool_)
    monthly = []
    for index in range(count):
        row = dict(signal_index=index, date=meta_dates[index], initial=index == 0,
                   initial_partial=index == 0, window_start=None, window_end=None,
                   window_size=0)
        if index == 0:
            for name in SOURCES:
                row[name] = dict(passed=True, observations=0, latest_rank=None, m=0.0)
            monthly.append(row)
            continue
        end = positions[dates[index]]
        require(end >= WINDOW, 'Insufficient calendar warmup at ' + meta_dates[index])
        window = calendar_sessions[end - WINDOW:end]
        row.update(window_start=window[0], window_end=window[-1], window_size=WINDOW)
        for name in SOURCES:
            source = values_by_source[name]
            observations = []
            for session in window:
                require(session in source, name + ' missing close at ' + session)
                observations.append(_observation(source[session], name, session))
            rank, state = _latest_midrank(observations)
            states[name][index] = state
            row[name] = dict(passed=True, observations=WINDOW, latest_rank=rank, m=state)
        # Set only after *both* windows succeed: there is no unilateral fallback.
        common[index] = True
        monthly.append(row)
    return dict(states=states, common_available=common, audit=dict(monthly=monthly))


def build_deltas(states, vol126, limit=None):
    """Apply already-scaled deltas to the original centered volatility input.

    Returns only ``delta_vix`` and ``delta_vvix``, each (prefix,49), bounded by
    +/-0.01. Incumbent delta is implicitly zero. The caller adds each delta
    exactly once to incumbent pre-score, without any further dose/centering.
    No eligibility mask or cross-sectional ranking is accepted or performed.

    ``vol126`` is a two-dimensional real numeric NumPy array with at least the
    requested prefix and exactly 49 original columns. Only its prefix is
    validated/read numerically. Consumed rows, including initial, must be
    finite and in [-0.5,0.5]; initial output is nevertheless exactly zero.
    """
    require(isinstance(states, Mapping) and isinstance(states.get('states'), Mapping),
            'build_states result required')
    vectors = states['states']
    require(set(vectors) == set(SOURCES), 'Both state vectors required')
    require(all(isinstance(vectors[name], np.ndarray) and vectors[name].ndim == 1
                and vectors[name].dtype.kind in 'fiu' for name in SOURCES),
            'Real numeric one-dimensional state arrays required')
    length = len(vectors['VIX'])
    require(len(vectors['VVIX']) == length, 'State vector length mismatch')
    common = states.get('common_available')
    require(isinstance(common, np.ndarray) and common.dtype.kind == 'b'
            and common.shape == (length,), 'Common boolean availability vector required')
    count = _limit(limit, length)
    expected = np.ones(count, dtype=np.bool_)
    if count:
        expected[0] = False
    require(np.array_equal(common[:count], expected), 'Both sources must have common complete windows')
    require(isinstance(vol126, np.ndarray) and vol126.ndim == 2
            and vol126.shape[1] == WIDTH and vol126.shape[0] >= count
            and vol126.dtype.kind in 'fiu', 'Original numeric vol126 array must have 49 columns')
    vol = vol126[:count]
    require(np.isfinite(vol).all() and ((vol >= -0.5) & (vol <= 0.5)).all(),
            'Consumed original vol126 must be finite and within [-0.5,0.5]')
    result = {}
    for name, field in zip(SOURCES, EXTRA):
        values = vectors[name][:count]
        require(np.isfinite(values).all() and ((values >= -1.0) & (values <= 1.0)).all(),
                name + ' state outside finite [-1,1] bounds')
        require(not count or values[0] == 0.0, 'Initial state must be exactly neutral')
        result[field] = COEFFICIENT * values[:, None] * vol
        if count:
            result[field][0, :] = 0.0
        require((np.abs(result[field]) <= 0.01).all(), 'Delta outside fixed +/-0.01 bounds')
    return result
