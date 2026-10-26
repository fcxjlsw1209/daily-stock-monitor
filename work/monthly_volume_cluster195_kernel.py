"""Pure unsigned stored-volume event clustering; no file, model or portfolio I/O.

Each 273-session window supplies 210 events after a strictly prior 63-session
median.  Constant positive volumes are valid and produce zero frequency/covariance.
"""
from collections import Counter
import numbers

import numpy as np
import pandas as pd

VOLUME_COUNT = 273
EVENT_COUNT = 210
PAIR_COUNT = 209
MEDIAN_WINDOW = 63
SKIP = 42
INITIAL_PARTIAL = '2016-08-17'


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'),
        control=float('nan'), median_values=[], high_volume_events=[],
        event_ones=None, left_ones=None, right_ones=None,
        adjacent_one_pairs=None, covariance_numerator=None,
        event_observations=0, adjacent_pair_observations=0,
        rolling_medians_computed=0)


def measure(volume):
    """Return shared-support raw covariance and frequency from exactly273 values."""
    if np.shape(volume) != (VOLUME_COUNT,):
        raise ValueError('Expected exactly273 stored volume observations')
    raw = np.asarray(volume, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return _empty('nonnumeric_volume')
    try:
        source = np.asarray(raw, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        return _empty('nonnumeric_volume')
    if not (np.isfinite(source) & (source > 0)).all():
        return _empty('invalid_volume')
    medians, events = [], []
    for end in range(MEDIAN_WINDOW, VOLUME_COUNT):
        median = float(np.partition(source[end-MEDIAN_WINDOW:end], MEDIAN_WINDOW//2)[MEDIAN_WINDOW//2])
        medians.append(median)
        events.append(bool(source[end] > median))
    total = sum(events)
    left, right = sum(events[:-1]), sum(events[1:])
    adjacent = sum(a and b for a, b in zip(events[:-1], events[1:]))
    numerator = PAIR_COUNT * adjacent - left * right
    return dict(available=True, reason='available', candidate=numerator/(PAIR_COUNT*PAIR_COUNT),
        control=total/EVENT_COUNT, median_values=medians, high_volume_events=events,
        event_ones=total, left_ones=left, right_ones=right,
        adjacent_one_pairs=adjacent, covariance_numerator=numerator,
        event_observations=EVENT_COUNT, adjacent_pair_observations=PAIR_COUNT,
        rolling_medians_computed=EVENT_COUNT)


def centered_rank(raw, available):
    """Higher is preferred: average tied zero-based rank/(n-1)-0.5."""
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.zeros(raw.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) < 2:
        return result
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
        raise TypeError('Volume requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Volume sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                         for s in frame.columns)):
        raise ValueError('Volume columns must be unique nonempty security IDs')


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
    """Return five aligned arrays and a complete JSON-safe cell/event audit.

    Only requested windows are converted or inspected numerically. Metadata axes
    are validated as a whole. The supplied official calendar remains authoritative.
    """
    if not isinstance(panel, dict) or set(panel) != {'volume'}:
        raise ValueError('Panel requires exactly volume')
    volume = panel['volume']
    _frame(volume)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
        available=np.zeros(shape, dtype=bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons = [], Counter()
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        dates = sessions[t-(VOLUME_COUNT-1)-SKIP:t-SKIP+1] if t >= VOLUME_COUNT-1+SKIP else None
        missing = dates.difference(volume.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells = []
        for j, symbol in enumerate(symbols):
            if dates is None:
                cell = _empty('insufficient_history')
            elif symbol not in volume.columns:
                cell = _empty('missing_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(volume.loc[dates, symbol].to_numpy(copy=True))
            for key in ('candidate', 'control', 'available'):
                values[key][i, j] = cell[key]
            reasons[cell['reason']] += 1
            cell = dict(cell, symbol=symbol, date=iso)
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        for arm in ('candidate', 'control'):
            values[arm+'_rank'][i] = centered_rank(values[arm][i], values['available'][i])
        for j, cell in enumerate(cells):
            cell.update(candidate_rank=float(values['candidate_rank'][i, j]),
                        control_rank=float(values['control_rank'][i, j]))
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            volume_start=str(dates[0].date()) if dates is not None else None,
            volume_end=str(dates[-1].date()) if dates is not None else None,
            event_start=str(dates[MEDIAN_WINDOW].date()) if dates is not None else None,
            event_end=str(dates[-1].date()) if dates is not None else None,
            missing_official_sessions=missing, cells=cells))
    available = int(values['available'].sum())
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        volume_count=VOLUME_COUNT, skip_sessions=SKIP, event_count=EVENT_COUNT, pair_count=PAIR_COUNT,
        median_window=MEDIAN_WINDOW, available_cells=available,
        unavailable_cells=int(values['available'].size)-available, reason_counts=dict(reasons),
        event_observations=available*EVENT_COUNT, adjacent_pair_observations=available*PAIR_COUNT,
        rolling_medians_computed=available*EVENT_COUNT, predictive_model_fits=0, new_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False, event_history=history)
