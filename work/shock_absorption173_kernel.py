"""Pure historical down-shock absorption measurement; no I/O or portfolios.

Sources are the caller's frozen price vintage. Stored close times volume is
activity, not observed investor flows or certified historical share turnover.
The label is a matured historical price change, not a forecast or trade fill.
"""
import math

import numpy as np
import pandas as pd


SKIP = 42
HISTORY = 1260
HORIZON = 21
COOLDOWN = 21
BASELINE = 20
PRIOR_MASS = 20
MIN_GROUP = 3
ADJUSTED_COUNT = HISTORY + 1
LAST_ANCHOR = HISTORY - HORIZON
ACTIVITY_COUNT = LAST_ANCHOR + BASELINE
ADJUSTED_START_OFFSET = HISTORY + SKIP
ACTIVITY_START_OFFSET = ADJUSTED_START_OFFSET + BASELINE - 1
ACTIVITY_END_OFFSET = SKIP + HORIZON


def _empty(reason):
    return dict(candidate=float('nan'), control=float('nan'), available=False,
                reason=reason, selected_anchors=[], high_anchors=[], low_anchors=[],
                invalid_activity_anchors=[], n_all=0, n_high=0, n_low=0,
                labels=[], classifications=[])


def _activity_units(close, volume):
    """Exact dyadic products in one integer unit, with no float overflow.

    Every finite IEEE float is a rational with a power-of-two denominator.
    Common integer scaling preserves strict comparison to the arithmetic
    previous-20 mean, even at equality or a one-ULP difference.
    """
    products = []
    greatest_denominator_power = 0
    for c, v in zip(close, volume):
        cn, cd = float(c).as_integer_ratio()
        vn, vd = float(v).as_integer_ratio()
        power = cd.bit_length() + vd.bit_length() - 2
        products.append((cn * vn, power))
        greatest_denominator_power = max(greatest_denominator_power, power)
    return [numerator << (greatest_denominator_power - power)
            for numerator, power in products]


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


def measure(adjusted, close, volume):
    """Measure 1261 prices and 1259 activity observations on exact axes.

    Price index 0 is t-1302; activity index 0 is t-1321. An anchor at
    price index s uses activity index s+19 and label price index s+21.
    Sampling uses only negative price days and a common 21-session cooldown;
    activity and label values cannot move or remove a sampled anchor.
    """
    arrays = []
    for value, label in ((adjusted, 'adjusted_price'), (close, 'stored_close'),
                         (volume, 'stored_volume')):
        try:
            arrays.append(np.asarray(value, dtype=float))
        except (TypeError, ValueError, OverflowError):
            return _empty('nonnumeric_' + label)
    adjusted, close, volume = arrays
    if (adjusted.shape != (ADJUSTED_COUNT,) or close.shape != (ACTIVITY_COUNT,)
            or volume.shape != (ACTIVITY_COUNT,)):
        raise ValueError('Expected exactly 1261 adjusted prices and 1259 activity rows')
    for value, label, positive in ((adjusted, 'adjusted_price', True),
                                   (close, 'stored_close', True),
                                   (volume, 'stored_volume', False)):
        valid = np.isfinite(value) & ((value > 0.) if positive else (value >= 0.))
        if not valid.all():
            return _empty('invalid_' + label)
    result = _empty('insufficient_groups')
    next_allowed = 1
    for s in range(1, LAST_ANCHOR + 1):
        if s >= next_allowed and adjusted[s] < adjusted[s-1]:
            result['selected_anchors'].append(s)
            next_allowed = s + COOLDOWN
    units = _activity_units(close, volume)
    sums = [0]
    for unit in units:
        sums.append(sums[-1] + unit)
    high_labels, low_labels = [], []
    for s in result['selected_anchors']:
        label = _log_ratio(adjusted[s + HORIZON], adjusted[s])
        if not math.isfinite(label):
            raise ArithmeticError('Positive finite prices produced a nonfinite log label')
        result['labels'].append(label)
        j = s + BASELINE - 1
        prior = sums[j] - sums[j - BASELINE]
        if prior == 0:
            result['classifications'].append(None)
            result['invalid_activity_anchors'].append(s)
        elif BASELINE * units[j] > prior:
            result['classifications'].append('H')
            result['high_anchors'].append(s)
            high_labels.append(label)
        else:
            result['classifications'].append('L')
            result['low_anchors'].append(s)
            low_labels.append(label)
    result.update(n_all=len(result['selected_anchors']),
                  n_high=len(high_labels), n_low=len(low_labels))
    if result['invalid_activity_anchors']:
        result['reason'] = 'zero_prior_activity'
        return result
    if min(result['n_high'], result['n_low']) < MIN_GROUP:
        return result
    q_high = math.fsum(high_labels) / (result['n_high'] + PRIOR_MASS)
    q_low = math.fsum(low_labels) / (result['n_low'] + PRIOR_MASS)
    result.update(candidate=q_high - q_low,
                  control=math.fsum(result['labels']) / (result['n_all'] + PRIOR_MASS),
                  available=True, reason='available')
    return result


def _day(value, name):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(name + ' must be a timezone-naive midnight date')
    return day


def _panel(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'close', 'volume'}:
        raise ValueError('Panel requires exactly adj, close, volume')
    for frame in panel.values():
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError('Panel values require DataFrames with DatetimeIndex')
        dates = frame.index
        if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
                or dates.tz is not None or not dates.equals(dates.normalize())):
            raise ValueError('Panel sessions must be unique, increasing, naive midnight dates')
        if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                                for s in frame.columns)):
            raise ValueError('Security columns must be unique nonempty IDs')
        if not dates.equals(panel['adj'].index) or not frame.columns.equals(panel['adj'].columns):
            raise ValueError('Panel axes must be exactly aligned')


def _calendar(calendar):
    dates = []
    for value in calendar:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()) or day.weekday() >= 5:
            raise ValueError('Calendar requires ISO weekday session dates')
        if dates and day <= dates[-1]:
            raise ValueError('Calendar must be unique and increasing')
        dates.append(day)
    if not dates:
        raise ValueError('Calendar cannot be empty')
    return pd.DatetimeIndex(dates)


def _events(events, sessions):
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, symbols in events:
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(symbols, (str, bytes)):
            raise TypeError('Event symbols must be an explicit sequence')
        symbols = list(symbols)
        if (not symbols or len(set(symbols)) != len(symbols)
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must be unique nonempty IDs')
        if width is not None and width != len(symbols):
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, lookup[day]))
    return rows, width or 0


def build_features(panel, events, calendar, limit=None):
    """Return values(candidate/control/available) and fully dated source audit.

    The supplied calendar is the host's pinned official calendar. A missing
    source session makes the whole measurement unavailable; reindexing here
    locates absence, never fills it. Numeric conversion follows time slicing.
    """
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool))
    history, reasons = [], {}
    for i, (day, symbols, t) in enumerate(queries):
        enough = t >= ACTIVITY_START_OFFSET
        adjusted_dates = sessions[t-ADJUSTED_START_OFFSET:t-SKIP+1] if enough else None
        activity_dates = sessions[t-ACTIVITY_START_OFFSET:t-ACTIVITY_END_OFFSET+1] if enough else None
        missing = []
        source = None
        if enough:
            needed = adjusted_dates.union(activity_dates)
            missing = needed.difference(panel['adj'].index).strftime('%Y-%m-%d').tolist()
            source = {key: frame.reindex(index=adjusted_dates if key == 'adj' else activity_dates,
                                         columns=symbols)
                      for key, frame in panel.items()}
        cells = []
        for j, symbol in enumerate(symbols):
            if not enough:
                result = _empty('insufficient_history')
            elif symbol not in panel['adj'].columns:
                result = _empty('missing_symbol')
            elif missing:
                result = _empty('missing_official_session')
            else:
                result = measure(*(source[key].iloc[:, j].to_numpy(copy=True)
                                   for key in ('adj', 'close', 'volume')))
            for key in values:
                values[key][i, j] = result[key]
            reasons[result['reason']] = reasons.get(result['reason'], 0) + 1
            cell = dict(result, symbol=symbol)
            for key in ('candidate', 'control'):
                cell[key] = float(result[key]) if math.isfinite(result[key]) else None
            for source_key, date_key in (('selected_anchors', 'selected_anchor_dates'),
                                         ('high_anchors', 'high_anchor_dates'),
                                         ('low_anchors', 'low_anchor_dates'),
                                         ('invalid_activity_anchors', 'invalid_activity_anchor_dates')):
                cell[date_key] = [str(adjusted_dates[s].date()) for s in result[source_key]]
            cell['label_end_dates'] = [str(adjusted_dates[s+HORIZON].date())
                                       for s in result['selected_anchors']]
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=list(symbols), session_position=t,
                            adjusted_start=str(adjusted_dates[0].date()) if enough else None,
                            adjusted_end=str(adjusted_dates[-1].date()) if enough else None,
                            activity_start=str(activity_dates[0].date()) if enough else None,
                            activity_end=str(activity_dates[-1].date()) if enough else None,
                            missing_official_sessions=missing, cells=cells))
    return values, dict(events=len(queries), symbols_per_event=width,
                        adjusted_count=ADJUSTED_COUNT, activity_count=ACTIVITY_COUNT,
                        history_sessions=HISTORY, skip_sessions=SKIP, horizon_sessions=HORIZON,
                        cooldown_sessions=COOLDOWN, activity_baseline_sessions=BASELINE,
                        prior_mass=PRIOR_MASS, minimum_group_anchors=MIN_GROUP,
                        available_cells=int(values['available'].sum()),
                        unavailable_cells=int(values['available'].size-values['available'].sum()),
                        reason_counts=reasons, event_history=history, common_availability=True,
                        activity_threshold_arithmetic='exact_dyadic_products_and_integer_sum',
                        historical_matured_price_outcomes_used=True,
                        portfolio_returns_calculated=False, ranking_performed=False,
                        trading_eligibility_modified=False)
