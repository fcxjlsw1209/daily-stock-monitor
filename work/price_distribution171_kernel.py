"""Pure skip-42 price-location distribution with stored dollar activity weights.

No data loading, ranks, orders, returns, or file I/O. Event timing and trading
eligibility remain the caller's responsibility. Missing observations never
shorten the official-session window and are never filled.
"""
import math
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd


LOOKBACK = 252
SKIP = 42
WINDOW = LOOKBACK-SKIP
ADJUSTED_COUNT = WINDOW+1


def _day(value, name):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name+' must be a valid session date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(name+' must be a timezone-naive midnight date')
    return day


def _frame(value):
    if not isinstance(value, pd.DataFrame) or not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError('Panel values must be DataFrames with DatetimeIndex')
    dates = value.index
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Panel dates must be unique and increasing')
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Panel dates must be timezone-naive midnight dates')
    if not value.columns.is_unique or any(not isinstance(s, str) or not s for s in value.columns):
        raise ValueError('Panel security columns must be unique nonempty strings')


def _panel(value):
    if not isinstance(value, dict) or set(value) != {'adj', 'close', 'volume'}:
        raise ValueError('Panel requires exactly adj, close, volume')
    for frame in value.values():
        _frame(frame)
        if not frame.index.equals(value['adj'].index) or not frame.columns.equals(value['adj'].columns):
            raise ValueError('Panel axes must be exactly aligned')


def _calendar(values):
    days = []
    for value in values:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()):
            raise ValueError('Calendar dates must be ISO YYYY-MM-DD strings')
        if day.weekday() >= 5:
            raise ValueError('Weekend in supplied official session calendar')
        if days and day <= days[-1]:
            raise ValueError('Calendar dates must be unique and increasing')
        days.append(day)
    if not days:
        raise ValueError('Calendar must not be empty')
    return pd.DatetimeIndex(days)


def _events(values, sessions):
    lookup = {day:i for i, day in enumerate(sessions)}
    result, width = [], None
    for value, names in values:
        day = _day(value, 'Event')
        if result and day <= result[-1][0]:
            raise ValueError('Events must be unique and increasing')
        if day not in lookup:
            raise ValueError('Event must be an official calendar session')
        if isinstance(names, (str, bytes)):
            raise TypeError('Event members must be a sequence of security IDs')
        names = list(names)
        if not names or any(not isinstance(s, str) or not s for s in names) or len(set(names)) != len(names):
            raise ValueError('Event members must be unique nonempty strings')
        if width is not None and len(names) != width:
            raise ValueError('Event membership width must be constant')
        width = len(names)
        result.append((day, names, lookup[day]))
    return result, width or 0


def _unavailable(reason):
    return dict(candidate=float('nan'), control=float('nan'), available=False,
                reason=reason, positive_activity_days=0)


def _activity_weights(close, volume, positive):
    """Return scaled weights and differences without logarithmic rounding.

    Differences are taken before float scaling so nearly equal activities do
    not collapse. The Decimal path is only for finite-input products that
    overflow/underflow double precision; it preserves their common scale.
    """
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        products = close*volume
    if np.isfinite(products).all() and (products[positive] > 0.).all():
        scale = products.max()
        return products/scale, (products-products[0])/scale, bool(np.all(products == products[0]))
    with localcontext() as context:
        context.prec = 80
        products = [Decimal.from_float(float(c))*Decimal.from_float(float(v))
                    for c, v in zip(close, volume)]
        scale = max(products)
        weights = np.array([float(v/scale) for v in products])
        differences = np.array([float((v-products[0])/scale) for v in products])
        uniform = all(v == products[0] for v in products)
    return weights, differences, uniform


def measure(adjusted, close, volume):
    """Use 211 adjusted prices (last=reference) and 210 historical close/volume.

    Candidate is activity-weighted mean(g) minus equal-weight mean(g), where
    g=(reference-price)/(reference+price). Removing the common positive weight
    scale stabilizes the weighted average. Its centered covariance identity
    avoids subtracting two almost equal means. Adjusted-price ratios are also
    evaluated after pairwise scaling to avoid overflow.
    """
    arrays = []
    for source, label in ((adjusted, 'adjusted_price'), (close, 'stored_close'),
                          (volume, 'stored_volume')):
        try:
            value = np.asarray(source, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return _unavailable('nonnumeric_'+label)
        arrays.append(value)
    adjusted, close, volume = arrays
    if adjusted.shape != (ADJUSTED_COUNT,) or close.shape != (WINDOW,) or volume.shape != (WINDOW,):
        raise ValueError('Expected exactly 211 adjusted prices and 210 stored close/volume values')
    if not np.all(np.isfinite(adjusted) & (adjusted > 0.)):
        return _unavailable('invalid_adjusted_price')
    if not np.all(np.isfinite(close) & (close > 0.)):
        return _unavailable('invalid_stored_close')
    if not np.all(np.isfinite(volume) & (volume >= 0.)):
        return _unavailable('invalid_stored_volume')
    positive = volume > 0.
    if not positive.any():
        return _unavailable('zero_total_activity')
    reference, history = adjusted[-1], adjusted[:-1]
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            scale = np.maximum(reference, history)
            r, p = reference/scale, history/scale
            g = (r-p)/(r+p)
            weights, differences, uniform = _activity_weights(close, volume, positive)
            denominator = math.fsum(float(x) for x in weights)
            control = math.fsum(float(x) for x in g)/WINDOW
            candidate = math.fsum(float(w)*(float(x)-control)
                                  for w, x in zip(differences, g))/denominator
        # Algebraically identical constant cases have exact zero effect; no
        # tolerance or equality of rounded logarithms is used as evidence.
        if uniform or np.all(history == history[0]):
            candidate = 0.
        if not math.isfinite(candidate) or not math.isfinite(control):
            return _unavailable('nonfinite_numeric_result')
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        return _unavailable('nonfinite_numeric_result')
    return dict(candidate=float(candidate), control=float(control), available=True,
                reason='available', positive_activity_days=int(positive.sum()))


def build_features(panel, events, calendar, limit=None):
    """Return event-by-member candidate/control/available matrices and audit.

    At official session index t, adjusted prices are [t-252,t-42] inclusive;
    the 210 historical activity weights are [t-252,t-43]. A reference-day
    close/volume or any more recent value cannot enter either factor.
    The kernel accepts any official event session, leaving month-end/initial
    allocation policy to its host. No event is silently omitted or neutralized.
    """
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    selected = queries if limit is None else queries[:limit]
    shape = (len(selected), width)
    candidate = np.full(shape, np.nan)
    control = np.full(shape, np.nan)
    available = np.zeros(shape, bool)
    histories = []
    reason_counts = {}
    for row, (day, symbols, position) in enumerate(selected):
        enough = position >= LOOKBACK
        dates = sessions[position-LOOKBACK:position-SKIP+1] if enough else None
        source = None
        missing_sessions = []
        if enough:
            missing_sessions = dates.difference(panel['adj'].index).strftime('%Y-%m-%d').tolist()
            # Reindex/slice before numeric conversion; future/object poison is
            # untouched, and an isolated bad symbol cannot spoil its peers.
            source = {key:frame.reindex(index=dates if key == 'adj' else dates[:-1], columns=symbols)
                      for key, frame in panel.items()}
        reasons, positive_counts = [], []
        for column, symbol in enumerate(symbols):
            if not enough:
                result = _unavailable('insufficient_history')
            elif symbol not in panel['adj'].columns:
                result = _unavailable('missing_symbol')
            elif missing_sessions:
                result = _unavailable('missing_official_session')
            else:
                result = measure(*(source[key].iloc[:, column].to_numpy(copy=True)
                                   for key in ('adj','close','volume')))
            candidate[row, column] = result['candidate']
            control[row, column] = result['control']
            available[row, column] = result['available']
            reasons.append(result['reason'])
            positive_counts.append(result['positive_activity_days'])
            reason_counts[result['reason']] = reason_counts.get(result['reason'], 0)+1
        histories.append(dict(date=str(day.date()), symbols=list(symbols), session_position=position,
            window_start=str(dates[0].date()) if enough else None,
            history_end=str(dates[-2].date()) if enough else None,
            reference_date=str(dates[-1].date()) if enough else None,
            required_adjusted_count=ADJUSTED_COUNT, required_activity_count=WINDOW,
            missing_official_sessions=missing_sessions, available=available[row].tolist(),
            candidate=[float(v) if np.isfinite(v) else None for v in candidate[row]],
            control=[float(v) if np.isfinite(v) else None for v in control[row]],
            reasons=reasons, positive_activity_days=positive_counts))
    return dict(candidate=candidate, control=control, available=available), dict(
        events=len(selected), symbols_per_event=width, lookback_sessions=LOOKBACK,
        skip_sessions=SKIP, history_sessions=WINDOW, adjusted_count=ADJUSTED_COUNT,
        weighting='stored_close * stored_volume; common scale removed numerically',
        formula='weighted_mean((ref-p)/(ref+p))-mean((ref-p)/(ref+p))',
        control='equal_weight_mean((ref-p)/(ref+p))',
        common_availability=True, available_cells=int(available.sum()),
        unavailable_cells=int(available.size-available.sum()), reason_counts=reason_counts,
        ranking_performed=False, trading_eligibility_modified=False, outcomes_used=False,
        event_history=histories)
