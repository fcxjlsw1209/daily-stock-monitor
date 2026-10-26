"""Pure, monthly-horizon conditional historical payoffs; no I/O or trading.

All inputs precede the signal by 42 sessions. The statistic is descriptive
and uses already matured historical blocks, not future or live prices.
"""
import math

import numpy as np
import pandas as pd


SKIP = 42
HISTORY = 1260
BLOCK = 21
BLOCK_COUNT = 60
TRANSITION_COUNT = 59
PRIOR_MASS = 20
MIN_MATCHES = 10
ADJUSTED_COUNT = HISTORY + 1
ADJUSTED_START_OFFSET = HISTORY + SKIP


def _empty(reason):
    return dict(candidate=float('nan'), control=float('nan'), available=False,
                reason=reason, endpoint_indices=[], states=[], log_returns=[],
                training_labels=[], current_state=None,
                matching_transition_indices=[], matching_labels=[],
                matching_count=0, unconditional_mean=None, matching_sum=None,
                matching_mean=None, shrunken_payoff=None)


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


def measure(adjusted):
    """Measure exactly 1261 session prices, inclusive t-1302 through t-42.

Block k (one-based) spans indices 21*(k-1) through 21*k. Train the
59 transitions state[k] -> return[k+1], then query with state[60].
The last block is a matured training outcome as well as the query state;
it is never used as a predictor whose next outcome is already known.
"""
    try:
        prices = np.asarray(adjusted, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _empty('nonnumeric_adjusted_price')
    if prices.shape != (ADJUSTED_COUNT,):
        raise ValueError('Expected exactly 1261 adjusted session prices')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        return _empty('invalid_adjusted_price')
    endpoints = list(range(0, ADJUSTED_COUNT, BLOCK))
    states, returns = [], []
    for a, b in zip(endpoints[:-1], endpoints[1:]):
        states.append(1 if prices[b] > prices[a] else -1 if prices[b] < prices[a] else 0)
        value = _log_ratio(prices[b], prices[a])
        if not math.isfinite(value):
            raise ArithmeticError('Positive finite prices produced a nonfinite log payoff')
        returns.append(value)
    labels = returns[1:]
    mean = math.fsum(labels) / TRANSITION_COUNT
    current = states[-1]
    matching = [j for j in range(1, BLOCK_COUNT) if states[j-1] == current]
    matches = [labels[j-1] for j in matching]
    count = len(matches)
    total = math.fsum(matches)
    result = _empty('insufficient_matches')
    result.update(endpoint_indices=endpoints, states=states, log_returns=returns,
                  training_labels=labels, current_state=current,
                  matching_transition_indices=matching, matching_labels=matches,
                  matching_count=count, unconditional_mean=mean,
                  matching_sum=total,
                  matching_mean=total/count if count else None)
    if count < MIN_MATCHES:
        return result
    candidate = (total + PRIOR_MASS*mean)/(count + PRIOR_MASS)
    if not math.isfinite(candidate):
        raise ArithmeticError('Finite historical payoffs produced a nonfinite shrinkage mean')
    result.update(candidate=candidate, control=mean, available=True,
                  reason='available', shrunken_payoff=candidate)
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
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    frame = panel['adj']
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Panel adj requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Panel sessions must be unique, increasing, naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                           for s in frame.columns)):
        raise ValueError('Security columns must be unique nonempty IDs')


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
    """Return three aligned arrays and a fully dated source-statistic audit.

The host pins an official calendar: absence of any required session makes
both arms unavailable. No missing data is filled. Slice before converting
numeric values so later observations cannot influence an earlier event.
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
        enough = t >= ADJUSTED_START_OFFSET
        dates = sessions[t-ADJUSTED_START_OFFSET:t-SKIP+1] if enough else None
        missing, source = [], None
        if enough:
            missing = dates.difference(panel['adj'].index).strftime('%Y-%m-%d').tolist()
            source = panel['adj'].reindex(index=dates, columns=symbols)
        cells = []
        for j, symbol in enumerate(symbols):
            if not enough:
                result = _empty('insufficient_history')
            elif symbol not in panel['adj'].columns:
                result = _empty('missing_symbol')
            elif missing:
                result = _empty('missing_official_session')
            else:
                result = measure(source.iloc[:, j].to_numpy(copy=True))
            for key in values:
                values[key][i, j] = result[key]
            reasons[result['reason']] = reasons.get(result['reason'], 0) + 1
            cell = dict(result, symbol=symbol)
            for key in ('candidate', 'control'):
                cell[key] = float(result[key]) if math.isfinite(result[key]) else None
            cell['endpoint_dates'] = [str(dates[k].date()) for k in result['endpoint_indices']]
            cell['matching_predictor_end_dates'] = [str(dates[j*BLOCK].date())
                                                    for j in result['matching_transition_indices']]
            cell['matching_label_end_dates'] = [str(dates[(j+1)*BLOCK].date())
                                                for j in result['matching_transition_indices']]
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=list(symbols), session_position=t,
                            adjusted_start=str(dates[0].date()) if enough else None,
                            adjusted_end=str(dates[-1].date()) if enough else None,
                            missing_official_sessions=missing, cells=cells))
    return values, dict(events=len(queries), symbols_per_event=width,
                        adjusted_count=ADJUSTED_COUNT, history_sessions=HISTORY,
                        skip_sessions=SKIP, block_sessions=BLOCK, block_count=BLOCK_COUNT,
                        training_transition_count=TRANSITION_COUNT,
                        prior_mass=PRIOR_MASS, minimum_matches=MIN_MATCHES,
                        available_cells=int(values['available'].sum()),
                        unavailable_cells=int(values['available'].size-values['available'].sum()),
                        reason_counts=reasons, event_history=history, common_availability=True,
                        predictive_model_fits=0, historical_matured_price_outcomes_used=True,
                        portfolio_returns_calculated=False, ranking_performed=False,
                        trading_eligibility_modified=False)
