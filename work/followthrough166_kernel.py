"""Pure skip-42 volume-followthrough partial correlation; no file or outcome I/O.

The caller supplies aligned adjusted prices and volume, original membership and an official
session calendar. Metadata is an unused compatibility argument. Neither input
eligibility nor the momentum core is read or changed by this module.
"""
import numpy as np
import pandas as pd


LOOKBACK = 252
SKIP = 42
WINDOW = LOOKBACK - SKIP
HISTORY = LOOKBACK + 1
ACTIVITY_BASELINE = 21
VOLUME_HISTORY = LOOKBACK + ACTIVITY_BASELINE
RELATIVE_EPS = 1e-12
ROUNDING_EPS = 64 * np.finfo(float).eps
INITIAL_PARTIAL = pd.Timestamp("2016-08-17")


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " must be a valid session date") from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + " must be a timezone-naive midnight date")
    return day


def _validate_frame(prices):
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be a DataFrame with a DatetimeIndex")
    index = prices.index
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("Price dates must be timezone-naive midnight dates")
    if not prices.columns.is_unique:
        raise ValueError("Price security columns must be unique")
    if any(not isinstance(symbol, str) or not symbol for symbol in prices.columns):
        raise ValueError("Price security columns must be nonempty strings")


def _validate_panel(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'volume'}:
        raise ValueError("Panel requires exactly adj, volume DataFrames")
    for frame in panel.values():
        _validate_frame(frame)
    for frame in panel.values():
        if not frame.index.equals(panel['adj'].index) or not frame.columns.equals(panel['adj'].columns):
            raise ValueError("Panel DataFrames must have exactly aligned axes")


def _calendar(calendar):
    days = []
    for value in calendar:
        day = _day(value, "Calendar")
        if not isinstance(value, str) or value != str(day.date()):
            raise ValueError("Calendar dates must be ISO YYYY-MM-DD strings")
        if day.weekday() >= 5:
            raise ValueError("Weekend in supplied official session calendar")
        if days and day <= days[-1]:
            raise ValueError("Calendar dates must be unique and increasing")
        days.append(day)
    if not days:
        raise ValueError("Calendar must not be empty")
    return pd.DatetimeIndex(days)


def _events(events, sessions):
    positions = {day: index for index, day in enumerate(sessions)}
    queries = []
    width = None
    for value, members in events:
        day = _day(value, "Event")
        if queries and day <= queries[-1][0]:
            raise ValueError("Event dates must be unique and increasing")
        if isinstance(members, (str, bytes)):
            raise TypeError("Event symbols must be a sequence")
        symbols = list(members)
        if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError("Event symbols must be nonempty strings in a nonempty pool")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in Event")
        if width is not None and len(symbols) != width:
            raise ValueError("All events must have the same number of symbols")
        width = len(symbols)
        if day not in positions:
            raise ValueError("Event is not an official calendar session: " + str(day.date()))
        position = positions[day]
        if day != INITIAL_PARTIAL:
            if position + 1 >= len(sessions):
                raise ValueError("Calendar needs the next session to verify month end: " + str(day.date()))
            if sessions[position + 1].to_period("M") == day.to_period("M"):
                raise ValueError("Noninitial event must be an official month end: " + str(day.date()))
        queries.append((day, symbols, position))
    return queries, (width or 0)


def _center_unit(values):
    """Scale before centering, rejecting dimensionless near-constant inputs."""
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0.:
        return None
    scaled = values / scale
    centered = scaled - scaled.mean()
    length = np.linalg.norm(centered)
    if not np.isfinite(length) or length <= RELATIVE_EPS * np.linalg.norm(scaled):
        return None
    return centered / length


def _roundoff_clip(value):
    if not np.isfinite(value) or abs(value) > 1. + ROUNDING_EPS:
        return np.nan
    return float(np.clip(value, -1., 1.))


def _paired_correlations(adj, volume):
    """Followthrough/control from exactly 212 adjusted and 231 volume values."""
    adj, volume = (np.asarray(v, dtype=float) for v in (adj, volume))
    if adj.shape != (WINDOW + 2,) or volume.shape != (WINDOW + ACTIVITY_BASELINE,):
        raise ValueError("Followthrough requires exactly 212 adjusted and 231 volume values")
    if not all((np.isfinite(v) & (v > 0.)).all() for v in (adj, volume)):
        return np.nan, np.nan
    returns = np.diff(np.log(adj))
    r, y = returns[:-1], returns[1:]
    log_volume = np.log(volume)
    activity = log_volume[ACTIVITY_BASELINE:] - np.asarray([
        log_volume[i:i + ACTIVITY_BASELINE].mean() for i in range(WINDOW)])
    ur, ua, uy = (_center_unit(v) for v in (r, activity, y))
    if any(v is None for v in (ur, ua, uy)):
        return np.nan, np.nan
    # Form the interaction only after normalization to avoid under/overflow.
    uq = _center_unit(ur * ua)
    if uq is None:
        return np.nan, np.nan
    activity_basis = ua - np.dot(ua, ur) * ur
    basis_norm = np.linalg.norm(activity_basis)
    if not np.isfinite(basis_norm) or basis_norm <= RELATIVE_EPS:
        return np.nan, np.nan
    activity_basis /= basis_norm
    residuals = []
    for vector in (uq, uy):
        residual = vector - np.dot(vector, ur) * ur
        residual -= np.dot(residual, activity_basis) * activity_basis
        length = np.linalg.norm(residual)
        if not np.isfinite(length) or length <= RELATIVE_EPS:
            return np.nan, np.nan
        residuals.append(residual / length)
    followthrough = _roundoff_clip(np.dot(*residuals))
    control = _roundoff_clip(np.dot(ur, uy))
    if not np.isfinite(followthrough) or not np.isfinite(control):
        return np.nan, np.nan
    return followthrough, control


def _followthrough_values(panel, symbols, sessions, position):
    values = np.full((2, len(symbols)), np.nan, dtype=float)
    if position < VOLUME_HISTORY:
        return values
    dates = {'adj': sessions[position - HISTORY:position - SKIP + 1],
             'volume': sessions[position - VOLUME_HISTORY:position - SKIP]}
    # Slice official source windows BEFORE conversion; ignore every other value.
    sliced = {key: frame.reindex(index=dates[key], columns=symbols).to_numpy(dtype=float)
              for key, frame in panel.items()}
    for column in range(len(symbols)):
        values[:, column] = _paired_correlations(sliced['adj'][:, column], sliced['volume'][:, column])
    return values


def _json_floats(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def build_features(panel, training_meta, events, calendar, limit=None):
    """Return two score-delta matrices and a JSON-safe source-only audit.

    For s=t-252..t-43 (210 pairs), r=diff(log(adjusted_close)),
    y_s=r_(s+1), a_s=log(V_s)-mean(log(V_(s-21:s-1))).
    The centered, unit-L2 r and a product defines q; partial corr(q,y | r,a)
    is the followthrough score; corr(r,y) is the matched
    control. Both are unavailable together on any missing/invalid source or
    numerical degeneracy. The shared finite original-pool members receive
    separate ascending average percentile ranks minus .5 and doses +.04*rank.
    Initial 2016-08-17 is forced neutral while raw measurements stay auditable.

    Official month-end validation requires the next calendar session, never
    that session's price. All event/axis schema is checked before taking a
    prefix; only selected event windows are converted to numeric prices.
    The supplied calendar's provenance is the caller's responsibility.
    """
    _validate_panel(panel)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError("limit must be an integer event-prefix length")
    selected = queries if limit is None else queries[:limit]
    followthrough = np.zeros((len(selected), width), dtype=float)
    control = np.zeros_like(followthrough)
    history = []
    support_cells = 0
    for row, (day, symbols, position) in enumerate(selected):
        values = _followthrough_values(panel, symbols, sessions, position)
        available = np.isfinite(values).all(axis=0)
        values[:, ~available] = np.nan
        ranks = np.asarray([pd.Series(v).rank(method="average", ascending=True, pct=True).to_numpy() - .5
                            for v in values])
        initial = bool(day == INITIAL_PARTIAL)
        support = available & (not initial)
        followthrough[row, support] = .04 * ranks[0, support]
        control[row, support] = .04 * ranks[1, support]
        support_cells += int(support.sum())
        history.append({
            "date": str(day.date()), "symbols": list(symbols),
            "window_start": str(sessions[position - HISTORY].date()) if position >= HISTORY else None,
            "window_end": str(sessions[position - SKIP].date()) if position >= SKIP else None,
            "volume_window_start": str(sessions[position - VOLUME_HISTORY].date()) if position >= VOLUME_HISTORY else None,
            "volume_window_end": str(sessions[position - SKIP - 1].date()) if position > SKIP else None,
            "pair_start": str(sessions[position - LOOKBACK].date()) if position >= LOOKBACK else None,
            "pair_end": str(sessions[position - SKIP - 1].date()) if position > SKIP else None,
            "common_support": support.tolist(),
            "raw_followthrough": _json_floats(values[0]), "raw_control": _json_floats(values[1]),
            "ranks_followthrough": _json_floats(ranks[0]), "ranks_control": _json_floats(ranks[1]),
            "initial_forced_neutral": initial,
        })
    audit = {
        "events": len(selected), "symbols_per_event": width,
        "window_returns": WINDOW, "skip_sessions": SKIP,
        "window_pairs": WINDOW, "adjusted_window_prices": WINDOW + 2,
        "volume_window_values": WINDOW + ACTIVITY_BASELINE, "activity_baseline_sessions": ACTIVITY_BASELINE, "relative_degeneracy_epsilon": RELATIVE_EPS,
        "common_support_cells": support_cells,
        "neutral_cells": len(selected) * width - support_cells,
        "trading_eligibility_modified": False, "outcomes_used": False,
        "event_history": history,
    }
    return {"delta_followthrough": followthrough, "delta_control": control}, audit
