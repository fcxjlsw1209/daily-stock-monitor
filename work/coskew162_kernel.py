"""Pure skip-42 residual coskewness; no file, strategy, or outcome I/O.

The caller supplies adjusted prices, original event membership and an official
session calendar. Metadata is an unused compatibility argument. Neither input
eligibility nor the momentum core is read or changed by this module.
"""
import numpy as np
import pandas as pd


LOOKBACK = 252
SKIP = 42
WINDOW = LOOKBACK - SKIP
MARKET_VARIANCE_MIN = 1e-24
RESIDUAL_SD_MIN = 1e-12
INITIAL_PARTIAL = pd.Timestamp("2016-08-17")


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " must be a valid session date") from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + " must be a timezone-naive midnight date")
    return day


def _validate_prices(prices):
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


def _coskewness(prices, symbols, sessions, position):
    values = np.full(len(symbols), np.nan, dtype=float)
    if position < LOOKBACK:
        return values
    dates = sessions[position - LOOKBACK:position - SKIP + 1]
    # Select only the causal window. A missing official session is a NaN;
    # extra rows in the input never shift the calendar offsets or fill gaps.
    market_prices = prices.reindex(index=dates, columns=["SPY"]).to_numpy(dtype=float)[:, 0]
    if not (np.isfinite(market_prices) & (market_prices > 0.)).all():
        return values
    stock_prices = prices.reindex(index=dates, columns=symbols).to_numpy(dtype=float)
    good = (np.isfinite(stock_prices) & (stock_prices > 0.)).all(axis=0)
    if not good.any():
        return values
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        market_returns = market_prices[1:] / market_prices[:-1] - 1.
        if not np.isfinite(market_returns).all():
            return values
        m = market_returns - np.mean(market_returns)
        squared = m * m
        market_variance = np.mean(squared)
        denominator = np.sum(squared)
        if (not np.isfinite(market_variance) or market_variance <= MARKET_VARIANCE_MIN
                or not np.isfinite(denominator)):
            return values
        # Scalar columns keep results independent of event membership ordering.
        for column in np.flatnonzero(good):
            returns = stock_prices[1:, column] / stock_prices[:-1, column] - 1.
            if not np.isfinite(returns).all():
                continue
            y = returns - np.mean(returns)
            beta = np.sum(m * y) / denominator
            residual = y - beta * m
            residual_sd = np.sqrt(np.mean(residual * residual))
            if (not np.isfinite(beta) or not np.isfinite(residual).all()
                    or not np.isfinite(residual_sd) or residual_sd <= RESIDUAL_SD_MIN):
                continue
            value = np.mean(residual * squared) / (residual_sd * market_variance)
            if np.isfinite(value):
                values[column] = value
    return values


def _json_floats(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def build_features(prices, training_meta, events, calendar, limit=None):
    """Return two score-delta matrices and a JSON-safe source-only audit.

    At official session t, 211 prices [t-252, t-42] give 210 simple
    returns. For centered market m and stock y, beta=sum(m*y)/sum(m*m),
    e=y-beta*m, and c=mean(e*m*m)/(sqrt(mean(e*e))*mean(m*m)). All
    finite c in the original event pool, including ETFs, receive ascending
    average percentile ranks minus 0.5. Missing c is never zero-filled into
    ranking. Positive and negative deltas are +/-0.02*rank. The initial
    2016-08-17 event is forced neutral; its raw values remain auditable.

    Official month-end validation requires the next calendar session, never
    that session's price. All event/axis schema is checked before taking a
    prefix; only selected event windows are converted to numeric prices.
    The supplied calendar's provenance is the caller's responsibility.
    """
    _validate_prices(prices)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError("limit must be an integer event-prefix length")
    selected = queries if limit is None else queries[:limit]
    positive = np.zeros((len(selected), width), dtype=float)
    negative = np.zeros_like(positive)
    history = []
    support_cells = 0
    for row, (day, symbols, position) in enumerate(selected):
        values = _coskewness(prices, symbols, sessions, position)
        ranks = pd.Series(values).rank(method="average", ascending=True, pct=True).to_numpy() - .5
        initial = bool(day == INITIAL_PARTIAL)
        support = np.isfinite(values) & (not initial)
        positive[row, support] = .02 * ranks[support]
        negative[row, support] = -positive[row, support]
        support_cells += int(support.sum())
        history.append({
            "date": str(day.date()), "symbols": list(symbols),
            "window_start": str(sessions[position - LOOKBACK].date()) if position >= LOOKBACK else None,
            "window_end": str(sessions[position - SKIP].date()) if position >= SKIP else None,
            "common_support": support.tolist(),
            "raw_coskewness": _json_floats(values), "ranks": _json_floats(ranks),
            "initial_forced_neutral": initial,
        })
    audit = {
        "events": len(selected), "symbols_per_event": width,
        "window_returns": WINDOW, "skip_sessions": SKIP,
        "common_support_cells": support_cells,
        "neutral_cells": len(selected) * width - support_cells,
        "trading_eligibility_modified": False, "outcomes_used": False,
        "event_history": history,
    }
    return {"delta_positive": positive, "delta_negative": negative}, audit
