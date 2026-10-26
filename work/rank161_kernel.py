"""Past-only rank residence and chronological rank trend; no file or outcome IO.

The caller supplies original historical membership snapshots and a fixed,
official session calendar. Each query uses exactly its current calendar month
and the preceding five months. The two deltas share an all-six-observations
support mask; neither delta controls the caller's trading eligibility.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SKIP = 42
LOOKBACK = 252
WARMUP = 295
MONTHS = 6
INITIAL_PARTIAL = pd.Timestamp("2016-08-17")


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " must be a valid session date") from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + " must be a timezone-naive midnight date")
    return day


def _normalize_events(events, label):
    result = []
    for value, members in events:
        day = _day(value, label)
        if isinstance(members, (str, bytes)):
            raise TypeError(label + " symbols must be a sequence")
        symbols = list(members)
        if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError(label + " symbols must be nonempty strings")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in " + label)
        if result and day < result[-1][0]:
            raise ValueError(label + " dates must be chronological")
        result.append((day, symbols))
    return result


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


def _validate_calendar(calendar):
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


def _json_floats(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in array]
    return [_json_floats(row) for row in array]


def _snapshot(prices, day, symbols, calendar, position):
    """Rank all endpoint-valid original members before the individual gate."""
    count = len(symbols)
    momentum = np.full(count, np.nan)
    endpoint_valid = np.zeros(count, dtype=bool)
    complete = np.zeros(count, dtype=bool)
    if position >= LOOKBACK:
        endpoint_days = calendar[[position - LOOKBACK, position - SKIP]]
        endpoints = prices.reindex(index=endpoint_days, columns=symbols).to_numpy(dtype=float)
        endpoint_valid = (np.isfinite(endpoints) & (endpoints > 0.)).all(axis=0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            computed = endpoints[1] / endpoints[0] - 1.
        endpoint_valid &= np.isfinite(computed)
        momentum[endpoint_valid] = computed[endpoint_valid]
    required = calendar[max(0, position - WARMUP + 1):position + 1]
    if position >= WARMUP - 1:
        # Reindexing against official dates exposes missing sessions as NaN;
        # extra noncalendar price rows never shift the offsets or fill a gap.
        history = prices.reindex(index=required, columns=symbols).to_numpy(dtype=float)
        complete = (np.isfinite(history) & (history > 0.)).all(axis=0)
    positive = endpoint_valid & (momentum > 0.)
    usable = complete & positive
    ranks = pd.Series(momentum, index=symbols).rank(method="average", pct=True).to_numpy() - .5
    missing_dates = required[~required.isin(prices.index)]
    audit = {
        "date": str(day.date()),
        "symbols": list(symbols),
        "member_count": count,
        "endpoint_valid_count": int(endpoint_valid.sum()),
        "usable_count": int(usable.sum()),
        "momentum": _json_floats(momentum),
        "centered_percentile_ranks": _json_floats(ranks),
        "endpoint_valid": endpoint_valid.tolist(),
        "complete_295_prices": complete.tolist(),
        "positive_momentum": positive.tolist(),
        "usable": usable.tolist(),
        "calendar_history_sufficient": bool(position >= WARMUP - 1),
        "required_window_start": str(required[0].date()),
        "required_window_end": str(day.date()),
        "required_window_sessions": len(required),
        "missing_price_sessions": [str(value.date()) for value in missing_dates],
    }
    return dict(zip(symbols, ranks)), dict(zip(symbols, usable)), audit


def build_features(prices: pd.DataFrame, training_meta: Mapping,
                   events: Sequence[Tuple[object, Sequence[str]]],
                   calendar: Sequence[str], limit=None) -> Tuple[dict, dict]:
    """Return ``delta_residence``, ``delta_rank_trend`` arrays and JSON-safe audit.

    Arrays have shape (processed queries, current event width); a neutral cell
    is exactly zero. ``limit`` selects an event prefix without changing the
    available historical union. Metadata has matching ``dates``/``symbols``
    rows. Same-date reordered equal sets are accepted; conflicting sets raise.
    The full metadata/event schema is validated even when ``limit`` is used.

    Noninitial snapshots must be official month ends, demonstrated by the next
    supplied calendar session falling in another month. This requires calendar
    coverage beyond the last snapshot, but never future prices. Calendar and
    price axes must be sorted and unique. Missing official price sessions stay
    missing. No current membership is copied into earlier months.

    ``rank_trajectories`` contains usable centered ranks in oldest-to-newest
    order (six rows by current width); ``raw_rank_trajectories`` additionally
    exposes endpoint-valid ranks failing the warmup/positive-momentum gate.
    ``snapshot_audits`` records complete original membership and gate details.
    The special initialization may appear only in its own query's history;
    that query is always neutral regardless of historical support.
    """
    _validate_prices(prices)
    sessions = _validate_calendar(calendar)
    positions = {day: index for index, day in enumerate(sessions)}
    queries = _normalize_events(events, "Event")
    dates, symbol_rows = list(training_meta["dates"]), list(training_meta["symbols"])
    if len(dates) != len(symbol_rows):
        raise ValueError("Training dates and symbols have different lengths")
    histories = _normalize_events(zip(dates, symbol_rows), "Training")
    width = len(queries[0][1]) if queries else 0
    if any(len(symbols) != width for _, symbols in queries):
        raise ValueError("All events must have the same number of symbols")
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError("limit must be an integer event-prefix length")
    selected_queries = queries if limit is None else queries[:limit]
    snapshots = {}
    for day, symbols in histories + queries:
        if day in snapshots and set(snapshots[day]) != set(symbols):
            raise ValueError("Conflicting same-date membership: " + str(day.date()))
        snapshots.setdefault(day, symbols)
    by_month = {}
    for day in sorted(snapshots):
        if day not in positions:
            raise ValueError("Snapshot is not an official calendar session: " + str(day.date()))
        if day == INITIAL_PARTIAL:
            continue
        month = day.to_period("M")
        if month in by_month:
            raise ValueError("Multiple noninitial snapshots in calendar month: " + str(month))
        by_month[month] = day
    # Validate monthly uniqueness first, so a conflicting monthly archive is
    # distinguished from an individually mistimed snapshot.
    for day in by_month.values():
        position = positions[day]
        if position + 1 >= len(sessions):
            raise ValueError("Calendar needs the next session to verify month end: " + str(day.date()))
        if sessions[position + 1].to_period("M") == day.to_period("M"):
            raise ValueError("Noninitial snapshot must be an official month end: " + str(day.date()))
    cache = {}

    def snapshot(day):
        if day not in cache:
            cache[day] = _snapshot(prices, day, snapshots[day], sessions, positions[day])
        return cache[day]

    residence = np.zeros((len(selected_queries), width), dtype=float)
    trend = np.zeros_like(residence)
    histogram = Counter()
    event_history = []
    support_cells = 0
    for row, (day, symbols) in enumerate(selected_queries):
        month = day.to_period("M")
        months = pd.period_range(month - (MONTHS - 1), month, freq="M")
        selected = [by_month.get(period) for period in months]
        selected = [prior if prior is not None and prior <= day else None for prior in selected]
        if day == INITIAL_PARTIAL:
            selected[-1] = day
        raw = np.full((MONTHS, width), np.nan)
        observed = np.full_like(raw, np.nan)
        usable_mask = np.zeros((MONTHS, width), dtype=bool)
        for offset, prior in enumerate(selected):
            if prior is None:
                continue
            ranks, usable, _ = snapshot(prior)
            for col, symbol in enumerate(symbols):
                if symbol in ranks:
                    raw[offset, col] = ranks[symbol]
                if usable.get(symbol, False):
                    usable_mask[offset, col] = True
                    observed[offset, col] = ranks[symbol]
        counts = usable_mask.sum(axis=0)
        support = (counts == MONTHS) & (day != INITIAL_PARTIAL)
        support_cells += int(support.sum())
        histogram.update(int(value) for value in counts)
        if support.any():
            supported = observed[:, support]
            residence[row, support] = .02 * np.clip(supported.mean(axis=0) - supported.std(axis=0, ddof=0), -.5, .5)
            # Pair symmetric dates so a constant rank path is exactly zero,
            # without a dot-product cancellation residual changing scores.
            numerator = (2.5 * (supported[5] - supported[0])
                         + 1.5 * (supported[4] - supported[1])
                         + .5 * (supported[3] - supported[2]))
            delta = .01 * numerator / 4.5
            if np.any(np.abs(delta) > .01 + 1e-14):
                raise ArithmeticError("Rank trend exceeds its mathematical bound")
            trend[row, support] = np.clip(delta, -.01, .01)
        event_history.append({
            "date": str(day.date()), "symbols": list(symbols),
            "calendar_months": [str(period) for period in months],
            "snapshots": [str(prior.date()) if prior is not None else None for prior in selected],
            "missing_snapshot_months": [str(period) for period, prior in zip(months, selected) if prior is None],
            "usable_observations": counts.astype(int).tolist(),
            "common_support": support.tolist(),
            "usable_mask": usable_mask.tolist(),
            "rank_trajectories": _json_floats(observed),
            "raw_rank_trajectories": _json_floats(raw),
            "initial_forced_neutral": bool(day == INITIAL_PARTIAL),
        })
    audit = {
        "events": len(selected_queries), "symbols_per_event": width,
        "months": MONTHS, "minimum_observations": MONTHS,
        "return_price_offsets": [-LOOKBACK, -SKIP],
        "required_complete_positive_prices": WARMUP,
        "rank_basis": "all original snapshot members with finite positive endpoints and finite momentum; average percentile minus 0.5",
        "observation_gate": "positive momentum and 295 exact official-session finite positive prices ending at the historical snapshot",
        "residence_formula": "0.02*clip(mean(r)-population_std(r),-0.5,0.5)",
        "rank_trend_formula": "0.01*sum((k-2.5)*r[k],k=0..5)/4.5; roundoff-clipped to [-0.01,0.01]",
        "trajectory_order": "oldest to newest; six calendar-month rows by current event symbol columns",
        "partial_initialization": "2016-08-17 query forced neutral; its snapshot never supplies later history",
        "missing_policy": "no month or price filling, no older substitutes; both additions exactly zero unless all six observations usable",
        "common_support_cells": support_cells,
        "neutral_cells": int(residence.size - support_cells),
        "observation_count_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
        "snapshots_computed": len(cache),
        "snapshot_audits": {str(day.date()): cache[day][2] for day in sorted(cache)},
        "event_history": event_history,
        "trading_eligibility_modified": False, "outcomes_used": False,
    }
    return {"delta_residence": residence, "delta_rank_trend": trend}, audit
