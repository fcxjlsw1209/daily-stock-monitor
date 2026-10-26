"""Past-only residence in monthly long-momentum cross-sectional rankings.

This describes competitive trend persistence, not past portfolio ownership.
For each actual historical monthly membership snapshot, rank the 210-session
price return ending 42 sessions earlier. A candidate's rank observation is
usable only with positive momentum and 295 complete positive prices then.
Use at most the six calendar months ending in the signal month, never reaching
farther back to replace a missing month. Four usable observations are required;
otherwise return the current ordinary raw-momentum rank. Trading eligibility
remains exclusively the caller's responsibility.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SKIP = 42
LOOKBACK = 252
WARMUP = 295
MONTHS = 6
MIN_OBSERVATIONS = 4
INITIAL_PARTIAL = pd.Timestamp("2016-08-17")


def _validate_panel(panel):
    if "adj" not in panel:
        raise KeyError("Missing required panel: adj")
    prices = panel["adj"]
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("adj must be a DataFrame with a DatetimeIndex of sessions")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not prices.columns.is_unique:
        raise ValueError("Security columns must be unique")
    return prices


def _normalize_events(events):
    result = []
    for day, symbols in events:
        day, symbols = pd.Timestamp(day), list(symbols)
        if pd.isna(day):
            raise ValueError("Invalid snapshot date")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in a snapshot")
        result.append((day, symbols))
    return result


def _snapshot(prices, day, symbols):
    """Current ordinary ranks and historical-rank availability are separate."""
    if day not in prices.index:
        raise ValueError("Snapshot date is not an exact panel session: " + str(day))
    t = prices.index.get_loc(day)
    raw_return = pd.Series(np.nan, index=symbols, dtype=float)
    usable = pd.Series(False, index=symbols, dtype=bool)
    positions = prices.columns.get_indexer(symbols)
    present = positions >= 0
    names = np.asarray(symbols, dtype=object)[present]
    columns = positions[present]
    if t >= LOOKBACK and len(columns):
        endpoints = prices.iloc[[t - LOOKBACK, t - SKIP], columns].to_numpy(dtype=float)
        good = (np.isfinite(endpoints) & (endpoints > 0.)).all(axis=0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            momentum = endpoints[1] / endpoints[0] - 1.
        good &= np.isfinite(momentum)
        raw_return.loc[names[good]] = momentum[good]
        if t >= WARMUP - 1:
            history = prices.iloc[t - WARMUP + 1:t + 1, columns].to_numpy(dtype=float)
            warm = (np.isfinite(history) & (history > 0.)).all(axis=0)
            usable.loc[names] = good & warm & (momentum > 0.)
    ranks = raw_return.rank(method="average", pct=True) - .5
    return ranks, usable


def build_features(panel: Mapping[str, pd.DataFrame], training_meta: Mapping,
                   events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return two arrays and an audit without reading outcomes or trading state.

    ``training_meta`` has equally-sized ``dates`` and ``symbols`` lists; each
    symbols row is that date's historical membership, not today's constituents.
    Events normally have 49 securities; uniform smaller widths work for tests.
    The union of metadata and events supplies historical snapshots, so later
    events remain available after the training cache ends. Same-day reordered
    identical sets are accepted; conflicting memberships raise. An event never
    consumes a later snapshot or later prices. Missing calendar months are not
    filled. The 2016-08-17 initialization is current-only, never later history.

    ``rank_residence`` is mean(observed rank) - population std(observed rank).
    ``rank_residence_fallback`` is boolean and marks the fewer-than-four rule.
    Current missing raw momentum always stays NaN, even with enough history.
    A fallback may also be NaN. No missing price or rank is replaced by zero.
    """
    prices = _validate_panel(panel)
    queries = _normalize_events(events)
    dates, symbol_rows = list(training_meta["dates"]), list(training_meta["symbols"])
    if len(dates) != len(symbol_rows):
        raise ValueError("Training dates and symbols have different lengths")
    histories = _normalize_events(zip(dates, symbol_rows))
    width = len(queries[0][1]) if queries else 0
    if any(len(symbols) != width for _, symbols in queries):
        raise ValueError("All events must have the same number of symbols")
    snapshots = {}
    for day, symbols in histories + queries:
        if day in snapshots and set(snapshots[day]) != set(symbols):
            raise ValueError("Conflicting same-date membership: " + str(day.date()))
        snapshots.setdefault(day, symbols)
    chronological = sorted(snapshots)
    cache = {}

    def snapshot(day):
        if day not in cache:
            cache[day] = _snapshot(prices, day, snapshots[day])
        return cache[day]

    values = np.full((len(queries), width), np.nan)
    fallback = np.zeros((len(queries), width), dtype=bool)
    counts = Counter()
    audit_events = []
    for row, (day, symbols) in enumerate(queries):
        month = day.to_period("M")
        months = pd.period_range(month - (MONTHS - 1), month, freq="M")
        by_month = {}
        for previous in chronological:
            if previous > day:
                break
            if previous == INITIAL_PARTIAL and previous != day:
                continue
            period = previous.to_period("M")
            if period not in months:
                continue
            if period in by_month and previous != by_month[period]:
                raise ValueError("Multiple noninitial snapshots in calendar month: " + str(period))
            by_month[period] = previous
        selected = [by_month.get(period) for period in months]
        ranks_now, _ = snapshot(day)
        observed = np.full((MONTHS, width), np.nan)
        for offset, prior in enumerate(selected):
            if prior is None:
                continue
            ranks, usable = snapshot(prior)
            for col, symbol in enumerate(symbols):
                if symbol in usable.index and usable.at[symbol]:
                    observed[offset, col] = ranks.at[symbol]
        nobs = np.isfinite(observed).sum(axis=0)
        for col, symbol in enumerate(symbols):
            fallback[row, col] = nobs[col] < MIN_OBSERVATIONS
            counts[int(nobs[col])] += 1
            current = ranks_now.at[symbol]
            if not np.isfinite(current):
                continue
            if fallback[row, col]:
                values[row, col] = current
            else:
                available = observed[np.isfinite(observed[:, col]), col]
                values[row, col] = float(available.mean() - available.std(ddof=0))
        audit_events.append({
            "date": str(day.date()),
            "calendar_months": [str(period) for period in months],
            "snapshots": [str(prior.date()) if prior is not None else None for prior in selected],
            "usable_observations": nobs.astype(int).tolist(),
        })
    audit = {
        "events": len(queries), "symbols_per_event": width,
        "months": MONTHS, "minimum_observations": MIN_OBSERVATIONS,
        "return_price_offsets": [-LOOKBACK, -SKIP],
        "return_sessions": LOOKBACK - SKIP,
        "required_complete_positive_prices": WARMUP,
        "rank_basis": "all snapshot members with finite positive endpoint prices; average tie percentile minus 0.5",
        "observation_gate": "positive raw long momentum and 295 finite positive prices ending at each historical signal",
        "residence_formula": "mean(usable centered percentile ranks)-population_std(usable centered percentile ranks)",
        "fallback": "fewer than 4 usable observations among last 6 calendar months: current ordinary raw rank",
        "partial_initialization": "2016-08-17 usable only as its own current event; never future history",
        "missing_policy": "no calendar-month filling, no zero ranks, current missing raw rank stays NaN",
        "fallback_cells": int(fallback.sum()),
        "finite_cells": int(np.isfinite(values).sum()),
        "missing_cells": int(np.isnan(values).sum()),
        "observation_count_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
        "snapshots_computed": len(cache), "event_history": audit_events,
        "trading_eligibility_modified": False, "outcomes_used": False,
    }
    return {"rank_residence": values, "rank_residence_fallback": fallback}, audit
