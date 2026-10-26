"""Causal two-segment trend quality inside the existing skip-42 window.

The old price segment is t-252..t-126 (126 daily returns); the newer price
segment is t-126..t-42 (84 returns). Both include their shared boundary price,
but no return is counted twice. The caller owns eligibility, ranking and trades.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


OLD_START = 252
BOUNDARY = 126
SKIP = 42
FEATURE_NAMES = ("continuity_floor", "quality_improvement", "downside_improvement")


def _validate(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if "adj" not in panel:
        raise KeyError("Missing required panel: adj")
    ref = panel["adj"]
    if not isinstance(ref, pd.DataFrame):
        raise TypeError("adj must be a pandas DataFrame")
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex of exchange sessions")
    if not ref.index.is_unique or not ref.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not ref.columns.is_unique:
        raise ValueError("Security columns must be unique")
    return ref


def _quality(prices: np.ndarray):
    """Return signed path efficiency and daily downside RMS for valid columns."""
    returns = np.diff(np.log(prices), axis=0)
    absolute = np.abs(returns).sum(axis=0)
    efficiency = np.divide(returns.sum(axis=0), absolute,
                           out=np.zeros(prices.shape[1], dtype=float), where=absolute > 0.)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.) ** 2, axis=0))
    return efficiency, downside


def build_features(panel: Mapping[str, pd.DataFrame],
                   events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return ``(raw, audit)`` with three arrays shaped (events, symbols).

    Production events contain 49 symbols; uniform smaller widths work in tests.
    Each date must be an exact panel session. Unknown security columns retain
    their position as NaN. No ranking, filling, forward filling or eligibility
    changes occur. Missing/nonpositive/nonfinite prices in either segment make
    all three comparisons NaN, independently of the caller's trading eligibility.

    E = sum(log returns)/sum(abs(log returns)), with a completely flat segment
    defined as E=0. Downside RMS averages over every day, including zero/up days.
    The 126-day and 84-day segment metrics are therefore on comparable scales.
    Higher feature values represent better continuity or greater improvement.
    """
    ref = _validate(panel)
    normalized = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    width = len(normalized[0][1]) if normalized else 0
    if any(len(symbols) != width for _, symbols in normalized):
        raise ValueError("All events must have the same number of symbols")
    raw = {key: np.full((len(normalized), width), np.nan) for key in FEATURE_NAMES}
    reasons = Counter()
    old_valid_count = new_valid_count = 0
    for row, (day, symbols) in enumerate(normalized):
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in an event")
        if day not in ref.index:
            raise ValueError("Event date is not an exact panel session: " + str(day))
        t = ref.index.get_loc(day)
        positions = ref.columns.get_indexer(symbols)
        present = positions >= 0
        reasons["unknown_symbol"] += int((~present).sum())
        destinations = np.flatnonzero(present)
        columns = positions[present]
        if not len(columns):
            continue
        if t < OLD_START:
            reasons["insufficient_252_session_price_history"] += len(columns)
            continue
        old = ref.iloc[t - OLD_START:t - BOUNDARY + 1, columns].to_numpy(dtype=float, copy=True)
        new = ref.iloc[t - BOUNDARY:t - SKIP + 1, columns].to_numpy(dtype=float, copy=True)
        old_good = (np.isfinite(old) & (old > 0.)).all(axis=0)
        new_good = (np.isfinite(new) & (new > 0.)).all(axis=0)
        old_valid_count += int(old_good.sum())
        new_valid_count += int(new_good.sum())
        reasons["invalid_old_segment_only"] += int((~old_good & new_good).sum())
        reasons["invalid_new_segment_only"] += int((old_good & ~new_good).sum())
        reasons["invalid_both_segments"] += int((~old_good & ~new_good).sum())
        good = old_good & new_good
        if not good.any():
            continue
        old_e, old_d = _quality(old[:, good])
        new_e, new_d = _quality(new[:, good])
        target = destinations[good]
        raw["continuity_floor"][row, target] = np.minimum(old_e, new_e)
        raw["quality_improvement"][row, target] = new_e - old_e
        raw["downside_improvement"][row, target] = old_d - new_d
    audit = {
        "events": len(normalized), "symbols_per_event": width,
        "old_price_offsets_inclusive": [-OLD_START, -BOUNDARY],
        "new_price_offsets_inclusive": [-BOUNDARY, -SKIP],
        "old_return_count": OLD_START - BOUNDARY,
        "new_return_count": BOUNDARY - SKIP, "skip_sessions": SKIP,
        "old_valid_segment_cells": old_valid_count, "new_valid_segment_cells": new_valid_count,
        "feature_finite_cells": {key: int(np.isfinite(value).sum()) for key, value in raw.items()},
        "feature_missing_cells": {key: int(np.isnan(value).sum()) for key, value in raw.items()},
        "missing_reasons": dict(reasons),
        "price_basis": "adjusted-close daily log returns",
        "efficiency_definition": "sum(log return)/sum(abs(log return)); fully flat segment=0",
        "downside_definition": "sqrt(mean(min(daily log return,0)^2)); all segment days in mean",
        "missing_policy": "Preserve NaN; never substitute zero or change caller eligibility",
        "trading_eligibility_modified": False, "cross_sectional_ranking_applied": False,
        "sector_adjustment_applied": False,
    }
    return raw, audit
