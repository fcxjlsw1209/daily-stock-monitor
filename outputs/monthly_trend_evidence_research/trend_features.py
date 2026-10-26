"""Two causal, unranked confirmations of the existing long momentum window.

The caller owns the session calendar, universe, eligibility, ranks and trading.
At signal row t both features end at t-42. Returns span t-251..t-42; the
adjusted-price path spans t-252..t-42. Volume normalization additionally needs
raw dollar volume back to t-314. Missing inputs remain NaN, never zero.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


WINDOW = 210
SKIP = 42
VOLUME_LOOKBACK = 63
WEIGHT_MIN = .25
WEIGHT_MAX = 4.
NEW_HIGH_TOLERANCE = 1e-12
FEATURE_NAMES = ("volume_confirmation", "recovery_speed")


def _validate(panel: Mapping[str, pd.DataFrame]):
    for key in ("adj", "close", "volume"):
        if key not in panel:
            raise KeyError("Missing required panel: " + key)
        if not isinstance(panel[key], pd.DataFrame):
            raise TypeError(key + " must be a pandas DataFrame")
    ref = panel["adj"]
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex of exchange sessions")
    if not ref.index.is_unique or not ref.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not ref.columns.is_unique:
        raise ValueError("Security columns must be unique")
    for key in ("close", "volume"):
        if not panel[key].index.equals(ref.index) or not panel[key].columns.equals(ref.columns):
            raise ValueError(key + " must have exactly the adj index and columns")
    return ref


def _path_recovery(prices: np.ndarray) -> np.ndarray:
    """Mean age below the running high, including the initial age-zero price.

    The running high never declines. A price within 1e-12 relative distance of
    that high counts as a new high for age only, including an exactly flat day.
    """
    high = prices[0].copy()
    age = np.zeros(prices.shape[1], dtype=float)
    age_sum = age.copy()
    for price in prices[1:]:
        new_high = price >= high * (1. - NEW_HIGH_TOLERANCE)
        age = np.where(new_high, 0., age + 1.)
        high = np.maximum(high, price)
        age_sum += age
    return -(age_sum / len(prices)) / WINDOW


def build_features(panel: Mapping[str, pd.DataFrame],
                   events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return ``(features, audit)``; each feature has shape (events, symbols).

    The production caller supplies 49 symbols per event. Smaller uniform widths
    are supported for independent tests. An event date must be an exact session;
    an unknown security is represented by NaN without shrinking the universe.

    Volume confirmation is 210*sum(w*r)/sum(w)-sum(r), where r is adjusted
    close-to-close log return and w=clip(dv/mean(previous 63 dv), .25, 4).
    dv uses contemporaneous raw close times raw volume. All 210 returns and all
    273 positive finite dollar-volume observations must exist. Recovery requires
    its 211 positive finite adjusted prices independently of volume availability.
    """
    ref = _validate(panel)
    normalized = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    width = len(normalized[0][1]) if normalized else 0
    if any(len(symbols) != width for _, symbols in normalized):
        raise ValueError("All events must have the same number of symbols")
    result = {key: np.full((len(normalized), width), np.nan) for key in FEATURE_NAMES}
    reasons = {key: Counter() for key in FEATURE_NAMES}
    for row, (day, symbols) in enumerate(normalized):
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in an event")
        if day not in ref.index:
            raise ValueError("Event date is not an exact panel session: " + str(day))
        t = ref.index.get_loc(day)
        end, start = t - SKIP, t - SKIP - WINDOW
        positions = ref.columns.get_indexer(symbols)
        present = positions >= 0
        columns = positions[present]
        destinations = np.flatnonzero(present)
        for key in FEATURE_NAMES:
            reasons[key]["unknown_symbol"] += int((~present).sum())
        if not len(columns):
            continue
        if start < 0:
            for key in FEATURE_NAMES:
                reasons[key]["insufficient_adjusted_price_history"] += len(columns)
            continue
        prices = ref.iloc[start:end + 1, columns].to_numpy(dtype=float, copy=True)
        good_price = (np.isfinite(prices) & (prices > 0.)).all(axis=0)
        reasons["recovery_speed"]["invalid_adjusted_price_window"] += int((~good_price).sum())
        reasons["volume_confirmation"]["invalid_adjusted_price_window"] += int((~good_price).sum())
        if good_price.any():
            result["recovery_speed"][row, destinations[good_price]] = _path_recovery(prices[:, good_price])
        dv_start = start + 1 - VOLUME_LOOKBACK
        if dv_start < 0:
            reasons["volume_confirmation"]["insufficient_dollar_volume_history"] += int(good_price.sum())
            continue
        close = panel["close"].iloc[dv_start:end + 1, columns].to_numpy(dtype=float, copy=True)
        volume = panel["volume"].iloc[dv_start:end + 1, columns].to_numpy(dtype=float, copy=True)
        with np.errstate(invalid="ignore", over="ignore"):
            dv = close * volume
        good_dv = (np.isfinite(close) & (close > 0.) & np.isfinite(volume) &
                   (volume > 0.) & np.isfinite(dv) & (dv > 0.)).all(axis=0)
        reasons["volume_confirmation"]["invalid_dollar_volume_window"] += int((good_price & ~good_dv).sum())
        good = good_price & good_dv
        if not good.any():
            continue
        d = dv[:, good]
        # Each window ends on the day before its corresponding return, not on
        # that return's day. There are 63 warmup observations plus 210 targets.
        previous = np.lib.stride_tricks.sliding_window_view(d, VOLUME_LOOKBACK, axis=0)[:-1].mean(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            weights = np.clip(d[VOLUME_LOOKBACK:] / previous, WEIGHT_MIN, WEIGHT_MAX)
            returns = np.diff(np.log(prices[:, good]), axis=0)
            value = WINDOW * (weights * returns).sum(axis=0) / weights.sum(axis=0) - returns.sum(axis=0)
        finite = np.isfinite(value) & np.isfinite(previous).all(axis=0) & (previous > 0.).all(axis=0)
        reasons["volume_confirmation"]["nonfinite_calculation"] += int((~finite).sum())
        target = destinations[good]
        result["volume_confirmation"][row, target[finite]] = value[finite]
    audit = {
        "events": len(normalized), "symbols_per_event": width,
        "window_returns": WINDOW, "skip_sessions": SKIP,
        "price_path_observations": WINDOW + 1, "volume_baseline_sessions": VOLUME_LOOKBACK,
        "volume_baseline_lag_sessions": 1, "weight_clip": [WEIGHT_MIN, WEIGHT_MAX],
        "new_high_relative_tolerance": NEW_HIGH_TOLERANCE,
        "first_required_price_offset": -(WINDOW + SKIP),
        "first_required_dollar_volume_offset": -(WINDOW + SKIP + VOLUME_LOOKBACK - 1),
        "last_used_offset": -SKIP,
        "feature_finite_cells": {key: int(np.isfinite(value).sum()) for key, value in result.items()},
        "feature_missing_cells": {key: int(np.isnan(value).sum()) for key, value in result.items()},
        "missing_reasons": {key: dict(count) for key, count in reasons.items()},
        "price_basis": "adjusted-close log returns and adjusted-close running-high path",
        "dollar_volume_basis": "contemporaneous raw close multiplied by raw volume",
        "missing_policy": "Preserve NaN; never substitute zero or change caller eligibility",
        "trading_eligibility_modified": False, "cross_sectional_ranking_applied": False,
    }
    return result, audit
