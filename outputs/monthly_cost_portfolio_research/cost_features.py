"""Causal close/volume reference-price proxies, not observed investor cost bases.

Only signal-time data are consumed. The caller owns ranking, eligibility and
trading. Missing observations remain NaN with explicit validity masks; paired
time-only controls deliberately use the SAME price/volume validity mask.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

WINDOWS = (126, 252)
HALF_LIFE = 63
PRICE_RELATIVE_TOLERANCE = 1e-12
KINDS = ("volume_gap", "time_gap", "volume_upper_mass", "time_upper_mass")
FEATURE_NAMES = tuple(kind + str(window) for window in WINDOWS for kind in KINDS)


def _validate(panel):
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


def _normalized_weights(log_weights):
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


def _window_features(adjusted, close, volume):
    """Four dimensionless raw values, with no ranks or trading decisions.

    q_s = stored_close_s * stored_volume_s / adjusted_close_s is an economic
    adjusted-unit volume proxy. With h_s=2**(-(t-s)/63), the reference is the
    h*q weighted arithmetic adjusted price. The time control uses h alone.
    Both return log(A_t/reference) and the weighted fraction of price levels
    strictly above A_t (relative tolerance 1e-12). Each daily volume is assigned
    to that day's close, not to the unknown intraday distribution of prices.
    """
    n = len(adjusted)
    log_price = np.log(adjusted)
    age = np.arange(n - 1, -1, -1, dtype=float)
    log_decay = -np.log(2.) * age / HALF_LIFE
    log_units = np.log(close) + np.log(volume) - log_price
    volume_weights = _normalized_weights(log_decay + log_units)
    time_weights = _normalized_weights(log_decay)
    # A flat path has no gap and no overhead price mass, exactly. This also
    # suppresses insignificant floating-point noise after unit conversions.
    if np.max(log_price) - np.min(log_price) <= np.log1p(PRICE_RELATIVE_TOLERANCE):
        return (0., 0., 0., 0.)
    above = log_price - log_price[-1] > np.log1p(PRICE_RELATIVE_TOLERANCE)
    max_price = np.max(log_price)
    scaled_price = np.exp(log_price - max_price)
    log_volume_reference = max_price + np.log(np.dot(volume_weights, scaled_price))
    log_time_reference = max_price + np.log(np.dot(time_weights, scaled_price))
    return (float(log_price[-1] - log_volume_reference),
            float(log_price[-1] - log_time_reference),
            float(np.clip(volume_weights[above].sum(), 0., 1.)),
            float(np.clip(time_weights[above].sum(), 0., 1.)))


def build_features(panel: Mapping[str, pd.DataFrame],
                   events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return ``(raw, audit)`` with arrays shaped (events, symbols per event).

    Production events contain 49 symbols. Smaller uniform widths are allowed
    for independent tests. Windows contain exactly 126/252 observations ending
    at the signal close: [t-window+1, t]. The auxiliary signals need not skip
    42 sessions; the caller must preserve its independently defined long core.

    Every adjusted price, stored close and stored volume in the chosen window
    must be positive and finite. Zero volume is conservatively treated as
    unavailable, not silently dropped. A failure masks all four paired features
    for that window, including controls whose formula needs only prices.
    """
    ref = _validate(panel)
    normalized = [(pd.Timestamp(day), list(symbols)) for day, symbols in events]
    width = len(normalized[0][1]) if normalized else 0
    if any(len(symbols) != width for _, symbols in normalized):
        raise ValueError("All events must have the same number of symbols")
    for day, symbols in normalized:
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate symbols in an event")
        if day not in ref.index:
            raise ValueError("Event date is not an exact panel session: " + str(day))
    shape = (len(normalized), width)
    raw = {key: np.full(shape, np.nan) for key in FEATURE_NAMES}
    missing = {window: Counter() for window in WINDOWS}
    flat = {window: 0 for window in WINDOWS}
    for row, (day, symbols) in enumerate(normalized):
        t = ref.index.get_loc(day)
        for column, symbol in enumerate(symbols):
            for window in WINDOWS:
                first = t - window + 1
                if symbol not in ref.columns:
                    missing[window]["unknown_symbol"] += 1
                    continue
                if first < 0:
                    missing[window]["insufficient_window_history"] += 1
                    continue
                values = {}
                reason = None
                for key in ("adj", "close", "volume"):
                    value = panel[key][symbol].iloc[first:t + 1].to_numpy(dtype=float, copy=True)
                    if not (np.isfinite(value) & (value > 0.)).all():
                        reason = "invalid_" + key + "_window"
                        break
                    values[key] = value
                if reason:
                    missing[window][reason] += 1
                    continue
                result = _window_features(values["adj"], values["close"], values["volume"])
                if not np.isfinite(result).all():
                    missing[window]["nonfinite_calculation"] += 1
                    continue
                for kind, value in zip(KINDS, result):
                    raw[kind + str(window)][row, column] = value
                flat[window] += int(all(value == 0. for value in result))
    for key in FEATURE_NAMES:
        raw[key + "_valid"] = np.isfinite(raw[key])
    audit = {
        "events": len(normalized), "symbols_per_event": width,
        "windows_price_observations": list(WINDOWS), "half_life_sessions": HALF_LIFE,
        "first_used_offsets": {str(window): -(window - 1) for window in WINDOWS},
        "last_used_offset": 0, "signal_session_included": True,
        "long_core_modified": False, "trading_eligibility_modified": False,
        "cross_sectional_ranking_applied": False, "all_asset_types_share_formula": True,
        "adjusted_unit_volume_formula": "q_s=stored_close_s*stored_volume_s/adjusted_close_s",
        "decay_formula": "h_s=2**(-(signal_session-s)/63)",
        "reference_formula": "sum(h_s*q_s*A_s)/sum(h_s*q_s); control replaces q_s with 1",
        "gap_formula": "log(A_signal/reference)",
        "upper_mass_formula": "sum(normalized_weights*I[A_s>A_signal*(1+1e-12)])",
        "upper_price_relative_tolerance": PRICE_RELATIVE_TOLERANCE,
        "feature_finite_cells": {key: int(raw[key + "_valid"].sum()) for key in FEATURE_NAMES},
        "feature_missing_cells": {key: int((~raw[key + "_valid"]).sum()) for key in FEATURE_NAMES},
        "missing_reasons": {kind + str(window): dict(missing[window]) for window in WINDOWS for kind in KINDS},
        "flat_zero_cells": {str(window): flat[window] for window in WINDOWS},
        "controls_share_price_close_volume_mask": True,
        "zero_volume_policy": "Conservative missing: every window volume must be finite and strictly positive",
        "missing_policy": "Raw NaN and valid=False; caller applies exactly zero correction without changing eligibility",
        "price_basis": "Inherited adjusted close; cash-dividend effects are included in its total-return adjustment",
        "stored_input_basis": "Inherited panel close and volume; sources include Yahoo-style OHLCV and Tiingo archives, plus spliced early history",
        "split_consistency": "Invariant under inverse rescaling of stored close and volume with unchanged adjusted price; not independent certification of every feed's adjustments",
        "limitations": [
            "No historical free-float shares: this is not share turnover or survival-weighted holder inventory",
            "Adjusted-unit weights include dividend adjustments and are not literal shares held",
            "All daily volume is assigned to daily close, not measured intraday volume-at-price",
            "Historical input splices and vendor adjustment conventions are inherited, not re-certified here",
            "Signals are hypothetical supply-pressure proxies, not observed investor acquisition costs",
        ],
    }
    return raw, audit
