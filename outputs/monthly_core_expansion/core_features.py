"""Causal raw trend scores, all on 210 simple returns ending at t-42.

No ranking, security-type exception, eligibility filtering or trading is done
here. The caller supplies the exact signal-time candidate list. Missing price
windows stay NaN. Missing auxiliary history falls back to ordinary Sharpe,
with separate flags and reasons, without changing that candidate list.
"""
from collections import Counter
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


WINDOW = 210
SKIP = 42
HAC_LAGS = 5
BETA_WINDOW = 252
STD_TOLERANCE = 1e-12
RESIDUAL_ZERO_TOLERANCE = 1e-12
MARKET = "SPY"
FEATURE_NAMES = ("ordinary_sharpe", "hac_sharpe", "market_residual_sharpe")


def _validate(panel):
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


def _market_beta(returns):
    """Beta for return u uses only returns u-252 through u-1, inclusive.

    Both rolling moments use 252 complete observations and sample covariance /
    variance. No intercept is subtracted when beta later forms the residual.
    """
    empty = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    if MARKET not in returns.columns:
        return empty, pd.Series(np.nan, index=returns.index)
    market = returns[MARKET]
    market_var = market.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var(ddof=1).shift(1)
    covariance = returns.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(market, ddof=1).shift(1)
    safe_var = market_var.where(np.isfinite(market_var) & (market_var > STD_TOLERANCE ** 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        beta = covariance.div(safe_var, axis=0)
    return beta.where(np.isfinite(beta)), market_var


def _ordinary(returns):
    deviation = returns.std(ddof=1)
    if deviation <= STD_TOLERANCE:
        return 0., True
    return float(returns.mean() / deviation), False


def _hac_variance(returns):
    centered = returns - returns.mean()
    value = float(np.dot(centered, centered) / len(returns))
    for lag in range(1, HAC_LAGS + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / len(returns))
        value += 2. * (1. - lag / (HAC_LAGS + 1.)) * gamma
    return value


def build_features(panel: Mapping[str, pd.DataFrame],
                   events: Sequence[Tuple[object, Sequence[str]]]) -> Tuple[dict, dict]:
    """Return raw scores and same-shaped validity/fallback masks plus audit.

    All scores are unannualized. Ordinary uses mean / sample standard deviation.
    HAC replaces its denominator with the square root of Bartlett lag-5 long
    variance, with every autocovariance divided by 210 (including lag zero).
    Residual uses r_i,u - beta_i,u * r_SPY,u, with beta strictly lagged one day,
    followed by the ordinary mean/sample-std ratio of its 210-return window.

    Zero ordinary variance is a documented zero score, not an infinite score.
    A numerically zero residual, including SPY versus itself, also scores zero
    (zero market excess, not a neutral percentile). Nonzero constant residuals
    instead fall back to ordinary Sharpe. ETFs receive exactly the same logic.
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
    for key in FEATURE_NAMES:
        raw[key + "_fallback"] = np.zeros(shape, dtype=bool)
    missing = {key: Counter() for key in FEATURE_NAMES}
    fallback = {key: Counter() for key in FEATURE_NAMES}
    degenerate = Counter()

    # Sanitize prices before pct_change and never fill absent observations.
    clean = ref.where(np.isfinite(ref) & (ref > 0.))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        returns = clean.pct_change(fill_method=None)
    returns = returns.where(np.isfinite(returns))
    beta, market_var = _market_beta(returns)
    residuals = returns.sub(beta.mul(returns[MARKET], axis=0)) if MARKET in returns else returns * np.nan

    def use_fallback(key, row, column, reason):
        raw[key][row, column] = raw["ordinary_sharpe"][row, column]
        raw[key + "_fallback"][row, column] = True
        fallback[key][reason] += 1

    for row, (day, symbols) in enumerate(normalized):
        t = ref.index.get_loc(day)
        first, last = t - SKIP - WINDOW, t - SKIP
        for column, symbol in enumerate(symbols):
            reason = None
            if symbol not in ref.columns:
                reason = "unknown_symbol"
            elif first < 0:
                reason = "insufficient_adjusted_price_history"
            else:
                prices = clean[symbol].iloc[first:last + 1].to_numpy()
                vals = returns[symbol].iloc[first + 1:last + 1].to_numpy()
                if not (np.isfinite(prices).all() and np.isfinite(vals).all()):
                    reason = "invalid_adjusted_price_window"
            if reason:
                for key in FEATURE_NAMES:
                    missing[key][reason] += 1
                continue
            ordinary, constant = _ordinary(vals)
            if not np.isfinite(ordinary):
                for key in FEATURE_NAMES:
                    missing[key]["nonfinite_ordinary_score"] += 1
                continue
            raw["ordinary_sharpe"][row, column] = ordinary
            degenerate["ordinary_zero_standard_deviation"] += int(constant)
            variance = _hac_variance(vals)
            if not np.isfinite(variance) or variance <= 0.:
                use_fallback("hac_sharpe", row, column, "nonpositive_or_nonfinite_hac_variance")
            elif constant:
                # Avoid numerical artifacts for mathematically constant prices /
                # returns; the ordinary tolerance has already classified them.
                raw["hac_sharpe"][row, column] = 0.
                degenerate["hac_zero_standard_deviation"] += 1
            else:
                value = float(vals.mean() / np.sqrt(variance))
                if np.isfinite(value):
                    raw["hac_sharpe"][row, column] = value
                else:
                    use_fallback("hac_sharpe", row, column, "nonfinite_hac_score")

            key = "market_residual_sharpe"
            window = slice(first + 1, last + 1)
            if MARKET not in returns:
                use_fallback(key, row, column, "missing_market_symbol")
                continue
            if first < BETA_WINDOW:
                use_fallback(key, row, column, "insufficient_beta_history")
                continue
            variance_window = market_var.iloc[window].to_numpy()
            if not np.isfinite(variance_window).all():
                use_fallback(key, row, column, "invalid_market_beta_history")
                continue
            if np.any(variance_window <= STD_TOLERANCE ** 2):
                use_fallback(key, row, column, "zero_or_tiny_market_variance")
                continue
            b = beta[symbol].iloc[window].to_numpy()
            if not np.isfinite(b).all():
                use_fallback(key, row, column, "invalid_asset_beta_history")
                continue
            res = residuals[symbol].iloc[window].to_numpy()
            if not np.isfinite(res).all():
                use_fallback(key, row, column, "invalid_residual_return_window")
                continue
            if np.max(np.abs(res)) <= RESIDUAL_ZERO_TOLERANCE:
                raw[key][row, column] = 0.
                degenerate["zero_market_residual"] += 1
                continue
            deviation = res.std(ddof=1)
            if deviation <= STD_TOLERANCE:
                use_fallback(key, row, column, "nonzero_constant_market_residual")
                continue
            value = float(res.mean() / deviation)
            if np.isfinite(value):
                raw[key][row, column] = value
            else:
                use_fallback(key, row, column, "nonfinite_residual_score")

    for key in FEATURE_NAMES:
        raw[key + "_valid"] = np.isfinite(raw[key])
    audit = {
        "events": len(normalized), "symbols_per_event": width,
        "window_returns": WINDOW, "skip_sessions": SKIP,
        "price_path_observations": WINDOW + 1,
        "first_required_ordinary_price_offset": -(WINDOW + SKIP),
        "first_required_residual_price_offset": -(WINDOW + SKIP + BETA_WINDOW),
        "last_used_offset": -SKIP,
        "ordinary_formula": "mean(simple returns) / sample standard deviation, ddof=1; unannualized",
        "hac_lags": HAC_LAGS, "hac_kernel": "Bartlett: weight(lag)=1-lag/6",
        "hac_autocovariance_denominator": WINDOW,
        "hac_normalization_note": "gamma0 uses population denominator 210, versus ordinary sample denominator 209",
        "market_symbol": MARKET, "beta_window_returns": BETA_WINDOW, "beta_lag_sessions": 1,
        "residual_formula": "asset return - prior252 beta * same-day SPY return; no intercept removed",
        "std_tolerance": STD_TOLERANCE, "market_variance_tolerance": STD_TOLERANCE ** 2,
        "residual_zero_max_abs_tolerance": RESIDUAL_ZERO_TOLERANCE,
        "feature_finite_cells": {key: int(raw[key + "_valid"].sum()) for key in FEATURE_NAMES},
        "feature_missing_cells": {key: int((~raw[key + "_valid"]).sum()) for key in FEATURE_NAMES},
        "feature_fallback_cells": {key: int(raw[key + "_fallback"].sum()) for key in FEATURE_NAMES},
        "missing_reasons": {key: dict(value) for key, value in missing.items()},
        "fallback_reasons": {key: dict(value) for key, value in fallback.items()},
        "degenerate_counts": dict(degenerate),
        "missing_policy": "Missing 211-price formation stays NaN; auxiliary failure falls back to same-window ordinary",
        "zero_score_policy": "Zero ordinary variance or numerical zero market residual gives raw zero, not a neutral percentile",
        "trading_eligibility_modified": False, "cross_sectional_ranking_applied": False,
        "all_asset_types_share_formula": True,
    }
    return raw, audit
