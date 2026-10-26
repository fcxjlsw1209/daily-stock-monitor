"""Causal overnight/intraday decomposition for isolated monthly research.

The caller owns the exchange-session calendar, universe, eligibility and ranking.
This module preserves missing values and never changes trading eligibility. Both
legs use the vendor's adjusted-close basis; their sum is the adjusted daily log
return, not a claim to reconstruct an independently observed cash dividend.
"""
from typing import Mapping, Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "overnight_log_sum", "intraday_log_sum", "session_log_spread",
    "overnight_sign_mean", "intraday_sign_mean",
)
SIGN_ZERO_TOLERANCE = 1e-12


def _validated_inputs(panel: Mapping[str, pd.DataFrame]):
    required = ("adj", "open", "close")
    for key in required:
        if key not in panel:
            raise KeyError("Missing required price panel: " + key)
        if not isinstance(panel[key], pd.DataFrame):
            raise TypeError(key + " must be a pandas DataFrame")
    ref = panel["adj"]
    if not isinstance(ref.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex of exchange sessions")
    if not ref.index.is_unique or not ref.index.is_monotonic_increasing:
        raise ValueError("Price sessions must be unique and increasing")
    if not ref.columns.is_unique:
        raise ValueError("Security columns must be unique")
    clean, invalid = {}, {}
    for key in required:
        frame = panel[key]
        if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
            raise ValueError(key + " must have exactly the adj index and columns")
        frame = frame.astype(float, copy=True)
        finite = np.isfinite(frame)
        positive = finite & frame.gt(0.)
        invalid[key] = {
            "missing": int(frame.isna().to_numpy().sum()),
            "nonfinite_nonmissing": int((~finite & frame.notna()).to_numpy().sum()),
            "nonpositive_finite": int((finite & frame.le(0.)).to_numpy().sum()),
        }
        clean[key] = frame.where(positive)
    return clean, invalid


def _identity_error(left: pd.DataFrame, right: pd.DataFrame):
    valid = np.isfinite(left) & np.isfinite(right)
    values = (left - right).where(valid).to_numpy()
    finite_values = values[np.isfinite(values)]
    maximum = float(np.max(np.abs(finite_values))) if finite_values.size else None
    return int(valid.to_numpy().sum()), maximum


def decompose_sessions(panel: Mapping[str, pd.DataFrame]) -> Tuple[dict, dict]:
    """Return adjusted-open and daily log-return frames, plus numeric audits.

    ``adjusted_open[t] = open[t] * adj[t] / close[t]``. Invalid or absent
    observations remain NaN, including a missing previous adjusted close for the
    overnight leg. A valid flat session has zero log return and zero sign.
    Adjustment-factor changes are reported without labelling them as verified
    splits or dividends; independently certified corporate actions are upstream.
    """
    prices, invalid = _validated_inputs(panel)
    adj, opening, close = (prices[k] for k in ("adj", "open", "close"))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        adjustment = adj / close
        adjusted_open = opening * adjustment
    adjustment = adjustment.where(np.isfinite(adjustment) & adjustment.gt(0.))
    adjusted_open = adjusted_open.where(np.isfinite(adjusted_open) & adjusted_open.gt(0.))
    log_adj = np.log(adj)
    total = log_adj - log_adj.shift(1)
    # The contemporaneous adjustment factor cancels within the session. Using
    # the identical raw operands makes an exactly flat open/close exactly zero;
    # subtracting separately rounded adjusted prices can instead produce 1e-15.
    intraday = (np.log(close) - np.log(opening)).where(adjusted_open.notna() & adj.notna())
    overnight = total - intraday
    count, error = _identity_error(overnight + intraday, total)
    if error is not None and error > 1e-11:
        raise ValueError("Daily adjusted return decomposition failed")
    factor_move = np.log(adjustment).diff()
    changed = factor_move.abs().gt(1e-10) & factor_move.notna()
    finite_raw = opening.notna() & adj.notna() & close.notna()
    return {
        "adjusted_open": adjusted_open,
        "adjustment_factor": adjustment,
        "overnight_log": overnight,
        "intraday_log": intraday,
        "total_log": total,
    }, {
        "sessions": len(adj.index),
        "securities": len(adj.columns),
        "invalid_inputs": invalid,
        "adjusted_open_unrepresentable": int((finite_raw & adjusted_open.isna()).to_numpy().sum()),
        "missing_overnight_cells": int(overnight.isna().to_numpy().sum()),
        "missing_intraday_cells": int(intraday.isna().to_numpy().sum()),
        "decomposition_cells_checked": count,
        "decomposition_max_abs_error": error,
        "adjustment_factor_changes": int(changed.to_numpy().sum()),
        "basis": "raw open multiplied by contemporaneous adjusted-close/raw-close ratio",
        "action_classification": "Factor changes alone do not certify a split or dividend",
        "missing_policy": "Preserve NaN; never substitute zero or change caller eligibility",
        "intraday_numerics": "log(raw close)-log(raw open); factor cancels, overnight=total-intraday",
    }


def build_session_features(panel: Mapping[str, pd.DataFrame], window: int = 210,
                           skip: int = 42) -> Tuple[dict, dict]:
    """Return five unranked frames using complete windows ending ``skip`` ago.

    At row t, the default contains returns t-251 through t-42 inclusive. Thus
    cumulative overnight plus intraday equals log(adj[t-42]/adj[t-252]).
    Persistence is mean(sign(log return)); up/flat/down count as +1/0/-1.
    Absolute log returns <=1e-12 count as numerical zero only for persistence.
    Full-window ``min_periods`` preserves every missing leg. No fill/forward-fill,
    security filtering, rolling eligibility test or cross-sectional rank occurs.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer")
    if isinstance(skip, bool) or not isinstance(skip, (int, np.integer)) or skip < 0:
        raise ValueError("skip must be a nonnegative integer")
    daily, audit = decompose_sessions(panel)
    overnight, intraday = daily["overnight_log"], daily["intraday_log"]
    aggregate = lambda z: z.rolling(window, min_periods=window).sum().shift(skip)
    signed = lambda z: np.sign(z).where(z.abs().gt(SIGN_ZERO_TOLERANCE), 0.).where(z.notna())
    on_sum, in_sum = aggregate(overnight), aggregate(intraday)
    frames = {
        "overnight_log_sum": on_sum,
        "intraday_log_sum": in_sum,
        "session_log_spread": on_sum - in_sum,
        "overnight_sign_mean": aggregate(signed(overnight)) / window,
        "intraday_sign_mean": aggregate(signed(intraday)) / window,
    }
    count, error = _identity_error(on_sum + in_sum, aggregate(daily["total_log"]))
    if error is not None and error > 1e-10:
        raise ValueError("Rolling adjusted return decomposition failed")
    audit.update(window=int(window), skip=int(skip),
                 rolling_decomposition_cells_checked=count,
                 rolling_decomposition_max_abs_error=error,
                 feature_missing_cells={k: int(v.isna().to_numpy().sum()) for k, v in frames.items()},
                 trading_eligibility_modified=False,
                 persistence="mean sign; abs(log return)<=1e-12 is numerical zero; missing remains NaN",
                 sign_zero_tolerance=SIGN_ZERO_TOLERANCE)
    return frames, audit
