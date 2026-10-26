"""Annual paired ridge forecasts isolating the addition of earnings features.

The caller supplies point-in-time features, forward targets, and eligibility.
Both arms use identical observations and weights. An unavailable forecast is
NaN; converting that into a zero portfolio correction belongs to the caller.
"""
import importlib.util
from pathlib import Path

import numpy as np


_HELPER_PATH = Path(__file__).resolve().parents[1] / "monthly_architecture_restart" / "residual_model.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("_earnings_ridge_helpers", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)
_fit_ridge = _HELPERS._fit_ridge
_predict = _HELPERS._predict
month_balanced_weights = _HELPERS.month_balanced_weights
_years_before = _HELPERS._years_before


def _days(values, name):
    result = np.asarray(values, dtype="datetime64[D]")
    if result.ndim != 1 or np.isnat(result).any():
        raise ValueError("%s must be a one-dimensional array of valid dates" % name)
    return result


def annual_forecasts(dates, ends, X_plain, X_earn, y, train_good,
                     event_dates, event_plain, event_earn, event_good,
                     alpha=10.0, half_life_months=24.0, min_months=18):
    """Return paired {plain, earn, available, audit} for decision events.

    The plain arm uses ``X_plain``; the earnings arm uses ``X_plain`` together
    with ``X_earn``. For signal year Y, both are frozen at December 31 of Y-1,
    using only labels completed by that date and signals strictly within the
    preceding five years. Training requires at least ``min_months`` distinct
    calendar months after the common eligibility/finite-value filter.

    A calendar month's rows have equal aggregate weight before age decay.
    Each model standardizes using its weighted training sample only. Defaults
    are fixed research choices; this routine performs no parameter selection.
    Both event feature blocks must be finite and ``event_good`` must be true.
    Insufficient history and invalid events leave both forecasts NaN and their
    shared availability false. There is no OOF or residual-target layer.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X_plain, X_earn = np.asarray(X_plain, float), np.asarray(X_earn, float)
    event_plain, event_earn = np.asarray(event_plain, float), np.asarray(event_earn, float)
    y, train_good = np.asarray(y, float), np.asarray(train_good, bool)
    event_good = np.asarray(event_good, bool)
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or y.shape != (n,) or train_good.shape != (n,):
        raise ValueError("training date, end, target, and eligibility lengths differ")
    if event_good.shape != (q,):
        raise ValueError("event eligibility length differs")
    for name, values, rows in (("X_plain", X_plain, n), ("X_earn", X_earn, n),
                               ("event_plain", event_plain, q),
                               ("event_earn", event_earn, q)):
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
            raise ValueError("%s must be a nonempty-feature [rows, features] matrix" % name)
    if X_plain.shape[1] != event_plain.shape[1] or X_earn.shape[1] != event_earn.shape[1]:
        raise ValueError("training and event feature dimensions differ")
    if (ends <= dates).any():
        raise ValueError("monthly label ends must be later than their signals")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(half_life_months) or half_life_months < 0:
        raise ValueError("half_life_months must be finite and nonnegative")
    if isinstance(min_months, (bool, np.bool_)) or not isinstance(min_months, (int, np.integer)) or min_months < 1:
        raise ValueError("min_months must be a positive integer")

    usable = (train_good & np.isfinite(y) & np.isfinite(X_plain).all(axis=1)
              & np.isfinite(X_earn).all(axis=1))
    valid_events = (event_good & np.isfinite(event_plain).all(axis=1)
                    & np.isfinite(event_earn).all(axis=1))
    plain, earn = np.full(q, np.nan), np.full(q, np.nan)
    available = np.zeros(q, dtype=bool)
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    annual_audit = []
    for year in np.unique(years):
        in_year = years == year
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        start = _years_before(cutoff)
        train = usable & (ends <= cutoff) & (dates > start)
        n_months = int(len(np.unique(dates[train].astype("datetime64[M]"))))
        predict = in_year & valid_events
        enough_history = n_months >= min_months
        record = {
            "year": int(year), "cutoff": str(cutoff),
            "lookback_exclusive_start": str(start),
            "training_rows": int(train.sum()), "training_months": n_months,
            "latest_label_end": str(ends[train].max()) if train.any() else None,
            "first_training_signal": str(dates[train].min()) if train.any() else None,
            "latest_training_signal": str(dates[train].max()) if train.any() else None,
            "eligible_event_rows": int(predict.sum()),
            "prediction_rows": 0,
            "invalid_or_ineligible_event_rows": int(in_year.sum() - predict.sum()),
            "status": "cold_start" if not enough_history else "no_valid_events",
            "plain_and_earn_share_rows": True,
        }
        if enough_history and predict.any():
            weights = month_balanced_weights(dates[train], cutoff, half_life_months)
            train_plain, target = X_plain[train], y[train]
            train_combined = np.column_stack((train_plain, X_earn[train]))
            event_combined = np.column_stack((event_plain[predict], event_earn[predict]))
            plain_model = _fit_ridge(train_plain, target, weights, alpha)
            earn_model = _fit_ridge(train_combined, target, weights, alpha)
            plain[predict] = _predict(plain_model, event_plain[predict])
            earn[predict] = _predict(earn_model, event_combined)
            available[predict] = True
            record.update(status="fit", prediction_rows=int(predict.sum()),
                          weight_sum=float(weights.sum()))
        annual_audit.append(record)
    audit = {
        "method": "annual_frozen_plain_vs_plain_plus_earnings_ridge",
        "alpha": float(alpha), "half_life_months": float(half_life_months),
        "min_months": int(min_months), "lookback_years": 5,
        "weighting": "equal_month_total_then_age_decay_normalized_to_row_count",
        "scale_fit": "training_rows_only",
        "plain_features": int(X_plain.shape[1]),
        "earn_features": int(X_plain.shape[1] + X_earn.shape[1]),
        "plain_and_earn_share_rows": True,
        "plain_and_earn_share_availability": True,
        "unavailable_prediction": "NaN",
        "linear_algebra": "einsum_optimize_false_with_finite_guards",
        "target": "caller_supplied_forward_return",
        "annual": annual_audit,
    }
    return {"plain": plain, "earn": earn, "available": available, "audit": audit}
