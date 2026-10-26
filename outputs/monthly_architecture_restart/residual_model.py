"""Causal annual direct-versus-residual monthly forecasts.

The residual target is computed from rolling historical base-model predictions,
not an in-sample refit. Neither model is tuned here. All targets and feature
construction belong to the caller; trading eligibility must be known at signal
time and must never be inferred from future price completeness.
"""
from datetime import date

import numpy as np


def _days(values, name):
    result = np.asarray(values, dtype="datetime64[D]")
    if result.ndim != 1 or np.isnat(result).any():
        raise ValueError("%s must be a one-dimensional array of valid dates" % name)
    return result


def _years_before(value, years=5):
    value = date.fromisoformat(str(np.datetime64(value, "D")))
    try:
        result = value.replace(year=value.year - years)
    except ValueError:  # February 29.
        result = value.replace(year=value.year - years, day=28)
    return np.datetime64(result.isoformat(), "D")


def month_balanced_weights(dates, cutoff, half_life_months=24.0):
    """Equal aggregate weight per calendar month, followed by causal age decay.

    The weights sum to the row count, as in sample-weighted Ridge. With decay
    disabled (half_life_months=0), a month with more securities has no larger
    total weight than a month with fewer securities.
    """
    dates = _days(dates, "dates")
    cutoff = np.datetime64(cutoff, "D")
    if np.isnat(cutoff) or (dates > cutoff).any():
        raise ValueError("weight cutoff must be on or after all training dates")
    if not np.isfinite(half_life_months) or half_life_months < 0:
        raise ValueError("half_life_months must be finite and nonnegative")
    if not len(dates):
        return np.empty(0, dtype=float)
    months, inverse, counts = np.unique(dates.astype("datetime64[M]"),
                                        return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse].astype(float)
    if half_life_months:
        # Calendar-month ages give every observation in one month the same decay.
        age = (cutoff.astype("datetime64[M]").astype(int) - months.astype(int))
        weight *= np.exp2(-age[inverse] / half_life_months)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("age weighting underflowed")
    return weight * (len(weight) / total)


def _require_finite(name, values):
    if not np.isfinite(values).all():
        raise ValueError("nonfinite %s in monthly residual model" % name)


def _fit_ridge(X, y, weights, alpha):
    """Weighted fit; explicitly avoid this host's warning-prone BLAS matmul."""
    for name, values in (("features", X), ("target", y), ("weights", weights)):
        _require_finite(name, values)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Ridge weights must be nonnegative with positive total")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = np.average(X, axis=0, weights=weights)
        centered = X - mean
        scale = np.sqrt(np.average(centered * centered, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = centered / scale
        intercept = float(np.average(y, weights=weights))
        lhs = np.einsum("ni,n,nj->ij", z, weights, z, optimize=False)
        lhs += alpha * np.eye(X.shape[1])
        rhs = np.einsum("ni,n,n->i", z, weights, y - intercept, optimize=False)
    for name, values in (("means", mean), ("scales", scale),
                         ("normal-equation lhs", lhs), ("normal-equation rhs", rhs),
                         ("intercept", intercept)):
        _require_finite(name, values)
    try:
        coefficient = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    _require_finite("coefficients", coefficient)
    return mean, scale, coefficient, intercept


def _predict(model, X):
    mean, scale, coefficient, intercept = model
    for name, values in (("prediction features", X), ("prediction means", mean),
                         ("prediction scales", scale), ("prediction coefficients", coefficient),
                         ("prediction intercept", intercept)):
        _require_finite(name, values)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        z = (X - mean) / scale
        prediction = np.einsum("ni,i->n", z, coefficient, optimize=False) + intercept
    _require_finite("predictions", prediction)
    return prediction


def _month_count(dates):
    return int(len(np.unique(dates.astype("datetime64[M]"))))


def _oof_base_predictions(dates, ends, X, y, eligible, alpha,
                          half_life_months, min_months):
    """Predict each historical signal using earlier, completed labels only."""
    prediction = np.full(len(y), np.nan, dtype=float)
    usable = eligible & np.isfinite(y) & np.isfinite(X).all(axis=1)
    months = dates.astype("datetime64[M]")
    audit = []
    for signal_date in np.unique(dates):
        current = (dates == signal_date) & eligible & np.isfinite(X).all(axis=1)
        prior = (usable & (months < signal_date.astype("datetime64[M]"))
                 & (ends <= signal_date) & (dates > _years_before(signal_date)))
        n_months = _month_count(dates[prior])
        record = {
            "signal_date": str(signal_date),
            "cutoff": str(signal_date),
            "training_rows": int(prior.sum()),
            "training_months": n_months,
            "latest_training_signal": str(dates[prior].max()) if prior.any() else None,
            "latest_training_end": str(ends[prior].max()) if prior.any() else None,
            "prediction_rows": int(current.sum()),
            "status": "fit" if n_months >= min_months else "cold_start",
            "base_predictor_origin": "prior_completed_months_only",
        }
        if n_months >= min_months and current.any():
            weights = month_balanced_weights(dates[prior], signal_date,
                                              half_life_months)
            model = _fit_ridge(X[prior], y[prior], weights, alpha)
            prediction[current] = _predict(model, X[current])
        audit.append(record)
    return prediction, audit


def annual_forecasts(dates, ends, X_base, X_aux, y, eligible,
                     event_dates, event_base, event_aux, event_eligible,
                     alpha=10.0, half_life_months=24.0, min_months=18):
    """Return {direct, residual, audit} for flattened monthly decision events.

    ``direct`` forecasts y using auxiliary features. ``residual`` forecasts the
    difference between y and a base forecast available at its historical signal.
    It is an auxiliary correction, not a base-plus-correction prediction.

    Both annual models share exactly the same OOF-available rows and weights.
    A year's models are frozen using labels completed by the preceding Dec 31
    and training signals in the preceding five calendar years. Cold starts,
    ineligible events, and nonfinite event features receive zero correction.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X_base, X_aux = np.asarray(X_base, float), np.asarray(X_aux, float)
    event_base = np.asarray(event_base, float)
    event_aux = np.asarray(event_aux, float)
    y, eligible = np.asarray(y, float), np.asarray(eligible, bool)
    event_eligible = np.asarray(event_eligible, bool)
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or y.shape != (n,) or eligible.shape != (n,):
        raise ValueError("training date, end, target, and eligibility lengths differ")
    if event_eligible.shape != (q,):
        raise ValueError("event eligibility length differs")
    for name, values, rows in (("X_base", X_base, n), ("X_aux", X_aux, n),
                               ("event_base", event_base, q),
                               ("event_aux", event_aux, q)):
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
            raise ValueError("%s must be a nonempty-feature [rows, features] matrix" % name)
    if X_base.shape[1] != event_base.shape[1] or X_aux.shape[1] != event_aux.shape[1]:
        raise ValueError("training and event feature dimensions differ")
    if (ends <= dates).any():
        raise ValueError("monthly label ends must be later than their signals")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not isinstance(min_months, (int, np.integer)) or min_months < 1:
        raise ValueError("min_months must be a positive integer")
    if not np.isfinite(half_life_months) or half_life_months < 0:
        raise ValueError("half_life_months must be finite and nonnegative")

    oof, oof_audit = _oof_base_predictions(
        dates, ends, X_base, y, eligible, alpha, half_life_months, min_months)
    residual_target = y - oof
    usable = (eligible & np.isfinite(y) & np.isfinite(oof)
              & np.isfinite(X_aux).all(axis=1))
    direct, residual = np.zeros(q), np.zeros(q)
    valid_events = (event_eligible & np.isfinite(event_base).all(axis=1)
                    & np.isfinite(event_aux).all(axis=1))
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    annual_audit = []
    for year in np.unique(years):
        in_year = years == year
        cutoff = min(np.datetime64("%04d-12-31" % (year - 1), "D"),
                     event_dates[in_year].min())
        train = usable & (ends <= cutoff) & (dates > _years_before(cutoff))
        n_months = _month_count(dates[train])
        predict = in_year & valid_events
        record = {
            "year": int(year), "cutoff": str(cutoff),
            "lookback_exclusive_start": str(_years_before(cutoff)),
            "training_rows": int(train.sum()), "residual_months": n_months,
            "latest_label_end": str(ends[train].max()) if train.any() else None,
            "first_training_signal": str(dates[train].min()) if train.any() else None,
            "latest_training_signal": str(dates[train].max()) if train.any() else None,
            "prediction_rows": int(predict.sum()),
            "invalid_or_ineligible_event_rows": int(in_year.sum() - predict.sum()),
            "status": "fit" if n_months >= min_months else "cold_start",
            "base_predictor_origin": "rolling_oof_before_each_historical_signal",
            "direct_and_residual_share_rows": True,
        }
        if n_months >= min_months and predict.any():
            weights = month_balanced_weights(dates[train], cutoff, half_life_months)
            direct_model = _fit_ridge(X_aux[train], y[train], weights, alpha)
            residual_model = _fit_ridge(X_aux[train], residual_target[train],
                                        weights, alpha)
            direct[predict] = _predict(direct_model, event_aux[predict])
            residual[predict] = _predict(residual_model, event_aux[predict])
            record["weight_sum"] = float(weights.sum())
        annual_audit.append(record)
    audit = {
        "method": "annual_auxiliary_ridge_with_historical_oof_base_residuals",
        "alpha": float(alpha), "half_life_months": float(half_life_months),
        "min_months": int(min_months), "lookback_years": 5,
        "weighting": "equal_month_total_then_age_decay_normalized_to_row_count",
        "scale_fit": "training_rows_only", "residual_is_auxiliary_correction": True,
        "linear_algebra": "einsum_optimize_false_with_finite_guards",
        "target": "caller_supplied_next_calendar_month_cross_section_centered_return",
        "oof_rows": int(np.isfinite(oof).sum()),
        "annual": annual_audit, "oof_months": oof_audit,
    }
    return {"direct": direct, "residual": residual, "audit": audit}
