"""Annual honest Ridge-residual trees and historical prediction-error shrinkage.

All four forecasts retain the same fixed ordinary Ridge. Trees learn only
annual out-of-year residuals; calibration sees only matured predictions and
payoffs. No returns selection, portfolio selection or feature expansion occurs.
"""
from pathlib import Path
import importlib.util

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "monthly_architecture_restart/residual_model.py"
_SPEC = importlib.util.spec_from_file_location("conditional70_residual_numerics", _PATH)
NUM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(NUM)

ARMS = ("ridge_residual_global", "ridge_residual_weak", "ridge_cal_global", "ridge_cal_state")
ALPHA = 10.
HALF_LIFE = 24.
FEATURES = 10
GROUP_MIN_MONTHS = 12
GROUP_MIN_YEARS = 2
GROUP_MIN_ROWS = 240
GROUP_MIN_EFFECTIVE_MONTHS = 8.
CAL_PRIOR_MONTHS = 6.
TREE_PARAMS = dict(max_depth=2, max_leaf_nodes=4, max_iter=100, learning_rate=.05,
                   min_samples_leaf=40, l2_regularization=1., random_state=1729,
                   early_stopping=False)


def _booleans(values, n, name):
    value = np.asarray(values)
    if value.shape != (n,) or value.dtype.kind != "b":
        raise ValueError(name + " must be a flat boolean array with matching rows")
    return value


def _validate(dates, ends, X, y, good, event_dates, eventX, event_good, train_weak, event_weak):
    dates, ends = NUM._days(dates, "dates"), NUM._days(ends, "ends")
    event_dates = NUM._days(event_dates, "event_dates")
    n, q = len(dates), len(event_dates)
    X, y, eventX = np.asarray(X, float), np.asarray(y, float), np.asarray(eventX, float)
    if ends.shape != (n,) or y.shape != (n,) or X.shape != (n, FEATURES) or eventX.shape != (q, FEATURES):
        raise ValueError("Training/event rows must match and have exactly10 features")
    if ((ends <= dates).any() or
            ((ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)) != 1).any()):
        raise ValueError("Targets must end in the next calendar month")
    good = _booleans(good, n, "good")
    event_good = _booleans(event_good, q, "event_good")
    train_weak = _booleans(train_weak, n, "train_weak")
    event_weak = _booleans(event_weak, q, "event_weak")
    for month in np.unique(dates.astype("datetime64[M]")):
        mask = dates.astype("datetime64[M]") == month
        if (len(np.unique(dates[mask])) != 1 or len(np.unique(ends[mask])) != 1
                or len(np.unique(train_weak[mask])) != 1):
            raise ValueError("Each training month must have one signal, end and market state")
    for day in np.unique(event_dates):
        if len(np.unique(event_weak[event_dates == day])) != 1:
            raise ValueError("Each event date must have one market state")
    return dates, ends, X, y, good, event_dates, eventX, event_good, train_weak, event_weak


def _raw_month_weights(dates, cutoff):
    """A month's total is its age decay, preserving prior-month units."""
    dates = NUM._days(dates, "dates")
    cutoff = np.datetime64(cutoff, "D")
    if (dates > cutoff).any():
        raise ValueError("Cannot weight future training dates")
    if not len(dates):
        return np.empty(0)
    months, inverse, counts = np.unique(dates.astype("datetime64[M]"), return_inverse=True, return_counts=True)
    age = cutoff.astype("datetime64[M]").astype(int) - months.astype(int)
    return np.exp2(-age[inverse] / HALF_LIFE) / counts[inverse]


def _group_readiness(dates, cutoff):
    dates = np.asarray(dates, dtype="datetime64[D]")
    weights = _raw_month_weights(dates, cutoff)
    months, inverse = np.unique(dates.astype("datetime64[M]"), return_inverse=True)
    totals = np.bincount(inverse, weights=weights) if len(dates) else np.empty(0)
    kish = float(totals.sum() ** 2 / np.sum(totals ** 2)) if len(totals) else 0.
    years = len(np.unique(dates.astype("datetime64[Y]")))
    ready = (len(months) >= GROUP_MIN_MONTHS and years >= GROUP_MIN_YEARS
             and len(dates) >= GROUP_MIN_ROWS and kish >= GROUP_MIN_EFFECTIVE_MONTHS)
    return dict(ready=bool(ready), months=len(months), signal_years=years, rows=len(dates),
                effective_months=kish, decayed_month_mass=float(weights.sum()))


def _ridge_predictions(dates, ends, X, y, usable, event_dates, eventX, event_good, min_months):
    """Annual frozen predictions, including when events are historical rows."""
    prediction = np.full(len(event_dates), np.nan)
    available = np.zeros(len(event_dates), dtype=bool)
    event_year = event_dates.astype("datetime64[Y]").astype(int) + 1970
    valid = event_good & np.isfinite(eventX).all(axis=1)
    logs = []
    for year in np.unique(event_year):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        start = NUM._years_before(cutoff)
        train = usable & (dates > start) & (ends <= cutoff)
        events = valid & (event_year == year)
        months = NUM._month_count(dates[train])
        ready = months >= min_months
        row = dict(year=int(year), cutoff=str(cutoff), exclusive_start=str(start),
                   training_rows=int(train.sum()), training_months=months,
                   latest_label_end=str(ends[train].max()) if train.any() else None,
                   prediction_rows=int(events.sum()), status="fit" if ready and events.any() else "cold_start" if not ready else "no_events")
        if ready and events.any():
            weights = NUM.month_balanced_weights(dates[train], cutoff, HALF_LIFE)
            fitted = NUM._fit_ridge(X[train], y[train], weights, ALPHA)
            prediction[events] = NUM._predict(fitted, eventX[events])
            available[events] = True
        logs.append(row)
    return prediction, available, logs


def _fit_tree(X, target, weights):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # The workspace already bundles threadpoolctl with sklearn. Limit only this
    # computation to prevent tiny annual fits spawning excessive OpenMP threads.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return HistGradientBoostingRegressor(**TREE_PARAMS).fit(X, target, sample_weight=weights)


def _tree_predict(model, X):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return model.predict(X)


def calibration_slope(prediction, target, dates, cutoff):
    """A NaN slope denotes missing calibration evidence, never perfect errors."""
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    dates = NUM._days(dates, "calibration_dates")
    if p.shape != dates.shape or y.shape != dates.shape:
        raise ValueError("Calibration row counts differ")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Calibration needs finite actual OOF predictions and labels")
    audit = _group_readiness(dates, cutoff)
    audit.update(slope=None, reason="insufficient_history")
    if not audit["ready"]:
        return np.nan, audit
    weights = _raw_month_weights(dates, cutoff)
    square_sum = float(np.sum(weights * p * p))
    square_mean = square_sum / weights.sum()
    if not np.isfinite(square_mean) or square_mean <= 1e-20:
        audit.update(ready=False, reason="zero_or_tiny_prediction_second_moment")
        return np.nan, audit
    numerator = float(np.sum(weights * p * y))
    denominator = square_sum + CAL_PRIOR_MONTHS * square_mean
    slope = float(np.clip(numerator / denominator, 0., 1.))
    audit.update(slope=slope, reason="fit", prediction_second_moment=square_mean,
                 prior_months=CAL_PRIOR_MONTHS, numerator=numerator, denominator=denominator)
    return slope, audit


def annual_forecasts(dates, ends, X, y, good, event_dates, eventX, event_good,
                     train_weak, event_weak, min_global_months=18,
                     min_weak_months=6, min_weak_rows=200):
    """Return ``(forecasts, audit)`` with four arms and ``available_<arm>``.

    The two optional weak thresholds are signature compatibility arguments;
    residual/calibration readiness always uses the registered12months/2years/
    240rows/Kish8 rule. Annual Ridge uses ``min_global_months`` (default18).
    Unavailable raw Ridge is NaN. Unavailable trees add zero; unavailable
    calibration falls back to known global slope, then identity slope1.
    """
    if (isinstance(min_global_months, (bool, np.bool_))
            or not isinstance(min_global_months, (int, np.integer)) or min_global_months < 1):
        raise ValueError("min_global_months must be a positive integer")
    (dates, ends, X, y, good, event_dates, eventX, event_good,
     train_weak, event_weak) = _validate(dates, ends, X, y, good, event_dates, eventX, event_good, train_weak, event_weak)
    usable = good & np.isfinite(y) & np.isfinite(X).all(axis=1)
    raw, raw_available, raw_audit = _ridge_predictions(dates, ends, X, y, usable,
        event_dates, eventX, event_good, min_global_months)
    # Annual historical predictions use all finite features as query rows;
    # future label completeness is used only to admit matured residual targets.
    oof, oof_available, oof_audit = _ridge_predictions(dates, ends, X, y, usable,
        dates, X, np.isfinite(X).all(axis=1), min_global_months)
    residual_good = usable & oof_available & np.isfinite(oof)
    residual = np.full(len(y), np.nan)
    residual[residual_good] = y[residual_good] - oof[residual_good]
    out = {arm: raw.copy() for arm in ARMS}
    out.update({"available_" + arm: raw_available.copy() for arm in ARMS})
    event_year = event_dates.astype("datetime64[Y]").astype(int) + 1970
    annual = []
    for year in np.unique(event_year):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        train = residual_good & (dates > NUM._years_before(cutoff)) & (ends <= cutoff)
        weak_train = train & train_weak
        events = raw_available & (event_year == year)
        weak_events = events & event_weak
        weak_ready = _group_readiness(dates[weak_train], cutoff)
        row = dict(year=int(year), cutoff=str(cutoff), residual_rows=int(train.sum()),
                   latest_residual_label_end=str(ends[train].max()) if train.any() else None,
                   tree_shared_weak_readiness=weak_ready,
                   tree_global_training=_group_readiness(dates[train], cutoff),
                   tree_status="insufficient_weak_history", tree_prediction_rows=int(weak_events.sum()))
        if weak_ready["ready"] and weak_events.any():
            for arm, mask in (("ridge_residual_global", train), ("ridge_residual_weak", weak_train)):
                weights = NUM.month_balanced_weights(dates[mask], cutoff, HALF_LIFE)
                fitted = _fit_tree(X[mask], residual[mask], weights)
                delta = _tree_predict(fitted, eventX[weak_events])
                if not np.isfinite(delta).all():
                    raise ValueError("Nonfinite residual tree forecast")
                out[arm][weak_events] = raw[weak_events] + delta
            row["tree_status"] = "fit"
        elif weak_ready["ready"]:
            row["tree_status"] = "ready_no_weak_events"
        global_slope, global_log = calibration_slope(oof[train], y[train], dates[train], cutoff)
        b_global = global_slope if np.isfinite(global_slope) else 1.
        out["ridge_cal_global"][events] = b_global * raw[events]
        row["calibration_global"] = global_log
        row["calibration_states"] = {}
        for state in (False, True):
            mask = train & (train_weak == state)
            slope, log = calibration_slope(oof[mask], y[mask], dates[mask], cutoff)
            known = np.isfinite(slope)
            b_state = slope if known else b_global
            selected = events & (event_weak == state)
            out["ridge_cal_state"][selected] = b_state * raw[selected]
            log.update(applied_slope=float(b_state), fallback="none" if known else "global" if np.isfinite(global_slope) else "identity")
            row["calibration_states"]["weak" if state else "normal"] = log
        global_log["applied_slope"] = float(b_global)
        global_log["fallback"] = "none" if np.isfinite(global_slope) else "identity"
        annual.append(row)
    for arm in ARMS:
        if not np.isfinite(out[arm][raw_available]).all() or np.isfinite(out[arm][~raw_available]).any():
            raise ValueError("Forecast/availability mismatch for " + arm)
    audit = dict(method="annual_honest_ridge_residual_trees_and_error_calibration",
        source_paths=[str(_PATH)], feature_count=FEATURES, alpha=ALPHA, half_life_months=HALF_LIFE,
        lookback_years=5, min_ridge_months=int(min_global_months), tree_parameters=TREE_PARAMS,
        group_readiness=dict(months=GROUP_MIN_MONTHS, signal_years=GROUP_MIN_YEARS,
                             rows=GROUP_MIN_ROWS, effective_months=GROUP_MIN_EFFECTIVE_MONTHS),
        compatible_weak_threshold_arguments_used=False,
        prior_dec31_mature_labels_only=True, oof_schedule="annual_previous_dec31_frozen_not_monthly_refit",
        oof_target="y minus actual historical annual out-of-year Ridge forecast; unknown excluded",
        residual_tree_activation="event_weak only; global/weak share weak readiness; otherwise raw Ridge",
        calibration_formula="clip(sum(w*p*y)/(sum(w*p*p)+6*weighted_mean(p*p)),0,1)",
        calibration_weight_units="each signal month sums to24month age decay; no row-count normalization",
        calibration_missing="unknown slope; global fallback identity1, state fallback ready global then identity1",
        forecast_availability="all four arms exactly match raw Ridge availability; unknown raw is NaN",
        historical_oof_available_rows=int(oof_available.sum()), matured_residual_eligible_rows=int(residual_good.sum()),
        raw_ridge=raw_audit, historical_annual_oof=oof_audit, annual=annual)
    return out, audit
