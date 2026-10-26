"""Fixed annual global/weak-market tree and Ridge development forecasts.

Pure modeling only: no file reads/writes, target construction, return evaluation,
hyperparameter selection, or portfolio rules. The caller supplies point-in-time
features, market-state masks, eligibility, and the already defined target.
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


_HELPER_PATH = Path(__file__).resolve().parents[1] / "monthly_architecture_restart" / "residual_model.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("_conditional70_ridge_helpers", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)
_days = _HELPERS._days
_years_before = _HELPERS._years_before
_fit_ridge = _HELPERS._fit_ridge
_predict = _HELPERS._predict
month_balanced_weights = _HELPERS.month_balanced_weights

ARMS = ("global_tree", "weak_tree", "global_ridge", "weak_ridge")
TREE_PARAMETERS = dict(loss="squared_error", max_iter=100, learning_rate=.05,
                       max_depth=2, min_samples_leaf=50, l2_regularization=1.,
                       random_state=1729, early_stopping=False)
RIDGE_ALPHA = 10.
HALF_LIFE_MONTHS = 24.


def _mask(values, length, name):
    values = np.asarray(values)
    if values.shape != (length,) or values.dtype.kind != "b":
        raise ValueError(name + " must be an aligned one-dimensional boolean array")
    return values


def _minimum(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _fit_tree(X, y, weights):
    return HistGradientBoostingRegressor(**TREE_PARAMETERS).fit(X, y, sample_weight=weights)


def _training_audit(dates, ends, train, minimum_months, minimum_rows):
    count = int(train.sum())
    months = int(len(np.unique(dates[train].astype("datetime64[M]"))))
    return dict(training_rows=count, training_months=months,
                minimum_months=minimum_months, minimum_rows=minimum_rows,
                enough_history=bool(months >= minimum_months and count >= minimum_rows),
                first_training_signal=str(dates[train].min()) if count else None,
                latest_training_signal=str(dates[train].max()) if count else None,
                latest_label_end=str(ends[train].max()) if count else None,
                prediction_rows=0, weak_event_prediction_rows=0,
                strong_event_prediction_rows=0,
                tree_and_ridge_share_rows=True, tree_and_ridge_share_weights=True)


def annual_forecasts(dates, ends, X, y, good, event_dates, eventX, event_good,
                     train_weak, event_weak, min_global_months=18,
                     min_weak_months=6, min_weak_rows=200):
    """Return four raw forecast arrays, four availability masks, and audit.

    Output keys are global_tree, weak_tree, global_ridge, weak_ridge and
    available_<arm>, each [event_rows], together with a JSON-serializable audit.
    Cold starts and invalid events remain NaN/False. Caller handles neutrality
    and weak-market portfolio activation; BOTH training scopes predict every
    eligible event with finite features, including events where event_weak=False.

    For event year Y, training is frozen at December 31 of Y-1. Only good rows
    with finite features/targets, signal > cutoff minus five calendar years,
    and label end <= cutoff are admitted. The weak scope additionally requires
    train_weak=True; event data never influence either fit or scaling. Each
    scope independently balances calendar-month weight, applies a 24-month
    half-life, and normalizes weights to that scope's row count.

    Global scope requires 18 distinct admitted months; weak scope independently
    requires 6 distinct admitted months AND 200 admitted rows. The three minimum
    overrides exist for synthetic tests; production uses the stated defaults.
    Model parameters are fixed. Ridge uses training-only weighted scaling and
    alpha=10. Tree fit and prediction and Ridge linear algebra run with native
    thread pools limited to one, restoring the caller's limits on exit.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X, eventX, y = np.asarray(X, float), np.asarray(eventX, float), np.asarray(y, float)
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or y.shape != (n,):
        raise ValueError("training date, label end, and target lengths differ")
    good = _mask(good, n, "good")
    event_good = _mask(event_good, q, "event_good")
    train_weak = _mask(train_weak, n, "train_weak")
    event_weak = _mask(event_weak, q, "event_weak")
    for name, values, rows in (("X", X, n), ("eventX", eventX, q)):
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
            raise ValueError(name + " must be a nonempty-feature [rows, features] matrix")
    if X.shape[1] != eventX.shape[1]:
        raise ValueError("training and event feature dimensions differ")
    if (ends <= dates).any():
        raise ValueError("label ends must be strictly later than training signals")
    min_global_months = _minimum(min_global_months, "min_global_months")
    min_weak_months = _minimum(min_weak_months, "min_weak_months")
    min_weak_rows = _minimum(min_weak_rows, "min_weak_rows")

    usable = good & np.isfinite(X).all(axis=1) & np.isfinite(y)
    valid_events = event_good & np.isfinite(eventX).all(axis=1)
    results = {arm: np.full(q, np.nan, float) for arm in ARMS}
    results.update({"available_" + arm: np.zeros(q, bool) for arm in ARMS})
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    annual = []
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            in_year = years == year
            cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
            start = _years_before(cutoff)
            global_train = usable & (dates > start) & (ends <= cutoff)
            predict = in_year & valid_events
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                          event_rows=int(in_year.sum()), eligible_event_rows=int(predict.sum()),
                          weak_eligible_event_rows=int((predict & event_weak).sum()),
                          strong_eligible_event_rows=int((predict & ~event_weak).sum()),
                          invalid_or_ineligible_event_rows=int((in_year & ~valid_events).sum()))
            for scope, train, minimum_months, minimum_rows in (
                    ("global", global_train, min_global_months, 1),
                    ("weak", global_train & train_weak, min_weak_months, min_weak_rows)):
                scope_audit = _training_audit(dates, ends, train, minimum_months, minimum_rows)
                scope_audit["status"] = "cold_start" if not scope_audit["enough_history"] else "no_valid_events"
                if scope_audit["enough_history"] and predict.any():
                    weights = month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
                    tree = _fit_tree(X[train], y[train], weights)
                    ridge = _fit_ridge(X[train], y[train], weights, RIDGE_ALPHA)
                    forecasts = {"tree": np.asarray(tree.predict(eventX[predict]), float),
                                 "ridge": _predict(ridge, eventX[predict])}
                    for kind, forecast in forecasts.items():
                        if forecast.shape != (int(predict.sum()),) or not np.isfinite(forecast).all():
                            raise FloatingPointError("nonfinite or misaligned " + scope + "_" + kind + " prediction")
                        arm = scope + "_" + kind
                        results[arm][predict] = forecast
                        results["available_" + arm][predict] = True
                    scope_audit.update(status="fit", weight_sum=float(weights.sum()),
                                       tree_iterations=int(tree.n_iter_), prediction_rows=int(predict.sum()),
                                       weak_event_prediction_rows=int((predict & event_weak).sum()),
                                       strong_event_prediction_rows=int((predict & ~event_weak).sum()))
                record[scope] = scope_audit
            annual.append(record)

    results["audit"] = dict(
        method="annual_frozen_global_vs_weak_tree_and_ridge", lookback_years=5,
        half_life_months=HALF_LIFE_MONTHS, ridge_alpha=RIDGE_ALPHA,
        tree_parameters=dict(TREE_PARAMETERS), native_thread_limit=1,
        minimum_global_months=min_global_months, minimum_weak_months=min_weak_months,
        minimum_weak_rows=min_weak_rows, feature_count=int(X.shape[1]),
        weighting="equal_month_total_then_age_decay_normalized_to_scope_row_count",
        ridge_scaling="weighted_training_rows_only", tree_scaling="none",
        weak_training="usable_global_training_rows_with_train_weak_true",
        prediction_eligibility="event_good_and_finite_features_independent_of_event_weak",
        weak_portfolio_activation="caller_responsibility", unavailable_prediction="NaN",
        target="caller_supplied", parameter_search=False, annual=annual)
    return results
