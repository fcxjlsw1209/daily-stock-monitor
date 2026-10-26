"""Causal monthly Ridge for a third holding's marginal portfolio log return.

The caller freezes the first two holdings and defines nearby challengers using
signal-time information only. This module constructs arithmetic/features and
fits annual models; it does not select stocks, change a ledger, or tune rules.
"""
import importlib.util
from pathlib import Path

import numpy as np

_PREVIOUS = Path(__file__).resolve().parents[1] / "monthly_architecture_restart" / "residual_model.py"
_spec = importlib.util.spec_from_file_location("monthly_portfolio_pair_numerics", _PREVIOUS)
_numeric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numeric)
_days = _numeric._days
_years_before = _numeric._years_before
_month_count = _numeric._month_count
_require_finite = _numeric._require_finite
month_balanced_weights = _numeric.month_balanced_weights

ALPHA = 10.
HALF_LIFE_MONTHS = 24.
MIN_MONTHS = 18
MIN_PAIRS = 30


def pair_features(X_i, X_j, X_A, X_B):
    """[challenger-incumbent features, difference * mean anchor features].

    Ten input features produce twenty features. No raw anchor-only column,
    constant intercept, date or identifier is added. Swapping i/j negates the
    complete vector; swapping the two anchors leaves it unchanged.
    """
    arrays = [np.asarray(v, float) for v in (X_i, X_j, X_A, X_B)]
    if any(v.ndim < 1 or not v.shape[-1] for v in arrays):
        raise ValueError("pair features require nonempty final feature axes")
    if len({v.shape[-1] for v in arrays}) != 1:
        raise ValueError("pair feature counts differ")
    i, j, a, b = np.broadcast_arrays(*arrays)
    with np.errstate(over="raise", invalid="ignore"):
        difference = j - i
        context = (a + b) / 2.
        return np.concatenate((difference, difference * context), axis=-1)


def pair_targets(r_A, r_B, r_i, r_j):
    """Native portfolio and stock log-return differences for matured labels.

    portfolio = log(1+(r_A+r_B+r_j)/3)-log(1+(r_A+r_B+r_i)/3)
    stock = log(1+r_j)-log(1+r_i), deliberately NOT divided by three here.
    Both targets use one common valid mask for a matched-sample comparison.
    Invalid labels are NaN; this mask must never filter live trading choices.
    """
    a, b, i, j = np.broadcast_arrays(*[np.asarray(v, float) for v in (r_A, r_B, r_i, r_j)])
    valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(i) & np.isfinite(j)
             & (a >= -1.) & (b >= -1.) & (i > -1.) & (j > -1.))
    portfolio, stock = np.full(a.shape, np.nan), np.full(a.shape, np.nan)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        portfolio[valid] = (np.log1p((a[valid] + b[valid] + j[valid]) / 3.)
                            - np.log1p((a[valid] + b[valid] + i[valid]) / 3.))
        stock[valid] = np.log1p(j[valid]) - np.log1p(i[valid])
    return dict(portfolio=portfolio, stock=stock, valid=valid)


def _fit_rms_ridge(X, y, weights):
    """No intercept/centering: weighted RMS scaling preserves antisymmetry."""
    for name, values in (("features", X), ("target", y), ("weights", weights)):
        _require_finite(name, values)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        scale = np.sqrt(np.average(X * X, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.)
        z = X / scale
        lhs = np.einsum("ni,n,nj->ij", z, weights, z, optimize=False)
        lhs += ALPHA * np.eye(X.shape[1])
        rhs = np.einsum("ni,n,n->i", z, weights, y, optimize=False)
    _require_finite("RMS scales", scale)
    _require_finite("normal-equation lhs", lhs)
    _require_finite("normal-equation rhs", rhs)
    coefficient = np.linalg.solve(lhs, rhs)
    _require_finite("coefficients", coefficient)
    return scale, coefficient


def _predict(model, X):
    scale, coefficient = model
    _require_finite("prediction features", X)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        forecast = np.einsum("ni,i->n", X / scale, coefficient, optimize=False)
    _require_finite("predictions", forecast)
    return forecast


def annual_pair_forecasts(dates, ends, X, y, event_dates, event_X, event_good):
    """Return {pred, available, audit} for a caller-selected single target.

    Pointwise rows are historical candidate pairs, not independently chosen
    asset observations. Only finite mature pair labels enter annual training.
    A signal year's fit ends at previous Dec 31, uses the preceding five years,
    and needs at least 18 distinct signal months and 30 pairs. Each month has
    equal total pre-decay weight, then a fixed 24-month half-life is applied.
    Cold starts or invalid event features return 0 with available=False.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X, y, event_X = np.asarray(X, float), np.asarray(y, float), np.asarray(event_X, float)
    event_good = np.asarray(event_good, bool)
    n, q = len(dates), len(event_dates)
    if X.ndim != 2 or X.shape[0] != n or not X.shape[1]:
        raise ValueError("X must be a nonempty-feature pair matrix")
    if event_X.shape != (q, X.shape[1]):
        raise ValueError("event_X pair-feature dimensions differ")
    if ends.shape != (n,) or y.shape != (n,) or event_good.shape != (q,):
        raise ValueError("dates, target or event eligibility lengths differ")
    horizon = ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)
    if (ends <= dates).any() or (horizon != 1).any():
        raise ValueError("labels must end in the next calendar month")
    usable = np.isfinite(y) & np.isfinite(X).all(axis=1)
    valid_events = event_good & np.isfinite(event_X).all(axis=1)
    years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    prediction, available = np.zeros(q), np.zeros(q, dtype=bool)
    logs = []
    for year in np.unique(years):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        train = usable & (ends <= cutoff) & (dates > _years_before(cutoff))
        events = valid_events & (years == year)
        months, count = _month_count(dates[train]), int(train.sum())
        ready = months >= MIN_MONTHS and count >= MIN_PAIRS
        row = dict(year=int(year), cutoff=str(cutoff),
            lookback_exclusive_start=str(_years_before(cutoff)),
            training_pairs=count, training_months=months,
            first_training_signal=str(dates[train].min()) if count else None,
            latest_training_signal=str(dates[train].max()) if count else None,
            latest_label_end=str(ends[train].max()) if count else None,
            prediction_pairs=int(events.sum()),
            invalid_or_ineligible_event_pairs=int(((years == year) & ~valid_events).sum()),
            status="fit" if ready else "cold_start")
        if ready and events.any():
            weights = month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
            model = _fit_rms_ridge(X[train], y[train], weights)
            prediction[events] = _predict(model, event_X[events])
            available[events] = True
            row["weight_sum"] = float(weights.sum())
        logs.append(row)
    audit = dict(method="annual_pair_target_zero_intercept_ridge", alpha=ALPHA,
        half_life_months=HALF_LIFE_MONTHS, lookback_years=5,
        min_training_months=MIN_MONTHS, min_training_pairs=MIN_PAIRS,
        feature_count=int(X.shape[1]),
        feature_contract="difference_and_difference_times_mean_two_anchor_features;no_identifiers",
        scale_fit="training_only_weighted_RMS;no_centering", intercept=False,
        weighting="equal_month_total_then_age_decay;sum_weights_equals_training_pair_count",
        target_units="caller_supplied_native_log_return_difference;stock_target_not_divided_here",
        event_eligibility="caller_signal_time_only;never_uses_future_label_completeness",
        linear_algebra="einsum_optimize_false_with_finite_guards", annual=logs)
    return dict(pred=prediction, available=available, audit=audit)
