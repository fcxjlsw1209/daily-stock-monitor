"""Annual leave-one-signal-year-out Ridge stability for monthly comparisons.

This is a historical-block sensitivity proxy, NOT a calibrated prediction
interval or a claimed probability of profitable trading. All fits predict the
same actual-next-calendar-month target using pointwise Ridge. The caller owns
signal-time candidate eligibility/ranks and all monthly trading rules.
"""
import importlib.util
from pathlib import Path

import numpy as np

_PREVIOUS = Path(__file__).resolve().parents[1] / "monthly_architecture_restart" / "residual_model.py"
_spec = importlib.util.spec_from_file_location("monthly_block_proven_numerics", _PREVIOUS)
_numeric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numeric)
_days = _numeric._days
_years_before = _numeric._years_before
_month_count = _numeric._month_count
_fit_ridge = _numeric._fit_ridge
_predict = _numeric._predict
month_balanced_weights = _numeric.month_balanced_weights

ALPHA = 10.
HALF_LIFE_MONTHS = 24.
MIN_MONTHS = 36
MIN_BLOCKS = 4
MIN_FOLD_MONTHS = 18


def annual_block_forecasts(dates, ends, X, y, good, event_dates, event_X, event_good):
    """Return full/folds/available/fold_available/audit for annual frozen fits.

    Full and every deleted-year fit independently standardize their own past
    training rows and compute month-balanced, 24-month-decayed sample weights.
    Annual cutoff is previous Dec 31; the trailing five-year signal window is
    lower-exclusive and every target must already have completed. There must
    be >=36 mature months and >=4 signal-year blocks; every deleted-year fit
    must retain >=18 months. Unavailable values are 0 with False masks.

    folds has [maximum_actual_blocks, number_of_events] shape. Slots map to
    ascending excluded signal years within each annual model; audit records
    this mapping. Compare candidates in the SAME event, never across years.
    """
    dates, ends = _days(dates, "dates"), _days(ends, "ends")
    event_dates = _days(event_dates, "event_dates")
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    y, good, event_good = np.asarray(y, float), np.asarray(good, bool), np.asarray(event_good, bool)
    n, q = len(dates), len(event_dates)
    if X.ndim != 2 or X.shape[0] != n or not X.shape[1]:
        raise ValueError("X must be a nonempty-feature training matrix")
    if event_X.shape != (q, X.shape[1]):
        raise ValueError("event_X feature shape differs")
    if ends.shape != (n,) or y.shape != (n,) or good.shape != (n,) or event_good.shape != (q,):
        raise ValueError("dates, labels, or eligibility lengths differ")
    horizon = ends.astype("datetime64[M]").astype(int) - dates.astype("datetime64[M]").astype(int)
    if (ends <= dates).any() or (horizon != 1).any():
        raise ValueError("labels must end in the next calendar month")
    usable = good & np.isfinite(y) & np.isfinite(X).all(axis=1)
    valid_events = event_good & np.isfinite(event_X).all(axis=1)
    signal_years = dates.astype("datetime64[Y]").astype(int) + 1970
    event_years = event_dates.astype("datetime64[Y]").astype(int) + 1970
    plans = []
    for year in np.unique(event_years):
        cutoff = np.datetime64("%04d-12-31" % (year - 1), "D")
        train = usable & (ends <= cutoff) & (dates > _years_before(cutoff))
        blocks = np.unique(signal_years[train])
        plans.append((int(year), cutoff, train, blocks))
    max_blocks = max((len(p[3]) for p in plans), default=0)
    full, folds = np.zeros(q), np.zeros((max_blocks, q))
    available = np.zeros(q, dtype=bool)
    fold_available = np.zeros((max_blocks, q), dtype=bool)
    logs = []
    for year, cutoff, train, blocks in plans:
        events = valid_events & (event_years == year)
        months = _month_count(dates[train])
        row = dict(year=year, cutoff=str(cutoff),
            lookback_exclusive_start=str(_years_before(cutoff)),
            training_rows=int(train.sum()), training_months=months,
            signal_year_blocks=[int(b) for b in blocks],
            latest_label_end=str(ends[train].max()) if train.any() else None,
            prediction_rows=int(events.sum()),
            invalid_or_ineligible_events=int(((event_years == year) & ~valid_events).sum()),
            status="cold_start", folds=[])
        fold_masks = [train & (signal_years != excluded) for excluded in blocks]
        ready = (months >= MIN_MONTHS and len(blocks) >= MIN_BLOCKS
                 and all(_month_count(dates[m]) >= MIN_FOLD_MONTHS for m in fold_masks))
        row["status"] = "fit" if ready else "cold_start"
        for slot, (excluded, mask) in enumerate(zip(blocks, fold_masks)):
            row["folds"].append(dict(slot=slot, excluded_signal_year=int(excluded),
                training_rows=int(mask.sum()), training_months=_month_count(dates[mask]),
                included_signal_years=[int(v) for v in np.unique(signal_years[mask])],
                latest_label_end=str(ends[mask].max()) if mask.any() else None,
                status="fit" if ready else "cold_start"))
        if ready and events.any():
            weights = month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
            model = _fit_ridge(X[train], y[train], weights, ALPHA)
            full[events] = _predict(model, event_X[events])
            for slot, mask in enumerate(fold_masks):
                weights = month_balanced_weights(dates[mask], cutoff, HALF_LIFE_MONTHS)
                model = _fit_ridge(X[mask], y[mask], weights, ALPHA)
                folds[slot, events] = _predict(model, event_X[events])
                fold_available[slot, events] = True
            available[events] = True
        logs.append(row)
    audit = dict(method="pointwise_ridge_leave_one_signal_year_out_stability", alpha=ALPHA,
        half_life_months=HALF_LIFE_MONTHS, lookback_years=5,
        min_full_months=MIN_MONTHS, min_year_blocks=MIN_BLOCKS, min_fold_months=MIN_FOLD_MONTHS,
        block_definition="calendar_year_of_historical_signal;actual_blocks_not_fixed_count",
        scale_and_weights="independently_refit_on_each_full_or_deleted_block_training_set",
        weighting="equal_month_total_before_age_decay;normalized_to_training_row_count",
        prediction_units="native_next_calendar_month_return;no_cross_section_rank_amplification",
        uncertainty="pair_difference_delete_year_jackknife_SE;sensitivity_proxy_not_calibrated_confidence",
        max_blocks=int(max_blocks), linear_algebra="proven_einsum_finite_guard_routines", annual=logs)
    return dict(full=full, folds=folds, available=available,
                fold_available=fold_available, audit=audit)


def pair_evidence(full, folds, available, fold_available, i, j, k=1.):
    """Evidence for candidate j outperforming candidate i in the same event.

    Compute each deleted-year difference jointly, preserving prediction
    covariance: SE=sqrt((B-1)/B * sum((difference_b-mean_difference)^2)).
    lower=full_difference-k*SE is a stability margin, not a 95% bound. Missing,
    mismatched, or nonfinite evidence returns available=False and lower=0.
    """
    full, folds = np.asarray(full, float), np.asarray(folds, float)
    available, fold_available = np.asarray(available, bool), np.asarray(fold_available, bool)
    if full.ndim != 1 or folds.ndim != 2 or folds.shape[1] != len(full):
        raise ValueError("forecast shapes differ")
    if available.shape != full.shape or fold_available.shape != folds.shape:
        raise ValueError("availability shapes differ")
    if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
        raise ValueError("pair indices must be integers")
    if i < 0 or j < 0 or i >= len(full) or j >= len(full) or i == j:
        raise ValueError("pair indices must identify two distinct candidates")
    if not np.isfinite(k) or k < 0:
        raise ValueError("stability multiplier must be finite and nonnegative")
    no = dict(delta_full=0., se=0., lower=0., available=False, fold_count=0)
    if not available[i] or not available[j] or not np.isfinite(full[[i, j]]).all():
        return no
    mask = fold_available[:, i]
    if not np.array_equal(mask, fold_available[:, j]) or int(mask.sum()) < MIN_BLOCKS:
        return no
    pair = folds[mask][:, [i, j]]
    if not np.isfinite(pair).all():
        return no
    with np.errstate(over="raise", invalid="raise"):
        differences = pair[:, 1] - pair[:, 0]
        count = len(differences)
        se = float(np.sqrt((count - 1.) / count * np.sum((differences - differences.mean()) ** 2)))
        delta = float(full[j] - full[i])
        lower = float(delta - k * se)
    if not np.isfinite([delta, se, lower]).all():
        return no
    return dict(delta_full=delta, se=se, lower=lower, available=True, fold_count=count)
