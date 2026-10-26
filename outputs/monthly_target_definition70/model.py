"""Four continuous targets with identical annual, causal Ridge training rows.

This module does not construct labels, rank predictions or evaluate portfolios.
The caller's common cohort is intersected with finite values for ALL four arms.
"""
import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1]/'monthly_architecture_restart/residual_model.py'
_SPEC = importlib.util.spec_from_file_location('target70_ridge_numeric', _PATH)
numeric = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(numeric)

TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
ALPHA = 10.
HALF_LIFE_MONTHS = 12.
MIN_MONTHS = 18


def _bool(values, rows, name):
    values = np.asarray(values)
    if values.shape != (rows,) or values.dtype.kind != 'b':
        raise ValueError(name+' must be an exact boolean row vector')
    return values


def annual_forecasts(dates, ends, X, targets, train_good, event_dates, event_X,
                     event_good, *, min_months=MIN_MONTHS):
    """Return four raw forecast vectors, common availability and annual audit.

    Every year's model uses only labels ending by preceding December 31 and
    signals strictly after the five-year lower bound. The standardization,
    month-balanced 12-month decay, alpha10 and training cohort are shared.
    Event eligibility is signal-time input; future label validity is never used
    to decide which current events can be predicted. Cold starts remain NaN.
    ``min_months`` is exposed only to allow small synthetic fixtures; production
    callers must use the fixed default of18.
    """
    dates, ends = numeric._days(dates, 'dates'), numeric._days(ends, 'ends')
    event_dates = numeric._days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or (ends <= dates).any():
        raise ValueError('label ends must align and be strictly after signals')
    if np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1):
        raise ValueError('labels must end in the next calendar month')
    if (isinstance(min_months, (bool, np.bool_)) or
            not isinstance(min_months, (int, np.integer)) or min_months < 1):
        raise ValueError('min_months must be a positive integer')
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    if X.ndim != 2 or X.shape[0] != n or X.shape[1] == 0:
        raise ValueError('X must be a nonempty-feature training matrix')
    if event_X.shape != (q, X.shape[1]):
        raise ValueError('event_X must match the training feature width')
    good = _bool(train_good, n, 'train_good')
    event_good = _bool(event_good, q, 'event_good')
    if set(targets) != set(TARGETS):
        raise ValueError('exactly the four registered targets are required')
    ys = {key: np.asarray(targets[key], float) for key in TARGETS}
    if any(y.shape != (n,) for y in ys.values()):
        raise ValueError('all targets must have exact training row length')
    usable = good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        usable &= np.isfinite(y)
    # A caller cannot accidentally train any arm on a singleton monthly cohort.
    for month in np.unique(dates.astype('datetime64[M]')):
        in_month = dates.astype('datetime64[M]') == month
        if (usable & in_month).sum() < 2:
            usable[in_month] = False
    events = event_good & np.isfinite(event_X).all(axis=1)
    output = {key: np.full(q, np.nan) for key in TARGETS}
    available = np.zeros(q, bool)
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    audit = []
    for year in np.unique(years):
        cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
        start = numeric._years_before(cutoff)
        train = usable & (dates > start) & (ends <= cutoff)
        predict = (years == year) & events
        months = numeric._month_count(dates[train])
        record = dict(year=int(year), cutoff=str(cutoff),
            lookback_exclusive_start=str(start), training_rows=int(train.sum()),
            training_months=months, prediction_rows=int(predict.sum()),
            latest_label_end=str(ends[train].max()) if train.any() else None,
            first_training_signal=str(dates[train].min()) if train.any() else None,
            latest_training_signal=str(dates[train].max()) if train.any() else None,
            status='fit' if months >= min_months else 'cold_start')
        if months >= min_months and predict.any():
            weights = numeric.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
            for key in TARGETS:
                fit = numeric._fit_ridge(X[train], ys[key][train], weights, ALPHA)
                output[key][predict] = numeric._predict(fit, event_X[predict])
            available[predict] = True
            record['weight_sum'] = float(weights.sum())
        audit.append(record)
    output['available'] = available
    output['audit'] = dict(alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS,
        min_months=int(min_months), lookback_years=5, targets=list(TARGETS),
        shared_rows_and_weights=True, training_only_standardization=True,
        weighting='equal_month_total_then_age_decay_normalized_to_row_count',
        predictions_are_raw=True, label_ranks_used=False, annual=audit)
    return output
