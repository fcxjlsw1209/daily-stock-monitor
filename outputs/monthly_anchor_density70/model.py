"""Matched annual month21/union21 Ridge; shared month support and weight mass."""
from pathlib import Path
import importlib.util
import numpy as np
import threadpoolctl
from threadpoolctl import threadpool_limits

P = Path(__file__).resolve().parent
ORIGINAL_PATH = P.parent/'monthly_target_definition70/model.py'
_sp = importlib.util.spec_from_file_location('density70_original_model', ORIGINAL_PATH)
original = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(original)
numeric = original.numeric
TARGETS = original.TARGETS
MODES = ('month21', 'union21')
ALPHA = 10.
HALF_LIFE_MONTHS = 12.
MIN_MONTHS = 18


def dependency_files():
    return (Path(__file__).resolve(), ORIGINAL_PATH, Path(numeric.__file__).resolve(),
            Path(threadpoolctl.__file__).resolve())


def anchor_weights(dates, cutoff, total_weight):
    """Equal month decay mass, equal anchors per month, equal names per anchor.

    total_weight is the mature monthly-control row count, never union row count.
    """
    dates = numeric._days(dates, 'training dates'); cutoff = np.datetime64(cutoff, 'D')
    if not len(dates) or np.isnat(cutoff) or (dates > cutoff).any():
        raise ValueError('Nonempty past-only dates required')
    if not np.isfinite(total_weight) or total_weight <= 0: raise ValueError('Positive fixed weight mass required')
    anchors, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    amonth = anchors.astype('datetime64[M]')
    months, minverse, acounts = np.unique(amonth, return_inverse=True, return_counts=True)
    ages = cutoff.astype('datetime64[M]').astype(int)-months.astype(int)
    weight = 1. / counts[inverse].astype(float)
    weight /= acounts[minverse[inverse]]
    weight *= np.exp2(-ages[minverse[inverse]]/HALF_LIFE_MONTHS)
    return weight*(float(total_weight)/weight.sum())


def _used_valid(X, y):
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('A declared-good mature density row is nonfinite; do not drop rows')
    if (np.abs(X) > .5).any(): raise ValueError('Actual plain10 training ranks out of bounds')


def annual_forecasts(source_dates, source_ends, source_X, source_targets, source_good,
                     event_dates, event_X, event_good, density_dates, density_ends,
                     density_X, density_y, density_good, density_is_monthly, *, min_months=MIN_MONTHS):
    """Independent original four-target replay, then two fixed annual fits.

    available is the original source mask. candidate_available never changes U;
    missing candidate predictions are NaN for the driver's whole-month fallback.
    Unused future labels/features cannot affect past fits or their scalers.
    """
    if isinstance(min_months, (bool, np.bool_)) or not isinstance(min_months, (int, np.integer)) or min_months < 1:
        raise ValueError('Positive integer minimum months required')
    sd, se = numeric._days(source_dates, 'source dates'), numeric._days(source_ends, 'source ends')
    ed = numeric._days(event_dates, 'event dates')
    dates, ends = numeric._days(density_dates, 'density dates'), numeric._days(density_ends, 'density ends')
    X, y, Q = np.asarray(density_X, float), np.asarray(density_y, float), np.asarray(event_X, float)
    if np.asarray(source_X).shape != (len(sd), 10) or Q.shape != (len(ed), 10) or X.shape != (len(dates), 10) or y.shape != (len(dates),):
        raise ValueError('Source and candidate must retain aligned ten-column features')
    if ends.shape != dates.shape or (ends <= dates).any(): raise ValueError('Density ends must follow signals')
    good = original._bool(density_good, len(dates), 'density good')
    monthly = original._bool(density_is_monthly, len(dates), 'monthly mask')
    for d in np.unique(dates):
        same = dates == d
        if len(np.unique(ends[same])) != 1 or len(np.unique(monthly[same])) != 1:
            raise ValueError('Rows of one anchor must share endpoint and monthly status')
    md = np.unique(dates[monthly])
    if len(np.unique(md.astype('datetime64[M]'))) != len(md):
        raise ValueError('Only one original monthly anchor per month')
    years = ed.astype('datetime64[Y]').astype(int)+1970
    annual=[]
    with threadpool_limits(limits=1):
        source = original.annual_forecasts(sd, se, source_X, source_targets, source_good,
                                          ed, Q, event_good, min_months=min_months)
        raw = dict(source_log=source['log_excess'].copy(), **{k: np.full(len(ed), np.nan) for k in MODES})
        available = source['available'].copy(); candidate_available = np.zeros(len(ed), bool)
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D'); start = numeric._years_before(cutoff)
            mature = good & (dates > start) & (ends <= cutoff)
            # Structural cohort validation only for labels actually matured by this cutoff.
            for d in np.unique(dates[mature]):
                if (mature & (dates == d)).sum() < 2:
                    raise ValueError('Declared-good mature anchor cohort has fewer than two names')
            a = mature & monthly; amonths = np.unique(dates[a].astype('datetime64[M]'))
            b = mature & np.isin(dates.astype('datetime64[M]'), amonths)
            predict = (years == year) & available
            n_a = int(a.sum()); count = len(amonths)
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                monthly_rows=n_a, monthly_months=count, union_rows=int(b.sum()),
                monthly_support=[str(z) for z in amonths],
                union_support=[str(z) for z in np.unique(dates[b].astype('datetime64[M]'))],
                excluded_weekly_only_rows=int((mature & ~b).sum()),
                excluded_weekly_only_months=[str(z) for z in np.unique(dates[mature & ~b].astype('datetime64[M]'))],
                prediction_rows=int(predict.sum()), fixed_weight_total=n_a,
                status='fit' if count >= min_months else 'cold_start', arms={})
            if count >= min_months and predict.any():
                if not np.isfinite(Q[predict]).all() or (np.abs(Q[predict]) > .5).any():
                    raise ValueError('Actual query plain10 must be finite original ranks')
                for mode, train in zip(MODES, [a, b]):
                    _used_valid(X[train], y[train])
                    weight = anchor_weights(dates[train], cutoff, n_a)
                    if mode == 'month21':
                        np.testing.assert_array_equal(weight, numeric.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS))
                    fit = numeric._fit_ridge(X[train], y[train], weight, ALPHA)
                    pred = numeric._predict(fit, Q[predict])
                    if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite density prediction')
                    raw[mode][predict] = pred
                    record['arms'][mode] = dict(rows=int(train.sum()), anchors=len(np.unique(dates[train])),
                        months=numeric._month_count(dates[train]), weight_sum=float(weight.sum()),
                        first_signal=str(dates[train].min()), latest_signal=str(dates[train].max()),
                        latest_label_end=str(ends[train].max()), mean=fit[0].tolist(), scale=fit[1].tolist(),
                        coefficient=fit[2].tolist(), intercept=float(fit[3]))
                candidate_available[predict] = True
            annual.append(record)
    return dict(raw=raw, available=available, candidate_available=candidate_available,
        audit=dict(alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS, min_months=int(min_months),
                   source=source['audit'], annual=annual, same_calendar_month_support=True,
                   weight_total='mature monthly control row count N_A for both arms',
                   source_masks_unchanged=True, training_only_standardization=True,
                   label_overlap_not_independent=True))
