"""Matched annual close/open Ridge plus original own-security local correction.

Only the target changes between arms. Calendar anchors and information maturity
are separate. No label construction, market loading, ranking or trades occur.
"""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
from threadpoolctl import threadpool_limits
import after142_paired_target_labels as labels

_PATH = Path(__file__).resolve().parents[1]/'outputs/monthly_local_linear70/model.py'
_SPEC = importlib.util.spec_from_file_location('paired143_original_local', _PATH)
original = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(original)
numeric = original.numeric
ARMS = ('close', 'open')
ALPHA = original.ALPHA
MIN_MONTHS = original.MIN_MONTHS
MIN_OWN_MONTHS = original.MIN_OWN_MONTHS
HALF_LIFE_MONTHS = original.HALF_LIFE_MONTHS


def dependency_files():
    return (Path(__file__).resolve(), Path(labels.__file__).resolve(),
            labels._VALIDATOR_PATH, *original.dependency_files())


def _matrix(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'fiu':
        raise ValueError(name+' must be an exact numeric array of shape '+str(shape))
    return value.astype(float, copy=True)


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def annual_forecasts(dates, calendar_ends, available_sessions, symbols, X, targets,
                     common_good, event_dates, event_symbols, event_X, event_good,
                     *, min_months=MIN_MONTHS, min_own_months=MIN_OWN_MONTHS):
    """Two raw local-linear predictions with shared annual cohorts and scalers.

    The caller must verify exact session anchors with the official calendar.
    Here next-month calendar ends and month+2 information availability are kept
    distinct and validated. Only availability gates an annual training cutoff.
    Minimum arguments exist solely for small artificial tests; production18.
    """
    original._minimum(min_months, 'min_months')
    original._minimum(min_own_months, 'min_own_months')
    days = labels.old._days
    dates, ends, available_days = (days(dates, 'dates'), days(calendar_ends, 'calendar_ends'),
                                  days(available_sessions, 'available_sessions'))
    event_dates = days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    months = dates.astype('datetime64[M]')
    if ends.shape != (n,) or available_days.shape != (n,):
        raise ValueError('Calendar and availability vectors must align')
    if ((ends <= dates).any() or (available_days <= ends).any() or
        (ends.astype('datetime64[M]') != months+1).any() or
        (available_days.astype('datetime64[M]') != months+2).any()):
        raise ValueError('Separate next-month calendar end and following-month availability required')
    symbols = original._symbols(symbols, n, 'symbols')
    event_symbols = original._symbols(event_symbols, q, 'event_symbols')
    X, event_X = _matrix(X, (n, 10), 'X'), _matrix(event_X, (q, 10), 'event_X')
    good = original.original._bool(common_good, n, 'common_good')
    event_good = original.original._bool(event_good, q, 'event_good')
    if set(targets) != set(ARMS):
        raise ValueError('Exactly close and open targets required')
    ys = {key: _matrix(targets[key], (n,), key) for key in ARMS}
    usable = good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        usable &= np.isfinite(y)
    for month in np.unique(months):
        rows = months == month
        if (rows & usable).sum() < 2:
            usable[rows] = False
    events = event_good & np.isfinite(event_X).all(axis=1)
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    raw = {key: np.full(q, np.nan) for key in ARMS}
    pooled_raw = {key: np.full(q, np.nan) for key in ARMS}
    available = np.zeros(q, bool)
    local_available = np.zeros(q, bool)
    annual, local_audit = [], []
    fits = 0
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = numeric._years_before(cutoff)
            train = usable & (dates > start) & (available_days <= cutoff)
            predict = (years == year) & events
            ids = np.flatnonzero(train)
            count = numeric._month_count(dates[train])
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                training_rows=len(ids), training_months=count, prediction_rows=int(predict.sum()),
                first_training_signal=str(dates[train].min()) if len(ids) else None,
                latest_calendar_end=str(ends[train].max()) if len(ids) else None,
                latest_available_session=str(available_days[train].max()) if len(ids) else None,
                training_indices_sha256=_digest(ids),
                status='fit' if count >= min_months and predict.any() else 'cold_start' if count < min_months else 'no_queries')
            if count >= min_months and predict.any():
                identities = list(zip(months[train].astype(str), symbols[train]))
                if len(set(identities)) != len(identities):
                    raise ValueError('Repeated actual training security/month')
                original._ranks('actual training ranks', X[train])
                original._ranks('actual query ranks', event_X[predict])
                weights = numeric.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
                models = {key: numeric._fit_ridge(X[train], ys[key][train], weights, ALPHA) for key in ARMS}
                fits += len(models)
                for index in (0, 1):
                    np.testing.assert_array_equal(models['close'][index], models['open'][index])
                record.update(weights_sha256=_digest(weights), weight_sum=float(weights.sum()),
                              scaler_mean_sha256=_digest(models['close'][0]), scaler_scale_sha256=_digest(models['close'][1]))
                residuals = {}
                for key in ARMS:
                    pooled_raw[key][predict] = numeric._predict(models[key], event_X[predict])
                    raw[key][predict] = pooled_raw[key][predict]
                    residuals[key] = ys[key][train]-numeric._predict(models[key], X[train])
                    original._finite('mature residuals', residuals[key])
                available[predict] = True
                by_symbol = {s: np.flatnonzero(symbols[train] == s) for s in set(symbols[train])}
                for j in np.flatnonzero(predict):
                    own = by_symbol.get(event_symbols[j], np.empty(0, dtype=int))
                    own_ids = ids[own]
                    own_months = numeric._month_count(dates[own_ids])
                    detail = dict(event_row=int(j), date=str(event_dates[j]), symbol=event_symbols[j],
                        year=int(year), cutoff=str(cutoff), own_rows=len(own), own_months=own_months,
                        latest_available_session=str(available_days[own_ids].max()) if len(own) else None,
                        status='insufficient_own_history', corrections={key:0. for key in ARMS})
                    if own_months >= min_own_months:
                        w = original.local_weights(dates[own_ids], cutoff, X[own_ids], event_X[j])
                        infos = []
                        for key in ARMS:
                            _, correction, info = original.local_corrections(X[own_ids], event_X[j],
                                models[key][1], residuals[key][own], w)
                            infos.append(info)
                            if correction != 0.:
                                raw[key][j] += correction
                            detail['corrections'][key] = correction
                        local_available[j] = float(w.sum()) > 0.
                        detail.update(status='fit' if local_available[j] else 'zero_weight',
                                      weight_mass=float(w.sum()), weights_sha256=_digest(w))
                    local_audit.append(detail)
            annual.append(record)
    for key in ARMS:
        if not np.array_equal(np.isfinite(raw[key]), available) or not np.array_equal(np.isfinite(pooled_raw[key]), available):
            raise FloatingPointError('Nonfinite model output or mismatched paired availability')
        np.testing.assert_array_equal(raw[key][available & ~local_available], pooled_raw[key][available & ~local_available])
    return dict(raw=raw, pooled=pooled_raw, available=available, local_available=local_available,
        audit=dict(targets=list(ARMS), new_pooled_model_fits=fits, annual=annual, local=local_audit,
        alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS, min_months=int(min_months),
        min_own_months=int(min_own_months), bandwidth=original.BANDWIDTH,
        intercept_penalty=original.INTERCEPT_PENALTY, slope_penalty=original.SLOPE_PENALTY,
        shared_rows_weights_scaler=True, local_weights_normalized=False,
        residuals='in-sample within mature annual cohort, not OOF',
        maturity='shared next opening after next-calendar-month end; not calendar_end',
        query_eligibility='only supplied original signal-time mask and finite features',
        exact_calendar_validation='caller must validate with frozen official session calendar'))
