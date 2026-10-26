"""Independent source229 annual query replay using SHA-bound augmented LS.

No source229 producer, frozen production model, ranking or portfolio engine is
imported. Only the independently implemented round206 least-squares/local-fit
primitives are reused; all annual, identity and query-union control flow is local.
"""
from pathlib import Path
import hashlib
import importlib.util
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = ROOT / 'work/monthly_identity_transfer206_independent.py'
NUMERICAL_SHA = 'a321b2cd07eb2b19c08f7747842e18a47438d781bed9b474c8ef321ec9f5611e'
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
RAW_KEYS = ('source_log', 'local_constant', 'local_linear')
ATOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dependency_files():
    return (Path(__file__).resolve(), NUMERICAL)


def numerical():
    require(hashlib.sha256(NUMERICAL.read_bytes()).hexdigest() == NUMERICAL_SHA,
            'Independent numerical primitive changed')
    spec = importlib.util.spec_from_file_location('early229_independent_numeric', NUMERICAL)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def days(values, name):
    a = np.asarray(values)
    require(a.ndim == 1, name + ': vector')
    if not len(a):
        return np.array([], dtype='datetime64[D]')
    if a.dtype.kind == 'M':
        require(a.dtype == np.dtype('datetime64[D]'), name + ': day-unit dates')
        out = a.copy()
    else:
        require(a.dtype.kind in 'USO' and all(isinstance(v, (str, np.str_)) and
            len(v) == 10 and v[4] == '-' and v[7] == '-' for v in a), name + ': canonical ISO dates')
        try:
            out = a.astype('datetime64[D]')
        except (ValueError, TypeError):
            raise ValueError(name + ': invalid ISO date')
        require(np.array_equal(out.astype(str), a.astype(str)), name + ': canonical ISO dates')
    require(not np.isnat(out).any(), name + ': no NaT')
    return out



def bools(values, size, name):
    a = np.asarray(values)
    require(a.shape == (size,) and a.dtype.kind == 'b', name + ': exact boolean vector')
    return a.copy()


def ids(values, size, name):
    a = np.asarray(values)
    require(a.shape == (size,) and all(isinstance(x, (str, np.str_)) and x and x == x.strip() for x in a),
            name + ': exact security IDs')
    return a.astype(str, copy=True)


def numbers(values, shape, name):
    a = np.asarray(values)
    require(a.shape == shape and a.dtype.kind in 'fiu', name + ': numeric shape')
    return a.astype(float, copy=True)


def model_inputs(args):
    require(len(args) == 10, 'Original ten forecast arguments')
    d, e, s, X, ys, g, qd, qs, Q, qg = args
    d, e, qd = days(d, 'training dates'), days(e, 'training ends'), days(qd, 'query dates')
    n, q = len(d), len(qd)
    require(e.shape == d.shape and np.all(e > d) and
            np.all(e.astype('datetime64[M]') == d.astype('datetime64[M]') + 1), 'Next-month label ends')
    s, qs = ids(s, n, 'training symbols'), ids(qs, q, 'query symbols')
    require(len(set(zip(d.astype(str), s))) == n, 'Duplicate training identity')
    require(len(set(zip(qd.astype(str), qs))) == q, 'Duplicate query identity')
    require(not n or np.all(d[1:] >= d[:-1]), 'Training signals ordered')
    require(not q or np.all(qd[1:] >= qd[:-1]), 'Query signals ordered')
    require(np.all(qd < np.datetime64('2016-08-17')), 'Queries must precede original first signal')
    for axis in (d, qd):
        for month in np.unique(axis.astype('datetime64[M]')):
            require(len(np.unique(axis[axis.astype('datetime64[M]') == month])) == 1, 'One signal per natural month')
    for month in np.unique(d.astype('datetime64[M]')):
        require(len(np.unique(e[d.astype('datetime64[M]') == month])) == 1, 'One label end per month')
    X, Q = numbers(X, (n, 10), 'train plain10'), numbers(Q, (q, 10), 'query plain10')
    g, qg = bools(g, n, 'common_good'), bools(qg, q, 'event_good')
    require(isinstance(ys, dict) and set(ys) == set(TARGETS), 'Four original targets')
    ys = {k: numbers(ys[k], (n,), 'target ' + k) for k in TARGETS}
    return d, e, s, X, ys, g, qd, qs, Q, qg


def close(actual, expected, label, atol=ATOL):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, label + ': shape')
    require(a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu', label + ': numeric dtype')
    require(not np.isinf(a).any() and not np.isinf(b).any(), label + ': infinity')
    require(np.array_equal(np.isnan(a), np.isnan(b)), label + ': NaN mask')
    used = np.isfinite(b)
    require(np.all(np.abs(a[used] - b[used]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(actual, np.generic):
        actual = actual.item()
    if isinstance(expected, np.generic):
        expected = expected.item()
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for k in expected:
            record_check(actual[k], expected[k], label + '/' + k)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) is float and math.isfinite(actual) and abs(actual-expected) <= ATOL,
                label + ': float')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def input_signature(value):
    """No serialization of raw input content; compare structure and raw bytes."""
    if isinstance(value, dict):
        return ('dict', tuple((k, input_signature(v)) for k, v in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(input_signature(v) for v in value))
    if isinstance(value, np.ndarray):
        return ('array', value.shape, value.dtype.str, hashlib.sha256(value.tobytes()).hexdigest())
    return (type(value).__name__, repr(value))


def replay_oracle(*args):
    d, e, s, X, ys, g, qd, qs, Q, qg = model_inputs(args)
    q = len(qd)
    num = numerical()
    months = d.astype('datetime64[M]')
    years = qd.astype('datetime64[Y]').astype(int) + 1970
    good = g & np.isfinite(X).all(axis=1)
    for y in ys.values():
        good &= np.isfinite(y)
    for month in np.unique(months):
        group = months == month
        if np.count_nonzero(good & group) < 2:
            good[group] = False
    query = qg & np.isfinite(Q).all(axis=1)
    raw = {k: np.full(q, np.nan) for k in RAW_KEYS}
    available, local_available = np.zeros(q, bool), np.zeros(q, bool)
    source_annual, annual, local = [], [], []
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = np.datetime64('%04d-12-31' % (year-6), 'D')
            use = np.flatnonzero(good & (d > start) & (e <= cutoff))
            selected = np.flatnonzero((years == year) & query)
            count = len(np.unique(months[use]))
            row = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                training_rows=len(use), training_months=count, prediction_rows=len(selected),
                latest_label_end=str(e[use].max()) if len(use) else None,
                first_training_signal=str(d[use].min()) if len(use) else None,
                latest_training_signal=str(d[use].max()) if len(use) else None,
                status='fit' if count >= 18 else 'cold_start')
            if count >= 18 and len(selected):
                require(np.all(np.abs(X[use]) <= .5) and np.all(np.abs(Q[selected]) <= .5),
                        'Consumed original centered percentile bounds')
                unique, inverse, counts = np.unique(months[use], return_inverse=True, return_counts=True)
                w = 1. / counts[inverse].astype(float)
                age = cutoff.astype('datetime64[M]').astype(int) - unique.astype(int)
                w *= np.exp2(-age[inverse] / 12.)
                w *= len(w) / w.sum()
                mean = np.average(X[use], axis=0, weights=w)
                centered = X[use] - mean
                scale = np.sqrt(np.average(centered * centered, axis=0, weights=w))
                scale = np.where(scale > 1e-12, scale, 1.)
                Z = centered / scale
                # Independently fit all four targets on the identical common cohort.
                fits = {}
                for key in TARGETS:
                    intercept = float(np.average(ys[key][use], weights=w))
                    coef = num.least_squares(Z, ys[key][use] - intercept, w, np.full(10, 10.))
                    fits[key] = (coef, intercept)
                coef, intercept = fits['log_excess']
                base = np.sum(((Q[selected]-mean)/scale) * coef, axis=1) + intercept
                require(np.isfinite(base).all(), 'Finite original pooled predictions')
                for key in RAW_KEYS:
                    raw[key][selected] = base
                available[selected] = True
                row['weight_sum'] = float(w.sum())
                # Independent pooled log refit mirrors the production fit accounting.
                intercept = float(np.average(ys['log_excess'][use], weights=w))
                coef = num.least_squares(Z, ys['log_excess'][use] - intercept, w, np.full(10, 10.))
                residual = ys['log_excess'][use] - (np.sum(Z * coef, axis=1) + intercept)
                require(np.isfinite(residual).all(), 'Finite mature residuals')
                for j in selected:
                    own = np.flatnonzero(s[use] == qs[j])
                    oi = use[own]
                    nc = len(np.unique(months[oi]))
                    detail = dict(event_row=int(j), date=str(qd[j]), symbol=str(qs[j]), year=int(year),
                        cutoff=str(cutoff), own_rows=len(oi), own_months=nc,
                        latest_label_end=str(e[oi].max()) if len(oi) else None,
                        status='insufficient_own_history', weight_mass=0., effective_samples=0.,
                        constant_correction=0., linear_correction=0.)
                    if nc >= 18:
                        ages = cutoff.astype('datetime64[M]').astype(int) - months[oi].astype(int)
                        # Separate scalar feature-distance path, same fixed kernel.
                        lw = np.array([float(np.exp2(-float(a)/12.)) *
                            math.exp(-math.fsum((float(x)-float(v))**2 for x, v in zip(X[k], Q[j])) / 10. / .5)
                            for k, a in zip(oi, ages)])
                        c, l, info = num.local_fit(X[oi], Q[j], scale, residual[own], lw)
                        detail.update(info, constant_correction=c, linear_correction=l)
                        local_available[j] = info['weight_mass'] > 0.
                        for key, correction in (('local_constant', c), ('local_linear', l)):
                            if correction != 0.:
                                raw[key][j] += correction
                    local.append(detail)
            source_annual.append(row)
            ar = {k:v for k,v in row.items() if k != 'latest_training_signal'}
            ar['prediction_rows'] = int(np.count_nonzero((years == year) & available))
            annual.append(ar)
    require(all(np.array_equal(np.isfinite(v), available) for v in raw.values()), 'Raw common availability')
    source = dict(alpha=10., half_life_months=12., min_months=18, lookback_years=5, targets=list(TARGETS),
        shared_rows_and_weights=True, training_only_standardization=True,
        weighting='equal_month_total_then_age_decay_normalized_to_row_count', predictions_are_raw=True,
        label_ranks_used=False, annual=source_annual)
    audit = dict(modes=['local_constant','local_linear'], feature_count=10,
        target='unchanged cached centered log_excess', alpha=10., half_life_months=12., min_months=18,
        min_own_months=18, bandwidth=.5, intercept_penalty=12., slope_penalty=10.,
        same_local_rows_residuals_and_weights=True, local_weights_normalized=False,
        local_recency_clock='annual prior December cutoff calendar-month age',
        kernel='mean squared difference in original centered percentile plain10 units',
        linear_design='query-centered original X differences divided by pooled annual training scales',
        residuals='in-sample residuals within already matured annual training cohort, not OOF',
        source_control='independent original four-target annual model and exact log fit replay',
        no_own_history='exact source raw copy; never remove a query or alter eligibility',
        source=source, annual=annual, local=local)
    return dict(raw=raw, available=available, local_available=local_available, audit=audit)

COUNT_FIELDS = ('query_rows', 'original_query_rows', 'expansion_requested_rows',
    'union_requested_rows', 'original_available_rows', 'added_available_rows',
    'available_rows', 'local_available_rows', 'source_target_fits', 'pooled_log_refits',
    'local_evaluations', 'local_solves', 'actual_regression_solves')
FORECAST_FLAGS = ('passed', 'independent_annual_control_flow', 'independent_augmented_least_squares',
    'same_training_cohort', 'four_targets_unchanged', 'annual_prior_december_cutoffs',
    'five_year_mature_window', 'minimum_18_months', 'same_security_local_history',
    'local_history_not_query_filter', 'early_query_date_bound', 'exact_query_identity',
    'original_eligibility_preserved', 'expansion_warmup_mask_enforced', 'union_query_mask_checked',
    'no_future_label_query_filter', 'no_query_rerank', 'fit_counts_checked', 'audit_checked',
    'input_bytes_unchanged_by_checker')


def forecasts_oracle(*args, expansion_allowed):
    a = list(model_inputs(args))
    q = len(a[6])
    allow = bools(expansion_allowed, q, 'expansion_allowed')
    finite = np.isfinite(a[8]).all(axis=1)
    original = a[9] & finite
    expansion = (~a[9]) & allow & finite
    union = original | expansion
    a[9] = union
    replay = replay_oracle(*a)
    av, la = replay['available'], replay['local_available']
    ra = replay['audit']
    counts = dict(query_rows=q, original_query_rows=int(original.sum()),
        expansion_requested_rows=int(expansion.sum()), union_requested_rows=int(union.sum()),
        original_available_rows=int(np.count_nonzero(av & original)),
        added_available_rows=int(np.count_nonzero(av & expansion)), available_rows=int(av.sum()),
        local_available_rows=int(la.sum()),
        source_target_fits=4*sum(r['status']=='fit' and r['prediction_rows']>0 for r in ra['source']['annual']),
        pooled_log_refits=sum(r['prediction_rows']>0 for r in ra['annual']),
        local_evaluations=len(ra['local']), local_solves=sum(r['status']=='fit' for r in ra['local']))
    counts['actual_regression_solves'] = counts['source_target_fits'] + counts['pooled_log_refits'] + counts['local_solves']
    audit = dict(method='early_original_query_history_extension', query_rows=q, feature_count=10,
        raw_keys=list(RAW_KEYS), query_date_exclusive_upper_bound='2016-08-17',
        original_query_mask=original.tolist(), expansion_query_mask=expansion.tolist(),
        union_query_mask=union.tolist(), expansion_allowed=allow.tolist(),
        query_qualification='finite cached plain10 AND (original train_eligible OR caller-verified 295-price warmup)',
        query_future_labels_used=False, training_targets_changed=False, training_common_good_changed=False,
        new_features_computed=False, annual_forecasts_calls=1, replay_audit=ra, counts=counts)
    return dict(raw=replay['raw'], available=av, local_available=la, audit=audit)


def check_forecasts(*args, expansion_allowed, result):
    before = input_signature((args, expansion_allowed, result))
    expected = forecasts_oracle(*args, expansion_allowed=expansion_allowed)
    require(isinstance(result, dict) and set(result)==set(expected), 'Forecast return schema')
    require(isinstance(result['raw'], dict) and set(result['raw'])==set(RAW_KEYS), 'Raw vector keys')
    for key in RAW_KEYS:
        close(result['raw'][key], expected['raw'][key], key)
    q = len(expected['available'])
    for key in ('available', 'local_available'):
        actual = bools(result[key], q, key)
        require(np.array_equal(actual, expected[key]), key + ': exact mask')
    record_check(result['audit'], expected['audit'])
    require(before == input_signature((args, expansion_allowed, result)), 'Checker mutated supplied input')
    return dict(**{k:True for k in FORECAST_FLAGS}, counts=expected['audit']['counts'], query_rows=q,
        numerical_tolerance=ATOL, producer_modules_imported=False, computes_returns=False,
        shared_numerical_primitives=False, independent_primitive_path=str(NUMERICAL.relative_to(ROOT)),
        independent_primitive_sha256=NUMERICAL_SHA)
