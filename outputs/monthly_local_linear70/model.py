"""Same-security local residual constant/linear corrections to frozen-style Ridge.

Only already matured annual training rows enter either correction. This module
constructs no labels, cross-sectional ranks, portfolio scores or trading orders.
"""
from pathlib import Path
import importlib.util

import numpy as np
import threadpoolctl
from threadpoolctl import threadpool_limits

O = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = O/'monthly_target_definition70/model.py'
_SPEC = importlib.util.spec_from_file_location('local_linear70_original', ORIGINAL_PATH)
original = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(original)
numeric = original.numeric
TARGETS = original.TARGETS
ALPHA = 10.
HALF_LIFE_MONTHS = 12.
MIN_MONTHS = 18
MIN_OWN_MONTHS = 18
BANDWIDTH = .5
INTERCEPT_PENALTY = 12.
SLOPE_PENALTY = 10.
MODES = ('local_constant', 'local_linear')


def dependency_files():
    return (Path(__file__).resolve(), ORIGINAL_PATH, Path(numeric.__file__).resolve(),
            Path(threadpoolctl.__file__).resolve())


def _minimum(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')


def _symbols(value, length, name):
    value = np.asarray(value)
    if value.shape != (length,) or any(not isinstance(x, (str, np.str_)) or not x for x in value):
        raise ValueError(name+' must contain one explicit nonempty security ID per row')
    return value.astype(str, copy=True)


def _finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite '+name)


def _ranks(name, value):
    _finite(name, value)
    if (np.abs(value) > .5).any():
        raise ValueError(name+' must retain original centered percentile bounds')


def local_weights(dates, cutoff, X, query):
    """Unnormalized annual-cutoff decay times a kernel in ORIGINAL rank units.

    There is one observation per security/month. The most recent possible row
    has unit recency weight; no rescaling to a sum, row count or effective size
    is performed. Pooled standardization is intentionally absent here.
    """
    dates = numeric._days(dates, 'own dates')
    cutoff = np.datetime64(cutoff, 'D')
    X, query = np.asarray(X, float), np.asarray(query, float)
    if X.shape != (len(dates), 10) or query.shape != (10,):
        raise ValueError('Local kernel requires original ten-column rank features')
    if np.isnat(cutoff) or (dates > cutoff).any():
        raise ValueError('Local recency cutoff must not precede a used training signal')
    _ranks('own training ranks', X)
    _ranks('query ranks', query)
    ages = cutoff.astype('datetime64[M]').astype(int)-dates.astype('datetime64[M]').astype(int)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        distance = np.mean((X-query)**2, axis=1)
        weights = np.exp2(-ages/HALF_LIFE_MONTHS)*np.exp(-distance/(2*BANDWIDTH**2))
    _finite('local weights', weights)
    if (weights < 0).any():
        raise ValueError('Negative local weight')
    return weights


def local_corrections(X, query, scale, residual, weights):
    """Return matched constant and linear query corrections, plus a fit audit.

    The query-centered design has row [1, (X_h-X_query)/pooled_scale], so the
    linear correction at the query is its penalized intercept. Both arms use
    precisely the supplied rows, residuals and unnormalized weights. A failed
    solve aborts; it is never disguised as absent history or a zero prediction.
    """
    X, query, scale = np.asarray(X, float), np.asarray(query, float), np.asarray(scale, float)
    residual, weights = np.asarray(residual, float), np.asarray(weights, float)
    if X.ndim != 2 or X.shape[1] != 10 or query.shape != (10,) or scale.shape != (10,):
        raise ValueError('Local fit requires original ten features and pooled scales')
    n = len(X)
    if residual.shape != (n,) or weights.shape != (n,):
        raise ValueError('Local fit rows, residuals and weights must align')
    for name, values in (('own features', X), ('query', query), ('pooled scale', scale),
                         ('mature residual', residual), ('local weights', weights)):
        _finite(name, values)
    if (scale <= 0).any() or (weights < 0).any():
        raise ValueError('Scales must be positive and weights nonnegative')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        mass = float(weights.sum())
    _finite('local weight mass', mass)
    if mass == 0. or not np.any(residual != 0.):
        return 0., 0., dict(weight_mass=mass, effective_samples=0. if mass == 0. else
                            float(mass**2/np.einsum('n,n->', weights, weights, optimize=False)),
                            status='zero_weight' if mass == 0. else 'zero_residual')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        d = (X-query)/scale
        Z = np.column_stack((np.ones(n), d))
        lhs = np.einsum('ni,n,nj->ij', Z, weights, Z, optimize=False)
        lhs += np.diag([INTERCEPT_PENALTY]+[SLOPE_PENALTY]*10)
        rhs = np.einsum('ni,n,n->i', Z, weights, residual, optimize=False)
        constant = float(np.einsum('n,n->', weights, residual, optimize=False)/(mass+INTERCEPT_PENALTY))
        effective = float(mass**2/np.einsum('n,n->', weights, weights, optimize=False))
    for name, values in (('query-centered design', d), ('local lhs', lhs), ('local rhs', rhs),
                         ('constant correction', constant), ('effective sample size', effective)):
        _finite(name, values)
    theta = np.linalg.solve(lhs, rhs)
    _finite('local coefficients', theta)
    return constant, float(theta[0]), dict(weight_mass=mass, effective_samples=effective, status='fit')


def _fit_pooled(X, y, weights):
    return numeric._fit_ridge(X, y, weights, ALPHA)


def annual_forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
                     event_symbols, event_X, event_good, *, min_months=MIN_MONTHS,
                     min_own_months=MIN_OWN_MONTHS):
    """Return source_log/local_constant/local_linear raw vectors and audit.

    `available` is independently reproduced original pooled availability. Both
    local outputs start as exact copies of its raw predictions. `local_available`
    only audits sufficient own history/positive local mass; it MUST NOT filter
    event eligibility or cross-sectional ranking population. Labels are neither
    rebuilt nor recentered. Local residuals are in-sample in the already mature
    annual cohort, not historical OOF residuals. Production defaults are fixed;
    the two minimum-month arguments support small synthetic tests only.
    """
    _minimum(min_months, 'min_months')
    _minimum(min_own_months, 'min_own_months')
    dates, ends = numeric._days(dates, 'dates'), numeric._days(ends, 'ends')
    event_dates = numeric._days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or (ends <= dates).any() or np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    symbols, event_symbols = _symbols(symbols, n, 'symbols'), _symbols(event_symbols, q, 'event_symbols')
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    if X.shape != (n, 10) or event_X.shape != (q, 10):
        raise ValueError('Original ten-feature matrices are required')
    common_good = original._bool(common_good, n, 'common_good')
    event_good = original._bool(event_good, q, 'event_good')
    if set(targets) != set(TARGETS):
        raise ValueError('All four original target arrays are required')
    ys = {key: np.asarray(targets[key], float) for key in TARGETS}
    if any(y.shape != (n,) for y in ys.values()):
        raise ValueError('Original targets must have exact training row length')
    usable = common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        usable &= np.isfinite(y)
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        selected = months == month
        if (usable & selected).sum() < 2:
            usable[selected] = False
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    annual, local_audit = [], []
    with threadpool_limits(limits=1):
        ref = original.annual_forecasts(dates, ends, X, ys, common_good, event_dates,
                                        event_X, event_good, min_months=min_months)
        raw = {key: ref['log_excess'].copy() for key in ('source_log',)+MODES}
        available = ref['available'].copy()
        local_available = np.zeros(q, bool)
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = numeric._years_before(cutoff)
            train = usable & (dates > start) & (ends <= cutoff)
            predict = (years == year) & available
            ids = np.flatnonzero(train)
            count = numeric._month_count(dates[train])
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                          training_rows=len(ids), training_months=count, prediction_rows=int(predict.sum()),
                          first_training_signal=str(dates[train].min()) if len(ids) else None,
                          latest_label_end=str(ends[train].max()) if len(ids) else None,
                          status='fit' if count >= min_months else 'cold_start')
            if predict.any():
                if count < min_months:
                    raise AssertionError('Independent pooled availability disagrees with source cohort')
                # Only actual mature/predicted rows are rank-validated. Poisoned
                # unused/future rows cannot make an earlier year fail.
                _ranks('actual pooled training ranks', X[train])
                _ranks('actual query ranks', event_X[predict])
                identities = list(zip(months[train].astype(str), symbols[train]))
                if len(set(identities)) != len(identities):
                    raise ValueError('Each actual training security must have one row per signal month')
                weights = numeric.month_balanced_weights(dates[train], cutoff, HALF_LIFE_MONTHS)
                pooled = _fit_pooled(X[train], ys['log_excess'][train], weights)
                replay = numeric._predict(pooled, event_X[predict])
                np.testing.assert_array_equal(replay, raw['source_log'][predict])
                with np.errstate(over='raise', invalid='raise'):
                    residual = ys['log_excess'][train]-numeric._predict(pooled, X[train])
                _finite('mature pooled residuals', residual)
                record['weight_sum'] = float(weights.sum())
                by_symbol = {s: np.flatnonzero(symbols[train] == s) for s in set(symbols[train])}
                for j in np.flatnonzero(predict):
                    own = by_symbol.get(event_symbols[j], np.empty(0, dtype=int))
                    own_ids = ids[own]
                    own_months = numeric._month_count(dates[own_ids])
                    detail = dict(event_row=int(j), date=str(event_dates[j]), symbol=event_symbols[j],
                                  year=int(year), cutoff=str(cutoff), own_rows=len(own), own_months=own_months,
                                  latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                                  status='insufficient_own_history', weight_mass=0., effective_samples=0.,
                                  constant_correction=0., linear_correction=0.)
                    if own_months >= min_own_months:
                        w = local_weights(dates[own_ids], cutoff, X[own_ids], event_X[j])
                        constant, linear, info = local_corrections(X[own_ids], event_X[j], pooled[1], residual[own], w)
                        detail.update(info, constant_correction=constant, linear_correction=linear)
                        local_available[j] = info['weight_mass'] > 0.
                        for mode, correction in zip(MODES, (constant, linear)):
                            if correction != 0.:
                                with np.errstate(over='raise', invalid='raise'):
                                    raw[mode][j] += correction
                    local_audit.append(detail)
            annual.append(record)
    for key, values in raw.items():
        if not np.array_equal(np.isfinite(values), available):
            raise ValueError(key+' changed source prediction availability or became nonfinite')
    return dict(raw=raw, available=available, local_available=local_available, audit=dict(
        modes=list(MODES), feature_count=10, target='unchanged cached centered log_excess',
        alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS, min_months=int(min_months),
        min_own_months=int(min_own_months), bandwidth=BANDWIDTH,
        intercept_penalty=INTERCEPT_PENALTY, slope_penalty=SLOPE_PENALTY,
        same_local_rows_residuals_and_weights=True, local_weights_normalized=False,
        local_recency_clock='annual prior December cutoff calendar-month age',
        kernel='mean squared difference in original centered percentile plain10 units',
        linear_design='query-centered original X differences divided by pooled annual training scales',
        residuals='in-sample residuals within already matured annual training cohort, not OOF',
        source_control='independent original four-target annual model and exact log fit replay',
        no_own_history='exact source raw copy; never remove a query or alter eligibility',
        source=ref['audit'], annual=annual, local=local_audit))
