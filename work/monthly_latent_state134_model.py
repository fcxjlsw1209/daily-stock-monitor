"""Prospective two-state SPY model. No I/O, fitting, or data loading on import.

All model inputs are simple daily returns. Parameter estimation uses only the
supplied training vector; the calendar wrapper controls its causal boundary.
Missing source history is unavailable. Numerical failures raise NumericalError.
"""
import math
from datetime import date

import numpy as np
import pandas as pd

MIN_OBSERVATIONS = 756
MAX_UPDATES = 300
LL_TOLERANCE = 1e-7
VARIANCE_FLOOR = 1e-4
PROBABILITY_FLOOR = 1e-8
MIN_SCALE = 1e-12
MIN_OCCUPANCY = 20.0
MIN_VARIANCE_RATIO = 1.05


class NumericalError(FloatingPointError):
    """A failed computation must not silently become a neutral market state."""


def _array(x, name, ndim=1):
    a = np.asarray(x)
    if a.ndim != ndim or a.dtype.kind not in 'iuf':
        raise ValueError(name + ' must be a real numeric array of the required dimension')
    a = a.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise NumericalError(name + ' contains nonfinite values')
    return a


def _finite(x, name):
    if not np.isfinite(x).all():
        raise NumericalError(name + ' is nonfinite')
    return x


def _probabilities(x, name, size):
    a = _array(x, name)
    if a.shape != (size,) or (a < 0).any() or not math.isclose(
            math.fsum(a), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(name + ' must be a normalized nonnegative probability vector')
    return a


def _transition(x):
    a = _array(x, 'transition', 2)
    if a.shape != (2, 2):
        raise ValueError('transition must be 2 by 2')
    for row in a:
        _probabilities(row, 'transition row', 2)
    return a


def stationary(trans):
    """Unique stationary probabilities for a two-state stochastic matrix."""
    a = _transition(trans)
    total = float(a[0, 1] + a[1, 0])
    if not total > 0:
        raise NumericalError('transition has no unique stationary distribution')
    return _finite(np.array([a[1, 0] / total, a[0, 1] / total]), 'stationary')


def _log(x):
    return math.log(x) if x > 0 else -math.inf


def _lse(values):
    m = max(values)
    if not math.isfinite(m):
        raise NumericalError('invalid log probability normalization')
    return m + math.log(math.fsum(math.exp(v - m) for v in values))


def _emissions(z, means, variances):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            b = -.5 * (math.log(2 * math.pi) + np.log(variances)[None, :]
                       + (z[:, None] - means[None, :]) ** 2 / variances[None, :])
        except FloatingPointError as exc:
            raise NumericalError('Gaussian log likelihood arithmetic failed') from exc
    return _finite(b, 'Gaussian log likelihood')


def _forward(b, initial, trans, predict_first=False):
    n = len(b)
    loga = np.array([[_log(v) for v in row] for row in trans])
    prior = np.array([_log(v) for v in initial])
    alpha = np.empty((n, 2)); scales = np.empty(n)
    for t in range(n):
        if t or predict_first:
            p = alpha[t - 1] if t else prior
            pred = [_lse([p[0] + loga[0, j], p[1] + loga[1, j]]) for j in range(2)]
        else:
            pred = prior
        v = [pred[j] + b[t, j] for j in range(2)]
        scales[t] = _lse(v)
        alpha[t] = [v[j] - scales[t] for j in range(2)]
    # log(0)=-inf is valid for an explicitly zero prior, not failed arithmetic.
    if np.isnan(alpha).any() or np.isposinf(alpha).any():
        raise NumericalError('invalid filtered log posterior')
    return alpha, _finite(scales, 'likelihood scales')


def _expectation(z, means, variances, initial, trans):
    """Scaled log-domain forward/backward sufficient statistics for EM."""
    b = _emissions(z, means, variances)
    alpha, scales = _forward(b, initial, trans)
    n = len(z); beta = np.zeros((n, 2))
    loga = np.array([[_log(v) for v in row] for row in trans])
    for t in range(n - 2, -1, -1):
        for i in range(2):
            beta[t, i] = _lse([loga[i, j] + b[t + 1, j] + beta[t + 1, j]
                               for j in range(2)]) - scales[t + 1]
    gamma = np.empty((n, 2)); counts = np.zeros((2, 2))
    for t in range(n):
        values = alpha[t] + beta[t]
        norm = _lse(values)
        gamma[t] = [math.exp(v - norm) for v in values]
        if t + 1 < n:
            values = [alpha[t, i] + loga[i, j] + b[t + 1, j] + beta[t + 1, j]
                      for i in range(2) for j in range(2)]
            norm = _lse(values)
            counts += np.array([math.exp(v - norm) for v in values]).reshape(2, 2)
    ll = math.fsum(scales)
    _finite(np.array([ll]), 'total likelihood')
    return dict(log_likelihood=ll, gamma=_finite(gamma, 'responsibilities'),
                counts=_finite(counts, 'transition counts'),
                posterior=np.array([math.exp(v) for v in alpha[-1]]))


def _floor_normalize(prob):
    p = np.maximum(prob, PROBABILITY_FLOOR)
    return _finite(p / math.fsum(p), 'floored probability')


def _maximization(z, expectation):
    gamma, counts = expectation['gamma'], expectation['counts']
    occupancy = np.array([math.fsum(gamma[:, j]) for j in range(2)])
    if (occupancy <= 0).any():
        raise NumericalError('zero state occupancy during EM')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            means = np.einsum('ni,n->i', gamma, z, optimize=False) / occupancy
            variances = np.einsum('ni,ni->i', gamma, (z[:, None] - means) ** 2,
                                  optimize=False) / occupancy
        except FloatingPointError as exc:
            raise NumericalError('M-step arithmetic failed') from exc
    _finite(means, 'means'); _finite(variances, 'variances')
    trans = np.empty((2, 2))
    for i in range(2):
        total = math.fsum(counts[i])
        if not total > 0:
            raise NumericalError('zero transition occupancy during EM')
        # The floor applies to PROBABILITIES, not to unnormalized counts.
        trans[i] = _floor_normalize(counts[i] / total)
    return means, np.maximum(variances, VARIANCE_FLOOR), _floor_normalize(gamma[0]), trans


def _empty_fit(n, reason, scaler=None):
    return dict(available=False, status='unavailable', reason=reason,
                fit_attempted=False, n_observations=int(n), scaler=scaler,
                parameters=None, filtered_posterior=None, log_likelihood=None,
                log_likelihood_history=[], updates=0, occupancy=None,
                variance_ratio=None, certificate=dict(converged=False))


def fit(x):
    """Fit the fixed model to a caller-selected training vector of simple returns."""
    x = _array(x, 'training returns'); n = len(x)
    if n < MIN_OBSERVATIONS:
        return _empty_fit(n, 'insufficient_observations')
    try:
        mean = math.fsum(x) / n
        with np.errstate(over='raise', invalid='raise'):
            delta = x - mean
        sd = math.sqrt(math.fsum(float(v) ** 2 for v in delta) / n)
    except (OverflowError, ValueError, FloatingPointError) as exc:
        raise NumericalError('training scaler arithmetic failed') from exc
    _finite(np.array([mean, sd]), 'training scaler')
    scaler = dict(mean=float(mean), sd=float(sd))
    if sd <= MIN_SCALE:
        return _empty_fit(n, 'degenerate_training_scale', scaler)
    z = _finite((x - mean) / sd, 'standardized training returns')
    means = np.array([0., 0.]); variances = np.array([.5, 2.])
    initial = np.array([.5, .5]); trans = np.array([[.97, .03], [.03, .97]])
    est = _expectation(z, means, variances, initial, trans)
    history = [float(est['log_likelihood'])]
    converged = False
    for update in range(1, MAX_UPDATES + 1):
        means, variances, initial, trans = _maximization(z, est)
        est = _expectation(z, means, variances, initial, trans)
        ll = float(est['log_likelihood']); improvement = (ll - history[-1]) / n
        history.append(ll)
        if improvement < -LL_TOLERANCE:
            raise NumericalError('material per-observation likelihood decrease')
        if abs(improvement) <= LL_TOLERANCE:
            converged = True
            break
    if not converged:
        raise NumericalError('EM did not converge within the fixed update limit')
    order = np.argsort(variances, kind='stable')
    means, variances, initial = means[order], variances[order], initial[order]
    trans = trans[np.ix_(order, order)]
    occupancy = np.array([math.fsum(est['gamma'][:, j]) for j in order])
    ratio = float(variances[1] / variances[0])
    reason = ('insufficient_effective_state_occupancy' if (occupancy < MIN_OCCUPANCY).any()
              else 'unseparated_state_variances' if ratio < MIN_VARIANCE_RATIO else None)
    return dict(available=reason is None, status='available' if reason is None else 'unavailable',
                reason=reason, fit_attempted=True, n_observations=n, scaler=scaler,
                parameters=dict(means=means.tolist(), variances=variances.tolist(),
                                initial=initial.tolist(), transition=trans.tolist()),
                filtered_posterior=est['posterior'][order].tolist(),
                log_likelihood=history[-1], log_likelihood_history=history, updates=update,
                occupancy=occupancy.tolist(), variance_ratio=ratio,
                certificate=dict(converged=True, all_arithmetic_finite=True,
                                 low_variance_state_first=True,
                                 likelihood_units='standardized_returns',
                                 last_improvement_per_observation=improvement,
                                 tolerance=LL_TOLERANCE, maximum_updates=MAX_UPDATES,
                                 variance_floor=VARIANCE_FLOOR,
                                 probability_floor_then_normalize=PROBABILITY_FLOOR))


def _model(model):
    if not isinstance(model, dict) or model.get('available') is not True:
        raise ValueError('inference requires an available fitted model')
    p = model['parameters']; s = model['scaler']
    means = _array(p['means'], 'means'); variances = _array(p['variances'], 'variances')
    if means.shape != (2,) or variances.shape != (2,) or (variances <= 0).any():
        raise ValueError('invalid Gaussian parameters')
    scaler = _array([s['mean'], s['sd']], 'scaler')
    if scaler[1] <= 0:
        raise ValueError('inference scale must be positive')
    return means, variances, _probabilities(p['initial'], 'initial', 2), _transition(p['transition']), scaler


def _standardize(x, scaler):
    x = _array(x, 'query returns')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            return _finite((x - scaler[0]) / scaler[1], 'standardized query returns')
        except FloatingPointError as exc:
            raise NumericalError('query standardization failed') from exc


def filter_path(x, model, initial=None, predict_first=False):
    """Nx2 filtered probabilities; explicit initial can be a preceding posterior."""
    if not isinstance(predict_first, (bool, np.bool_)):
        raise ValueError('predict_first must be boolean')
    means, variances, pi, trans, scaler = _model(model)
    if initial is not None:
        pi = _probabilities(initial, 'inference initial', 2)
    z = _standardize(x, scaler)
    if not len(z):
        return np.empty((0, 2))
    alpha, _ = _forward(_emissions(z, means, variances), pi, trans, predict_first)
    return np.array([[math.exp(v) for v in row] for row in alpha])


def memoryless(x, model):
    """Same emissions and scaler; stationary prior resets at every observation."""
    means, variances, _, trans, scaler = _model(model)
    z = _standardize(x, scaler); b = _emissions(z, means, variances)
    pi = stationary(trans); lp = [_log(v) for v in pi]
    result = np.empty((len(z), 2))
    for i in range(len(z)):
        values = [lp[j] + b[i, j] for j in range(2)]; norm = _lse(values)
        result[i] = [math.exp(v - norm) for v in values]
    return result


def _dates(values, name):
    index = pd.DatetimeIndex(values)
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(name + ' must be sorted unique finite dates')
    if not (index == index.normalize()).all():
        raise ValueError(name + ' must contain normalized session dates')
    # A supplied exchange calendar may carry UTC midnight labels.
    return [v.date() for v in index]


def _returns(prices):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            return _finite(prices[1:] / prices[:-1] - 1.0, 'simple returns')
        except FloatingPointError as exc:
            raise NumericalError('simple return arithmetic failed') from exc


def annual_probabilities(price, sessions, signal_dates):
    """Annual past-only fit and low-state probabilities on explicit signal dates.

    Only caller-supplied official sessions at or before the final signal are
    inspected. The function does not infer holidays or load any source files.
    The audit is JSON-safe; unavailable output probabilities remain NaN.
    """
    if not isinstance(price, pd.Series):
        raise ValueError('price must be a pandas Series')
    cal = _dates(sessions, 'official sessions'); signals = _dates(signal_dates, 'signals')
    if not signals:
        return dict(filtered=np.empty(0), memoryless=np.empty(0),
                    audit=dict(years=[], signals=[], new_model_fits=0))
    if not cal or any(d not in set(cal) for d in signals):
        raise ValueError('all signal dates must be official sessions')
    # Index validation is structural; future price VALUES are never inspected.
    pdates = _dates(price.index, 'price index')
    end = signals[-1]; keep = [i for i, d in enumerate(pdates) if d <= end]
    selected = price.iloc[keep]
    if selected.dtype.kind not in 'iuf':
        raise ValueError('historical prices must be numeric')
    observed = {pdates[i]: float(price.iloc[i]) for i in keep}
    cal = [d for d in cal if d <= end]; positions = {d:i for i,d in enumerate(cal)}
    values = np.array([observed.get(d, math.nan) for d in cal])
    filtered = np.full(len(signals), np.nan); independent = np.full(len(signals), np.nan)
    years = []; signal_audit = [None] * len(signals)
    for year in sorted({d.year for d in signals}):
        cutoff = date(year - 1, 12, 31); lower = date(year - 6, 12, 31)
        ti = [i for i,d in enumerate(cal) if lower < d <= cutoff]
        si = [i for i,d in enumerate(signals) if d.year == year]
        row = dict(year=year, cutoff=str(cutoff), lower_bound_exclusive=str(lower),
                   training_start=str(cal[ti[0]]) if ti else None,
                   training_end=str(cal[ti[-1]]) if ti else None,
                   training_observations=len(ti), preceding_price_date=None,
                   fit_attempted=False, status='unavailable', reason=None,
                   model=None, first_invalid_query_date=None, query_observations=0)
        if not ti or ti[0] == 0 or cal[0] > lower:
            row['reason'] = 'incomplete_training_calendar'
        elif len(ti) < MIN_OBSERVATIONS:
            row['reason'] = 'insufficient_observations'
        else:
            row['preceding_price_date'] = str(cal[ti[0] - 1])
            train_prices = values[ti[0] - 1:ti[-1] + 1]
            if not np.isfinite(train_prices).all() or (train_prices <= 0).any():
                row['reason'] = 'incomplete_training_prices'
            else:
                row['model'] = fit(_returns(train_prices))
                row['fit_attempted'] = row['model']['fit_attempted']
                row['reason'] = row['model']['reason']; row['status'] = row['model']['status']
        if row['status'] == 'available':
            last_query = positions[signals[si[-1]]]
            q0 = ti[-1] + 1
            query_dates = cal[q0:last_query + 1]
            query_prices = values[q0 - 1:last_query + 1]
            invalid = np.flatnonzero(~np.isfinite(query_prices) | (query_prices <= 0))
            # Invalid price j poisons its own return (j-1) and all later states.
            valid_n = max(0, int(invalid[0]) - 1) if len(invalid) else len(query_dates)
            if valid_n < len(query_dates):
                row['first_invalid_query_date'] = str(query_dates[valid_n])
            row['query_observations'] = len(query_dates)
            query = _returns(query_prices[:valid_n + 1])
            fp = filter_path(query, row['model'], row['model']['filtered_posterior'], True)
            mp = memoryless(query, row['model'])
            for i in si:
                offset = positions[signals[i]] - q0
                if offset < valid_n:
                    filtered[i], independent[i] = fp[offset, 0], mp[offset, 0]
        for i in si:
            ok = math.isfinite(filtered[i])
            signal_audit[i] = dict(signal_index=i, signal_date=str(signals[i]), year=year,
                                   available=ok, reason=None if ok else row['reason'] or 'invalid_query_history',
                                   first_invalid_query_date=None if ok else row['first_invalid_query_date'],
                                   filtered_probability=float(filtered[i]) if ok else None,
                                   memoryless_probability=float(independent[i]) if ok else None)
        years.append(row)
    return dict(filtered=filtered, memoryless=independent,
                audit=dict(years=years, signals=signal_audit,
                           new_model_fits=sum(row['fit_attempted'] for row in years)))
