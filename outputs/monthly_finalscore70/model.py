"""Constrained annual final scores on four frozen centered-rank columns.

Only caller-supplied arrays are used. No prices, ranks, holdings or returns are
constructed here. SLSQP status is diagnostic; the fixed primal and global
Frank-Wolfe certificates alone determine numerical admission.
"""
from functools import lru_cache
import importlib
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import scipy.optimize
from threadpoolctl import threadpool_limits

VALIDATOR_PATH = Path(__file__).resolve().parents[1] / 'monthly_listwise_top3_70/model.py'
_SPEC = importlib.util.spec_from_file_location('finalscore119_frozen_validation', VALIDATOR_PATH)
legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(legacy)

COLUMNS = (0, 2, 3, 5)
PRIOR = np.array([.65, .20, .10, .05])
MODES = ('ordinal_final', 'listwise_final')
TARGETS = legacy.TARGETS
MIN_MONTHS, HALF_LIFE_MONTHS = 18, 12.
ALPHA, TEMPERATURE = 1., .1
FTOL, MAX_ITERATIONS = 1e-15, 2000
PRIMAL_TOLERANCE, GAP_TOLERANCE = 1e-10, 1e-8
VERTEX_DECIMALS = 12


def dependency_files():
    names = ('scipy', 'scipy.optimize', 'scipy.optimize._minimize',
             'scipy.optimize._slsqp_py', 'scipy.optimize._slsqp',
             'scipy.optimize._optimize', 'scipy.optimize._numdiff',
             'scipy.optimize._constraints', 'scipy.optimize._differentiable_functions',
             'scipy._lib._util')
    paths = {Path(__file__).resolve(), VALIDATOR_PATH, *legacy.dependency_files()}
    paths.update(Path(importlib.import_module(name).__file__).resolve() for name in names)
    return tuple(sorted(paths))


def constraints():
    """Return A,b,names for A @ w <= b; sum(w)=1 is separate."""
    a = [-row for row in np.eye(4)]
    b = [0.] * 4
    names = ['nonnegative_' + str(i) for i in range(4)]
    a.extend([[-1., 0., 0., 0.], [0., 1., 0., 0.],
              [0., 0., 1., 0.], [0., 0., 0., 1.]])
    b.extend([-.5, .3, .2, .1])
    names.extend(['minimum_core', 'maximum_vol', 'maximum_sector', 'maximum_cmf'])
    for sign in itertools.product((-1., 1.), repeat=4):
        a.append(sign)
        b.append(float(np.einsum('i,i->', sign, PRIOR, optimize=False)) + .20)
        names.append('l1_' + ''.join('+' if x > 0 else '-' for x in sign))
    return np.asarray(a), np.asarray(b), names


def primal_violation(coefficient):
    w = np.asarray(coefficient, float)
    if w.shape != (4,):
        raise ValueError('Four coefficients required')
    legacy._finite('coefficients', w)
    a, b, _ = constraints()
    return float(max(0., abs(float(w.sum()) - 1.),
                     float(np.max(np.einsum('ni,i->n', a, w, optimize=False) - b))))


@lru_cache(maxsize=1)
def _vertices():
    a, b, _ = constraints()
    found = {}
    for active in itertools.combinations(range(len(b)), 3):
        matrix = np.vstack((np.ones(4), a[list(active)]))
        if np.linalg.matrix_rank(matrix) < 4:
            continue
        w = np.linalg.solve(matrix, np.r_[1., b[list(active)]])
        legacy._finite('polytope vertex', w)
        if primal_violation(w) <= PRIMAL_TOLERANCE:
            key = tuple(np.round(w, VERTEX_DECIMALS))
            found.setdefault(key, w)
    if not found:
        raise FloatingPointError('No feasible polytope vertices')
    result = np.stack([found[key] for key in sorted(found)])
    result.setflags(write=False)
    return result


def vertices():
    return _vertices().copy()


def certificate(coefficient, gradient):
    w, g = np.asarray(coefficient, float), np.asarray(gradient, float)
    if w.shape != (4,) or g.shape != (4,):
        raise ValueError('Four coefficients and gradient entries required')
    legacy._finite('coefficients', w)
    legacy._finite('gradient', g)
    gaps = np.einsum('vi,i->v', w - _vertices(), g, optimize=False)
    legacy._finite('vertex directional gaps', gaps)
    worst = int(np.argmax(gaps))
    gap, violation = float(gaps[worst]), primal_violation(w)
    return dict(primal_violation=violation, fw_gap=gap,
                worst_vertex=_vertices()[worst].tolist(), vertex_count=len(_vertices()),
                primal_tolerance=PRIMAL_TOLERANCE, fw_gap_tolerance=GAP_TOLERANCE,
                certified=violation <= PRIMAL_TOLERANCE and gap <= GAP_TOLERANCE)


def _inputs(X, ordinal, row_weights, groups):
    x, y, weights = np.asarray(X, float), np.asarray(ordinal, float), np.asarray(row_weights, float)
    if x.ndim != 2 or x.shape[1] != 4 or not len(x) or y.shape != (len(x),) or weights.shape != y.shape:
        raise ValueError('Expected N by four features and aligned target/weights')
    for name, value in [('features', x), ('ordinal', y), ('row weights', weights)]:
        legacy._finite(name, value)
    if (weights <= 0).any() or abs(float(weights.sum()) - 1.) > 1e-12:
        raise ValueError('Positive row weights normalized to total one required')
    parsed, q = [], np.empty(len(y))
    for group in groups:
        ids = np.asarray(group)
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
                or (ids < 0).any() or (ids >= len(y)).any()):
            raise ValueError('Invalid list indices')
        parsed.append(ids)
        q[ids] = legacy.top3_distribution(y[ids])
    if not parsed or not np.array_equal(np.sort(np.concatenate(parsed)), np.arange(len(y))):
        raise ValueError('Complete disjoint monthly list partition required')
    monthly = np.array([weights[ids].sum() for ids in parsed])
    return x, y, weights, parsed, q, monthly


def predict_scores(X, coefficient):
    """Four terms in fixed left-to-right order, independent of array layout."""
    x, w = np.asarray(X, float), np.asarray(coefficient, float)
    if x.ndim != 2 or x.shape[1] != 4 or w.shape != (4,):
        raise ValueError('Four raw rank columns and coefficients required')
    legacy._finite('prediction features', x)
    legacy._finite('prediction coefficient', w)
    with np.errstate(over='raise', invalid='raise'):
        result = ((x[:, 0]*w[0] + x[:, 1]*w[1]) + x[:, 2]*w[2]) + x[:, 3]*w[3]
    legacy._finite('direct prediction', result)
    return result


def _objective(mode, data, coefficient):
    x, y, weights, groups, q, monthly = data
    w = np.asarray(coefficient, float)
    if w.shape != (4,):
        raise ValueError('Four coefficients required')
    legacy._finite('coefficients', w)
    delta = w - PRIOR
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        value = .5 * ALPHA * float(np.einsum('i,i->', delta, delta, optimize=False))
        gradient, hessian = ALPHA * delta.copy(), ALPHA * np.eye(4)
        if mode == 'ordinal_final':
            residual = predict_scores(x, w) - y / 2.
            value += .5 * float(np.einsum('n,n,n->', weights, residual, residual, optimize=False))
            gradient += np.einsum('ni,n,n->i', x, weights, residual, optimize=False)
            hessian += np.einsum('ni,n,nj->ij', x, weights, x, optimize=False)
        elif mode == 'listwise_final':
            for ids, mass in zip(groups, monthly):
                # A common feature row cancels from the listwise likelihood.
                z = (x[ids] - x[ids[0]]) / TEMPERATURE
                score = predict_scores(z, w)
                shifted = score - np.max(score)
                exps = np.exp(shifted)
                probabilities = exps / exps.sum()
                value += float(mass) * (float(np.log(exps.sum())) -
                    float(np.einsum('i,i->', q[ids], shifted, optimize=False)))
                gradient += mass * np.einsum('ni,n->i', z, probabilities-q[ids], optimize=False)
                center = np.einsum('ni,n->i', z, probabilities, optimize=False)
                zc = z - center
                hessian += mass * np.einsum('ni,n,nj->ij', zc, probabilities, zc, optimize=False)
        else:
            raise ValueError('Unknown final-score loss')
    for name, item in [('objective', value), ('gradient', gradient), ('Hessian', hessian)]:
        legacy._finite(name, item)
    return float(value), gradient, hessian


def value_gradient_hessian(mode, X, ordinal, row_weights, groups, coefficient):
    return _objective(mode, _inputs(X, ordinal, row_weights, groups), coefficient)


def fit_model(mode, X, ordinal, row_weights, groups):
    data = _inputs(X, ordinal, row_weights, groups)
    if mode not in MODES:
        raise ValueError('Unknown final-score loss')
    a, b, _ = constraints()
    def fun(w):
        value, gradient, _ = _objective(mode, data, w)
        return value, gradient
    result = scipy.optimize.minimize(fun, PRIOR.copy(), jac=True, method='SLSQP',
        constraints=[dict(type='eq', fun=lambda w: float(w.sum())-1., jac=lambda w: np.ones(4)),
                     dict(type='ineq', fun=lambda w: b-np.einsum('ni,i->n', a, w, optimize=False), jac=lambda w: -a)],
        options=dict(ftol=FTOL, maxiter=MAX_ITERATIONS, disp=False))
    w = np.asarray(result.x, float)
    value, gradient, hessian = _objective(mode, data, w)
    cert = certificate(w, gradient)
    if not cert['certified']:
        raise FloatingPointError('Final-score certificate failed: primal=%r FW=%r status=%r' %
                                 (cert['primal_violation'], cert['fw_gap'], result.status))
    audit = dict(cert, coefficient=w.tolist(), objective=value, gradient=gradient.tolist(),
        hessian_min_eigenvalue=float(np.linalg.eigvalsh(hessian).min()),
        solver='SLSQP', solver_success=bool(result.success), solver_status=int(result.status),
        solver_message=str(result.message), iterations=int(result.nit), ftol=FTOL,
        max_iterations=MAX_ITERATIONS, start=PRIOR.tolist(),
        admission='finite objective/coefficients/derivatives and primal/FW certificates; solver status is diagnostic')
    return w, audit


def _validate(dates, ends, symbols, X, ordinal, targets, train_eligible, common_good,
              event_dates, event_symbols, event_X, event_good):
    dates, ends = legacy._days(dates, 'training dates'), legacy._days(ends, 'label ends')
    event_dates = legacy._days(event_dates, 'event dates')
    n, count = len(dates), len(event_dates)
    symbols = legacy._symbols(symbols, n, 'training symbols')
    event_symbols = legacy._symbols(event_symbols, count, 'event symbols')
    if (ends.shape != (n,) or (ends <= dates).any()
            or np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1)):
        raise ValueError('Native next-calendar-month ends required')
    if len(set(zip(dates.astype(str), symbols))) != n or len(set(zip(event_dates.astype(str), event_symbols))) != count:
        raise ValueError('Duplicate date/security identity')
    X, event_X, y = np.asarray(X, float), np.asarray(event_X, float), np.asarray(ordinal, float)
    if X.shape != (n, 10) or event_X.shape != (count, 10) or y.shape != (n,):
        raise ValueError('Original plain10 and ordinal dimensions required')
    eligible = legacy._bool(train_eligible, n, 'train_eligible')
    common = legacy._bool(common_good, n, 'common_good')
    event_good = legacy._bool(event_good, count, 'event_good')
    if (common & ~eligible).any() or np.isfinite(y[~eligible]).any():
        raise ValueError('Future-valid target outside original eligible list')
    if set(targets) != set(TARGETS):
        raise ValueError('All four original targets required')
    ys = {key:np.asarray(targets[key], float) for key in TARGETS}
    if any(item.shape != (n,) for item in ys.values()):
        raise ValueError('Target dimensions')
    for name, value in [('plain10', X), ('event plain10', event_X), ('ordinal', y), *ys.items()]:
        if np.isinf(value).any():
            raise FloatingPointError('Infinite ' + name)
    usable, cohorts = np.zeros(n, bool), []
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        all_ids = np.flatnonzero(months == month)
        if len(np.unique(dates[all_ids])) != 1 or len(np.unique(ends[all_ids])) != 1:
            raise ValueError('One signal and label end per month required')
        ids = all_ids[eligible[all_ids]]
        reasons = []
        if not len(ids): reasons.append('no_original_eligible')
        if len(ids) and not common[ids].all(): reasons.append('not_all_common_good')
        if len(ids) and not np.isfinite(X[ids]).all(): reasons.append('not_all_finite_plain10')
        if len(ids) and not np.isfinite(y[ids]).all(): reasons.append('not_all_finite_ordinal')
        if len(ids) and not all(np.isfinite(item[ids]).all() for item in ys.values()):
            reasons.append('not_all_four_targets_finite')
        if len(ids) and np.isfinite(y[ids]).all(): legacy.validate_ordinal(y[ids])
        if not reasons: usable[ids] = True
        cohorts.append(dict(month=str(month), signal_date=str(dates[all_ids[0]]),
            label_end=str(ends[all_ids[0]]), eligible_indices=ids.tolist(), eligible_count=len(ids),
            complete=not reasons, at_most_three=len(ids) <= 3, reasons=reasons))
    valid = event_good & np.isfinite(event_X).all(axis=1)
    return dates, ends, X[:, COLUMNS], y, months, usable, cohorts, event_dates, event_X[:, COLUMNS], valid


def annual_forecasts(dates, ends, symbols, Xplain10, ordinal, targets, train_eligible,
                     common_good, event_dates, event_symbols, event_Xplain10, event_good,
                     *, min_months=MIN_MONTHS):
    if isinstance(min_months, bool) or not isinstance(min_months, (int, np.integer)) or min_months < 1:
        raise ValueError('Positive min_months required')
    dates, ends, x, y, months, usable, cohorts, event_dates, event_x, valid = _validate(
        dates, ends, symbols, Xplain10, ordinal, targets, train_eligible, common_good,
        event_dates, event_symbols, event_Xplain10, event_good)
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    raw = {mode:np.full(len(event_dates), np.nan) for mode in MODES}
    annual, fits = [], 0
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = legacy.numeric._years_before(cutoff)
            mask = usable & (dates > start) & (ends <= cutoff)
            ids = np.flatnonzero(mask)
            queries = np.flatnonzero(valid & (years == year))
            count = legacy.numeric._month_count(dates[mask])
            status = 'no_queries' if not len(queries) else 'cold_start_prior' if count < min_months else 'fit'
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                training_indices=ids.tolist(), training_indices_sha256=legacy.array_hash(ids),
                query_indices=queries.tolist(), training_rows=len(ids), training_months=count,
                prediction_rows=len(queries), latest_label_end=str(ends[mask].max()) if len(ids) else None,
                status=status, scaler='none', prior_fallback=status == 'cold_start_prior')
            coefficients = {mode:PRIOR.copy() for mode in MODES}
            if status == 'fit':
                weights = legacy.numeric.month_balanced_weights(dates[mask], cutoff, HALF_LIFE_MONTHS)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    weights = weights / weights.sum()
                legacy._finite('normalized annual weights', weights)
                groups = [np.flatnonzero(months[mask] == month) for month in np.unique(months[mask])]
                data = _inputs(x[mask], y[mask], weights, groups)
                for mode in MODES:
                    coefficients[mode], record[mode] = fit_model(mode, x[mask], y[mask], weights, groups)
                record.update(weights=weights.tolist(), weights_sha256=legacy.array_hash(weights),
                    weight_sum=float(weights.sum()), q=data[4].tolist(), q_sha256=legacy.array_hash(data[4]),
                    group_months=np.unique(months[mask]).astype(str).tolist(),
                    group_rows=[g.tolist() for g in groups], group_weights=data[5].tolist())
                fits += 1
            else:
                for mode in MODES:
                    record[mode] = dict(coefficient=PRIOR.tolist(), fitted=False)
            for mode in MODES:
                raw[mode][queries] = predict_scores(event_x[queries], coefficients[mode])
            annual.append(record)
    for mode in MODES:
        if not np.array_equal(np.isfinite(raw[mode]), valid):
            raise FloatingPointError('Final-score availability mismatch')
    return dict(raw=raw, available=valid.copy(), audit=dict(
        columns=list(COLUMNS), prior=PRIOR.tolist(), alpha=ALPHA, temperature=TEMPERATURE,
        scaler='none', half_life_months=HALF_LIFE_MONTHS, min_months=int(min_months),
        training_usable=usable.tolist(), training_usable_sha256=legacy.array_hash(usable),
        cohorts=cohorts, annual=annual, ordinal_fits=fits, listwise_fits=fits, actual_model_fits=2*fits,
        weights='original month-balanced12-month decay, normalized to total one',
        output='direct X[:,[0,2,3,5]] @ coefficient; no rank or probability transform',
        availability='event_good and all original plain10 finite; cold start uses prior',
        at_most_three_policy='included; uniform top3 q', own_history_filter=False,
        vertex_count=len(_vertices()), vertex_sha256=legacy.array_hash(_vertices())))
