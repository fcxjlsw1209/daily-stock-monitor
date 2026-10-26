"""Small, constrained annual calibration of four fixed guidance-factor budgets.

This module neither constructs features/labels nor selects securities. Numeric
training values are inspected only after the annual metadata clock admits their
whole signal month. Future values never determine a previous year's fit.
"""
from itertools import combinations, product
import importlib.util
from pathlib import Path

import numpy as np

NUMERIC_PATH = Path(__file__).resolve().parents[1]/'monthly_architecture_restart/residual_model.py'
_SPEC = importlib.util.spec_from_file_location('_budget_calibration70_numeric', NUMERIC_PATH)
numeric = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(numeric)

PRIOR = np.array([.65, .15, .10, .10], dtype=float)
PRIOR.flags.writeable = False
CORE_MIN = .60
RADIUS = .10
PENALTY = .10
TOL = 1e-9
MIN_MONTHS = 18
HALF_LIFE_MONTHS = 24.
FEATURES = ('M', 'R', 'V', 'P')


def dependency_files():
    return (Path(__file__).resolve(), NUMERIC_PATH)


def constraints():
    """Return independent copies of A,b,names for A @ w <= b."""
    signs = np.array(list(product((-1., 1.), repeat=4)))
    A = np.vstack([-np.eye(4), [-1., 0., 0., 0.], signs])
    b = np.r_[np.zeros(4), -CORE_MIN,
              RADIUS+np.einsum('ij,j->i', signs, PRIOR, optimize=False)]
    names = ['nonnegative_'+s for s in FEATURES]+['momentum_minimum']
    names += ['l1_'+''.join('+' if v > 0 else '-' for v in s) for s in signs]
    return A, b, names


def _finite(name, values):
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite '+name)


def _bounded(name, values):
    if not np.isfinite(values).all() or (np.abs(values) > .5).any():
        raise ValueError(name+' must be finite and bounded in [-.5,.5]')


def _bool(values, n, name):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind != 'b':
        raise ValueError(name+' must be a length-N boolean array')
    return result


def _timeline(dates, ends):
    dates = numeric._days(dates, 'train_dates')
    ends = numeric._days(ends, 'train_ends')
    if ends.shape != dates.shape:
        raise ValueError('Training dates and ends must align')
    months = dates.astype('datetime64[M]')
    if ((ends <= dates).any() or
            (ends.astype('datetime64[M]') != months+1).any()):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    for month in np.unique(months):
        rows = months == month
        if len(np.unique(dates[rows])) != 1 or len(np.unique(ends[rows])) != 1:
            raise ValueError('Each signal calendar month needs one signal date and one label end')
    return dates, ends, months


def objective(w, X, y, weights):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        residual = np.einsum('ij,j->i', X, w, optimize=False)-y
        value = (np.einsum('i,i,i->', weights, residual, residual, optimize=False)
                 + PENALTY*np.sum((w-PRIOR)**2))
    _finite('objective', value)
    return float(value)


def fit_weights(X, y, weights):
    """Solve the fixed strictly convex four-dimensional QP, or fail explicitly.

    Enumeration is deterministic: lexicographic subsets of zero through three
    inequalities, each combined with sum(w)=1. Only independent active rows are
    solved; candidates must pass primal, dual and stationarity checks. No solver
    fallback, fitted intercept, standardization, bound clipping or renormalizing
    a candidate is allowed.
    """
    X, y, weights = np.asarray(X, float), np.asarray(y, float), np.asarray(weights, float)
    if X.ndim != 2 or X.shape[1] != 4 or y.shape != (len(X),) or weights.shape != y.shape or not len(X):
        raise ValueError('Nonempty four-feature rows, labels and weights must align')
    _bounded('Features', X); _bounded('Target', y)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Sample weights must be finite and strictly positive')
    if not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12):
        raise ValueError('Sample weights must already sum to one')
    A, b, names = constraints()
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        H = 2.*(np.einsum('ni,n,nj->ij', X, weights, X, optimize=False)+PENALTY*np.eye(4))
        g = -2.*(np.einsum('ni,n,n->i', X, weights, y, optimize=False)+PENALTY*PRIOR)
    _finite('Hessian', H); _finite('linear term', g)
    best = None
    counts = dict(enumerated=0, dependent=0, singular=0, solved=0, verified=0)
    for size in range(4):
        for active in combinations(range(len(A)), size):
            counts['enumerated'] += 1
            E = np.vstack([np.ones(4), A[list(active)]])
            rhs_constraint = np.r_[1., b[list(active)]]
            if np.linalg.matrix_rank(E) != size+1:
                counts['dependent'] += 1
                continue
            K = np.block([[H, E.T], [E, np.zeros((size+1, size+1))]])
            try:
                solution = np.linalg.solve(K, np.r_[-g, rhs_constraint])
            except np.linalg.LinAlgError:
                counts['singular'] += 1
                continue
            _finite('KKT solution', solution)
            counts['solved'] += 1
            w, lagrange = solution[:4], solution[4:]
            dual = np.zeros(len(A))
            dual[list(active)] = lagrange[1:]
            with np.errstate(over='raise', invalid='raise'):
                violation = np.einsum('ij,j->i', A, w, optimize=False)-b
                stationarity = (np.einsum('ij,j->i', H, w, optimize=False)+g+lagrange[0]
                                + np.einsum('ij,i->j', A, dual, optimize=False))
                residuals = dict(
                    equality=float(abs(w.sum()-1.)),
                    inequality=float(max(0., violation.max())),
                    dual=float(max(0., -dual.min())),
                    stationarity=float(np.max(np.abs(stationarity))),
                    complementarity=float(np.max(np.abs(dual*violation))),
                    l1=float(max(0., np.abs(w-PRIOR).sum()-RADIUS)),
                    momentum=float(max(0., CORE_MIN-w[0])),
                    nonnegative=float(max(0., -w.min())))
            _finite('KKT residuals', np.array(list(residuals.values())))
            if max(residuals.values()) > TOL:
                continue
            value = objective(w, X, y, weights)
            counts['verified'] += 1
            if best is None or value < best[0]:
                best = (value, w.copy(), active, float(lagrange[0]), dual.copy(), residuals)
    if best is None:
        raise RuntimeError('No verified constrained solution; refusing a fallback or relaxed bounds')
    value, w, active, equality_dual, dual, residuals = best
    audit = dict(objective=value, prior_objective=objective(PRIOR, X, y, weights),
        weights=w.tolist(), active_constraints=[names[i] for i in active],
        active_indices=list(active), equality_dual=equality_dual,
        inequality_duals=dual.tolist(), kkt_residuals=residuals,
        inequality_names=names, inequality_matrix=A.tolist(), inequality_bounds=b.tolist(),
        stationarity_convention='2 X.T a (Xw-y) + .2(w-prior) + equality_dual*ones + A.T*inequality_duals = 0',
        tolerance=TOL, enumeration=counts, l1_distance=float(np.abs(w-PRIOR).sum()),
        momentum_weight=float(w[0]), verified=True)
    return w, audit


def annual_weights(train_X, train_target, train_dates, train_ends,
                   train_eligible, train_label_valid, event_dates):
    """Return (one four-weight vector per event, annual JSON audit).

    A month is unavailable when any eligible row lacks a finite feature or
    label, or has label_valid=False. Ineligible numeric values are not read.
    Metadata is globally validated, but numeric values beyond each event-year
    clock are never converted. No event feature or query is used to fit weights.
    """
    dates, ends, months = _timeline(train_dates, train_ends)
    events = numeric._days(event_dates, 'event_dates')
    X = np.asarray(train_X)
    target = np.asarray(train_target)
    n = len(dates)
    if X.shape != (n, 4) or target.shape != (n,):
        raise ValueError('Training matrix must have four columns and matching target rows')
    eligible = _bool(train_eligible, n, 'train_eligible')
    label_valid = _bool(train_label_valid, n, 'train_label_valid')
    years = events.astype('datetime64[Y]').astype(int)+1970
    result = np.tile(PRIOR, (len(events), 1))
    records = []
    for year in np.unique(years):
        cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
        start = numeric._years_before(cutoff)
        clock = (dates > start) & (ends <= cutoff)
        selected_X, selected_y, selected_dates, month_records = [], [], [], []
        for month in np.unique(months[clock]):
            all_rows = np.flatnonzero((months == month) & clock)
            rows = all_rows[eligible[all_rows]]
            entry = dict(month=str(month), signal_date=str(dates[all_rows[0]]),
                label_end=str(ends[all_rows[0]]), eligible_rows=int(len(rows)), status='no_eligible_rows')
            if not len(rows):
                month_records.append(entry)
                continue
            if not label_valid[rows].all():
                entry['status'] = 'whole_month_invalid_label'
                month_records.append(entry)
                continue
            # Deliberately convert only clock-admitted eligible rows.
            xx, yy = np.asarray(X[rows], float), np.asarray(target[rows], float)
            if not (np.isfinite(xx).all() and np.isfinite(yy).all()):
                entry['status'] = 'whole_month_nonfinite_values'
                month_records.append(entry)
                continue
            _bounded('Features', xx); _bounded('Target', yy)
            entry['status'] = 'available'
            selected_X.append(xx); selected_y.append(yy); selected_dates.append(dates[rows])
            month_records.append(entry)
        usable_months = sum(row['status'] == 'available' for row in month_records)
        record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
            status='cold_start', months=usable_months,
            rows=sum(len(x) for x in selected_X), event_count=int(np.sum(years == year)),
            required_months=MIN_MONTHS, weights=PRIOR.tolist(), month_records=month_records,
            excluded_months=[r['month'] for r in month_records if r['status'] != 'available'])
        if usable_months >= MIN_MONTHS:
            xx, yy, dd = np.vstack(selected_X), np.concatenate(selected_y), np.concatenate(selected_dates)
            # Canonical order removes caller row-order effects without changing
            # any examples or relative weighting within a calendar month.
            order = np.lexsort((yy, xx[:, 3], xx[:, 2], xx[:, 1], xx[:, 0], dd.astype(int)))
            xx, yy, dd = xx[order], yy[order], dd[order]
            sample_weights = numeric.month_balanced_weights(dd, cutoff, HALF_LIFE_MONTHS)
            sample_weights = sample_weights/sample_weights.sum()
            weights, fit = fit_weights(xx, yy, sample_weights)
            result[years == year] = weights
            used_months = dd.astype('datetime64[M]')
            record.update(status='fitted', weights=weights.tolist(), fit=fit,
                earliest_signal=str(dd.min()), latest_signal=str(dd.max()),
                latest_label_end=max(r['label_end'] for r in month_records if r['status'] == 'available'),
                sample_weight_sum=float(sample_weights.sum()),
                month_weights={str(m):float(sample_weights[used_months == m].sum())
                               for m in np.unique(used_months)})
        records.append(record)
    audit = dict(method='annual_constrained_guidance_budget_calibration', feature_order=list(FEATURES),
        fixed_prior=PRIOR.tolist(), core_minimum=CORE_MIN, l1_radius=RADIUS,
        penalty=PENALTY, half_life_months=HALF_LIFE_MONTHS, minimum_months=MIN_MONTHS,
        clock='signal_date > cutoff minus five years; label_end <= prior December 31',
        whole_month_validity=True, no_intercept=True, no_standardization=True,
        query_features_used=False, no_eligibility_changes=True,
        rows_inspected='Only eligible numeric rows from each admitted annual clock; invalid-label months not converted',
        cold_start='Exact fixed prior when fewer than eighteen whole valid months', years=records)
    return result, audit
