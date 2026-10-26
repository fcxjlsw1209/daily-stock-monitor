"""Matched-month own versus other-security residual corrections, round206.

Pure caller-array model. No data loading, labels, ranks, orders or returns.
The frozen local model is replayed for exact source/constant/linear parity;
all such regression solves and new scalar correction estimates are counted.
"""
from pathlib import Path
import importlib.util

import numpy as np


LOCAL_PATH = Path(__file__).resolve().parents[1]/'outputs/monthly_local_linear70/model.py'
_spec = importlib.util.spec_from_file_location('identity206_original_local', LOCAL_PATH)
local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local)
numeric = local.numeric
TARGETS = local.TARGETS
MIN_MONTHS = 18
MIN_OWN_MONTHS = 18
ALPHA = 10.
HALF_LIFE_MONTHS = 12.
BANDWIDTH = .5
INTERCEPT_PENALTY = 12.
ARMS = ('peer_residual_transfer', 'own_residual_control')
RAW_KEYS = ('source_log', 'local_linear', 'local_constant', 'candidate', 'control')
COUNT_FIELDS = ('source_target_fits', 'reference_pooled_log_refits',
    'reference_local_evaluations', 'reference_local_solves',
    'transfer_pooled_log_refits', 'own_constant_evaluations',
    'peer_query_evaluations', 'peer_month_evaluations', 'peer_row_observations',
    'actual_regression_solves')


def dependency_files():
    return tuple(sorted(set(local.dependency_files()) | {Path(__file__).resolve()}))


def _finite(name, value):
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite '+name)


def peer_month_average(X, query, residual):
    """One same-month other-security kernel mean, without any age multiplier."""
    X, query, residual = np.asarray(X, float), np.asarray(query, float), np.asarray(residual, float)
    if X.ndim != 2 or X.shape[1] != 10 or len(X) == 0 or query.shape != (10,) or residual.shape != (len(X),):
        raise ValueError('Nonempty aligned ten-feature peer month required')
    local._ranks('peer features', X)
    local._ranks('peer query', query)
    _finite('peer residual', residual)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        distance = np.mean((X-query)**2, axis=1)
        weight = np.exp(-distance/(2*BANDWIDTH**2))
        mass = float(weight.sum())
        numerator = float(np.einsum('n,n->', weight, residual, optimize=False))
        value = numerator/mass
    if mass <= 0.:
        raise FloatingPointError('Nonpositive peer kernel mass')
    for label, item in (('peer weights', weight), ('peer mass', mass),
                        ('peer numerator', numerator), ('peer mean', value)):
        _finite(label, item)
    return value, dict(peer_count=len(X), peer_weight_mass=mass,
                       peer_weighted_residual=numerator, peer_mean=value)


def _inputs(dates, ends, symbols, X, targets, common_good, event_dates,
            event_symbols, event_X, event_good):
    """Validate axes, but interpret training numbers only inside query-year support.

    The host also supplies physically truncated source prefixes. Keeping unused
    rows as NaN/False here preserves caller row IDs and makes future object
    poison irrelevant without rewriting the frozen original model.
    """
    dates = numeric._days(dates, 'dates')
    ends = numeric._days(ends, 'ends')
    event_dates = numeric._days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if ends.shape != (n,) or (ends <= dates).any() or np.any(
            ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1):
        raise ValueError('Labels must mature in the next calendar month')
    symbols = local._symbols(symbols, n, 'symbols')
    event_symbols = local._symbols(event_symbols, q, 'event symbols')
    X, event_X = np.asarray(X), np.asarray(event_X)
    if X.shape != (n, 10) or event_X.shape != (q, 10):
        raise ValueError('Original plain10 matrices required')
    common_good = np.asarray(common_good)
    event_good = local.original._bool(event_good, q, 'event good')
    if common_good.shape != (n,) or set(targets) != set(TARGETS):
        raise ValueError('Original common cohort and four targets required')
    ys = {key: np.asarray(targets[key]) for key in TARGETS}
    if any(y.shape != (n,) for y in ys.values()):
        raise ValueError('All four targets must match training rows')
    consumed = np.zeros(n, bool)
    if q:
        years = event_dates.astype('datetime64[Y]').astype(int)+1970
        first = np.datetime64('%04d-12-31' % (int(years.min())-6))
        last = np.datetime64('%04d-12-31' % (int(years.max())-1))
        consumed = (dates > first) & (ends <= last)
    used_good = common_good[consumed]
    if any(not isinstance(x, (bool, np.bool_)) for x in used_good):
        raise ValueError('Consumed common_good entries must be exact booleans')
    xf = np.full((n, 10), np.nan)
    xf[consumed] = np.asarray(X[consumed], float)
    yf = {key: np.full(n, np.nan) for key in TARGETS}
    for key in TARGETS:
        yf[key][consumed] = np.asarray(ys[key][consumed], float)
    good = np.zeros(n, bool)
    good[consumed] = used_good
    return (dates, ends, symbols, xf, yf, good, event_dates, event_symbols,
            np.asarray(event_X, float), event_good)


def annual_forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
                     event_symbols, event_X, event_good, *, min_months=MIN_MONTHS,
                     min_own_months=MIN_OWN_MONTHS):
    """Rebuild original forecasts and two matched corrections; never rerank here."""
    local._minimum(min_months, 'min_months')
    local._minimum(min_own_months, 'min_own_months')
    args = _inputs(dates, ends, symbols, X, targets, common_good, event_dates,
                   event_symbols, event_X, event_good)
    dates, ends, symbols, X, ys, common_good, event_dates, event_symbols, event_X, event_good = args
    months = dates.astype('datetime64[M]')
    usable = common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        usable &= np.isfinite(y)
    for month in np.unique(months):
        rows = months == month
        if (usable & rows).sum() < 2:
            usable[rows] = False
    years = event_dates.astype('datetime64[Y]').astype(int)+1970
    annual, details = [], []
    counts = {key: 0 for key in COUNT_FIELDS}
    with local.threadpool_limits(limits=1):
        ref = local.annual_forecasts(*args, min_months=min_months,
                                     min_own_months=min_own_months)
        raw = {key: ref['raw'][key].copy() for key in RAW_KEYS[:3]}
        raw.update(candidate=raw['source_log'].copy(), control=raw['local_constant'].copy())
        available, local_available = ref['available'].copy(), ref['local_available'].copy()
        reference_rows = {row['event_row']: row for row in ref['audit']['local']}
        counts['source_target_fits'] = 4*sum(r['status'] == 'fit' and r['prediction_rows'] > 0
                                            for r in ref['audit']['source']['annual'])
        counts['reference_pooled_log_refits'] = sum(r['prediction_rows'] > 0
                                                   for r in ref['audit']['annual'])
        counts['reference_local_evaluations'] = sum(r['own_months'] >= min_own_months
                                                   for r in reference_rows.values())
        counts['reference_local_solves'] = sum(r['status'] == 'fit' for r in reference_rows.values())
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1))
            start = numeric._years_before(cutoff)
            train = usable & (dates > start) & (ends <= cutoff)
            ids = np.flatnonzero(train)
            predict = np.flatnonzero((years == year) & available)
            nm = numeric._month_count(dates[ids])
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                training_rows=len(ids), training_months=nm, prediction_rows=len(predict),
                training_row_ids=ids.tolist(), first_training_signal=str(dates[ids].min()) if len(ids) else None,
                latest_label_end=str(ends[ids].max()) if len(ids) else None,
                status='fit' if nm >= min_months else 'cold_start', weight_sum=None)
            if len(predict):
                identities = list(zip(months[ids].astype(str), symbols[ids]))
                if len(set(identities)) != len(ids):
                    raise ValueError('Duplicate monthly security identity')
                local._ranks('mature training ranks', X[ids])
                local._ranks('actual query ranks', event_X[predict])
                weights = numeric.month_balanced_weights(dates[ids], cutoff, HALF_LIFE_MONTHS)
                pooled = numeric._fit_ridge(X[ids], ys['log_excess'][ids], weights, ALPHA)
                counts['transfer_pooled_log_refits'] += 1
                np.testing.assert_array_equal(numeric._predict(pooled, event_X[predict]), raw['source_log'][predict])
                with np.errstate(over='raise', invalid='raise'):
                    residual = ys['log_excess'][ids]-numeric._predict(pooled, X[ids])
                _finite('mature pooled residuals', residual)
                record['weight_sum'] = float(weights.sum())
                for j in predict:
                    own = np.flatnonzero(symbols[ids] == event_symbols[j])
                    own_ids = ids[own]
                    own_months = numeric._month_count(dates[own_ids])
                    old = reference_rows[int(j)]
                    row = dict(event_row=int(j), date=str(event_dates[j]), symbol=event_symbols[j],
                        year=int(year), cutoff=str(cutoff), own_rows=len(own), own_months=own_months,
                        own_row_ids=own_ids.tolist(), own_dates=dates[own_ids].astype(str).tolist(),
                        own_label_ends=ends[own_ids].astype(str).tolist(),
                        latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                        status='insufficient_own_history', weight_mass=0., effective_samples=0.,
                        own_weights=[], own_residuals=[], own_numerator=0., peer_numerator=0.,
                        correction_denominator=None, constant_correction=0.,
                        linear_correction=float(old['linear_correction']), peer_correction=0., peer_months=[])
                    if own_months >= min_own_months:
                        w = local.local_weights(dates[own_ids], cutoff, X[own_ids], event_X[j])
                        mass = float(w.sum())
                        with np.errstate(over='raise', invalid='raise', divide='raise'):
                            own_num = float(np.einsum('n,n->', w, residual[own], optimize=False))
                            denominator = mass+INTERCEPT_PENALTY
                            c_own = own_num/denominator if own_num != 0. else 0.
                            effective = float(mass**2/np.einsum('n,n->', w, w, optimize=False)) if mass else 0.
                        np.testing.assert_array_equal(c_own, old['constant_correction'])
                        means, peer_records = [], []
                        for position, own_id, wk in zip(own, own_ids, w):
                            peers = np.flatnonzero((months[ids] == months[own_id]) &
                                                   (symbols[ids] != event_symbols[j]))
                            if not len(peers):
                                raise ValueError('Own mature month has no other common-cohort security')
                            mean, info = peer_month_average(X[ids[peers]], event_X[j], residual[peers])
                            means.append(mean)
                            peer_records.append(dict(month=str(months[own_id]), own_row_id=int(own_id),
                                own_weight=float(wk), own_residual=float(residual[position]),
                                peer_row_ids=ids[peers].tolist(), **info))
                            counts['peer_month_evaluations'] += 1
                            counts['peer_row_observations'] += len(peers)
                        with np.errstate(over='raise', invalid='raise', divide='raise'):
                            peer_num = float(np.einsum('n,n->', w, np.asarray(means), optimize=False))
                            c_peer = peer_num/denominator if peer_num != 0. else 0.
                        for label, value in (('own numerator', own_num), ('peer numerator', peer_num),
                                             ('own correction', c_own), ('peer correction', c_peer)):
                            _finite(label, value)
                        counts['own_constant_evaluations'] += 1
                        counts['peer_query_evaluations'] += 1
                        row.update(status='fit', weight_mass=mass, effective_samples=effective,
                            own_weights=w.tolist(), own_residuals=residual[own].tolist(),
                            own_numerator=own_num, peer_numerator=peer_num,
                            correction_denominator=denominator, constant_correction=c_own,
                            peer_correction=c_peer, peer_months=peer_records)
                        if c_peer != 0.:
                            with np.errstate(over='raise', invalid='raise'):
                                raw['candidate'][j] += c_peer
                    details.append(row)
            annual.append(record)
    counts['actual_regression_solves'] = sum(counts[key] for key in (
        'source_target_fits', 'reference_pooled_log_refits',
        'reference_local_solves', 'transfer_pooled_log_refits'))
    for key, value in raw.items():
        if not np.array_equal(np.isfinite(value), available) or np.isinf(value).any():
            raise FloatingPointError(key+' changed original availability or became nonfinite')
    np.testing.assert_array_equal(raw['control'], raw['local_constant'])
    return dict(raw=raw, available=available, local_available=local_available, audit=dict(
        arms=list(ARMS), raw_keys=list(RAW_KEYS), feature_count=10,
        target='unchanged cached centered log_excess', alpha=ALPHA,
        half_life_months=HALF_LIFE_MONTHS, min_months=int(min_months),
        min_own_months=int(min_own_months), bandwidth=BANDWIDTH,
        intercept_penalty=INTERCEPT_PENALTY, counts=counts,
        source=ref['audit']['source'], annual=annual, local=details))
