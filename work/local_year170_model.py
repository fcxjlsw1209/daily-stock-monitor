"""Fixed-pooled, same-security local-correction calendar-year sensitivity.

This module accepts caller-provided arrays only. It constructs no labels, ranks,
orders or returns and reads no prepared data. Deletion folds assess sensitivity
of a local residual correction, not out-of-sample error: the pooled fit still
uses the complete mature annual cohort.
"""
from pathlib import Path
import importlib.util
import math

import numpy as np


LOCAL_PATH = Path(__file__).resolve().parents[1]/'outputs/monthly_local_linear70/model.py'
_spec = importlib.util.spec_from_file_location('local_year170_frozen_local', LOCAL_PATH)
local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local)
numeric = local.numeric
TARGETS = local.TARGETS
MIN_MONTHS = local.MIN_MONTHS
MIN_OWN_MONTHS = local.MIN_OWN_MONTHS
ARMS = ('local_year_stability', 'local_half')
RAW_KEYS = ('source_log', 'local_linear')+ARMS


def dependency_files():
    return tuple(sorted(set(local.dependency_files()) | {Path(__file__).resolve()}))


def weighted_sign_shrink(full_correction, year_weight_mass, fold_corrections):
    """Return raw weighted agreement and the single preregistered shrink rule.

    A zero fold correction is not strictly same-sign as a nonzero full one.
    Zero-mass years do not contribute to either the fraction or positive count.
    Undefined agreement at zero total mass is represented by None, not NaN.
    The zero-full-correction override precedes the fewer-than-two-years rule.
    """
    c0 = float(full_correction)
    mass = np.asarray(year_weight_mass, dtype=float)
    correction = np.asarray(fold_corrections, dtype=float)
    if mass.ndim != 1 or correction.shape != mass.shape:
        raise ValueError('Year masses and fold corrections must be matching vectors')
    local._finite('full correction', c0)
    local._finite('year weight mass', mass)
    local._finite('fold correction', correction)
    if (mass < 0.).any():
        raise ValueError('Negative year weight mass')
    same = ((correction > 0.) & (c0 > 0.)) | ((correction < 0.) & (c0 < 0.))
    total = math.fsum(float(x) for x in mass)
    agreed = math.fsum(float(x) for x in mass[same])
    count = int(np.count_nonzero(mass > 0.))
    local._finite('total year weight mass', total)
    p = agreed/total if total > 0. else None
    g = 1. if c0 == 0. else 0. if count < 2 else max(0., 2.*p-1.)
    return dict(p=p, shrink=g, positive_mass_years=count,
                total_weight_mass=total, same_sign_weight_mass=agreed,
                same_sign=same.tolist())


def _year_folds(dates, ends, row_ids, X, query, scale, residual, weights, c0):
    """Delete own rows only; all inputs and the pooled scales remain fixed."""
    years = dates.astype('datetime64[Y]').astype(int)+1970
    folds = []
    for year in np.unique(years):
        removed = years == year
        keep = ~removed
        _, correction, info = local.local_corrections(
            X[keep], query, scale, residual[keep], weights[keep])
        folds.append(dict(removed_year=int(year),
            removed_row_ids=row_ids[removed].tolist(),
            retained_rows=int(keep.sum()),
            latest_label_end=str(ends[keep].max()) if keep.any() else None,
            weight_mass=float(weights[removed].sum()), correction=correction,
            retained_weight_mass=info['weight_mass'], status=info['status']))
    agreement = weighted_sign_shrink(c0, [f['weight_mass'] for f in folds],
                                    [f['correction'] for f in folds])
    for fold, same_sign in zip(folds, agreement.pop('same_sign')):
        fold['strictly_same_sign'] = same_sign
    return folds, agreement


def annual_forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
                     event_symbols, event_X, event_good, *, min_months=MIN_MONTHS,
                     min_own_months=MIN_OWN_MONTHS):
    """Preserve frozen pooled/local predictions; append two raw local variants.

    Production minimums stay 18. Keyword overrides exist only for synthetic
    fixtures. Original local availability is determined before deletion folds;
    no fold minimum, agreement or correction magnitude may alter availability.
    A failed numeric operation raises, rather than silently dropping a fold.
    """
    local._minimum(min_months, 'min_months')
    local._minimum(min_own_months, 'min_own_months')
    dates, ends = numeric._days(dates, 'dates'), numeric._days(ends, 'ends')
    event_dates = numeric._days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if (ends.shape != (n,) or (ends <= dates).any() or
            np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]')+1)):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    symbols = local._symbols(symbols, n, 'symbols')
    event_symbols = local._symbols(event_symbols, q, 'event_symbols')
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    if X.shape != (n, 10) or event_X.shape != (q, 10):
        raise ValueError('Original ten-feature matrices are required')
    common_good = local.original._bool(common_good, n, 'common_good')
    event_good = local.original._bool(event_good, q, 'event_good')
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
    annual, details = [], []
    counts = dict(source_target_fits=0, pooled_log_refits=0,
                  full_local_evaluations=0, full_local_solves=0,
                  deletion_evaluations=0, deletion_solves=0)
    with local.threadpool_limits(limits=1):
        ref = local.original.annual_forecasts(dates, ends, X, ys, common_good,
            event_dates, event_X, event_good, min_months=min_months)
        counts['source_target_fits'] = 4*sum(
            r['status'] == 'fit' and r['prediction_rows'] > 0
            for r in ref['audit']['annual'])
        raw = {key: ref['log_excess'].copy() for key in RAW_KEYS}
        available = ref['available'].copy()
        local_available = np.zeros(q, bool)
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year-1), 'D')
            start = numeric._years_before(cutoff)
            train = usable & (dates > start) & (ends <= cutoff)
            predict = (years == year) & available
            ids = np.flatnonzero(train)
            count = numeric._month_count(dates[train])
            record = dict(year=int(year), cutoff=str(cutoff),
                lookback_exclusive_start=str(start), training_rows=len(ids),
                training_months=count, prediction_rows=int(predict.sum()),
                training_row_ids=ids.tolist(),
                first_training_signal=str(dates[train].min()) if len(ids) else None,
                latest_label_end=str(ends[train].max()) if len(ids) else None,
                status='fit' if count >= min_months else 'cold_start')
            if predict.any():
                if count < min_months:
                    raise AssertionError('Pooled availability disagrees with source cohort')
                local._ranks('actual pooled training ranks', X[train])
                local._ranks('actual query ranks', event_X[predict])
                identities = list(zip(months[train].astype(str), symbols[train]))
                if len(set(identities)) != len(identities):
                    raise ValueError('Each training security must have one row per signal month')
                weights = numeric.month_balanced_weights(dates[train], cutoff,
                                                        local.HALF_LIFE_MONTHS)
                pooled = local._fit_pooled(X[train], ys['log_excess'][train], weights)
                counts['pooled_log_refits'] += 1
                np.testing.assert_array_equal(numeric._predict(pooled, event_X[predict]),
                                              raw['source_log'][predict])
                with np.errstate(over='raise', invalid='raise'):
                    residual = ys['log_excess'][train]-numeric._predict(pooled, X[train])
                local._finite('mature pooled residuals', residual)
                record['weight_sum'] = float(weights.sum())
                by_symbol = {s: np.flatnonzero(symbols[train] == s) for s in set(symbols[train])}
                for j in np.flatnonzero(predict):
                    own = by_symbol.get(event_symbols[j], np.empty(0, dtype=int))
                    own_ids = ids[own]
                    own_months = numeric._month_count(dates[own_ids])
                    detail = dict(event_row=int(j), date=str(event_dates[j]),
                        symbol=event_symbols[j], year=int(year), cutoff=str(cutoff),
                        own_rows=len(own), own_months=own_months,
                        own_row_ids=own_ids.tolist(),
                        own_dates=dates[own_ids].astype(str).tolist(),
                        own_label_ends=ends[own_ids].astype(str).tolist(),
                        latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                        status='insufficient_own_history', weight_mass=0.,
                        effective_samples=0., constant_correction=0., full_correction=0.,
                        own_weights=[], folds=[], p=None, shrink=1.,
                        positive_mass_years=0, total_weight_mass=0., same_sign_weight_mass=0.)
                    if own_months >= min_own_months:
                        w = local.local_weights(dates[own_ids], cutoff, X[own_ids], event_X[j])
                        constant, c0, info = local.local_corrections(
                            X[own_ids], event_X[j], pooled[1], residual[own], w)
                        counts['full_local_evaluations'] += 1
                        counts['full_local_solves'] += int(info['status'] == 'fit')
                        folds, agreement = _year_folds(dates[own_ids], ends[own_ids], own_ids,
                            X[own_ids], event_X[j], pooled[1], residual[own], w, c0)
                        counts['deletion_evaluations'] += len(folds)
                        counts['deletion_solves'] += sum(f['status'] == 'fit' for f in folds)
                        detail.update(info, constant_correction=constant,
                            full_correction=c0, own_weights=w.tolist(), folds=folds, **agreement)
                        local_available[j] = info['weight_mass'] > 0.
                        corrections = dict(local_linear=c0,
                            local_year_stability=c0*agreement['shrink'], local_half=.5*c0)
                        for key, correction in corrections.items():
                            if correction != 0.:
                                with np.errstate(over='raise', invalid='raise'):
                                    raw[key][j] += correction
                    details.append(detail)
            annual.append(record)
    for key, values in raw.items():
        if not np.array_equal(np.isfinite(values), available):
            raise ValueError(key+' changed source availability or became nonfinite')
    return dict(raw=raw, available=available, local_available=local_available, audit=dict(
        arms=list(ARMS), feature_count=10, target='unchanged centered log_excess',
        alpha=local.ALPHA, half_life_months=local.HALF_LIFE_MONTHS,
        min_months=int(min_months), min_own_months=int(min_own_months),
        bandwidth=local.BANDWIDTH, intercept_penalty=local.INTERCEPT_PENALTY,
        slope_penalty=local.SLOPE_PENALTY, counts=counts,
        pooled_fit_fixed_across_deletions=True, deletion_weight_renormalization=False,
        deletion_minimum_months=None, local_availability_unchanged=True,
        agreement='strict same-sign fold mass / all original own-year weight mass',
        shrink='c0==0:1; fewer2 positive-mass years:0; otherwise max(0,2*p-1)',
        validation_scope='local-correction sensitivity, not held-out or OOF validation',
        source=ref['audit'], annual=annual, local=details))
