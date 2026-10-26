"""Independent numerical preparation, rank-slot and execution audit for round170.

No production model, strategy or replay engine is imported. Callers supply all
arrays/exports after enforcing their registered input hashes and phase gates.
Ridge systems are solved as augmented least squares rather than normal equations.
The existing independent scalar/accounting auditor is SHA-bound below.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
RAW_KEYS = ('source_log', 'local_linear', 'local_year_stability', 'local_half')
ARMS = ('incumbent_local67', 'local_year_stability', 'local_half_control')
NEW_RAW = {'local_year_stability': 'local_year_stability', 'local_half_control': 'local_half'}
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('year170_independent_accounting', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': nan mask')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    use = np.isfinite(expected)
    require(np.all(np.abs(actual[use] - expected[use]) <= atol), label + ': values')


def same(actual, expected, label):
    require(np.asarray(actual).shape == np.asarray(expected).shape and
            np.array_equal(actual, expected), label)


def record_check(actual, expected, label):
    """Check every independently produced field, allowing descriptive extra fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list length')
        for i, (a, e) in enumerate(zip(actual, expected)):
            record_check(a, e, label + '/' + str(i))
    elif isinstance(expected, float):
        close(actual, expected, label)
    else:
        require(actual == expected, label + ': differs')


def shrink(c0, masses, corrections):
    """Independent scalar weighted sign agreement, including zero precedence."""
    masses, corrections = list(masses), list(corrections)
    require(len(masses) == len(corrections), 'Fold vectors differ')
    require(math.isfinite(c0) and all(math.isfinite(x) and x >= 0 for x in masses)
            and all(math.isfinite(x) for x in corrections), 'Invalid shrink input')
    agree = [(c > 0 and c0 > 0) or (c < 0 and c0 < 0) for c in corrections]
    total = math.fsum(masses)
    matched = math.fsum(m for m, yes in zip(masses, agree) if yes)
    require(math.isfinite(total), 'Infinite mass')
    positive = sum(m > 0 for m in masses)
    p = matched / total if total > 0 else None
    g = 1. if c0 == 0 else 0. if positive < 2 else max(0., 2. * p - 1.)
    return dict(p=p, shrink=g, positive_mass_years=positive,
                total_weight_mass=total, same_sign_weight_mass=matched), agree


def least_squares(design, response, weights, penalties):
    """All penalties positive; solve a full-rank augmented system with SVD."""
    A = np.vstack((np.sqrt(weights)[:, None] * design, np.diag(np.sqrt(penalties))))
    b = np.concatenate((np.sqrt(weights) * response, np.zeros(design.shape[1])))
    require(np.isfinite(A).all() and np.isfinite(b).all(), 'Nonfinite ridge system')
    coefficient, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    require(rank == design.shape[1] and np.isfinite(coefficient).all(), 'Invalid ridge solution')
    return coefficient


def local_fit(X, query, scale, residual, weights):
    require(X.shape == (len(weights), 10) and query.shape == scale.shape == (10,)
            and residual.shape == weights.shape, 'Invalid local shapes')
    require(all(np.isfinite(v).all() for v in (X, query, scale, residual, weights))
            and (scale > 0).all() and (weights >= 0).all(), 'Invalid local inputs')
    mass = float(weights.sum())
    effective = 0. if mass == 0 else float(mass * mass / np.dot(weights, weights))
    status = 'zero_weight' if mass == 0 else 'zero_residual' if not np.any(residual) else 'fit'
    if status != 'fit':
        return 0., 0., dict(weight_mass=mass, effective_samples=effective, status=status)
    design = np.column_stack((np.ones(len(X)), (X - query) / scale))
    c = least_squares(design, residual, weights, np.array([12.] + [10.] * 10))[0]
    constant = float(np.dot(weights, residual) / (mass + 12.))
    return constant, float(c), dict(weight_mass=mass, effective_samples=effective, status=status)


def reconstruct(dates, ends, symbols, X, targets, common_good, event_dates,
                event_symbols, event_X, event_good, *, min_months=18, min_own_months=18):
    """Independent model reconstruction; synthetic overrides never used by check_prepare."""
    require(type(min_months) is int and min_months > 0 and type(min_own_months) is int
            and min_own_months > 0, 'Positive integer minimums required')
    dates, ends, event_dates = [np.asarray(v, dtype='datetime64[D]') for v in (dates, ends, event_dates)]
    require(all(v.ndim == 1 and not np.isnat(v).any() for v in (dates, ends, event_dates)), 'Invalid dates')
    n, q = len(dates), len(event_dates)
    require(ends.shape == dates.shape and (ends > dates).all() and
            np.all(ends.astype('datetime64[M]') == dates.astype('datetime64[M]') + 1), 'Invalid label ends')
    X, event_X = np.asarray(X, float), np.asarray(event_X, float)
    require(X.shape == (n, 10) and event_X.shape == (q, 10), 'Ten features required')
    symbols, event_symbols = np.asarray(symbols), np.asarray(event_symbols)
    require(symbols.shape == (n,) and event_symbols.shape == (q,)
            and all(isinstance(v, (str, np.str_)) and v for v in list(symbols) + list(event_symbols)), 'Invalid symbols')
    common_good, event_good = np.asarray(common_good), np.asarray(event_good)
    require(common_good.dtype == event_good.dtype == np.dtype(bool) and
            common_good.shape == (n,) and event_good.shape == (q,), 'Exact bool masks required')
    require(set(targets) == set(TARGETS), 'All original targets required')
    ys = {k: np.asarray(v, float) for k, v in targets.items()}
    require(all(y.shape == (n,) for y in ys.values()), 'Target shape differs')
    valid = common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():
        valid &= np.isfinite(y)
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        use = months == month
        if np.count_nonzero(valid & use) < 2:
            valid[use] = False
    years = event_dates.astype('datetime64[Y]').astype(int) + 1970
    available, local_available = np.zeros(q, bool), np.zeros(q, bool)
    raw = {k: np.full(q, np.nan) for k in RAW_KEYS}
    annual, details = [], []
    counts = dict(source_target_fits=0, pooled_log_refits=0, full_local_evaluations=0,
                  full_local_solves=0, deletion_evaluations=0, deletion_solves=0)
    for year in np.unique(years):
        cutoff = np.datetime64('%04d-12-31' % (year - 1))
        start = np.datetime64('%04d-12-31' % (year - 6))
        ids = np.flatnonzero(valid & (dates > start) & (ends <= cutoff))
        nm = len(np.unique(months[ids]))
        pred = np.flatnonzero((years == year) & event_good & np.isfinite(event_X).all(axis=1)) if nm >= min_months else np.array([], int)
        annual_row = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(start),
                          training_rows=len(ids), training_months=nm, prediction_rows=len(pred),
                          training_row_ids=ids.tolist(), first_training_signal=str(dates[ids].min()) if len(ids) else None,
                          latest_label_end=str(ends[ids].max()) if len(ids) else None,
                          status='fit' if nm >= min_months else 'cold_start')
        if len(pred):
            require(np.abs(X[ids]).max() <= .5 and np.abs(event_X[pred]).max() <= .5, 'Actual features outside rank units')
            identities = [(str(months[i]), str(symbols[i])) for i in ids]
            require(len(identities) == len(set(identities)), 'Duplicate security/signal month')
            # Balance each month's mass, then normalize only the original pooled fit.
            _, inv, num = np.unique(months[ids], return_inverse=True, return_counts=True)
            age = int(cutoff.astype('datetime64[M]').astype(int)) - months[ids].astype(int)
            w = np.exp2(-age / 12.) / num[inv]
            w *= len(w) / w.sum()
            annual_row['weight_sum'] = float(w.sum())
            counts['source_target_fits'] += 4
            counts['pooled_log_refits'] += 1
            mean = np.average(X[ids], axis=0, weights=w)
            scale = np.sqrt(np.average((X[ids] - mean) ** 2, axis=0, weights=w))
            scale = np.where(scale > 1e-12, scale, 1.)
            intercept = float(np.average(ys['log_excess'][ids], weights=w))
            z = (X[ids] - mean) / scale
            coefficient = least_squares(z, ys['log_excess'][ids] - intercept, w, np.full(10, 10.))
            # Explicit sums avoid platform BLAS floating-status leakage and use
            # an arithmetic path separate from the production prediction code.
            residual = ys['log_excess'][ids] - (np.sum(z * coefficient, axis=1) + intercept)
            source = np.sum(((event_X[pred] - mean) / scale) * coefficient, axis=1) + intercept
            for key in RAW_KEYS:
                raw[key][pred] = source
            available[pred] = True
            for j in pred:
                own = np.flatnonzero(symbols[ids] == event_symbols[j])
                own_ids = ids[own]
                own_months = len(np.unique(months[own_ids]))
                detail = dict(event_row=int(j), date=str(event_dates[j]), symbol=str(event_symbols[j]), year=int(year),
                              cutoff=str(cutoff), own_rows=len(own), own_months=own_months, own_row_ids=own_ids.tolist(),
                              own_dates=dates[own_ids].astype(str).tolist(), own_label_ends=ends[own_ids].astype(str).tolist(),
                              latest_label_end=str(ends[own_ids].max()) if len(own_ids) else None,
                              status='insufficient_own_history', weight_mass=0., effective_samples=0.,
                              constant_correction=0., full_correction=0., own_weights=[], folds=[], p=None, shrink=1.,
                              positive_mass_years=0, total_weight_mass=0., same_sign_weight_mass=0.)
                if own_months >= min_own_months:
                    local_age = int(cutoff.astype('datetime64[M]').astype(int)) - months[own_ids].astype(int)
                    lw = np.exp2(-local_age / 12.) * np.exp(-np.mean((X[own_ids] - event_X[j]) ** 2, axis=1) / .5)
                    constant, c0, info = local_fit(X[own_ids], event_X[j], scale, residual[own], lw)
                    counts['full_local_evaluations'] += 1
                    counts['full_local_solves'] += int(info['status'] == 'fit')
                    folds = []
                    own_years = dates[own_ids].astype('datetime64[Y]').astype(int) + 1970
                    for deleted in np.unique(own_years):
                        keep = own_years != deleted
                        _, correction, fi = local_fit(X[own_ids][keep], event_X[j], scale, residual[own][keep], lw[keep])
                        folds.append(dict(removed_year=int(deleted), removed_row_ids=own_ids[~keep].tolist(),
                                          retained_rows=int(keep.sum()), latest_label_end=str(ends[own_ids][keep].max()) if keep.any() else None,
                                          weight_mass=float(lw[~keep].sum()), correction=correction,
                                          retained_weight_mass=fi['weight_mass'], status=fi['status']))
                    agreement, signs = shrink(c0, [f['weight_mass'] for f in folds], [f['correction'] for f in folds])
                    for fold, yes in zip(folds, signs):
                        fold['strictly_same_sign'] = yes
                    counts['deletion_evaluations'] += len(folds)
                    counts['deletion_solves'] += sum(f['status'] == 'fit' for f in folds)
                    detail.update(info, constant_correction=constant, full_correction=c0,
                                  own_weights=lw.tolist(), folds=folds, **agreement)
                    local_available[j] = info['weight_mass'] > 0
                    for key, correction in [('local_linear', c0), ('local_year_stability', agreement['shrink'] * c0), ('local_half', .5 * c0)]:
                        if correction != 0:
                            raw[key][j] += correction
                details.append(detail)
        annual.append(annual_row)
    require(all(np.array_equal(np.isfinite(v), available) for v in raw.values()), 'Reconstructed availability differs')
    return dict(raw=raw, available=available, local_available=local_available,
                audit=dict(annual=annual, local=details, counts=counts))


def check_prepare(train, meta, a, forecast):
    m, tm = meta['monthly'], meta['training']
    symbols = np.array([s for row in tm['symbols'] for s in row])
    event_symbols = np.array([s for row in m['symbols'] for s in row])
    for key, metadata, source in [('dates', tm, 'dates'), ('ends', tm, 'ends'), ('dates', m, 'event_dates')]:
        expanded = np.array([d for d, row in zip(metadata[key], metadata['symbols']) for _ in row], dtype='datetime64[D]')
        same(np.asarray(train[source], dtype='datetime64[D]'), expanded, 'Metadata axis ' + source)
    require(np.asarray(a['eligible']).dtype == np.dtype(bool) and a['eligible'].shape == (len(m['dates']), len(m['symbols'][0])), 'Eligibility axis')
    expected = reconstruct(train['dates'], train['ends'], symbols, train['train_plain'],
        {k: train['target_' + k] for k in TARGETS}, train['common_good'], train['event_dates'],
        event_symbols, train['event_plain'], a['eligible'].ravel() & np.isfinite(train['event_plain']).all(axis=1))
    require(set(forecast['raw']) == set(RAW_KEYS), 'Raw forecast keys changed')
    for key in RAW_KEYS:
        close(forecast['raw'][key], expected['raw'][key], 'Raw ' + key)
    for key in ('available', 'local_available'):
        require(np.asarray(forecast[key]).dtype == np.dtype(bool), 'Forecast mask dtype')
        same(forecast[key], expected[key], 'Forecast ' + key)
    same(forecast['available'], a['model_available'].ravel(), 'Original pooled availability changed')
    same(forecast['local_available'], a['local_available'].ravel(), 'Original local availability changed')
    for key in ('source_log', 'local_linear'):
        close(forecast['raw'][key], a['raw_' + key].ravel(), 'Original ' + key + ' parity')
    record_check(forecast['audit'], expected['audit'], 'Model audit')
    return dict(passed=True, independent_augmented_least_squares=True, atol=1e-12,
                original_source_and_local_parity=True, availability_unchanged=True,
                annual_cohorts=len(expected['audit']['annual']), local_queries=len(expected['audit']['local']),
                deletion_folds=expected['audit']['counts']['deletion_evaluations'],
                not_held_out_validation=True, computes_returns=False)


def centered_rank(v):
    """Stable average rank / population minus 0.5, without pandas/production rank."""
    v = np.asarray(v, float)
    require(v.ndim == 1 and np.isfinite(v).all() and len(v), 'Invalid rank population')
    order = np.argsort(v, kind='stable')
    answer = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]:
            j += 1
        answer[order[i:j]] = ((i + 1 + j) / 2.) / len(v) - .5
        i = j
    return answer


def rank_oracle(a, forecast):
    shape = a['eligible'].shape
    original = np.asarray(a['rank_log_excess'], float)
    require(original.shape == shape and np.isfinite(original).all(), 'Invalid original ranks')
    require(np.asarray(a['forecast_available']).dtype == np.dtype(bool) and np.asarray(a['eligible']).dtype == np.dtype(bool), 'Bool universe masks required')
    use = a['forecast_available'] & a['eligible']
    source = a['raw_source_log']

    def rank(values, available, other=None):
        values, available = values.reshape(shape), available.reshape(shape)
        other = values if other is None else other.reshape(shape)
        result, ready = original.copy(), np.zeros(shape[0], bool)
        for i, members in enumerate(use):
            if members.any() and available[i, members].all() and np.isfinite(values[i, members]).all() and np.isfinite(other[i, members]).all():
                ready[i] = True
                if not np.array_equal(values[i, members], source[i, members]):
                    result[i, members] = centered_rank(values[i, members])
        return result, ready

    old, old_ready = rank(a['raw_local_linear'], a['model_available'], a['raw_local_constant'])
    weak, calm = a['state_weak_narrow'], a['state_calm10']
    coefficient = .25 * (.05 * weak[:, None] + .01 * (calm & ~weak)[:, None] * a['extension_available'])
    ranks, deltas, ready = {}, {}, {}
    for arm, key in NEW_RAW.items():
        rr, yes = rank(np.asarray(forecast['raw'][key]), np.asarray(forecast['available']))
        same(yes, old_ready, 'Candidate changed original row readiness')
        delta = np.zeros(shape)
        active = use & yes[:, None]
        delta[active] = (coefficient * (rr - old))[active]
        if len(delta):
            delta[0] = 0.  # Preserve the original partial 2016-08-17 initialization.
        ranks[arm], deltas[arm], ready[arm] = rr, delta, yes
    return ranks, deltas, ready


def verify_rank_slot(a, forecast, ranks, deltas):
    require(set(ranks) == set(deltas) == set(NEW_RAW), 'Candidate slot names differ')
    rr, dd, ready = rank_oracle(a, forecast)
    for arm in NEW_RAW:
        close(ranks[arm], rr[arm], arm + ' full-U rank', atol=0.)
        close(deltas[arm], dd[arm], arm + ' replacement delta')
    return dict(passed=True, full_original_universe=True, includes_eligible_etfs=True,
                no_local_history_filter=True, replacement_not_additive_local_slot=True,
                ready_rows={k: int(v.sum()) for k, v in ready.items()},
                changed_cells={k: int(np.count_nonzero(v)) for k, v in dd.items()})


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            record_check(wanted['metrics']['annual_returns'], annual, 'Annual returns')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
