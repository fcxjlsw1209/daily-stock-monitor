"""Caller-supplied clock-only pooled Ridge replication; no I/O of market data.

Annual calibration precedes the callback and every candidate fit. The monthly
clock changes maturity, the five-calendar-year window, decay and scaler together.
Same-close label maturity is an idealization, not strict point-in-time evidence.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_PATH = ROOT / 'outputs/monthly_architecture_restart/residual_model.py'
NUMERIC_SHA = 'ea294b9a0a1e9493799f1014dcec464effa93dce70f44fb5de53a604cd832886'
if hashlib.sha256(NUMERIC_PATH.read_bytes()).hexdigest() != NUMERIC_SHA:
    raise ValueError('Changed frozen numeric dependency')
_spec = importlib.util.spec_from_file_location('clock273_original_numeric', NUMERIC_PATH)
numeric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(numeric)
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
INITIAL_PARTIAL = '2016-08-17'
ARMS = ('incumbent_local67', 'monthly_pooled_clock')
BUNDLE_KEYS = ('n', 'width', 'control', 'candidate', 'audit')
FORECAST_KEYS = ('raw', 'available', 'rank', 'audit')
FIT_KEYS = ('kind', 'query_indices', 'cutoff', 'lower_bound', 'training_row_ids',
            'query_row_ids', 'training_months', 'status', 'signal_dates',
            'label_end_dates', 'symbols', 'weights', 'weights_sha256', 'mean',
            'scale', 'beta', 'intercept', 'gram', 'rhs', 'raw_predictions')
MODEL_COUNT_FIELDS = ('events', 'width', 'training_rows', 'valid_training_rows',
                      'annual_groups', 'annual_ridge_fits', 'monthly_ridge_fits',
                      'annual_prediction_rows', 'monthly_prediction_rows',
                      'actual_model_fits', 'local_model_fits', 'other_target_model_fits')
MODEL_AUDIT_KEYS = ('schema', 'counts', 'annual_control_exact',
                    'original_ranking_population', 'paired_ready_preserved',
                    'local_models_reused', 'other_targets_not_fitted',
                    'full_clock_change', 'strict_pit', 'computes_returns',
                    'control_gate', 'label_preservation', 'source_scope')
RANK_AUDIT_KEYS = ('schema', 'events', 'width', 'ranking_cells', 'active_indices',
                   'changed_cells', 'original_ranking_population',
                   'paired_ready_preserved', 'local_basis_preserved', 'weak_only',
                   'partial_exact', 'calm_slot_preserved', 'no_post_rerank',
                   'computes_returns')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dependency_files():
    return (NUMERIC_PATH,)


def _schema(name):
    return 'monthly_matured_clock273_' + name + '_v1'


def _digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def _days(values, name):
    a = np.asarray(values)
    require(a.ndim == 1, name + ' vector')
    out = []
    for value in a:
        if isinstance(value, np.datetime64):
            require(not np.isnat(value), name + ' NaT')
            text = str(value.astype('datetime64[D]'))
            require(value == np.datetime64(text), name + ' whole date')
        else:
            require(isinstance(value, (str, np.str_)), name + ' ISO string')
            text = str(value)
        require(date.fromisoformat(text).isoformat() == text, name + ' canonical date')
        out.append(text)
    return np.asarray(out, dtype='datetime64[D]')


def _axes(arrays, meta, n):
    shape = np.shape(arrays['baseline'])
    require(len(shape) == 2 and shape[1] > 0, 'Original axes')
    total, width = shape
    n = total if n is None else n
    require(type(n) is int and 0 <= n <= total, 'Query prefix')
    require(all(isinstance(meta.get(k), list) and len(meta[k]) == total
                for k in ('dates', 'ends', 'symbols')), 'Monthly metadata axes')
    qd, qe = _days(meta['dates'][:n], 'Query dates'), _days(meta['ends'][:n], 'Query ends')
    require(np.all(qe > qd) and (n < 2 or np.all(qd[1:] > qd[:-1])), 'Query chronology')
    for row in meta['symbols'][:n]:
        require(isinstance(row, list) and len(row) == width and len(set(row)) == width
                and all(isinstance(s, str) and s and s.strip() == s for s in row), 'Query identities')
    return n, width, qd


def _array(arrays, key, shape, n, kind='f', finite=True):
    a = np.asarray(arrays[key])
    dtype = np.dtype(bool) if kind == 'b' else np.dtype(float)
    require(a.shape == shape and a.dtype == dtype, 'Original layout ' + key)
    out = a[:n].copy()
    if kind == 'f' and finite:
        require(np.isfinite(out).all(), 'Finite original ' + key)
    return out


def _rank(raw, use, fallback):
    out = fallback.copy()
    for i in range(len(out)):
        if use[i].any():
            out[i, use[i]] = (pd.Series(raw[i]).where(use[i]).rank(pct=True)
                              .fillna(.5).to_numpy()[use[i]] - .5)
    return out


def _original(arrays, n, width):
    shape = np.shape(arrays['baseline'])
    E = _array(arrays, 'eligible', shape, n, 'b')
    available = _array(arrays, 'forecast_available', shape, n, 'b')
    use = E & available
    raw = _array(arrays, 'raw_source_log', shape, n, finite=False)
    rank = _array(arrays, 'rank_log_excess', shape, n)
    require(not np.isinf(raw).any() and np.isfinite(raw[use]).all(), 'Original pooled support')
    require(np.all(np.abs(rank) <= .5 + 1e-12), 'Original rank bounds')
    np.testing.assert_array_equal(_rank(raw, use, rank), rank)
    ma = _array(arrays, 'model_available', shape, n, 'b')
    linear = _array(arrays, 'raw_local_linear', shape, n, finite=False)
    constant = _array(arrays, 'raw_local_constant', shape, n, finite=False)
    require(not np.isinf(linear).any() and not np.isinf(constant).any(), 'Original local finite or NaN')
    ready = use.any(axis=1) & np.all(~use | (ma & np.isfinite(linear) & np.isfinite(constant)), axis=1)
    if 'paired_month_ready' in arrays:
        paired = _array(arrays, 'paired_month_ready', (shape[0],), n, 'b')
        np.testing.assert_array_equal(ready, paired)
    weak = _array(arrays, 'state_weak_narrow', (shape[0],), n, 'b')
    return raw, available, rank, use, ready, weak


def physical_inputs(train, combined_meta, arrays, n):
    """Materialize only query-prefix arrays and mature whole training cohorts.

    Date/offset headers may be inspected to locate the cut; future numeric
    feature/target bodies and future identity lists are never consumed.
    """
    n, width, qd = _axes(arrays, combined_meta['monthly'], n)
    _, _, _, use, ready, weak = _original(arrays, n, width)
    formal = qd != np.datetime64(INITIAL_PARTIAL)
    cutoffs = [np.datetime64('%04d-12-31' % (int(str(d)[:4]) - 1)) for d in qd[formal]]
    active = formal & weak & ready & use.any(axis=1)
    cutoffs.extend(qd[active])
    latest = max(cutoffs) if cutoffs else None
    tm = combined_meta['training']
    require(all(isinstance(tm.get(k), list) for k in ('dates', 'ends', 'symbols')), 'Training metadata lists')
    td, te = _days(tm['dates'], 'Training signal headers'), _days(tm['ends'], 'Training end headers')
    require(len(td) == len(te) == len(tm['symbols']), 'Training metadata lengths')
    require(np.all(te > td) and np.all(te.astype('datetime64[M]') == td.astype('datetime64[M]') + 1), 'Next-month labels')
    require(len(td) < 2 or (np.all(td[1:] > td[:-1]) and np.all(te[1:] >= te[:-1])), 'Training cohort chronology')
    offsets = np.asarray(train['offsets'])
    require(offsets.dtype.kind in 'iu' and offsets.shape == (len(td) + 1,)
            and offsets[0] == 0 and np.all(np.diff(offsets) > 0), 'Training offsets')
    count = int(np.count_nonzero(te <= latest)) if latest is not None else 0
    stop, rows = int(offsets[count]), int(offsets[-1])
    total = np.shape(arrays['baseline'])[0]
    sizes = dict(dates=(rows,), ends=(rows,), train_plain=(rows, 10), common_good=(rows,),
                 event_dates=(total * width,), event_plain=(total * width, 10))
    sizes.update({'target_' + key: (rows,) for key in TARGETS})
    t = {}
    for key, shape in sizes.items():
        require(np.shape(train[key]) == shape, 'Source shape ' + key)
        limit = n * width if key.startswith('event_') else stop
        t[key] = np.asarray(train[key])[:limit].copy()
    t['offsets'] = offsets[:count + 1].copy()
    m = deepcopy({key: combined_meta['monthly'][key][:n] for key in ('dates', 'ends', 'symbols')})
    m['etfs'] = deepcopy(combined_meta['monthly']['etfs'])
    mt = deepcopy({key: tm[key][:count] for key in ('dates', 'ends', 'symbols')})
    a = {}
    for key, value in arrays.items():
        require(np.shape(value) and np.shape(value)[0] == total, 'Original array first axis ' + key)
        a[key] = np.asarray(value)[:n].copy()
    scope = dict(events=n, width=width, consumed_training_rows=stop,
                 training_months=count, latest_needed_cutoff=str(latest) if latest is not None else None,
                 future_training_numeric_accessed=False, future_queries_accessed=False)
    return t, dict(monthly=m, training=mt), a, scope


def _floats(value, name):
    a = np.asarray(value)
    if not a.size:
        return np.empty(a.shape, float)
    require(a.dtype.kind in 'fiu', 'Numeric nonboolean ' + name)
    return a.astype(float, copy=True)


def model_inputs(train, combined_meta, arrays, n=None):
    n, width, qd = _axes(arrays, combined_meta['monthly'], n)
    t, meta, a, scope = physical_inputs(train, combined_meta, arrays, n)
    tm = meta['training']
    td, te = _days(tm['dates'], 'Consumed signals'), _days(tm['ends'], 'Consumed ends')
    widths = []
    symbols = []
    for row in tm['symbols']:
        require(isinstance(row, list) and len(row) > 0 and len(set(row)) == len(row)
                and all(isinstance(s, str) and s and s.strip() == s for s in row), 'Training identities')
        widths.append(len(row)); symbols.extend(row)
    dates, ends = _days(t['dates'], 'Training dates'), _days(t['ends'], 'Training ends')
    require(np.array_equal(dates, np.repeat(td, widths)) and np.array_equal(ends, np.repeat(te, widths)), 'Flat training dates exact')
    require(np.array_equal(t['offsets'], np.r_[0, np.cumsum(widths)]), 'Identity offsets exact')
    event_dates = _days(t['event_dates'], 'Flat query dates')
    require(np.array_equal(event_dates, np.repeat(qd, width)), 'Flat query dates exact')
    # Initial-only and empty prefixes do not convert any model feature values.
    formal = qd != np.datetime64(INITIAL_PARTIAL)
    Q = np.zeros((n * width, 10), float)
    if formal.any():
        qi = np.flatnonzero(np.repeat(formal, width))
        Q[qi] = _floats(t['event_plain'][qi], 'Consumed formal queries')
    X = _floats(t['train_plain'], 'Consumed training features')
    good = np.asarray(t['common_good'])
    require(good.dtype == np.dtype(bool), 'Exact boolean common_good')
    targets = {key: _floats(t['target_' + key], 'Target ' + key) for key in TARGETS}
    valid = good & np.isfinite(X).all(axis=1)
    for y in targets.values():
        valid &= np.isfinite(y)
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        selection = months == month
        if np.count_nonzero(selection & valid) < 2:
            valid[selection] = False
    egood = a['eligible'].reshape(-1) & np.isfinite(Q).all(axis=1)
    return dict(dates=dates, ends=ends, symbols=np.asarray(symbols, str), X=X, targets=targets,
                common_good=good.copy(), valid=valid, Q=Q, event_good=egood,
                query_dates=qd, n=n, width=width, scope=scope)


def fit_ridge(X, y, weights):
    """Frozen operation order; singular/nonfinite equations fail without rescue."""
    require(X.ndim == 2 and X.shape[1] == 10 and y.shape == weights.shape == (len(X),), 'Ridge axes')
    require(all(np.isfinite(v).all() for v in (X, y, weights)) and np.all(weights > 0), 'Finite positive Ridge inputs')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        mean = np.average(X, axis=0, weights=weights)
        centered = X - mean
        scale = np.sqrt(np.average(centered * centered, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.)
        z = centered / scale
        intercept = float(np.average(y, weights=weights))
        gram = np.einsum('ni,n,nj->ij', z, weights, z, optimize=False)
        gram += 10. * np.eye(10)
        rhs = np.einsum('ni,n,n->i', z, weights, y - intercept, optimize=False)
    require(all(np.isfinite(v).all() for v in (mean, scale, gram, rhs)) and np.isfinite(intercept), 'Finite Ridge equations')
    beta = np.linalg.solve(gram, rhs)
    require(np.isfinite(beta).all(), 'Finite Ridge coefficients')
    return (mean, scale, beta, intercept), gram, rhs


def _record(s, kind, query_indices, cutoff, query_rows):
    lower = numeric._years_before(cutoff)
    ids = np.flatnonzero(s['valid'] & (s['dates'] > lower) & (s['ends'] <= cutoff))
    months = len(np.unique(s['dates'][ids].astype('datetime64[M]')))
    status = 'cold_start' if months < 18 else 'no_prediction_rows' if not len(query_rows) else 'fit'
    if kind == 'monthly':
        require(status == 'fit', 'Monthly required U lacks 18-month mature fit; no fallback')
    r = dict(kind=kind, query_indices=list(query_indices), cutoff=str(cutoff), lower_bound=str(lower),
             training_row_ids=ids.tolist(), query_row_ids=query_rows.tolist(), training_months=months,
             status=status, signal_dates=s['dates'][ids].astype(str).tolist(),
             label_end_dates=s['ends'][ids].astype(str).tolist(), symbols=s['symbols'][ids].tolist(),
             weights=[], weights_sha256=None, mean=None, scale=None, beta=None, intercept=None,
             gram=None, rhs=None, raw_predictions=[])
    if status == 'fit':
        X, Q = s['X'][ids], s['Q'][query_rows]
        require(np.isfinite(Q).all() and np.max(np.abs(X)) <= .5 + 1e-12
                and np.max(np.abs(Q)) <= .5 + 1e-12, 'Original plain10 rank bounds')
        require(len(set(zip(s['dates'][ids].astype('datetime64[M]').astype(str), s['symbols'][ids]))) == len(ids), 'Unique security-month')
        weights = numeric.month_balanced_weights(s['dates'][ids], cutoff, 12.)
        fit, gram, rhs = fit_ridge(X, s['targets']['log_excess'][ids], weights)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            prediction = numeric._predict(fit, Q)
        require(prediction.shape == (len(query_rows),) and np.isfinite(prediction).all(), 'Finite complete U prediction')
        r.update(weights=weights.tolist(), weights_sha256=_digest(weights), mean=fit[0].tolist(),
                 scale=fit[1].tolist(), beta=fit[2].tolist(), intercept=fit[3],
                 gram=gram.tolist(), rhs=rhs.tolist(), raw_predictions=prediction.tolist())
    return r


def _snapshot(s, arrays, n):
    def bit(a):
        a = np.asarray(a)
        return a.dtype.str, a.shape, a.tobytes()
    source = {k: ({q: bit(v) for q, v in value.items()} if k == 'targets' else
                   deepcopy(value) if k == 'scope' else bit(value)) for k, value in s.items()}
    return source, {k: bit(np.asarray(v)[:n]) for k, v in arrays.items()}


def fit_models(train, combined_meta, arrays, n=None, *, control_validator=None):
    s = model_inputs(train, combined_meta, arrays, n)
    n, width, dates = s['n'], s['width'], s['query_dates']
    snap = _snapshot(s, arrays, n)
    raw, available, rank, use, ready, weak = _original(arrays, n, width)
    formal = dates != np.datetime64(INITIAL_PARTIAL)
    active = formal & weak & ready & use.any(axis=1)
    expected = available.copy(); expected[formal] = False
    control_raw = raw.copy(); annual = []
    years = sorted({int(str(d)[:4]) for d in dates[formal]})
    counts = dict.fromkeys(MODEL_COUNT_FIELDS, 0)
    counts.update(events=n, width=width, training_rows=len(s['dates']),
                  valid_training_rows=int(s['valid'].sum()), annual_groups=len(years))
    with threadpool_limits(limits=1):
        for year in years:
            rows = [i for i in range(n) if formal[i] and int(str(dates[i])[:4]) == year]
            query_rows = np.concatenate([i * width + np.flatnonzero(use[i]) for i in rows]).astype(int)
            cutoff = np.datetime64('%04d-12-31' % (year - 1))
            r = _record(s, 'annual', rows, cutoff, query_rows)
            if r['training_months'] >= 18:
                expected[rows] = s['event_good'].reshape(n, width)[rows]
            if r['status'] == 'fit':
                predicted = np.asarray(r['raw_predictions'])
                np.testing.assert_array_equal(predicted, raw.reshape(-1)[query_rows])
                control_raw.reshape(-1)[query_rows] = predicted
                counts['annual_ridge_fits'] += 1
                counts['annual_prediction_rows'] += len(query_rows)
            annual.append(r)
    np.testing.assert_array_equal(expected, available)
    np.testing.assert_array_equal(control_raw, raw)
    control_rank = _rank(control_raw, use, rank)
    np.testing.assert_array_equal(control_rank, rank)
    counts['actual_model_fits'] = counts['annual_ridge_fits']
    lp = dict(target_keys=list(TARGETS), target_sha256={k: _digest(v) for k, v in s['targets'].items()},
              common_good_sha256=_digest(s['common_good']), labels_unchanged=True,
              common_good_unchanged=True, original_row_order_preserved=True,
              recentered=False, raw_target_reconstructed=False)
    audit = dict(schema=_schema('model_audit'), counts=counts, annual_control_exact=True,
                 original_ranking_population=True, paired_ready_preserved=True,
                 local_models_reused=True, other_targets_not_fitted=True, full_clock_change=True,
                 strict_pit=False, computes_returns=False, control_gate=None,
                 label_preservation=lp, source_scope=deepcopy(s['scope']))
    bundle = dict(n=n, width=width,
                  control=dict(raw=control_raw, available=available.copy(), rank=control_rank, audit=dict(fits=annual)),
                  candidate=None, audit=audit)
    if control_validator is None:
        gate = dict(passed=True, synthetic_no_callback=True)
    else:
        require(callable(control_validator), 'Callable control gate')
        gate = control_validator(deepcopy(bundle))
        require(type(gate) is dict and gate.get('passed') is True, 'Independent control gate failed')
    after = model_inputs(train, combined_meta, arrays, n)
    require(snap == _snapshot(after, arrays, n), 'Consumed source changed by control gate')
    candidate_raw = control_raw.copy(); monthly = []
    with threadpool_limits(limits=1):
        for i in np.flatnonzero(active):
            query_rows = i * width + np.flatnonzero(use[i])
            r = _record(s, 'monthly', [int(i)], dates[i], query_rows)
            candidate_raw.reshape(-1)[query_rows] = r['raw_predictions']
            counts['monthly_ridge_fits'] += 1
            counts['monthly_prediction_rows'] += len(query_rows)
            monthly.append(r)
    candidate_rank = _rank(candidate_raw, use, rank)
    counts['actual_model_fits'] += counts['monthly_ridge_fits']
    bundle['candidate'] = dict(raw=candidate_raw, available=available.copy(), rank=candidate_rank, audit=dict(fits=monthly))
    bundle['audit']['control_gate'] = deepcopy(gate)
    require(snap == _snapshot(model_inputs(train, combined_meta, arrays, n), arrays, n), 'Consumed source changed during fit')
    return bundle


def rank_and_delta(arrays, monthly_meta, bundle, n=None):
    n, width, dates = _axes(arrays, monthly_meta, n)
    require(type(bundle) is dict and set(bundle) == set(BUNDLE_KEYS)
            and type(bundle['n']) is int and n <= bundle['n'] and bundle['width'] == width, 'Model bundle prefix')
    raw, available, old_rank, use, ready, weak = _original(arrays, n, width)
    formal = dates != np.datetime64(INITIAL_PARTIAL)
    active = use & (formal & ready & weak)[:, None]
    for role in ('control', 'candidate'):
        f = bundle[role]
        require(type(f) is dict and set(f) == set(FORECAST_KEYS), 'Forecast schema')
        shape = (bundle['n'], width)
        values = _array(f, 'raw', shape, n, finite=False)
        av = _array(f, 'available', shape, n, 'b')
        rank = _array(f, 'rank', shape, n)
        np.testing.assert_array_equal(av, available)
        require(np.isfinite(values[use]).all() and not np.isinf(values).any(), 'Complete forecast U')
        np.testing.assert_array_equal(_rank(values, use, old_rank), rank)
        if role == 'control':
            np.testing.assert_array_equal(values, raw)
            np.testing.assert_array_equal(rank, old_rank)
        else:
            np.testing.assert_array_equal(values[~active], raw[~active])
            np.testing.assert_array_equal(rank[~active], old_rank[~active])
            candidate_rank = rank
    delta = np.zeros((n, width), float)
    delta[active] = .0375 * (candidate_rank[active] - old_rank[active])
    require(np.isfinite(delta).all(), 'Finite exact pooled slot delta')
    audit = dict(schema=_schema('rank_delta'), events=n, width=width, ranking_cells=int(use.sum()),
                 active_indices=[np.flatnonzero(row).tolist() for row in active],
                 changed_cells=int(np.count_nonzero(delta)), original_ranking_population=True,
                 paired_ready_preserved=True, local_basis_preserved=True, weak_only=True,
                 partial_exact=True, calm_slot_preserved=True, no_post_rerank=True, computes_returns=False)
    return candidate_rank, delta, audit
