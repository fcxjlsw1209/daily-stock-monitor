"""Independent prospective267 mathematics on caller-supplied synthetic inputs.

The author previously reviewed the producer: this is not a blind review.
No producer/legacy imports, file access, actual-source admission, fitting of
other models, or portfolio replay occur here. Model verification rebuilds the
cohort and uses augmented SVD, not the producer's normal-equation solver.
Canonical Gram/rhs checks deliberately share the declared NumPy einsum
contraction; only the separate solution algorithm is independent.
Caller-owned source authentication and before/after input bindings remain
necessary. A returned digest is a mutation binding, not receipt authentication.
"""
from datetime import date
import hashlib
import json
import math
from numbers import Real

import numpy as np

ARMS = ('top5_all_pairs_control', 'original_p34_pair_candidate')
INITIAL_PARTIAL = '2016-08-17'
CONTINUOUS_ATOL = CONTINUOUS_RTOL = 2e-10
WORK_FIELDS = ('cohort_reconstructions', 'date_header_rows', 'mature_month_bodies',
               'selected_feature_rows', 'endpoint_return_evaluations',
               'control_pair_rows', 'candidate_pair_rows',
               'independent_equation_checks', 'independent_svd_solves',
               'independent_query_predictions', 'effect_mapping_checks',
               'independent_selector_calls', 'selector_events', 'selector_candidate_cells')
MODEL_FLAGS = ('maturity_before_numeric_body', 'original_topk_before_validity',
               'common_months_exact', 'pair_arrays_and_weights_exact',
               'canonical_month_weight_rule', 'no_training_gap_filter',
               'alpha10_no_intercept_no_scaling', 'independent_augmented_svd',
               'actual_solve_counts_checked', 'model_fingerprints_bound')
ADJUSTMENT_FLAGS = ('same_year_availability_checked', 'original_boundary_pair_checked',
                    'effect_mapping_exact', 'untouched_effect_bytes_exact',
                    'zero_and_inapplicable_identity_exact', 'no_forced_pair_winner')
SELECTION_FLAGS = ('original_top2_and_boundary_rule', 'single_original_episode_checked',
                   'original_pre_score_bytes_preserved', 'ordered_choices_exact',
                   'final_scores_exact', 'no_global_rerank')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _work():
    return dict.fromkeys(WORK_FIELDS, 0)


def _number(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
        return None
    try:
        x = float(v)
    except (ValueError, TypeError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _date(s):
    require(type(s) is str and len(s) == 10, 'Canonical date string')
    try:
        d = date.fromisoformat(s)
    except ValueError as exc:
        raise ValueError('Invalid date') from exc
    require(d.isoformat() == s, 'Canonical date spelling')
    return d


def _row(v, n, name):
    require(isinstance(v, (list, tuple, np.ndarray)) and len(v) == n, name + ' length')
    if isinstance(v, np.ndarray):
        require(v.ndim == 1, name + ' dimension')


def _features(v):
    if not isinstance(v, (list, tuple, np.ndarray)):
        return None
    if isinstance(v, np.ndarray) and v.ndim != 1:
        return None
    if len(v) != 10:
        return None
    x = [_number(z) for z in v]
    return None if any(z is None or z < -.5 or z > .5 for z in x) else np.array(x, dtype=np.float64)


def _order(p, eligible, symbols):
    require(isinstance(symbols, (list, tuple)) and len(symbols) > 0, 'Original identities')
    require(all(type(s) is str and s and s.strip() == s for s in symbols)
            and len(set(symbols)) == len(symbols), 'Unique canonical identities')
    _row(p, len(symbols), 'Original P'); _row(eligible, len(symbols), 'Original E')
    ranked = []
    for j, s in enumerate(symbols):
        require(isinstance(eligible[j], (bool, np.bool_)), 'Original E boolean')
        if eligible[j]:
            value = _number(p[j])
            require(value is not None, 'Eligible P finite before sorting')
            ranked.append((-value, s, j))
    ranked.sort()
    return [j for _, _, j in ranked]


def _packed(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), values=[float(v).hex() for v in a.flat])


def _exact(a, b, where):
    if isinstance(b, np.ndarray):
        require(isinstance(a, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes(order='C') == b.tobytes(order='C'), 'Exact array bytes: ' + where)
    elif type(b) is float:
        require(type(a) is float and a.hex() == b.hex(), 'Exact float bits: ' + where)
    elif type(b) is dict:
        require(type(a) is dict and set(a) == set(b), 'Exact dictionary keys: ' + where)
        for k in b:
            _exact(a[k], b[k], where + '/' + str(k))
    elif type(b) in (list, tuple):
        require(type(a) is type(b) and len(a) == len(b), 'Exact sequence: ' + where)
        for j, (x, y) in enumerate(zip(a, b)):
            _exact(x, y, where + '/' + str(j))
    else:
        require(type(a) is type(b) and a == b, 'Exact value/type: ' + where)


def model_fingerprint(model):
    """Bind the already checked model; this digest does not authenticate a gate."""
    def typed(x):
        if isinstance(x, np.ndarray):
            require(x.dtype != object, 'No object array in model binding')
            return ['array', x.dtype.str, list(x.shape), x.tobytes(order='C').hex()]
        if type(x) is float:
            return ['float', x.hex()]
        if type(x) is dict:
            require(all(type(k) is str for k in x), 'Model string keys')
            return ['dict', [[k, typed(x[k])] for k in sorted(x)]]
        if type(x) in (list, tuple):
            return [type(x).__name__, [typed(v) for v in x]]
        require(x is None or type(x) in (bool, int, str), 'Supported model binding type')
        return [type(x).__name__, x]
    return hashlib.sha256(json.dumps(typed(model), separators=(',', ':')).encode()).hexdigest()


def reference_cohort(months, query_year):
    """Rebuild calendar/identity/pair rows independently; no producer result input."""
    require(type(query_year) is int and 7 <= query_year <= 9999, 'Native query year')
    require(isinstance(months, (list, tuple)), 'Monthly records')
    cutoff, lower = date(query_year - 1, 12, 31), date(query_year - 6, 12, 31)
    work = _work(); work['cohort_reconstructions'] = 1
    headers, seen, previous = [], set(), None
    for i, record in enumerate(months):
        require(type(record) is dict, 'Monthly dictionary')
        signal, end = _date(record['signal']), _date(record['end'])
        work['date_header_rows'] += 1
        require(previous is None or previous < signal, 'Strict chronological signals')
        previous = signal
        if record['signal'] == INITIAL_PARTIAL:
            require(end > signal, 'Initial partial endpoint')
            status = 'initial_partial'
        else:
            ym = (signal.year, signal.month)
            require(ym not in seen, 'One original signal per month'); seen.add(ym)
            following = (signal.year + int(signal.month == 12), 1 if signal.month == 12 else signal.month + 1)
            require(end > signal and (end.year, end.month) == following, 'Next natural-month endpoint')
            status = 'outside_window' if signal <= lower else 'not_mature' if end > cutoff else 'eligible_dates'
        headers.append((i, signal, end, status))
    audit, kept = [], []
    for index, signal, end, status in headers:
        record = dict(source_index=index, signal=signal.isoformat(), end=end.isoformat(), status=status)
        audit.append(record)
        if status != 'eligible_dates':
            continue
        work['mature_month_bodies'] += 1
        source = months[index]
        require(all(k in source for k in ('symbols', 'pre_scores', 'eligible', 'features', 'signal_prices', 'end_prices')),
                'Complete mature month body')
        order = _order(source['pre_scores'], source['eligible'], source['symbols'])
        chosen = order[:5]
        record.update(eligible_count=len(order), selected_indices=chosen,
                      selected_symbols=[source['symbols'][j] for j in chosen])
        if len(order) < 4:
            record['status'] = 'fewer_than_four_eligible'
            continue
        width = len(source['symbols'])
        require(isinstance(source['features'], (list, tuple, np.ndarray)) and len(source['features']) == width,
                'Original feature rows')
        _row(source['signal_prices'], width, 'Signal prices'); _row(source['end_prices'], width, 'End prices')
        values, bad = [], []
        for j in chosen:
            work['selected_feature_rows'] += 1
            x = _features(source['features'][j])
            a, b = _number(source['signal_prices'][j]), _number(source['end_prices'][j])
            if x is None or a is None or b is None or a <= 0 or b <= 0:
                bad.append(j)
            else:
                values.append((j, x, a, b))
        if bad:
            record.update(status='invalid_selected_name', invalid_selected_indices=bad)
            continue
        returns = []
        for j, x, start, endprice in values:
            gross = endprice / start
            require(math.isfinite(gross) and gross > 0., 'Finite positive adjusted gross')
            ret = float(gross) - 1.0
            require(math.isfinite(ret), 'Finite endpoint return')
            work['endpoint_return_evaluations'] += 1
            returns.append((j, x, ret))
        record.update(status='common_valid_month', boundary_pair_indices=[chosen[2], chosen[3]])
        kept.append((index, signal, end, source['symbols'], returns))
    count = len(kept)
    ages = [12 * (cutoff.year - s.year) + cutoff.month - s.month for _, s, _, _, _ in kept]
    require(all(a >= 1 for a in ages), 'Positive mature decay age')
    decays = [float(np.exp2(-a / 12.)) for a in ages]
    require(all(math.isfinite(v) and v > 0 for v in decays), 'Positive finite decay')
    mass = math.fsum(decays)
    weights = [count * v / mass for v in decays]
    require(all(math.isfinite(v) and v > 0 for v in weights), 'Positive finite common weight')
    arms = {}
    for which, arm in enumerate(ARMS):
        X, y, w, pairs = [], [], [], []
        for month_index, (source_index, signal, end, names, rows) in enumerate(kept):
            ranks = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))] if which == 0 else [(2, 3)]
            weight = weights[month_index] / len(ranks)
            require(math.isfinite(weight) and weight > 0, 'Positive finite pair weight')
            for a, b in ranks:
                i, xi, ri = rows[a]; j, xj, rj = rows[b]
                delta = np.array([float(xi[k]) - float(xj[k]) for k in range(10)], dtype=np.float64)
                label = ri - rj
                require(np.isfinite(delta).all() and math.isfinite(label), 'Finite pair values')
                X.append(delta); y.append(label); w.append(weight)
                pairs.append(dict(source_index=source_index, signal=signal.isoformat(), end=end.isoformat(),
                                  indices=[i, j], symbols=[names[i], names[j]], original_ranks=[a + 1, b + 1],
                                  common_month_index=month_index))
        arrays = dict(X=np.array(X, dtype=np.float64).reshape(-1, 10),
                      y=np.array(y, dtype=np.float64), weights=np.array(w, dtype=np.float64))
        actual_mass = math.fsum(w)
        arms[arm] = dict(arrays, pairs=pairs, weight_mass=actual_mass, weight_mass_hex=actual_mass.hex(),
                         float_evidence={k: _packed(a) for k, a in arrays.items()})
        work['control_pair_rows' if which == 0 else 'candidate_pair_rows'] = len(y)
    result = dict(schema='monthly_boundary_pair267_cohort_v1', query_year=query_year, cutoff=cutoff.isoformat(),
                  signal_lower_exclusive=lower.isoformat(), minimum_months=18, common_months=count,
                  status='ready' if count >= 18 else 'cold', months=audit,
                  common_source_indices=[m[0] for m in kept], decay_ages=ages, decays=decays,
                  month_weights=weights, month_weights_hex=[v.hex() for v in weights], mathematical_weight_mass=count,
                  weight_rule='d=exp2(-age/12); wm=(M*d)/fsum(d); control=wm/C(K,2), candidate=wm; no residual correction',
                  training_gap_filter=False, returns_rule='float(end/signal)-1.0, then ri-rj; no clipping/excess/third',
                  source_authentication='Caller responsibility; official endpoints and original P/E are not authenticated here',
                  actual_coverage_claim=False, arms=arms)
    return result, work


def _near(a, b, label):
    require(isinstance(a, np.ndarray) and a.dtype == np.dtype('float64') and a.shape == b.shape
            and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite continuous array: ' + label)
    require(np.all(np.abs(a - b) <= CONTINUOUS_ATOL + CONTINUOUS_RTOL * np.abs(b)),
            'Independent SVD tolerance: ' + label)
    return float(np.max(np.abs(a - b))) if a.size else 0.


def _svd(X, y, w):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        root = np.sqrt(w)
        a = np.concatenate((X * root[:, None], math.sqrt(10.) * np.eye(10)), axis=0)
        target = np.concatenate((y * root, np.zeros(10)))
        require(np.isfinite(a).all() and np.isfinite(target).all(), 'Finite augmented system')
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        require(s.shape == (10,) and np.isfinite(s).all() and np.all(s > 0), 'Full-rank regularized SVD')
        # Explicit fsum projection/back-substitution avoids depending on BLAS
        # matmul's floating-status flags after LAPACK SVD on this runtime.
        # Each product/sum must remain finite; this is not a solver fallback.
        projection = [math.fsum(float(u[n, j]) * float(target[n]) for n in range(len(target)))
                      for j in range(10)]
        scaled = [projection[j] / float(s[j]) for j in range(10)]
        require(all(math.isfinite(v) for v in projection + scaled), 'Finite SVD projection')
        beta = np.array([math.fsum(float(vt[j, k]) * scaled[j] for j in range(10))
                         for k in range(10)], dtype=np.float64)
    require(np.isfinite(beta).all(), 'Finite independent SVD solution')
    return beta


def verify_models(months, query_year, supplied_bundle):
    cohort, work = reference_cohort(months, query_year)
    require(type(supplied_bundle) is dict and set(supplied_bundle) == {
        'schema', 'query_year', 'cohort', 'models', 'actual_solve_calls',
        'pair_loss_set_only_difference', 'actual_coverage_claim'}, 'Full pair model bundle')
    _exact(supplied_bundle['schema'], 'monthly_boundary_pair267_models_v1', 'models schema')
    _exact(supplied_bundle['query_year'], query_year, 'query year')
    _exact(supplied_bundle['cohort'], cohort, 'full independently reconstructed cohort')
    require(type(supplied_bundle['models']) is dict and set(supplied_bundle['models']) == set(ARMS), 'Both pair models')
    comparisons, bindings, solutions = {}, {}, {}
    for arm in ARMS:
        row = cohort['arms'][arm]; model = supplied_bundle['models'][arm]
        expected = dict(schema='monthly_boundary_pair267_model_v1', arm=arm, query_year=query_year,
                        status=cohort['status'], common_months=cohort['common_months'], alpha=10., intercept=0.,
                        rescaled=False, reverse_augmented=False, rows=len(row['y']),
                        actual_solve_calls=int(cohort['status'] == 'ready'))
        require(type(model) is dict and set(model) == set(expected) | {'beta', 'gram', 'rhs', 'float_evidence'}, 'Complete model fields')
        for key, value in expected.items():
            _exact(model[key], value, 'model/' + key)
        if cohort['status'] == 'cold':
            _exact(model['beta'], np.zeros(10), 'cold beta bits')
            _exact(model['gram'], None, 'cold Gram'); _exact(model['rhs'], None, 'cold rhs')
            error = 0.
            solutions[arm] = None
        else:
            X, y, w = row['X'], row['y'], row['weights']
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                gram = np.einsum('ni,n,nj->ij', X, w, X, optimize=False)
                gram += 10.0 * np.eye(10)
                rhs = np.einsum('ni,n,n->i', X, w, y, optimize=False)
            work['independent_equation_checks'] += 1
            _exact(model['gram'], gram, 'canonical weighted Gram')
            _exact(model['rhs'], rhs, 'canonical weighted rhs')
            beta = _svd(X, y, w); work['independent_svd_solves'] += 1
            error = _near(model['beta'], beta, arm + '/beta')
            solutions[arm] = _packed(beta)
        expected_evidence = {k: None if model[k] is None else _packed(model[k]) for k in ('beta', 'gram', 'rhs')}
        _exact(model['float_evidence'], expected_evidence, 'saved exact fitted arrays')
        comparisons[arm] = dict(max_beta_absolute_error=error, common_months=cohort['common_months'],
                                rows=len(row['y']), actual_weight_mass_hex=row['weight_mass_hex'])
        bindings[arm] = model_fingerprint(model)
    _exact(supplied_bundle['actual_solve_calls'], 2 if cohort['status'] == 'ready' else 0, 'Producer solve count')
    _exact(supplied_bundle['pair_loss_set_only_difference'], True, 'Only pair loss differs')
    _exact(supplied_bundle['actual_coverage_claim'], False, 'No actual coverage admission')
    return dict(schema='monthly_boundary_pair267_independent_models_v1', passed=True, query_year=query_year,
                common_months=cohort['common_months'], status=cohort['status'], work=work,
                comparisons=comparisons, model_sha256=bindings, svd_beta=solutions,
                continuous_atol=CONTINUOUS_ATOL, continuous_rtol=CONTINUOUS_RTOL,
                exact_cohort_bits=True, canonical_equations_exact=True,
                numerical_method='Independent augmented SVD; shared caller data/NumPy/canonical einsum, non-blind author',
                actual_coverage_claim=False, receipt_authentication_performed=False,
                **dict.fromkeys(MODEL_FLAGS, True))


def verify_adjustment(event, model, supplied_effects, supplied_audit, *, model_verification=None):
    """Exact map from a caller's already independently checked production beta.

    Passing the current verify_models receipt checks its model binding and the
    independent SVD query prediction, without another solve. A receipt is data,
    not authentication: the host must actually invoke the model verifier and
    bind the returned object before deployment. Without it, only the canonical
    effect map is verified, and independent_svd_prediction_checked is false.
    """
    work = _work(); work['effect_mapping_checks'] = 1
    binding, checked_svd, prediction_error = None, False, None
    require(type(event) is dict and type(model) is dict, 'Event/model dictionaries')
    signal = _date(event['signal'])
    require(model.get('schema') == 'monthly_boundary_pair267_model_v1' and model.get('arm') in ARMS
            and type(model.get('query_year')) is int and model['query_year'] == signal.year,
            'Same-year pair model')
    M = model.get('common_months'); status = model.get('status')
    require(type(M) is int and M >= 0 and status in ('ready', 'cold') and (M >= 18) == (status == 'ready'),
            'Common eighteen-month model availability')
    original = event['effects']
    require(isinstance(original, np.ndarray) and original.dtype == np.dtype('float64') and original.ndim == 1
            and np.isfinite(original).all(), 'Original finite effects')
    expected = original.copy()
    audit = dict(signal=event['signal'], arm=model['arm'], status='initial_partial' if event['signal'] == INITIAL_PARTIAL else 'cold',
                 pair_indices=[], predicted_return_gap=None, q=0., changed_indices=[], original_selector_calls=0,
                 original_pre_scores_unchanged=True, new_global_score_ranks_computed=False, winner_selected=False)
    if event['signal'] != INITIAL_PARTIAL and status == 'ready':
        order = _order(event['pre_scores'], event['eligible'], event['symbols'])
        require(len(original) == len(event['symbols']), 'Effects identity width')
        if len(order) < 4:
            audit['status'] = 'fewer_than_four_eligible'
        else:
            i, j = order[2], order[3]
            gap = float(event['pre_scores'][i]) - float(event['pre_scores'][j])
            require(math.isfinite(gap) and gap >= 0, 'Boundary gap')
            audit.update(pair_indices=[i, j], original_gap=gap)
            if gap > .02:
                audit['status'] = 'outside_deployment_gap'
            else:
                xs = event['features']
                require(isinstance(xs, (list, tuple, np.ndarray)) and len(xs) == len(original), 'Query features axis')
                xi, xj = _features(xs[i]), _features(xs[j])
                require(xi is not None and xj is not None, 'Required pair feature validity')
                beta = model['beta']
                require(isinstance(beta, np.ndarray) and beta.shape == (10,) and beta.dtype == np.dtype('float64')
                        and np.isfinite(beta).all(), 'Finite production beta')
                for key, value in [('alpha', 10.), ('intercept', 0.), ('rescaled', False), ('reverse_augmented', False)]:
                    _exact(model.get(key), value, 'Deployment model/' + key)
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    prediction = float(np.dot(xi - xj, beta))
                work['independent_query_predictions'] += 1
                require(math.isfinite(prediction) and math.isfinite(prediction / .05), 'Finite prediction and scale')
                binding = model_fingerprint(model)
                if model_verification is not None:
                    proof = model_verification
                    require(type(proof) is dict and proof.get('schema') == 'monthly_boundary_pair267_independent_models_v1'
                            and proof.get('passed') is True and all(proof.get(k) is True for k in MODEL_FLAGS),
                            'Complete prior model verification')
                    _exact(proof.get('query_year'), signal.year, 'Verified model year')
                    _exact(proof.get('status'), 'ready', 'Verified model availability')
                    _exact(proof.get('common_months'), M, 'Verified common month count')
                    _exact(proof.get('continuous_atol'), CONTINUOUS_ATOL, 'Declared atol')
                    _exact(proof.get('continuous_rtol'), CONTINUOUS_RTOL, 'Declared rtol')
                    require(type(proof.get('model_sha256')) is dict
                            and proof['model_sha256'].get(model['arm']) == binding, 'Unchanged verified model binding')
                    require(type(proof.get('svd_beta')) is dict, 'Independent SVD coefficient evidence')
                    packed = proof['svd_beta'].get(model['arm'])
                    require(type(packed) is dict and set(packed) == {'dtype', 'shape', 'values'}
                            and packed['dtype'] == np.dtype('float64').str and packed['shape'] == [10]
                            and type(packed['values']) is list and len(packed['values']) == 10
                            and all(type(v) is str for v in packed['values']), 'Typed SVD coefficient shape')
                    svd_beta = np.array([float.fromhex(v) for v in packed['values']], dtype=np.float64)
                    _exact(_packed(svd_beta), packed, 'Canonical SVD float encodings')
                    require(np.isfinite(svd_beta).all(), 'Finite verified SVD coefficients')
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        svd_prediction = float(np.dot(xi - xj, svd_beta))
                    work['independent_query_predictions'] += 1
                    prediction_error = _near(np.array([prediction]), np.array([svd_prediction]), 'Query prediction')
                    checked_svd = True
                q = .5 * math.tanh(prediction / .05)
                require(math.isfinite(q) and -.5 <= q <= .5, 'Bounded correction')
                audit.update(predicted_return_gap=prediction, predicted_return_gap_hex=prediction.hex(), q=q, q_hex=q.hex())
                if q == 0.:
                    audit['status'] = 'zero_prediction'
                else:
                    expected[i] = min(.5, max(-.5, float(original[i]) + q))
                    expected[j] = min(.5, max(-.5, float(original[j]) - q))
                    audit['status'] = 'adjusted'
                    audit['changed_indices'] = [z for z in (i, j) if expected[z].tobytes() != original[z].tobytes()]
    _exact(supplied_effects, expected, 'Exact deployed effect bytes')
    _exact(supplied_audit, audit, 'Exact deployment audit')
    return dict(schema='monthly_boundary_pair267_independent_adjustment_v1', passed=True, signal=event['signal'],
                arm=model['arm'], status=audit['status'], work=work, model_sha256=binding,
                requires_prior_independent_model_gate=True, model_fitting_repeated=False,
                independent_svd_prediction_checked=checked_svd,
                max_prediction_absolute_error=prediction_error,
                continuous_atol=CONTINUOUS_ATOL, continuous_rtol=CONTINUOUS_RTOL,
                numeric_tolerance=0., actual_coverage_claim=False, **dict.fromkeys(ADJUSTMENT_FLAGS, True))


def verify_selection(pre_scores, eligible, symbols, effects, source, supplied_choices, supplied_final):
    """Independently rebuild original episode2 Top2 + rank3..5 selection once."""
    work = _work(); work['independent_selector_calls'] = 1
    require(isinstance(pre_scores, np.ndarray) and pre_scores.dtype == np.dtype('float64') and pre_scores.ndim == 2,
            'Original P float64 matrix')
    n, width = pre_scores.shape
    require(isinstance(eligible, np.ndarray) and eligible.dtype == np.dtype(bool) and eligible.shape == (n, width), 'Original E shape')
    require(isinstance(effects, np.ndarray) and effects.dtype == np.dtype('float64') and effects.shape == (n, width)
            and np.isfinite(effects).all(), 'Effects shape and validity')
    require(isinstance(symbols, (list, tuple)) and len(symbols) == n and type(source) is dict, 'Original source identities')
    expected = pre_scores.copy(); choices = []
    # This is the frozen selector's support/budget arithmetic, not a new score.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support = (source['baseline'][:n] - source['core'][:n]) / (1 - source['core_weight'][:n])
        budget = np.minimum(.05, np.maximum(0., .10 - (.025 * source['state_calm10'][:n] + .05 * source['state_weak_narrow'][:n])))
    require(isinstance(support, np.ndarray) and support.shape == (n, width)
            and isinstance(budget, np.ndarray) and budget.shape == (n,), 'Original support/budget shapes')
    for i, names in enumerate(symbols):
        work['selector_events'] += 1
        order = _order(pre_scores[i], eligible[i], names)
        if len(order) < 3:
            choices.append(order if len(order) == 2 else [])
            continue
        candidates = [j for j in order[2:5] if pre_scores[i, order[2]] - pre_scores[i, j] <= .02]
        for j in candidates:
            require(math.isfinite(float(support[i, j])) and math.isfinite(float(budget[i])), 'Finite original candidate support')
            target = min(.5, max(-.5, float(support[i, j]) + float(effects[i, j])))
            expected[i, j] += budget[i] * (target - support[i, j])
            work['selector_candidate_cells'] += 1
        winner = sorted(candidates, key=lambda j: (-expected[i, j], names[j]))[0]
        choices.append(sorted(order[:2] + [winner], key=lambda j: (-expected[i, j], names[j])))
    require(type(supplied_choices) is list and len(supplied_choices) == n, 'Ordered choices event axis')
    normalized = []
    for selected in supplied_choices:
        require(type(selected) is list and all(isinstance(j, (int, np.integer))
                and not isinstance(j, (bool, np.bool_)) for j in selected), 'Ordered choice integer indices')
        normalized.append([int(j) for j in selected])
    _exact(normalized, choices, 'Original ordered choices')
    _exact(supplied_final, expected, 'Original final P/F bytes')
    return dict(schema='monthly_boundary_pair267_independent_selection_v1', passed=True, events=n, width=width,
                work=work, original_episode_calls=1, policy=dict(budget=.05, ceiling=5, gap=.02),
                numerical_tolerance=0., new_model_fits=0, actual_coverage_claim=False,
                **dict.fromkeys(SELECTION_FLAGS, True))


verify_incumbent_selection = verify_selection
