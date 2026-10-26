"""Pure joint versus sequential security-intercept fits; caller-owned inputs only."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / 'work/monthly_local_distance256_kernel.py'
BASE_SHA = '97ada679902942f31e4881354fcf95b1fccd2119f77c09760904b8c328364e30'
CONTEXT_PATH = ROOT / 'work/monthly_joint_identity259_context.json'
CONTEXT_SHA = '0b71847a2fbfc3f44e5b0e8b86b2ecfcfaaf15a94d2fa88406f8995a46471527'
CONTRACT_PATH = ROOT / 'work/monthly_joint_identity259_IMPLEMENTATION_CONTRACT.json'
CONTRACT_SHA = '72565c3821be025ae22a2e6fc3827a8129eea002b08f135fbb93705779e37cf4'
for _path, _digest in ((BASE_PATH, BASE_SHA), (CONTEXT_PATH, CONTEXT_SHA), (CONTRACT_PATH, CONTRACT_SHA)):
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _digest:
        raise ValueError('Frozen dependency changed: ' + str(_path))
_spec = importlib.util.spec_from_file_location('identity259_original_control', BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
model, bridge, require = base.model, base.bridge, base.require
TARGETS, RAW_KEYS, RANK_KEYS = base.TARGETS, base.RAW_KEYS, base.RANK_KEYS
INITIAL_PARTIAL, POLICY = base.INITIAL_PARTIAL, dict(base.POLICY)
ARMS = ('incumbent_local67', 'sequential_identity', 'joint_identity')
ROLES = {'sequential_identity': 'diagnostic', 'joint_identity': 'candidate'}
BUNDLE_KEYS = ('schema','n','width','training_row_ids','control','diagnostic','candidate','label_values','label_preservation','control_gate','audit')
LABEL_KEYS = base.LABEL_KEYS
MODEL_COUNT_FIELDS = ('training_rows','valid_training_rows','control_target_fits','control_log_refits','control_local_evaluations','control_local_solves','annual_design_builds','qualified_security_columns','qualified_query_rows','cold_query_rows','no_qualified_columns_years','no_qualified_query_years','zero_centered_design_years','diagnostic_pooled_fits','diagnostic_dummy_solves','candidate_pooled_fits','candidate_joint_solves','actual_new_model_solves','actual_regression_solves')
FORECAST_AUDIT_KEYS = ('schema','role','feature_count','target','source_log_copied','local_constant_copied','availability_preserved','local_support_preserved','security_intercepts_only','global_penalty','dummy_penalty','source_global_parameters_fixed','common_weights','annual')
ANNUAL_KEYS = ('year','cutoff','training_row_ids','query_row_ids','qualified_symbols','qualified_query_row_ids','unqualified_query_row_ids','own_month_counts','weight_sum','weights','source_mean','source_scale','source_intercept','source_beta','dummy_mean','raw_dummy_sha256','centered_dummy_sha256','standardized_design_sha256','weighted_dummy_sum','feature_dummy_cross','centered_dummy_zero','status','global_beta','dummy_delta','gram','rhs','penalty','fit_performed')
COUNT_FIELDS = ('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells','ranking_cells','weak_applicable_cells','calm_applicable_cells','diagnostic_changed_delta_cells','candidate_changed_delta_cells','diagnostic_changed_holdings_events','candidate_changed_holdings_events','candidate_vs_diagnostic_changed_holdings_events','incumbent_episode_calls','diagnostic_episode_calls','candidate_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS = ('query_index','date','initial_partial','weak','calm','ranking_indices','active_indices','diagnostic_changed_delta_indices','candidate_changed_delta_indices','diagnostic_delta','candidate_delta','incumbent_choices','diagnostic_choices','candidate_choices','episode_budget')

# Frozen source/control helpers only; no256 geometry or candidate fitter is called.
model_inputs = base.model_inputs
control_parity = base.control_parity
_control_forecasts = base._control_forecasts
_rank = base._rank

def dependency_files():
    return tuple(dict.fromkeys((BASE_PATH, CONTEXT_PATH, CONTRACT_PATH) + tuple(base.dependency_files())))

def _digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()

def _solve(design, weights, response, penalty):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        gram = np.einsum('ni,n,nj->ij', design, weights, design, optimize=False)
        gram += np.diag(penalty)
        rhs = np.einsum('ni,n,n->i', design, weights, response, optimize=False)
    model._finite('new penalized Gram', gram)
    model._finite('new rhs', rhs)
    theta = np.linalg.solve(gram, rhs)
    model._finite('new coefficients', theta)
    return theta, gram, rhs

def identity_design(dates, symbols, X, weights, pooled, min_own_months=18):
    """Annual descriptive design. All training IDs, not just query IDs, qualify."""
    model._minimum(min_own_months, 'min_own_months')
    mean, scale, beta, intercept = pooled
    months = dates.astype('datetime64[M]')
    counts = {str(s): int(len(np.unique(months[symbols == s]))) for s in sorted(set(symbols))}
    qualified = [s for s in counts if counts[s] >= min_own_months]
    H = np.asarray(symbols[:, None] == np.asarray(qualified, dtype=str)[None, :], dtype=float)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        weight_sum = float(np.sum(weights))
        require(np.isfinite(weight_sum) and weight_sum > 0 and np.all(weights > 0), 'Positive original weights')
        dummy_mean = np.einsum('i,ij->j', weights, H, optimize=False) / weight_sum
        Hc = H - dummy_mean
        Z = (X - mean) / scale
        centered_sum = np.einsum('i,ij->j', weights, Hc, optimize=False)
        cross = np.einsum('ni,n,nj->ij', Z, weights, Hc, optimize=False)
    for name, value in (('dummy design', Hc), ('original standardized design', Z),
                        ('dummy centering', dummy_mean), ('feature-dummy cross', cross),
                        ('centered dummy sums', centered_sum)):
        model._finite(name, value)
    return dict(qualified=qualified, own_month_counts=counts, H=H, Hc=Hc, Z=Z,
                dummy_mean=dummy_mean, weight_sum=weight_sum,
                weighted_dummy_sum=centered_sum, feature_dummy_cross=cross)

def new_forecasts(args, control, contexts, *, min_own_months=18):
    dates, ends, symbols, X, targets, good, event_dates, event_symbols, event_X, event_good = args
    outputs = {}
    for role in ('diagnostic', 'candidate'):
        raw = {k: v.copy() for k, v in control['raw'].items()}
        raw['local_linear'] = raw['source_log'].copy()
        audit = dict(schema='monthly_joint_identity259_forecast_audit_v1', role=role,
            feature_count=10, target='unchanged cached centered log_excess',
            source_log_copied=True, local_constant_copied=True, availability_preserved=True,
            local_support_preserved=True, security_intercepts_only=True,
            global_penalty=10., dummy_penalty=12., source_global_parameters_fixed=role == 'diagnostic',
            common_weights=True, annual=[])
        outputs[role] = dict(raw=raw, available=control['available'].copy(),
                             local_available=control['local_available'].copy(), audit=audit)
    with threadpool_limits(limits=1):
        for context in contexts:
            ids, qids = context['ids'], context['predict']
            mean, scale, beta0, intercept = context['pooled']
            weights = model.numeric.month_balanced_weights(dates[ids], context['cutoff'], model.HALF_LIFE_MONTHS)
            design = identity_design(dates[ids], symbols[ids], X[ids], weights,
                                     context['pooled'], min_own_months)
            columns, Hc, Z = design['qualified'], design['Hc'], design['Z']
            J = len(columns)
            qualified_q = np.asarray([j for j in qids if event_symbols[j] in columns], dtype=int)
            cold_q = np.asarray([j for j in qids if event_symbols[j] not in columns], dtype=int)
            zero = not np.any(Hc != 0.)
            status = ('no_qualified_columns' if not J else
                      'no_qualified_query' if not len(qualified_q) else
                      'zero_centered_design' if zero else 'fit')
            for role in ('diagnostic', 'candidate'):
                penalty = np.full(J, 12.) if role == 'diagnostic' else np.r_[np.full(10, 10.), np.full(J, 12.)]
                beta, delta = beta0.copy(), np.zeros(J)
                gram = rhs = None
                if status == 'fit':
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        centered_y = targets['log_excess'][ids] - intercept
                        if role == 'diagnostic':
                            response = centered_y - np.einsum('ij,j->i', Z, beta0, optimize=False)
                            delta, gram, rhs = _solve(Hc, weights, response, penalty)
                        else:
                            B = np.column_stack((Z, Hc))
                            theta, gram, rhs = _solve(B, weights, centered_y, penalty)
                            beta, delta = theta[:10].copy(), theta[10:].copy()
                        Zq = (event_X[qualified_q] - mean) / scale
                        Hq = np.asarray(event_symbols[qualified_q, None] ==
                                        np.asarray(columns, dtype=str)[None, :], dtype=float) - design['dummy_mean']
                        prediction = (intercept + np.einsum('ij,j->i', Zq, beta, optimize=False)
                                      + np.einsum('ij,j->i', Hq, delta, optimize=False))
                    model._finite('new qualified query prediction', prediction)
                    outputs[role]['raw']['local_linear'][qualified_q] = prediction
                annual = dict(year=context['year'], cutoff=str(context['cutoff']),
                    training_row_ids=ids.tolist(), query_row_ids=qids.tolist(),
                    qualified_symbols=columns.copy(), qualified_query_row_ids=qualified_q.tolist(),
                    unqualified_query_row_ids=cold_q.tolist(), own_month_counts=design['own_month_counts'].copy(),
                    weight_sum=design['weight_sum'], weights=weights.tolist(), source_mean=mean.tolist(),
                    source_scale=scale.tolist(), source_intercept=float(intercept), source_beta=beta0.tolist(),
                    dummy_mean=design['dummy_mean'].tolist(), raw_dummy_sha256=_digest(design['H']),
                    centered_dummy_sha256=_digest(Hc), standardized_design_sha256=_digest(Z),
                    weighted_dummy_sum=design['weighted_dummy_sum'].tolist(),
                    feature_dummy_cross=design['feature_dummy_cross'].tolist(),
                    centered_dummy_zero=bool(zero), status=status,
                    global_beta=beta.tolist(), dummy_delta=delta.tolist(),
                    gram=None if gram is None else gram.tolist(), rhs=None if rhs is None else rhs.tolist(),
                    penalty=penalty.tolist(), fit_performed=status == 'fit')
                outputs[role]['audit']['annual'].append(annual)
    return outputs['diagnostic'], outputs['candidate']

def fit_models(train, meta, arrays, n=None, *, control_validator=None, min_months=18, min_own_months=18):
    args, ids, n, width = base._inputs(train, meta, arrays, n)
    snapshot = bridge._snapshot(arrays)
    dates, ends, symbols, X, targets, good, event_dates, event_symbols, event_X, event_good = args
    identity = list(zip(dates.astype('datetime64[M]').astype(str), symbols.tolist()))
    require(len(identity) == len(set(identity)), 'Duplicate security-month')
    require(np.isfinite(targets['log_excess'][good]).all(), 'Nonfinite valid original log target')
    before = {k: _digest(v) for k, v in targets.items()}
    good_before = _digest(good)
    control, contexts = _control_forecasts(*args, min_months=min_months, min_own_months=min_own_months)
    parity = control_parity(arrays, control, n)
    receipt = parity
    if control_validator is not None:
        require(callable(control_validator), 'Control validator callable')
        receipt = control_validator(deepcopy(control))
        require(type(receipt) is dict and receipt.get('passed') is True, 'Independent original model gate failed')
    diagnostic, candidate = new_forecasts(args, control, contexts, min_own_months=min_own_months)
    for out in (diagnostic, candidate):
        base._forecast_layout(out, n, width)
        for key in ('source_log', 'local_constant'):
            require(out['raw'][key].tobytes() == control['raw'][key].tobytes(), 'Preserved forecast changed ' + key)
        for key in ('available', 'local_available'):
            require(out[key].tobytes() == control[key].tobytes(), 'Preserved support changed ' + key)
    require(before == {k: _digest(v) for k, v in targets.items()} and good_before == _digest(good),
            'Original labels/common_good changed')
    label = dict(schema='monthly_joint_identity259_labels_preserved_v1', rows=len(ids),
        valid_rows=int(good.sum()), target_keys=list(TARGETS), target_sha256=before,
        common_good_sha256=good_before, labels_unchanged=True, common_good_unchanged=True,
        original_row_order_preserved=True, recentered=False, raw_target_reconstructed=False)
    counts = dict.fromkeys(MODEL_COUNT_FIELDS, 0)
    counts.update(training_rows=len(ids), valid_training_rows=int(good.sum()))
    for suffix, value in zip(('target_fits','log_refits','local_evaluations','local_solves'), base._fit_counts(control)):
        counts['control_' + suffix] = int(value)
    annual = candidate['audit']['annual']
    counts.update(annual_design_builds=len(annual),
        qualified_security_columns=sum(len(x['qualified_symbols']) for x in annual),
        qualified_query_rows=sum(len(x['qualified_query_row_ids']) for x in annual),
        cold_query_rows=sum(len(x['unqualified_query_row_ids']) for x in annual),
        no_qualified_columns_years=sum(x['status'] == 'no_qualified_columns' for x in annual),
        no_qualified_query_years=sum(x['status'] == 'no_qualified_query' for x in annual),
        zero_centered_design_years=sum(x['status'] == 'zero_centered_design' for x in annual),
        diagnostic_dummy_solves=sum(x['fit_performed'] for x in diagnostic['audit']['annual']),
        candidate_joint_solves=sum(x['fit_performed'] for x in annual))
    counts['actual_new_model_solves'] = counts['diagnostic_dummy_solves'] + counts['candidate_joint_solves']
    counts['actual_regression_solves'] = sum(counts[k] for k in ('control_target_fits','control_log_refits','control_local_solves','actual_new_model_solves'))
    require(snapshot == bridge._snapshot(arrays), 'Original arrays changed')
    audit = dict(schema='monthly_joint_identity259_model_audit_v1', counts=counts,
        original_cache_parity=parity, original_row_order_preserved=True,
        original_common_good_preserved=True, all_targets_preserved=True,
        diagnostic_source_log_copied=True, candidate_source_log_copied=True,
        diagnostic_local_constant_copied=True, candidate_local_constant_copied=True,
        availability_preserved=True, local_support_preserved=True,
        common_design_weights_penalties=True, security_intercepts_only=True,
        candidate_global_coefficients_refit=True, computes_returns=False)
    return dict(schema='monthly_joint_identity259_models_v1', n=n, width=width,
        training_row_ids=ids, control=control, diagnostic=diagnostic, candidate=candidate,
        label_values=targets['log_excess'].copy(), label_preservation=label,
        control_gate=deepcopy(receipt), audit=audit)

def rank_and_delta(arrays, meta, bundle, n=None, *, arm='joint_identity'):
    total, width = np.shape(arrays['baseline'])
    n = total if n is None else n
    require(type(n) is int and 0 <= n <= total and type(bundle) is dict and
        bundle.get('schema') == 'monthly_joint_identity259_models_v1' and
        type(bundle.get('n')) is int and n <= bundle['n'] and bundle.get('width') == width,
        'Model prefix axes')
    require(arm in ROLES, 'New arm name')
    require(type(meta) is dict and isinstance(meta.get('dates'), list) and len(meta['dates']) == total, 'Rank metadata')
    old_pool, old_local, use, old_ready = base._old_ranks(arrays, n, width)
    forecast = bundle[ROLES[arm]]
    base._forecast_layout(forecast, bundle['n'], width, limit=n)
    available = forecast['available'][:n*width].reshape(n, width)
    pool = forecast['raw']['source_log'][:n*width].reshape(n, width)
    local = forecast['raw']['local_linear'][:n*width].reshape(n, width)
    require(np.array_equal(available, np.asarray(arrays['model_available'])[:n]), 'New model support changed')
    for key in ('source_log', 'local_constant'):
        require(forecast['raw'][key][:n*width].tobytes() ==
                np.asarray(arrays['raw_' + key])[:n].reshape(-1).tobytes(), 'Preserved raw changed ' + key)
    ready = use.any(axis=1) & np.all(~use | (available & np.isfinite(pool) & np.isfinite(local)), axis=1)
    new_pool, new_local = old_pool.copy(), old_local.copy()
    for i in range(n):
        if ready[i]:
            new_local[i] = new_pool[i] if np.array_equal(local[i, use[i]], pool[i, use[i]]) else base._rank(local[i:i+1], use[i:i+1], new_pool[i:i+1])[0]
    weak = bridge._bool(arrays['state_weak_narrow'], (total,), n, 'weak')
    calm = bridge._bool(arrays['state_calm10'], (total,), n, 'calm')
    extension = bridge._bool(arrays['extension_available'], (total, width), n, 'extension')
    formal = np.asarray([d != INITIAL_PARTIAL for d in meta['dates'][:n]], dtype=bool)
    coefficient = .25 * (.05 * weak[:, None] + .01 * (calm & ~weak)[:, None] * extension)
    active = use & ready[:, None] & old_ready[:, None] & (coefficient != 0.) & formal[:, None]
    delta = np.zeros((n, width))
    with np.errstate(over='raise', invalid='raise'):
        delta[active] = coefficient[active] * (new_local[active] - old_local[active])
    delta[new_local == old_local] = 0.
    require(np.isfinite(delta).all() and (np.abs(delta) <= .0125).all(), 'Bounded original-quarter replacement')
    ranks = dict(old_pool=old_pool, old_local=old_local, new_pool=new_pool, new_local=new_local)
    audit = dict(schema='monthly_joint_identity259_rank_delta_v1', arm=arm, events=n,
        ranking_cells=int(use.sum()), ready_rows=ready.tolist(),
        active_indices=[np.flatnonzero(x).tolist() for x in active],
        changed_cells=int(np.count_nonzero(delta)), coefficient=coefficient.tolist(),
        original_ranking_population=True, no_group_ranking=True, no_own_history_filter=True,
        pooled_ranks_preserved=True, full_original_local_quarter=True,
        partial_exact=True, calm_quarter_included=True, no_post_rerank=True)
    return ranks, delta, audit

def decisions(study, arrays, meta, bundle, n=None, *, incumbent_validator=None, expected_incumbent_orders=None):
    total, width = np.shape(arrays['baseline'])
    n = total if n is None else n
    before = bridge._snapshot(arrays)
    cache = bridge._cache(study, arrays, meta, n)
    source = dict(baseline=cache['base'], core=cache['core'], core_weight=cache['cw'],
        eligible=cache['good'], state_calm10=cache['calm'], state_weak_narrow=cache['weak'])
    orders, scores, pres, choices, deltas, rank_audit = {}, {}, {}, {}, {}, {}
    receipt = None
    for position, arm in enumerate(ARMS):
        pre = cache['pre'].copy()
        if position:
            ranks, delta, detail = rank_and_delta(arrays, meta, bundle, n, arm=arm)
            deltas[arm], rank_audit[arm] = delta, detail
            changed = delta != 0.
            pre[changed] += delta[changed]
        chosen, post = study.ep.select_boundary(pre.copy(), cache['good'].copy(),
            deepcopy(cache['meta']['symbols']), cache['effects'].copy(),
            {k: v.copy() for k, v in source.items()}, POLICY.copy())
        post = bridge._matrix(post, (n, width), n, 'Final scores')
        choices[arm] = [[int(j) for j in row] for row in chosen]
        orders[arm] = study.rp.make_orders(choices[arm], post, cache['meta'])
        scores[arm], pres[arm] = post, pre
        if not position:
            if expected_incumbent_orders is not None:
                require(orders[arm] == expected_incumbent_orders, 'Frozen incumbent orders changed')
            if incumbent_validator is not None:
                require(callable(incumbent_validator), 'Incumbent validator callable')
                receipt = incumbent_validator(deepcopy(orders[arm]), post.copy(), pre.copy())
                require(type(receipt) is dict and receipt.get('passed') is True, 'Independent incumbent gate failed')
    use = cache['good'] & np.asarray(arrays['forecast_available'])[:n]
    diagnostic, candidate = ARMS[1:]
    events = []
    for i, day in enumerate(meta['dates'][:n]):
        require(rank_audit[diagnostic]['active_indices'][i] == rank_audit[candidate]['active_indices'][i], 'Shared model support')
        events.append(dict(query_index=i, date=day, initial_partial=day == INITIAL_PARTIAL,
            weak=bool(cache['weak'][i]), calm=bool(cache['calm'][i]),
            ranking_indices=np.flatnonzero(use[i]).tolist(), active_indices=rank_audit[candidate]['active_indices'][i],
            diagnostic_changed_delta_indices=np.flatnonzero(deltas[diagnostic][i] != 0.).tolist(),
            candidate_changed_delta_indices=np.flatnonzero(deltas[candidate][i] != 0.).tolist(),
            diagnostic_delta=deltas[diagnostic][i].tolist(), candidate_delta=deltas[candidate][i].tolist(),
            incumbent_choices=choices[ARMS[0]][i], diagnostic_choices=choices[diagnostic][i],
            candidate_choices=choices[candidate][i], episode_budget=float(cache['episode'][i])))
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    partial = sum(e['initial_partial'] for e in events)
    counts.update(events=n, symbols_per_event=width, initial_partial_events=partial,
        formal_events=n-partial, original_eligible_cells=int(cache['good'].sum()), ranking_cells=int(use.sum()),
        weak_applicable_cells=sum(len(e['active_indices']) for e in events if e['weak']),
        calm_applicable_cells=sum(len(e['active_indices']) for e in events if e['calm'] and not e['weak']),
        diagnostic_changed_delta_cells=int(np.count_nonzero(deltas[diagnostic])),
        candidate_changed_delta_cells=int(np.count_nonzero(deltas[candidate])),
        diagnostic_changed_holdings_events=sum(set(a) != set(b) for a, b in zip(choices[ARMS[0]], choices[diagnostic])),
        candidate_changed_holdings_events=sum(set(a) != set(b) for a, b in zip(choices[ARMS[0]], choices[candidate])),
        candidate_vs_diagnostic_changed_holdings_events=sum(set(a) != set(b) for a, b in zip(choices[diagnostic], choices[candidate])),
        incumbent_episode_calls=1, diagnostic_episode_calls=1, candidate_episode_calls=1)
    require(before == bridge._snapshot(arrays), 'Original cache modified')
    return orders, scores, pres, dict(schema='monthly_joint_identity259_decisions_v1',
        counts=counts, events=events, rank_delta=rank_audit, incumbent_gate=deepcopy(receipt),
        original_arrays_unchanged=True, original_eligibility_unchanged=True,
        original_core_support_preserved=True, single_original_episode_per_arm=True,
        full_original_local_quarter=True, partial_exact=True, computes_returns=False)
