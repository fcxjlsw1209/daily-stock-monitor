"""Round182: route only the existing Local67 same-security local increment.

All actual features and portfolio returns require a prospectively bound PLAN.
No new model fits, eligibility changes, or replacement of the original episode.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_trend_gate182'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_upside_retention181'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'trend_gated_local', 'uniform_local_shrink')
TRAIN, INCREMENT_LIMIT = 88, .0125
ROUTING_VECTORS = ('uniform_removed_fraction', 'candidate_removed_abs_increment', 'control_removed_abs_increment')
ROUTING_KEYS = {'pre_local67', 'upstream_pre_score', 'local_increment', *NAMES[1:], *ROUTING_VECTORS}
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('block_return_estimates',)
MODEL_FIT_ACCOUNTING = 'No new predictive ML/Ridge/GBDT fits; block_return_estimates counts three endpoint-log-return estimates per complete valid211-price cell. Prefix-validation repetitions are excluded. Routing arithmetic is not a model fit.'
PRIOR_EXPECTED = {
    'PLAN.json': 'a49e9b0842d5dec69eaa9061dceaa861337b91c3a62b2ddc4d4b216364cf212a',
    'COMPLETION.json': '461d1bd72c95833124552c817a5141c3ff07da269b26a81f86203ef157a0836f',
    'results/COMPLETION.json': 'a167e30cc9f32ed29642c1f5c24c3850e2f705f4edc607370aa1b311ffe529d3',
}
CONTEXT = ROOT / 'work/monthly_trend_gate182_context.json'
CONTEXT_SHA = 'f5ec1c2b0c0c91688f364e0f4d31646227711019a15dab03de1bd47a174fcd36'
TEST_RECEIPT = ROOT / 'work/monthly_trend_gate182_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'ROUTING.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_trend_gate182_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        prospective_context_sha256=CONTEXT_SHA,
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='When three skipped formation blocks all rise, remove only the frozen same-security local-model increment; compare conditional routing with equal-absolute-budget uniform shrinkage.',
        family='Conditional routing of an existing frozen increment; no new predictive ML fit or parameter search.',
        measure='Use211 positive finite adjusted prices for official sessions t-252..t-42. The three70-return endpoint-log gains use indices0/70/140/210. I is true iff all three gains are strictly positive; unavailable inputs set available=false and I=false.',
        source_window=dict(price_count=211, earliest_offset=252, latest_offset=42,
            formation_return_count=210, block_count=3, block_sessions=70),
        component_binding='For original config4, obtain P=study.modified_pre_score, B=study.scope.modified_pre_score(study.source_spec(config4)), and D=study.score_delta, using the same arrays and prefix. Never infer D by subtraction. Require exact P from copying B then adding nonzero D in the original order, D=0 on original ineligible cells, and abs(D)<=0.0125 with1e-12 tolerance.',
        numerical_policy='Full211 prices must be finite and positive; no filling, shortened history, floor or fallback. Nonfinite arithmetic raises. P/B/D share the same prefix shape; D is finite and P/B share any NaNs only where D=0, with no infinities. Preserve originalP exactly when no change, D=0 or feature unavailable; full removal copies B; partial removal computes B+(1-b)*D.',
        availability='Measure all49 original stock/ETF candidates without eligibility filtering. The routing universe V is the common valid-feature universe. Original ineligible D is zero, so using all V is equivalent for the removal budget to the eligible subset; do not alter eligibility.',
        scoring='Candidate copies B on valid I=true cells and keeps P elsewhere. Each row uniform b=sum_V(I*abs(D))/sum_V(abs(D)), with denominator0 giving b0; retain(1-b)*D on V. The first partial row keeps P exactly for both new arms and records zero removal. No new score budget, ranking or normalization. Each new pre-score then goes through exactly one original episode.',
        budget='Match the theoretical L1 magnitude of removed D, not the floating-point observed score difference. Candidate and uniform removed amounts agree within1e-12. Fractions, amounts, P/B/D and final routed scores are saved explicitly; route itself applies zero episodes.',
        portfolio='Unchanged27 original arrays, long core/support, forecasts, historical proxy Top32+theme17 and eligibility; long-only EW3/two halves/under2 cash; fixed adjusted units intramonth; difference rebalance; no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_uniform_shrink=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES), all123_order_and_routing_prefixes=True),
        predecessor='Round181 completed without a new execution repair. Inherit its complete input/protection chain, including177 repair evidence and sources, and bind181 PLAN and root/result completions. Preserve historical central amendment references.',
        novelty='The3x70 formation information already exists in path70; only routing the frozen local increment is the new contrast. Global removal already exists in Local67 mode=none. Earlier shrinkage/budget studies overlap conceptually; unavailable historical sources prevent exhaustive novelty claims.',
        limitation='Repeatedly reused development sample, historical proxy pool and retrospectively retrieved prices; not pristine OOS/PIT. The original local increment changed only one historical holding month versus its upstream control and is bounded by0.0125 (0.0025 in calm nonweak extension members), so an outcome with no holding changes is plausible. Mixed B/P vectors can nevertheless change other rankings. The routing idea preceded181 outcomes but the exact gate/matched budget followed them; this is continued development, not a wholly unseen-data hypothesis.',
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9,
        descriptive_statistics='block_return_estimates=3 per complete valid211-price cell, zero for missing/invalid inputs; repeated prefix validation is excluded.',
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_trend_gate182_host.py',
        'test_monthly_trend_gate182_host.py', 'monthly_trend_gate182_kernel.py',
        'test_monthly_trend_gate182_kernel.py', 'monthly_trend_gate182_independent.py',
        'test_monthly_trend_gate182_independent.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')] + [CONTEXT]


def verify(pins):
    require(isinstance(pins, dict) and pins, 'Nonempty source pins required')
    for name, digest in pins.items():
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require('CloudStorage' not in path.parts and (path.is_relative_to(ROOT) or
                str(path).startswith('/Library/Developer/CommandLineTools/')), 'Unapproved source path')
        require(path.is_file() and sha(path) == digest, 'Changed or missing source: ' + str(path))


def bind(pins, path, digest=None):
    key, actual = str(path), sha(path)
    require(digest is None or actual == digest, 'Source differs from declared digest: ' + key)
    require(key not in pins or pins[key] == actual, 'Conflicting inherited source digest: ' + key)
    pins[key] = actual


def context_binding():
    pins = {}
    bind(pins, CONTEXT, CONTEXT_SHA)
    context = read(CONTEXT)
    for name, digest in context['local_evidence_sha256'].items():
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts, 'Nonlocal prospective context evidence')
        bind(pins, path, digest)
    return pins


def predecessor_binding(state):
    """Bind completed181 and preserve inherited evidence without requiring a new repair."""
    pins = {}
    for name, digest in PRIOR_EXPECTED.items():
        bind(pins, PRIOR / name, digest)

    def local_path(name):
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts, 'Predecessor reference path')
        return path

    def ref(value, expected=None):
        require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Predecessor reference schema')
        path = local_path(value['path'])
        require(expected is None or path == expected.resolve(), 'Predecessor reference target')
        bind(pins, path, value['sha256'])
        return read(path)

    closed = ref(state['completed'][-1]['completion'], PRIOR / 'COMPLETION.json')
    plan = read(PRIOR / 'PLAN.json')
    require(closed.get('status') == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed.get('plan_sha256') == PRIOR_EXPECTED['PLAN.json'] and
            closed.get('result_completion_sha256') == PRIOR_EXPECTED['results/COMPLETION.json'] and
            closed.get('protected_files') == 21444 and closed.get('observers_modified') is False,
            'Predecessor closure incomplete')
    result = read(PRIOR / 'results/COMPLETION.json')
    require(result.get('status') == 'COMPLETE' and result.get('plan_sha256') == PRIOR_EXPECTED['PLAN.json'],
            'Predecessor result completion differs')
    inherited = plan.get('predecessor_binding', {}).get('input_sha256')
    require(type(inherited) is dict and inherited, 'Missing inherited predecessor evidence')
    for name, digest in inherited.items():
        require(plan['input_sha256'].get(name) == digest, 'Historical evidence omitted from inherited input pins')
        bind(pins, local_path(name), digest)
    bind(pins, PRIOR / 'STATE_BEFORE.json', plan['prior_central_sha256'])
    before = read(PRIOR / 'STATE_BEFORE.json')
    historical_refs = {}
    for key in ('active_execution_amendment', 'latest_execution_repair_receipt'):
        expected = before.get(key)
        require(state.get(key) == expected, 'Historical central reference changed: '+key)
        historical_refs[key] = expected
        if expected is not None:
            ref(expected)
    return dict(directory=PRIOR.name, input_sha256=pins, inherited_evidence_count=len(inherited),
                predecessor_requires_new_repair=False, historical_execution_references=historical_refs)


def feature_inputs(source):
    require(isinstance(source.get('adj'), pd.DataFrame), 'Original adjusted price panel required')
    return {'adj': source['adj']}


def estimate_counts(check):
    count, available = check.get('block_return_estimates'), check.get('available_cells')
    require(type(count) is int and type(available) is int and 0 <= available <= 122*49 and count == 3*available,
            'Descriptive block-return accounting differs')
    return dict(block_return_estimates=count)


def state_check(state):
    require(len(state['completed']) == 179 and sum(r['trials'] for r in state['completed']) == 1052,
            'Expected179 completed rounds /1052 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round181 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_TREND_GATE182_' + tag + '_' + expected + '.json')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(before)
    require(sha(backup) == expected, 'Central backup mismatch')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        require(CENTRAL.read_bytes() == before, 'Concurrent central update before commit')
        os.replace(temporary, CENTRAL)
    finally:
        temporary.unlink(missing_ok=True)


def tests():
    names = ('test_monthly_trend_gate182_host.py', 'test_monthly_trend_gate182_kernel.py',
             'test_monthly_trend_gate182_independent.py')
    modules = [module('test_trend182_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = TEST_RECEIPT
    require(not receipt.exists(), 'Test receipt is write-once')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p): sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(), 'Fresh experiment directory required')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    require(not state['registered_queue'], 'Existing registered experiment')
    prior_binding = predecessor_binding(state)
    validation = TEST_RECEIPT; t = read(validation)
    require(t['passed'] is True and t['synthetic_only'] is True and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p): sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])
    raw = read(LOCAL / 'PLAN.json')['raw_source_hashes']
    for mapping in (prior_binding['input_sha256'], context_binding()):
        for name, digest in mapping.items(): bind(pins, Path(name), digest)
    calendar_sources = read(CALENDAR)['provenance']['source_hashes']
    for p in code_files() + [PRIOR / 'PLAN.json', PRIOR / 'COMPLETION.json', validation,
                            CALENDAR, PROTECTED, LOCAL / 'PLAN.json']:
        bind(pins, p)
    for mapping in (raw, calendar_sources):
        for name, digest in mapping.items(): bind(pins, Path(name), digest)
    for name in ('DATA.npz', 'META.json', 'local_linear_ORDERS.json', 'INCUMBENT_POLICY.json'):
        bind(pins, LOCAL / name)
    for name, digest in ev.ENGINE_HASHES.items(): bind(pins, ev.ENGINE_DIR / name, digest)
    verify(pins); protected = read(PROTECTED); verify(protected)
    require(len(protected) == 21444, 'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT / 'STATE_BEFORE.json').open('xb') as stream: stream.write(before)
    plan = dict(schema='monthly_trend_gate182_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_descriptive_block_return_estimates_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_trend_routing_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed three-block consistency and matched local-increment routing; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(all(p['input_sha256'].get(name) == digest for name, digest in context_binding().items()),
            'Prospective context binding changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor evidence chain changed')
    return p


def feature_prefix(values, n):
    require(set(values) == {'consistent', 'available'}, 'Feature keys changed')
    available, consistent = np.asarray(values['available']), np.asarray(values['consistent'])
    require(available.ndim == 2 and available.shape[1] > 0 and consistent.shape == available.shape and
            available.dtype == consistent.dtype == np.dtype(bool), 'Feature boolean shapes differ')
    require(type(n) is int and 0 <= n <= len(available), 'Invalid feature prefix')
    require(not np.any(consistent[:n] & ~available[:n]), 'Unavailable feature cannot be consistent')
    return dict(consistent=consistent[:n], available=available[:n])


def original_components(study, arrays, n):
    shape = np.asarray(arrays['baseline']).shape
    require(len(shape) == 2 and type(n) is int and 0 <= n <= shape[0], 'Invalid original prefix')
    specification = study.configs()[4]
    local_pre = np.asarray(study.modified_pre_score(specification, arrays, n))
    upstream = np.asarray(study.scope.modified_pre_score(study.source_spec(specification), arrays, n))
    increment = np.asarray(study.score_delta(specification, arrays, n))
    require(local_pre.shape == upstream.shape == shape and increment.shape == (n, shape[1]), 'Original component shapes')
    local_pre, upstream, increment = (np.asarray(x, float).copy() for x in (local_pre[:n], upstream[:n], increment))
    eligible = np.asarray(arrays['eligible'])
    require(eligible.shape == shape and eligible.dtype == np.dtype(bool), 'Original eligibility shape')
    require(np.isfinite(increment).all() and np.max(np.abs(increment), initial=0.) <= INCREMENT_LIMIT+1e-12,
            'Original local increment exceeds fixed bound')
    require(np.all(increment[~eligible[:n]] == 0.), 'Original ineligible local increment must be zero')
    require(not np.isinf(local_pre).any() and not np.isinf(upstream).any() and
            np.array_equal(np.isnan(local_pre), np.isnan(upstream)) and
            np.all(increment[np.isnan(upstream)] == 0.), 'Invalid original component values')
    rebuilt = upstream.copy(); changed = increment != 0.; rebuilt[changed] += increment[changed]
    np.testing.assert_array_equal(rebuilt, local_pre)
    return dict(pre_local67=local_pre, upstream_pre_score=upstream, local_increment=increment)


def routing_arrays(components, scores, audit):
    require(set(components) == {'pre_local67', 'upstream_pre_score', 'local_increment'} and
            set(scores) == set(NAMES[1:]), 'Routing component/arm keys')
    shape = components['pre_local67'].shape
    result = dict(components, **scores)
    for key in ROUTING_VECTORS:
        value = np.asarray(audit[key], float)
        require(value.shape == (shape[0],) and np.isfinite(value).all(), 'Routing budget vector shape')
        result[key] = value
    require(all(np.asarray(value).shape == shape for value in scores.values()), 'Routed score shapes')
    require(np.all((result['uniform_removed_fraction'] >= 0.) & (result['uniform_removed_fraction'] <= 1.)), 'Routing fraction bound')
    require(np.all(result['candidate_removed_abs_increment'] >= 0.) and
            np.all(result['control_removed_abs_increment'] >= 0.), 'Negative removal budget')
    np.testing.assert_allclose(result['candidate_removed_abs_increment'], result['control_removed_abs_increment'], atol=1e-12, rtol=0)
    return result


def decisions(study, arrays, meta, values, n):
    import monthly_trend_gate182_kernel as kernel
    small = feature_prefix(values, n)
    components = original_components(study, arrays, n)
    scores, audit = kernel.route_scores(components['pre_local67'], components['upstream_pre_score'],
        components['local_increment'], small['consistent'], small['available'])
    routing = routing_arrays(components, scores, audit)
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, old_scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, old_scores, meta)}
    for arm in NAMES[1:]:
        # Padding serves only the original adapter's shape contract; it reads the prefix.
        pre = np.zeros(np.asarray(arrays['baseline']).shape); pre[:n] = scores[arm]
        chosen, score = study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, routing, audit


def require_feature_audit(check, values):
    actual = feature_prefix(values, len(values['available']))
    for key in ('passed', 'independent_scalar_rebuild', 'exact_common_availability',
                'no_future_or_outcome_inputs', 'audit_checked', 'full_211_price_support_checked',
                'three_70_return_blocks_checked', 'strict_positive_consistency_checked'):
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, symbols_per_event=49, checked_cells=122*49,
                    available_cells=int(actual['available'].sum()), unavailable_cells=int((~actual['available']).sum()),
                    consistent_cells=int(actual['consistent'].sum()))
    require(actual['available'].shape == (122, 49), 'Independent feature shape')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value, 'Independent feature coverage: ' + key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0, 'No predictive model fits')
    estimate_counts(check)


def require_routing_audit(check, routing):
    require(set(routing) == ROUTING_KEYS, 'Saved routing fields differ')
    for key in ('passed', 'independent_scalar_rebuild', 'original_increment_identity_checked', 'exact_unchanged_cells_checked',
                'matched_absolute_removal_checked', 'no_cross_sectional_ranking', 'initial_scores_unchanged',
                'no_new_eligibility_filter', 'original_pre_reconstruction_exact', 'unchanged_cell_copy_exact',
                'matched_removed_budgets'):
        require(check.get(key) is True, 'Independent routing evidence missing '+key)
    for key in ('predictive_model_fits', 'episode_applications'):
        require(type(check.get(key)) is int and check[key] == 0, 'Routing must not fit or apply an episode')
    pre = routing['pre_local67']
    require(check.get('portfolio_returns_calculated') is False, 'Routing audit must not calculate portfolio returns')
    for key, expected in dict(events=len(pre), symbols_per_event=pre.shape[1], checked_cells=pre.size).items():
        require(type(check.get(key)) is int and check[key] == expected, 'Independent routing coverage: '+key)
    np.testing.assert_array_equal(check.get('uniform_removed_fraction'), routing['uniform_removed_fraction'])
    changed = {arm: int(np.count_nonzero(~((routing[arm] == pre) | (np.isnan(routing[arm]) & np.isnan(pre)))))
               for arm in NAMES[1:]}
    require(check.get('changed_cells') == changed, 'Independent routing coverage')


def require_outcome_audit(check):
    for key in ('passed', 'independent_accounts', 'independent_of_production_model_and_engine'):
        require(check.get(key) is True, 'Independent outcome audit missing ' + key)
    require(check.get('checked_paths') == 9 and check.get('checked_slices') == 45 and
            check.get('terminal_liquidation') is False, 'Independent outcome coverage')
    require(check.get('accounting_source_sha256') == sha(ROOT / 'work/ptr169_results_independent.py'), 'Accounting source changed')
    require(set(check.get('paths', {})) == set(NAMES), 'Missing independent arms')
    for paths in check['paths'].values():
        require(set(paths) == {p[0] for p in ev.PATHS}, 'Missing independent paths')
        for path in paths.values():
            require(path.get('passed') is True and path.get('monthly_executions') == 122 and
                    set(path.get('slices', {})) == {'full122', 'first88', 'early53', 'recent35', 'later34'}, 'Incomplete path audit')


def prepare(expected):
    plan = gate(expected); require(not (OUT / 'prepared').exists(), 'Preparation is write-once')
    import monthly_trend_gate182_kernel as kernel
    import monthly_trend_gate182_independent as independent
    study = local(); meta, arrays, px, op = study.load()
    require(len(meta['dates']) == 122 and arrays['baseline'].shape == (122, 49), 'Original122x49 required')
    require(len(arrays) == 27, 'Original27 arrays required')
    require(all(len(s) == 49 for s in meta['symbols']), 'Original monthly candidate widths changed')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    source, info = study.arch.old.panel_load(meta['etfs'])
    raw_hashes = study.raw_source_hashes(info)
    require(raw_hashes == plan['raw_source_hashes'], 'Original raw source lineage changed')
    verify(raw_hashes)
    parity = study.verify_panel_parity(source, px, op)
    panel = feature_inputs(source)
    calendar = read(CALENDAR)['sessions']
    values, audit = kernel.build_features(panel, meta, calendar)
    check = independent.verify_features(panel, meta, calendar, values, audit)
    require_feature_audit(check, values)
    require(check['predictive_model_fits'] == audit['predictive_model_fits'] == 0,
            'Independent model accounting differs')
    counts = estimate_counts(check)
    require(counts == estimate_counts(audit), 'Independent descriptive statistics accounting differs')
    orders, routing, routing_detail = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    route_check = independent.verify_routing(routing['pre_local67'], routing['upstream_pre_score'],
        routing['local_increment'], values['consistent'], values['available'],
        {arm: routing[arm] for arm in NAMES[1:]}, routing_detail)
    require_routing_audit(route_check, routing)
    for n in FEATURE_PREFIXES:
        small, _ = kernel.build_features(panel, meta, calendar, limit=n)
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
    for n in range(123):
        oo, rr, detail = decisions(study, arrays, meta, values, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in ROUTING_KEYS: np.testing.assert_array_equal(rr[k], routing[k][:n])
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'ROUTING.npz', **routing)
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, feature_panel='original source adj', execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, routing=route_check, routing_detail=routing_detail, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27,
        original_components_direct=True, original_ineligible_increment_zero=True, original_episode_once_per_arm=True, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, **counts))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays), **counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, values, routing, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_routing_audit(audit['routing'], routing)
    for key in ('original_components_direct', 'original_ineligible_increment_zero', 'original_episode_once_per_arm'):
        require(audit.get(key) is True, 'Missing component/episode invariant '+key)
    detail = audit['routing_detail']
    for key in ROUTING_VECTORS: np.testing.assert_array_equal(routing[key], np.asarray(detail[key], float))
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and audit.get('original_orders_exact') is True and
            audit.get('original_arrays_unchanged') is True and audit.get('original_array_count') == 27 and audit.get('new_model_fits') == 0 and
            audit.get('predictive_model_fits') == 0 and audit.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING,
            'Prepared invariants not verified')
    for record in (audit, c):
        require(type(record.get('new_predictive_model_fits')) is int and record['new_predictive_model_fits'] == 0,
                'New predictive model fits not verified')
        expected_counts = estimate_counts(audit['feature'])
        require(all(type(record.get(key)) is int and record[key] == value
                    for key, value in expected_counts.items()), 'Descriptive estimate receipt accounting differs')
    require(c.get('new_returns_computed') is False and c.get('new_model_fits') == 0 and
            c.get('predictive_model_fits') == 0 and c.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING and
            c.get('original_array_count') == 27, 'Prepared model/array accounting changed')


def prepared(expected):
    gate(expected); dest = OUT / 'prepared'; c = read(dest / 'COMPLETION.json')
    require(c['status'] == 'COMPLETE' and c['plan_sha256'] == expected, 'Preparation incomplete')
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    for name, digest in c['artifacts_sha256'].items(): require(sha(dest / name) == digest, 'Changed prepared artifact ' + name)
    with np.load(dest / 'FEATURES.npz') as data: values = {k: data[k] for k in data.files}
    with np.load(dest / 'ROUTING.npz') as data: routing = {k: data[k] for k in data.files}
    evidence = read(dest / 'INDEPENDENT_PREPARE.json')
    verify_prepared_evidence(c, values, routing, evidence)
    import monthly_trend_gate182_independent as independent
    check = independent.verify_routing(routing['pre_local67'], routing['upstream_pre_score'],
        routing['local_increment'], values['consistent'], values['available'],
        {arm: routing[arm] for arm in NAMES[1:]}, evidence['routing_detail'])
    require_routing_audit(check, routing)
    return read(dest / 'META.json'), read(dest / 'ORDERS.json')


def train_metrics(engine, eq, meta):
    answer = {}
    for name, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
        part = eq.loc[meta['dates'][lo]:meta['ends'][hi]]
        answer[name] = engine.stats(part / part.iloc[0] * 100000.)
    return answer


def guard(candidate, reference, gain):
    return (candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12 and
        candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05 and
        candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03 and
        all(candidate[e]['cagr'] >= reference[e]['cagr'] - .025 for e in ('early', 'recent')))


def select_first88(metrics, orders):
    changed = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[0]][i]['symbols'])]
    guards = dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], .005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], 0.),
        close_vs_uniform_shrink=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_uniform_shrink=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
        changed_support=len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed))
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, first88_only=True, later34_used=False)


def select(expected):
    meta, orders = prepared(expected); require(not (OUT / 'selected').exists(), 'Selection is write-once')
    px, op = ev.prices(); engine = ev.engine(); metrics = {}
    for arm in NAMES:
        metrics[arm] = {}
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[2]):
            _, eq, _, _ = engine.replay(px, op, orders[arm][:TRAIN], cost=cost, execution=mode,
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date()) == meta['ends'][87], 'Selection escaped first88 cutoff')
            metrics[arm][path] = train_metrics(engine, eq, meta)
    result = select_first88(metrics, orders); gate(expected)
    dest = OUT / 'selected'; dest.mkdir(); dump(dest / 'TRAIN_RESULTS.json', metrics)
    dump(dest / 'FROZEN.json', dict(created_utc=now(), plan_sha256=expected,
        prepared_sha256=sha(OUT / 'prepared/COMPLETION.json'), orders_sha256=sha(OUT / 'prepared/ORDERS.json'),
        training_sha256=sha(dest / 'TRAIN_RESULTS.json'), selection=result))
    print(json.dumps(result), flush=True)


def frozen_selection(expected, orders):
    frozen = read(OUT / 'selected/FROZEN.json')
    require(frozen['plan_sha256'] == expected and frozen['orders_sha256'] == sha(OUT / 'prepared/ORDERS.json') and
        frozen['training_sha256'] == sha(OUT / 'selected/TRAIN_RESULTS.json') and
        frozen['prepared_sha256'] == sha(OUT / 'prepared/COMPLETION.json'), 'Frozen selection dependencies changed')
    require(select_first88(read(OUT / 'selected/TRAIN_RESULTS.json'), orders) == frozen['selection'], 'Selection mismatch')
    return frozen


def verify_training_against_audit(training, audit):
    import monthly_trend_gate182_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Training paths differ')
        for path, slices in training[arm].items():
            require(set(slices) == {'training', 'early', 'recent'}, 'Training slices differ')
            for name, full_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                helper.check_metrics(audit['paths'][arm][path]['slices'][full_name], slices[name],
                                     'Frozen training/' + arm + '/' + path + '/' + name)


def diagnose(expected):
    meta, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(not (OUT / 'results').exists(), 'Diagnostics are write-once')
    px, op = ev.prices(); engine = ev.engine(); dest = OUT / 'results'; dest.mkdir(); results, exports = {}, {}
    for arm in NAMES:
        results[arm], exports[arm] = {}, {}
        for path, mode, cost in ev.PATHS:
            result, eq, trades, holds, positions = ev.run_path(engine, px, op, orders[arm], meta, mode, cost)
            result['slices'] = dict(full122=result['metrics'], **ev.split_stats(engine, eq, meta))
            results[arm][path] = result
            data = dict(DAILY=[dict(date=str(d.date()), equity=float(v)) for d, v in eq.items()],
                TRADES=trades, HOLDINGS=holds, DAILY_POSITIONS=positions)
            exports[arm][path] = data
            for kind, value in data.items(): dump(dest / (arm + '_' + path + '_' + kind + '.json'), value)
    import monthly_trend_gate182_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, routing_only_existing_local_increment=True, matched_budget_is_theoretical=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=9, slices=45,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 9 and completion['slices'] == 45, 'Incomplete diagnostics')
    for name, digest in completion['artifacts_sha256'].items(): require(sha(OUT / 'results' / name) == digest, 'Changed result ' + name)
    require_outcome_audit(read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    _, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(completion['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection receipt changed')
    results = read(OUT / 'results/RESULTS.json')
    require(results['selection'] == frozen['selection'], 'Closure selection mismatch')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    protected = read(PROTECTED); verify(protected); require(len(protected) == 21444, 'Incomplete protection')
    require(not (OUT / 'COMPLETION.json').exists(), 'Closure is write-once')
    best = state['best_training_selected_candidate']; pick = results['selection']['primary']; arm = results['arms'][pick]
    metrics = arm['same_close_zero']['metrics']
    if results['selection']['admitted'] and metrics['cagr'] > best['cagr']:
        best = dict(directory=OUT.name, spec=dict(arm=pick), cagr=metrics['cagr'], sharpe=metrics['sharpe'],
            max_drawdown=metrics['max_drawdown'], next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],
            selected_on_first88=True, observer_promotion=False, limitations=contract()['limitation'])
    receipt = dict(status='COMPLETED_INDEPENDENTLY_VERIFIED', created_utc=now(), plan_sha256=expected,
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False,
        **{key: read(OUT / 'prepared/COMPLETION.json')[key] for key in ESTIMATE_FIELDS})
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review fixed local-increment routing before a distinct new hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 180 and sum(r['trials'] for r in state['completed']) == 1055, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['test', 'register', 'prepare', 'select', 'diagnose', 'close'])
    parser.add_argument('--plan-sha256'); args = parser.parse_args()
    if args.phase == 'test': tests()
    elif args.phase == 'register': register()
    else:
        require(bool(args.plan_sha256), 'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
