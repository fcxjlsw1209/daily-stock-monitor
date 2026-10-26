"""Round180: one smooth or uniform replacement of the frozen episode selector.

Actual scores and outcomes require a prospectively bound PLAN. This changes no
raw feature, model, source eligibility or underlying Local67 support/core array.
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
OUT = ROOT / 'outputs/monthly_episode_neighborhood180'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_downside_mix179'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'score_neighborhood_episode', 'uniform_episode')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, WIDTH, MAX_BUDGET = 88, .02, .05
SELECTOR_PREFIXES = (0, 1, 53, 88, 122)
MODEL_FIT_ACCOUNTING = 'No new predictive ML fit, raw-price feature, or descriptive statistical estimation; reuse frozen Local67 pre-scores and episode effects, changing only their deterministic selection operator'
PRIOR_EXPECTED = {
    'PLAN.json': '84f567e96ded9840cd3972b2329edf131191cf75c054ee20863ea39ff96a2c5c',
    'COMPLETION.json': '8eef60eac90450d90476bae5053f4e6627738fb6cd7cf68497bc42a688fb7938',
    'results/COMPLETION.json': '621164ba991123a57846cf4b6997d81120b378a344908b9bc5d047ee8a65f089',
}
CONTEXT = ROOT / 'work/monthly_episode_neighborhood180_context.json'
CONTEXT_SHA = '9c1b7b96eadab99ccfcde1135ee7955f2355c69c4202d0875a5d9e04fab508d4'
TEST_RECEIPT = ROOT / 'work/monthly_episode_neighborhood180_TESTS.json'
PREPARED_FILES = {'SELECTION.npz', 'CHOICES.json', 'EPISODE_AUDIT.json',
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
    return module('monthly_episode_neighborhood180_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        prospective_context_sha256=CONTEXT_SHA,
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='A smooth score-distance taper of the existing episode correction may improve near-tie selection compared with hard top2 locking and a rank3-5 boundary.',
        family='Deterministic replacement of one frozen episode operator; no new raw feature, model, statistical estimate or parameter search.',
        measure='Start exact Local67 modified_pre_score. Original support=(baseline-core)/(1-core_weight). B=min(0.05,max(0,0.10-0.025*state_calm10-0.05*state_weak_narrow)). With at least3 eligible names, q is the third-highest pre-score; uniform correction=B*(clip(support+effect,-0.5,0.5)-support); candidate correction=uniform*exp(-0.5*((pre-q)/0.02)^2).',
        numerical_policy='Frozen original eligibility is unchanged. Eligible pre and support must be finite; invalid eligible values raise rather than deleting a name. NaN effects give zero correction without excluding the name; infinite effects raise. Core weights broadcast from (n,), (n,1), or (n,width), must be finite0<w<1; source pins and original parity preserve actual weights. The host also verifies actual eligible support in[-0.5,0.5] and absolute corrections<=0.05 with tolerance1e-12, without clipping support. Ineligible scores are untouched. Arithmetic nonfinite intermediates fail explicitly except a Gaussian tail may underflow to zero.',
        availability='The same original eligible universe is used by both new arms; no effect-availability or score-distance filter changes eligibility. There is no new raw-data availability mask.',
        scoring='Exactly one episode operator per arm. Incumbent retains its original hard selector. New arms replace it and go directly to make_orders; do not call the old adapter/episode afterwards. All eligible names may change score, including top2 and ranks beyond5. Width0.02 and maximum budget0.05 are fixed inherited policy scales; no width, shape or dose search. First partial row corrections are zero; its new scores equal unmodified pre, without assuming incumbent chooses identical names.',
        portfolio='Global descending corrected score then ascending symbol; at least3 eligible selects3, exactly2 selects both, fewer than2 cash. With fewer than3 eligible both corrections are zero. Unchanged27 original arrays, core, source eligibility, forecast/support, proxy Top32+theme17 pools, equal weights, fixed adjusted units intramonth, difference rebalance, and no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_uniform_episode=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, selector_prefixes=list(SELECTOR_PREFIXES), all123_order_prefixes=True),
        predecessor='Round179 completed without a new execution repair. Inherit its complete source/protection hash chain including177 repair evidence and sources, and bind179 PLAN and root/result completions. Preserve historical central amendment references; no new repair adapter is used.',
        novelty='The tested distinction is continuous score-neighborhood weighting of an existing episode correction. Unlocking top2 or globally selecting the trio is not novel: formation142 and later additive studies overlap. Readable boundary funding/preserve code retained hard cohorts. Missing original episode_retention70 and episode_confidence70 source prevents an exhaustive novelty claim.',
        limitation='Repeatedly reused development sample, estimated historical membership and retrospectively retrieved prices; not pristine OOS/PIT or author reproduction. Smooth score corrections do not make discrete top3 holdings continuous. Uniform control isolates the taper relative to the same full eligible correction. Prior single-event observations motivate a mechanism, not a target security/date or forced trade.',
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0,
        new_descriptive_estimates=0, new_raw_features=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9,
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_episode_neighborhood180_host.py',
        'test_monthly_episode_neighborhood180_host.py', 'monthly_episode_neighborhood180_kernel.py',
        'test_monthly_episode_neighborhood180_kernel.py', 'monthly_episode_neighborhood180_independent.py',
        'test_monthly_episode_neighborhood180_independent.py', 'ptr169_rebuild_evaluate.py',
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
    """Bind completed179 and preserve inherited evidence without requiring a new repair."""
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


def state_check(state):
    require(len(state['completed']) == 177 and sum(r['trials'] for r in state['completed']) == 1046,
            'Expected177 completed rounds /1046 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round179 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_EPISODE_NEIGHBORHOOD180_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_episode_neighborhood180_host.py', 'test_monthly_episode_neighborhood180_kernel.py',
             'test_monthly_episode_neighborhood180_independent.py')
    modules = [module('test_episode180_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    plan = dict(schema='monthly_episode_neighborhood180_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_selector_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_new_raw_features_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_episode_selection',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed score-neighborhood and uniform episode replacements; select only on first88 before later34 diagnostics.')
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


def result_arrays(result):
    require(type(result) is dict and set(result) == {'scores', 'corrections', 'choices', 'eligible'}, 'Selector result keys')
    eligible = np.asarray(result['eligible'])
    require(eligible.ndim == 2 and eligible.shape[1] > 0 and eligible.dtype == np.dtype(bool), 'Selector eligibility shape/type')
    n, width = eligible.shape
    packed = {'eligible': eligible}
    for field in ('scores', 'corrections', 'choices'):
        require(type(result[field]) is dict and set(result[field]) == {'candidate', 'control'}, 'Selector paired arm keys')
    for key in RAW.values():
        score = np.asarray(result['scores'][key])
        delta = np.asarray(result['corrections'][key])
        require(score.shape == delta.shape == eligible.shape and score.dtype.kind == delta.dtype.kind == 'f', 'Selector numeric shape/type')
        require(np.isfinite(score[eligible]).all() and np.isfinite(delta).all(), 'Invalid selector numerics')
        require(np.all(delta[~eligible] == 0.) and (not n or np.all(delta[0] == 0.)), 'Noneligible/initial correction')
        choices = result['choices'][key]
        require(type(choices) is list and len(choices) == n, 'Selector choices rows')
        for i, ids in enumerate(choices):
            count = int(eligible[i].sum())
            require(type(ids) is list and len(ids) == (min(3, count) if count >= 2 else 0), 'Selector cardinality')
            require(all(type(j) is int and 0 <= j < width and eligible[i, j] for j in ids) and
                    len(ids) == len(set(ids)), 'Selector choice index')
        packed['score_' + key], packed['correction_' + key] = score, delta
    return packed


def prefix_result(result, n):
    shape = np.asarray(result['eligible']).shape
    require(len(shape) == 2 and type(n) is int and 0 <= n <= shape[0], 'Invalid selector prefix')
    small = {field: {key: np.asarray(value[:n], float) for key, value in result[field].items()}
             for field in ('scores', 'corrections')}
    small['choices'] = {key: rows[:n] for key, rows in result['choices'].items()}
    small['eligible'] = result['eligible'][:n]
    result_arrays(small)
    return small


def verify_frozen_support(pre, arrays, result, n):
    """Real source support is bounded; never clip or silently shrink its universe."""
    result_arrays(result)
    full_shape = np.asarray(arrays['baseline']).shape
    require(len(full_shape) == 2 and type(n) is int and 0 <= n <= full_shape[0], 'Invalid source prefix')
    require(result['eligible'].shape == (n, full_shape[1]), 'Result/source prefix mismatch')
    eligible = np.asarray(arrays['eligible'])
    require(eligible.shape == full_shape and eligible.dtype == np.dtype(bool) and
            np.array_equal(result['eligible'], eligible[:n]), 'Original eligibility changed')
    matrix = {}
    for name in ('baseline', 'core'):
        value = np.asarray(arrays[name])
        require(value.shape == full_shape, 'Original matrix shape')
        matrix[name] = np.asarray(value[:n], float)
    weights = np.asarray(arrays['core_weight'])
    if weights.shape == (full_shape[0],): weights = weights[:, None]
    require(weights.shape in ((full_shape[0], 1), full_shape), 'Original core weight shape')
    weights = np.broadcast_to(np.asarray(weights[:n], float), (n, full_shape[1]))
    require(np.isfinite(weights).all() and np.all((weights > 0.) & (weights < 1.)), 'Original core weights invalid')
    use = eligible[:n]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support = (matrix['baseline'][use] - matrix['core'][use]) / (1. - weights[use])
    require(np.isfinite(support).all() and np.all(np.abs(support) <= .5 + 1e-12), 'Frozen eligible support outside original bound')
    original_pre = np.asarray(pre)
    require(original_pre.shape == full_shape, 'Original pre-score shape')
    original_pre = np.asarray(original_pre[:n], float)
    require(np.isfinite(original_pre[use]).all(), 'Original eligible pre-score invalid')
    for key in RAW.values():
        correction = result['corrections'][key]
        require(np.max(np.abs(correction), initial=0.) <= MAX_BUDGET + 1e-12, 'Original episode correction budget exceeded')
        if n:
            np.testing.assert_array_equal(result['scores'][key][0], original_pre[0])
        np.testing.assert_array_equal(result['scores'][key][~use], original_pre[~use])
    return dict(eligible_support_bound_checked=True, correction_budget_checked=True,
                initial_pre_score_unchanged=True, original_eligibility_exact=True,
                support_abs_limit=.5, correction_abs_limit=MAX_BUDGET, tolerance=1e-12)


def decisions(study, arrays, meta, result, n):
    small = prefix_result(result, n)
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    for arm, key in RAW.items():
        # The new selector already applied its only episode operation.
        orders[arm] = study.rp.make_orders(small['choices'][key], small['scores'][key], meta)
    return orders


def require_selection_audit(check, result):
    result_arrays(result)
    for key in ('passed', 'independent_scalar_rebuild', 'original_support_checked',
                'clipped_episode_budget_checked', 'gaussian_neighborhood_checked', 'global_top3_checked',
                'first_partial_neutral_checked', 'exact_cardinality_checked', 'no_future_or_outcome_inputs',
                'audit_checked', 'eligibility_unchanged'):
        require(check.get(key) is True, 'Independent selector evidence missing ' + key)
    n, width = result['eligible'].shape
    expected = dict(events=n, symbols_per_event=width, checked_cells=n*width,
                    eligible_cells=int(result['eligible'].sum()))
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value, 'Independent selector coverage: ' + key)
    require(type(check.get('episode_applications')) is int and check['episode_applications'] == 1, 'One episode application required')
    require(check.get('computes_returns') is False, 'Selector audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0, 'No predictive fits')
    require(check.get('correction_cells') == {key: int(np.count_nonzero(value))
            for key, value in result['corrections'].items()}, 'Independent correction coverage')


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
    import monthly_episode_neighborhood180_kernel as kernel
    import monthly_episode_neighborhood180_independent as independent
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
    pre = study.modified_pre_score(study.configs()[4], arrays, 122)
    pre_before = np.array(pre, copy=True)
    result, audit = kernel.apply_episode(pre, arrays, meta)
    support_check = verify_frozen_support(pre, arrays, result, 122)
    check = independent.verify_selection(pre, arrays, meta, result, audit)
    require_selection_audit(check, result)
    orders = decisions(study, arrays, meta, result, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    # Prefix checks reconstruct both the frozen pre-score and the new operator.
    for n in range(123):
        small_pre = study.modified_pre_score(study.configs()[4], arrays, n)
        small, small_audit = kernel.apply_episode(small_pre, arrays, meta, limit=n)
        verify_frozen_support(small_pre, arrays, small, n)
        expected_result = prefix_result(result, n)
        for key, value in result_arrays(small).items():
            np.testing.assert_array_equal(value, result_arrays(expected_result)[key])
        require(small['choices'] == expected_result['choices'], 'Selector choices prefix failed')
        oo = decisions(study, arrays, meta, small, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        if n in SELECTOR_PREFIXES:
            require_selection_audit(independent.verify_selection(small_pre, arrays, meta, small, small_audit, limit=n), small)
    np.testing.assert_array_equal(pre, pre_before)
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'SELECTION.npz', **result_arrays(result))
    dump(dest / 'CHOICES.json', result['choices'])
    dump(dest / 'EPISODE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, execution_prices=parity,
        selector_source='exact frozen Local67 modified_pre_score and original27 arrays; no new raw-price feature'))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(selection=check, frozen_support=support_check, prefixes=123, selector_prefixes=list(SELECTOR_PREFIXES),
        original_orders_exact=True, original_pre_score_exact=True, original_arrays_unchanged=True, original_array_count=27,
        new_arms_direct_make_orders=True, original_episode_not_reapplied=True,
        new_model_fits=0, predictive_model_fits=0, new_predictive_model_fits=0,
        new_raw_features=0, new_descriptive_estimates=0, model_fit_accounting=MODEL_FIT_ACCOUNTING))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0, new_raw_features=0, new_descriptive_estimates=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays)))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, result, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_selection_audit(audit['selection'], result)
    require(result['eligible'].shape == (122, 49), 'Original full selector shape')
    support = audit.get('frozen_support', {})
    for key in ('eligible_support_bound_checked', 'correction_budget_checked', 'initial_pre_score_unchanged', 'original_eligibility_exact'):
        require(support.get(key) is True, 'Missing frozen support evidence ' + key)
    require(support.get('support_abs_limit') == .5 and support.get('correction_abs_limit') == MAX_BUDGET and
            support.get('tolerance') == 1e-12, 'Frozen support limits changed')
    require(audit.get('prefixes') == 123 and audit.get('selector_prefixes') == list(SELECTOR_PREFIXES), 'Missing prefix evidence')
    for key in ('original_orders_exact', 'original_pre_score_exact', 'original_arrays_unchanged',
                'new_arms_direct_make_orders', 'original_episode_not_reapplied'):
        require(audit.get(key) is True, 'Missing prepared invariant ' + key)
    for record in (audit, c):
        require(record.get('original_array_count') == 27 and record.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING, 'Array/model accounting changed')
        for key in ('new_model_fits', 'predictive_model_fits', 'new_predictive_model_fits', 'new_raw_features', 'new_descriptive_estimates'):
            require(type(record.get(key)) is int and record[key] == 0, 'Unexpected estimation: ' + key)
    require(c.get('new_returns_computed') is False, 'Preparation computed portfolio returns')


def prepared(expected):
    gate(expected); dest = OUT / 'prepared'; c = read(dest / 'COMPLETION.json')
    require(c['status'] == 'COMPLETE' and c['plan_sha256'] == expected, 'Preparation incomplete')
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    for name, digest in c['artifacts_sha256'].items(): require(sha(dest / name) == digest, 'Changed prepared artifact ' + name)
    with np.load(dest / 'SELECTION.npz') as data:
        require(set(data.files) == {'eligible', 'score_candidate', 'score_control', 'correction_candidate', 'correction_control'}, 'Saved selector arrays')
        result = dict(eligible=data['eligible'], scores={k: data['score_' + k] for k in RAW.values()},
                      corrections={k: data['correction_' + k] for k in RAW.values()}, choices=read(dest / 'CHOICES.json'))
    verify_prepared_evidence(c, result, read(dest / 'INDEPENDENT_PREPARE.json'))
    meta, orders = read(dest / 'META.json'), read(dest / 'ORDERS.json')
    study = local()
    for arm, key in RAW.items():
        require(orders[arm] == study.rp.make_orders(result['choices'][key], result['scores'][key], meta), 'Saved orders differ from verified selector')
    return meta, orders


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
        close_vs_uniform_episode=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_uniform_episode=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_episode_neighborhood180_independent as independent
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
    import monthly_episode_neighborhood180_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, smooth_correction_not_continuous_holdings=True, original_episode_replaced_once=True, observer_promotion=False))
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
        new_raw_features=0, new_descriptive_estimates=0, original_array_count=27,
        protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False)
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review fixed episode-operator results before a distinct new hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 178 and sum(r['trials'] for r in state['completed']) == 1049, 'Wrong final counters')
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
