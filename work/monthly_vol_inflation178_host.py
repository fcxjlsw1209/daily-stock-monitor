"""Round178: frozen Local67 plus bounded formation volatility-inflation penalties.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests fit no predictive model. Historical sample volatilities are separately
counted descriptive statistics, not new predictive ML fits.
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
OUT = ROOT / 'outputs/monthly_vol_inflation178'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_breakpoint177'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'formation_vol_inflation', 'current_vol_inflation')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE = 88, .01
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('volatility_estimates',)
MODEL_FIT_ACCOUNTING = 'new_model_fits and new_predictive_model_fits count only new predictive ML/Ridge/GBDT fits; volatility_estimates counts three sample standard deviations per complete valid-price cell, including floor failures, in the returned full feature audit; prefix-validation repeats are excluded'
PRIOR_EXPECTED = {
    'PLAN.json': 'cef087df4824c24438d16852fc8bfb079295568ea347b14e126b4e2c8fe1b5fd',
    'COMPLETION.json': '8ad8c25f05a80636955ae1356f6fd238d3ccc23b45373508cd57230b982a3617',
    'AMENDMENT_01.json': '2744c6f646294814e3bf33611ea28920300a6e2b3d30d544edd7e24a7fb55f98',
    'REPAIR_COMPLETION_01.json': '7da696e15193e1d5f9d4fdbb499e2da7363a19b9655723713e642a7ae5fd8d2c',
}
TEST_RECEIPT = ROOT / 'work/monthly_vol_inflation178_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
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
    return module('monthly_vol_inflation178_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='A bounded penalty for current volatility exceeding the volatility of the momentum formation period may reduce stale risk-normalization in Local67; compare an otherwise identical current252-volatility reference.',
        family='Descriptive simple-return sample volatility inflation; no predictive ML fit or parameter search.',
        measure='At signal t use all295 adjusted prices t-294..t. sigma_old252 uses253 prices t-294..t-42; sigma_current63 uses64 prices t-63..t; sigma_current252 uses253 prices t-252..t. All returns are simple adjacent price returns, all standard deviations use ddof=1. Candidate=-tanh(max(0,log(sigma_current63)-log(sigma_old252))); control=-tanh(max(0,log(sigma_current63)-log(sigma_current252))).',
        source_window=dict(price_count=295, earliest_offset=294, latest_offset=0,
            old_return_count=252, old_latest_offset=42, current_return_count=63,
            control_return_count=252, sample_sd_floor=1e-12, ddof=1),
        numerical_policy='All295 prices must be finite and strictly positive; every standard deviation must be finite and exceed1e-12. Missing or invalid prices and floor failures make both arms unavailable. Arithmetic overflow/nonfinite calculations raise. Use log differences to avoid ratio overflow. Valid zero is retained; unavailable raw is NaN. No filling, shortened history or fallback.',
        availability='Identical complete295-price and three-volatility support for both arms; all49 stock/ETF candidates are measured without an eligibility or sector filter.',
        scoring='Add0.01*raw directly to Local67 pre-score then exactly one original episode. Raw lies in[-1,0]; no second tanh or cross-sectional ranking. Unavailable delta0 and first partial2016-08-17 allocation delta0; maximum penalty magnitude0.01. Original baseline/core/support metadata are unchanged.',
        portfolio='Unchanged27 original arrays, long-momentum main score and skip42 core; original historical proxy Top32+theme17 and eligibility; long-only EW3/two halves/under2 cash; fixed adjusted units intramonth; difference rebalance; no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_current_volatility=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES), all123_order_prefixes=True),
        predecessor='Original177 PLAN and completion plus its separately registered audit-shape AMENDMENT_01 and REPAIR_COMPLETION_01, referenced receipts and repair sources are verified and pinned. The repair belongs to177 and adds no new strategy trial;178 does not inherit its half-core replacement or shape adapter.',
        novelty='Prior volatility-window alignment and negative std21/std126 were tested. This exact bounded negative formation/current volatility ratio was not found in readable prior implementations. Some older source was unavailable, so this is not proof that an equivalent hypothesis has never been tested.',
        limitation='Repeatedly reused development sample, estimated historical membership and retrospectively retrieved vendor prices; not pristine OOS/PIT. Current std63 already penalizes the momentum core, so this may duplicate an existing risk response. The current252 control overlaps and smooths current63; only the matched comparison isolates the reference window under this fixed correction.',
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9,
        descriptive_statistics='volatility_estimates is three per complete valid295-price cell, including standard-deviation floor failures; missing or invalid-price cells contribute zero. Count only the returned events, excluding repeated prefix validation calls.',
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_vol_inflation178_host.py',
        'test_monthly_vol_inflation178_host.py', 'monthly_vol_inflation178_kernel.py',
        'test_monthly_vol_inflation178_kernel.py', 'monthly_vol_inflation178_independent.py',
        'test_monthly_vol_inflation178_independent.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')]


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


def predecessor_binding(state):
    """Verify177's closed audit repair without executing any predecessor phase."""
    pins = {}
    for name, digest in PRIOR_EXPECTED.items():
        bind(pins, PRIOR / name, digest)

    def ref(value, expected):
        require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Predecessor reference schema')
        path = Path(value['path'])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path == expected.resolve() and path.is_relative_to(ROOT), 'Predecessor reference path')
        bind(pins, path, value['sha256'])
        return read(path) if path.suffix == '.json' else None

    plan_sha = PRIOR_EXPECTED['PLAN.json']
    closed = ref(state['completed'][-1]['completion'], PRIOR / 'COMPLETION.json')
    require(closed.get('status') == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed.get('plan_sha256') == plan_sha and closed.get('protected_files') == 21444 and
            closed.get('observers_modified') is False, 'Predecessor closure incomplete')
    bind(pins, PRIOR / 'results/COMPLETION.json', closed['result_completion_sha256'])
    result = read(PRIOR / 'results/COMPLETION.json')
    require(result.get('status') == 'COMPLETE' and result.get('plan_sha256') == plan_sha,
            'Predecessor result completion differs')
    amendment = ref(state.get('active_execution_amendment'), PRIOR / 'AMENDMENT_01.json')
    repaired = ref(state.get('latest_execution_repair_receipt'), PRIOR / 'REPAIR_COMPLETION_01.json')
    for doc in (amendment, repaired):
        require(doc.get('round') == 177 and doc.get('same_experiment') is True and
                type(doc.get('additional_strategy_trials')) is int and doc['additional_strategy_trials'] == 0,
                'Predecessor repair must be the same experiment')
    require(amendment.get('original_plan_sha256') == plan_sha and
            amendment.get('kind') == 'independent_core_weight_representation' and
            all(amendment.get(key) is False for key in ('formula_changed', 'parameters_changed',
                'input_data_changed', 'original_arrays_changed', 'source_eligibility_changed')),
            'Predecessor repair changed strategy semantics')
    ref(repaired.get('original_completion'), PRIOR / 'COMPLETION.json')
    ref(repaired.get('amendment'), PRIOR / 'AMENDMENT_01.json')
    failure = ref(amendment.get('failure_record'), PRIOR / 'PREPARE_INTERFACE_FAILURE_01.json')
    tests = ref(amendment.get('tests'), PRIOR / 'REPAIR_TESTS_01.json')
    review = ref(amendment.get('independent_prereview'), PRIOR / 'REPAIR_PREREVIEW_01.json')
    require(repaired.get('failure_record') == amendment['failure_record'] and
            repaired.get('independent_prereview') == amendment['independent_prereview'], 'Repair closure evidence differs')
    ref(amendment.get('previous_active_execution_amendment'), ROOT / 'outputs/monthly_annual_lag158/AMENDMENT_01.json')
    ref(failure.get('excerpt'), PRIOR / 'PREPARE_INTERFACE_FAILURE_01_EXCERPT.txt')
    require(failure.get('artifact_kind') == 'retained_execution_context_excerpt' and
            failure.get('source') == 'tool_output_excerpt_not_full_log' and
            failure.get('full_raw_log_available') is False and
            failure.get('strategy_returns_calculated') == 0 and failure.get('prepared_artifacts_published') == 0,
            'Predecessor failure provenance differs')
    require(tests.get('passed') is True and tests.get('synthetic_only') is True and
            all(review.get(key) is True for key in ('passed', 'independent_review',
                'semantics_preserved', 'original_frozen_files_unchanged')), 'Repair review or tests incomplete')
    require(review.get('tests') == amendment['tests'] and review.get('failure_record') == amendment['failure_record'],
            'Repair prereview evidence differs')
    ref(review.get('reviewer_script'), ROOT / 'work/breakpoint177_repair_prereview.py')
    implementation = amendment.get('implementation_hashes')
    expected_files = {'work/monthly_breakpoint177_repaired_runner.py', 'work/test_monthly_breakpoint177_repaired_runner.py',
                      'work/breakpoint177_weight_repair.py', 'work/test_breakpoint177_weight_repair.py'}
    require(type(implementation) is dict and set(implementation) == expected_files and
            tests.get('implementation_hashes') == implementation == review.get('implementation_hashes'),
            'Predecessor repair implementation differs')
    for name, digest in implementation.items(): bind(pins, ROOT / name, digest)
    prepared = ref(repaired.get('independent_prepare'), PRIOR / 'prepared/INDEPENDENT_PREPARE.json')
    outcome = ref(repaired.get('independent_outcome'), PRIOR / 'results/INDEPENDENT_OUTCOME.json')
    delta = prepared.get('delta', {})
    require(delta.get('passed') is True and delta.get('original_plan_sha256') == plan_sha and
            delta.get('execution_amendment') == repaired['amendment'] and
            delta.get('repair_source_hashes') == implementation and
            delta.get('repair_prereview') == amendment['independent_prereview'] and
            delta.get('weight_shape_adapter', {}).get('passed') is True,
            'Predecessor saved repair proof missing')
    require(outcome.get('passed') is True and outcome.get('checked_paths') == 9 and
            outcome.get('checked_slices') == 45, 'Predecessor independent outcomes incomplete')
    return dict(directory=PRIOR.name, input_sha256=pins, same_experiment_repair=True,
                repair_additional_trials=0)


def feature_inputs(source):
    require(isinstance(source.get('adj'), pd.DataFrame), 'Original adjusted price panel required')
    return {'adj': source['adj']}


def estimate_counts(check):
    count = check.get('volatility_estimates')
    available = check.get('available_cells')
    require(type(count) is int and type(available) is int and 0 <= available <= 122*49 and
            3*available <= count <= 3*122*49 and count % 3 == 0, 'Descriptive volatility accounting differs')
    return dict(volatility_estimates=count)


def state_check(state):
    require(len(state['completed']) == 175 and sum(r['trials'] for r in state['completed']) == 1040,
            'Expected175 completed rounds /1040 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round177 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_VOL_INFLATION178_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_vol_inflation178_host.py', 'test_monthly_vol_inflation178_kernel.py',
             'test_monthly_vol_inflation178_independent.py')
    modules = [module('test_vol178_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    for name, digest in prior_binding['input_sha256'].items(): bind(pins, Path(name), digest)
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
    plan = dict(schema='monthly_vol_inflation178_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_descriptive_volatility_estimates_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_volatility_inflation_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed formation volatility inflation and matched current252 reference; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor repair chain changed')
    return p


def scale_and_delta(values, n):
    require(set(values) == {'candidate', 'control', 'available'}, 'Feature keys changed')
    available = np.asarray(values['available'])
    shape = available.shape
    require(len(shape) == 2 and shape[1] > 0 and type(n) is int and 0 <= n <= shape[0], 'Invalid feature prefix')
    require(available.dtype == np.dtype(bool), 'Boolean mask required')
    transforms, deltas = {}, {}
    for arm, key in RAW.items():
        raw = np.asarray(values[key])
        require(raw.shape == shape, 'Invalid raw feature shape')
        x = np.asarray(raw[:n], float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x), available[:n]), 'Common feature mask invalid')
        transformed = np.zeros((n, shape[1]), float)
        require(np.all((x[available[:n]] >= -1.) & (x[available[:n]] <= 0.)), 'Raw volatility penalty must be in[-1,0]')
        transformed[available[:n]] = x[available[:n]]
        delta = DOSE * transformed
        if n: delta[0] = 0.
        require(np.max(np.abs(delta), initial=0.) <= DOSE, 'Score budget exceeded')
        transforms[arm], deltas[arm] = transformed, delta
    return transforms, deltas


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    ranks, deltas = scale_and_delta(values, n)
    for arm in NAMES[1:]:
        pre = study.modified_pre_score(study.configs()[4], arrays, n)
        view = pre[:n].copy(); changed = deltas[arm] != 0.
        view[changed] += deltas[arm][changed]; pre[:n] = view
        chosen, score = study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, ranks, deltas


def require_feature_audit(check, values):
    for key in ('passed', 'independent_scalar_rebuild', 'exact_common_availability',
                'no_future_or_outcome_inputs', 'audit_checked', 'three_simple_sample_volatilities_checked',
                'official_session_windows_checked', 'full_295_price_support_checked'):
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, symbols_per_event=49, checked_cells=122*49,
                    available_cells=int(values['available'].sum()),
                    unavailable_cells=int((~values['available']).sum()))
    require(values['available'].shape == (122, 49), 'Independent feature shape')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value,
                'Independent feature coverage: ' + key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0,
            'No predictive model fits allowed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in ('passed', 'full_candidate_universe', 'initial_delta_zero', 'common_available_only',
                'no_eligibility_or_model_filter', 'no_cross_sectional_ranking', 'direct_linear_mapping'):
        require(check.get(key) is True, 'Independent delta evidence missing '+key)
    require(check.get('dose') == DOSE, 'Independent delta dose differs')
    require(check.get('changed_cells') == {k: int(np.count_nonzero(v)) for k, v in deltas.items()}, 'Independent delta coverage')


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
    import monthly_vol_inflation178_kernel as kernel
    import monthly_vol_inflation178_independent as independent
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
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_deltas(values, deltas, dose=DOSE)
    require_delta_audit(rank_check, deltas)
    for n in FEATURE_PREFIXES:
        small, _ = kernel.build_features(panel, meta, calendar, limit=n)
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
    for n in range(123):
        oo, rr, dd = decisions(study, arrays, meta, values, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in RAW:
            np.testing.assert_array_equal(rr[k], ranks[k][:n]); np.testing.assert_array_equal(dd[k], deltas[k][:n])
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'TRANSFORMS_DELTAS.npz', **{'transform_' + k: v for k, v in ranks.items()}, **{'delta_' + k: v for k, v in deltas.items()})
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, feature_panel='original source adj', execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, **counts))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}, new_returns_computed=False, new_model_fits=0,
        predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays), **counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)


def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_delta_audit(audit['delta'], deltas)
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
    with np.load(dest / 'TRANSFORMS_DELTAS.npz') as data: deltas = {k: data['delta_' + k] for k in RAW}
    verify_prepared_evidence(c, values, deltas, read(dest / 'INDEPENDENT_PREPARE.json'))
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
        close_vs_current_volatility=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_current_volatility=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_vol_inflation178_independent as independent
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
    import monthly_vol_inflation178_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, formation_reference_overlap_with_current63=True, current252_control_overlaps_current63=True, observer_promotion=False))
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
        next_step='Review formation volatility-inflation results before a distinct new hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 176 and sum(r['trials'] for r in state['completed']) == 1043, 'Wrong final counters')
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
