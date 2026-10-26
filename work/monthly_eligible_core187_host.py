"""Round187: half-core eligible midranks versus a matched affine-span control.

No actual source values or returns may be evaluated before prospective registration.
All fitted predictive coefficients, original arrays and original episode support are retained.
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
OUT = ROOT / 'outputs/monthly_eligible_core187'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_aux_reallocation186'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'eligible_core_rank', 'affine_core_span')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, HALF = 88, .5
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
QUALITY = ROOT / 'outputs/monthly_quality_research/FEATURES.npz'
QUALITY_SHA = '310768a7b3a0e5b1832b2ec10a58f1bf234c009a77ba0f27e1bad2cd1cffcdce'
ESTIMATE_FIELDS = ('eligible_cells', 'raw_core_finite_cells', 'ready_rows', 'fallback_rows',
    'equivalent_ready_rows', 'candidate_control_different_rows', 'core_rank_reconstructions',
    'max_candidate_control_difference')
MODEL_FIT_ACCOUNTING = 'Zero new predictive model fits and zero fitted coefficients. Counts describe canonical core reconstructions, source-valid cells and conditional rank/span mappings in returned events; repeated prefix checks are not added to those descriptive counts.'


PRIOR_EXPECTED = {
    'PLAN.json': '5cb9f1b6a7167d7bef0e5e4235563abd4c9dea2a5fafc3e28454f56737db33d9',
    'COMPLETION.json': '1835c8c819b88c53a9179953733c4227c45062a1bf8dbd796112001409ec17f1',
    'results/COMPLETION.json': 'bb8797b4f98710367a5934ba94550cde6561fb31cb854ec40f407e715692633d',
}
CONTEXT = ROOT / 'work/monthly_eligible_core187_context.json'
CONTEXT_SHA = 'f1d28bcb2020eb901520e11d4da161378385d58749b198f92967d2d652e69569'
NUMERICS = ROOT / 'work/monthly_eligible_core187_numerics.json'
NUMERICS_SHA = '2ef4d7a40dbd0b2462da0b709305b9bc283ff8ba6870c75b345267ba4b3695bb'
TEST_RECEIPT = ROOT / 'work/monthly_eligible_core187_TESTS.json'
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
sha, read, require, dump = ev.sha, ev.read, ev.require, ev.dump
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'original_raw_core_checked',
    'original_core_rank_parity_checked', 'original_market_state_checked', 'original_eligibility_checked',
    'eligible_domain_preserved', 'conditional_midrank_checked', 'affine_span_control_checked',
    'identity_branches_exact_checked', 'exact_common_availability', 'no_future_or_outcome_inputs', 'audit_checked')
DELTA_FLAGS = ('passed', 'independent_scalar_rebuild', 'half_core_replacement_checked',
    'initial_delta_zero', 'unavailable_exact_zero', 'original_support_unchanged',
    'original_model_corrections_unchanged', 'no_extra_rank_or_model_filter', 'audit_checked')


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def local():
    return module('monthly_eligible_core187_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        prospective_context_sha256=CONTEXT_SHA, prospective_numerics_sha256=NUMERICS_SHA,
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='Conditioning the dominant original core rank reference set on existing trading eligibility may improve selection beyond merely changing its score span.',
        family='Structural long-core rank calibration, not new information or a volatility-dose sweep. Replace HALF of the core using eligible midranks versus a matched affine-span control.',
        source='Original complete offline adjusted-price history through each query. Reconstruct canonical full49 core ranks using the original pandas rolling sample-standard-deviation convention; independently verify weighted core, original calm state and original295-observation eligibility. Raw validity remains separate from a neutral missing rank. Pin quality FEATURES core/core_weight and exact parity of baseline/core/core_weight/state_calm10. Never infer undefined raw core from a stored neutral zero.',
        raw_core='Calm: px.shift42/px.shift252-1,210 returns, weight0.50. Noncalm: (px.shift42/px.shift294-1)/current63 simple-return std(ddof1),252 formation returns, weight0.52. Exactly zero std becomes NaN; no new floor. Calm is original SPY63 sample std times sqrt252 strictly below0.10; original eligibility is295 nonmissing price observations, positive current price and positive sanitized skip42 raw210 momentum (original field raw252); original feature_panel replaces derived plus/minusInf with NaN before the eligibility comparison.',
        measure='U is the canonical original full49 average ascending percentile minus0.5, reconstructed from original raw core and checked against original weighted C. For E=original eligible, candidate e=average_rank(U[E])/len(E)-0.5. Control a=e_min+(U-U_min)*(e_max-e_min)/(U_max-U_min) on E; e_min/e_max include average ties. Assign affine endpoints exactly. With exactly two distinct eligible U levels, explicitly copy a=e. No second ranking or nonlinear mapping.',
        availability='Both arms are unavailable, targetNaN and delta0 on row0, fewer than2 eligible names, all49 eligible names, all eligible U exactly tied, or any eligible name with undefined original raw core. Otherwise both arms are available exactly on E. Never remove an undefined eligible member, shrink E or add a new eligibility filter; both stock and ETF eligible members remain in the same domain.',
        numerical_policy='Exact source ties use average ranks, without epsilon or symbol perturbation. Raw validity is independently reconstructed before rank filling. The first partial row and identity branches copy original scores exactly. Weighted core parity uses atol1e-12 and canonical rank reconstruction is independently checked; no clipping or acceptance of a reduced raw-core domain. Affine endpoints and two-level equivalence are explicit identities. The pinned numerical supplement preserves original feature_panel sanitization: raw252 plus/minusInf becomes NaN before eligibility raw252>0; this is original-source parity, not a new eligibility filter.',
        scoring='delta=0.5*(core_weight*target_rank-original_core), evaluated in this exact operation order only on available cells. Add only nonzero deltas to exact Local67 modified_pre_score(config4), then exactly one original episode with ORIGINAL baseline/core/core_weight/support. All original27arrays and model corrections stay unchanged. This is a halfCORE remapping with maximum absolute delta0.25 calm or0.26 noncalm, not a minor auxiliary adjustment. HALF0.5 is fixed; no dose search.',
        portfolio='Unchanged original eligibility and historical proxyTop32+theme17; long-only EW3/two halves/under2 cash; fixed adjusted units intramonth; difference rebalance; no terminal liquidation. Original source arrays, original support and nominal50/52 core weights remain fixed.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            changed_support_against=['incumbent_local67', 'affine_core_span'],
            require_noninferior_to_affine_span=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES),
            all123_order_prefixes=True, all123_delta_prefixes=True, original_quality_array_parity=True),
        predecessor='Round186 completed without a new execution repair. Inherit its complete input/protection hash chain including177 repair evidence; bind186 PLAN and both completions, preserving central historical amendment references.',
        novelty='Readable prior quality research tests stock/ETF grouping and own-history CDFs; core_amplitude changes fullpool normalization;177 keeps the original full49 rank domain;184 remaps volatility using core peers. The exact existing-eligibility core reference set with matched-span control was not found in readable records. Missing early source prevents exhaustive novelty; this remains a rank-calibration family. Holding-retention buffers were already tested and are not introduced here.',
        limitation='Repeatedly reused history, historical proxyTop32+theme17 and retrospectively retrieved prices, not pristine OOS or certified strictPIT. Same-close execution is idealized; next-open5bp is reported separately. Eligibility partly truncates momentum, so the candidate and affine control may coincide; identification requires distinct holding sets in both training eras. This is not an author-confirmed formula or new independent evidence; no observer promotion.',
        historical_best_tracking='First88 selects only eligible_core_rank. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0,
        fitted_coefficients=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9, training_metric_slices=18,
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_eligible_core187_host.py',
        'test_monthly_eligible_core187_host.py', 'monthly_eligible_core187_kernel.py',
        'test_monthly_eligible_core187_kernel.py', 'monthly_eligible_core187_independent.py',
        'test_monthly_eligible_core187_independent.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')] + [CONTEXT, NUMERICS]


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
    bind(pins, NUMERICS, NUMERICS_SHA)
    require(read(NUMERICS).get('context_sha256') == CONTEXT_SHA, 'Numerical supplement context differs')
    context = read(CONTEXT)
    for name, digest in context['local_evidence_sha256'].items():
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts, 'Nonlocal prospective context evidence')
        bind(pins, path, digest)
    return pins


def predecessor_binding(state):
    """Bind completed186 and preserve inherited evidence without requiring a new repair."""
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


def quality_source_binding():
    require(sha(QUALITY) == QUALITY_SHA, 'Original quality source changed')
    return dict(path=str(QUALITY), sha256=QUALITY_SHA,
        source_fields=['core', 'core_weight'],
        parity_fields=['baseline', 'core', 'core_weight', 'state_calm10'],
        loader='original study.arch.old.panel_load', price_column='Adj Close',
        rolling_history='complete original source through query; original pandas rolling ddof1')


def quality_inputs(original):
    """Called only after PLAN gate; never obtain actual arrays at registration."""
    quality_source_binding()
    with np.load(QUALITY) as data:
        for key in ('baseline', 'core', 'core_weight', 'state_calm10'):
            np.testing.assert_array_equal(data[key], original[key], err_msg='Quality/original parity '+key)
        source = {key: data[key].copy() for key in ('core', 'core_weight')}
    return source, dict(passed=True, original_quality_arrays_exact=True,
        parity_fields=['baseline', 'core', 'core_weight', 'state_calm10'], source=quality_source_binding())


def estimate_counts(check):
    fields = dict(events=122, symbols_per_event=49, checked_cells=5978, available_cells=5978,
        eligible_cells=5978, raw_core_finite_cells=5978, ready_rows=122, fallback_rows=122,
        equivalent_ready_rows=122, candidate_control_different_rows=122, core_rank_reconstructions=122)
    for key, upper in fields.items():
        value = check.get(key)
        require(type(value) is int and 0 <= value <= upper, 'Core mapping accounting differs: '+key)
    require(check['checked_cells'] == check['events'] * check['symbols_per_event'] and
        check['available_cells'] <= check['eligible_cells'] <= check['checked_cells'] and
        check['available_cells'] <= check['raw_core_finite_cells'] <= check['checked_cells'] and
        check['ready_rows'] + check['fallback_rows'] == check['events'] and
        check['equivalent_ready_rows'] + check['candidate_control_different_rows'] == check['ready_rows'] and
        check['core_rank_reconstructions'] == check['events'], 'Inconsistent core mapping accounting')
    maximum = check.get('max_candidate_control_difference')
    require(type(maximum) is float and np.isfinite(maximum) and 0. <= maximum <= 1.,
        'Candidate-control difference accounting differs')
    return {key: check[key] for key in ESTIMATE_FIELDS}


def state_check(state):
    require(len(state['completed']) == 184 and sum(r['trials'] for r in state['completed']) == 1069,
            'Expected184 completed rounds /1069 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round186 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_ELIGIBLE_CORE187_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_eligible_core187_host.py', 'test_monthly_eligible_core187_kernel.py',
             'test_monthly_eligible_core187_independent.py')
    modules = [module('test_eligible187_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    source_binding = quality_source_binding()
    bind(pins, QUALITY, QUALITY_SHA)
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
    plan = dict(schema='monthly_eligible_core187_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        quality_source_binding=source_binding,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_comparison_evaluations_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_eligible_core_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare eligible-core ranks and matched affine span; require first88 differences versus both controls before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['quality_source_binding'] == quality_source_binding(), 'Registered quality source binding changed')
    require(all(p['input_sha256'].get(name) == digest for name, digest in context_binding().items()),
            'Prospective context binding changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor evidence chain changed')
    return p


def decisions(study, arrays, meta, values, source, n):
    import monthly_eligible_core187_kernel as kernel
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    short, audit = kernel.build_deltas(values, arrays, source, limit=n)
    require(set(short) == {'candidate', 'control'}, 'Unexpected delta arm keys')
    deltas = {arm: short[key] for arm, key in RAW.items()}
    for arm in NAMES[1:]:
        require(deltas[arm].shape == (n, arrays['baseline'].shape[1]) and np.isfinite(deltas[arm]).all(), 'Invalid delta shape/values')
        require(np.all(np.abs(deltas[arm]) <= .26 + 1e-12) and (n == 0 or np.all(deltas[arm][0] == 0.)), 'Core remapping bound or first partial changed')
        pre = study.modified_pre_score(study.configs()[4], arrays, n)
        view = pre[:n].copy(); changed = deltas[arm] != 0.
        view[changed] += deltas[arm][changed]; pre[:n] = view
        chosen, score = study.adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, audit, deltas


def require_feature_audit(check, values):
    for key in FEATURE_FLAGS:
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, symbols_per_event=49, checked_cells=122*49,
        available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()))
    require(values['available'].shape == (122,49), 'Independent feature shape')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value, 'Independent feature coverage: '+key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0,
        'No predictive model fits allowed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in DELTA_FLAGS:
        require(check.get(key) is True, 'Independent delta evidence missing '+key)
    require(check.get('replacement_fraction') == HALF, 'Independent half replacement differs')
    require(type(check.get('predictive_model_fits')) is int and check['predictive_model_fits'] == 0 and
            type(check.get('episode_applications')) is int and check['episode_applications'] == 0,
            'Delta audit must not fit models or execute episode')
    require(check.get('changed_cells') == {key:int(np.count_nonzero(deltas[arm])) for arm,key in RAW.items()},
        'Independent delta coverage')


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
    import monthly_eligible_core187_kernel as kernel
    import monthly_eligible_core187_independent as independent
    study = local(); meta, arrays, px, op = study.load()
    require(len(meta['dates']) == 122 and arrays['baseline'].shape == (122,49), 'Original122x49 required')
    require(len(arrays) == 27, 'Original27 arrays required')
    require(all(len(s) == 49 for s in meta['symbols']), 'Original monthly candidate widths changed')
    before = {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}
    original_source, info = study.arch.old.panel_load(meta['etfs'])
    raw_hashes = study.raw_source_hashes(info)
    require(raw_hashes == plan['raw_source_hashes'], 'Original raw source lineage changed')
    verify(raw_hashes)
    parity = study.verify_panel_parity(original_source, px, op)
    panel = {'adj':original_source['adj']}
    source, quality_parity = quality_inputs(arrays)
    source_before = {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in source.items()}
    calendar = read(CALENDAR)['sessions']
    values, audit = kernel.build_features(panel, meta, calendar, arrays, source)
    check = independent.verify_features(panel, meta, calendar, arrays, source, values, audit)
    require_feature_audit(check, values)
    require(check['predictive_model_fits'] == audit['predictive_model_fits'] == 0, 'Model accounting differs')
    counts = estimate_counts(check)
    require(counts == estimate_counts(audit), 'Independent comparison accounting differs')
    orders, delta_audit, deltas = decisions(study, arrays, meta, values, source, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    short_deltas = {key:deltas[arm] for arm,key in RAW.items()}
    delta_check = independent.verify_deltas(values, arrays, source, short_deltas, delta_audit)
    require_delta_audit(delta_check, deltas)
    for n in FEATURE_PREFIXES:
        small, _ = kernel.build_features(panel, meta, calendar, arrays, source, limit=n)
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
    for n in range(123):
        oo, da, dd = decisions(study, arrays, meta, values, source, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in RAW: np.testing.assert_array_equal(dd[k], deltas[k][:n])
        for key in ('core_weights',):
            require(da[key] == delta_audit[key][:n], 'Prefix original core coefficient differs')
    require(before == {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in arrays.items()}, 'Original arrays modified')
    require(source_before == {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in source.items()}, 'Quality source arrays modified')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'TRANSFORMS_DELTAS.npz',
        **{'transform_'+arm:values[key] for arm,key in RAW.items()},
        **{'delta_'+arm:value for arm,value in deltas.items()},
        original_weighted_core=arrays['core'], original_eligible=arrays['eligible'],
        core_weights=np.asarray(delta_audit['core_weights']))
    dump(dest / 'FEATURE_AUDIT.json', dict(feature=audit, delta=delta_audit))
    dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes,
        quality_source_binding=plan['quality_source_binding'], original_quality_parity=quality_parity,
        feature_panel='complete original source adj', execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=delta_check, prefixes=123,
        feature_prefixes=list(FEATURE_PREFIXES), original_orders_exact=True, original_arrays_unchanged=True,
        original_array_count=27, original_quality_arrays_exact=True, quality_arrays_unchanged=True,
        new_model_fits=0, predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, **counts))
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', plan_sha256=expected, created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()}, new_returns_computed=False,
        new_model_fits=0, predictive_model_fits=0, new_predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, original_array_count=len(arrays), **counts))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest / 'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(c, values, deltas, audit):
    require(set(c['artifacts_sha256']) == PREPARED_FILES, 'Prepared artifact set incomplete')
    require_feature_audit(audit['feature'], values)
    require_delta_audit(audit['delta'], deltas)
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and audit.get('original_orders_exact') is True and
            audit.get('original_quality_arrays_exact') is True and audit.get('quality_arrays_unchanged') is True and
            audit.get('original_arrays_unchanged') is True and audit.get('original_array_count') == 27 and audit.get('new_model_fits') == 0 and
            audit.get('predictive_model_fits') == 0 and audit.get('model_fit_accounting') == MODEL_FIT_ACCOUNTING,
            'Prepared invariants not verified')
    for record in (audit, c):
        require(type(record.get('new_predictive_model_fits')) is int and record['new_predictive_model_fits'] == 0,
                'New predictive model fits not verified')
        expected_counts = estimate_counts(audit['feature'])
        require(all(type(record.get(key)) is type(value) and record[key] == value
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
    def changed_against(reference):
        return [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[reference][i]['symbols'])]
    def supported(changed):
        return len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed)
    changed = changed_against(NAMES[0]); changed_control = changed_against(NAMES[2])
    guards = dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], .005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], 0.),
        close_vs_affine_span=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_affine_span=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
        changed_support=supported(changed), changed_support_vs_control=supported(changed_control))
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, changed_months_vs_control=changed_control,
        first88_only=True, later34_used=False)


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
    import monthly_eligible_core187_independent as independent
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
    import monthly_eligible_core187_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, eligible_core_reference_remapping=True, matched_affine_span=True, observer_promotion=False))
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
        next_step='Review eligible-core versus affine-span identification and results before a distinct hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 185 and sum(r['trials'] for r in state['completed']) == 1072, 'Wrong final counters')
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
