"""Round194: Local67 plus same-report annual cash before the three signed trade working-capital contributions.

All actual features and portfolio returns require a prospectively bound PLAN.
All source admission and frozen tests precede actual financial features or returns.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_trade_wc194'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_cash_ppe193'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'same_support_cashflow_control', 'cash_before_trade_working_capital')
RAW = {NAMES[1]: 'control_bounded', NAMES[2]: 'candidate_bounded'}
TRAIN, DOSE, MAX_RAW, MAX_DELTA = 88, .01, 1., .01
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('available_cells', 'unavailable_cells')
MODEL_FIT_ACCOUNTING = 'Zero new predictive or descriptive model fits; same-report rational accounting ratios are bounded once. Prefix and independent checks do not create model fits.'
CONTEXT = ROOT / 'work/working_capital194_experiment_concept.json'
CONTEXT_SHA = '380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
SOURCE_DIR = ROOT / 'outputs/monthly_working_capital194_source/admitted'
SOURCE_POLICY = SOURCE_DIR / 'POLICY.json'
SOURCE_DATA = SOURCE_DIR / 'DATASET.json'
SOURCE_ADMISSION = SOURCE_DIR / 'ADMISSION.json'
SOURCE_EXPECTED = {
    'POLICY.json': 'd37e9606c03bb2e384a482fe9d326043127c86f471e0242424c6da1dbb9327a6',
    'DATASET.json': '0eca5d80051f4081db8d7f0bf0405efa895365090feb3c989c21786c44a17133',
    'ADMISSION.json': '8371d21eb44f1590f545b0808bfdf05ad63a52bf6ce63c285a3e889b758fde1c',
}
SOURCE_REVIEW = ROOT / 'outputs/monthly_working_capital194_source/independent/ADMISSION_REVIEW.json'
SOURCE_REVIEW_SHA = 'db43540f98c865dac5a718845ac5917c0c6305569edc7260a5136d1298a1b736'
SOURCE_REVIEW_CODE = ROOT / 'work/wc194_independent_admission_review.py'
SOURCE_REVIEW_CODE_SHA = '936997743dfb57d7470d958c4212755ed0cc2105862c556df407d38a8441eba8'
GUARD_PATH = ROOT / 'work/monthly_same_report145_runner.py'
GUARD_SHA = 'f400350c20a6d95f1d7ed60753ff529b5544b2b8901abbd0e801999843500939'
WALKFORWARD = ((36, 60), (60, 76), (76, 88))
PRIOR_EXPECTED = {
    'PLAN.json': '548653cc06de37290867c4384a4d748d4bf70c3f47373fe8c64542b739340b1c',
    'COMPLETION.json': '586821986e6dfc10d7070215c1379f22ddefaaadc42e816b325f972938afa2ef',
    'results/COMPLETION.json': '8cc920b6b505b37a73143d89036f76f7ca3b603f1eabf39e3cfe99203858bfc8',
}
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_common_availability',
    'latest_report_selected_before_validity', 'no_older_report_fallback', 'current_annual_only',
    'same_report_assets_checked', 'exact_rational_arithmetic_checked', 'trade_working_capital_scope_checked',
    'no_future_or_outcome_inputs', 'audit_checked')
TEST_RECEIPT = ROOT / 'work/monthly_trade_wc194_TESTS.json'
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
    return module('monthly_trade_wc194_original', LOCAL / 'study.py')


def contract():
    require(sha(CONTEXT) == CONTEXT_SHA, 'Changed prospective context')
    context = read(CONTEXT)
    return dict(context=context, prospective_context_sha256=CONTEXT_SHA,
        arms=list(NAMES), candidate=NAMES[2], control_nonpromotable=NAMES[1],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        dose=DOSE, max_absolute_raw=MAX_RAW, max_absolute_delta=MAX_DELTA,
        scoring='On original eligible cells add .01 times bounded financial value to complete original Local67 pre-score, then apply exactly one original episode, as in financial146. No additional first-row suppression; original selector controls initial allocation.',
        financial_source='Only new194 SOURCE_POLICY complete closure; old shared PDFium receipts retained as historical evidence, never reasserted as current runtime verification.',
        feature_prefix_source='Truncate query metadata and release-available events before rebuilding each feature prefix. Select latest known report before all validity gates; held latest blocks old reports.',
        walkforward_prefixes=[list(x) for x in WALKFORWARD],
        guard_source=dict(path=str(GUARD_PATH), sha256=GUARD_SHA, function='guard'),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES), all123_order_prefixes=True),
        limitation=context['data_limitations'],
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9,
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_trade_wc194_host.py',
        'test_monthly_trade_wc194_host.py', 'monthly_trade_wc194_factor.py',
        'test_monthly_trade_wc194_factor.py', 'monthly_trade_wc194_independent.py',
        'test_monthly_trade_wc194_independent.py', 'working_capital194_experiment_concept.json', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py', 'monthly_sector_recurrence192_independent.py',
        'monthly_same_report145_runner.py', 'monthly_cash_quality146_factor.py')]


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
    pins = {}
    for name, digest in PRIOR_EXPECTED.items(): bind(pins, PRIOR / name, digest)
    reference = state['completed'][-1]['completion']
    require(Path(reference['path']).resolve() == (PRIOR/'COMPLETION.json').resolve()
            and reference['sha256'] == PRIOR_EXPECTED['COMPLETION.json'], 'Predecessor central receipt')
    closed = read(PRIOR/'COMPLETION.json')
    require(closed.get('status') == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed.get('plan_sha256') == PRIOR_EXPECTED['PLAN.json'] and closed.get('protected_files') == 21444 and
            closed.get('observers_modified') is False, 'Predecessor closure incomplete')
    bind(pins, PRIOR/'results/COMPLETION.json', closed['result_completion_sha256'])
    result = read(PRIOR/'results/COMPLETION.json')
    require(result['status']=='COMPLETE' and result['plan_sha256']==PRIOR_EXPECTED['PLAN.json'] and
            result['paths']==9 and result['slices']==45, 'Predecessor results incomplete')
    return dict(directory=PRIOR.name, input_sha256=pins)


def financial_source_binding():
    """Validate only the new194 closure, not old shared-runtime manifests recursively."""
    require(set(SOURCE_EXPECTED) == {'POLICY.json', 'DATASET.json', 'ADMISSION.json'},
            'Completed source-only admission digests required before registration')
    pins = {}
    for name, digest in SOURCE_EXPECTED.items(): bind(pins, SOURCE_DIR/name, digest)
    policy, admission = read(SOURCE_POLICY), read(SOURCE_ADMISSION)
    require(admission.get('passed') is True and admission.get('events') == 44 and
            admission.get('admitted_events') == 29 and admission.get('held_events') == 15,
            'Complete reviewed44 source census with29 admitted15 held required')
    require(admission.get('data_sha256') == sha(SOURCE_DATA) and
            admission.get('policy_sha256') == sha(SOURCE_POLICY), 'Source admission root differs')
    require(admission.get('all_original_event_values_preserved') is True and
            admission.get('inherited_admitted_events') == 31 and admission.get('inherited_held_events') == 13,
            'Inherited source census changed')
    for key in ('factor_calculations', 'financial_ratios', 'return_calculations', 'observer_changes'):
        require(type(admission.get(key)) is int and admission[key] == 0, 'Source-only stage was not source-only')
    require(policy.get('concept_sha256') == CONTEXT_SHA and policy.get('conditional_research_only') is True and
            policy.get('current_report_trade_working_capital_only') is True, 'Wrong admitted source policy')
    closure = policy.get('input_hashes')
    require(isinstance(closure, dict) and closure, 'Complete new194 source closure required')
    verify(closure)
    for name, digest in closure.items(): bind(pins, Path(name) if Path(name).is_absolute() else ROOT/name, digest)
    reference = admission.get('independent_review', {})
    path = Path(reference.get('path', ''))
    path = path if path.is_absolute() else ROOT/path
    require(str(path) in pins and pins[str(path)] == reference.get('sha256'), 'Independent source review not pinned')
    review = read(path)
    require(review.get('passed') is True, 'Independent source review failed')
    bind(pins, SOURCE_REVIEW, SOURCE_REVIEW_SHA)
    bind(pins, SOURCE_REVIEW_CODE, SOURCE_REVIEW_CODE_SHA)
    admitted_review = read(SOURCE_REVIEW)
    require(admitted_review.get('passed') is True and admitted_review.get('source_only') is True and
            admitted_review.get('events') == 44 and admitted_review.get('admitted_events') == 29 and
            admitted_review.get('held_events') == 15 and admitted_review.get('inherited_holds_preserved') == 13 and
            admitted_review.get('closure_pins_verified') == len(closure) and len(closure) > 200 and
            admitted_review.get('review_code_sha256') == SOURCE_REVIEW_CODE_SHA,
            'Independent normalized-source admission review missing')
    for key in ('all_original_event_values_preserved','all_raw_producer_evidence_preserved',
                'normalized_fields_and_source_clock_checked','source_review_closure_preserved'):
        require(admitted_review.get(key) is True, 'Source admission review check missing: '+key)
    for key in ('ratios_computed','factors_computed','returns_computed','production_modules_imported'):
        require(admitted_review.get(key) is False, 'Source review exceeded source-only scope')
    for group in ('input_sha256','source_code_sha256'):
        verify(admitted_review[group])
        for name,digest in admitted_review[group].items():
            bind(pins, Path(name) if Path(name).is_absolute() else ROOT/name, digest)
    event_reviews=admitted_review.get('event_reviews',[])
    require(len(event_reviews)==44 and len({r['event_id'] for r in event_reviews})==44 and
            {r['event_id'] for r in event_reviews if r['admitted']}==set(admission['admitted_event_ids']) and
            {r['event_id'] for r in event_reviews if not r['admitted']}==set(admission['held_event_ids']),
            'Independent source admission event census differs')
    require(all(all(r.get(k) is True for k in ('original_values_exact','raw_evidence_exact',
                'normalization_exact','source_clock_verified')) for r in event_reviews), 'Incomplete per-event source review')
    return dict(directory=str(SOURCE_DIR), input_sha256=pins,
                data_sha256=sha(SOURCE_DATA), policy_sha256=sha(SOURCE_POLICY),
                admission_sha256=sha(SOURCE_ADMISSION), events=44, admitted_events=29, held_events=15,
                independent_review=reference,
                normalized_admission_review=dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA),
                old_shared_pdf_runtime_revalidated=False)


def estimate_counts(check):
    available = check.get('available_cells')
    require(type(available) is int and 0 <= available <= 122*49, 'Invalid available-cell count')
    unavailable = check.get('unavailable_cells')
    require(type(unavailable) is int and available + unavailable == 122*49, 'Source support coverage differs')
    return dict(available_cells=available, unavailable_cells=unavailable)


def feature_prefix_inputs(meta, dataset, n):
    """Keep latest-report semantics while physically removing not-yet-known events."""
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Invalid source prefix')
    small_meta = deepcopy(meta)
    for key in ('dates', 'ends', 'symbols'):
        if key in small_meta: small_meta[key] = small_meta[key][:n]
    source = {k: deepcopy(v) for k,v in dataset.items() if k != 'events'}
    cutoff = None if n == 0 else pd.Timestamp(meta['dates'][n-1])
    source['events'] = []
    for event in dataset['events']:
        # Inspect only the source clock before deciding whether to copy its numerical fields.
        available = pd.Timestamp(event['available_from'])
        require(not pd.isna(available) and available.tz is None, 'Invalid source availability date')
        if cutoff is not None and available <= cutoff: source['events'].append(deepcopy(event))
    return small_meta, source

def state_check(state):
    require(len(state['completed']) == 191 and sum(r['trials'] for r in state['completed']) == 1090,
            'Expected191 completed entries /1090 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round193 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_TRADE_WC194_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_trade_wc194_host.py', 'test_monthly_trade_wc194_factor.py',
             'test_monthly_trade_wc194_independent.py')
    modules = [module('test_wc194_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    require(result.wasSuccessful(), 'Synthetic validation failed')
    receipt = TEST_RECEIPT
    require(not receipt.exists(), 'Test receipt is write-once')
    dump(receipt, dict(passed=True, synthetic_only=True, tests=result.testsRun,
        runtime=ev.runtime(), code_sha256={str(p): sha(p) for p in code_files()}))


def code_review_binding():
    path = ROOT / 'outputs/monthly_working_capital194_source/independent/CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:6]}
    require(review.get('passed') is True and review.get('unresolved_blockers') == [], 'Independent numerical code review incomplete')
    require(review.get('code_sha256') == expected, 'Numerical files changed after independent review')
    for key in ('actual_features_computed', 'actual_financial_ratios_computed', 'actual_portfolio_returns_computed', 'production_files_modified', 'strategy_registered_by_reviewer'):
        require(review.get(key) is False, 'Independent review scope differs: '+key)
    pins = {str(path): sha(path)}
    for name, digest in review.get('input_sha256', {}).items():
        bind(pins, Path(name) if Path(name).is_absolute() else ROOT/name, digest)
    verify(pins)
    return dict(path=str(path), sha256=sha(path), input_sha256=pins)


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
    source_binding = financial_source_binding()
    numerical_review = code_review_binding()
    for name,digest in numerical_review['input_sha256'].items(): bind(pins, Path(name), digest)
    for name,digest in source_binding['input_sha256'].items(): bind(pins, Path(name), digest)
    require(sha(GUARD_PATH) == GUARD_SHA, 'Frozen financial-study guard changed')
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
    plan = dict(schema='monthly_trade_wc194_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        financial_source_binding=source_binding, numerical_code_review=numerical_review,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_financial_ratios_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_trade_working_capital_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed same-report cash-before-trade-working-capital and same-support CFO; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['financial_source_binding'] == financial_source_binding(), 'Registered financial source binding changed')
    require(p.get('numerical_code_review') == code_review_binding(), 'Registered numerical code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p


def scale_and_delta(values, eligibility, n):
    require(set(values) == {'candidate_bounded', 'control_bounded', 'common_available'}, 'Feature keys changed')
    available = np.asarray(values['common_available'])
    shape = available.shape
    require(len(shape) == 2 and shape[1] > 0 and type(n) is int and 0 <= n <= shape[0], 'Invalid feature prefix')
    require(available.dtype == np.dtype(bool), 'Boolean mask required')
    eligible = np.asarray(eligibility)
    require(eligible.shape == shape and eligible.dtype == np.dtype(bool), 'Original eligibility mask required')
    transforms, deltas = {}, {}
    for arm, key in RAW.items():
        raw = np.asarray(values[key])
        require(raw.shape == shape, 'Invalid raw feature shape')
        x = np.asarray(raw[:n], float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x), available[:n]), 'Common feature mask invalid')
        transformed = np.zeros((n, shape[1]), float)
        require((np.abs(x[available[:n]]) <= MAX_RAW).all(), 'Bounded financial value exceeded')
        transformed[available[:n]] = x[available[:n]]
        delta = np.zeros_like(transformed)
        active = eligible[:n] & available[:n] & (transformed != 0.)
        delta[active] = DOSE * transformed[active]
        require(np.max(np.abs(delta), initial=0.) <= DOSE * MAX_RAW, 'Score budget exceeded')
        transforms[arm], deltas[arm] = transformed, delta
    return transforms, deltas


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    ranks, deltas = scale_and_delta(values, arrays['eligible'], n)
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
    for key in FEATURE_FLAGS:
        require(check.get(key) is True, 'Independent feature evidence missing '+key)
    expected = dict(events=122, width=49, checked_cells=122*49,
                    available_cells=int(values['common_available'].sum()),
                    unavailable_cells=int((~values['common_available']).sum()))
    require(values['common_available'].shape == (122, 49), 'Independent feature shape')
    for key, value in expected.items():
        require(type(check.get(key)) is int and check[key] == value,
                'Independent feature coverage: ' + key)
    require(check.get('computes_returns') is False, 'Feature audit must not compute portfolio returns')
    require(type(check.get('new_model_fits')) is int and check['new_model_fits'] == 0,
            'No predictive model fits allowed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in ('passed', 'eligibility_preserved', 'eligible_only', 'common_available_only',
                'no_new_availability_filter', 'no_cross_sectional_ranking', 'direct_linear_mapping'):
        require(check.get(key) is True, 'Independent delta evidence missing '+key)
    require(check.get('dose') == DOSE and
            check.get('max_absolute_delta') == MAX_DELTA, 'Independent delta dose/bound differs')
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
    import monthly_trade_wc194_factor as factor
    import monthly_trade_wc194_independent as independent
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
    dataset = read(SOURCE_DATA)
    before_dataset = deepcopy(dataset)
    built = factor.build_features(meta, dataset); values, audit = built['arrays'], built['audit']
    check = independent.verify_features(meta, dataset, built)
    require_feature_audit(check, values)
    require(check['new_model_fits'] == audit['new_model_fits'] == 0,
            'Independent model accounting differs')
    counts = estimate_counts(check)
    require(counts['available_cells'] == audit['common_available_cells'] and
            len(audit['cells']) == 122*49, 'Independent financial support accounting differs')
    orders, ranks, deltas = decisions(study, arrays, meta, values, 122)
    require(orders[NAMES[0]] == read(LOCAL / 'local_linear_ORDERS.json'), 'Original full order parity failed')
    rank_check = independent.verify_deltas(values, deltas, arrays['eligible'], dose=DOSE)
    require_delta_audit(rank_check, deltas)
    for n in FEATURE_PREFIXES:
        small_meta, small_dataset = feature_prefix_inputs(meta, dataset, n)
        small_built = factor.build_features(small_meta, small_dataset)
        small, small_audit = small_built['arrays'], small_built['audit']
        require(small_audit['cells'] == audit['cells'][:n*49], 'Truncated report prefix audit differs')
        for key in values: np.testing.assert_array_equal(small[key], values[key][:n])
    for n in range(123):
        oo, rr, dd = decisions(study, arrays, meta, values, n)
        require(all(oo[k] == orders[k][:n] for k in NAMES), 'Prefix order parity failed')
        for k in RAW:
            np.testing.assert_array_equal(rr[k], ranks[k][:n]); np.testing.assert_array_equal(dd[k], deltas[k][:n])
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays modified')
    require(dataset == before_dataset, 'Admitted source dataset modified')
    empty = {key: np.full_like(value, False if key == 'common_available' else np.nan)
             for key,value in values.items()}
    neutral_orders, _, _ = decisions(study, arrays, meta, empty, 122)
    require(all(neutral_orders[name] == orders[NAMES[0]] for name in NAMES), 'Missing financial values must preserve exact original orders')
    zero = {key: value.copy() for key,value in values.items()}
    for key in RAW.values(): zero[key][zero['common_available']] = 0.
    zero_orders, _, _ = decisions(study, arrays, meta, zero, 122)
    require(all(zero_orders[name] == orders[NAMES[0]] for name in NAMES), 'Zero financial values must preserve exact original orders')
    gate(expected)
    dest = OUT / 'prepared'; dest.mkdir()
    np.savez_compressed(dest / 'FEATURES.npz', **values)
    np.savez_compressed(dest / 'TRANSFORMS_DELTAS.npz', **{'transform_' + k: v for k, v in ranks.items()}, **{'delta_' + k: v for k, v in deltas.items()})
    dump(dest / 'FEATURE_AUDIT.json', audit); dump(dest / 'ORDERS.json', orders); dump(dest / 'META.json', meta)
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, financial_source_binding=plan['financial_source_binding'],
        feature_dataset_sha256=sha(SOURCE_DATA), execution_prices=parity, financial_source_unmodified=True))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True, original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27,
        zero_missing_exact_fallback=True, financial_source_unmodified=True, new_model_fits=0,
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
    require(audit.get('prefixes') == 123 and audit.get('feature_prefixes') == list(FEATURE_PREFIXES) and
            audit.get('feature_prefix_sources_truncated') is True and audit.get('original_orders_exact') is True and
            audit.get('zero_missing_exact_fallback') is True and audit.get('financial_source_unmodified') is True and
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


def train_metrics(engine, eq, meta, n=TRAIN):
    require(type(n) is int and 2 <= n <= TRAIN, 'Training prefix outside first88')
    split = 53 if n == TRAIN else n//2
    answer = {}
    for name, lo, hi in (('training', 0, n-1), ('early', 0, split-1), ('recent', split, n-1)):
        part = eq.loc[meta['dates'][lo]:meta['ends'][hi]]
        require(len(part) >= 2 and np.isfinite(part.to_numpy()).all() and (part.to_numpy() > 0).all(),
                'Invalid training equity window')
        answer[name] = engine.stats(part / part.iloc[0] * 100000.)
    return answer


def guard(candidate, reference, strict=True):
    """Execute only the exact pinned pure145 guard, without its old lifecycle imports."""
    require(sha(GUARD_PATH) == GUARD_SHA, 'Frozen financial guard changed')
    parsed = ast.parse(GUARD_PATH.read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'guard']
    require(len(nodes) == 1, 'Unique original financial guard required')
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GUARD_PATH), 'exec'), scope)
    return scope['guard'](candidate, reference, strict=strict)


def select_first88(metrics, orders, n=TRAIN):
    require(type(n) is int and 2 <= n <= TRAIN, 'Selection prefix outside first88')
    require(set(metrics) == set(orders) == set(NAMES), 'Three fixed arms required')
    for name in NAMES:
        require(len(orders[name]) >= n and set(metrics[name]) >= {'same_close_zero','next_open_cost'},
                'Incomplete frozen training data')
    candidate = NAMES[2]
    guards = dict(close_vs_incumbent=guard(metrics[candidate]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], True),
        open_vs_incumbent=guard(metrics[candidate]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], False),
        close_vs_same_support_cashflow=guard(metrics[candidate]['same_close_zero'], metrics[NAMES[1]]['same_close_zero'], True),
        open_vs_same_support_cashflow=guard(metrics[candidate]['next_open_cost'], metrics[NAMES[1]]['next_open_cost'], True))
    changed = [i for i in range(n) if set(orders[candidate][i]['symbols']) != set(orders[NAMES[0]][i]['symbols'])]
    changed_control = [i for i in range(n) if set(orders[candidate][i]['symbols']) != set(orders[NAMES[1]][i]['symbols'])]
    return dict(primary=candidate if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, changed_months_vs_control=changed_control,
        training_months=n, first88_only=True, later34_used=False)


def training_curves(engine, px, op, orders, meta, n):
    require(type(n) is int and 2 <= n <= TRAIN, 'Training escaped first88')
    metrics, curves = {}, {}
    for arm in NAMES:
        metrics[arm], curves[arm] = {}, {}
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[2]):
            _, eq, _, _ = engine.replay(px, op, orders[arm][:n], cost=cost, execution=mode,
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][n-1]))
            require(str(eq.index[-1].date()) == meta['ends'][n-1], 'Selection escaped training cutoff')
            metrics[arm][path] = train_metrics(engine, eq, meta, n)
            curves[arm][path] = eq
    return metrics, curves


def walkforward_training(engine, px, op, orders, meta, curves):
    """Prefix selection and same-close stitched diagnostic, as in the prior financial studies."""
    folds, stitched, reference = [], [], []
    for n, end in WALKFORWARD:
        metrics, _ = training_curves(engine, px, op, orders, meta, n)
        choice = select_first88(metrics, orders, n)
        pieces = []
        for arm in (choice['primary'], NAMES[0]):
            eq = curves[arm]['same_close_zero'].loc[meta['dates'][n]:meta['ends'][end-1]]
            require(len(eq) >= 2 and str(eq.index[-1].date()) == meta['ends'][end-1], 'Incomplete walkforward window')
            pieces.append(eq.pct_change(fill_method=None).iloc[1:])
        require(pieces[0].index.equals(pieces[1].index), 'Walkforward reference calendar mismatch')
        stitched.append(pieces[0]); reference.append(pieces[1])
        folds.append(dict(n=n, end=end, selection=choice, training=metrics,
            first_evaluation_signal=meta['dates'][n], last_evaluation_end=meta['ends'][end-1]))
    result = dict(folds=folds, pristine_oos=False, used_for_primary_selection=False,
                  execution='same_close_zero', later34_used=False)
    for key, pieces in (('metrics', stitched), ('incumbent', reference)):
        returns = pd.concat(pieces)
        require(returns.index.is_unique and returns.index.is_monotonic_increasing, 'Overlapping walkforward windows')
        initial = pd.Series([100000.], index=[pd.Timestamp(meta['dates'][36])])
        equity = pd.concat([initial, 100000.*(1.+returns).cumprod()])
        result[key] = engine.stats(equity)
    return result


def select(expected):
    meta, orders = prepared(expected); require(not (OUT / 'selected').exists(), 'Selection is write-once')
    px, op = ev.prices(); engine = ev.engine()
    metrics, curves = training_curves(engine, px, op, orders, meta, TRAIN)
    result = select_first88(metrics, orders)
    walk = walkforward_training(engine, px, op, orders, meta, curves)
    gate(expected)
    dest = OUT / 'selected'; dest.mkdir(); dump(dest / 'TRAIN_RESULTS.json', metrics)
    dump(dest / 'WALKFORWARD.json', walk)
    dump(dest / 'FROZEN.json', dict(created_utc=now(), plan_sha256=expected,
        prepared_sha256=sha(OUT / 'prepared/COMPLETION.json'), orders_sha256=sha(OUT / 'prepared/ORDERS.json'),
        training_sha256=sha(dest / 'TRAIN_RESULTS.json'), walkforward_sha256=sha(dest/'WALKFORWARD.json'), selection=result))
    print(json.dumps(result), flush=True)


def frozen_selection(expected, orders):
    frozen = read(OUT / 'selected/FROZEN.json')
    require(frozen['plan_sha256'] == expected and frozen['orders_sha256'] == sha(OUT / 'prepared/ORDERS.json') and
        frozen['training_sha256'] == sha(OUT / 'selected/TRAIN_RESULTS.json') and
        frozen['prepared_sha256'] == sha(OUT / 'prepared/COMPLETION.json') and
        frozen['walkforward_sha256'] == sha(OUT/'selected/WALKFORWARD.json'), 'Frozen selection dependencies changed')
    require(select_first88(read(OUT / 'selected/TRAIN_RESULTS.json'), orders) == frozen['selection'], 'Selection mismatch')
    walk = read(OUT/'selected/WALKFORWARD.json')
    require(walk.get('later34_used') is False and walk.get('used_for_primary_selection') is False and
            [(f['n'],f['end']) for f in walk['folds']] == list(WALKFORWARD), 'Walkforward contract changed')
    for fold in walk['folds']:
        require(select_first88(fold['training'], orders, fold['n']) == fold['selection'], 'Walkforward prefix selection mismatch')
    return frozen


def verify_training_against_audit(training, audit):
    import monthly_trade_wc194_independent as independent
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
    import monthly_trade_wc194_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, current_annual_cash_before_trade_working_capital=True, same_support_both_arms=True, observer_promotion=False))
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
        next_step='Review same-support annual cash-before-trade-working-capital versus CFO results before another distinct hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 192 and sum(r['trials'] for r in state['completed']) == 1093, 'Wrong final counters')
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
