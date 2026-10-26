"""Round183: frozen Local67 plus the lower own-sector conditional relative mean.

All actual features and portfolio returns require a prospectively bound PLAN.
Conditional and pooled means are descriptive statistics; no predictive model is fitted.
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
OUT = ROOT / 'outputs/monthly_sector_floor183'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_trend_gate182'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'sector_two_state_floor', 'sector_relative_mean_control')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE = 88, .05
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
SECTOR_ETFS = ('XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY')
MAPPING_PATH = ROOT / 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTORS_PATH = ROOT / 'outputs/index_union/alias_resolved_inputs/sectors.csv.gz'
ETF_MANIFESTS = tuple(ROOT / path for path in ('outputs/four_factor_rebound/ETF_SOURCES.json',
    'outputs/monthly_reconstruction_v2/ETF_MANIFEST.json',
    'outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'))
ESTIMATE_FIELDS = ('conditional_mean_estimates', 'pooled_mean_estimates')
MODEL_FIT_ACCOUNTING = 'new_model_fits and new_predictive_model_fits count only new predictive ML/Ridge/GBDT fits; conditional_mean_estimates counts two conditional means and pooled_mean_estimates one pooled mean per available stock cell, with zero estimates for unavailable cells; these are descriptive statistics, not independent samples; prefix-validation repeats are excluded'


PRIOR_EXPECTED = {
    'PLAN.json': 'c0ba9cebaa01fa03b86a051dd89dd4efedebb05cda9128fc96f180d0d12f0e38',
    'COMPLETION.json': 'ed21ccb10143c6f1c371c76c84ab68d59864087c96fd3b8f8206f40ab8fd9511',
    'results/COMPLETION.json': '30d7d01295531f0c7f4a6d098f833b3a58bf6f91cf4cdb848a1680164bf558d7',
}
CONTEXT = ROOT / 'work/monthly_sector_floor183_context.json'
CONTEXT_SHA = 'ff9850f59d0e0acd55c9e5d34803a4a0d522e0102b2f1a1e00b1ac5ccd1f3a39'
TEST_RECEIPT = ROOT / 'work/monthly_sector_floor183_TESTS.json'
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
    return module('monthly_sector_floor183_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        prospective_context_sha256=CONTEXT_SHA,
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='Own-sector-relative gains persistent in both rising-sector and falling-sector environments may distinguish durable long-momentum stocks from stocks reliant on one environment.',
        family='Conditional extension of the existing industry-relative family, with a matched observed-frequency pooled-mean control; no predictive ML fit or parameter search.',
        measure='Use211 adjusted prices of the stock and its original own-sector ETF ending at signal t-42. Let e be210 stock log returns minus own-sector ETF log returns. Define Up and Down by exact adjacent original sector price comparisons > and <; equal prices are excluded. Candidate=min(mean(e on Up),mean(e on Down)). Control=mean(e on Up union Down), equivalently the count-weighted mean of the two group means. Require at least40 observations in each group; no volatility normalization, variance floor, beta fit or current-pullback requirement.',
        source_window=dict(price_count=211, earliest_offset=252, latest_offset=42,
            return_count=210, minimum_group_count=40),
        numerical_policy='All211 stock and own-sector prices must be finite and strictly positive. Missing source, missing mapping or either group below40 makes both raw values NaN. Nonfinite arithmetic or overflow raises. Exact sector price comparisons define groups, without a zero-return tolerance. No filling, shortened history or proxy fallback.',
        mapping='Use original study.ex.t.strict.benchmark_map at the query date with source sectors truncated through query and frozen17 ETF identities. Pin the mapping source, original sectors panel, ETF manifests and original industry ETF files. ETF candidates and unknown mappings are neutral. XLC/XLRE are valid original benchmarks; missing their required price history is neutral without substitution.',
        availability='Identical valid stock/own-sector211-price support and minimum40 observations in each group for both arms. Rank across the same available original stock candidates before original eligibility filtering; ETFs are unavailable.',
        scoring='Independently rank candidate and control on the common available stock domain using average percentile rank minus0.5. Fewer than2 available stocks gives whole-row transform0. Unavailable cells get transform0. Add0.05*transform directly to original Local67 pre-score, then exactly one original episode; no second tanh or nonlinear mapping. First partial2016-08-17 delta0; maximum absolute delta0.025. The inherited percentile convention retains the common positive1/(2N) offset for all-tied ranks. Original baseline/core/support and eligibility remain unchanged. Fixed dose0.05 before actual183 evaluation; no rank, dose, threshold or window search.',
        portfolio='Unchanged27 original arrays, long-momentum main score and skip42 core; original historical proxy Top32+theme17 and eligibility; long-only EW3/two halves/under2 cash; fixed adjusted units intramonth; difference rebalance; no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_pooled_relative=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, feature_prefixes=list(FEATURE_PREFIXES), all123_order_prefixes=True),
        predecessor='Round182 completed without a new execution repair. Inherit its complete input/protection hash chain, including177 repair evidence and sources, and bind182 PLAN and root/result completions. Preserve historical central amendment references; do not import a new repair adapter.',
        novelty='An extension within industry-relative research, not a new information family. Readable downside_minimax70 concerned portfolio downside synchronization and unchanged top2; beta_asymmetry70 and sector_asymmetry138 concern conditional beta/correlation. Original sector_down70 and sector_residual70 sources are missing and their descriptions overlap with sector-down excess and residual momentum, so exhaustive novelty is unproven. A normalized/balanced-mean alternative was rejected prospectively without actual evaluation.',
        limitation='Repeatedly reused development sample, historical proxyTop32+theme17 and sector mapping, retrospectively retrieved vendor prices; not pristine OOS or certified strictPIT. Sector ETFs may include the stock. The lower conditional relative mean may penalize high-beta winners and favor near-beta1 behavior; it is descriptive evidence, not a rebound forecast or guaranteed downside protection. ETFs remain neutral; finite-rank common stock offset is shared by both arms. Group observations are not independent samples.',
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        new_arms=2, registered_trials=3, execution_paths=9,
        descriptive_statistics='Each available stock cell counts two conditional_mean_estimates and one pooled_mean_estimate; unavailable cells count0. Count only returned events, excluding repeated prefix validation calls.',
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_sector_floor183_host.py',
        'test_monthly_sector_floor183_host.py', 'monthly_sector_floor183_kernel.py',
        'test_monthly_sector_floor183_kernel.py', 'monthly_sector_floor183_independent.py',
        'test_monthly_sector_floor183_independent.py', 'ptr169_rebuild_evaluate.py',
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
    """Bind completed182 and preserve inherited evidence without requiring a new repair."""
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


def sector_source_binding(raw):
    """Bind metadata only; never load price values before registration."""
    require(sha(MAPPING_PATH) == MAPPING_SHA, 'Original industry mapping source changed')
    required = set(SECTOR_ETFS)
    entries = {}
    manifests = {}
    require(raw.get(str(SECTORS_PATH)) == sha(SECTORS_PATH), 'Sector panel missing from original raw pins')
    for manifest in ETF_MANIFESTS:
        digest = sha(manifest)
        require(raw.get(str(manifest)) == digest, 'ETF manifest missing from original raw pins')
        manifests[str(manifest)] = digest
        for entry in read(manifest):
            symbol = entry['symbol']
            if symbol not in required:
                continue
            require(symbol not in entries, 'Ambiguous sector ETF source ' + symbol)
            path = Path(entry['file']).resolve()
            require(path.is_relative_to(ROOT) and 'CloudStorage' not in path.parts, 'Nonlocal sector ETF source')
            require(raw.get(str(path)) == entry['sha256'] == sha(path), 'Sector ETF raw pin mismatch ' + symbol)
            entries[symbol] = dict(path=str(path), sha256=entry['sha256'], manifest=str(manifest))
    require(set(entries) == required, 'Missing source for required industry ETF')
    return dict(mapping_source=dict(path=str(MAPPING_PATH), sha256=MAPPING_SHA),
        sectors_source=dict(path=str(SECTORS_PATH), sha256=raw[str(SECTORS_PATH)]),
        manifest_sha256=manifests, own_sector_etfs=list(SECTOR_ETFS), sources=entries,
        price_column='Adj Close', loader='original study.arch.old.panel_load')


def sector_inputs(study, source, meta):
    strict = study.ex.t.strict
    require(Path(strict.__file__).resolve() == MAPPING_PATH.resolve() and sha(MAPPING_PATH) == MAPPING_SHA,
            'Loaded industry mapping is not the pinned original')
    require(len(meta['etfs']) == 17 and len(set(meta['etfs'])) == 17, 'Original17 ETF identities required')
    require(isinstance(source.get('adj'), pd.DataFrame) and isinstance(source.get('sectors'), pd.DataFrame),
            'Original adjusted prices and sector classifications required')
    require(set(SECTOR_ETFS) <= set(source['adj'].columns), 'Original sector ETF injection incomplete')
    return {'adj': source['adj'], 'sectors': source['sectors']}, strict.benchmark_map


def estimate_counts(check):
    available = check.get('available_cells')
    require(type(available) is int and 0 <= available <= 122*49, 'Available-cell accounting differs')
    expected = dict(conditional_mean_estimates=2*available, pooled_mean_estimates=available)
    require(all(type(check.get(key)) is int and check[key] == value for key, value in expected.items()),
            'Descriptive conditional/pooled mean accounting differs')
    return expected


def state_check(state):
    require(len(state['completed']) == 180 and sum(r['trials'] for r in state['completed']) == 1055,
            'Expected180 completed rounds /1055 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round182 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_SECTOR_FLOOR183_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_sector_floor183_host.py', 'test_monthly_sector_floor183_kernel.py',
             'test_monthly_sector_floor183_independent.py')
    modules = [module('test_sector183_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    source_binding = sector_source_binding(raw)
    bind(pins, MAPPING_PATH, MAPPING_SHA)
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
    plan = dict(schema='monthly_sector_floor183_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw, predecessor_binding=prior_binding,
        sector_source_binding=source_binding,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_conditional_mean_estimates_before_registration=0, actual_pooled_mean_estimates_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_sector_floor_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed own-sector two-state floor and pooled relative means; select only on first88 before later34 diagnostics.')
    commit_state(before, state, 'register')
    print(json.dumps(dict(status='REGISTERED', plan_sha256=sha(OUT / 'PLAN.json'))), flush=True)


def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['sector_source_binding'] == sector_source_binding(p['raw_source_hashes']), 'Registered sector source binding changed')
    require(all(p['input_sha256'].get(name) == digest for name, digest in context_binding().items()),
            'Prospective context binding changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor evidence chain changed')
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
        for i in range(n):
            mask = available[i]
            if np.count_nonzero(mask) >= 2:
                transformed[i, mask] = pd.Series(x[i, mask]).rank(method='average', pct=True).to_numpy() - .5
        delta = DOSE * transformed
        if n: delta[0] = 0.
        require(np.max(np.abs(delta), initial=0.) <= DOSE*.5, 'Score budget exceeded')
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
                'no_future_or_outcome_inputs', 'audit_checked', 'own_sector_mapping_checked',
                'exact_sector_price_sign_groups_checked', 'two_conditional_means_checked',
                'count_weighted_control_checked', 'minimum_40_per_group_checked',
                'official_session_windows_checked', 'full_211_price_support_checked', 'reconstructed_raw_ranks_exact'):
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
    require(check.get('mapping_source_sha256') == MAPPING_SHA, 'Independent mapping source changed')
    estimate_counts(check)


def require_delta_audit(check, deltas):
    for key in ('passed', 'full_candidate_universe', 'initial_delta_zero', 'common_available_only',
                'no_eligibility_or_model_filter', 'average_percentile_ranking_checked',
                'no_additional_nonlinearity', 'direct_linear_mapping'):
        require(check.get(key) is True, 'Independent delta evidence missing '+key)
    require(check.get('dose') == DOSE and check.get('max_absolute_delta') == .025, 'Independent delta budget differs')
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
    import monthly_sector_floor183_kernel as kernel
    import monthly_sector_floor183_independent as independent
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
    require(plan['sector_source_binding'] == sector_source_binding(raw_hashes), 'Prepared sector lineage differs')
    panel, mapper = sector_inputs(study, source, meta)
    calendar = read(CALENDAR)['sessions']
    values, audit = kernel.build_features(panel, meta, calendar, mapper)
    check = independent.verify_features(panel, meta, calendar, mapper, values, audit)
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
        small, _ = kernel.build_features(panel, meta, calendar, mapper, limit=n)
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
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, sector_source_binding=plan['sector_source_binding'], feature_panel='original source adj and sectors', execution_prices=parity))
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
        close_vs_pooled_relative=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_pooled_relative=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_sector_floor183_independent as independent
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
    import monthly_sector_floor183_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, own_sector_conditional_extension=True, descriptive_groups_not_independent_samples=True, observer_promotion=False))
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
        next_step='Review fixed own-sector two-state floor results before a distinct new hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 181 and sum(r['trials'] for r in state['completed']) == 1058, 'Wrong final counters')
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
