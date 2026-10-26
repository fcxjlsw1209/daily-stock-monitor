"""Round177: noncalm half-core breakpoint versus fixed-midpoint drift.

All actual features and portfolio returns require a prospectively bound PLAN.
Synthetic tests fit no predictive model. Historical drift estimates are separately
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
OUT = ROOT / 'outputs/monthly_breakpoint177'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRIOR = ROOT / 'outputs/monthly_sector_dimension176'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67', 'breakpoint_core', 'midpoint_core')
RAW = {NAMES[1]: 'candidate', NAMES[2]: 'control'}
TRAIN, DOSE = 88, .5
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
CORE_ADAPTER_PATH = ROOT / 'outputs/monthly_core_timing70_research/core_adapter.py'
CORE_ADAPTER_SHA = 'b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a'
PATH_ADAPTER_PATH = ROOT / 'outputs/monthly_path70_research/incumbent_adapter.py'
PATH_ADAPTER_SHA = '5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b'
ESTIMATE_FIELDS = ('constant_mean_estimates', 'split_mean_estimates', 'candidate_breaks', 'control_breaks')
MODEL_FIT_ACCOUNTING = 'new_model_fits and new_predictive_model_fits count only predictive ML/Ridge/GBDT fits; within-formation means and breakpoint selections are descriptive estimates counted separately over the full feature audit, excluding prefix-validation repeats'
TEST_RECEIPT = ROOT / 'work/monthly_breakpoint177_TESTS.json'
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
    return module('monthly_breakpoint177_original', LOCAL / 'study.py')


def contract():
    return dict(arms=list(NAMES), candidate=NAMES[1], control_nonpromotable=NAMES[2],
        period=['2016-08-17', '2026-09-24'], training_months=88, later_months=34,
        hypothesis='A single persistent change in historical return drift may make the last drift segment a better long-momentum estimate than a fixed midpoint split.',
        family='Within-formation structural-break estimation replacing half of the noncalm long core; no additional auxiliary factor or new predictive ML fit.',
        formation='At signal session t,253 adjusted prices t-294..t-42 supply252 logreturns. Test every split k=63..189, first k returns versus remaining252-k. Find minimum total within-segment SSE; all splits with SSE<=minimum_SSE+1e-12*SSE0 are numerical ties, selecting the earliest k. This declared relative-SSE tolerance only resolves floating-point tie ambiguity; the acceptance criterion remains strict. Compare the selected split with a fixed k=126 control, each using its own improvement criterion.',
        criterion='For n=252, accept a split iff n*(log(SSE0)-log(SSE1))>2*log(n), using the actual SSE1 of the earliest selected numerical-tie split, not the search minimum SSE. Formation log-return sample std<=1e-12 is numerically constant: do not compute either log-SSE ratio, record statistic=None and accepted=False, and copy baseline exactly. Candidate k63 is only a canonical inactive placeholder, not an argmin claim; control retains k126. All127 split SSE estimates remain audited and counted. Otherwise SSE0=0 rejects both; SSE1=0<SSE0 accepts. This is a fixed descriptive complexity penalty, not a calibrated hypothesis-test p-value. No outcome-based tuning.',
        raw='baseline=(P[t-42]/P[t-294]-1)/sample_std(last63 simple returns through t). Baseline needs only those endpoints and64 prices t-63..t, with positive finite std. Candidate/control require all253 formation prices and64 recent prices, std>1e-12, and both finite outputs. Accepted split raw=expm1(252*post-split mean logreturn)/same std; rejected split copies baseline exactly.',
        source_window=dict(formation_prices=253, formation_returns=252, earliest_offset=294,
            formation_latest_offset=42, volatility_prices=64, volatility_returns=63,
            volatility_earliest_offset=63, volatility_latest_offset=0,
            minimum_segment_returns=63, first_split=63, last_split=189, midpoint=126,
            feature_std_floor=1e-12, baseline_std_floor=0., formation_log_std_floor=1e-12,
            split_sse_relative_tolerance=1e-12),
        ranking='Average percentile rank minus0.5 across all49 original same-date candidates, before eligibility. Missing baseline raw gets neutral rank0. Exact baseline centered ranks times0.52 must equal original weighted core for every noncalm row. Any noncalm row with an unavailable new feature for a baseline-finite member is wholly neutral; availability outside baseline-finite domain is forbidden. Never shrink the original rank domain.',
        scoring='Only noncalm ready rows after the first partial allocation receive delta=0.5*(0.52*new_rank-original_weighted_core). Calm uses its original0.50 raw210-session core without any modification. Original noncalm total core weight0.52 remains half old and half new. Initial,calm and unready rows pass all-NaN ranks to the original core adapter, giving exact zero delta.',
        metadata='Use original core_adapter.score_and_choose with half blend. It adds identical delta to copies of baseline,weighted core,and Local67 pre-score, preserving auxiliary support. Its episode wrapper passes original baseline/core/weights to the original single episode selector, while episode candidates use modified pre-score. No post-episode rerank, duplicated episode,changed eligibility or hidden support recalculation.',
        portfolio='Unchanged27 original arrays and original Top32+theme17 proxy universe. Long-only EW3/two halves/under2 cash; monthly fixed adjusted units, difference rebalance, no terminal liquidation.',
        selection=dict(first88_only=True, minimum_cagr_gain=.005, open_5bp_noninferior=True,
            sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
            changed_set_months_min=3, changed_set_months_each_era_min=1,
            require_noninferior_to_midpoint=True, later34_for_selection=False),
        verification=dict(independent_all_cells=True, exact_noncalm_original_core_rank_parity=True,
            whole_row_readiness=True, feature_prefixes=list(FEATURE_PREFIXES), all123_order_prefixes=True),
        novelty='Core timing tested calendar anchors and5/21 observation smoothing; core memory tested9/11percent volatility hysteresis. Fixed-stage path growth,OLS/TheilSen and Kalman local-trend families were already tested. The available Kalman receipts identify adaptive/fixed local-trend and constant-drift quarter-core arms, but its exact formula files are unavailable locally. No exact single-SSE-break selection was found in inspected readable records; this is a bounded distinction within trend estimation, not an exhaustive novelty claim.',
        limitation='Repeatedly reused development sample, estimated historical membership and retrospectively retrieved vendor prices; not pristine OOS/PIT. A detected historical change need not persist. Split selection estimates are noisy and the fixed penalty is not a calibrated inference guarantee. Momentum numerator ends t-42 but the unchanged risk denominator deliberately includes returns through t.',
        historical_best_tracking='First88 selects this fixed candidate. Historical best is descriptive full-period performance among previously training-selected candidates, not independent validation or observer promotion.',
        new_model_fits=0, new_predictive_model_fits=0, predictive_model_fits=0,
        model_fit_accounting=MODEL_FIT_ACCOUNTING, descriptive_statistics='One constant mean and127 two-segment split estimates per completed formation cell; the fixed126 control reuses that split. Accepted-break counts include any subsequent nonfinite forecast rejection. Prefix-validation repeats are excluded.', new_arms=2, registered_trials=3, execution_paths=9,
        no_observer_writes=True, original_pool_preserved=True, no_parameter_search=True)


def code_files():
    return [ROOT / 'work' / n for n in ('monthly_breakpoint177_host.py',
        'test_monthly_breakpoint177_host.py', 'monthly_breakpoint177_kernel.py',
        'test_monthly_breakpoint177_kernel.py', 'monthly_breakpoint177_independent.py',
        'test_monthly_breakpoint177_independent.py', 'ptr169_rebuild_evaluate.py',
        'ptr169_results_independent.py', 'price_distribution171_independent.py')] + [CORE_ADAPTER_PATH, PATH_ADAPTER_PATH]


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


def core_adapter():
    require(sha(CORE_ADAPTER_PATH) == CORE_ADAPTER_SHA and sha(PATH_ADAPTER_PATH) == PATH_ADAPTER_SHA,
            'Original core/path adapter source changed')
    return module('breakpoint177_frozen_core_adapter', CORE_ADAPTER_PATH)


def estimate_counts(check):
    counts = {key: check.get(key) for key in ESTIMATE_FIELDS}
    require(all(type(value) is int and value >= 0 for value in counts.values()), 'Invalid descriptive estimate counts')
    total = counts['constant_mean_estimates']
    require(total <= 122*49 and counts['split_mean_estimates'] == 127*total and
            counts['candidate_breaks'] <= total and counts['control_breaks'] <= total,
            'Descriptive estimate accounting differs')
    return counts


def state_check(state):
    require(len(state['completed']) == 174 and sum(r['trials'] for r in state['completed']) == 1037,
            'Expected174 completed rounds /1037 trials')
    require(state['best_training_selected_candidate']['directory'] == 'monthly_local_linear70', 'Incumbent changed')
    require(state['completed'][-1]['directory'] == PRIOR.name, 'Round176 must be the completed predecessor')


def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_BREAKPOINT177_' + tag + '_' + expected + '.json')
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
    names = ('test_monthly_breakpoint177_host.py', 'test_monthly_breakpoint177_kernel.py',
             'test_monthly_breakpoint177_independent.py')
    modules = [module('test_breakpoint177_' + str(i), ROOT / 'work' / name) for i, name in enumerate(names)]
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
    predecessor = read(PRIOR / 'COMPLETION.json')
    require(state['completed'][-1]['completion']['sha256'] == sha(PRIOR / 'COMPLETION.json') and
            predecessor['result_completion_sha256'] == sha(PRIOR / 'results/COMPLETION.json'),
            'Predecessor differs from registered completion')
    require(predecessor['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            predecessor['plan_sha256'] == sha(PRIOR / 'PLAN.json'), 'Predecessor not verified')
    validation = TEST_RECEIPT; t = read(validation)
    require(t['passed'] is True and t['synthetic_only'] is True and t['runtime'] == ev.runtime(), 'Synthetic tests required')
    require(t['code_sha256'] == {str(p): sha(p) for p in code_files()}, 'Code changed since tests')
    pins = dict(read(PRIOR / 'PLAN.json')['input_sha256'])
    raw = read(LOCAL / 'PLAN.json')['raw_source_hashes']
    bind(pins, CORE_ADAPTER_PATH, CORE_ADAPTER_SHA)
    bind(pins, PATH_ADAPTER_PATH, PATH_ADAPTER_SHA)
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
    plan = dict(schema='monthly_breakpoint177_v1', created_utc=now(), runtime=ev.runtime(),
        contract=contract(), input_sha256=pins, raw_source_hashes=raw,
        calendar_source_hashes=calendar_sources, prior_central_sha256=hashlib.sha256(before).hexdigest(),
        actual_new_feature_evaluations_before_registration=0, actual_new_model_fits_before_registration=0,
        actual_descriptive_breakpoint_estimates_before_registration=0,
        actual_new_returns_before_registration=0, model_fit_accounting=MODEL_FIT_ACCOUNTING,
        protected_files=len(protected))
    dump(OUT / 'PLAN.json', plan)
    state.update(current=OUT.name, current_status='registered_before_new_breakpoint_features',
        registered_queue=[dict(directory=OUT.name, plan_sha256=sha(OUT / 'PLAN.json'))],
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Prepare fixed within-formation breakpoint and midpoint control; select only on first88 before later34 diagnostics.')
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
    return p


def core_ranks_and_deltas(values, arrays, n):
    require(set(values) == {'baseline', 'candidate', 'control', 'available'}, 'Feature keys changed')
    available = np.asarray(values['available'])
    shape = available.shape
    require(len(shape) == 2 and shape[1] > 0 and type(n) is int and 0 <= n <= shape[0], 'Invalid feature prefix')
    require(available.dtype == np.dtype(bool), 'Boolean mask required')
    a = available[:n]
    raw = {}
    for key in ('baseline', 'candidate', 'control'):
        data = np.asarray(values[key])
        require(data.shape == shape, 'Invalid raw feature shape')
        raw[key] = np.asarray(data[:n], float)
        require(not np.isinf(raw[key]).any(), 'Infinite feature value')
    base_finite = np.isfinite(raw['baseline'])
    require(not np.any(a & ~base_finite), 'New availability exceeds baseline rank domain')
    require(all(np.array_equal(np.isfinite(raw[key]), a) for key in ('candidate', 'control')), 'Common feature mask invalid')
    calm = np.asarray(arrays['state_calm10'])
    require(calm.shape == (shape[0],) and calm.dtype == np.dtype(bool), 'Original calm flags required')
    calm = calm[:n]
    core = np.asarray(arrays['core'])
    require(core.shape == shape, 'Original core shape differs')
    core = np.asarray(core[:n], float)
    require(np.isfinite(core).all(), 'Original weighted core must be finite')
    weights = np.asarray(arrays['core_weight'])
    if weights.shape == (shape[0],): weights = weights[:, None]
    require(weights.shape in ((shape[0], 1), shape), 'Original core weight shape differs')
    weights = np.broadcast_to(np.asarray(weights[:n], float), (n, shape[1]))
    require(np.array_equal(weights, np.broadcast_to(np.where(calm, .50, .52)[:, None], weights.shape)),
            'Original calm/noncalm core weights changed')
    def ranks(data):
        return pd.DataFrame(data).rank(axis=1, method='average', pct=True).fillna(.5).to_numpy()-.5
    baseline_rank = ranks(raw['baseline'])
    require(np.array_equal(.52*baseline_rank[~calm], core[~calm]), 'Original noncalm core rank parity failed')
    ready = ~calm & np.all(~base_finite | a, axis=1)
    if n: ready[0] = False
    transformed, deltas = {}, {}
    for arm, key in RAW.items():
        rank = ranks(raw[key])
        active = ready[:, None] & a
        selected = np.full((n, shape[1]), np.nan)
        selected[active] = rank[active]
        delta = np.zeros((n, shape[1]))
        delta[active] = DOSE*(weights[active]*rank[active]-core[active])
        require(np.max(np.abs(delta), initial=0.) <= .26+1e-15, 'Half-core replacement budget exceeded')
        transformed[arm], deltas[arm] = selected, delta
    return transformed, deltas


def decisions(study, arrays, meta, values, n):
    policy = read(LOCAL / 'INCUMBENT_POLICY.json')
    choices, scores = study.choices(study.configs()[4], arrays, meta, n, policy=policy)
    orders = {NAMES[0]: study.rp.make_orders(choices, scores, meta)}
    ranks, deltas = core_ranks_and_deltas(values, arrays, n)
    adapter = core_adapter()
    for arm in NAMES[1:]:
        pre = study.modified_pre_score(study.configs()[4], arrays, n)
        full_rank = np.full(np.asarray(arrays['core']).shape, np.nan)
        full_rank[:n] = ranks[arm]
        chosen, score = adapter.score_and_choose('incumbent66', DOSE, full_rank,
            arrays, meta, n, pre_score=pre, effects=arrays['effects'], episode_policy=policy,
            boundary_selector=study.ep.select_boundary)
        orders[arm] = study.rp.make_orders(chosen, score, meta)
    return orders, ranks, deltas


def require_feature_audit(check, values):
    for key in ('passed', 'independent_scalar_rebuild', 'exact_common_availability',
                'no_future_or_outcome_inputs', 'audit_checked', 'breakpoint_search_checked',
                'fixed_midpoint_control_checked', 'formation_and_volatility_dates_checked', 'endpoint_fallback_checked',
                'numerical_constant_fallback_checked', 'raw_rank_parity_checked'):
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
                'no_eligibility_or_model_filter', 'noncalm_core_parity_checked', 'baseline_rank_domain_checked',
                'whole_row_readiness_checked', 'calm_rows_neutral'):
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
    import monthly_breakpoint177_kernel as kernel
    import monthly_breakpoint177_independent as independent
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
    require('adj' in source, 'Original adjusted source panel required')
    panel = {'adj': source['adj']}
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
    rank_check = independent.verify_deltas(values, deltas, arrays, dose=DOSE)
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
    dump(dest / 'SOURCE_PARITY.json', dict(raw_source_hashes=raw_hashes, execution_prices=parity))
    dump(dest / 'INDEPENDENT_PREPARE.json', dict(feature=check, delta=rank_check, prefixes=123, feature_prefixes=list(FEATURE_PREFIXES),
        original_orders_exact=True, original_arrays_unchanged=True, original_array_count=27,
        original_core_metadata_and_support_preserved=True, single_original_episode=True, new_model_fits=0,
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
    require(audit.get('original_core_metadata_and_support_preserved') is True and
            audit.get('single_original_episode') is True, 'Core metadata or episode evidence missing')
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
        close_vs_midpoint=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_midpoint=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_breakpoint177_independent as independent
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
    import monthly_breakpoint177_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, within_formation_break_is_not_predictive_validation=True, unchanged_recent_risk_denominator=True, observer_promotion=False))
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
        next_step='Review within-formation breakpoint results before a distinct new hypothesis; preserve frozen observers.')
    require(len(state['completed']) == 175 and sum(r['trials'] for r in state['completed']) == 1040, 'Wrong final counters')
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
