"""Independent prospective runner review; only fabricated financial inputs."""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/monthly_revenue_persistence156_runner_prereview.json'
PROPOSAL = 'work/monthly_revenue_persistence156_proposal.json'
DESIGN = 'work/after155_revenue_experiment_path.json'
SOURCE_PLAN = 'work/after155_revenue_pilot_source_plan.json'
SOURCE_MANIFEST = 'outputs/monthly_revenue_pilot_source156/MANIFEST.json'
CORE = {
    'work/after155_revenue_pilot_factor.py': '63666fad56d1c4b225ecbd3496cd7423bc471ef0054c29663408531ad8a1931a',
    'work/monthly_revenue_persistence156_runner.py': '129cba53a74c3a1c8a1fafb1fe767a91588f56b14c0d9c83add586af558aaa27',
    'work/monthly_revenue_persistence156_lifecycle.py': '4cb3cd9b26791f9c2af8c4e00ed58b4eb08fc48fe403bb6fc66c12238a25c645',
    'work/prepare_revenue_persistence156_proposal.py': '557a0acf266479ee45b8fd814d91a6bf2076e8271f476f6fd91ccf4ef17a3da0',
}
TESTS = {
    'work/test_after155_revenue_pilot_factor.py': '5daf61a5683a2c04e41e214f0e2f1b7c9a155a20af50baa6da947535aaf9f040',
    'work/test_monthly_revenue_persistence156_runner.py': '32377f6b620641d0633c8e7da0729d7b6dd54baf55c634b9e043ea4b1fe5d48f',
    'work/test_monthly_revenue_persistence156_lifecycle.py': '0c771ac28c3b1ddfa591ad170210ecce61d588a90638c745a0e3706316b87ae5',
    'work/test_monthly_revenue_persistence156_proposal.py': '5de228ed33a037ffc7ab1a80cdb8abf5a71c8238de5ce9911fb4d40092a2f4da',
}
INPUTS = {
    PROPOSAL: '18101719da500f23aff007a6249b2c224a99b28829b562f5db2a78484bc58d4b',
    DESIGN: 'a84440879cf3db80782190de8eb7b43c402c25bdb88fa07e1d9b3401bf092806',
    SOURCE_PLAN: '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
    SOURCE_MANIFEST: 'e4b5b71004186b1262128cc3e8b1286ea639da50ccd426daea572e6af61201bf',
    'work/after155_revenue_pilot_output_independent.json': '3e44c700ef288ee3471c9d050648672d89a9290933927112b8f49cdcbb59422d',
}
SEEN = {}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    runtime = {Path(sys.executable).resolve(), Path(r.np.__file__).resolve(), Path(r.pd.__file__).resolve()} if 'r' in globals() else set()
    require((ROOT in p.parents or p in runtime) and p.resolve() == p, 'Nonlocal review input: ' + str(p))
    raw = p.read_bytes()
    h = digest(raw)
    require(expected is None or h == expected, 'Hash mismatch: ' + str(p))
    require(str(p) not in SEEN or SEEN[str(p)] == h, 'Changed input during prereview')
    SEEN[str(p)] = h
    return raw

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item

def independent_measurement():
    return dict(levels='four_consecutive_own_year_originals_oldest_first', latest='h(R3/R2-1)',
        annualized='h((R3/R0)^(1/3)-1)', weakest='h(min(R1/R0-1,R2/R1-1,R3/R2-1))',
        bounded='h(x)=x/(1+abs(x))', common_mask='identical_independently_reviewed_four_original_chain_for_all_three_features',
        unavailable='NaN_with_reason_and_exact_incumbent_score_fallback',
        placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
        source_scope=['MSFT', 'AAPL', 'ABBV'], original_candidate_axis=[122, 49], unchanged_original_arrays=27,
        invalid_latest_blocks_fallback=True, maximum_annual_age_days=550, no_parameter_search=True,
        new_model_fits=0, etf_financial_features=False, monthly_equal_weight_positions=3, intramonth_trades=False)

def independent_selection():
    return dict(total_months=122, training_months=88, early_months=53, recent_months=35, later_months=34,
        walkforward=[[36, 60], [60, 76], [76, 88]], execution_paths=[
            dict(execution='close', cost=0., key='same_close'), dict(execution='open', cost=0., key='next_open_zero'),
            dict(execution='open', cost=.0005, key='next_open_cost')],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05, max_drawdown_degradation=.03,
        max_early_cagr_degradation=.025, max_recent_cagr_degradation=.025, new_model_fits=0,
        intramonth_trades=False, later34_for_selection=False)

class IndependentProspectiveTests(unittest.TestCase):
    def test_contracts_match_independent_frozen_specification(self):
        self.assertEqual(r.measurement_contract(), independent_measurement())
        self.assertEqual(r.selection_contract(), independent_selection())
        self.assertEqual((r.TRAIN, r.MONTHS, r.WIDTH, r.DOSE), (88, 122, 49, .01))

    def test_temporal_minimum_differs_from_latest_and_endpoint(self):
        got = r.factor.measure_levels(['100', '200', '50', '100'])
        self.assertEqual(got[r.HANDS[0]], .5)
        self.assertEqual(got[r.HANDS[1]], 0.)
        self.assertEqual(got[r.HANDS[2]], float(Fraction(-3, 7)))

    def test_fraction_inputs_scale_and_decline(self):
        base = ['0.008', '0.004', '0.002', '0.001']
        before = base[:]
        got = r.factor.measure_levels(base)
        self.assertEqual(base, before)
        self.assertEqual(got, r.factor.measure_levels(['800', '400', '200', '100']))
        for value in got.values():
            self.assertAlmostEqual(value, -1/3, places=14)

    def test_final_proposal_is_validated_before_calculation(self):
        r.validate_proposal(PROPOSAL_DOC)
        for key in ('actual_new_factors', 'actual_new_orders', 'actual_new_returns', 'actual_new_model_fits'):
            self.assertIs(type(PROPOSAL_DOC[key]), int)
            self.assertEqual(PROPOSAL_DOC[key], 0)
        bad = deepcopy(PROPOSAL_DOC)
        bad['measurement']['maximum_annual_age_days'] = 900
        with self.assertRaises(ValueError):
            r.validate_proposal(bad)

    def test_lifecycle_and_runner_agree_on_declared_controls(self):
        def row(name, cagr):
            return dict(name=name, training=dict(cagr=cagr, sharpe=1.5, max_drawdown=-.3),
                        early=dict(cagr=.2), recent=dict(cagr=.4))
        for control in range(3):
            closed = [row(name, .5) for name in r.NAMES]
            opened = deepcopy(closed)
            closed[3]['training']['cagr'] = opened[3]['training']['cagr'] = .6
            self.assertTrue(r.select(closed, opened)['admitted'])
            self.assertEqual(r.select(closed, opened), c.selection(closed, opened))
            closed[control]['training']['cagr'] = .6
            self.assertFalse(r.select(closed, opened)['admitted'])
            self.assertEqual(r.select(closed, opened), c.selection(closed, opened))

    def test_model_and_search_do_not_enter_measurement(self):
        m = r.measurement_contract()
        self.assertEqual(m['new_model_fits'], 0)
        self.assertIs(m['no_parameter_search'], True)
        self.assertIs(m['etf_financial_features'], False)
        self.assertIs(m['intramonth_trades'], False)
        self.assertIs(r.corrected, r.prior.corrected)
        self.assertIs(r.Backend.pre, r.prior.Backend.pre)

def main():
    global r, c, PROPOSAL_DOC
    require(not (ROOT / OUT).exists(), 'Runner prereview is write-once')
    for p, h in {**CORE, **TESTS, **INPUTS}.items():
        read(p, h)
    import monthly_revenue_persistence156_runner as r
    import monthly_revenue_persistence156_lifecycle as c
    PROPOSAL_DOC = json.loads(read(PROPOSAL, INPUTS[PROPOSAL]))
    design = json.loads(read(DESIGN, INPUTS[DESIGN]))
    source_plan = json.loads(read(SOURCE_PLAN, INPUTS[SOURCE_PLAN]))
    require(PROPOSAL_DOC['measurement'] == independent_measurement(), 'Prospective measurement differs')
    require(PROPOSAL_DOC['selection_contract'] == independent_selection(), 'Prospective selection differs')
    require(PROPOSAL_DOC['design'] == dict(path=DESIGN, sha256=INPUTS[DESIGN]), 'Frozen design identity')
    require([x['name'] for x in design['proposed_mechanism']['arms']] == PROPOSAL_DOC['arms'] == list(r.NAMES), 'Original fixed arms')
    require(design['proposed_mechanism']['dose'] == PROPOSAL_DOC['dose'] == .01, 'Original fixed dose')
    require(source_plan['fixed_issuer_scope'] == ['MSFT', 'AAPL', 'ABBV'], 'Fixed source scope')
    require(PROPOSAL_DOC['fixed']['first_training_months'] == 88 and PROPOSAL_DOC['fixed']['execution_paths'] == 12 and PROPOSAL_DOC['fixed']['monthly_orders'] == 488, 'Prospective experiment dimensions')
    for ref in PROPOSAL_DOC['source_inputs'].values():
        read(ref['path'], ref['sha256'])
    # Pin inherited code/runtime, without creating a Backend or opening prices.
    for path, h in r.source_files(ROOT).items():
        read(path, h)
    read('work/complete_monthly_same_report145.py')
    for p in ('AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        read(p)
    forbidden = ('DATA.npz', 'SCORES.npz', 'PRETRAIN_ORDERS.json', 'CACHE.json', 'RESULTS.json', 'FROZEN.json')
    study = ROOT / 'outputs/monthly_revenue_persistence156'
    require(not any((study / name).exists() for name in forbidden), 'Actual calculation occurred before prospective review')
    outcomes = []
    for index, path in enumerate(TESTS):
        item = module(path, 'prospective_published_tests_%s' % index)
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(item))
        require(result.wasSuccessful(), log.getvalue())
        outcomes.append(dict(path=path, tests=result.testsRun, passed=True, output=log.getvalue()))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentProspectiveTests))
    require(result.wasSuccessful(), log.getvalue())
    outcomes.append(dict(path=str(Path(__file__).relative_to(ROOT)), tests=result.testsRun, passed=True, output=log.getvalue()))
    require(not any((study / name).exists() for name in forbidden), 'Synthetic tests created actual artifacts')
    self_path = str(Path(__file__).resolve())
    read(self_path)
    for path, h in list(SEEN.items()):
        read(path, h)
    receipt = dict(passed=True, round=156, stage='preregistration_runner_review', synthetic_only=True,
        proposal_sha256=INPUTS[PROPOSAL], created_utc=datetime.now(timezone.utc).isoformat(),
        actual_factors_calculated=0, actual_returns_calculated=0, actual_orders_calculated=0, actual_model_fits=0,
        source_hashes={self_path: SEEN[self_path]},
        implementation_hashes={str(ROOT / p): h for p, h in CORE.items()}, input_hashes=dict(sorted(SEEN.items())),
        synthetic_tests=outcomes, synthetic_test_count=sum(x['tests'] for x in outcomes),
        checks=dict(fixed_four_arms_and_one_primary=True, independently_checked_measurement=True,
            common_source_mask=True, latest_source_and_550_day_rules=True, msft_original_source_policy_preserved=True,
            aapl_and_unresolved_edges_remain_unavailable=True, registered_plan_required_before_actual_calculation=True,
            source_gate_before_parsing_financial_amounts=True, original_27_arrays_and_all_5978_candidates=True,
            exact_missing_or_zero_fallback=True, fixed_point01_correction_before_one_selector=True,
            first88_only_selection=True, three_prespecified_walkforward_folds=True, later34_diagnostic_only=True,
            independent_prepare_required_before_selection_and_diagnostics=True,
            twelve_paths_and_488_orders=True, immutable_phase_artifacts=True, hash_bound_prereviews_and_reviewers=True),
        findings=[], limitations=['This prospective review used only fabricated factor inputs, synthetic orders and mocked replay returns.',
            'The known sparse source coverage has no usable early53 pair; a passing source floor does not establish broad representativeness.',
            'Actual feature/order preparation and all12 outcome paths still require their separately registered independent audits.',
            'No source acquisition, accounting or organic-growth certification is added by this runner review.'])
    raw = (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    with (ROOT / OUT).open('xb') as f:
        f.write(raw)
    print(json.dumps(dict(path=OUT, sha256=digest(raw), passed=True, synthetic_test_count=receipt['synthetic_test_count'],
                          proposal_sha256=INPUTS[PROPOSAL], actual_factors_calculated=0, actual_returns_calculated=0)))

if __name__ == '__main__':
    main()
