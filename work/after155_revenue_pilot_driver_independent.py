"""Source-only prereview of the pilot driver; never evaluates the actual grid."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import ast
import hashlib
import importlib.util
import io
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after155_revenue_pilot_driver_independent.json'
DRIVER = 'work/after155_revenue_pilot_prepare.py'
TEST = 'work/test_after155_revenue_pilot_prepare.py'
POLICY = 'work/after155_multi_issuer_chain_policy.json'
PLAN = 'work/after155_revenue_pilot_source_plan.json'
EXPECTED = {
    DRIVER: '19ad848a521bbc1db3829cebe8fdbdcf96562faa122b060793977056a89cb1d0',
    TEST: '2190fc856cdf377b943e8b10a1765c3db02c2570bf71c16138065873e0e24520',
    POLICY: '25820e3cdb6273a9bedadfc8253a415bc2f6515b3d61e9d59ab7a465cdc0acc0',
    PLAN: '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
}
SHAS = {
    'msft': '8dd79daa2891cd093fb16dd01015be92e59d38c9545d4ff04a3e45826ae4dadc',
    'abbv_source': '0fc8b40786cfb3df532d94700ac46a7d78ae416d1f39a7a3bd0c3faee08a8f9d',
    'abbv_accounting': '2cadcda44b07096ac02ad6c60ac81ef881c2a33ee840b4d6aa5492964efb22be',
    'chain': '1aabe00960b8c09c3794ad2d0bc9de2d03159fadaddb7acb211a07ea32232e8c',
    'gate': 'c260e8ba0cbee0e637131c1c4cf04ab02f08b11ab8d539e2e02249f6a1da719c',
}
HASHES = {}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve() == p and ROOT in p.parents, 'Nonlocal review input')
    key = str(p.relative_to(ROOT))
    raw = p.read_bytes()
    h = digest(raw)
    require(expected is None or h == expected, 'Input SHA mismatch: ' + key)
    require(key not in HASHES or HASHES[key] == h, 'Input changed: ' + key)
    HASHES[key] = h
    return raw

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class SyntheticClose:
    def __init__(self, day):
        self.day = day
        self.value = int(datetime.fromisoformat(day + 'T21:00:00+00:00').timestamp()) * 10**9

    def isoformat(self):
        return self.day + 'T21:00:00+00:00'

    def tz_convert(self, zone):
        return self

    def date(self):
        return date.fromisoformat(self.day)

class IndependentChecks(unittest.TestCase):
    def normalized(self, extraction=None, source=None, accounting=None):
        return driver.normalize_abbv(extraction or DATA, source or REVIEWS['abbv_source'],
            accounting or REVIEWS['abbv_accounting'], SHAS['abbv_source'], SHAS['abbv_accounting'],
            EXPECTED[POLICY], DAYS, CLOSES)

    def test_original_cells_preserved_and_early_clock_not_rewritten(self):
        original = deepcopy(DATA)
        normalized = self.normalized()
        self.assertEqual(DATA, original)
        self.assertEqual(len(normalized), 13)
        for src, item in zip(DATA['reports'], normalized):
            self.assertEqual(item['current'], src['revenue']['current']['amount_decimal'])
            self.assertEqual(item['comparative'], src['revenue']['comparative']['amount_decimal'])
            self.assertEqual(item['original_publication_clock'], src['publication_clock'])
            self.assertEqual(item['currency_policy_sha256'], EXPECTED[POLICY])
            self.assertIsNone(item['currency_iso'])
            self.assertIs(item['currency_iso_certified'], False)
        self.assertIsNone(normalized[0]['original_publication_clock']['available_session'])
        self.assertEqual(normalized[0]['known_by'], '2014-02-03T21:00:00+00:00')

    def test_swapped_review_year_source_fails(self):
        source = deepcopy(REVIEWS['abbv_source'])
        source['reports'][0]['source_path'] = source['reports'][1]['source_path']
        with self.assertRaisesRegex(ValueError, 'source identity mismatch'):
            self.normalized(source=source)

    def test_explicit_current_year_vintage_mismatch_fails(self):
        data = deepcopy(DATA)
        data['reports'][0]['revenue']['current']['fact_year'] += 1
        with self.assertRaisesRegex(ValueError, 'cell review binding mismatch'):
            self.normalized(extraction=data)

    def test_release_date_cannot_replace_available_close(self):
        source = deepcopy(REVIEWS['abbv_source'])
        source['reports'][0]['independent_available_session'] = source['reports'][0]['release_date']
        with self.assertRaisesRegex(ValueError, 'next-session mismatch'):
            self.normalized(source=source)

    def test_false_accounting_review_not_admitted(self):
        class Rejected(driver.Inputs):
            def load(self, path, expected=None):
                obj = super().load(path, expected)
                if path == driver.REVIEW_PATHS['abbv_accounting']:
                    obj['passed'] = False
                return obj
        with self.assertRaisesRegex(ValueError, 'Independent review not passed: abbv_accounting'):
            driver.review_inputs(Rejected(), SHAS, require_driver=False)

    def test_review_specific_binding_cannot_be_supplied_by_another_review(self):
        class MissingBinding(driver.Inputs):
            def load(self, path, expected=None):
                obj = super().load(path, expected)
                if path == driver.REVIEW_PATHS['chain']:
                    del obj['input_hashes'][PLAN]
                return obj
        with self.assertRaisesRegex(ValueError, 'Review does not bind required input'):
            driver.review_inputs(MissingBinding(), SHAS, require_driver=False)

    def test_bad_review_digest_fails_before_use(self):
        altered = dict(SHAS, msft='0' * 64)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            driver.review_inputs(driver.Inputs(), altered, require_driver=False)

    def test_prepare_review_mandatory(self):
        with self.assertRaisesRegex(ValueError, 'review SHA missing: driver'):
            driver.review_inputs(driver.Inputs(), SHAS, require_driver=True)

    def test_actual_source_edge_identities_and_pending_status_survive_normalization(self):
        reports = self.normalized()
        edges = driver.normalize_edges(PROPOSAL, REVIEWS['abbv_accounting'], reports,
            SHAS['abbv_accounting'], HASHES[driver.EDGE_PROPOSAL], EXPECTED[POLICY])
        self.assertEqual(len(edges), 12)
        self.assertEqual(sum(e['status'] == 'checked_reported_total_edge' for e in edges), 11)
        self.assertEqual(edges[4]['status'], 'pending_identified_reclassification_note')
        for edge in edges:
            self.assertEqual(edge['source_review_sha256'], SHAS['abbv_accounting'])
            self.assertEqual(edge['producer_review_sha256'], HASHES[driver.EDGE_PROPOSAL])
        driver.verify_source_objects(INPUTS, reports, edges)

    def test_later_evidence_cannot_be_backdated_to_original_edge(self):
        changed = deepcopy(PROPOSAL)
        changed['edges'][0]['evidence'][0]['known_ns'] = changed['edges'][0]['known_ns'] + 1
        with self.assertRaisesRegex(ValueError, 'note source/clock'):
            driver.normalize_edges(changed, REVIEWS['abbv_accounting'], self.normalized(),
                SHAS['abbv_accounting'], HASHES[driver.EDGE_PROPOSAL], EXPECTED[POLICY])

    def test_independent_edge_report_identity_required(self):
        changed = deepcopy(REVIEWS['abbv_accounting'])
        changed['edges'][0]['later_report_id'] = changed['edges'][1]['later_report_id']
        with self.assertRaisesRegex(ValueError, 'identity/clock mismatch'):
            driver.normalize_edges(PROPOSAL, changed, self.normalized(), SHAS['abbv_accounting'],
                HASHES[driver.EDGE_PROPOSAL], EXPECTED[POLICY])

    def test_full_synthetic_axis_uncensored_catalog_and_untouched_msft_contract(self):
        dates = [(date(2010, 1, 1) + timedelta(days=i)).isoformat() for i in range(122)]
        names = ['MSFT', 'ABBV', 'AAPL', 'ETF'] + ['S%02d' % i for i in range(45)]
        meta = dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['ETF'])
        closes = {day: SyntheticClose(day) for day in dates}
        prior = [dict(signal_index=i, candidate_index=j, symbol=s, decision_date=day,
            decision_cutoff_ns=closes[day].value, status='old_status_%s' % i,
            policy_id='frozen_original_msft_policy', report_ids=['old_original'], edge_ids=['old_edge'],
            original_specific_marker={'keep': i}, factor_computed=False, factor_admitted=False,
            strategy_admitted=False) for i, day in enumerate(dates) for j, s in enumerate(names)]
        preserved = deepcopy(prior)
        # Deliberately invalid latest source must reach the evaluator unchanged.
        reports = [dict(report_id='older', known_ns=1, source_checked=True),
                   dict(report_id='latest_invalid', known_ns=2, source_checked=False)]
        edges = [dict(edge_id='pending', status='pending_identified_note')]
        seen = []
        def evaluator(issuer, observed_reports, observed_edges, cutoff, day):
            self.assertIs(observed_reports, reports)
            self.assertIs(observed_edges, edges)
            self.assertEqual((issuer, cutoff), ('ABBV', closes[day].value))
            seen.append(day)
            return dict(status='unresolved_source_or_comparability', reasons=['latest_source_not_checked'],
                        latest_report_id='latest_invalid', report_ids=[], edge_ids=[], factor_computed=False)
        rows = driver.grid(meta, prior, reports, edges, closes, 'new_policy',
            dict(SHAS, msft_rows='f' * 64), evaluator)
        self.assertEqual(seen, dates)
        self.assertEqual(len(rows), 122 * 49)
        self.assertEqual(prior, preserved)
        for old, row in zip(prior, rows):
            self.assertEqual((old['signal_index'], old['candidate_index'], old['symbol']),
                             (row['signal_index'], row['candidate_index'], row['symbol']))
            if row['symbol'] == 'MSFT':
                self.assertTrue(all(row[k] == v for k, v in old.items()))
            if row['symbol'] == 'AAPL':
                self.assertEqual(row['status'], 'unresolved_source_or_comparability')
                self.assertEqual(row['report_ids'], [])
                self.assertEqual(row['edge_ids'], [])
            self.assertIs(row['factor_computed'], False)
            self.assertIs(row['factor_admitted'], False)
            self.assertIs(row['strategy_admitted'], False)

    def test_usable_row_cannot_contain_a_future_source(self):
        reports = [dict(report_id=str(i), source={'raw': {}}, known_by='x', known_ns=1) for i in range(4)]
        reports[-1]['known_ns'] = 3
        edges = [dict(edge_id=str(i), known_by='x', known_ns=1) for i in range(3)]
        row = dict(status=driver.USABLE, report_ids=[str(i) for i in range(4)],
                   edge_ids=[str(i) for i in range(3)], decision_cutoff_ns=2)
        with self.assertRaisesRegex(ValueError, 'after decision cutoff'):
            driver.attach_provenance(row, reports, edges, {})

def main():
    global driver, INPUTS, REVIEWS, DATA, PROPOSAL, DAYS, CLOSES
    require(not (ROOT / OUT).exists(), 'Independent driver prereview is write-once')
    require(not (ROOT / 'outputs/monthly_revenue_pilot_source156').exists(), 'Actual grid exists before prereview')
    for p, h in EXPECTED.items():
        read(p, h)
    tree = ast.parse(read(DRIVER).decode())
    imports = sorted({x.module for x in ast.walk(tree) if isinstance(x, ast.ImportFrom)} |
                     {a.name for x in ast.walk(tree) if isinstance(x, ast.Import) for a in x.names})
    require(set(imports) <= {'argparse', 'copy', 'datetime', 'hashlib', 'importlib.util', 'json', 'pathlib', 're', 'after155_multi_issuer_chain', 'after155_revenue_coverage_gate'}, 'Unexpected driver dependency')
    driver = load_module(DRIVER, 'reviewed_pilot_driver')
    INPUTS = driver.Inputs()
    INPUTS.read(DRIVER, EXPECTED[DRIVER])
    INPUTS.read(TEST, EXPECTED[TEST])
    REVIEWS, paths = driver.review_inputs(INPUTS, SHAS, require_driver=False)
    for p, h in INPUTS.hashes.items():
        read(p, h)
    DATA = INPUTS.load(driver.EXTRACT)
    PROPOSAL = INPUTS.load(driver.EDGE_PROPOSAL)
    DAYS, CLOSES = driver.calendar(INPUTS)
    published = load_module(TEST, 'published_pilot_driver_tests')
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(published),
                               unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks)])
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    require(result.wasSuccessful(), capture.getvalue())
    INPUTS.unchanged()
    for p, h in INPUTS.hashes.items():
        read(p, h)
    read(str(Path(__file__)))
    for p, h in list(HASHES.items()):
        read(p, h)
    require(not (ROOT / 'outputs/monthly_revenue_pilot_source156').exists(), 'Prereview generated actual output')
    output = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        input_hashes=dict(sorted(HASHES.items())), actual_grid_verified=False,
        scope='Independent code and synthetic-test prereview plus actual source joins; no actual candidate grid evaluated',
        manual_review=dict(source_only_dependencies=imports,
            msft_original_policy_and_row_decisions_preserved=True, aapl_explicit_pending=True,
            fixed_122_by_49_axis_preserved=True, latest_known_catalog_not_censored_by_driver=True,
            source_review_closures_and_specific_role_bindings_required=True,
            driver_and_tests_bound_before_prepare=True, write_once_new_output_directory=True,
            gate_uses_frozen_source_plan_before_any_financial_factor=True,
            finite_named_output_set=['ROWS.json', 'REPORTS.json', 'EDGES.json', 'COVERAGE_GATE.json', 'SUMMARY.json', 'MANIFEST.json']),
        checks=dict(published_tests=19, independent_tests=13, total_tests=result.testsRun,
            actual_source_reports_normalized=13, actual_edges_normalized=12,
            actual_pending_2017_2018_preserved=True, calendar_rebuilt_from_frozen_sessions=True,
            full_input_closure_verified=True, actual_grid_evaluated=False),
        test_execution=dict(command='./run_local.sh work/after155_revenue_pilot_driver_independent.py',
            output=capture.getvalue(), failures=len(result.failures), errors=len(result.errors)),
        approved_review_sha256s=SHAS,
        remaining_work=['Execute the actual source-only grid with this exact independent review digest.',
                        'Independently audit all actual candidate slots, cutoff decisions and output hashes before later use.'],
        activity_counts=dict(actual_grid_rows_created=0, financial_factors=0, rankings=0, models=0, returns=0, strategies_registered=0, observer_writes=0))
    raw = (json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    with (ROOT / OUT).open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(path=OUT, sha256=digest(raw), passed=True, tests=result.testsRun,
                          input_files=len(HASHES), actual_grid_evaluated=False)))

if __name__ == '__main__':
    main()
