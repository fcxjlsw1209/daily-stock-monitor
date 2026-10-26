"""Independent synthetic checks of the pure chain gate; no actual source evaluation."""
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from decimal import localcontext
import copy
import hashlib
import io
import json
import sys
import unittest

import after155_revenue_chain as candidate

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'work/after155_revenue_chain.py': 'b7921dc1d20b6ccf602a1a8e14e3544d2b8e41bba86d4843b3dc0f445ed2b83c',
    'work/test_after155_revenue_chain.py': '3490f5c0c90b165f63ebb10fa38967754934de0cde6613dae895055eeffce970',
    'work/after155_revenue_chain_policy_review.json': '8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36',
    'work/after155_revenue_chain_implementation_review.json': '7fca9eccc92ffbc97fe71e22a0b6866bcf534932feef7498639fc7913e815843',
}
PASS = 'checked_reported_total_chain_with_declared_source_scope'
HOLD = 'unresolved_source_or_comparability'


def clock(day):
    return int(datetime.fromisoformat(day + 'T23:59:00+00:00').timestamp()) * 1000000000


def fabricated():
    records, edges = [], []
    for y in range(2015, 2022):
        known = clock(f'{y}-07-29')
        records.append(dict(report_id=f'synthetic_{y}', known_ns=known, fiscal_year=y,
            source_checked=True, issuer='MSFT', currency='USD',
            measure='consolidated_gaap_total_revenue',
            period_start=f'{y-1}-07-01', period_end=f'{y}-06-30',
            current='137', scale='1000000', comparative='137',
            comparative_scale='1000000', comparative_currency='USD',
            comparative_measure='consolidated_gaap_total_revenue',
            comparative_source_report_id=f'synthetic_{y}',
            comparative_period_start=f'{y-2}-07-01', comparative_period_end=f'{y-1}-06-30'))
        if y > 2015:
            edges.append(dict(edge_id=f'synthetic_edge_{y-1}_{y}', known_ns=known,
                earlier_fiscal_year=y-1, later_fiscal_year=y,
                earlier_report_id=f'synthetic_{y-1}', later_report_id=f'synthetic_{y}',
                evidence_known_ns=[clock(f'{y-1}-07-29'), known],
                status='checked_reported_total_edge'))
    return records, edges


class IndependentCases(unittest.TestCase):
    def setUp(self):
        self.r, self.e = fabricated()

    def run_gate(self, day='2021-08-02', cutoff=None):
        before = copy.deepcopy((self.r, self.e))
        out = candidate.evaluate(self.r, self.e, clock(day) if cutoff is None else cutoff, day)
        self.assertEqual((self.r, self.e), before, 'Evaluator mutated supplied evidence')
        self.assertIs(out['factor_computed'], False)
        return out

    def reason(self, expected, **kwargs):
        out = self.run_gate(**kwargs)
        self.assertEqual(out['reasons'], [expected])
        self.assertNotEqual(out['status'], PASS)

    def test_01_exact_four_originals(self):
        out = self.run_gate()
        self.assertEqual(out['status'], PASS)
        self.assertEqual(out['report_ids'], [f'synthetic_{y}' for y in range(2018, 2022)])
        self.assertEqual(len(out['edge_ids']), 3)

    def test_02_future_latest_not_selected(self):
        out = self.run_gate(day='2020-08-02')
        self.assertEqual(out['status'], PASS)
        self.assertEqual(out['latest_report_id'], 'synthetic_2020')
        self.assertNotIn('synthetic_2021', out['report_ids'])

    def test_03_latest_invalid_no_fallback(self):
        self.r[-1]['source_checked'] = False
        self.reason('latest_source_not_checked')

    def test_04_missing_original_no_comparative_replacement(self):
        self.r = [r for r in self.r if r['fiscal_year'] != 2019]
        self.reason('missing_own_year_2019')

    def test_05_duplicate_latest(self):
        self.r.append(copy.deepcopy(self.r[-1]))
        self.reason('conflicting_latest_release')

    def test_06_duplicate_historical_original(self):
        self.r.append(copy.deepcopy(self.r[-2]))
        self.reason('conflicting_own_year_2020')

    def test_07_quarter_member_rejected(self):
        self.r[-2]['period_start'] = '2020-04-01'
        self.reason('invalid_period_2020')

    def test_08_wrong_current_scope(self):
        for key, value in [('issuer', 'OTHER'), ('currency', 'EUR'), ('measure', 'segment_revenue')]:
            with self.subTest(key=key):
                self.r, self.e = fabricated()
                self.r[-2][key] = value
                self.reason('invalid_member_2020')

    def test_09_invalid_current_and_scale(self):
        for key, value in [('current', '0'), ('current', '-1'), ('current', 'NaN'),
                           ('current', 'Infinity'), ('scale', '0'), ('scale', '-1'),
                           ('scale', 'NaN')]:
            with self.subTest(key=key, value=value):
                self.r, self.e = fabricated()
                self.r[-2][key] = value
                self.reason('invalid_value_2020')

    def test_10_witness_scope_and_owner(self):
        cases = [('comparative_period_start', '2020-04-01'),
                 ('comparative_period_end', '2020-03-31'),
                 ('comparative_measure', 'segment_revenue'),
                 ('comparative_currency', 'EUR'),
                 ('comparative_source_report_id', 'other_document')]
        for key, value in cases:
            with self.subTest(key=key):
                self.r, self.e = fabricated()
                self.r[-1][key] = value
                self.reason('incompatible_witness_2021')

    def test_11_both_edge_report_ids(self):
        for key in ['earlier_report_id', 'later_report_id']:
            with self.subTest(key=key):
                self.r, self.e = fabricated()
                self.e[-1][key] = 'wrong_original'
                self.reason('unbound_or_mistimed_edge_2021')

    def test_12_edge_cannot_predate_later_report(self):
        self.e[-1]['known_ns'] = self.r[-1]['known_ns'] - 1
        self.reason('unbound_or_mistimed_edge_2021')

    def test_13_future_edge_ignored(self):
        self.e[-1]['known_ns'] = clock('2022-01-01')
        self.reason('missing_or_conflicting_edge_2021')

    def test_14_future_supporting_note_blocks(self):
        self.e[-1]['evidence_known_ns'].append(clock('2022-01-01'))
        self.reason('unbound_or_mistimed_edge_2021')

    def test_15_empty_supporting_evidence_blocks(self):
        self.e[-1]['evidence_known_ns'] = []
        self.reason('unbound_or_mistimed_edge_2021')

    def test_16_known_break_blocks(self):
        self.e[-2]['status'] = 'blocked_known_basis_break'
        self.assertEqual(self.run_gate()['status'], 'blocked_known_basis_break')

    def test_17_unresolved_edge_blocks(self):
        self.e[-2]['status'] = 'unresolved_accounting_note'
        self.assertEqual(self.run_gate()['status'], HOLD)

    def test_18_changed_witness_not_rebased(self):
        self.r[-1]['comparative'] = '138'
        self.reason('comparative_mismatch_2021')
        self.assertEqual(self.r[-2]['current'], '137')
        self.assertEqual(self.run_gate(day='2020-08-02')['status'], PASS)

    def test_19_different_scales_even_equal_money_rejected(self):
        self.r[-1].update(comparative='137000', comparative_scale='1000')
        self.reason('incompatible_witness_scale_2021')

    def test_20_equal_scale_numeric_not_spelling(self):
        self.r[-1]['comparative_scale'] = '1E6'
        self.assertEqual(self.run_gate()['status'], PASS)

    def test_21_long_unequal_decimals_under_low_precision(self):
        for r in self.r:
            r.update(current='10000000000000000000000000001', scale='1',
                     comparative='10000000000000000000000000001', comparative_scale='1')
        self.r[-1]['comparative'] = '10000000000000000000000000002'
        with localcontext() as context:
            context.prec = 3
            self.reason('comparative_mismatch_2021')

    def test_22_long_equal_decimals_under_low_precision(self):
        for r in self.r:
            r.update(current='10000000000000000000000000001', scale='1',
                     comparative='10000000000000000000000000001', comparative_scale='1')
        with localcontext() as context:
            context.prec = 3
            self.assertEqual(self.run_gate()['status'], PASS)

    def test_23_invalid_witness(self):
        for value in ['0', '-1', 'NaN', 'Infinity']:
            with self.subTest(value=value):
                self.r, self.e = fabricated()
                self.r[-1]['comparative'] = value
                self.reason('invalid_witness_2021')

    def test_24_missing_fields_explicit_hold(self):
        for key in ['current', 'period_end', 'known_ns']:
            with self.subTest(key=key):
                self.r, self.e = fabricated()
                del self.r[-1][key]
                self.reason('malformed_required_source_record')

    def test_25_future_period_rejected(self):
        self.reason('future_latest_period', day='2021-05-01', cutoff=clock('2021-08-02'))

    def test_26_freshness_550_inclusive_551_stale(self):
        period_end = date(2021, 6, 30)
        self.assertEqual(self.run_gate(day=str(period_end + timedelta(days=550)))['status'], PASS)
        self.reason('latest_annual_period_older_than_550_days', day=str(period_end + timedelta(days=551)))

    def test_27_permutation_does_not_change_output(self):
        original = self.run_gate()
        self.r.reverse()
        self.e.reverse()
        self.assertEqual(self.run_gate(), original)

    def test_28_duplicate_known_edges_block(self):
        self.e.append(copy.deepcopy(self.e[-1]))
        self.reason('missing_or_conflicting_edge_2021')

    def test_29_empty_catalog(self):
        self.r = []
        self.reason('no_known_original_annual_report')

    def test_30_cutoff_is_inclusive(self):
        out = self.run_gate(cutoff=self.r[-1]['known_ns'])
        self.assertEqual(out['status'], PASS)
        before = self.run_gate(cutoff=self.r[-1]['known_ns'] - 1)
        self.assertEqual(before['latest_report_id'], 'synthetic_2020')
        self.assertEqual(before['status'], PASS)


def main():
    for rel, sha in PINS.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, 'Changed input: ' + rel
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentCases)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    log = output.getvalue()
    print(log, end='')
    if not result.wasSuccessful():
        raise SystemExit(1)
    for rel, sha in PINS.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, 'Input changed during tests: ' + rel
    script = Path(__file__).resolve()
    hashes = dict(PINS)
    hashes[str(script.relative_to(ROOT))] = hashlib.sha256(script.read_bytes()).hexdigest()
    hashes['run_local.sh'] = hashlib.sha256((ROOT / 'run_local.sh').read_bytes()).hexdigest()
    hashes['local_run.py'] = hashlib.sha256((ROOT / 'local_run.py').read_bytes()).hexdigest()
    logpath = ROOT / 'work/after155_revenue_chain_v2_independent_tests.log'
    receipt = ROOT / 'work/after155_revenue_chain_implementation_review_v2.json'
    data = {
        'stage': 'independent_static_and_synthetic_source_gate_review',
        'passed': True,
        'scope': 'The pure evaluator only; no actual source-chain admission, original-report extraction, financial growth, prices, models, orders, returns or observer actions.',
        'input_hashes': hashes,
        'reviewed_prior_findings': {'R1': 'resolved_in_pure_contract', 'R2': 'resolved_matching_scale_required', 'R3': 'resolved_exact_decimal_comparison_without_multiplication'},
        'checks': {
            'latest_known_before_validity_no_fallback': True,
            'four_own_year_originals_required': True,
            'annual_comparative_period_measure_and_owner_bound': True,
            'edge_both_report_ids_bound': True,
            'all_supplied_evidence_times_no_later_than_edge_and_cutoff': True,
            'future_edges_cannot_validate_current_chain': True,
            'known_break_and_unresolved_edges_block': True,
            'same_scale_and_exact_unrounded_amount_equality': True,
            'negative_age_rejected_550_inclusive_551_stale': True,
            'missing_required_fields_explicit_hold': True,
            'inputs_preserved': True
        },
        'validation': {'independent_unittest_methods': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'all_input_records_fabricated': True, 'producer_test_fixture_imported': False,
            'producer_test_suite_rerun': False,
            'log': str(logpath.relative_to(ROOT)), 'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
            'command': './run_local.sh work/after155_revenue_chain_v2_independent_review.py',
            'python': sys.version, 'actual_source_chains_evaluated': 0,
            'financial_ratios_computed': 0, 'models_fit': 0, 'returns_computed': 0},
        'integration_requirements': [
            'The pinned driver must bind the complete declared catalog, unique original release identities, raw hashes, source/receipt/row locators and independently reviewed status evidence. The pure core does not verify source bytes.',
            'The driver must derive finite integer known_ns values for reports and every evidence note from the pinned calendar, bind decision_date to the actual frozen cutoff, and reject unknown or malformed clocks before use.',
            'evidence_known_ns must enumerate all evidence used by the edge, including later corrections or accounting notes. The pure core checks supplied times but cannot discover an omitted note.',
            'Preserve separately timed corrections and declared archival/first-public-version limitations. No claim of exhaustive worldwide source or correction completeness is introduced.',
            'The writer must publish the complete policy output metadata and explicit holds on the predeclared candidate/date scope; this review is not actual source admission.'
        ],
        'nonblocking_behavior': 'Duplicate candidate records and duplicate edges conservatively hold; no corroborating duplicate is silently selected.',
        'preservation': {'root_code_and_tests_modified': False, 'old_review_modified': False,
            'frozen_inputs_modified': False, 'observer_modified': False}
    }
    with logpath.open('x') as handle:
        handle.write(log)
    with receipt.open('x') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    assert json.loads(receipt.read_text())['validation']['independent_unittest_methods'] == 30
    print(json.dumps({'passed': True, 'tests': result.testsRun,
        'review_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'log_sha256': hashlib.sha256(logpath.read_bytes()).hexdigest(),
        'review_code_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}, sort_keys=True))

if __name__ == '__main__':
    main()
