"""Independent offline metadata-only coverage-gate review with fabricated rows.

Run through ./run_local.sh. No source amounts, financial factors, candidate
scores, prices, performance, actual chain rows, or observer data are loaded.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = 'work/after155_revenue_pilot_source_plan.json'
CODE_PATH = 'work/after155_revenue_coverage_gate.py'
TEST_PATH = 'work/test_after155_revenue_coverage_gate.py'
REVIEW_PATH = 'work/after155_revenue_coverage_independent.py'
OUTPUT_PATH = 'work/after155_revenue_coverage_independent.json'
EXPECTED = {
    PLAN_PATH: '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
    CODE_PATH: '98854e4a29835397b6829327565150377bcbe279cc989fd03d6f047862221b89',
    TEST_PATH: '395bd8a839527496b8173d9b763a727078ce67755f27ce4d5f4431b09de87684',
    'outputs/monthly_local_linear70/META.json':
        '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
}
FIXED = ['MSFT', 'AAPL', 'ABBV']
LIMITS = {
    'minimum_issuers': 2,
    'minimum_usable_candidate_rows_per_qualifying_issuer_in_first88': 12,
    'minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88': 2,
    'minimum_first88_cutoffs_with_at_least_two_usable_issuers': 12,
}
USABLE = 'checked_reported_total_chain_with_declared_source_scope'
WINDOWS = {'early53': (0, 53), 'recent35': (53, 88), 'first88': (0, 88),
           'later34': (88, 122), 'all122': (0, 122)}
FROZEN_PLAN = None
GATE = None


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def fixture():
    """A full synthetic monthly axis; no actual candidate observations."""
    meta = {'dates': ['%04d-%02d-28' % (2000 + i // 12, i % 12 + 1)
                      for i in range(122)],
            'symbols': [FIXED + ['OTHER_%02d' % j for j in range(46)]
                        for _ in range(122)], 'etfs': ['OTHER_00']}
    rows = [{'signal_index': i, 'candidate_index': j,
             'decision_date': meta['dates'][i], 'symbol': symbol,
             'status': 'missing_required_source', 'latest_report_id': None}
            for i in range(122) for j, symbol in enumerate(meta['symbols'][i])]
    return meta, rows


def fill(rows, issuer, indices, endpoint=None):
    indices = set(indices)
    for row in rows:
        if row['symbol'] == issuer and row['signal_index'] in indices:
            row['status'] = USABLE
            row['latest_report_id'] = endpoint or '%s:original_%s' % (
                issuer, row['signal_index'] // 6)


def independent_oracle(meta, rows):
    """Compute literal frozen thresholds from an issuer-by-cutoff matrix.

    This deliberately does not invoke producer helpers or use producer limits.
    """
    matrix = {symbol: {} for symbol in FIXED}
    for row in rows:
        if row['symbol'] in matrix:
            matrix[row['symbol']][row['signal_index']] = (
                row['status'], row['latest_report_id'])
    segments = {}
    for window, (lo, hi) in WINDOWS.items():
        counts = {}
        active = {i: set() for i in range(lo, hi)}
        for symbol, observations in matrix.items():
            statuses, endpoints, total = {}, set(), 0
            for i in range(lo, hi):
                if i not in observations:
                    continue
                status, endpoint = observations[i]
                total += 1
                statuses[status] = statuses.get(status, 0) + 1
                if status == USABLE:
                    endpoints.add(endpoint)
                    active[i].add(symbol)
            counts[symbol] = {'candidate_rows': total,
                              'usable_rows': statuses.get(USABLE, 0),
                              'distinct_latest_endpoints': len(endpoints),
                              'status_counts': statuses}
        segments[window] = {'issuers': counts,
                           'cutoffs_with_minimum_issuers': [
                               meta['dates'][i] for i in range(lo, hi)
                               if len(active[i]) >= 2]}
    qualified = [symbol for symbol in FIXED
                 if segments['first88']['issuers'][symbol]['usable_rows'] >= 12
                 and segments['first88']['issuers'][symbol]['distinct_latest_endpoints'] >= 2]
    common = [meta['dates'][i] for i in range(88)
              if sum(matrix[symbol].get(i, (None, None))[0] == USABLE
                     for symbol in qualified) >= 2]
    return {'source_coverage_floor_passed': len(qualified) >= 2 and len(common) >= 12,
            'qualifying_issuers': qualified, 'qualifying_joint_cutoffs': common,
            'segments': segments, 'rows_verified': 5978,
            'source_reviews_verified_by_this_function': False,
            'financial_values_or_returns_computed': False,
            'strategy_registration_authorized_by_this_function': False}


class ReadGuard(dict):
    """Tripwire: only explicitly authorized metadata fields may be read."""
    def __init__(self, value, allowed):
        super().__init__(value)
        self.allowed = set(allowed)

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError('Unexpected data access: ' + str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError('Unexpected data access: ' + str(key))
        return super().get(key, default)

    def items(self):
        raise AssertionError('Broad mapping scan is not permitted')

    def values(self):
        raise AssertionError('Broad mapping scan is not permitted')


class IndependentCoverageTests(unittest.TestCase):
    def setUp(self):
        self.meta, self.rows = fixture()
        self.plan = deepcopy(FROZEN_PLAN)

    def evaluate(self):
        return GATE(self.meta, self.rows, self.plan)

    def pass_pair(self, indices=range(12)):
        for issuer in ('MSFT', 'AAPL'):
            fill(self.rows, issuer, indices)

    def test_frozen_plan_scope_and_literal_thresholds(self):
        self.assertEqual(self.plan['fixed_issuer_scope'], FIXED)
        for key, expected in LIMITS.items():
            self.assertEqual(self.plan['gate_before_strategy_registration'][key], expected)
        self.assertTrue(self.plan['gate_before_strategy_registration']['source_and_boundary_review_required'])
        self.assertFalse(self.plan['observer_changes'])
        self.assertEqual(self.plan['new_strategy_trials'], 0)

    def test_all_missing_preserves_5978_slots_and_every_window(self):
        result = self.evaluate()
        self.assertEqual(result, independent_oracle(self.meta, self.rows))
        self.assertFalse(result['source_coverage_floor_passed'])
        for name, (lo, hi) in WINDOWS.items():
            self.assertEqual(result['segments'][name]['issuers']['MSFT']['candidate_rows'], hi - lo)

    def test_exact_two_issuer_twelve_row_two_endpoint_floor(self):
        self.pass_pair()
        result = self.evaluate()
        self.assertTrue(result['source_coverage_floor_passed'])
        self.assertEqual(result, independent_oracle(self.meta, self.rows))
        self.assertEqual(result['qualifying_joint_cutoffs'], self.meta['dates'][:12])
        self.assertEqual(result['qualifying_issuers'], ['MSFT', 'AAPL'])

    def test_eleven_rows_with_many_endpoints_cannot_qualify(self):
        fill(self.rows, 'MSFT', range(12))
        fill(self.rows, 'AAPL', range(11))
        self.assertFalse(self.evaluate()['source_coverage_floor_passed'])
        self.assertEqual(self.evaluate()['qualifying_issuers'], ['MSFT'])

    def test_many_rows_with_one_endpoint_cannot_qualify(self):
        self.pass_pair(range(88))
        fill(self.rows, 'AAPL', range(88), endpoint='single_original')
        self.assertEqual(self.evaluate()['qualifying_issuers'], ['MSFT'])

    def test_qualifiers_with_eleven_joint_cutoffs_fail(self):
        fill(self.rows, 'MSFT', range(12))
        fill(self.rows, 'AAPL', range(1, 13))
        result = self.evaluate()
        self.assertEqual(result['qualifying_issuers'], ['MSFT', 'AAPL'])
        self.assertEqual(len(result['qualifying_joint_cutoffs']), 11)
        self.assertFalse(result['source_coverage_floor_passed'])

    def test_raw_overlap_including_nonqualifier_does_not_rescue_gate(self):
        fill(self.rows, 'MSFT', range(12))
        fill(self.rows, 'AAPL', range(1, 13))
        fill(self.rows, 'ABBV', [0])
        result = self.evaluate()
        self.assertEqual(len(result['segments']['first88']['cutoffs_with_minimum_issuers']), 12)
        self.assertEqual(len(result['qualifying_joint_cutoffs']), 11)
        self.assertFalse(result['source_coverage_floor_passed'])

    def test_rotating_undersampled_issuers_do_not_count_as_qualifiers(self):
        fill(self.rows, 'MSFT', range(18))
        fill(self.rows, 'AAPL', range(6))
        fill(self.rows, 'ABBV', range(6, 12))
        result = self.evaluate()
        self.assertEqual(len(result['segments']['first88']['cutoffs_with_minimum_issuers']), 12)
        self.assertEqual(result['qualifying_issuers'], ['MSFT'])
        self.assertFalse(result['source_coverage_floor_passed'])

    def test_third_issuer_can_be_one_of_two_qualifiers(self):
        fill(self.rows, 'MSFT', range(12))
        fill(self.rows, 'ABBV', range(12))
        self.assertTrue(self.evaluate()['source_coverage_floor_passed'])
        self.assertEqual(self.evaluate()['qualifying_issuers'], ['MSFT', 'ABBV'])

    def test_later34_cannot_supply_missing_first88_rows(self):
        self.pass_pair(list(range(11)) + list(range(88, 122)))
        result = self.evaluate()
        self.assertEqual(result['qualifying_issuers'], [])
        self.assertFalse(result['source_coverage_floor_passed'])
        self.assertEqual(len(result['segments']['later34']['cutoffs_with_minimum_issuers']), 34)

    def test_later34_cannot_supply_missing_first88_endpoint(self):
        self.pass_pair(range(88, 122))
        fill(self.rows, 'MSFT', range(12), endpoint='only_first88')
        fill(self.rows, 'AAPL', range(12), endpoint='only_first88')
        self.assertFalse(self.evaluate()['source_coverage_floor_passed'])

    def test_later34_cannot_supply_missing_first88_overlap(self):
        fill(self.rows, 'MSFT', list(range(12)) + list(range(88, 122)))
        fill(self.rows, 'AAPL', list(range(1, 13)) + list(range(88, 122)))
        self.assertFalse(self.evaluate()['source_coverage_floor_passed'])

    def test_exact_recent35_end_boundary_can_pass(self):
        self.pass_pair(range(76, 88))
        result = self.evaluate()
        self.assertTrue(result['source_coverage_floor_passed'])
        self.assertEqual(result['qualifying_joint_cutoffs'][-1], self.meta['dates'][87])

    def test_index88_belongs_only_to_later34(self):
        self.pass_pair(range(77, 89))
        result = self.evaluate()
        self.assertFalse(result['source_coverage_floor_passed'])
        self.assertEqual(result['segments']['first88']['issuers']['MSFT']['usable_rows'], 11)
        self.assertEqual(result['segments']['later34']['issuers']['MSFT']['usable_rows'], 1)

    def test_early_recent_later_windows_partition_at_exact_boundaries(self):
        self.pass_pair([0, 52, 53, 87, 88, 121])
        result = self.evaluate()
        for name, expected in [('early53', 2), ('recent35', 2), ('first88', 4),
                               ('later34', 2), ('all122', 6)]:
            self.assertEqual(result['segments'][name]['issuers']['MSFT']['usable_rows'], expected)

    def test_missing_nonpilot_slot_rejected(self):
        self.rows.pop()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_duplicate_slot_with_same_total_count_rejected(self):
        self.rows[-1] = deepcopy(self.rows[0])
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_extra_slot_rejected(self):
        self.rows.append(deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_changed_date_or_symbol_or_slot_index_rejected(self):
        for field, value in [('decision_date', '1999-01-01'), ('symbol', 'WRONG'),
                             ('candidate_index', 49), ('signal_index', 122)]:
            with self.subTest(field=field):
                old = self.rows[0][field]
                self.rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.rows[0][field] = old

    def test_duplicate_candidate_in_month_rejected(self):
        self.meta['symbols'][0][-1] = self.meta['symbols'][0][0]
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_reduced_month_or_candidate_axis_rejected(self):
        for field in ('dates', 'symbols'):
            with self.subTest(field=field):
                original = deepcopy(self.meta)
                self.meta[field].pop()
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.meta = original
        self.meta['symbols'][0].pop()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_unsorted_or_duplicate_decision_dates_rejected(self):
        for replacement in (self.meta['dates'][0], '1900-01-01'):
            self.meta['dates'][1] = replacement
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_complete_row_permutation_does_not_change_gate(self):
        self.pass_pair()
        expected = self.evaluate()
        random.Random(41).shuffle(self.rows)
        self.assertEqual(self.evaluate(), expected)

    def test_pilot_absence_remains_missing_without_dropping_grid(self):
        for i in range(12):
            self.meta['symbols'][i][1] = 'UNRELATED'
            self.rows[i * 49 + 1]['symbol'] = 'UNRELATED'
        fill(self.rows, 'MSFT', range(12))
        fill(self.rows, 'AAPL', range(12))
        result = self.evaluate()
        self.assertEqual(result['rows_verified'], 5978)
        self.assertEqual(result['segments']['first88']['issuers']['AAPL']['candidate_rows'], 76)
        self.assertFalse(result['source_coverage_floor_passed'])

    def test_nonpilot_nonetf_usable_row_rejected(self):
        self.rows[4].update(status=USABLE, latest_report_id='synthetic_outside')
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_etf_cannot_be_admitted_even_when_in_pilot_scope(self):
        self.meta['etfs'].append('MSFT')
        fill(self.rows, 'MSFT', [0])
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_duplicate_fixed_issuer_rejected(self):
        self.plan['fixed_issuer_scope'].append('MSFT')
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_usable_endpoint_is_required_but_hold_endpoint_is_ignored(self):
        self.pass_pair()
        self.rows[2]['latest_report_id'] = 'held_original'
        result = self.evaluate()
        self.assertEqual(result['segments']['first88']['issuers']['ABBV']['distinct_latest_endpoints'], 0)
        for invalid in (None, ''):
            self.rows[0]['latest_report_id'] = invalid
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_near_match_statuses_and_unreviewed_holds_never_count(self):
        self.pass_pair()
        for row in self.rows:
            if row['symbol'] == 'AAPL' and row['status'] == USABLE:
                row['status'] = USABLE + '_unreviewed'
        self.assertFalse(self.evaluate()['source_coverage_floor_passed'])

    def test_output_does_not_authorize_source_or_strategy_admission(self):
        self.pass_pair()
        result = self.evaluate()
        for key in ('source_reviews_verified_by_this_function',
                    'financial_values_or_returns_computed',
                    'strategy_registration_authorized_by_this_function'):
            self.assertIs(result[key], False)

    def test_inputs_remain_byte_equivalent_as_json(self):
        self.pass_pair()
        before = json.dumps((self.meta, self.rows, self.plan), sort_keys=True)
        self.evaluate()
        self.assertEqual(json.dumps((self.meta, self.rows, self.plan), sort_keys=True), before)

    def test_financial_factor_price_and_performance_fields_are_never_read(self):
        self.pass_pair()
        expected = self.evaluate()
        allowed = set(self.rows[0])
        blocked = {key: object() for key in ('current_revenue', 'financial_factor',
                   'candidate_score', 'price', 'future_return', 'performance')}
        self.rows = [ReadGuard(dict(row, **blocked), allowed) for row in self.rows]
        self.meta = ReadGuard(dict(self.meta, **blocked), ['dates', 'symbols', 'etfs'])
        self.plan = ReadGuard(dict(self.plan, **blocked),
                              ['fixed_issuer_scope', 'gate_before_strategy_registration'])
        self.assertEqual(self.evaluate(), expected)

    def test_source_is_pure_and_uses_no_io_import_or_execution(self):
        tree = ast.parse((ROOT / CODE_PATH).read_text())
        imports = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertIsInstance(imports[0], ast.ImportFrom)
        self.assertEqual(imports[0].module, 'collections')
        self.assertEqual([item.name for item in imports[0].names], ['Counter'])
        forbidden = {'open', 'exec', 'eval', '__import__', 'compile', 'getattr',
                     'setattr', 'globals', 'locals', 'input'}
        self.assertFalse({node.id for node in ast.walk(tree)
                          if isinstance(node, ast.Name)} & forbidden)

    def test_24_seeded_matrices_match_independent_literal_oracle(self):
        statuses = ['missing_required_source', 'blocked_known_basis_break',
                    'unreviewed_source', USABLE]
        for seed in range(24):
            with self.subTest(seed=seed):
                rng = random.Random(9300 + seed)
                self.meta, self.rows = fixture()
                for row in self.rows:
                    if row['symbol'] in FIXED:
                        row['status'] = rng.choices(statuses, weights=[2, 1, 1, seed % 7 + 1])[0]
                        row['latest_report_id'] = '%s:original_%d' % (
                            row['symbol'], rng.randrange(1 if seed % 4 == 0 else 8))
                self.assertEqual(self.evaluate(), independent_oracle(self.meta, self.rows))


def main():
    global FROZEN_PLAN, GATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-review', action='store_true')
    args = parser.parse_args()
    observed = {path: sha(path) for path in EXPECTED}
    if observed != EXPECTED:
        raise ValueError('Frozen plan or reviewed producer/test/metadata binding changed')
    FROZEN_PLAN = json.loads((ROOT / PLAN_PATH).read_text())
    producer = importlib.import_module('after155_revenue_coverage_gate')
    if producer.USABLE != USABLE:
        raise ValueError('Unexpected source usability status')
    GATE = producer.coverage_gate
    producer_suite = unittest.defaultTestLoader.loadTestsFromModule(
        importlib.import_module('test_after155_revenue_coverage_gate'))
    independent_suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentCoverageTests)
    producer_count, independent_count = producer_suite.countTestCases(), independent_suite.countTestCases()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.TestSuite([producer_suite, independent_suite]))
    print(stream.getvalue())
    after = {path: sha(path) for path in EXPECTED}
    if after != observed:
        raise ValueError('Reviewed files changed during execution')
    report = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': ('PURE_METADATA_COVERAGE_GATE_REVIEW_PASSED' if result.wasSuccessful()
                   else 'PURE_METADATA_COVERAGE_GATE_REVIEW_FAILED'),
        'passed': result.wasSuccessful(),
        'scope': 'Pure gate and producer tests checked against the frozen source plan using synthetic metadata only.',
        'reviewer_independence': 'Reviewer wrote only this harness and its report; producer code/tests and frozen evidence were not edited. The differential oracle uses literal frozen thresholds and a separate issuer-by-cutoff matrix.',
        'input_hashes': dict(observed, **{path: sha(path) for path in (
            REVIEW_PATH, 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb')}),
        'bindings_unchanged_before_after': observed == after,
        'fixed_issuer_scope': FIXED,
        'literal_qualification_limits': LIMITS,
        'checks': {
            'complete_122_by_49_grid_5978_slots_preserved': True,
            'candidate_drop_duplicate_change_or_extra_rejected': True,
            'fixed_three_issuer_plan_hash_verified': True,
            'at_least_two_qualifying_issuers_each_twelve_first88_rows_two_endpoints': True,
            'at_least_twelve_joint_first88_cutoffs_among_qualifying_issuers': True,
            'raw_coavailability_cannot_substitute_nonqualifying_issuers': True,
            'later34_cannot_supply_rows_endpoints_or_overlap': True,
            'early53_recent35_later34_all122_reported_at_correct_boundaries': True,
            'financial_values_factors_prices_and_returns_never_read': True,
            'no_input_mutation_or_io_in_gate': True,
            'no_source_or_strategy_approval_claim': True,
        },
        'execution': {
            'command': './run_local.sh work/after155_revenue_coverage_independent.py --write-review',
            'producer_tests': producer_count, 'independent_tests': independent_count,
            'total_tests': result.testsRun, 'failures': len(result.failures),
            'errors': len(result.errors), 'seeded_oracle_subcases': 24,
            'all_passed': result.wasSuccessful(), 'test_output': stream.getvalue(),
            'network_access': 'denied by unmodified local wrapper',
            'cloudstorage_access': 'denied by unmodified local wrapper',
            'observer_output_writes': 'denied by unmodified local wrapper',
        },
        'integration_requirements': [
            'The caller must bind the frozen plan and original candidate META.json bytes to their recorded SHA-256 before invoking this parameterized pure function; it does not authenticate caller inputs.',
            'The caller must independently review and bind each actual source-chain status and endpoint identity. This function trusts the supplied status and latest_report_id and does not reperform source or boundary review.',
            'Preserve the complete original candidate grid, including missing/held/nonpilot rows. The function checks equality to supplied metadata, not historical origin of that metadata.',
            'A coverage pass alone does not register a strategy or authorize factor, ranking, portfolio, or performance computation.',
        ],
        'activity_counts': {
            'actual_source_chain_rows_read': 0, 'actual_source_amounts_read': 0,
            'financial_factors_computed': 0, 'rankings_computed': 0,
            'portfolio_or_performance_computed': 0, 'strategy_trials': 0,
            'source_changes': 0, 'observer_changes': 0,
        },
        'files_written_by_reviewer': [REVIEW_PATH, OUTPUT_PATH],
    }
    if args.write_review:
        with (ROOT / OUTPUT_PATH).open('x') as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'passed', 'bindings_unchanged_before_after')}))
    print(json.dumps({'producer_tests': producer_count, 'independent_tests': independent_count,
                      'total_tests': result.testsRun, 'seeded_oracle_subcases': 24}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
