"""Independent pure-factor audit; all numerical inputs are fabricated.

No real source value, price, return, portfolio, or registered strategy is read.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import random
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = 'work/after155_revenue_pilot_factor.py'
TEST = 'work/test_after155_revenue_pilot_factor.py'
DESIGN = 'work/after155_revenue_experiment_path.json'
SELF = 'work/after155_revenue_factor_independent.py'
OUT = 'work/after155_revenue_factor_independent.json'
EXPECTED = {
    CODE: '63666fad56d1c4b225ecbd3496cd7423bc471ef0054c29663408531ad8a1931a',
    TEST: '5daf61a5683a2c04e41e214f0e2f1b7c9a155a20af50baa6da947535aaf9f040',
    DESIGN: 'a84440879cf3db80782190de8eb7b43c402c25bdb88fa07e1d9b3401bf092806',
    'work/monthly_stock_comp149_runner.py':
        'f3c6264afca34f211acf0b2a68a548657c6565cb6d1ac77060eb25afc17788de',
    'work/monthly_stock_comp149_lifecycle.py':
        '6adf39b07f4796cb2d255ac3d47bfb862b579ec2c58512b386d33d101fffbaea',
}
KEYS = ('latest_revenue_growth', 'annualized_total_revenue_growth',
        'weakest_annual_revenue_growth')
MEASURE = None
DESIGN_DOC = None


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def integer_decimal(value):
    """Independent base-10 parsing, without the producer's Decimal conversion."""
    match = re.fullmatch(r'([+]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?', value)
    if not match:
        raise ValueError('Synthetic oracle requires a positive decimal literal')
    _, whole, fractional, exponent = match.groups()
    fractional = fractional or ''
    exponent = int(exponent or 0) - len(fractional)
    coefficient = int(whole + fractional)
    if coefficient <= 0:
        raise ValueError('Positive synthetic level required')
    return Fraction(coefficient * 10 ** max(exponent, 0), 10 ** max(-exponent, 0))


def cube_root_newton(value):
    """No fractional exponent or logarithm: Newton iteration at 90 digits."""
    x = Decimal(10) ** (value.adjusted() // 3)
    for _ in range(160):
        updated = (2 * x + value / (x * x)) / 3
        if updated == x:
            break
        x = updated
    return x


def oracle(levels):
    original = [integer_decimal(value) for value in levels]
    pairs = list(zip(original, original[1:]))
    weakest_a, weakest_b = min(pairs, key=lambda pair: pair[1] / pair[0])
    a, b = pairs[-1]
    # Algebraically reduced h(b/a-1), not the producer's growth/h helpers.
    latest = float((b - a) / (a + abs(b - a)))
    weakest = float((weakest_b - weakest_a) /
                    (weakest_a + abs(weakest_b - weakest_a)))
    total = original[-1] / original[0]
    with localcontext() as context:
        context.prec = 90
        ratio = Decimal(total.numerator) / Decimal(total.denominator)
        rate = cube_root_newton(ratio) - 1
        annualized = float(rate / (1 + abs(rate)))
    return dict(zip(KEYS, (latest, annualized, weakest)))


class IndependentFactorTests(unittest.TestCase):
    def assertOracle(self, levels):
        result, expected = MEASURE(levels), oracle(levels)
        self.assertEqual(set(result), set(KEYS))
        self.assertEqual(result[KEYS[0]], expected[KEYS[0]])
        self.assertEqual(result[KEYS[2]], expected[KEYS[2]])
        self.assertAlmostEqual(result[KEYS[1]], expected[KEYS[1]], delta=2e-15)
        self.assertTrue(all(math.isfinite(value) and -.5 <= value <= 1 for value in result.values()))
        return result

    def test_prospective_design_literal_formulas_arms_and_fixed_dose(self):
        mechanism = DESIGN_DOC['proposed_mechanism']
        self.assertEqual(mechanism['annual_growth'], 'gk=Rk/R(k-1)-1 for k=1,2,3')
        self.assertEqual(mechanism['bounded_transform'], 'h(x)=x/(1+abs(x))')
        self.assertEqual(mechanism['dose'], .01)
        self.assertEqual([arm['name'] for arm in mechanism['arms']], [
            'incumbent_local67', 'matched_latest_revenue_growth',
            'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth'])
        self.assertEqual([arm['feature'] for arm in mechanism['arms']], [
            None, 'h(g3)', 'h((R3/R0)^(1/3)-1)', 'h(min(g1,g2,g3))'])
        self.assertTrue(mechanism['arms'][-1]['primary'])
        self.assertEqual(DESIGN_DOC['status'], 'PROPOSED_PATH_NOT_STRATEGY_REGISTRATION')

    def test_flat_levels_zero_all_measures(self):
        self.assertEqual(self.assertOracle(['7.25'] * 4), dict.fromkeys(KEYS, 0.))

    def test_constant_doubling_is_exact_half_after_transform(self):
        self.assertEqual(self.assertOracle(['7', '14', '28', '56']), dict.fromkeys(KEYS, .5))

    def test_constant_halving_is_negative_one_third(self):
        for result in self.assertOracle(['56', '28', '14', '7']).values():
            self.assertAlmostEqual(result, -1 / 3, delta=2e-15)

    def test_known_geometric_growth_is_annualized_not_total(self):
        result = self.assertOracle(['8', '18', '36', '64'])
        self.assertAlmostEqual(result[KEYS[1]], .5, delta=2e-15)
        self.assertNotAlmostEqual(result[KEYS[1]], 7 / 8)

    def test_geometric_rate_is_not_mean_annual_growth(self):
        result = self.assertOracle(['100', '200', '100', '400'])
        arithmetic = (1 - .5 + 3) / 3
        self.assertNotAlmostEqual(result[KEYS[1]], arithmetic / (1 + arithmetic))

    def test_latest_is_last_adjacent_pair(self):
        result = self.assertOracle(['100', '1000', '50', '100'])
        self.assertEqual(result[KEYS[0]], .5)
        self.assertEqual(result[KEYS[1]], 0.)
        self.assertEqual(result[KEYS[2]], -19 / 39)

    def test_first_growth_can_be_weakest(self):
        result = self.assertOracle(['100', '50', '100', '400'])
        self.assertEqual(result[KEYS[2]], -1 / 3)
        self.assertEqual(result[KEYS[0]], .75)

    def test_middle_growth_can_be_weakest(self):
        result = self.assertOracle(['100', '200', '100', '200'])
        self.assertEqual(result[KEYS[2]], -1 / 3)
        self.assertEqual(result[KEYS[0]], .5)

    def test_last_growth_can_be_weakest(self):
        result = self.assertOracle(['100', '200', '400', '200'])
        self.assertEqual(result[KEYS[2]], result[KEYS[0]])

    def test_input_order_is_not_sorted_or_reversed(self):
        levels = ['100', '200', '50', '400']
        result = self.assertOracle(levels)
        self.assertNotEqual(result, MEASURE(sorted(levels, key=Decimal)))
        self.assertNotEqual(result, MEASURE(list(reversed(levels))))

    def test_same_endpoints_different_path_keeps_only_annualized_rate(self):
        a = self.assertOracle(['100', '150', '300', '400'])
        b = self.assertOracle(['100', '50', '100', '400'])
        self.assertEqual(a[KEYS[1]], b[KEYS[1]])
        self.assertNotEqual(a[KEYS[2]], b[KEYS[2]])
        self.assertNotEqual(a[KEYS[0]], b[KEYS[0]])

    def test_recovery_does_not_hide_earlier_decline(self):
        result = self.assertOracle(['100', '25', '100', '200'])
        self.assertGreater(result[KEYS[0]], 0)
        self.assertGreater(result[KEYS[1]], 0)
        self.assertLess(result[KEYS[2]], 0)

    def test_tiny_exact_positive_change_does_not_float_round_to_flat(self):
        result = self.assertOracle(['1000000000000000000', '1000000000000000001',
                                    '1000000000000000002', '1000000000000000003'])
        self.assertTrue(all(value > 0 for value in result.values()))

    def test_decimal_fractional_values_preserve_exact_growth(self):
        self.assertOracle(['0.12345', '0.99991', '0.05753', '0.24117'])

    def test_common_integer_and_fractional_scales_are_invariant(self):
        original = ['17.125', '14.25', '28.5', '30.375']
        expected = self.assertOracle(original)
        for scale in ('0.000001', '1000000', '7.5', '1e90', '1e-90'):
            with self.subTest(scale=scale), localcontext() as context:
                context.prec = 120
                scaled = [str(Decimal(value) * Decimal(scale)) for value in original]
                self.assertEqual(self.assertOracle(scaled), expected)

    def test_extreme_finite_ratios_remain_bounded(self):
        for levels in (['1e-600', '1', '1e300', '1e600'],
                       ['1e600', '1e300', '1', '1e-600'],
                       ['1e-300', '1e300', '1e-300', '1e300']):
            with self.subTest(levels=levels):
                self.assertOracle(levels)

    def test_declines_use_absolute_denominator_not_positive_only_formula(self):
        result = self.assertOracle(['100', '50', '25', '12.5'])
        self.assertEqual(result[KEYS[0]], -1 / 3)
        self.assertNotEqual(result[KEYS[0]], -1.)

    def test_weakest_is_no_greater_than_latest_or_annualized(self):
        for levels in (['13', '10', '23', '11'], ['13', '26', '52', '104'],
                       ['19', '12', '7', '3'], ['11', '40', '1', '44']):
            result = self.assertOracle(levels)
            self.assertLessEqual(result[KEYS[2]], result[KEYS[0]] + 2e-15)
            self.assertLessEqual(result[KEYS[2]], result[KEYS[1]] + 2e-15)

    def test_tuple_and_list_have_identical_results(self):
        levels = ['13', '10', '23', '11']
        self.assertEqual(MEASURE(levels), MEASURE(tuple(levels)))

    def test_input_and_decimal_context_not_mutated(self):
        levels = ['13.25', '10.5', '23.01', '11.17']
        before = deepcopy(levels)
        context = getcontext().copy()
        MEASURE(levels)
        self.assertEqual(levels, before)
        self.assertEqual(str(getcontext()), str(context))

    def test_small_external_decimal_precision_does_not_change_results(self):
        levels = ['13.25', '10.5', '23.01', '11.17']
        expected = MEASURE(levels)
        with localcontext() as context:
            context.prec = 7
            self.assertEqual(MEASURE(levels), expected)
            self.assertEqual(context.prec, 7)

    def test_invalid_level_count_and_nonsequence_rejected(self):
        for invalid in (None, {}, '1234', [], ['1'] * 3, ['1'] * 5, iter(['1'] * 4)):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                MEASURE(invalid)

    def test_nonstring_and_nonpositive_or_nonfinite_values_rejected(self):
        for invalid in (True, 1, 1., Decimal('1'), None, '0', '-0', '-1',
                        'NaN', 'sNaN', 'Infinity', '-Infinity'):
            with self.subTest(invalid=repr(invalid)), self.assertRaises((ValueError, ArithmeticError)):
                MEASURE(['1', '2', invalid, '4'])

    def test_malformed_decimal_strings_fail_closed(self):
        for invalid in ('', 'revenue', '1,000', '1.2.3', '0x20'):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, ArithmeticError)):
                MEASURE(['1', '2', invalid, '4'])

    def test_80_fabricated_exact_decimal_paths_match_separate_oracle(self):
        rng = random.Random(1562026)
        for case in range(80):
            levels = ['%d.%04d' % (rng.randrange(1, 1000000), rng.randrange(10000))
                      for _ in range(4)]
            with self.subTest(case=case):
                self.assertOracle(levels)

    def test_module_has_no_io_model_or_performance_dependencies(self):
        tree = ast.parse((ROOT / CODE).read_text())
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        self.assertEqual(modules, {'decimal', 'fractions', 'math'})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {'open', 'exec', 'eval', '__import__', 'input'})


def main():
    global DESIGN_DOC, MEASURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-review', action='store_true')
    args = parser.parse_args()
    before = {path: digest(path) for path in EXPECTED}
    if before != EXPECTED:
        raise ValueError('Reviewed factor/test/design/template bytes changed')
    DESIGN_DOC = json.loads((ROOT / DESIGN).read_text())
    module = importlib.import_module('after155_revenue_pilot_factor')
    if module.NAMES != KEYS:
        raise ValueError('Unexpected factor output names')
    MEASURE = module.measure_levels
    producer = unittest.defaultTestLoader.loadTestsFromModule(
        importlib.import_module('test_after155_revenue_pilot_factor'))
    independent = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentFactorTests)
    counts = {'producer': producer.countTestCases(), 'independent': independent.countTestCases()}
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite([producer, independent]))
    print(stream.getvalue())
    after = {path: digest(path) for path in EXPECTED}
    if before != after:
        raise ValueError('Reviewed bytes changed during tests')
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PURE_REVENUE_FACTOR_REVIEW_PASSED' if result.wasSuccessful() else 'PURE_REVENUE_FACTOR_REVIEW_FAILED',
        'passed': result.wasSuccessful(), 'synthetic_only': True,
        'scope': 'Three bounded four-original-level measures under the fixed proposed mechanism; no source admission or strategy registration.',
        'input_hashes': dict(before, **{path: digest(path) for path in (
            SELF, 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb',
            'work/monthly_forward_guidance132_runner.py', 'work/monthly_same_report145_runner.py',
            'work/monthly_endpoint155_lifecycle.py')}),
        'bindings_unchanged_before_after': before == after,
        'reviewer_independence': 'Only the independent harness/report were written for this audit. The oracle uses its own decimal-literal integer parser, reduced rational h formula, and 90-digit Newton cube-root iteration; no producer helper is used.',
        'checks': ['oldest-first order retained', 'latest uses last adjacent pair',
                   'weakest examines all three adjacent growths',
                   'annualized rate uses the cube root of last/first, not arithmetic mean or cumulative growth',
                   'h uses x/(1+abs(x)) for increases and declines',
                   'common positive scales cancel exactly', 'input and external Decimal context not mutated',
                   'invalid inputs fail closed', 'no I/O, model, price, or return dependencies'],
        'test_execution': dict(counts, total=result.testsRun, seeded_formula_cases=80,
            failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful(),
            command='./run_local.sh work/after155_revenue_factor_independent.py --write-review',
            output=stream.getvalue(), offline_wrapper_unchanged=True),
        'integration_requirements': [
            'Caller independently validates four consecutive original fiscal years, oldest-first ordering, own-year identity, common scale/unit class, same-release witnesses and cutoff availability. measure_levels accepts only four numeric strings and cannot authenticate that metadata.',
            'Build all three measures using the identical admitted chain mask on the complete original 122x49 axis. Hold/absent/ETF rows remain NaN with explicit reasons.',
            'Only after a separately frozen strategy PLAN may the caller add fixed 0.01*h to original eligible pre-scores, invoke the original episode selector once, and compute orders or returns.',
            'Malformed numeric literals currently raise Decimal arithmetic errors; callers must fail closed rather than discard or replace those rows.',
        ],
        'reusable_integration_architecture': {
            'approach': 'Use small explicit revenue adapters over stable scoring, accounting, and file-binding helpers. Keep frozen149/145/132 modules byte-identical.',
            'scoring': 'Call monthly_stock_comp149_runner.corrected unchanged: fixed .01 dose, original eligibility, bounded continuous h, ETF exclusion and one original selector. Write a small new choose/panels using revenue arm names and exact same-mask/prefix/fallback checks.',
            'backend': 'Subclass149.Backend for its prefix-safe pre() and inherited145.open_training(names=...). Override149 training/diagnostics loops only because they capture149 NAMES. Retain the same folds, replay engine, execution modes and costs.',
            'selection': 'Implement the small149 select loop with revenue NAMES, calling145.guard for its precise finite-value, CAGR epsilon and degradation bounds. Reuse149.selection_contract().',
            'runner': 'A new thin132.Runner subclass can reuse path/verify/bound_path/write/refresh/cache. Supply revenue-specific PLAN constructor, source closure validator and feature preparation; retain phase gates and independent preparation receipt checks.',
            'lifecycle': 'Use the current predecessor and prospective receipts in a small new lifecycle; reuse existing Reader/hash closure and generic state lock/publication helpers. Carry over149 sequence and 12-path/488-order completion scope, but not its44-event31/13 SBC constants or146 subset policy.',
            'independence': 'New preparation/outcome reviewers must not import the new producer runner/factor; reuse only prior independent replay infrastructure and new independent revenue oracles.',
            'code_references': [
                'work/monthly_stock_comp149_runner.py:36 (selection contract)',
                'work/monthly_stock_comp149_runner.py:170 (four-arm select)',
                'work/monthly_stock_comp149_runner.py:181 (corrected pre-score)',
                'work/monthly_stock_comp149_runner.py:217 (panels/prefix/fallback)',
                'work/monthly_stock_comp149_runner.py:247 (prefix-safe backend)',
                'work/monthly_stock_comp149_runner.py:439 (phase sequence)',
                'work/monthly_stock_comp149_lifecycle.py:268 (closure audit)',
                'work/monthly_forward_guidance132_runner.py:373 (generic binding helpers)',
                'work/monthly_same_report145_runner.py:107 (selection guard)',
                'work/monthly_same_report145_runner.py:140 (named open training)',
            ],
        },
        'activity_counts': {'actual_financial_values_read': 0, 'actual_factors_computed': 0,
            'actual_orders_computed': 0, 'actual_returns_computed': 0, 'strategy_registrations': 0,
            'frozen_files_changed': 0, 'observer_changes': 0},
        'files_written_by_this_review': [SELF, OUT],
    }
    if args.write_review:
        with (ROOT / OUT).open('x') as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun, 'counts': counts}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
