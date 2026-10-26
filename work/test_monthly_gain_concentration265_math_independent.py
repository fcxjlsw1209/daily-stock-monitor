"""Synthetic-only independent oracle checks; never opens research datasets."""
import ast
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import sys
import unittest

import monthly_gain_concentration265_math_independent as oracle
import monthly_gain_concentration265_kernel as producer


TINY = float.fromhex('0x0.0000000000001p-1022')
MAXIMUM = sys.float_info.max


def prices_from_grosses(grosses, initial=1.0):
    result = [initial]
    for gross in grosses:
        value = result[-1] * gross
        if not math.isfinite(value) or value <= 0:
            raise AssertionError('Bad synthetic fixture')
        result.append(value)
    return result


def two_jump_prices(second_slot=22, second_gross=2.0):
    grosses = [1.0] * 252
    grosses[0] = 2.0
    grosses[second_slot - 1] = second_gross
    return prices_from_grosses(grosses)


def as_fraction(record):
    return Fraction(record['numerator'], record['denominator'])


class IndependentConcentrationTests(unittest.TestCase):
    def assert_same(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, float):
            self.assertEqual(struct.pack('>d', left), struct.pack('>d', right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_same(a, b)
        else:
            self.assertEqual(left, right)

    def compare(self, prices, raw=1.75):
        actual = oracle.measure(prices, raw)
        self.assert_same(actual, producer.measure(prices, raw))
        self.assert_same(oracle.exact_qualities_from_gains(actual['daily_gains']),
                         producer.exact_qualities_from_gains(actual['daily_gains']))
        json.dumps(actual, allow_nan=False)
        return actual

    def reject_both(self, prices, raw=1.0, kind=ValueError):
        for module in (oracle, producer):
            with self.subTest(module=module.__name__):
                with self.assertRaises(kind):
                    module.measure(prices, raw)

    def test_oracle_has_only_standard_library_imports(self):
        tree = ast.parse(Path(oracle.__file__).read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module.split('.')[0])
        self.assertTrue(set(modules) <= {'collections', 'fractions', 'math', 'numbers'})

    def test_all_flat_neutral_with_undefined_concentrations(self):
        result = self.compare([1.0] * 253)
        self.assertTrue(result['no_positive_gains'])
        self.assertEqual(result['exact']['gain_sum'], {'numerator': 0, 'denominator': 1})
        self.assertIsNone(result['exact']['daily_concentration'])
        self.assertIsNone(result['exact']['block_concentration'])
        self.assertEqual(result['daily_quality'], 1.0)
        self.assertEqual(result['block_quality'], 1.0)

    def test_decreasing_path_and_nonpositive_raw_copy(self):
        prices = prices_from_grosses([0.5] * 252)
        for raw in (-3.5, -0.0, 0.0, 2.0):
            result = self.compare(prices, raw)
            for key in ('daily_raw', 'block_raw'):
                self.assertEqual(struct.pack('>d', result[key]), struct.pack('>d', raw))

    def test_equal_positive_daily_masses_exact_one(self):
        result = self.compare(prices_from_grosses([2.0] * 252))
        self.assertEqual(as_fraction(result['exact']['daily_concentration']), Fraction(1, 252))
        self.assertEqual(as_fraction(result['exact']['block_concentration']), Fraction(1, 12))
        self.assertEqual(as_fraction(result['exact']['daily_quality']), 1)
        self.assertEqual(as_fraction(result['exact']['block_quality']), 1)

    def test_single_gain_legitimate_zero_even_with_tiny_original(self):
        prices = [1.0] + [2.0] * 252
        for raw in (2.0, TINY):
            result = self.compare(prices, raw)
            self.assertEqual(result['daily_quality'], 0.0)
            self.assertEqual(result['block_quality'], 0.0)
            self.assertEqual(result['daily_raw'], 0.0)
            self.assertEqual(result['block_raw'], 0.0)

    def test_block_boundary_changes_only_temporal_concentration(self):
        inside = self.compare(two_jump_prices(21))
        outside = self.compare(two_jump_prices(22))
        self.assertEqual(inside['exact']['gain_sum'], outside['exact']['gain_sum'])
        self.assertEqual(inside['exact']['daily_quality'], outside['exact']['daily_quality'])
        self.assertEqual(inside['block_quality'], 0.0)
        self.assertEqual(as_fraction(outside['exact']['block_quality']), Fraction(6, 11))
        self.assertEqual(inside['positive_gain_indices'], [1, 21])
        self.assertEqual(outside['positive_gain_indices'], [1, 22])

    def test_within_block_reordering_is_invariant(self):
        one = [1.0] * 252
        one[0], one[1], one[21] = 2.0, 0.5, 1.5
        two = list(one)
        two[0], two[1] = two[1], two[0]
        left = self.compare(prices_from_grosses(one))
        right = self.compare(prices_from_grosses(two))
        self.assertEqual(left['exact'], right['exact'])
        self.assertEqual(left['exact_block_gain_sums'], right['exact_block_gain_sums'])

    def test_reordering_old_returns_keeps_current_risk_tail_unchanged(self):
        concentrated = [1.0] * 252
        spread = [1.0] * 252
        for i in range(11):
            concentrated[i] = 2.0
            spread[21 * i] = 2.0
        a = prices_from_grosses(concentrated)
        b = prices_from_grosses(spread)
        self.assertEqual(a[231:], b[231:])
        left, right = self.compare(a), self.compare(b)
        self.assertEqual(left['exact']['daily_quality'], right['exact']['daily_quality'])
        self.assertLess(left['block_quality'], right['block_quality'])

    def test_gross_ratio_rounded_first_not_exact_price_fraction(self):
        prices = [3.0] + [10.0] * 252
        result = self.compare(prices)
        canonical = Fraction.from_float(10.0 / 3.0 - 1.0)
        self.assertEqual(as_fraction(result['exact']['gain_sum']), canonical)
        self.assertNotEqual(canonical, Fraction(7, 3))

    def test_distinct_exact_qualities_can_tie_after_float_conversion(self):
        ordinary = prices_from_grosses([2.0] * 252)
        perturbed = prices_from_grosses([math.nextafter(2.0, math.inf)] + [2.0] * 251)
        one = self.compare(ordinary)
        two = self.compare(perturbed)
        for kind in ('daily', 'block'):
            exact = as_fraction(two['exact'][kind + '_quality'])
            self.assertLess(exact, 1)
            self.assertNotEqual(one['exact'][kind + '_quality'], two['exact'][kind + '_quality'])
            self.assertEqual(one[kind + '_quality'], two[kind + '_quality'])
            self.assertEqual(one[kind + '_raw'], two[kind + '_raw'])

    def test_power_of_two_price_scale_invariance(self):
        original = two_jump_prices(22, 1.5)
        base = self.compare(original)
        for exponent in (-900, -100, 100, 900):
            scaled = [math.ldexp(value, exponent) for value in original]
            self.assert_same(base, self.compare(scaled))

    def test_unequal_price_scales_and_exact_mass_sum_exceeding_float_max(self):
        result = self.compare([1.0 if i % 2 == 0 else MAXIMUM for i in range(253)])
        self.assertGreater(as_fraction(result['exact']['gain_sum']), Fraction.from_float(MAXIMUM))
        self.assertTrue(math.isfinite(result['daily_raw']))
        self.assertTrue(math.isfinite(result['block_raw']))
        self.compare([TINY if i % 2 == 0 else 2 * TINY for i in range(253)])

    def test_nonpositive_raw_does_not_skip_formation_or_quality_validation(self):
        result = self.compare(two_jump_prices(), -0.0)
        self.assertEqual(result['positive_gain_indices'], [1, 22])
        self.assertEqual(struct.pack('>d', result['block_raw']), struct.pack('>d', -0.0))
        invalid = [1.0] * 253
        invalid[151] = float('nan')
        self.reject_both(invalid, -3.0)

    def test_reject_malformed_vectors_and_finite_domains(self):
        invalid_vectors = ([1.0] * 252, [1.0] * 254, '1' * 253,
                           bytes(253), {i: 1.0 for i in range(253)}, iter([1.0] * 253))
        for values in invalid_vectors:
            self.reject_both(values)
        for bad in (True, False, None, '1.0', complex(1, 0), float('nan'), float('inf'), -float('inf'), 0, -1):
            prices = [1.0] * 253
            prices[125] = bad
            self.reject_both(prices)
        for bad in (True, None, '0', complex(1, 0), float('nan'), float('inf'), 10 ** 1000):
            self.reject_both([1.0] * 253, bad)

    def test_nonzero_input_conversion_underflow_is_failure(self):
        tiny_exact = Fraction(1, 2 ** 2000)
        prices = [1.0] * 253
        prices[20] = tiny_exact
        self.reject_both(prices, kind=ArithmeticError)
        self.reject_both([1.0] * 253, tiny_exact, ArithmeticError)

    def test_ratio_overflow_and_underflow_are_failures(self):
        self.reject_both([TINY, MAXIMUM] + [1.0] * 251, kind=ArithmeticError)
        self.reject_both([MAXIMUM, TINY] + [1.0] * 251, kind=ArithmeticError)

    def test_reachable_positive_quality_conversion_underflow(self):
        prices = [1.0] * 253
        prices[1] = prices[3] = MAXIMUM
        prices[22:] = [math.nextafter(1.0, math.inf)] * 231
        self.reject_both(prices, kind=ArithmeticError)
        # An original negative score still requires both exact qualities to convert.
        self.reject_both(prices, -1.0, ArithmeticError)

    def test_positive_raw_multiplication_underflow_is_failure(self):
        self.reject_both(two_jump_prices(22, 1.5), TINY, ArithmeticError)

    def test_quality_conversion_exact_zero_and_round_to_one(self):
        self.assertEqual(oracle.quality_to_binary64(Fraction(0)), 0.0)
        self.assertEqual(oracle.quality_to_binary64(1 - Fraction(1, 2 ** 100)), 1.0)
        with self.assertRaises(ArithmeticError):
            oracle.quality_to_binary64(Fraction(1, 2 ** 2000))
        for bad in (0.5, -Fraction(1, 10), Fraction(11, 10)):
            with self.assertRaises(ValueError):
                oracle.quality_to_binary64(bad)

    def test_equal_block_assignment_expectation_exact_identity(self):
        same_pairs = 12 * 21 * 20 // 2
        total_pairs = 252 * 251 // 2
        same_probability = Fraction(same_pairs, total_pairs)
        self.assertEqual(same_probability, Fraction(20, 251))
        self.assertEqual((1 - same_probability) * Fraction(12, 11), Fraction(252, 251))
        # Two equal nonzero masses: same-block Q=0; different-block Q=6/11.
        mean_block_quality = (1 - same_probability) * Fraction(6, 11)
        two = [0.0] * 252
        two[0] = two[1] = 1.0
        self.assertEqual(mean_block_quality, oracle.exact_qualities_from_gains(two)['daily_quality'])

    def test_generated_reachable_paths_full_evidence_agrees(self):
        rng = random.Random(265)
        for case in range(24):
            grosses = [rng.choice((0.75, 0.875, 1.0, 1.125, 1.25, 1.5)) for unused in range(252)]
            start = math.ldexp(1.0, (case % 7 - 3) * 100)
            raw = (-1.5, -0.0, 0.0, 0.125, 7.0)[case % 5]
            self.compare(prices_from_grosses(grosses, start), raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
