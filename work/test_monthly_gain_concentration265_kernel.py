"""Synthetic-only tests for the unregistered, pure 265 concentration kernel."""
import ast
from fractions import Fraction
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import monthly_gain_concentration265_kernel as k


def jump_prices(indices):
    value = 1.0
    prices = [value]
    for i in range(1, 253):
        if i in indices:
            value *= 2.0
        prices.append(value)
    return prices


def exact_record(record):
    return Fraction(record['numerator'], record['denominator'])


def independent_toy(gains, block_size):
    """Independent small-N proof helper, never used by the production kernel."""
    weights = [Fraction(a) / sum(gains) for a in gains]
    n = len(weights)
    blocks = n // block_size
    hd = sum(a ** 2 for a in weights)
    hb = sum(sum(weights[start:start + block_size]) ** 2
             for start in range(0, n, block_size))
    return (1 - hd) * n / (n - 1), (1 - hb) * blocks / (blocks - 1)


class ConcentrationMathematics(unittest.TestCase):
    def test_single_positive_jump_both_qualities_zero(self):
        for index in (1, 21, 22, 252):
            result = k.measure(jump_prices({index}), 3.0)
            self.assertEqual(result['positive_gain_indices'], [index])
            self.assertEqual(result['daily_quality'], 0.0)
            self.assertEqual(result['block_quality'], 0.0)
            self.assertEqual(result['daily_raw'].hex(), '0x0.0p+0')
            self.assertEqual(result['block_raw'].hex(), '0x0.0p+0')

    def test_all_equal_positive_gains_attain_both_upper_bounds(self):
        result = k.measure(jump_prices(set(range(1, 253))), 7.0)
        self.assertEqual(result['daily_gains'], [1.0] * 252)
        for role, count in (('daily', 252), ('block', 12)):
            self.assertEqual(exact_record(result['exact'][role + '_concentration']), Fraction(1, count))
            self.assertEqual(result[role + '_quality'], 1.0)
            self.assertEqual(result[role + '_raw'], 7.0)

    def test_same_gains_spaced_versus_clustered_has_no_pathwise_order(self):
        spaced = k.measure(jump_prices(set(range(21, 253, 21))), 2.0)
        clustered = k.measure(jump_prices(set(range(1, 13))), 2.0)
        self.assertEqual(sorted(spaced['daily_gains']), sorted(clustered['daily_gains']))
        expected_daily = Fraction(231, 251)
        self.assertEqual(exact_record(spaced['exact']['daily_quality']), expected_daily)
        self.assertEqual(spaced['daily_quality'], clustered['daily_quality'])
        self.assertEqual(spaced['block_quality'], 1.0)
        self.assertEqual(clustered['block_quality'], 0.0)
        self.assertGreater(spaced['block_quality'], spaced['daily_quality'])
        self.assertLess(clustered['block_quality'], clustered['daily_quality'])

    def test_block_boundaries_are_fixed_one_based_21_sessions(self):
        same = k.measure(jump_prices({20, 21}), 1.0)
        different = k.measure(jump_prices({21, 22}), 1.0)
        self.assertEqual(same['block_quality'], 0.0)
        self.assertEqual(exact_record(different['exact']['block_quality']), Fraction(6, 11))
        self.assertEqual(same['daily_quality'], different['daily_quality'])

    def test_no_positive_gains_is_explicit_neutral_not_fake_concentration(self):
        for prices in ([100.0] * 253, [float(300 - i) for i in range(253)]):
            result = k.measure(prices, 2.0)
            self.assertTrue(result['no_positive_gains'])
            self.assertEqual(result['daily_quality'], 1.0)
            self.assertEqual(result['block_quality'], 1.0)
            self.assertIsNone(result['exact']['daily_concentration'])
            self.assertIsNone(result['exact']['block_concentration'])
            self.assertEqual(exact_record(result['exact']['gain_sum']), 0)

    def test_nonpositive_original_is_exact_including_negative_zero(self):
        for raw in (-sys.float_info.max, -3.0, -math.ulp(0.0), -0.0, 0.0):
            for prices in (jump_prices({1}), jump_prices(set(range(21, 253, 21)))):
                result = k.measure(prices, raw)
                self.assertTrue(result['nonpositive_original_preserved'])
                self.assertEqual(result['daily_raw'].hex(), raw.hex())
                self.assertEqual(result['block_raw'].hex(), raw.hex())

    def test_binary64_returns_precede_exact_fraction_weights(self):
        prices = [1.0 + 0.001 * i + 0.002 * math.sin(i) for i in range(253)]
        result = k.measure(prices, 1.0)
        gains = [prices[i] / prices[i - 1] - 1.0 for i in range(1, 253)]
        self.assertEqual(result['daily_gains'], gains)
        expected = sum((Fraction.from_float(max(g, 0.0)) for g in gains), Fraction(0))
        self.assertEqual(exact_record(result['exact']['gain_sum']), expected)
        self.assertEqual(result['daily_quality'], float(exact_record(result['exact']['daily_quality'])))
        self.assertEqual(result['block_quality'], float(exact_record(result['exact']['block_quality'])))

    def test_exact_sum_can_exceed_binary64_without_float_overflow(self):
        values = [sys.float_info.max] * 252
        result = k.exact_qualities_from_gains(values)
        self.assertGreater(result['gain_sum'], Fraction.from_float(sys.float_info.max))
        self.assertEqual(result['daily_quality'], 1)
        self.assertEqual(result['block_quality'], 1)

    def test_exact_expected_block_quality_equals_daily_for_fixed_252_slots(self):
        same = [0.0] * 252
        same[0], same[1] = 1.0, 3.0
        separate = same.copy()
        separate[1], separate[21] = 0.0, 3.0
        a = k.exact_qualities_from_gains(same)
        b = k.exact_qualities_from_gains(separate)
        # Every ordered pair occupies one of 252*251 positions. Conditional
        # on the first position, exactly 20 of 251 share its fixed block.
        same_count = sum(1 for i in range(252) for j in range(252)
                         if i != j and i // 21 == j // 21)
        self.assertEqual(Fraction(same_count, 252 * 251), Fraction(20, 251))
        expected = Fraction(20, 251) * a['block_quality'] + Fraction(231, 251) * b['block_quality']
        self.assertEqual(expected, a['daily_quality'])
        self.assertEqual(a['daily_quality'], b['daily_quality'])

    def test_independent_small_toy_exhaustive_permutations_exact_identity(self):
        for gains, width in (((0, 1, 2, 4), 2), ((0, 0, 1, 1, 2, 4), 2)):
            values = [independent_toy(p, width) for p in set(permutations(gains))]
            self.assertTrue(all(daily == values[0][0] for daily, block in values))
            self.assertEqual(sum(block for daily, block in values) / len(values), values[0][0])

    def test_power_two_price_rescaling_is_exact(self):
        prices = jump_prices({1, 22, 64, 167, 251})
        self.assertEqual(k.measure(prices, 3.0), k.measure([p * 8 for p in prices], 3.0))

    def test_evidence_json_serializes_without_nonfinite_values(self):
        for prices in (jump_prices({1, 22}), [1.0] * 253):
            result = k.measure(prices, -0.0)
            self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)


class FailuresBeforeScore(unittest.TestCase):
    def reject_before_transform(self, prices, raw=1.0, exception=ValueError):
        with patch.object(k, '_transform', side_effect=AssertionError('Scoring ran')) as transform:
            with self.assertRaises(exception):
                k.measure(prices, raw)
            transform.assert_not_called()

    def test_bad_window_lengths_and_ambiguous_containers(self):
        for prices in ([1.0] * 252, [1.0] * 254, [], '1' * 253, {i: 1.0 for i in range(253)}, iter([1.0] * 253)):
            self.reject_before_transform(prices)

    def test_invalid_price_types_and_values_even_at_last_slot(self):
        for bad in (True, False, None, '1', 1 + 0j, 0.0, -0.0, -1.0, math.inf, -math.inf, math.nan):
            prices = [1.0] * 253
            prices[-1] = bad
            with self.subTest(value=repr(bad)):
                self.reject_before_transform(prices)

    def test_invalid_original_raw(self):
        for raw in (True, False, None, '1', 1 + 0j, math.inf, -math.inf, math.nan):
            with self.subTest(raw=repr(raw)):
                self.reject_before_transform([1.0] * 253, raw)

    def test_input_conversion_overflow_and_underflow(self):
        for bad, error in ((10 ** 1000, ValueError), (Fraction(1, 10 ** 1000), ArithmeticError)):
            self.reject_before_transform([bad] + [1.0] * 252, exception=error)
            self.reject_before_transform([1.0] * 253, bad, error)

    def test_daily_return_overflow_and_underflow_fail_before_score(self):
        tiny, huge = math.ulp(0.0), sys.float_info.max
        for first, second in ((tiny, huge), (huge, tiny)):
            self.reject_before_transform([first, second] + [second] * 251, exception=ArithmeticError)

    def test_positive_exact_quality_cast_underflow_is_not_clipped(self):
        gains = [sys.float_info.max, math.ulp(0.0)] + [0.0] * 250
        quality = k.exact_qualities_from_gains(gains)['daily_quality']
        self.assertGreater(quality, 0)
        with self.assertRaises(ArithmeticError):
            k._quality_float(quality)

    def test_reachable_price_path_quality_underflow_precedes_any_score(self):
        prices = ([1.0, sys.float_info.max, 1.0, sys.float_info.max]
                  + [1.0] * 18 + [math.nextafter(1.0, math.inf)] * 231)
        self.assertEqual(len(prices), 253)
        gains = [prices[i] / prices[i - 1] - 1.0 for i in range(1, 253)]
        exact = k.exact_qualities_from_gains(gains)
        self.assertGreater(exact['gain_sum'], Fraction.from_float(sys.float_info.max))
        self.assertGreater(exact['block_quality'], 0)
        with patch.object(k, '_transform', side_effect=AssertionError('Scoring ran')) as transform:
            with self.assertRaisesRegex(ArithmeticError, 'Positive exact quality conversion underflow'):
                k.measure(prices, 1.0)
            transform.assert_not_called()

    def test_positive_raw_multiplication_underflow_and_legitimate_zero(self):
        prices = [1.0, 2.0] + [math.nextafter(2.0, math.inf)] * 251
        with self.assertRaises(ArithmeticError):
            k.measure(prices, math.ulp(0.0))
        result = k.measure(jump_prices({1}), math.ulp(0.0))
        self.assertEqual(result['daily_raw'].hex(), '0x0.0p+0')
        self.assertEqual(result['block_raw'].hex(), '0x0.0p+0')

    def test_gain_helper_rejects_wrong_length_bool_and_nonfinite(self):
        for gains in ([0.0] * 251, [True] + [0.0] * 251, [math.inf] + [0.0] * 251,
                      [math.nan] + [0.0] * 251):
            with self.assertRaises(ValueError):
                k.exact_qualities_from_gains(gains)

    def test_kernel_has_only_stdlib_arithmetic_imports_and_no_io(self):
        tree = ast.parse(Path(k.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {'open', 'eval', 'exec', '__import__'})
        self.assertEqual(set(imports), {'collections.abc', 'fractions', 'math', 'numbers'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
