"""Synthetic-only arithmetic checks; no financial source input is opened."""
from fractions import Fraction as F
import copy
import itertools
import math
import unittest
import roa_stability_preflight_math as kernel


def rows(values):
    return [(F(v), F(1), F(1)) for v in values]


class TestProposedArithmetic(unittest.TestCase):
    def test_hand_computed_sample_dispersion(self):
        out = kernel.measure(rows([1, 2, 3]))
        self.assertEqual(out['mean'], F(2))
        self.assertEqual(out['sample_variance'], F(1))
        self.assertEqual(out['sample_std'], 1.)
        self.assertEqual(out['candidate'], .5)
        self.assertEqual(out['control'], float(F(2, 3)))
        self.assertFalse(out['equal_arms'])
        self.assertFalse(out['zero_dispersion'])

    def test_irrational_dispersion_rounding_contract(self):
        out = kernel.measure(rows([0, 0, 1]))
        self.assertEqual(out['mean'], F(1, 3))
        self.assertEqual(out['sample_variance'], F(1, 3))
        sigma = math.sqrt(float(F(1, 3)))
        p = F(1, 3) - F.from_float(sigma)
        self.assertEqual(out['penalized_mean'], p)
        self.assertEqual(out['candidate'], float(p / (1 + abs(p))))
        self.assertEqual(out['control'], .25)

    def test_zero_dispersion_exact_copy_positive_zero_negative(self):
        for v in [F(2, 5), F(0), F(-2, 5)]:
            with self.subTest(v=v):
                out = kernel.measure(rows([v]*3))
                self.assertEqual(out['sample_variance'], F(0))
                self.assertEqual(out['sample_std'], 0.)
                self.assertEqual(out['candidate'].hex(), out['control'].hex())
                self.assertTrue(out['zero_dispersion'])
                self.assertTrue(out['equal_arms'])

    def test_negative_net_income_is_valid(self):
        out = kernel.measure(rows([-3, -2, -1]))
        self.assertEqual(out['mean'], F(-2))
        self.assertEqual(out['candidate'], -.75)
        self.assertEqual(out['control'], float(F(-2, 3)))

    def test_own_year_endpoints_no_stitching(self):
        amounts = [(F(10), F(10), F(30)), (F(30), F(20), F(40)),
                   (F(60), F(50), F(70))]
        out = kernel.measure(amounts)
        self.assertEqual(out['roa'], (F(1, 2), F(1), F(1)))
        self.assertEqual(out['mean'], F(5, 6))
        self.assertEqual(out['sample_variance'], F(1, 12))

    def test_scale_invariance_each_report(self):
        amounts = [(F(1, 3), F(2), F(4)), (F(-2), F(8), F(10)),
                   (F(3), F(11), F(13))]
        reference = kernel.measure(amounts)
        scales = [F(1000000), F(1, 1000), F(7, 13)]
        scaled = [tuple(x*s for x in row) for row, s in zip(amounts, scales)]
        self.assertEqual(kernel.measure(scaled), reference)

    def test_permutation_invariance_of_summary(self):
        reference = kernel.measure(rows([F(1, 7), F(-1, 3), F(4, 9)]))
        for order in itertools.permutations([F(1, 7), F(-1, 3), F(4, 9)]):
            actual = kernel.measure(rows(order))
            self.assertEqual({k:v for k,v in actual.items() if k != 'roa'},
                             {k:v for k,v in reference.items() if k != 'roa'})

    def test_input_not_mutated(self):
        amounts = rows([1, 2, 3])
        before = copy.deepcopy(amounts)
        kernel.measure(amounts)
        self.assertEqual(amounts, before)

    def test_exactly_three_reports(self):
        for n in [0, 1, 2, 4, 8]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                kernel.measure(rows([1]*n))
        for bad in [None, '123', iter(rows([1, 2, 3])), {'x': rows([1, 2, 3])}]:
            with self.assertRaises(ValueError):
                kernel.measure(bad)

    def test_malformed_row_fails(self):
        for bad in [(F(1), F(2)), (F(1), F(2), F(3), F(4)), None,
                    {'ni': F(1), 'open': F(1), 'close': F(1)}]:
            with self.assertRaises(ValueError):
                kernel.measure([bad] + rows([1, 2]))

    def test_nonfractions_fail_in_every_field(self):
        for bad in [1, 1., True, '1', None, float('nan'), float('inf')]:
            for column in range(3):
                amounts = rows([1, 2, 3])
                row = list(amounts[0]); row[column] = bad; amounts[0] = tuple(row)
                with self.subTest(bad=bad, column=column), self.assertRaises(ValueError):
                    kernel.measure(amounts)

    def test_assets_must_be_strictly_positive(self):
        for column in [1, 2]:
            for bad in [F(0), F(-1)]:
                amounts = rows([1, 2, 3])
                row = list(amounts[0]); row[column] = bad; amounts[0] = tuple(row)
                with self.assertRaises(ValueError):
                    kernel.measure(amounts)

    def test_variance_overflow_fails(self):
        with self.assertRaises(FloatingPointError):
            kernel.measure(rows([-10**200, 0, 10**200]))

    def test_nonzero_variance_underflow_fails(self):
        tiny = F(1, 10**200)
        with self.assertRaises(FloatingPointError):
            kernel.measure(rows([0, tiny, tiny*2]))

    def test_final_bounded_score_underflow_fails(self):
        for sign in [-1, 1]:
            with self.assertRaises(FloatingPointError):
                kernel.measure(rows([F(sign, 10**400)]*3))

    def test_nonzero_subnormal_variance_is_retained(self):
        delta = F(1, 2**537)
        out = kernel.measure(rows([0, delta, delta*2]))
        self.assertEqual(float(out['sample_variance']), float.fromhex('0x0.0000000000001p-1022'))
        self.assertEqual(out['sample_std'], float(delta))
        self.assertFalse(out['zero_dispersion'])
        self.assertEqual(out['candidate'], 0.)
        self.assertGreater(out['control'], 0.)

    def test_negative_large_constant_saturates_only_after_bounding(self):
        out = kernel.measure(rows([F(-10**500)]*3))
        self.assertEqual(out['candidate'], -1.)
        self.assertTrue(out['zero_dispersion'])

    def test_large_constant_bounded_without_unneeded_conversion(self):
        out = kernel.measure(rows([F(10**500)]*3))
        self.assertEqual(out['candidate'], 1.)
        self.assertTrue(out['zero_dispersion'])

    def test_rounding_equal_arms_is_not_zero_dispersion(self):
        out = kernel.measure(rows([10**40-1, 10**40, 10**40+1]))
        self.assertEqual(out['sample_variance'], F(1))
        self.assertTrue(out['equal_arms'])
        self.assertFalse(out['zero_dispersion'])

    def test_monotone_bounded_penalty_grid(self):
        for values in itertools.product([-10, -1, 0, F(1, 7), 1, 10], repeat=3):
            out = kernel.measure(rows(values))
            self.assertLessEqual(out['candidate'], out['control'])
            self.assertTrue(math.isfinite(out['candidate']))
            self.assertGreaterEqual(out['candidate'], -1.)
            self.assertLessEqual(out['control'], 1.)

    def test_bounded_rejects_inexact_inputs(self):
        for value in [0, .1, True, '0.1']:
            with self.assertRaises(ValueError):
                kernel.bounded(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
