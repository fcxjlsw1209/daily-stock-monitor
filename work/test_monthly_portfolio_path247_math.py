"""Synthetic-only canonical trio statistics and numerical-boundary tests."""
from copy import deepcopy
from fractions import Fraction as F
import itertools
import json
import math
import sys
import unittest
import numpy as np
import monthly_portfolio_path247_math as kernel

DRIFT = [[1, 2, 2], [1, 1, 2], [1, 1, 1]]
OFFSET = [[1, 1.5, 1, 1.25, 1], [1, .5, 1, .75, 1], [1, 1, 1, 1, 1]]


def rational_reference(paths):
    prices = [[F(v) for v in row] for row in paths]
    x = [[v / row[0] for v in row] for row in prices]
    totals = [sum((row[k] for row in x), F(0)) for k in range(len(x[0]))]
    nav = [v / 3 for v in totals]
    returns = [nav[k] / nav[k-1] - 1 for k in range(1, len(nav))]
    contributions = [[(row[k] - row[k-1]) / totals[k-1] for k in range(1, len(nav))] for row in x]
    n = len(returns)
    def variance(row):
        mean = sum(row, F(0)) / n
        return sum(((v-mean)**2 for v in row), F(0)) / (n-1)
    return returns, sum(returns, F(0))/n, variance(returns), sum((variance(c) for c in contributions), F(0))


class FixedUnitTests(unittest.TestCase):
    def assert_close(self, a, b, tol=2e-12):
        self.assertLessEqual(abs(a-b), tol+tol*max(abs(a),abs(b)))

    def test_known_fraction_drift_counterexample(self):
        z=kernel.measure(DRIFT)
        for key, expected in [('mean_return',F(7,24)),('joint_variance',F(1,288)),
                              ('diagonal_variance',F(25,288)),('cross_covariance',F(-1,12))]:
            self.assert_close(z[key],float(expected))
        self.assertEqual(z['portfolio_returns'],[1/3,1/4])
        self.assert_close(z['joint_ratio'],float(F(7,24))/math.sqrt(float(F(1,288))))
        self.assertLess(z['cross_covariance'],0.)

    def test_exact_rational_reference_for_varied_paths(self):
        for paths in (DRIFT, [[1,2,1,3]]*3, OFFSET,
                      [[2,3,2,4],[3,3,5,4],[4,5,4,3]]):
            z=kernel.measure(paths);r,m,f,d=rational_reference(paths)
            for actual,expected in zip(z['portfolio_returns'],r): self.assert_close(actual,float(expected))
            for actual,expected in ((z['mean_return'],m),(z['joint_variance'],f),(z['diagonal_variance'],d)):
                self.assert_close(actual,float(expected))

    def test_drifting_weights_are_not_daily_rebalanced(self):
        z=kernel.measure(DRIFT)
        equal_daily=[sum(row[k]/row[k-1]-1 for row in DRIFT)/3 for k in (1,2)]
        self.assertEqual(equal_daily,[1/3,1/3])
        self.assertNotEqual(z['portfolio_returns'],equal_daily)
        self.assertEqual(z['contributions'][1][1],.25)
        self.assert_close(z['portfolio_values'][-1],5/3)
        self.assertNotEqual((1+equal_daily[0])*(1+equal_daily[1]),z['portfolio_values'][-1])

    def test_mean_return_not_terminal_gain(self):
        z=kernel.measure(DRIFT)
        self.assert_close(z['mean_return'],7/24)
        self.assertNotEqual(z['mean_return'],z['portfolio_values'][-1]-1)

    def test_wealth_increments_not_portfolio_returns(self):
        z=kernel.measure(DRIFT)
        increments=[z['portfolio_values'][k]-z['portfolio_values'][k-1] for k in (1,2)]
        self.assert_close(increments[0],increments[1])
        self.assertGreater(z['joint_variance'],0.)

    def test_joint_zero_diagonal_positive_common_unavailable(self):
        z=kernel.measure(OFFSET)
        self.assertEqual(z['joint_variance'],0.)
        self.assertGreater(z['diagonal_variance'],0.)
        self.assertFalse(z['available']);self.assertEqual(z['reason'],'zero_variance')
        self.assertIsNone(z['joint_ratio']);self.assertIsNone(z['diagonal_ratio'])
        self.assertEqual(z['portfolio_returns'],[0.]*4)
        self.assert_close(z['cross_covariance'],-z['diagonal_variance'])
        self.assertTrue(z['diagnostics']['covariance_identity_passed'])

    def test_both_zero_retains_all_statistics(self):
        z=kernel.measure([[2,2,2],[3,3,3],[4,4,4]])
        self.assertFalse(z['available'])
        for key in ('mean_return','joint_variance','diagonal_variance','joint_sd','diagonal_sd'):
            self.assertEqual(z[key],0.)
        self.assertEqual(z['contributions'],[[0.,0.]]*3)
        self.assertIsNone(z['joint_ratio']);self.assertIsNone(z['diagonal_ratio'])
        json.dumps(z,allow_nan=False)

    def test_identical_paths_allowed_and_upper_bound_attained(self):
        z=kernel.measure([[1,2,1,3]]*3)
        self.assertTrue(z['available'])
        self.assert_close(z['joint_variance'],3*z['diagonal_variance'])
        self.assert_close(z['cross_covariance'],2*z['diagonal_variance'])

    def test_zero_mean_with_positive_variances_is_available(self):
        z=kernel.measure([[1,1.5,.75]]*3)
        self.assertTrue(z['available']);self.assertEqual(z['mean_return'],0.)
        self.assertEqual(z['joint_ratio'],0.);self.assertEqual(z['diagonal_ratio'],0.)
        self.assertGreater(z['joint_variance'],0.)

    def test_negative_mean_ratios_valid(self):
        z=kernel.measure([[1,.5,.375]]*3)
        self.assertTrue(z['available']);self.assertLess(z['mean_return'],0.)
        self.assertLess(z['joint_ratio'],0.);self.assertLess(z['diagonal_ratio'],0.)

    def test_scale_and_permutation_invariance(self):
        baseline=kernel.measure([[2,3,2,4],[3,3,5,4],[4,5,4,3]])
        paths=[[2,3,2,4],[3,3,5,4],[4,5,4,3]]
        for order in itertools.permutations(range(3)):
            moved=[[v*scale for v in paths[i]] for i,scale in zip(order,[.125,7.,13.])]
            z=kernel.measure(moved)
            for key in ('mean_return','joint_variance','diagonal_variance','joint_ratio','diagonal_ratio'):
                self.assert_close(z[key],baseline[key])
            for value,expected in zip(z['portfolio_returns'],baseline['portfolio_returns']):
                self.assert_close(value,expected)

    def test_full_253_synthetic_window_and_sample_denominator(self):
        paths=[[1+.01*k+.05*math.sin((i+1)*k/9) for k in range(253)] for i in range(3)]
        z=kernel.measure(paths)
        self.assertEqual((z['price_count'],z['return_count']),(253,252))
        self.assertEqual(z['joint_variance'],z['joint_square_sum']/251.)
        self.assertEqual(z['diagonal_variance'],math.fsum(z['contribution_square_sums'])/251.)
        self.assertTrue(z['available'])
        self.assertEqual(len(z['contributions']),3)
        self.assertTrue(all(len(row)==252 for row in z['contributions']))

    def test_no_input_mutation_and_plain_serializable_output(self):
        paths=deepcopy(DRIFT);before=deepcopy(paths);z=kernel.measure(paths)
        self.assertEqual(paths,before)
        self.assertEqual(json.loads(json.dumps(z,allow_nan=False)),z)
        self.assertEqual([v['indices'] for v in z['pair_covariances']],[[0,1],[0,2],[1,2]])

    def test_numpy_and_tuple_inputs_have_same_result(self):
        self.assertEqual(kernel.measure(np.asarray(DRIFT,float)),kernel.measure(tuple(tuple(r) for r in DRIFT)))

    def test_diagnostics_tolerances_fixed_and_do_not_zero_small_risk(self):
        z=kernel.measure([[1,1+2**-20,1]]*3)
        self.assertGreater(z['joint_variance'],0.)
        self.assertLess(z['joint_variance'],kernel.DIAGNOSTIC_ATOL)
        self.assertTrue(z['available'])
        self.assertEqual(z['diagnostics']['nav_identity_atol'],2e-12)
        self.assertEqual(z['diagnostics']['covariance_atol'],2e-10)


class ValidationTests(unittest.TestCase):
    def test_invalid_shapes_rejected(self):
        invalid=[[],DRIFT[:2],DRIFT+[DRIFT[0]],[[1,2]]*3,[DRIFT[0],DRIFT[1],DRIFT[2][:-1]],
                 [1,2,3],None,'abc',{'a':[1,2,3]},[[[1],[2],[3]]]*3]
        for paths in invalid:
            with self.subTest(paths=repr(paths)):
                with self.assertRaises(ValueError): kernel.measure(paths)

    def test_invalid_prices_rejected(self):
        for value in [True,np.bool_(True),False,0.,-1.,float('nan'),float('inf'),-float('inf'),
                      '2',None,1+0j,10**10000,F(1,10**1000)]:
            paths=deepcopy(DRIFT);paths[1][1]=value
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(ValueError): kernel.measure(paths)

    def test_normalization_overflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'normalized price'):
            kernel.measure([[math.ulp(0.),1.,1.],[1,1,1],[1,1,1]])

    def test_normalization_underflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'Underflow.*normalized price'):
            kernel.measure([[sys.float_info.max,math.ulp(0.),1.],[1,1,1],[1,1,1]])

    def test_normalized_total_overflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'normalized total'):
            kernel.measure([[1,1e308,1]]*3)

    def test_centered_square_overflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'centered square'):
            kernel.measure([[1,1e200,1],[1,1,1],[1,1,1]])

    def test_contribution_division_underflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'Underflow.*daily contribution'):
            kernel.measure([[1,1,math.nextafter(1.,math.inf)],[1,1e308,1e308],[1,1,1]])

    def test_centered_square_underflow_is_arithmetic_failure(self):
        with self.assertRaisesRegex(ArithmeticError,'Underflow.*centered square'):
            kernel.measure([[1,1e-200,2e-200,1],[1,2,2,1],[1,1,1,1]])

    def test_nonzero_division_never_silently_becomes_zero(self):
        for label in ('mean','sample variance','ratio'):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ArithmeticError,'Underflow'):
                    kernel._divide(math.ulp(0.),2.,label)
        self.assertEqual(kernel._divide(0.,2.,'genuine zero'),0.)

    def test_pair_diagnostic_does_not_clip_or_accept_large_mismatch(self):
        with self.assertRaisesRegex(ArithmeticError,'Diagnostic failed'):
            kernel._close(1.,1.01,kernel.DIAGNOSTIC_ATOL,kernel.DIAGNOSTIC_RTOL,'synthetic mismatch')
        self.assertEqual(kernel._close(-.1,-.1,kernel.DIAGNOSTIC_ATOL,kernel.DIAGNOSTIC_RTOL,'signed'),0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
