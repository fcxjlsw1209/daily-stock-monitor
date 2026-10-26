"""Definition-only exact algebra checks; no repository data or strategy imports.

All observations below are invented.  This is not a production price loader,
portfolio selector, proposed lookback choice, or strategy performance test.
"""

import json
import math
from fractions import Fraction as F
from numbers import Real
import unittest


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Synthetic prices must be real, nonboolean numbers')
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError('Synthetic prices must be finite and positive')
    return F(value)


def sample_covariance(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('One common sample with at least two returns is required')
    n = len(left)
    ml, mr = sum(left, F(0)) / n, sum(right, F(0)) / n
    return sum(((a - ml) * (b - mr) for a, b in zip(left, right)), F(0)) / (n - 1)


def fixed_units(paths):
    """Exact initial-EW fixed adjusted-unit wealth and contribution covariance."""
    if len(paths) != 3 or len(paths[0]) < 3 or any(len(p) != len(paths[0]) for p in paths):
        raise ValueError('Exactly three complete common positive paths are required')
    prices = [[_positive(v) for v in p] for p in paths]
    x = [[v / p[0] for v in p] for p in prices]
    totals = [sum((p[d] for p in x), F(0)) for d in range(len(x[0]))]
    wealth = [v / 3 for v in totals]
    returns = [wealth[d] / wealth[d - 1] - 1 for d in range(1, len(wealth))]
    weights = [[p[d - 1] / totals[d - 1] for d in range(1, len(wealth))] for p in x]
    asset_returns = [[p[d] / p[d - 1] - 1 for d in range(1, len(p))] for p in prices]
    contributions = [[(p[d] - p[d - 1]) / totals[d - 1]
                      for d in range(1, len(p))] for p in x]
    covariance = [[sample_covariance(a, b) for b in contributions] for a in contributions]
    diagonal = sum((covariance[i][i] for i in range(3)), F(0))
    cross = 2 * sum((covariance[i][j] for i in range(3) for j in range(i + 1, 3)), F(0))
    full = sample_covariance(returns, returns)
    return dict(prices=prices, normalized=x, wealth=wealth, returns=returns,
                weights=weights, asset_returns=asset_returns, contributions=contributions,
                covariance=covariance, full=full, diagonal=diagonal, cross=cross,
                mean=sum(returns, F(0)) / len(returns), terminal=wealth[-1] - 1)


DRIFT = [[1, 2, 2], [1, 1, 2], [1, 1, 1]]
OFFSET = [[1, F(7, 5), 1, F(6, 5), 1],
          [1, F(3, 5), 1, F(4, 5), 1], [1, 1, 1, 1, 1]]
IDENTICAL = [[1, 2, 1, 3]] * 3


class FixedUnitAlgebraTests(unittest.TestCase):
    def assert_identities(self, result):
        r = result
        for d, ret in enumerate(r['returns']):
            self.assertEqual(sum((c[d] for c in r['contributions']), F(0)), ret)
            self.assertEqual(sum((w[d] for w in r['weights']), F(0)), 1)
            self.assertGreater(ret, -1)
            for i in range(3):
                self.assertEqual(r['contributions'][i][d],
                                 r['weights'][i][d] * r['asset_returns'][i][d])
        self.assertEqual(r['full'], r['diagonal'] + r['cross'])
        self.assertGreaterEqual(r['full'], 0)
        self.assertGreaterEqual(r['diagonal'], 0)
        self.assertLessEqual(r['full'], 3 * r['diagonal'])
        self.assertGreaterEqual(r['cross'], -r['diagonal'])
        self.assertLessEqual(r['cross'], 2 * r['diagonal'])

    def test_exact_wealth_and_daily_drift(self):
        r = fixed_units(DRIFT)
        self.assertEqual(r['wealth'], [F(1), F(4, 3), F(5, 3)])
        self.assertEqual(r['returns'], [F(1, 3), F(1, 4)])
        self.assertEqual([w[1] for w in r['weights']], [F(1, 2), F(1, 4), F(1, 4)])
        self.assert_identities(r)

    def test_signed_cross_terms_are_not_clipped(self):
        r = fixed_units(DRIFT)
        self.assertEqual((r['full'], r['diagonal'], r['cross']),
                         (F(1, 288), F(25, 288), F(-1, 12)))

    def test_daily_equal_weight_is_a_different_path(self):
        r = fixed_units(DRIFT)
        daily_equal = [sum((a[d] for a in r['asset_returns']), F(0)) / 3 for d in range(2)]
        self.assertEqual(daily_equal, [F(1, 3), F(1, 3)])
        self.assertEqual((1 + daily_equal[0]) * (1 + daily_equal[1]), F(16, 9))
        self.assertNotEqual(F(16, 9), r['wealth'][-1])

    def test_static_one_ninth_covariance_is_not_fixed_unit_risk(self):
        r = fixed_units(DRIFT)
        naive = sum((sample_covariance(a, b) for a in r['asset_returns']
                     for b in r['asset_returns']), F(0)) / 9
        self.assertEqual(naive, 0)
        self.assertNotEqual(naive, r['full'])

    def test_wealth_increments_are_not_daily_returns(self):
        r = fixed_units(DRIFT)
        increments = [r['wealth'][d] - r['wealth'][d - 1] for d in (1, 2)]
        self.assertEqual(increments, [F(1, 3), F(1, 3)])
        self.assertEqual(sample_covariance(increments, increments), 0)
        self.assertNotEqual(sample_covariance(increments, increments), r['full'])

    def test_full_zero_with_positive_diagonal(self):
        r = fixed_units(OFFSET)
        self.assert_identities(r)
        self.assertEqual(r['full'], 0)
        self.assertGreater(r['diagonal'], 0)
        self.assertEqual(r['cross'], -r['diagonal'])

    def test_identical_paths_attain_upper_bound(self):
        r = fixed_units(IDENTICAL)
        self.assert_identities(r)
        self.assertGreater(r['full'], 0)
        self.assertEqual(r['full'], 3 * r['diagonal'])
        self.assertEqual(r['cross'], 2 * r['diagonal'])

    def test_constant_paths_both_zero(self):
        r = fixed_units([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
        self.assert_identities(r)
        self.assertEqual((r['full'], r['diagonal'], r['cross']), (0, 0, 0))

    def test_scale_and_permutation_invariance(self):
        r = fixed_units(DRIFT)
        other = fixed_units([[v * factor for v in DRIFT[i]]
                             for i, factor in ((2, F(7, 3)), (0, 23), (1, F(2, 11)))])
        for key in ('wealth', 'returns', 'full', 'diagonal', 'cross', 'mean', 'terminal'):
            self.assertEqual(r[key], other[key])

    def test_anchor_and_reinitialization_are_different(self):
        prices = [[1, 2, 3, 3], [1, 1, 1, 2], [1, 1, 1, 1]]
        anchored = fixed_units(prices)
        rebased = fixed_units([p[1:] for p in prices])
        self.assertEqual(anchored['returns'][1:], [F(1, 4), F(1, 5)])
        self.assertEqual(rebased['returns'], [F(1, 6), F(2, 7)])
        self.assertNotEqual(anchored['returns'][1:], rebased['returns'])

    def test_daily_contributions_link_but_do_not_sum_to_terminal(self):
        r = fixed_units(DRIFT)
        self.assertEqual(sum(r['returns'], F(0)), F(7, 12))
        self.assertEqual(r['terminal'], F(2, 3))
        self.assertEqual(sum((r['wealth'][d] * ret for d, ret in enumerate(r['returns'])), F(0)),
                         r['terminal'])
        for i in range(3):
            linked = sum((r['wealth'][d] * c for d, c in enumerate(r['contributions'][i])), F(0))
            self.assertEqual(linked, (r['normalized'][i][-1] - 1) / 3)

    def test_shared_matrix_is_required_and_invalid_prices_fail(self):
        for bad in (True, 0, -1, float('nan'), float('inf'), '2', None):
            paths = [p[:] for p in DRIFT]
            paths[1][1] = bad
            with self.assertRaises(ValueError):
                fixed_units(paths)
        for bad in ([DRIFT[0]], [DRIFT[0], DRIFT[1], DRIFT[2][:-1]]):
            with self.assertRaises(ValueError):
                fixed_units(bad)


def synthetic_summary():
    result = {}
    for name, paths in (('drift', DRIFT), ('offset', OFFSET), ('identical', IDENTICAL)):
        r = fixed_units(paths)
        result[name] = {key: str(r[key]) for key in ('full', 'diagonal', 'cross', 'mean', 'terminal')}
    return result


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixedUnitAlgebraTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'scope': 'hard_coded_synthetic_exact_algebra_only',
                      'tests': outcome.testsRun, 'examples': synthetic_summary()}, sort_keys=True))
