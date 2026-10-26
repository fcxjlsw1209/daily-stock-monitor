"""Prospective score algebra on invented numbers only; no prices or strategy I/O."""

from fractions import Fraction as F
from itertools import combinations
import json
import unittest


def mean(values):
    return sum(values, F(0)) / len(values)


def utility(core, auxiliary, cw, budget, quality):
    if not 0 <= budget <= 1 - cw or not 0 < cw < 1:
        raise ValueError('Nonnegative funded auxiliary allocation')
    return core + (1 - budget / (1 - cw)) * auxiliary + budget * quality


def percentile(values):
    """Average ascending rank/N-.5, including its nonzero all-ties convention."""
    n = len(values)
    if n == 0:
        raise ValueError('Nonempty rank domain')
    return [(F(sum(y < x for y in values), 1) + F(sum(y == x for y in values) + 1, 2)) / n - F(1, 2)
            for x in values]


def canonical_choice(values, incumbent=None):
    best = max(values.values())
    winners = [key for key, value in values.items() if value == best]
    return incumbent if incumbent in winners else min(winners)


def toy_domain():
    # Invented scalar scores; no market returns, forecasts, source loading or fits.
    names = ('A', 'B', 'C', 'D', 'E')
    pre = dict(zip(names, map(F, ('0.30', '0.29', '0.28', '0.275', '0.27'))))
    final = dict(zip(names, map(F, ('0.30', '0.29', '0.28', '0.325', '0.32'))))
    core = {s: F('0.10') for s in names}
    ordered = sorted(names, key=lambda s: (-pre[s], s))
    boundary = [s for s in ordered[2:5] if pre[ordered[2]] - pre[s] <= F('.02')]
    third = min(boundary, key=lambda s: (-final[s], s))
    incumbent = tuple(sorted(ordered[:2] + [third]))
    cw = F(1, 2)
    reference_final = mean([final[s] for s in incumbent])
    reference_core = mean([core[s] / cw for s in incumbent])
    baskets = [tuple(sorted(row)) for row in combinations(ordered[:5], 3)]
    feasible = [row for row in baskets
                if mean([final[s] for s in row]) >= reference_final - F('.02')
                and mean([core[s] / cw for s in row]) >= reference_core - F('.02')]
    return names, pre, final, core, incumbent, feasible


class PortfolioBudgetAlgebraTests(unittest.TestCase):
    def test_funding_coefficients_sum_to_one(self):
        b = F(1, 40)
        for cw, factor in ((F(1, 2), F(19, 20)), (F(13, 25), F(91, 96))):
            self.assertEqual(1 - b / (1 - cw), factor)
            self.assertEqual(cw + factor * (1 - cw) + b, 1)

    def test_exact_replacement_identity(self):
        c, a, cw, b, q = F(1, 10), F(1, 5), F(1, 2), F(1, 40), F(3, 10)
        self.assertEqual(utility(c, a, cw, b, q) - (c + a), b * (q - a / (1 - cw)))

    def test_incumbent_is_inside_original_top5_cap_domain(self):
        _, _, _, _, inc, domain = toy_domain()
        self.assertIn(inc, domain)
        self.assertEqual(len(domain), 10)

    def test_structural_optimizer_is_not_original_selector(self):
        _, _, final, _, inc, domain = toy_domain()
        baseline = {row: mean([final[s] for s in row]) for row in domain}
        structural = canonical_choice(baseline, inc)
        self.assertEqual(inc, ('A', 'B', 'D'))
        self.assertEqual(structural, ('A', 'D', 'E'))
        self.assertNotEqual(structural, inc)

    def test_zero_budget_and_exact_noop_target_recover_structural(self):
        _, _, final, core, inc, domain = toy_domain()
        cw = F(1, 2)
        base, zero, noop = {}, {}, {}
        for row in domain:
            c = mean([core[s] for s in row])
            a = mean([final[s] - core[s] for s in row])
            base[row] = c + a
            zero[row] = utility(c, a, cw, F(0), F(1, 3))
            noop[row] = utility(c, a, cw, F(1, 40), a / (1 - cw))
        self.assertEqual(base, zero)
        self.assertEqual(base, noop)
        self.assertEqual(canonical_choice(zero, inc), canonical_choice(base, inc))

    def test_constant_quality_does_not_recover_structural(self):
        c = [F('.1'), F('.2')]
        a = [F('.2'), F('.099')]
        cw, b = F('.5'), F('.025')
        q = percentile([F(7), F(7)])
        self.assertEqual(q, [F(1, 4), F(1, 4)])
        self.assertGreater(c[0] + a[0], c[1] + a[1])
        self.assertLess(utility(c[0], a[0], cw, b, q[0]), utility(c[1], a[1], cw, b, q[1]))
        self.assertLess(utility(c[0], a[0], cw, b, F(0)), utility(c[1], a[1], cw, b, F(0)))

    def test_budget_is_not_unconditionally_delta_bound(self):
        c, a, cw, b, q = F(0), F(2), F('.5'), F('.025'), F('.5')
        delta = utility(c, a, cw, b, q) - (c + a)
        self.assertEqual(delta, F('-0.0875'))
        self.assertGreater(abs(delta), b)

    def test_score_loss_bound_uses_auxiliary_oscillation(self):
        c = [F('.1'), F('.2'), F('.19')]
        a = [F('.2'), F('.099'), F('.105')]
        q = percentile([F(2), F(3), F(1)])
        cw, b = F('.5'), F('.025')
        g = [x + y for x, y in zip(c, a)]
        u = [utility(x, y, cw, b, z) for x, y, z in zip(c, a, q)]
        structural, selected = max(range(3), key=g.__getitem__), max(range(3), key=u.__getitem__)
        delta = [x - y for x, y in zip(u, g)]
        s = [x / (1 - cw) for x in a]
        bound = b * ((max(q) - min(q)) + (max(s) - min(s)))
        self.assertLessEqual(g[structural] - g[selected], delta[selected] - delta[structural])
        self.assertLessEqual(max(delta) - min(delta), bound)

    def test_score_loss_bound_from_core_range_without_auxiliary_bound(self):
        c = [F('.1'), F('.2'), F('.19')]
        a = [F('.2'), F('.099'), F('.105')]
        q = percentile([F(2), F(3), F(1)])
        cw, b = F('.5'), F('.025')
        remainder = 1 - cw
        g = [x + y for x, y in zip(c, a)]
        u = [utility(x, y, cw, b, z) for x, y, z in zip(c, a, q)]
        structural, selected = max(range(3), key=g.__getitem__), max(range(3), key=u.__getitem__)
        bound = b / (remainder - b) * (max(c) - min(c))
        bound += remainder * b / (remainder - b) * (max(q) - min(q))
        self.assertLessEqual(g[structural] - g[selected], bound)

    def test_overlapping_three_of_five_mean_range_bound(self):
        values = [F(1, 2), F(1, 2), F(0), F(-1, 2), F(-1, 2)]
        basket_means = [mean(row) for row in combinations(values, 3)]
        self.assertEqual(max(basket_means) - min(basket_means), F(2, 3))
        self.assertEqual(max(basket_means) - min(basket_means),
                         F(2, 3) * (max(values) - min(values)))

    def test_nominal_majority_core_alone_is_not_nonconcession(self):
        cw, b = F('.52'), F('.025')
        c, a = [F('.26'), F(0)], [F('-.24'), F('.24')]
        u = [utility(x, y, cw, b, F(0)) for x, y in zip(c, a)]
        self.assertGreater(u[1], u[0])
        self.assertEqual((c[0] - c[1]) / cw, F('.5'))
        self.assertGreater((c[0] - c[1]) / cw, F('.02'))

    def test_admission_caps_give_core_and_final_score_bounds(self):
        eta = F('.02')
        for cw in (F('.5'), F('.52')):
            reference_core, reference_final = F('.2'), F('.3')
            proposals = [(F('.18'), F('.28')), (F('.179'), F('.3')), (F('.2'), F('.279'))]
            admitted = [(x, g) for x, g in proposals
                        if x >= reference_core - eta and g >= reference_final - eta]
            self.assertEqual(admitted, [(F('.18'), F('.28'))])
            for x, g in admitted:
                self.assertLessEqual(cw * (reference_core - x), cw * eta)
                self.assertLessEqual(reference_final - g, eta)

    def test_joint_and_diagonal_difference_has_no_funding_difference(self):
        c, a, cw, b = F('.1'), F('.2'), F('.5'), F('.025')
        joint, diagonal = F('.4'), F('-.2')
        self.assertEqual(utility(c, a, cw, b, joint) - utility(c, a, cw, b, diagonal),
                         b * (joint - diagonal))

    def test_rank_ties_and_permutation(self):
        points = [('A', F(3)), ('B', F(1)), ('C', F(3)), ('D', F(2))]
        reference = dict(zip([x[0] for x in points], percentile([x[1] for x in points])))
        reversed_points = list(reversed(points))
        other = dict(zip([x[0] for x in reversed_points], percentile([x[1] for x in reversed_points])))
        self.assertEqual(reference, other)
        self.assertEqual(reference['A'], reference['C'])
        self.assertEqual(percentile([F(0)]), [F(1, 2)])
        self.assertEqual(max(reference.values()) - min(reference.values()), F(5, 8))

    def test_tie_incumbent_preference_then_symbol_tuple(self):
        a, b, c = ('A', 'B', 'C'), ('A', 'B', 'D'), ('B', 'C', 'D')
        scores = {c: F(1), a: F(1), b: F(0)}
        self.assertEqual(canonical_choice(scores, c), c)
        self.assertEqual(canonical_choice(scores, b), a)
        self.assertEqual(canonical_choice(dict(reversed(list(scores.items()))), b), a)


if __name__ == '__main__':
    outcome = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PortfolioBudgetAlgebraTests))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'scope': 'invented_scalar_score_algebra_only', 'tests': outcome.testsRun,
                      'prices_read': False, 'market_returns_computed': False,
                      'strategy_execution': False, 'parameter_search': False}, sort_keys=True))
