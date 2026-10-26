"""Synthetic tests with only the real frozen pure boundary function injected.

Read/compile that single function from its source via AST. Never import its
driver, load research data, fit models, or compute strategy returns.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


P = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("path70_adapter_under_test", P/"incumbent_adapter.py")
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)
_EPISODE_PATH = P.parent/"monthly_daily_episodes"/"episodes.py"
_TREE = ast.parse(_EPISODE_PATH.read_text())
_SELECTOR_NODE = next(node for node in _TREE.body if isinstance(node, ast.FunctionDef)
                      and node.name == "select_boundary")
_NAMESPACE = {"np": np}
exec(compile(ast.Module(body=[_SELECTOR_NODE], type_ignores=[]), str(_EPISODE_PATH), "exec"), _NAMESPACE)
real_boundary = _NAMESPACE["select_boundary"]
POLICY = dict(id=7, family="level", model_id=2, budget=.05, ceiling=5, gap=.02)


def fixture():
    shape = (4, 6)
    baseline = np.tile([.24, .22, .20, .19, .181, .1805], (4, 1))
    cw = np.array([.52, .50, .52, .50])[:, None]
    support = np.tile([.1, -.1, .05, 0., -.05, .1], (4, 1))
    calm = np.array([False, True, False, True])
    weak = np.array([False, False, True, True])
    a = dict(baseline=baseline, core=baseline-(1-cw)*support, core_weight=cw,
             eligible=np.ones(shape, bool), state_calm10=calm, state_weak_narrow=weak)
    residual = np.tile([.1, .2, -.1, 0., .3, -.2], (4, 1))
    ml = np.tile([-.2, -.1, .1, .3, 0., -.3], (4, 1))
    pre = baseline+.025*calm[:, None]*(residual-support)+.05*weak[:, None]*(ml-support)
    effects = np.tile([0., 0., -.2, .1, .05, .5], (4, 1))
    factor = np.tile([-.5, -.3, -.1, .1, .3, .5], (4, 1))
    m = dict(symbols=[["A", "B", "C", "D", "E", "F"] for _ in range(4)])
    return a, m, pre, effects, factor, residual, ml


def boundary_mask(score, eligible, symbols):
    result = np.zeros(score.shape, bool)
    for i, names in enumerate(symbols):
        rank = sorted(np.flatnonzero(eligible[i]).tolist(), key=lambda j: (-score[i, j], names[j]))
        if len(rank) >= 3:
            result[i, [j for j in rank[2:5] if score[i, rank[2]]-score[i, j] <= .02]] = True
    return result


class IncumbentAdapterTests(unittest.TestCase):
    def setUp(self):
        self.a, self.m, self.pre, self.effects, self.factor, self.residual, self.ml = fixture()

    def choose(self, base="incumbent66", budget=.10, n=4, **kwargs):
        defaults = dict(pre_score=self.pre, effects=self.effects,
                        episode_policy=POLICY, boundary_selector=real_boundary)
        defaults.update(kwargs)
        return adapter.score_and_choose(base, budget, self.factor, self.a, self.m, n, **defaults)

    def original_incumbent(self, n=4):
        return real_boundary(self.pre[:n], self.a["eligible"][:n], self.m["symbols"][:n],
                             self.effects[:n], self.a, POLICY)

    def test_zero_hand_budget_exact_incumbent_and_original65(self):
        expected = self.original_incumbent()
        for budget in (0.,):
            actual = self.choose(budget=budget)
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(self.choose(base="original65", budget=0.)[1], self.a["baseline"])

    def test_all_missing_hand_exact_incumbent_and_original65(self):
        expected = self.original_incumbent()
        self.factor[:] = np.nan
        for value in (np.nan, np.inf, -np.inf):
            self.factor[:] = value
            actual = self.choose()
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(self.choose(base="original65")[1], self.a["baseline"])

    def test_finite_zero_rank_replaces_original_support_not_missing(self):
        self.factor[:] = 0.
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        actual = self.choose(base="original65", budget=.05)[1]
        np.testing.assert_array_equal(actual, self.a["baseline"]-.05*support)
        self.assertFalse(np.array_equal(actual, self.a["baseline"]))

    def test_finite_hand_only_changes_its_score_cells_before_boundary(self):
        self.factor[:, 1:4] = np.nan
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        actual = self.choose(base="original65", budget=.025)[1]
        np.testing.assert_array_equal(actual[:, 1:4], self.a["baseline"][:, 1:4])
        np.testing.assert_array_equal(actual[:, [0, 4, 5]],
            (self.a["baseline"]+.025*(self.factor-support))[:, [0, 4, 5]])

    def test_original_core_and_support_inputs_are_passed_without_feedback(self):
        captured = {}
        def capture(score, eligible, symbols, effects, original, policy):
            captured.update(score=score, original=original, effects=effects, policy=policy)
            return real_boundary(score, eligible, symbols, effects, original, policy)
        self.choose(boundary_selector=capture)
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        np.testing.assert_array_equal(captured["score"], self.pre+.1*(self.factor-support))
        np.testing.assert_array_equal(captured["original"]["baseline"], self.a["baseline"])
        np.testing.assert_array_equal(captured["original"]["core"], self.a["core"])
        np.testing.assert_array_equal(captured["effects"], self.effects)
        self.assertEqual(captured["policy"], POLICY)
        self.assertFalse(np.array_equal(captured["score"], captured["original"]["baseline"]))

    def test_weights_nonnegative_sum_one_and_exact_incumbent_score_decomposition(self):
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        for budget in (0., .025, .05, .10):
            with self.subTest(budget=budget):
                modified = self.pre+budget*(self.factor-support)
                mask = boundary_mask(modified, self.a["eligible"], self.m["symbols"])
                weights = adapter.budget_weights("incumbent66", budget, self.factor, self.a, 4, episode_mask=mask)
                for value in weights.values():
                    self.assertTrue((value >= 0).all())
                np.testing.assert_allclose(sum(weights.values()), np.ones(self.pre.shape), rtol=0, atol=1e-15)
                np.testing.assert_array_equal(weights["core"], np.broadcast_to(self.a["core_weight"], self.pre.shape))
                old_total = weights["prior_residual"]+weights["prior_ml"]+weights["episode"]
                self.assertLessEqual(float(old_total.max()), .10+1e-15)
                self.assertLessEqual(float((old_total+weights["hand"]).max()), .20+1e-15)
                self.assertGreaterEqual(float(weights["original_support"].min()), .28-1e-15)
                expected = (self.a["core"]+weights["original_support"]*support
                            +weights["prior_residual"]*self.residual+weights["prior_ml"]*self.ml
                            +weights["episode"]*np.clip(support+self.effects, -.5, .5)
                            +weights["hand"]*self.factor)
                np.testing.assert_allclose(self.choose(budget=budget)[1], expected, rtol=0, atol=1e-16)

    def test_conservative_reserved_budget_reaches_point28_and_missing_releases_hand(self):
        weights = adapter.budget_weights("incumbent66", .1, self.factor, self.a, 4)
        self.assertAlmostEqual(float(weights["original_support"][2, 0]), .28)
        self.assertAlmostEqual(float(weights["original_support"][3, 0]), .30)
        self.factor[2, 0] = np.nan
        missing = adapter.budget_weights("incumbent66", .1, self.factor, self.a, 4)
        self.assertEqual(missing["hand"][2, 0], 0.)
        self.assertAlmostEqual(missing["original_support"][2, 0], .38)

    def test_original65_weight_decomposition_has_no_old_corrections(self):
        weights = adapter.budget_weights("original65", .1, self.factor, self.a, 4)
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        expected = self.a["core"]+weights["original_support"]*support+weights["hand"]*self.factor
        np.testing.assert_allclose(self.choose(base="original65")[1], expected, rtol=0, atol=1e-16)
        for name in ("prior_residual", "prior_ml", "episode"):
            self.assertFalse(weights[name].any())

    def test_modified_prescore_can_change_all_three_members(self):
        self.a["baseline"][:] = [.22, .21, .20, .19, .18, .17]
        self.a["core"] = self.a["baseline"].copy()
        self.pre[:] = self.a["baseline"]
        self.a["state_calm10"][:] = False
        self.a["state_weak_narrow"][:] = False
        self.effects[:] = 0.
        self.factor[:] = [-.5, -.5, -.5, .5, .5, .5]
        self.assertEqual(self.original_incumbent()[0], [[0, 1, 2]]*4)
        chosen, _ = self.choose()
        self.assertEqual(chosen, [[3, 4, 5]]*4)

    def test_modified_pre_top_two_and_gap_use_new_anchor(self):
        support = (self.a["baseline"]-self.a["core"])/(1-self.a["core_weight"])
        modified = self.pre+.1*(self.factor-support)
        chosen, _ = self.choose()
        for i, names in enumerate(self.m["symbols"]):
            rank = sorted(np.flatnonzero(self.a["eligible"][i]), key=lambda j: (-modified[i, j], names[j]))
            allowed = {j for j in rank[2:5] if modified[i, rank[2]]-modified[i, j] <= .02}
            self.assertTrue(set(rank[:2]).issubset(chosen[i]))
            self.assertEqual(len(set(chosen[i])-set(rank[:2])), 1)
            self.assertTrue((set(chosen[i])-set(rank[:2])).issubset(allowed))

    def test_never_reranks_post_boundary_score_globally(self):
        self.a["baseline"][:] = [.30, .25, .20, .19, .181, .1805]
        self.a["core"] = self.a["baseline"].copy()
        self.pre[:] = self.a["baseline"]
        self.effects[:] = [0., 0., -.5, -.5, -.5, 0.]
        self.a["state_calm10"][:] = False
        self.a["state_weak_narrow"][:] = False
        chosen, score = self.choose(budget=0.)
        self.assertGreater(score[0, 5], score[0, 2])
        self.assertEqual(chosen, [[0, 1, 2]]*4)

    def test_eligibility_and_two_half_or_cash(self):
        for count in range(4):
            self.a["eligible"][:] = False
            self.a["eligible"][:, :count] = True
            for base in adapter.BASES:
                chosen, _ = self.choose(base=base)
                for row in chosen:
                    self.assertEqual(len(row), count if count >= 2 else 0)
                    self.assertTrue(all(j < count for j in row))

    def test_weight_shapes_monthly_column_and_full_are_equivalent(self):
        expected = self.choose()
        column = self.a["core_weight"].copy()
        for value in (column[:, 0], np.broadcast_to(column, self.pre.shape).copy()):
            self.a["core_weight"] = value
            got = self.choose()
            self.assertEqual(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])

    def test_inputs_and_original_injected_payload_cannot_be_mutated(self):
        before = copy.deepcopy((self.a, self.m, self.pre, self.effects, self.factor, POLICY))
        self.choose()[1][:] = -100.
        for old, new in zip(before[2:5], (self.pre, self.effects, self.factor)):
            np.testing.assert_array_equal(old, new)
        for key in self.a:
            np.testing.assert_array_equal(before[0][key], self.a[key])
        self.assertEqual(before[1], self.m)
        self.assertEqual(before[5], POLICY)

    def test_future_values_and_symbols_cannot_change_prefix(self):
        expected = self.choose(n=1)
        for key, value in self.a.items():
            value[1:] = False if value.dtype.kind == "b" else np.nan
        self.pre[1:] = np.nan
        self.effects[1:] = np.nan
        self.factor[1:] = np.nan
        self.m["symbols"][1:] = [[None]]*3
        actual = self.choose(n=1)
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_empty_prefix_and_bad_prefix_budget_or_policy_rejected(self):
        chosen, score = self.choose(n=0)
        self.assertEqual(chosen, [])
        self.assertEqual(score.shape, (0, 6))
        for n in (-1, 5, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.choose(n=n)
        for budget in (-.01, .1001, np.nan, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.choose(budget=budget)
        for name, value in (("budget", .10), ("ceiling", 4), ("gap", .01)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.choose(episode_policy=dict(POLICY, **{name: value}))
        with self.assertRaises(ValueError):
            self.choose(boundary_selector=None)

    def test_bad_full_shapes_masks_ranks_core_and_effects_rejected(self):
        self.factor = np.zeros((3, 6))
        with self.assertRaises(ValueError): self.choose(n=1)
        self.factor = np.zeros((4, 6))
        self.factor[0, 0] = .51
        with self.assertRaises(ValueError): self.choose()
        self.factor[0, 0] = 0.
        self.a["core_weight"] = np.full((4, 1), .51)
        with self.assertRaises(ValueError): self.choose()
        self.a["core_weight"] = np.full(6, .52)
        with self.assertRaises(ValueError): self.choose()
        self.a["core_weight"] = np.full((4, 1), .52)
        self.a["eligible"] = np.ones((4, 6))
        with self.assertRaises(ValueError): self.choose()
        self.a["eligible"] = np.ones((4, 6), bool)
        self.effects[0, 0] = np.nan
        with self.assertRaises(ValueError): self.choose()


if __name__ == "__main__":
    unittest.main()
