"""Synthetic core and frozen-boundary tests; no strategy data or returns."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


P = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("timing_core_adapter_under_test", P / "core_adapter.py")
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)
_EPISODE_PATH = P.parent / "monthly_daily_episodes" / "episodes.py"
_TREE = ast.parse(_EPISODE_PATH.read_text())
_SELECTOR = next(node for node in _TREE.body if isinstance(node, ast.FunctionDef)
                 and node.name == "select_boundary")
_NAMESPACE = {"np": np}
exec(compile(ast.Module(body=[_SELECTOR], type_ignores=[]), str(_EPISODE_PATH), "exec"), _NAMESPACE)
real_boundary = _NAMESPACE["select_boundary"]
POLICY = dict(id=7, family="level", model_id=2, budget=.05, ceiling=5, gap=.02)


def fixture():
    cw = np.array([.52, .50, .52, .50])[:, None]
    old_rank = np.tile([.40, .35, .30, .28, .26, .25], (4, 1))
    support = np.tile([.10, .08, .05, .03, .02, .01], (4, 1))
    core = cw * old_rank
    baseline = core + (1-cw) * support
    calm = np.array([False, True, False, True])
    weak = np.array([False, False, True, True])
    a = dict(baseline=baseline, core=core, core_weight=cw,
             eligible=np.ones((4, 6), bool), state_calm10=calm,
             state_weak_narrow=weak)
    residual = np.tile([.1, .2, -.1, 0., .3, -.2], (4, 1))
    ml = np.tile([-.2, -.1, .1, .3, 0., -.3], (4, 1))
    pre = baseline + .025*calm[:, None]*(residual-support) + .05*weak[:, None]*(ml-support)
    effects = np.tile([0., 0., -.2, .1, .05, .5], (4, 1))
    rank = np.tile([-.5, -.3, -.1, .1, .3, .5], (4, 1))
    m = dict(symbols=[["A", "B", "C", "D", "E", "F"] for _ in range(4)])
    return a, m, pre, effects, rank


class CoreAdapterTests(unittest.TestCase):
    def setUp(self):
        self.a, self.m, self.pre, self.effects, self.rank = fixture()

    def choose(self, base="incumbent66", blend=.5, n=4, **kwargs):
        options = dict(pre_score=self.pre, effects=self.effects,
                       episode_policy=POLICY, boundary_selector=real_boundary)
        options.update(kwargs)
        return adapter.score_and_choose(base, blend, self.rank, self.a, self.m, n, **options)

    def old(self, base="incumbent66", n=4):
        return adapter.path_adapter.score_and_choose(
            base, 0., np.full(self.pre.shape, np.nan), self.a, self.m, n,
            pre_score=self.pre, effects=self.effects, episode_policy=POLICY,
            boundary_selector=real_boundary)

    def delta(self, blend=.5, n=4):
        cw = np.broadcast_to(self.a["core_weight"], self.pre.shape)
        result = np.zeros((n, self.pre.shape[1]))
        valid = np.isfinite(self.rank[:n])
        result[valid] = blend*(cw[:n][valid]*self.rank[:n][valid]-self.a["core"][:n][valid])
        return result

    def test_zero_blend_is_bitwise_original_for_both_bases(self):
        for base in adapter.path_adapter.BASES:
            expected = self.old(base)
            got = self.choose(base=base, blend=0.)
            self.assertEqual(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])

    def test_all_nonfinite_rank_is_bitwise_original(self):
        for value in (np.nan, np.inf, -np.inf):
            self.rank[:] = value
            for base in adapter.path_adapter.BASES:
                expected = self.old(base)
                got = self.choose(base=base, blend=1.)
                self.assertEqual(got[0], expected[0])
                np.testing.assert_array_equal(got[1], expected[1])

    def test_half_and_full_rank_blend_change_only_weighted_core(self):
        for blend in (.5, 1.):
            delta = self.delta(blend)
            score = self.choose(base="original65", blend=blend)[1]
            np.testing.assert_array_equal(score, self.a["baseline"]+delta)
            new_core = self.a["core"]+delta
            expected = (1-blend)*self.a["core"]+blend*self.a["core_weight"]*self.rank
            np.testing.assert_allclose(new_core, expected, rtol=0, atol=1e-16)

    def test_neutral_zero_rank_is_valid_core_replacement(self):
        self.rank[:] = 0.
        score = self.choose(base="original65", blend=1.)[1]
        np.testing.assert_array_equal(score, self.a["baseline"]-self.a["core"])
        self.assertFalse(np.array_equal(score, self.a["baseline"]))

    def test_partial_missing_applies_zero_delta_without_changing_eligibility(self):
        self.rank[:, 1:4] = np.nan
        captured = {}
        def capture(score, eligible, symbols, effects, original, policy):
            captured.update(score=score.copy(), eligible=eligible.copy())
            return real_boundary(score, eligible, symbols, effects, original, policy)
        self.choose(boundary_selector=capture)
        np.testing.assert_array_equal(captured["score"], self.pre+self.delta())
        np.testing.assert_array_equal(captured["score"][:, 1:4], self.pre[:, 1:4])
        np.testing.assert_array_equal(captured["eligible"], self.a["eligible"])

    def test_frozen_adapter_receives_full_copies_same_delta_and_zero_hand_budget(self):
        real = adapter.path_adapter.score_and_choose
        before = copy.deepcopy(self.a)
        captured = {}
        def capture(base, budget, factor, a, m, n, **kwargs):
            captured.update(budget=budget, factor=factor.copy(), a=copy.deepcopy(a),
                            pre=kwargs["pre_score"].copy())
            return real(base, budget, factor, a, m, n, **kwargs)
        with patch.object(adapter.path_adapter, "score_and_choose", side_effect=capture):
            self.choose(n=2)
        self.assertEqual(captured["budget"], 0.)
        self.assertTrue(np.isnan(captured["factor"]).all())
        self.assertEqual(captured["factor"].shape, self.pre.shape)
        for name in ("baseline", "core"):
            np.testing.assert_array_equal(captured["a"][name][:2], before[name][:2]+self.delta(n=2))
            np.testing.assert_array_equal(captured["a"][name][2:], before[name][2:])
        np.testing.assert_array_equal(captured["pre"][:2], self.pre[:2]+self.delta(n=2))
        np.testing.assert_array_equal(captured["pre"][2:], self.pre[2:])
        np.testing.assert_array_equal(captured["a"]["core_weight"][:2],
                                      np.broadcast_to(before["core_weight"][:2], (2, 6)))
        for name in ("eligible", "state_calm10", "state_weak_narrow"):
            np.testing.assert_array_equal(captured["a"][name], before[name])

    def test_true_boundary_gets_exact_original_support_effects_states_and_policy(self):
        captured = {}
        def capture(score, eligible, symbols, effects, original, policy):
            captured.update(a=copy.deepcopy(original), effects=effects.copy(), policy=copy.deepcopy(policy))
            return real_boundary(score, eligible, symbols, effects, original, policy)
        got = self.choose(boundary_selector=capture)
        expected = real_boundary(self.pre+self.delta(), self.a["eligible"], self.m["symbols"],
                                 self.effects, self.a, POLICY)
        self.assertEqual(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
        for name in ("baseline", "core", "state_calm10", "state_weak_narrow"):
            np.testing.assert_array_equal(captured["a"][name], self.a[name])
        np.testing.assert_array_equal(captured["a"]["core_weight"],
                                      np.broadcast_to(self.a["core_weight"], self.pre.shape))
        np.testing.assert_array_equal(captured["effects"], self.effects)
        self.assertEqual(captured["policy"], POLICY)

    def test_cancellation_does_not_change_support_by_one_ulp(self):
        self.a["baseline"][:] = -.1
        self.a["core"][:] = -.25
        self.a["core_weight"][:] = .5
        self.a["state_calm10"][:] = False
        self.a["state_weak_narrow"][:] = False
        self.pre[:] = -.1
        self.rank[:] = -.3
        self.effects[:] = .1
        modified = self.pre+self.delta()
        changed_a = dict(self.a, baseline=self.a["baseline"]+self.delta(), core=self.a["core"]+self.delta())
        expected = real_boundary(modified, self.a["eligible"], self.m["symbols"], self.effects, self.a, POLICY)
        recomputed = real_boundary(modified, self.a["eligible"], self.m["symbols"], self.effects, changed_a, POLICY)
        self.assertFalse(np.array_equal(expected[1], recomputed[1]))
        got = self.choose()
        self.assertEqual(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])

    def test_modified_pre_can_change_all_three_members(self):
        self.a["core"][:] = [.22, .21, .20, .19, .18, .17]
        self.a["baseline"] = self.a["core"].copy()
        self.pre = self.a["baseline"].copy()
        self.effects[:] = 0.
        self.rank[:] = [-.5, -.5, -.5, .5, .5, .5]
        self.assertEqual(self.old()[0], [[0, 1, 2]]*4)
        self.assertEqual(self.choose(blend=1.)[0], [[3, 4, 5]]*4)

    def test_modified_pre_top_two_and_gap_are_the_actual_boundary_anchors(self):
        modified = self.pre+self.delta()
        chosen, _ = self.choose()
        for i, names in enumerate(self.m["symbols"]):
            order = sorted(np.flatnonzero(self.a["eligible"][i]), key=lambda j: (-modified[i, j], names[j]))
            allowed = {j for j in order[2:5] if modified[i, order[2]]-modified[i, j] <= .02}
            self.assertTrue(set(order[:2]).issubset(chosen[i]))
            self.assertTrue((set(chosen[i])-set(order[:2])).issubset(allowed))

    def test_postscore_is_never_globally_reranked(self):
        self.a["baseline"][:] = [.25, .24, .20, .19, .181, .1805]
        self.a["core"] = self.a["baseline"].copy()
        self.pre = self.a["baseline"].copy()
        self.a["state_calm10"][:] = False
        self.a["state_weak_narrow"][:] = False
        self.effects[:] = [0., 0., -.5, -.5, -.5, 0.]
        self.rank = self.a["core"]/self.a["core_weight"]
        chosen, score = self.choose()
        self.assertGreater(score[0, 5], score[0, 2])
        self.assertEqual(chosen, [[0, 1, 2]]*4)

    def test_eligibility_two_positions_or_cash_are_preserved(self):
        for count in range(4):
            self.a["eligible"][:] = False
            self.a["eligible"][:, :count] = True
            for base in adapter.path_adapter.BASES:
                chosen, _ = self.choose(base=base)
                for row in chosen:
                    self.assertEqual(len(row), count if count >= 2 else 0)
                    self.assertTrue(all(j < count for j in row))

    def test_core_weight_supported_shapes_are_identical(self):
        expected = self.choose()
        cw = self.a["core_weight"].copy()
        for value in (cw[:, 0], np.broadcast_to(cw, self.pre.shape).copy()):
            self.a["core_weight"] = value
            got = self.choose()
            self.assertEqual(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])

    def test_future_values_symbols_and_invalid_suffix_ranks_cannot_affect_prefix(self):
        expected = self.choose(n=1)
        for key, value in self.a.items():
            if value.dtype.kind == "b":
                value[1:] = False
            else:
                value[1:] = np.nan
        self.pre[1:] = np.nan
        self.effects[1:] = np.nan
        self.rank[1:] = 999.
        self.m["symbols"][1:] = [[None]]*3
        got = self.choose(n=1)
        self.assertEqual(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])

    def test_non_numeric_future_values_are_not_converted(self):
        expected = self.choose(n=1)
        for key in ("baseline", "core", "core_weight"):
            self.a[key] = self.a[key].astype(object)
            self.a[key][1:] = "unused future"
        for name in ("pre", "effects", "rank"):
            value = getattr(self, name).astype(object)
            value[1:] = "unused future"
            setattr(self, name, value)
        got = self.choose(n=1)
        self.assertEqual(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])

    def test_inputs_and_nested_payload_are_not_mutated(self):
        before = copy.deepcopy((self.a, self.m, self.pre, self.effects, self.rank, POLICY))
        def mutate(score, eligible, symbols, effects, original, policy):
            result = real_boundary(score, eligible, symbols, effects, original, policy)
            score[:] = 50
            eligible[:] = False
            effects[:] = 20
            symbols[0][0] = "MUTATED"
            for value in original.values(): value[:] = 0
            policy["gap"] = 100
            return result
        self.choose(boundary_selector=mutate)[1][:] = -100.
        for key in self.a: np.testing.assert_array_equal(self.a[key], before[0][key])
        self.assertEqual(self.m, before[1])
        for value, old in zip((self.pre, self.effects, self.rank), before[2:5]):
            np.testing.assert_array_equal(value, old)
        self.assertEqual(POLICY, before[5])

    def test_empty_prefix_and_invalid_blends_or_prefixes(self):
        chosen, score = self.choose(n=0)
        self.assertEqual(chosen, [])
        self.assertEqual(score.shape, (0, 6))
        for value in (True, -.5, .25, 1.01, np.nan, np.inf, [0.]):
            with self.subTest(blend=value), self.assertRaises(ValueError): self.choose(blend=value)
        for n in (-1, 5, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError): self.choose(n=n)

    def test_exact_full_shapes_and_bounded_ranks_are_required(self):
        original = self.rank.copy()
        for value in (np.zeros((1, 6)), np.zeros((4, 1)), np.zeros(6)):
            self.rank = value
            with self.assertRaises(ValueError): self.choose(n=1)
        self.rank = original.copy()
        self.rank[0, 0] = .5001
        with self.assertRaises(ValueError): self.choose()
        self.rank = original.copy()
        self.a["core_weight"] = np.full((4, 1), .51)
        with self.assertRaises(ValueError): self.choose()
        self.a["core_weight"] = np.full((4, 1), .52)
        for base in adapter.path_adapter.BASES:
            with self.assertRaises(ValueError): self.choose(base=base, n=1, pre_score=self.pre[:1])
            with self.assertRaises(ValueError): self.choose(base=base, n=1, effects=self.effects[:1])

    def test_frozen_episode_policy_and_required_inputs_stay_enforced(self):
        for name in ("pre_score", "effects", "episode_policy", "boundary_selector"):
            with self.subTest(name=name), self.assertRaises(ValueError): self.choose(**{name: None})
        with self.assertRaises(ValueError): self.choose(episode_policy=dict(POLICY, gap=.03))
        with self.assertRaises(ValueError): self.choose(base="unknown")


if __name__ == "__main__":
    unittest.main()
