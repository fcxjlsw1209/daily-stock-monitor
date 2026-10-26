"""Synthetic-only verification of the prospective common-scenario selector."""
from copy import deepcopy
from itertools import combinations
import math
import unittest

import numpy as np

from monthly_formation142_selector import select_portfolio


def fixture(n=5):
    return dict(pre_score=np.linspace(.1, -.1, n),
                original_selected=[0, 1, 2],
                original_weighted_core=np.zeros(n), core_weight=.5,
                scenario_core_ranks=np.full((2, n), .1),
                eligible=np.ones(n, dtype=bool),
                symbol_keys=["S%02d" % i for i in range(n)], arm="worst_scenario")


class SelectorTests(unittest.TestCase):
    def call(self, **updates):
        args = fixture()
        args.update(updates)
        return select_portfolio(**args)

    def test_joint_alignment_changes_worst_not_marginals_or_mean(self):
        args = fixture(4)
        args.update(pre_score=np.array([.1, .08, .06, 0]),
                    original_weighted_core=np.array([.1, .08, .06, 0]),
                    symbol_keys=list("ABCD"),
                    scenario_core_ranks=np.array([[.5, .4, .4, -.3],
                                                  [.5, -.4, -.4, .3]]))
        first = args["scenario_core_ranks"].copy()
        second = first.copy()
        second[:, 2] = second[::-1, 2]
        np.testing.assert_array_equal(np.sort(first, axis=0), np.sort(second, axis=0))
        np.testing.assert_array_equal(first.min(axis=0), second.min(axis=0))
        np.testing.assert_array_equal(first.mean(axis=0), second.mean(axis=0))
        self.assertEqual(select_portfolio(**args)["selected_keys"], list("ABD"))
        args["scenario_core_ranks"] = second
        self.assertEqual(select_portfolio(**args)["selected_keys"], list("ABC"))
        args["arm"] = "mean_scenario"
        self.assertEqual(select_portfolio(**args)["selected_keys"], list("ABC"))
        args["scenario_core_ranks"] = first
        self.assertEqual(select_portfolio(**args)["selected_keys"], list("ABC"))

    def test_formula_replaces_weighted_core_not_additive_double_count(self):
        result = self.call(pre_score=[.1, .08, .06, .04, .02],
                           original_weighted_core=[.2, .2, .2, -.2, -.2],
                           scenario_core_ranks=[[0, 0, 0, 0, 0]])
        self.assertEqual(result["selected_indices"], [0, 3, 4])
        self.assertAlmostEqual(result["objective"], (-.1 + .24 + .22) / 3)

    def test_original_set_wins_objective_tie_despite_lower_original_scores(self):
        args = fixture()
        args["original_weighted_core"] = args["pre_score"].copy()
        args["original_selected"] = [4, 3, 2]
        result = select_portfolio(**args)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["selected_indices"], [2, 3, 4])

    def test_secondary_original_score_sum_tiebreak(self):
        result = self.call(pre_score=[.2, .1, .05, .03, 0],
                           original_weighted_core=[.2, .1, .05, .03, 0],
                           original_selected=[0, 1, 4],
                           scenario_core_ranks=[[.5, .4, .3, .3, -.4]])
        self.assertEqual(result["selected_indices"], [0, 1, 2])

    def test_tertiary_lexicographic_keys_tiebreak(self):
        result = self.call(pre_score=[.2, .1, .05, .05, 0],
                           original_weighted_core=[.2, .1, .05, .05, 0],
                           original_selected=[0, 1, 4], symbol_keys=["A", "B", "Z", "C", "E"],
                           scenario_core_ranks=[[.5, .4, .3, .3, -.4]])
        self.assertEqual(result["selected_keys"], ["A", "B", "C"])

    def test_objective_tolerance_prefers_original(self):
        result = self.call(pre_score=[0, 0, 0, 0, 0],
                           original_selected=[0, 1, 3],
                           scenario_core_ranks=[[.5, .4, .3, .3 - 5e-12, -.4]])
        self.assertEqual(result["selected_indices"], [0, 1, 3])

    def test_outside_objective_tolerance_does_not_prefer_original(self):
        result = self.call(pre_score=[0, 0, 0, 0, 0],
                           original_selected=[0, 1, 3],
                           scenario_core_ranks=[[.5, .4, .3, .3 - 8e-12, -.4]])
        self.assertEqual(result["selected_indices"], [0, 1, 2])

    def test_top_eight_plus_original_only(self):
        args = fixture(12)
        args["pre_score"] = np.arange(12, 0, -1) / 100
        args["original_selected"] = [0, 1, 10]
        args["scenario_core_ranks"][:, [8, 9, 11]] = .5
        result = select_portfolio(**args)
        self.assertEqual(result["candidate_indices"], list(range(8)) + [10])
        self.assertTrue(set(result["selected_indices"]).issubset(set(range(8)) | {10}))

    def test_original_three_outside_top_eight_are_all_preserved_as_candidates(self):
        args = fixture(12)
        args["original_selected"] = [9, 10, 11]
        result = select_portfolio(**args)
        self.assertEqual(result["candidate_indices"], list(range(8)) + [9, 10, 11])

    def test_top_eight_uses_only_eligible_even_when_ineligible_has_high_score(self):
        args = fixture(12)
        args["eligible"][[3, 4]] = False
        args["pre_score"][[3, 4]] = 100
        self.assertEqual(select_portfolio(**args)["candidate_indices"], [0, 1, 2, 5, 6, 7, 8, 9])

    def test_top_eight_boundary_tie_uses_stable_keys(self):
        args = fixture(10)
        args["pre_score"][:] = 0
        args["symbol_keys"] = list("JIHGFEDCBA")
        result = select_portfolio(**args)
        # Original J/I/H are unioned with lexical top eight A through H.
        self.assertEqual(result["candidate_keys"], list("ABCDEFGHIJ"))

    def test_columns_permutation_preserves_both_choices_and_objectives(self):
        args = fixture(10)
        rng = np.random.default_rng(142)
        args["scenario_core_ranks"] = rng.uniform(-.5, .5, (5, 10))
        permutation = rng.permutation(10)
        inverse = {int(old): new for new, old in enumerate(permutation)}
        for arm in ("mean_scenario", "worst_scenario"):
            args["arm"] = arm
            first = select_portfolio(**args)
            shuffled = deepcopy(args)
            for field in ("pre_score", "original_weighted_core", "eligible"):
                shuffled[field] = args[field][permutation]
            shuffled["symbol_keys"] = [args["symbol_keys"][i] for i in permutation]
            shuffled["original_selected"] = [inverse[i] for i in args["original_selected"]]
            shuffled["scenario_core_ranks"] = args["scenario_core_ranks"][:, permutation]
            second = select_portfolio(**shuffled)
            self.assertEqual(first["selected_keys"], second["selected_keys"])
            self.assertEqual(first["candidate_keys"], second["candidate_keys"])
            self.assertAlmostEqual(first["objective"], second["objective"], places=15)

    def test_row_permutation_preserves_objectives(self):
        args = fixture()
        args["scenario_core_ranks"] = np.random.default_rng(10).uniform(-.5, .5, (7, 5))
        for arm in ("mean_scenario", "worst_scenario"):
            args["arm"] = arm
            a = select_portfolio(**args)
            args["scenario_core_ranks"] = args["scenario_core_ranks"][::-1]
            b = select_portfolio(**args)
            self.assertEqual(a["selected_keys"], b["selected_keys"])
            self.assertAlmostEqual(a["objective"], b["objective"], places=15)

    def test_mean_matches_sum_of_individual_means_without_ties(self):
        rng = np.random.default_rng(919)
        for _ in range(20):
            args = fixture(7)
            args["arm"] = "mean_scenario"
            args["scenario_core_ranks"] = rng.uniform(-.5, .5, (9, 7))
            args["original_weighted_core"] = rng.uniform(-.1, .1, 7)
            means = args["pre_score"] - args["original_weighted_core"] + .5 * args["scenario_core_ranks"].mean(axis=0)
            expected = set(np.argsort(means)[-3:].tolist())
            self.assertEqual(set(select_portfolio(**args)["selected_indices"]), expected)

    def test_worst_matches_independent_bruteforce_synthetic_oracle(self):
        args = fixture(6)
        args["scenario_core_ranks"] = np.random.default_rng(300).uniform(-.5, .5, (11, 6))
        rows = args["pre_score"] + .5 * args["scenario_core_ranks"]
        oracle = {combo: np.mean(rows[:, combo], axis=1).min() for combo in combinations(range(6), 3)}
        expected = max(oracle, key=oracle.get)
        result = select_portfolio(**args)
        self.assertEqual(set(result["selected_indices"]), set(expected))
        self.assertAlmostEqual(result["objective"], oracle[expected])

    def test_allowed_core_weights(self):
        for weight in (.5, .52, np.float64(.52)):
            self.assertFalse(self.call(core_weight=weight)["fallback"])

    def test_no_input_mutation_and_output_does_not_alias(self):
        args = fixture()
        args["history_available"] = np.ones(5, bool)
        before = deepcopy(args)
        result = select_portfolio(**args)
        result["selected_indices"][0] = 99
        result["candidate_keys"][0] = "changed"
        for field, value in args.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, before[field])
            else:
                self.assertEqual(value, before[field])


class FallbackTests(unittest.TestCase):
    def fallback(self, expected, **updates):
        args = fixture()
        args["original_selected"] = [2, 0, 1]
        args.update(updates)
        result = select_portfolio(**args)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["reason"], expected)
        self.assertEqual(result["selected_indices"], args["original_selected"])
        self.assertIsNone(result["objective"])
        return result

    def test_none_scenarios(self):
        self.fallback("missing_scenarios", scenario_core_ranks=None)

    def test_empty_list_scenarios(self):
        self.fallback("empty_scenarios", scenario_core_ranks=[])

    def test_empty_matrix_scenarios(self):
        self.fallback("empty_scenarios", scenario_core_ranks=np.empty((0, 5)))

    def test_nan_one_cell(self):
        scenarios = np.full((2, 5), .1)
        scenarios[0, 4] = np.nan
        self.fallback("nan_scenario_values", scenario_core_ranks=scenarios)

    def test_nan_entire_row(self):
        self.fallback("nan_scenario_values", scenario_core_ranks=[[np.nan] * 5, [.1] * 5])

    def test_history_scalar_false(self):
        self.fallback("candidate_history_unavailable", history_available=False)

    def test_history_missing_candidate(self):
        self.fallback("candidate_history_unavailable", history_available=[True, True, True, False, True])

    def test_history_missing_non_candidate_does_not_force_fallback(self):
        args = fixture(10)
        args["history_available"] = np.ones(10, bool)
        args["history_available"][9] = False
        self.assertFalse(select_portfolio(**args)["fallback"])

    def test_noop_exact_core_preserves_original_order(self):
        self.fallback("unchanged_weighted_core", original_weighted_core=[.05] * 5)

    def test_noop_tolerance_preserves_original_order(self):
        self.fallback("unchanged_weighted_core", original_weighted_core=[.05 + 5e-13] * 5)

    def test_core_change_beyond_tolerance_activates(self):
        result = select_portfolio(**dict(fixture(), original_weighted_core=[.05 + 2e-12] * 5))
        self.assertFalse(result["fallback"])

    def test_noop_requires_all_scenarios_not_just_mean(self):
        result = select_portfolio(**dict(fixture(), scenario_core_ranks=[[.1] * 5, [-.1] * 5]))
        self.assertFalse(result["fallback"])

    def test_fewer_than_three_preserves_order_and_does_not_fill(self):
        for chosen in ([], [1], [2, 0]):
            self.fallback("original_fewer_than_three", original_selected=chosen)

    def test_zero_candidate_universe(self):
        result = select_portfolio([], [], [], .5, [], np.array([], dtype=bool), [], arm="worst_scenario")
        self.assertEqual(result["reason"], "original_fewer_than_three")
        self.assertEqual(result["selected_indices"], [])


class ValidationTests(unittest.TestCase):
    def bad(self, **updates):
        args = fixture()
        args.update(updates)
        with self.assertRaises(ValueError):
            select_portfolio(**args)

    def test_invalid_weights(self):
        for weight in (.49, .51, .53, 0, 1, True, "0.5", float("nan"), float("inf")):
            with self.subTest(weight=weight):
                self.bad(core_weight=weight)

    def test_invalid_arm(self):
        self.bad(arm="individual_minimum")

    def test_original_score_infinity(self):
        self.bad(pre_score=[0, 0, 0, float("inf"), 0])

    def test_original_core_infinity(self):
        self.bad(original_weighted_core=[0, 0, 0, -float("inf"), 0])

    def test_original_scores_nan_are_not_scenario_missingness(self):
        self.bad(pre_score=[0, 0, 0, np.nan, 0])
        self.bad(original_weighted_core=[0, 0, 0, np.nan, 0])

    def test_scenario_infinity_is_error_not_fallback(self):
        self.bad(scenario_core_ranks=[[0, 0, 0, np.inf, 0]])
        self.bad(scenario_core_ranks=[[0, 0, 0, -np.inf, np.nan]])

    def test_rank_bounds(self):
        self.bad(scenario_core_ranks=[[0, 0, 0, .500001, 0]])
        self.bad(scenario_core_ranks=[[0, 0, 0, -.500001, 0]])

    def test_rank_endpoints_valid(self):
        args = fixture()
        args["scenario_core_ranks"] = [[-.5, .5, 0, 0, 0]]
        self.assertFalse(select_portfolio(**args)["fallback"])

    def test_wrong_array_dimensions(self):
        self.bad(pre_score=[[0] * 5])
        self.bad(original_weighted_core=[[0] * 5])
        self.bad(scenario_core_ranks=[.1] * 5)
        self.bad(scenario_core_ranks=np.zeros((1, 2, 5)))

    def test_wrong_array_lengths(self):
        self.bad(original_weighted_core=[0] * 4)
        self.bad(scenario_core_ranks=[[0] * 4])
        self.bad(eligible=[True] * 4)
        self.bad(history_available=[True] * 4)

    def test_numeric_strings_booleans_complex_rejected(self):
        self.bad(pre_score=["0"] * 5)
        self.bad(scenario_core_ranks=[[True] * 5])
        self.bad(scenario_core_ranks=[[1j] * 5])

    def test_eligibility_must_be_booleans(self):
        self.bad(eligible=[1] * 5)
        self.bad(eligible=True)
        self.bad(history_available=[1] * 5)
        self.bad(history_available=1)

    def test_duplicate_out_of_range_and_noninteger_selected(self):
        for chosen in ([0, 0, 1], [-1, 1, 2], [0, 1, 5], [0., 1., 2.], [True, 1, 2], [0, 1, 2, 3]):
            with self.subTest(chosen=chosen):
                self.bad(original_selected=chosen)

    def test_selected_must_be_eligible(self):
        self.bad(eligible=[True, False, True, True, True])

    def test_invalid_keys(self):
        for keys in (["A"] * 5, ["A", "B", "C", "D", ""], [1, 2, 3, 4, 5], ["A"] * 4, "ABCDE", None):
            with self.subTest(keys=keys):
                self.bad(symbol_keys=keys)

    def test_invalid_schemas_not_bypassed_by_missing_scenarios(self):
        self.bad(scenario_core_ranks=None, original_selected=[0, 0, 1])

    def test_invalid_infinity_not_bypassed_by_fewer_holdings(self):
        self.bad(original_selected=[0], scenario_core_ranks=[[0, 0, 0, np.inf, 0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
