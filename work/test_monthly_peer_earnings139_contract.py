"""Synthetic-only tests; no research sources, prices, orders or returns are read."""
from copy import deepcopy
import math
import unittest

import monthly_peer_earnings139_contract as contract


def rows():
    return [dict(name=name, training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                 early=dict(cagr=.2), recent=dict(cagr=.2)) for name in contract.NAMES]


class SelectionTests(unittest.TestCase):
    def test_four_arm_roles_schema_and_fresh_metadata(self):
        schema = contract.selection_schema()
        self.assertEqual(len(contract.NAMES), 4)
        self.assertEqual(schema["candidate_index"], 3)
        self.assertEqual(schema["reference_indices"], [0, 1, 2])
        self.assertEqual(schema["primary_challenger"], "peer_event10")
        self.assertEqual(set(schema["roles"]), set(contract.NAMES))
        self.assertFalse(schema["controls_can_be_promoted"])
        schema["arms"].append("mutation")
        schema["roles"].clear()
        self.assertEqual(contract.selection_schema()["arms"], list(contract.NAMES))
        self.assertEqual(len(contract.selection_schema()["roles"]), 4)

    def test_candidate_must_pass_all_three_references(self):
        trial = rows()
        trial[3]["training"]["cagr"] = .21
        result = contract.select(trial)
        self.assertEqual(result, {"primary": "peer_event10", "admitted": True, "guards": True,
                                 "guards_against": {name: True for name in contract.REFERENCE_NAMES}})

    def test_third_reference_alone_can_reject_candidate(self):
        trial = rows()
        trial[3]["training"]["cagr"] = .21
        trial[2]["training"]["cagr"] = .22
        result = contract.select(trial)
        self.assertEqual(list(result["guards_against"].values()), [True, True, False])
        self.assertEqual(result["primary"], "incumbent_local67")
        self.assertFalse(result["admitted"])

    def test_controls_cannot_be_promoted_when_highest(self):
        for index in (1, 2):
            trial = rows()
            trial[index]["training"]["cagr"] = .9
            trial[3]["training"]["cagr"] = .21
            with self.subTest(control=index):
                self.assertEqual(contract.select(trial)["primary"], contract.NAMES[0])
        trial = rows()
        trial[1]["training"]["cagr"] = .8
        trial[2]["training"]["cagr"] = .9
        self.assertEqual(contract.select(trial)["primary"], contract.NAMES[0])

    def test_tied_candidate_falls_back(self):
        result = contract.select(rows())
        self.assertEqual(result["primary"], contract.NAMES[0])
        self.assertFalse(result["admitted"])

    def test_strict_cagr_epsilon_boundary_against_each_reference(self):
        for index in range(3):
            trial = rows()
            for row in trial:
                row["training"]["cagr"] = -1.
            trial[index]["training"]["cagr"] = 0.
            trial[3]["training"]["cagr"] = contract.CAGR_EPSILON
            with self.subTest(reference=index):
                self.assertFalse(contract.select(trial)["admitted"])
                trial[3]["training"]["cagr"] = math.nextafter(contract.CAGR_EPSILON, math.inf)
                self.assertTrue(contract.select(trial)["admitted"])

    def test_each_shortfall_boundary_is_inclusive_for_each_reference(self):
        for index in range(3):
            for (group, field), tolerance in zip(contract.METRIC_FIELDS[1:], contract.GUARD_TOLERANCES):
                trial = rows()
                trial[3]["training"]["cagr"] = .21
                # Equal references isolate this guard without changing other guards.
                boundary = trial[index][group][field] - tolerance
                trial[3][group][field] = boundary
                with self.subTest(reference=index, metric=field, group=group):
                    self.assertTrue(contract.select(trial)["admitted"])
                    trial[3][group][field] = math.nextafter(boundary, -math.inf)
                    self.assertFalse(contract.select(trial)["admitted"])

    def test_each_reference_shortfall_can_independently_reject(self):
        for index in range(3):
            for (group, field), tolerance in zip(contract.METRIC_FIELDS[1:], contract.GUARD_TOLERANCES):
                trial = rows()
                trial[3]["training"]["cagr"] = .21
                trial[index][group][field] += tolerance + .001
                with self.subTest(reference=index, group=group, metric=field):
                    result = contract.select(trial)
                    self.assertFalse(result["admitted"])
                    self.assertFalse(result["guards_against"][contract.NAMES[index]])
                    self.assertTrue(all(result["guards_against"][name]
                                        for j, name in enumerate(contract.REFERENCE_NAMES) if j != index))

    def test_nan_and_infinity_in_any_required_arm_metric_fail_closed(self):
        for arm in range(4):
            for group, field in contract.METRIC_FIELDS:
                for value in (math.nan, math.inf, -math.inf):
                    trial = rows()
                    trial[3]["training"]["cagr"] = .21
                    trial[arm][group][field] = value
                    with self.subTest(arm=arm, group=group, field=field, value=value):
                        self.assertFalse(contract.select(trial)["admitted"])
                        self.assertEqual(contract.select(trial)["primary"], contract.NAMES[0])

    def test_bool_string_missing_and_overflowing_number_fail_closed(self):
        for arm in range(4):
            for value in (True, False, "0.3", None, 10**1000):
                trial = rows()
                trial[3]["training"]["cagr"] = .21
                trial[arm]["training"]["cagr"] = value
                with self.subTest(arm=arm, value=repr(value)[:40]):
                    self.assertFalse(contract.select(trial)["admitted"])

    def test_missing_or_nonmapping_metric_group_fails_closed(self):
        for value in (None, "bad", [], {}):
            trial = rows()
            trial[3]["training"]["cagr"] = .21
            trial[2]["early"] = value
            self.assertFalse(contract.select(trial)["admitted"])
        trial = rows()
        del trial[3]["training"]["cagr"]
        self.assertFalse(contract.select(trial)["admitted"])

    def test_exact_four_arm_names_and_order_are_required(self):
        for trial in (rows()[:3], rows() + [rows()[0]], rows()[::-1], None, {}):
            with self.subTest(trial=trial), self.assertRaisesRegex(ValueError, "Four fixed arms"):
                contract.select(trial)
        trial = rows()
        trial[2]["name"] = trial[1]["name"]
        with self.assertRaises(ValueError):
            contract.select(trial)

    def test_later_period_cannot_rescue_and_inputs_are_unchanged(self):
        trial = rows()
        trial[3]["later"] = {"cagr": 1e9, "sharpe": 1e9}
        before = deepcopy(trial)
        self.assertEqual(contract.select(trial)["primary"], contract.NAMES[0])
        self.assertEqual(trial, before)
        trial[3]["training"]["cagr"] = .21
        trial[3]["later"] = {"cagr": math.nan}
        self.assertTrue(contract.select(trial)["admitted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
