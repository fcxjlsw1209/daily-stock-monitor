"""Synthetic-only checks; no source amounts, prices or actual orders read."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import monthly_financial_coverage153_selection as m


def row(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


def passing_rows():
    return [row(name, .5 if i == 5 else .4) for i, name in enumerate(m.NAMES)]


class Adapter:
    def __init__(self):
        self.calls = []

    def score_and_choose(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ["synthetic"], kwargs["pre_score"][:args[5]].copy()


class Backend:
    def __init__(self):
        self.adapter = Adapter()
        self.selector = object()
        self.baseline_calls = 0
        self.pre_calls = []

    def pre(self, arrays, n):
        self.pre_calls.append(n)
        return arrays["baseline"][:n].copy()

    def baseline(self, arrays, meta, n, policy):
        self.baseline_calls += 1
        return ["unchanged"], arrays["baseline"][:n].copy()


def fixture():
    stocks = ["STOCK" + str(i) for i in range(32)]
    etfs = ["ETF" + str(i) for i in range(17)]
    meta = dict(symbols=[stocks + etfs for _ in range(3)], etfs=etfs)
    arrays = dict(baseline=np.arange(147, dtype=np.float64).reshape(3, 49)/100,
                  eligible=np.ones((3, 49), dtype=bool), effects=np.zeros((3, 49)))
    values = [.2, -.3, .4, -.5, .6]
    for i, hand in enumerate(m.HANDS):
        arrays[hand] = np.full((3, 49), np.nan)
        arrays[hand][:, :i+1] = values[:i+1]
    return arrays, meta, Backend()


class SelectionTests(unittest.TestCase):
    def test_exact_declared_arms_and_hands(self):
        self.assertEqual(m.NAMES, ("incumbent_local67", "original_earnings", "batch1_earnings",
                                   "batch2_earnings", "batch3_earnings", "expanded_earnings"))
        self.assertEqual(m.HANDS, ("earnings_original", "earnings_batch1", "earnings_batch2",
                                   "earnings_batch3", "earnings_expanded"))

    def test_exact_frozen_correction_function(self):
        self.assertIs(m.corrected, m.old.corrected)
        self.assertEqual(m.DOSE, .01)

    def test_contract_binds_the_four_frozen_controls(self):
        contract = m.selection_contract()
        self.assertEqual(contract["old_coverage"], "frozen149_earnings_control_including_SBC_common_mask")
        self.assertEqual(contract["batch1_coverage"], "frozen150_expanded_earnings_and_mask")
        self.assertEqual(contract["batch2_coverage"], "frozen151_expanded_earnings_and_mask")
        self.assertEqual(contract["batch3_coverage"], "frozen152_expanded_earnings_and_mask")
        self.assertEqual(contract["coverage_masks"], "distinct_original_batch1_batch2_batch3_and_expanded_masks")
        self.assertEqual(contract["dose"], .01)
        self.assertEqual(contract["episode"], "one_original_episode_selector_after_same_original_pre_score")

    def test_contract_keeps_three_execution_paths_splits_and_all_other_guards(self):
        original = m.old.selection_contract()
        contract = m.selection_contract()
        for key, value in original.items():
            if key not in ("same_close_cagr", "next_open_cost_cagr"):
                self.assertEqual(contract[key], value, key)
        self.assertEqual(contract["execution_paths"], [
            dict(execution="close", cost=0., key="same_close"),
            dict(execution="open", cost=0., key="next_open_zero"),
            dict(execution="open", cost=.0005, key="next_open_cost")])
        self.assertFalse(contract["later34_for_selection"])
        self.assertEqual(m.old.selection_contract(), original)

    def test_incumbent_bypasses_new_correction(self):
        arrays, meta, backend = fixture()
        result = m.choose(m.NAMES[0], arrays, meta, 3, backend, {})
        self.assertEqual(result[0], ["unchanged"])
        self.assertEqual(backend.baseline_calls, 1)
        self.assertEqual(backend.pre_calls, [])
        self.assertEqual(backend.adapter.calls, [])

    def test_different_masks_same_dose_and_one_episode_without_mutating_inputs(self):
        arrays, meta, backend = fixture()
        original_bytes = {k: v.tobytes() for k, v in arrays.items()}
        values = [.2, -.3, .4, -.5, .6]
        policy = {"synthetic": True}
        for i, name in enumerate(m.NAMES[1:]):
            _, scores = m.choose(name, arrays, meta, 3, backend, policy)
            expected = arrays["baseline"].copy()
            expected[:, :i+1] += .01 * np.array(values[:i+1])
            np.testing.assert_array_equal(scores, expected)
        self.assertEqual(backend.pre_calls, [3] * 5)
        self.assertEqual(len(backend.adapter.calls), 5)
        self.assertEqual(backend.baseline_calls, 0)
        for args, kwargs in backend.adapter.calls:
            self.assertEqual(args[:2], ("incumbent66", 0.))
            self.assertIs(kwargs["boundary_selector"], backend.selector)
            self.assertIs(kwargs["episode_policy"], policy)
        self.assertEqual(original_bytes, {k: v.tobytes() for k, v in arrays.items()})

    def test_frozen_corrected_called_with_each_declared_hand(self):
        arrays, meta, backend = fixture()
        for name, hand in zip(m.NAMES[1:], m.HANDS):
            with self.subTest(name=name):
                with patch.object(m, "corrected", return_value=("choice", "score")) as fn:
                    self.assertEqual(m.choose(name, arrays, meta, 2, backend, {}), ("choice", "score"))
                self.assertIs(fn.call_args.args[1], arrays[hand])
                self.assertEqual(fn.call_args.args[4], 2)

    def test_zero_and_missing_exact_per_hand_fallback(self):
        for value in (0., np.nan):
            arrays, meta, backend = fixture()
            for hand in m.HANDS:
                arrays[hand][:] = np.nan
                arrays[hand][:, :32] = value
            for name in m.NAMES[1:]:
                _, score = m.choose(name, arrays, meta, 3, backend, {})
                self.assertEqual(score.tobytes(), arrays["baseline"].tobytes())

    def test_ineligible_new_coverage_does_not_modify_score(self):
        arrays, meta, backend = fixture()
        arrays["eligible"][:, :5] = False
        for name in m.NAMES[1:]:
            _, score = m.choose(name, arrays, meta, 3, backend, {})
            self.assertEqual(score.tobytes(), arrays["baseline"].tobytes())

    def test_every_synthetic_prefix_and_future_isolation(self):
        arrays, meta, backend = fixture()
        for name, hand in zip(m.NAMES[1:], m.HANDS):
            _, full = m.choose(name, arrays, meta, 3, backend, {})
            for n in range(4):
                _, prefix = m.choose(name, arrays, meta, n, backend, {})
                self.assertEqual(prefix.tobytes(), full[:n].tobytes())
            arrays[hand][2, 0] = np.inf
            _, prefix = m.choose(name, arrays, meta, 2, backend, {})
            self.assertEqual(prefix.tobytes(), full[:2].tobytes())

    def test_no_half_point_clip(self):
        arrays, meta, backend = fixture()
        arrays["earnings_expanded"][:, :2] = [-1., 1.]
        _, score = m.choose(m.NAMES[5], arrays, meta, 3, backend, {})
        np.testing.assert_array_equal(score[:, 0], arrays["baseline"][:, 0]-.01)
        np.testing.assert_array_equal(score[:, 1], arrays["baseline"][:, 1]+.01)

    def test_invalid_or_etf_hand_rejected_for_every_earnings_arm(self):
        for name, hand in zip(m.NAMES[1:], m.HANDS):
            for col, value in ((0, np.inf), (0, 1.01), (32, 0.)):
                with self.subTest(name=name, col=col, value=value):
                    arrays, meta, backend = fixture()
                    arrays[hand][0, col] = value
                    with self.assertRaises(ValueError):
                        m.choose(name, arrays, meta, 3, backend, {})

    def test_unknown_arm_rejected(self):
        arrays, meta, backend = fixture()
        with self.assertRaises(ValueError):
            m.choose("cashflow_control", arrays, meta, 3, backend, {})

    def test_sameclose_requires_strict_improvement_against_all_five_controls(self):
        for i in range(5):
            with self.subTest(control=m.NAMES[i]):
                close, opened = passing_rows(), passing_rows()
                close[i] = row(m.NAMES[i], .5)
                result = m.select(close, opened)
                self.assertFalse(result["admitted"])
                self.assertEqual(result["primary"], m.NAMES[0])
                self.assertFalse(result["guards_against"][m.NAMES[i]])
                self.assertTrue(all(v for name, v in result["guards_against"].items() if name != m.NAMES[i]))

    def test_nextopen_incumbent_noninferior_and_other_four_strict(self):
        close, opened = passing_rows(), passing_rows()
        opened[0] = row(m.NAMES[0], .5)
        self.assertEqual(m.select(close, opened)["primary"], m.NAMES[5])
        for i in (1, 2, 3, 4):
            with self.subTest(control=m.NAMES[i]):
                opened = passing_rows()
                opened[i] = row(m.NAMES[i], .5)
                result = m.select(close, opened)
                self.assertFalse(result["admitted"])
                self.assertFalse(result["next_open_guards_against"][m.NAMES[i]])
                self.assertTrue(all(v for name, v in result["next_open_guards_against"].items() if name != m.NAMES[i]))

    def test_frozen_strict_epsilon_and_noninferior_tolerance(self):
        close, opened = passing_rows(), passing_rows()
        close[3] = row(m.NAMES[3], .5 - .5e-12)
        self.assertFalse(m.select(close, opened)["admitted"])
        close[3] = row(m.NAMES[3], .5 - 2e-12)
        opened[0] = row(m.NAMES[0], .5 + .5e-12)
        self.assertTrue(m.select(close, opened)["admitted"])
        opened[0] = row(m.NAMES[0], .5 + 2e-12)
        self.assertFalse(m.select(close, opened)["admitted"])

    def test_nextopen_regression_cannot_be_rescued_by_close(self):
        close, opened = passing_rows(), passing_rows()
        close[5] = row(m.NAMES[5], .9)
        opened[5] = row(m.NAMES[5], .39)
        self.assertEqual(m.select(close, opened)["primary"], m.NAMES[0])

    def test_risk_and_segment_guards_against_each_control_each_mode(self):
        for mode, key in (("close", "guards_against"), ("open", "next_open_guards_against")):
            for i in range(5):
                for metric, value in (("sharpe", 1.551), ("dd", -.219),
                                      ("early", .326), ("recent", .526)):
                    with self.subTest(mode=mode, control=m.NAMES[i], metric=metric):
                        close, opened = passing_rows(), passing_rows()
                        group = close if mode == "close" else opened
                        group[i] = row(m.NAMES[i], .4, **{metric: value})
                        result = m.select(close, opened)
                        self.assertFalse(result["admitted"])
                        self.assertFalse(result[key][m.NAMES[i]])
                        self.assertTrue(all(v for name, v in result[key].items() if name != m.NAMES[i]))

    def test_invalid_summary_metrics_cannot_pass(self):
        for side in (0, 1):
            for index in (0, 1, 2, 3, 4, 5):
                for value in (np.nan, np.inf, True, "0.5", None):
                    with self.subTest(side=side, index=index, value=value):
                        groups = [passing_rows(), passing_rows()]
                        groups[side][index]["training"]["cagr"] = value
                        self.assertFalse(m.select(*groups)["admitted"])

    def test_exact_arm_order_cardinality_and_no_extra_diagnostic_rescue(self):
        valid = passing_rows()
        duplicate = valid[:-1] + [row(m.NAMES[3], .9)]
        for wrong in (valid[::-1], valid[:4], valid + [row("diagnostic_winner", 99)], duplicate):
            for groups in ((wrong, valid), (valid, wrong)):
                with self.assertRaises(ValueError):
                    m.select(*groups)

    def test_selection_does_not_mutate_summaries(self):
        close, opened = passing_rows(), passing_rows()
        before = copy.deepcopy((close, opened))
        self.assertTrue(m.select(close, opened)["admitted"])
        self.assertEqual((close, opened), before)

    def test_mechanism_confirmation_does_not_override_failed_sameclose(self):
        close, opened = passing_rows(), passing_rows()
        close[3] = row(m.NAMES[3], .5)
        result = m.select(close, opened)
        self.assertTrue(result["mechanism_confirmed"])
        self.assertFalse(result["admitted"])
        self.assertEqual(result["primary"], m.NAMES[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
