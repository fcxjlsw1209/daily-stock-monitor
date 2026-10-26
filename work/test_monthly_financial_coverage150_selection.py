"""Synthetic-only checks; no source amounts, prices or actual orders read."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import monthly_financial_coverage150_selection as m


def row(name, cagr=.4, sharpe=1.5, dd=-.25, early=.3, recent=.5):
    return dict(name=name, training=dict(cagr=cagr, sharpe=sharpe, max_drawdown=dd),
                early=dict(cagr=early), recent=dict(cagr=recent))


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
                  eligible=np.ones((3, 49), dtype=bool), effects=np.zeros((3, 49)),
                  earnings_original=np.full((3, 49), np.nan),
                  earnings_expanded=np.full((3, 49), np.nan))
    arrays["earnings_original"][:, 0] = .2
    arrays["earnings_expanded"][:, :2] = [.2, -.3]
    return arrays, meta, Backend()


class SelectionTests(unittest.TestCase):
    def test_exact_frozen_correction_function(self):
        self.assertIs(m.corrected, m.old.corrected)
        self.assertEqual(m.DOSE, .01)

    def test_contract_keeps_three_execution_paths_and_split(self):
        contract = m.selection_contract()
        for key in ("execution_paths", "training_months", "early_months",
                    "recent_months", "later_months", "walkforward",
                    "max_sharpe_degradation", "max_drawdown_degradation",
                    "max_early_cagr_degradation", "max_recent_cagr_degradation"):
            self.assertEqual(contract[key], m.old.selection_contract()[key])
        self.assertFalse(contract["later34_for_selection"])

    def test_incumbent_bypasses_new_correction(self):
        arrays, meta, backend = fixture()
        result = m.choose(m.NAMES[0], arrays, meta, 3, backend, {})
        self.assertEqual(result[0], ["unchanged"])
        self.assertEqual(backend.baseline_calls, 1)
        self.assertEqual(backend.pre_calls, [])
        self.assertEqual(backend.adapter.calls, [])

    def test_different_masks_same_formula_dose_and_one_episode(self):
        arrays, meta, backend = fixture()
        original_bytes = {k: v.tobytes() for k, v in arrays.items()}
        _, old_scores = m.choose(m.NAMES[1], arrays, meta, 3, backend, {})
        _, new_scores = m.choose(m.NAMES[2], arrays, meta, 3, backend, {})
        self.assertEqual(len(backend.adapter.calls), 2)
        self.assertEqual(old_scores[:, 0].tobytes(), new_scores[:, 0].tobytes())
        np.testing.assert_array_equal(old_scores[:, 1], arrays["baseline"][:, 1])
        np.testing.assert_array_equal(new_scores[:, 1], arrays["baseline"][:, 1]-.003)
        for args, kwargs in backend.adapter.calls:
            self.assertEqual(args[:2], ("incumbent66", 0.))
            self.assertIs(kwargs["boundary_selector"], backend.selector)
        self.assertEqual(original_bytes, {k: v.tobytes() for k, v in arrays.items()})

    def test_frozen_corrected_called_with_declared_hand(self):
        arrays, meta, backend = fixture()
        with patch.object(m, "corrected", return_value=("choice", "score")) as fn:
            self.assertEqual(m.choose(m.NAMES[2], arrays, meta, 2, backend, {}),
                             ("choice", "score"))
        self.assertIs(fn.call_args.args[1], arrays["earnings_expanded"])
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
        arrays["eligible"][:, 1] = False
        _, score = m.choose(m.NAMES[2], arrays, meta, 3, backend, {})
        np.testing.assert_array_equal(score[:, 1], arrays["baseline"][:, 1])

    def test_every_synthetic_prefix_and_future_isolation(self):
        arrays, meta, backend = fixture()
        for name in m.NAMES[1:]:
            _, full = m.choose(name, arrays, meta, 3, backend, {})
            for n in range(4):
                _, prefix = m.choose(name, arrays, meta, n, backend, {})
                self.assertEqual(prefix.tobytes(), full[:n].tobytes())
        arrays["earnings_expanded"][2, 0] = np.inf
        _, score = m.choose(m.NAMES[2], arrays, meta, 2, backend, {})
        self.assertEqual(score.shape, (2, 49))

    def test_no_half_point_clip(self):
        arrays, meta, backend = fixture()
        arrays["earnings_expanded"][:, :2] = [-1., 1.]
        _, score = m.choose(m.NAMES[2], arrays, meta, 3, backend, {})
        np.testing.assert_array_equal(score[:, 0], arrays["baseline"][:, 0]-.01)
        np.testing.assert_array_equal(score[:, 1], arrays["baseline"][:, 1]+.01)

    def test_invalid_or_etf_hand_rejected(self):
        for col, value in ((0, np.inf), (0, 1.01), (32, 0.)):
            arrays, meta, backend = fixture()
            arrays["earnings_expanded"][0, col] = value
            with self.assertRaises(ValueError):
                m.choose(m.NAMES[2], arrays, meta, 3, backend, {})

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            m.choose("cashflow_control", *fixture()[:2], 3, fixture()[2], {})

    def test_both_sameclose_controls_require_strict_improvement(self):
        close = [row(m.NAMES[0], .4), row(m.NAMES[1], .42), row(m.NAMES[2], .42)]
        opened = [row(m.NAMES[0], .4), row(m.NAMES[1], .4), row(m.NAMES[2], .43)]
        self.assertFalse(m.select(close, opened)["admitted"])
        close[2] = row(m.NAMES[2], .43)
        self.assertEqual(m.select(close, opened)["primary"], m.NAMES[2])

    def test_open_incumbent_noninferior_but_original_strict(self):
        close = [row(m.NAMES[0], .4), row(m.NAMES[1], .4), row(m.NAMES[2], .43)]
        opened = [row(m.NAMES[0], .42), row(m.NAMES[1], .4), row(m.NAMES[2], .42)]
        self.assertTrue(m.select(close, opened)["admitted"])
        opened[1] = row(m.NAMES[1], .42)
        self.assertFalse(m.select(close, opened)["admitted"])

    def test_nextopen_regression_cannot_be_rescued_by_close(self):
        close = [row(m.NAMES[0]), row(m.NAMES[1]), row(m.NAMES[2], .9)]
        opened = [row(m.NAMES[0], .4), row(m.NAMES[1], .4), row(m.NAMES[2], .39)]
        self.assertEqual(m.select(close, opened)["primary"], m.NAMES[0])

    def test_risk_and_segment_guards_against_each_control_each_mode(self):
        for mode in ("close", "open"):
            for metric, value in (("sharpe", 1.44), ("dd", -.281),
                                  ("early", .274), ("recent", .474)):
                close = [row(m.NAMES[0]), row(m.NAMES[1]), row(m.NAMES[2], .8)]
                opened = [row(m.NAMES[0]), row(m.NAMES[1]), row(m.NAMES[2], .8)]
                selected = close if mode == "close" else opened
                selected[2] = row(m.NAMES[2], .8, **{metric: value})
                self.assertFalse(m.select(close, opened)["admitted"])

    def test_exact_arm_order_and_no_diagnostic_rescue(self):
        rows = [row(n) for n in m.NAMES]
        for wrong in (rows[::-1], rows[:2], rows+[row("diagnostic_winner", 99)]):
            with self.assertRaises(ValueError):
                m.select(rows, wrong)
        self.assertFalse(m.select(rows, rows)["admitted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
