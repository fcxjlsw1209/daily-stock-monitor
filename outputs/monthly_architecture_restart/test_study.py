"""Frozen design, causal features, and full-candidate monthly ranking checks."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
# Older experiments also use the generic module name study; isolate this driver.
_spec = importlib.util.spec_from_file_location("monthly_restart_study_for_tests", P / "study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def synthetic_panel(rows=420):
    rng = np.random.default_rng(331)
    dates = pd.bdate_range("2018-01-01", periods=rows)
    symbols = ["A", "B", "C", "D", "SPY", "XLK"]
    returns = rng.normal(.001, .012, size=(rows, len(symbols)))
    px = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=symbols)
    volume = pd.DataFrame(rng.uniform(1e5, 2e6, size=px.shape), index=dates, columns=symbols)
    return {"adj": px, "close": px.copy(), "high": px * 1.02,
            "low": px * .985, "volume": volume,
            "sectors": pd.DataFrame("Information Technology", index=dates, columns=symbols)}


def synthetic_scores():
    scores = np.array([[.49, .48, .47, .46, .45, .44]])
    a = {"eligible": np.ones(scores.shape, dtype=bool)}
    for normalization in study.NORMS:
        prefix = "event_" + normalization + "_"
        for family in study.FAMILIES + ["hand"]:
            a[prefix + family] = scores.copy()
    predictions = {}
    for architecture in study.architectures():
        for arm in ("direct", "residual"):
            predictions[str(architecture["id"]) + "_" + arm] = np.zeros(scores.shape)
    return a, predictions, {"symbols": [["A", "B", "C", "D", "E", "F"]]}


class TestStudySpecification(unittest.TestCase):
    def test_exactly_eighty_unique_configs_and_budget_ceiling(self):
        grid = study.grid()
        self.assertEqual(len(grid), 80)
        self.assertEqual(len({json.dumps(row, sort_keys=True) for row in grid}), 80)
        self.assertEqual([row["id"] for row in grid], list(range(80)))
        self.assertEqual(len(study.architectures()), 10)
        for row in grid:
            self.assertLessEqual(row["hand_budget"] + row["ml_budget"], .10000000001)
            self.assertGreaterEqual(row["hand_budget"], 0.)
            self.assertGreaterEqual(row["ml_budget"], 0.)

    def test_native_forecast_bounded_without_rank_amplification(self):
        x = np.array([-100., -.01, 0., .01, 100.])
        result = study.bounded_prediction(x)
        self.assertTrue(np.all(np.abs(result) <= .5))
        self.assertEqual(result[2], 0.)
        self.assertLess(float(study.bounded_prediction(np.array([1e-12]))[0]), 1e-10)
        np.testing.assert_allclose(result, -result[::-1], atol=0.)

    def test_zero_ml_forecast_preserves_base_scores_and_choices(self):
        a, predictions, meta = synthetic_scores()
        for row in study.grid():
            if row["hand_budget"]:
                continue
            base = row["architecture"]
            prefix = "event_" + base["normalization"] + "_"
            sc = study.score(row, a, predictions, 1)
            np.testing.assert_array_equal(sc, a[prefix + base["family"]])
            choices, _ = study.choices(row, a, predictions, meta, 1)
            self.assertEqual(choices, [[0, 1, 2]])

    def test_small_correction_can_change_all_three_slots(self):
        a, predictions, meta = synthetic_scores()
        row = next(r for r in study.grid() if r["architecture"]["id"] == 0
                   and r["arm"] == "residual" and r["ml_budget"] == .10)
        original, _ = study.choices(row, a, predictions, meta, 1)
        predictions["0_residual"][:] = [[-10., -10., -10., 10., 10., 10.]]
        selected, _ = study.choices(row, a, predictions, meta, 1)
        self.assertEqual(original, [[0, 1, 2]])
        self.assertEqual(selected, [[3, 4, 5]])
        # An otherwise winning name is still forbidden when signal-time ineligible.
        a["eligible"][0, 3] = False
        selected, _ = study.choices(row, a, predictions, meta, 1)
        self.assertNotIn(3, selected[0])
        a["eligible"][:] = False
        a["eligible"][0, 0] = True
        self.assertEqual(study.choices(row, a, predictions, meta, 1)[0], [[]])

    def test_normalization_does_not_change_eligibility(self):
        panel = synthetic_panel()
        frames, momenta, eligible = study.raw_frames(panel)
        day = panel["adj"].index[-1]
        symbols = ["A", "B", "C", "D"]
        eligible.loc[day, symbols] = [True, False, True, True]
        with patch.object(study.old, "state_row", return_value={"breadth": .75,
                          "trend": True, "highvol": False}):
            all_row, all_good, _ = study.snapshot(day, symbols, frames, momenta,
                                                eligible, panel, set(), "all")
            own_row, own_good, _ = study.snapshot(day, symbols, frames, momenta,
                                                eligible, panel, set(), "eligible")
        np.testing.assert_array_equal(all_good, own_good)
        np.testing.assert_array_equal(all_good, [True, False, True, True])
        self.assertEqual(own_row["raw"][1], 0.)
        self.assertFalse(np.array_equal(all_row["raw"], own_row["raw"]))


class TestCausalFeatureWindows(unittest.TestCase):
    def test_aligned_sharpe_uses_same_210_returns_ending_42_days_ago(self):
        panel = synthetic_panel()
        frames, _, _ = study.raw_frames(panel)
        t = len(panel["adj"]) - 1
        r = panel["adj"].pct_change(fill_method=None)
        window = r.iloc[t - 251:t - 41]
        self.assertEqual(len(window), 210)
        expected = window.mean() / window.std(ddof=1)
        np.testing.assert_allclose(frames["aligned_sharpe"].iloc[t], expected,
                                   atol=1e-13, rtol=1e-12)
        expected_raw = panel["adj"].iloc[t - 42] / panel["adj"].iloc[t - 252] - 1
        np.testing.assert_allclose(frames["raw"].iloc[t], expected_raw,
                                   atol=1e-14, rtol=1e-12)

    def test_latest_42_prices_do_not_change_skipped_momentum(self):
        panel = synthetic_panel()
        original, _, _ = study.raw_frames(panel)
        altered = {k: v.copy() for k, v in panel.items()}
        multiplier = np.linspace(1.05, 2., 42)[:, None]
        for key in ("adj", "close", "high", "low"):
            altered[key].iloc[-42:] *= multiplier
        changed, _, _ = study.raw_frames(altered)
        for name in ("raw", "aligned_sharpe"):
            np.testing.assert_array_equal(original[name].iloc[-1], changed[name].iloc[-1])
        self.assertFalse(np.allclose(original["recent_risk"].iloc[-1],
                                     changed["recent_risk"].iloc[-1]))

    def test_raw_feature_frames_are_prefix_invariant(self):
        panel = synthetic_panel()
        full, full_mom, full_good = study.raw_frames(panel)
        prefix_panel = {k: v.iloc[:350].copy() for k, v in panel.items()}
        prefix, prefix_mom, prefix_good = study.raw_frames(prefix_panel)
        for name in full:
            pd.testing.assert_frame_equal(prefix[name], full[name].iloc[:350])
        for name in full_mom:
            pd.testing.assert_frame_equal(prefix_mom[name], full_mom[name].iloc[:350])
        pd.testing.assert_frame_equal(prefix_good, full_good.iloc[:350])


@unittest.skipUnless((P / "DATA.npz").exists(), "prepared monthly data not available yet")
class TestPreparedMonthlyData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dict(np.load(P / "DATA.npz"))
        cls.meta = json.loads((P / "META.json").read_text())

    def test_eligibility_identical_to_prior_trading_data(self):
        old = np.load(study.ex.P / "DATA.npz")
        np.testing.assert_array_equal(self.data["eligible"], old["eligible"])
        self.assertEqual(self.data["eligible"].shape, (122, 49))
        for normalization in study.NORMS:
            for family in study.FAMILIES:
                scores = self.data["event_" + normalization + "_" + family]
                self.assertTrue(np.isfinite(scores).all())
        validation = json.loads((P / "PREPARE_VALIDATION.json").read_text())
        self.assertTrue(validation["future_label_validity_never_filters_trading"])

    def test_training_targets_centered_with_monthly_completed_end_dates(self):
        a = self.data
        self.assertTrue(np.all(a["ends"] > a["dates"]))
        for lo, hi in zip(a["offsets"][:-1], a["offsets"][1:]):
            good = a["train_good"][lo:hi]
            if good.any():
                self.assertAlmostEqual(float(a["y"][lo:hi][good].mean()), 0., places=13)

    @unittest.skipUnless((P / "FORECAST_AUDIT.json").exists(), "annual models not available yet")
    def test_forecast_label_cutoffs_and_oof_origins(self):
        audits = json.loads((P / "FORECAST_AUDIT.json").read_text())
        self.assertEqual(set(audits), {str(i) for i in range(10)})
        forecasts = np.load(P / "FORECASTS.npz")
        for architecture, audit in audits.items():
            self.assertEqual(audit["min_months"], 18)
            for row in audit["annual"]:
                self.assertEqual(row["cutoff"], "%04d-12-31" % (row["year"] - 1))
                if row["latest_label_end"]:
                    self.assertLessEqual(row["latest_label_end"], row["cutoff"])
                self.assertTrue(row["direct_and_residual_share_rows"])
            for row in audit["oof_months"]:
                if row["latest_training_end"]:
                    self.assertLessEqual(row["latest_training_end"], row["signal_date"])
                    self.assertLess(row["latest_training_signal"][:7], row["signal_date"][:7])
            for arm in ("direct", "residual"):
                pred = forecasts[architecture + "_" + arm]
                self.assertTrue(np.isfinite(pred).all())
                self.assertEqual(pred.shape, self.data["eligible"].shape)
                np.testing.assert_array_equal(pred[~self.data["eligible"]], 0.)


if __name__ == "__main__":
    unittest.main()
