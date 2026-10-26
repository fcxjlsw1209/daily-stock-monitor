"""Independent causality, weighting, path and missing-observation tests."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("trend_features_for_tests", Path(__file__).with_name("trend_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def panel(rows=420):
    rng = np.random.default_rng(391)
    dates = pd.bdate_range("2017-01-02", periods=rows)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .014, (rows, 2)), axis=0)),
                          index=dates, columns=["A", "B"])
    volume = pd.DataFrame(rng.lognormal(12, .7, (rows, 2)), index=dates, columns=prices.columns)
    return {"adj": prices.copy(), "close": prices.copy(), "volume": volume}


def calculate(p, t=-1, symbols=("A", "B")):
    return features.build_features(p, [(p["adj"].index[t], symbols)])


class TestTrendFeatures(unittest.TestCase):
    def test_exact_volume_formula_and_previous63_baseline(self):
        p = panel()
        output, audit = calculate(p)
        t = len(p["adj"]) - 1
        vals = []
        weights = []
        dv = (p["close"] * p["volume"])["A"].to_numpy()
        prices = p["adj"]["A"].to_numpy()
        for i in range(t - 251, t - 41):
            vals.append(np.log(prices[i]) - np.log(prices[i - 1]))
            weights.append(np.clip(dv[i] / dv[i - 63:i].mean(), .25, 4.))
        expected = np.average(vals, weights=weights) * 210 - np.sum(vals)
        self.assertAlmostEqual(output["volume_confirmation"][0, 0], expected, places=13)
        self.assertFalse(audit["trading_eligibility_modified"])
        self.assertEqual(audit["first_required_dollar_volume_offset"], -314)

    def test_prefix_and_future_perturbation_causality(self):
        p = panel()
        expected, _ = calculate(p, 350)
        prefix = {k: v.iloc[:351].copy() for k, v in p.items()}
        prefix_result, _ = calculate(prefix)
        for frame in p.values():
            frame.iloc[351:] = 1e9
        changed, _ = calculate(p, 350)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(expected[key], prefix_result[key])
            np.testing.assert_array_equal(expected[key], changed[key])

    def test_skip42_and_endpoint_included(self):
        p = panel()
        expected, _ = calculate(p)
        for frame in p.values():
            frame.iloc[-42:] = np.nan
        skipped, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(expected[key], skipped[key])
        p["adj"].iloc[-43, 0] = np.nan
        endpoint, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            self.assertTrue(np.isnan(endpoint[key][0, 0]))

    def test_constant_weight_confirmation_zero(self):
        p = panel()
        p["volume"] = 1e7 / p["close"]
        result, _ = calculate(p)
        np.testing.assert_allclose(result["volume_confirmation"], 0., atol=1e-14)

    def test_volume_scale_invariance(self):
        p = panel()
        expected, _ = calculate(p)
        p["volume"] *= 73.91
        scaled, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_allclose(expected[key], scaled[key], atol=1e-13)

    def test_split_preserving_dollar_volume_changes_no_feature(self):
        p = panel()
        expected, _ = calculate(p)
        p["close"].iloc[200:] /= 4.
        p["volume"].iloc[200:] *= 4.
        split, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(expected[key], split[key])

    def test_gradual_highs_outrank_underwater_path_and_known_age(self):
        p = panel()
        t = len(p["adj"]) - 1
        start, end = t - 252, t - 42
        p["adj"].iloc[start:end + 1, 0] = np.linspace(100., 110., 211)
        p["adj"].iloc[start:end + 1, 1] = np.r_[100., np.linspace(80., 99., 210)]
        result, _ = calculate(p)
        self.assertEqual(result["recovery_speed"][0, 0], 0.)
        self.assertEqual(result["recovery_speed"][0, 1], -.5)

    def test_flat_new_high_is_age_zero(self):
        p = panel()
        p["adj"].iloc[:, :] = 260.37
        p["adj"].iloc[::2, 0] *= (1. - 5e-13)
        result, _ = calculate(p)
        np.testing.assert_array_equal(result["recovery_speed"], 0.)

    def test_missing_volume_does_not_erase_recovery_or_become_zero(self):
        p = panel()
        expected, _ = calculate(p)
        # This is in the 63-session pre-window baseline, before price features.
        p["volume"].iloc[-300, 0] = np.nan
        result, audit = calculate(p)
        self.assertTrue(np.isnan(result["volume_confirmation"][0, 0]))
        np.testing.assert_array_equal(result["recovery_speed"], expected["recovery_speed"])
        self.assertTrue(np.isfinite(result["volume_confirmation"][0, 1]))
        self.assertEqual(audit["missing_reasons"]["volume_confirmation"]["invalid_dollar_volume_window"], 1)

    def test_nonpositive_or_nonfinite_inputs_rejected(self):
        for key in ("adj", "close", "volume"):
            for bad in (0., -1., np.nan, np.inf, -np.inf):
                with self.subTest(key=key, bad=bad):
                    p = panel()
                    p[key].iloc[-80, 0] = bad
                    result, _ = calculate(p)
                    self.assertTrue(np.isnan(result["volume_confirmation"][0, 0]))
                    self.assertEqual(np.isnan(result["recovery_speed"][0, 0]), key == "adj")

    def test_warmup_boundaries_and_unknown_security_preserve_shape(self):
        p = panel()
        at252, _ = calculate(p, 252)
        self.assertTrue(np.isfinite(at252["recovery_speed"]).all())
        self.assertTrue(np.isnan(at252["volume_confirmation"]).all())
        at314, _ = calculate(p, 314)
        self.assertTrue(np.isfinite(at314["volume_confirmation"]).all())
        result, audit = calculate(p, symbols=("A", "UNKNOWN"))
        for key in features.FEATURE_NAMES:
            self.assertEqual(result[key].shape, (1, 2))
            self.assertTrue(np.isnan(result[key][0, 1]))
            self.assertEqual(audit["missing_reasons"][key]["unknown_symbol"], 1)

    def test_input_frames_remain_unchanged(self):
        p = panel()
        saved = {k: v.copy(deep=True) for k, v in p.items()}
        calculate(p)
        for key in p:
            pd.testing.assert_frame_equal(p[key], saved[key])

    def test_misaligned_or_invalid_events_raise(self):
        p = panel()
        p["volume"] = p["volume"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "exactly the adj"):
            calculate(p)
        p = panel()
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            calculate(p, symbols=("A", "A"))
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            features.build_features(p, [(pd.Timestamp("2017-01-01"), ["A"])])


if __name__ == "__main__":
    unittest.main()
