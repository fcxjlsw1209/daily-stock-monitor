"""Exact-window, comparable-segment and causality checks without returns data."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("segment_features_for_tests", Path(__file__).with_name("segment_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def panel(rows=420):
    rng = np.random.default_rng(182)
    dates = pd.bdate_range("2017-01-02", periods=rows)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .015, (rows, 2)), axis=0)),
                          index=dates, columns=["A", "B"])
    return {"adj": prices}


def calculate(p, t=-1, symbols=("A", "B")):
    return features.build_features(p, [(p["adj"].index[t], symbols)])


def set_path(p, symbol, old_returns, new_returns):
    assert len(old_returns) == 126 and len(new_returns) == 84
    t = len(p["adj"]) - 1
    prices = 100. * np.exp(np.r_[0., np.cumsum(np.r_[old_returns, new_returns])])
    p["adj"].iloc[t - 252:t - 42 + 1, p["adj"].columns.get_loc(symbol)] = prices


class TestSegmentFeatures(unittest.TestCase):
    def test_exact_126_84_formula_and_daily_downside_normalization(self):
        p = panel()
        set_path(p, "A", np.tile([-.02, .01], 63), np.tile([-.01, .02], 42))
        result, audit = calculate(p)
        self.assertAlmostEqual(result["continuity_floor"][0, 0], -1. / 3., places=13)
        self.assertAlmostEqual(result["quality_improvement"][0, 0], 2. / 3., places=13)
        self.assertAlmostEqual(result["downside_improvement"][0, 0], .01 / np.sqrt(2), places=13)
        self.assertEqual(audit["old_return_count"], 126)
        self.assertEqual(audit["new_return_count"], 84)
        self.assertFalse(audit["trading_eligibility_modified"])

    def test_equal_return_distributions_have_zero_improvement_despite_lengths(self):
        p = panel()
        set_path(p, "A", np.tile([-.01, .02], 63), np.tile([-.01, .02], 42))
        result, _ = calculate(p)
        self.assertAlmostEqual(result["continuity_floor"][0, 0], 1. / 3., places=13)
        self.assertAlmostEqual(result["quality_improvement"][0, 0], 0., places=13)
        self.assertAlmostEqual(result["downside_improvement"][0, 0], 0., places=13)

    def test_strong_old_weak_new_is_negative_improvement(self):
        p = panel()
        set_path(p, "A", np.full(126, .002), np.full(84, -.001))
        set_path(p, "B", np.full(126, -.001), np.full(84, .002))
        result, _ = calculate(p)
        np.testing.assert_allclose(result["continuity_floor"], [[-1., -1.]], atol=1e-13)
        np.testing.assert_allclose(result["quality_improvement"], [[-2., 2.]], atol=1e-13)
        np.testing.assert_allclose(result["downside_improvement"], [[-.001, .001]], atol=1e-13)

    def test_flat_segments_are_zero(self):
        p = panel()
        p["adj"].iloc[:, :] = 260.37
        result, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(result[key], 0.)

    def test_prefix_and_future_perturbation_invariance(self):
        p = panel()
        expected, _ = calculate(p, 350)
        prefix = {"adj": p["adj"].iloc[:351].copy()}
        prefix_result, _ = calculate(prefix)
        p["adj"].iloc[351:] = 1e20
        future_changed, _ = calculate(p, 350)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(expected[key], prefix_result[key])
            np.testing.assert_array_equal(expected[key], future_changed[key])

    def test_exact_skip42_and_window_boundaries(self):
        p = panel()
        expected, _ = calculate(p)
        p["adj"].iloc[-42:] = np.nan
        p["adj"].iloc[:-253] = np.nan
        skipped, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(expected[key], skipped[key])
        for location in (-253, -127, -43):
            changed = {"adj": p["adj"].copy()}
            changed["adj"].iloc[location, 0] = np.nan
            bad, _ = calculate(changed)
            for key in features.FEATURE_NAMES:
                self.assertTrue(np.isnan(bad[key][0, 0]))

    def test_price_scale_invariance(self):
        p = panel()
        expected, _ = calculate(p)
        p["adj"]["A"] *= 73.91
        p["adj"]["B"] /= 127.
        scaled, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            np.testing.assert_allclose(expected[key], scaled[key], atol=1e-13)

    def test_missing_nonpositive_and_nonfinite_prices_stay_nan(self):
        for location, reason in ((-200, "invalid_old_segment_only"), (-80, "invalid_new_segment_only"),
                                 (-127, "invalid_both_segments")):
            for bad in (np.nan, 0., -1., np.inf, -np.inf):
                with self.subTest(location=location, bad=bad):
                    p = panel()
                    p["adj"].iloc[location, 0] = bad
                    result, audit = calculate(p)
                    for key in features.FEATURE_NAMES:
                        self.assertTrue(np.isnan(result[key][0, 0]))
                        self.assertTrue(np.isfinite(result[key][0, 1]))
                    self.assertEqual(audit["missing_reasons"][reason], 1)

    def test_history_boundary_and_unknown_symbol_preserve_shape(self):
        p = panel()
        before, _ = calculate(p, 251)
        at, _ = calculate(p, 252)
        unknown, audit = calculate(p, symbols=("A", "UNKNOWN"))
        for key in features.FEATURE_NAMES:
            self.assertTrue(np.isnan(before[key]).all())
            self.assertTrue(np.isfinite(at[key]).all())
            self.assertEqual(unknown[key].shape, (1, 2))
            self.assertTrue(np.isnan(unknown[key][0, 1]))
        self.assertEqual(audit["missing_reasons"]["unknown_symbol"], 1)

    def test_input_is_not_modified_and_events_preserve_symbol_order(self):
        p = panel()
        saved = p["adj"].copy(deep=True)
        original, _ = calculate(p)
        reordered, _ = calculate(p, symbols=("B", "A"))
        pd.testing.assert_frame_equal(p["adj"], saved)
        for key in features.FEATURE_NAMES:
            np.testing.assert_array_equal(reordered[key], original[key][:, ::-1])

    def test_invalid_dates_duplicate_symbols_and_variable_width_raise(self):
        p = panel()
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            calculate(p, symbols=("A", "A"))
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            features.build_features(p, [(pd.Timestamp("2017-01-01"), ["A"])])
        with self.assertRaisesRegex(ValueError, "same number"):
            features.build_features(p, [(p["adj"].index[-1], ["A"]), (p["adj"].index[-2], ["A", "B"])])


if __name__ == "__main__":
    unittest.main()
