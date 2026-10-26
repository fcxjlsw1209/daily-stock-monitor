"""Formula, unit, causality and missing-data tests; no strategy performance."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("cost_features_for_tests", Path(__file__).with_name("cost_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def panel(rows=450):
    rng = np.random.default_rng(11801)
    p = 100. * np.exp(np.cumsum(rng.normal(.0005, .012, (rows, 3)), axis=0))
    adjustment = np.linspace(.91, .99, rows)[:, None]
    index = pd.bdate_range("2014-01-02", periods=rows)
    cols = ["A", "ETF", "B"]
    frame = lambda values: pd.DataFrame(values, index=index, columns=cols)
    return {"adj": frame(p), "close": frame(p / adjustment),
            "volume": frame(rng.uniform(1e5, 1e7, (rows, 3)))}


def calculate(p, t=-1, symbols=("A", "ETF", "B")):
    return features.build_features(p, [(p["adj"].index[t], symbols)])


class TestCostFeatures(unittest.TestCase):
    def test_independent_direct_formula_both_windows(self):
        p = panel()
        out, audit = calculate(p)
        for window in features.WINDOWS:
            a = p["adj"]["A"].iloc[-window:].to_numpy()
            c = p["close"]["A"].iloc[-window:].to_numpy()
            v = p["volume"]["A"].iloc[-window:].to_numpy()
            decay = 2. ** (-np.arange(window - 1, -1, -1) / 63.)
            q = c * v / a
            for prefix, weight in (("volume", decay * q), ("time", decay)):
                reference = sum(weight * a) / sum(weight)
                self.assertAlmostEqual(out[prefix + "_gap" + str(window)][0, 0], np.log(a[-1] / reference), places=13)
                upper = sum(weight[a > a[-1] * (1. + 1e-12)]) / sum(weight)
                self.assertAlmostEqual(out[prefix + "_upper_mass" + str(window)][0, 0], upper, places=13)
        self.assertEqual(audit["first_used_offsets"], {"126": -125, "252": -251})
        self.assertEqual(audit["last_used_offset"], 0)
        self.assertFalse(audit["trading_eligibility_modified"])
        self.assertTrue(audit["controls_share_price_close_volume_mask"])

    def test_prefix_and_future_values_do_not_change_signal(self):
        p = panel()
        expected, _ = calculate(p, 350)
        prefix, _ = calculate({k: v.iloc[:351].copy() for k, v in p.items()})
        for value in p.values():
            value.iloc[351:] = np.nan
        perturbed, _ = calculate(p, 350)
        for key in expected:
            np.testing.assert_array_equal(expected[key], prefix[key])
            np.testing.assert_array_equal(expected[key], perturbed[key])

    def test_window_exact_boundaries_and_signal_day_are_used(self):
        p = panel()
        baseline, _ = calculate(p)
        # Observation -253 lies outside both windows; -252 enters only 252.
        p["volume"].iloc[-253, 0] = np.nan
        outside, _ = calculate(p)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], outside[key])
        p["volume"].iloc[-252, 0] = np.nan
        inside, _ = calculate(p)
        self.assertTrue(inside["volume_gap126_valid"][0, 0])
        self.assertFalse(inside["volume_gap252_valid"][0, 0])
        p = panel()
        p["adj"].iloc[-1, 0] *= 1.3
        signal_changed, _ = calculate(p)
        self.assertNotEqual(baseline["volume_gap126"][0, 0], signal_changed["volume_gap126"][0, 0])

    def test_prices_and_volume_units_do_not_change_dimensionless_scores(self):
        p = panel()
        saved = {k: v.copy(deep=True) for k, v in p.items()}
        expected, _ = calculate(p)
        for k in p:
            pd.testing.assert_frame_equal(p[k], saved[k])
        changes = [
            {"adj": [1e5, .01, 9.], "close": [1e5, .01, 9.], "volume": [1., 1., 1.]},
            {"adj": [3., .003, 171.], "close": [1., 1., 1.], "volume": [1., 1., 1.]},
            {"adj": [1., 1., 1.], "close": [1., 1., 1.], "volume": [.0001, 3., 170.]},
        ]
        for scales in changes:
            actual, _ = calculate({k: saved[k] * scales[k] for k in saved})
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=2e-12)

    def test_inverse_close_volume_split_scaling_is_invariant(self):
        p = panel()
        expected, _ = calculate(p)
        # Change the quoted share unit in only the earlier part of the window.
        # The same economic traded amount has price / 4 and shares * 4.
        p["close"].iloc[:330, 0] /= 4.
        p["volume"].iloc[:330, 0] *= 4.
        actual, _ = calculate(p)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=2e-12)

    def test_constant_adjusted_unit_volume_matches_time_control(self):
        p = panel()
        p["volume"] = 150000. * p["adj"] / p["close"]
        out, _ = calculate(p)
        for window in features.WINDOWS:
            for metric in ("gap", "upper_mass"):
                np.testing.assert_allclose(out["volume_" + metric + str(window)], out["time_" + metric + str(window)], atol=2e-14)

    def test_upper_volume_concentration_changes_reference_in_expected_direction(self):
        p = panel(300)
        for k in ("adj", "close"):
            p[k].loc[:, "A"] = 100.
            p[k].iloc[-100:-40, 0] = 120.
        p["volume"].loc[:, "A"] = 1e6
        equal, _ = calculate(p)
        p["volume"].iloc[-100:-40, 0] *= 10.
        concentrated, _ = calculate(p)
        for w in features.WINDOWS:
            self.assertLess(concentrated["volume_gap" + str(w)][0, 0], equal["volume_gap" + str(w)][0, 0])
            self.assertGreater(concentrated["volume_upper_mass" + str(w)][0, 0], equal["volume_upper_mass" + str(w)][0, 0])
            self.assertEqual(concentrated["time_gap" + str(w)][0, 0], equal["time_gap" + str(w)][0, 0])

    def test_flat_price_is_exact_zero_even_with_variable_volume_and_adjustments(self):
        p = panel()
        p["adj"].loc[:, "A"] = 123.4
        out, audit = calculate(p)
        for key in features.FEATURE_NAMES:
            self.assertEqual(out[key][0, 0], 0.)
            self.assertTrue(out[key + "_valid"][0, 0])
        self.assertEqual(audit["flat_zero_cells"], {"126": 1, "252": 1})

    def test_latest_tied_price_has_no_overhead_mass(self):
        p = panel()
        p["adj"].loc[:, "A"] = 100.
        p["adj"].iloc[-120:-60, 0] = 80.
        out, _ = calculate(p)
        for w in features.WINDOWS:
            self.assertEqual(out["volume_upper_mass" + str(w)][0, 0], 0.)
            self.assertEqual(out["time_upper_mass" + str(w)][0, 0], 0.)
            self.assertGreater(out["volume_gap" + str(w)][0, 0], 0.)

    def test_invalid_inputs_neutralize_matched_controls_too(self):
        for source in ("adj", "close", "volume"):
            for bad in (np.nan, np.inf, 0., -1.):
                with self.subTest(source=source, bad=bad):
                    p = panel()
                    p[source].iloc[-60, 0] = bad
                    out, audit = calculate(p)
                    for key in features.FEATURE_NAMES:
                        self.assertTrue(np.isnan(out[key][0, 0]))
                        self.assertFalse(out[key + "_valid"][0, 0])
                        self.assertEqual(audit["missing_reasons"][key]["invalid_" + source + "_window"], 1)
                    self.assertTrue(out["time_gap252_valid"][0, 1])

    def test_missing_volume_does_not_shorten_window_or_backfill(self):
        p = panel()
        p["volume"].iloc[-200, 0] = np.nan
        out, _ = calculate(p)
        self.assertTrue(out["volume_gap126_valid"][0, 0])
        self.assertFalse(out["volume_gap252_valid"][0, 0])
        self.assertFalse(out["time_gap252_valid"][0, 0])

    def test_warmup_unknown_symbol_and_empty_events_preserve_shape(self):
        p = panel()
        out, audit = calculate(p, 125, ("A", "UNKNOWN"))
        self.assertEqual(out["volume_gap126"].shape, (1, 2))
        self.assertTrue(out["volume_gap126_valid"][0, 0])
        self.assertFalse(out["volume_gap252_valid"][0, 0])
        self.assertFalse(out["volume_gap126_valid"][0, 1])
        self.assertEqual(audit["missing_reasons"]["volume_gap126"]["unknown_symbol"], 1)
        early, _ = calculate(p, 124)
        self.assertFalse(early["volume_gap126_valid"].any())
        ready, _ = calculate(p, 251)
        self.assertTrue(ready["volume_gap252_valid"].all())
        empty, _ = features.build_features(p, [])
        self.assertTrue(all(value.shape == (0, 0) for value in empty.values()))

    def test_candidate_order_panel_order_and_irrelevant_columns(self):
        p = panel()
        expected, _ = calculate(p)
        changed = {k: v[["ETF", "B", "A"]].assign(UNUSED=np.nan) for k, v in p.items()}
        day = p["adj"].index[-1]
        actual, _ = features.build_features(changed, [(day, ["ETF", "A"]), (day, ["A", "ETF"])])
        for key in expected:
            np.testing.assert_array_equal(actual[key][0], expected[key][0, [1, 0]])
            np.testing.assert_array_equal(actual[key][1], expected[key][0, [0, 1]])

    def test_49_symbol_production_shape_and_validity(self):
        p = panel()
        wide = {k: pd.concat([v["A"].rename("S" + str(i)) for i in range(49)], axis=1) for k, v in p.items()}
        out, audit = calculate(wide, symbols=list(wide["adj"].columns))
        self.assertEqual(audit["symbols_per_event"], 49)
        for key in features.FEATURE_NAMES:
            self.assertEqual(out[key].shape, (1, 49))
            self.assertTrue(out[key + "_valid"].all())

    def test_large_positive_units_do_not_overflow_products(self):
        p = panel()
        expected, _ = calculate(p)
        p["close"] *= 1e240
        p["volume"] *= 1e240
        actual, _ = calculate(p)
        for key in features.FEATURE_NAMES:
            self.assertTrue(np.isfinite(actual[key]).all())
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=2e-11)

    def test_event_and_panel_validation(self):
        p = panel()
        day = p["adj"].index[-1]
        with self.assertRaisesRegex(ValueError, "same number"):
            features.build_features(p, [(day, ["A"]), (day, ["A", "ETF"])])
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            calculate(p, symbols=("A", "A"))
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            features.build_features(p, [(pd.Timestamp("1900-01-01"), ["A"])])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            calculate({k: v.iloc[::-1] for k, v in p.items()})
        with self.assertRaisesRegex(ValueError, "columns must be unique"):
            calculate({k: v.rename(columns={"ETF": "A"}) for k, v in p.items()})
        with self.assertRaisesRegex(ValueError, "exactly the adj"):
            calculate({**p, "volume": p["volume"][["ETF", "A", "B"]]})
        with self.assertRaisesRegex(KeyError, "Missing required panel"):
            calculate({"adj": p["adj"]})


if __name__ == "__main__":
    unittest.main()
