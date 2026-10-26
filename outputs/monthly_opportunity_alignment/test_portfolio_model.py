"""Pure feature/label arithmetic and annual pair-model causality tests."""
import json
import unittest

import numpy as np

from portfolio_model import (annual_pair_forecasts, pair_features, pair_targets,
                             _fit_rms_ridge, _predict, month_balanced_weights)


def fixture():
    rng = np.random.default_rng(192)
    months = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2018-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 2)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 2)
    X = rng.normal(size=(len(dates), 20))
    y = .025 * X[:, 0] - .01 * X[:, 10] + rng.normal(0, .003, len(dates))
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-03-31"], dtype="datetime64[D]"), 2)
    event_X = rng.normal(size=(len(event_dates), 20))
    event_good = np.ones(len(event_dates), bool)
    return [dates, ends, X, y, event_dates, event_X, event_good]


class TestPairArithmetic(unittest.TestCase):
    def test_features_are_20_dimensional_and_antisymmetric(self):
        rng = np.random.default_rng(44)
        i, j, a, b = [rng.normal(size=(7, 10)) for _ in range(4)]
        features = pair_features(i, j, a, b)
        self.assertEqual(features.shape, (7, 20))
        np.testing.assert_array_equal(features[:, :10], j - i)
        np.testing.assert_array_equal(features[:, 10:], (j - i) * ((a + b) / 2))
        np.testing.assert_array_equal(pair_features(j, i, a, b), -features)
        np.testing.assert_array_equal(pair_features(i, j, b, a), features)

    def test_portfolio_target_is_log_of_equal_weight_portfolio_wealth(self):
        a, b, i, j = [.20, -.10, .05, .35]
        target = pair_targets(a, b, i, j)
        expected = np.log((1 + (a + b + j) / 3) / (1 + (a + b + i) / 3))
        self.assertTrue(target["valid"])
        self.assertAlmostEqual(float(target["portfolio"]), expected)
        self.assertAlmostEqual(float(target["stock"]), np.log1p(j) - np.log1p(i))
        self.assertNotAlmostEqual(float(target["portfolio"]), float(target["stock"]) / 3., places=6)
        swapped = pair_targets(a, b, j, i)
        self.assertAlmostEqual(float(swapped["portfolio"]), -float(target["portfolio"]))
        self.assertAlmostEqual(float(swapped["stock"]), -float(target["stock"]))

    def test_common_label_mask_preserves_matched_target_comparison(self):
        target = pair_targets([0., np.nan, 0., -1.], [0., 0., 0., -1.],
                              [0., 0., -1., .1], [.3, .3, .3, .2])
        np.testing.assert_array_equal(target["valid"], [True, False, False, True])
        for name in ("portfolio", "stock"):
            self.assertTrue(np.isnan(target[name][~target["valid"]]).all())
            self.assertTrue(np.isfinite(target[name][target["valid"]]).all())


class TestAnnualPairModel(unittest.TestCase):
    def test_future_labels_and_features_do_not_change_past_predictions(self):
        args = fixture()
        original = annual_pair_forecasts(*args)
        changed = [v.copy() for v in args]
        changed[3][changed[1] > np.datetime64("2015-12-31")] += 500.
        changed[2][changed[0] > np.datetime64("2015-12-31")] *= 1000.
        modified = annual_pair_forecasts(*changed)
        before = args[4] < np.datetime64("2017-01-01")
        np.testing.assert_array_equal(original["pred"][before], modified["pred"][before])
        np.testing.assert_array_equal(original["available"][before], True)
        json.dumps(original["audit"], allow_nan=False)

    def test_training_cutoff_and_RMS_scaler_match_independent_construction(self):
        args = fixture()
        dates, ends, X, y, event_dates, event_X, _ = args
        X[(dates >= np.datetime64("2014-01-01")) & (dates < np.datetime64("2015-01-01")), 0] += 20.
        result = annual_pair_forecasts(*args)
        for row in result["audit"]["annual"]:
            cutoff = np.datetime64(row["cutoff"])
            self.assertEqual(str(cutoff), "%04d-12-31" % (row["year"] - 1))
            self.assertLessEqual(row["latest_label_end"], row["cutoff"])
            mask = (ends <= cutoff) & (dates > np.datetime64(row["lookback_exclusive_start"]))
            weight = month_balanced_weights(dates[mask], cutoff, 24.)
            model = _fit_rms_ridge(X[mask], y[mask], weight)
            expected_scale = np.sqrt(np.average(X[mask] ** 2, axis=0, weights=weight))
            np.testing.assert_allclose(model[0], expected_scale, atol=0., rtol=1e-14)
            event = event_dates.astype("datetime64[Y]").astype(int) + 1970 == row["year"]
            np.testing.assert_array_equal(result["pred"][event], _predict(model, event_X[event]))
            self.assertGreaterEqual(row["training_pairs"], 30)
            self.assertGreaterEqual(row["training_months"], 18)

    def test_fit_matches_augmented_least_squares_without_intercept(self):
        rng = np.random.default_rng(744)
        X = rng.normal(loc=.5, size=(70, 6))
        y = rng.normal(0, .03, 70)
        weight = rng.uniform(.1, 2., 70)
        model = _fit_rms_ridge(X, y, weight)
        scale = np.sqrt(np.average(X ** 2, axis=0, weights=weight))
        z = X / scale
        design = np.vstack((np.sqrt(weight)[:, None] * z, np.sqrt(10.) * np.eye(6)))
        target = np.r_[np.sqrt(weight) * y, np.zeros(6)]
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
        np.testing.assert_allclose(model[1], coefficient, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(_predict(model, X), np.sum(z * coefficient, axis=1),
                                   atol=1e-14, rtol=1e-12)

    def test_swap_predictions_reverse_sign_and_zero_difference_is_zero(self):
        args = fixture()
        vector = args[5][0].copy()
        args[4][:] = np.datetime64("2016-06-30")
        args[5][:] = 0.
        args[5][0], args[5][1] = vector, -vector
        result = annual_pair_forecasts(*args)
        self.assertEqual(result["pred"][0], -result["pred"][1])
        np.testing.assert_array_equal(result["pred"][2:], 0.)
        np.testing.assert_array_equal(result["available"], True)

    def test_row_permutation_does_not_change_fitted_rule(self):
        args = fixture()
        original = annual_pair_forecasts(*args)
        perm = np.random.default_rng(31).permutation(len(args[0]))
        args[:4] = [v[perm] for v in args[:4]]
        other = annual_pair_forecasts(*args)
        np.testing.assert_allclose(original["pred"], other["pred"], atol=1e-13, rtol=1e-12)

    def test_minimum_months_and_pair_count_are_both_required(self):
        for months, copies, expected in [(18, 1, False), (17, 2, False), (18, 2, True)]:
            m = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2010-01", "M") + months)
            dates = np.repeat((m + 1).astype("datetime64[D]") - 1, copies)
            ends = np.repeat((m + 2).astype("datetime64[D]") - 1, copies)
            X = np.ones((len(dates), 20))
            result = annual_pair_forecasts(dates, ends, X, np.ones(len(dates)),
                np.array(["2012-01-31"], dtype="datetime64[D]"), np.ones((1, 20)), np.ones(1, bool))
            self.assertEqual(bool(result["available"][0]), expected)
            if not expected:
                self.assertEqual(result["pred"][0], 0.)

    def test_missing_event_features_never_trigger_available_forecast(self):
        args = fixture()
        args[5][0, 0] = np.nan
        args[6][1] = False
        result = annual_pair_forecasts(*args)
        np.testing.assert_array_equal(result["pred"][:2], 0.)
        np.testing.assert_array_equal(result["available"][:2], False)
        args = fixture()
        args[3][:] = 0.
        zero = annual_pair_forecasts(*args)
        np.testing.assert_array_equal(zero["pred"], 0.)
        np.testing.assert_array_equal(zero["available"], True)

    def test_future_event_cannot_change_training_scaler(self):
        args = fixture()
        original = annual_pair_forecasts(*args)
        args[5][-1] = 1e9
        changed = annual_pair_forecasts(*args)
        np.testing.assert_array_equal(original["pred"][:-1], changed["pred"][:-1])

    def test_bad_label_horizon_is_rejected(self):
        args = fixture()
        args[1][0] = np.datetime64("2010-03-31")
        with self.assertRaisesRegex(ValueError, "next calendar month"):
            annual_pair_forecasts(*args)


if __name__ == "__main__":
    unittest.main()
