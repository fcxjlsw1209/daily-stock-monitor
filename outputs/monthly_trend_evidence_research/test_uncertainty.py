"""Tests for causal block sensitivity; no strategy returns or parameter tuning."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from uncertainty import (annual_block_forecasts, pair_evidence,
                         _fit_ridge, _predict, month_balanced_weights)


def fixture():
    rng = np.random.default_rng(503)
    months = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2018-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 4)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 4)
    X = rng.normal(size=(len(dates), 10))
    y = .02 * X[:, 0] - .01 * X[:, 1] + rng.normal(0, .005, len(dates))
    good = np.ones(len(dates), dtype=bool)
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-03-31"], dtype="datetime64[D]"), 4)
    event_X = rng.normal(size=(len(event_dates), X.shape[1]))
    event_good = np.ones(len(event_dates), dtype=bool)
    return [dates, ends, X, y, good, event_dates, event_X, event_good]


class TestAnnualBlockModel(unittest.TestCase):
    def test_future_labels_and_features_do_not_change_earlier_models(self):
        args = fixture()
        original = annual_block_forecasts(*args)
        changed = [a.copy() for a in args]
        after_cutoff = changed[1] > np.datetime64("2015-12-31")
        changed[3][after_cutoff] += 500.
        changed[2][changed[0] > np.datetime64("2015-12-31")] *= 1000.
        other = annual_block_forecasts(*changed)
        early = args[5] < np.datetime64("2017-01-01")
        np.testing.assert_array_equal(original["full"][early], other["full"][early])
        np.testing.assert_array_equal(original["folds"][:, early], other["folds"][:, early])
        np.testing.assert_array_equal(original["available"][early], True)
        json.dumps(original["audit"], allow_nan=False)

    def test_full_fit_exactly_matches_previous_pointwise_pooled(self):
        path = Path(__file__).resolve().parents[1] / "monthly_session_type_research" / "models.py"
        spec = importlib.util.spec_from_file_location("previous_pointwise_for_block_test", path)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        dates, ends, X, y, good, edates, ex, egood = fixture()
        expected = previous.annual_forecasts(dates, ends, X, y, good, np.zeros(len(y), int),
            edates, ex, egood, np.zeros(len(edates), int))["pooled"]
        actual = annual_block_forecasts(dates, ends, X, y, good, edates, ex, egood)
        np.testing.assert_array_equal(actual["full"], expected)

    def test_every_fold_excludes_its_year_and_refits_scaler(self):
        args = fixture()
        dates, ends, X, y, good, event_dates, event_X, _ = args
        # A shifted historical year makes reusing the full-fit scaler incorrect.
        X[(dates >= np.datetime64("2014-01-01")) & (dates < np.datetime64("2015-01-01")), 0] += 100.
        result = annual_block_forecasts(*args)
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        for annual in result["audit"]["annual"]:
            cutoff = np.datetime64(annual["cutoff"])
            lower = np.datetime64(annual["lookback_exclusive_start"])
            event = event_dates.astype("datetime64[Y]").astype(int) + 1970 == annual["year"]
            for record in annual["folds"]:
                excluded = record["excluded_signal_year"]
                self.assertNotIn(excluded, record["included_signal_years"])
                self.assertLessEqual(record["latest_label_end"], annual["cutoff"])
                mask = good & (ends <= cutoff) & (dates > lower) & (years != excluded)
                weight = month_balanced_weights(dates[mask], cutoff, 24.)
                model = _fit_ridge(X[mask], y[mask], weight, 10.)
                expected = _predict(model, event_X[event])
                np.testing.assert_array_equal(result["folds"][record["slot"], event], expected)
                self.assertEqual(record["training_rows"], int(mask.sum()))
                self.assertGreaterEqual(record["training_months"], 18)

    def test_requires_36_mature_months_and_four_year_blocks(self):
        args = fixture()
        args[5][:] = np.datetime64("2012-01-31")
        cold = annual_block_forecasts(*args)
        np.testing.assert_array_equal(cold["full"], 0.)
        np.testing.assert_array_equal(cold["available"], False)
        np.testing.assert_array_equal(cold["fold_available"], False)
        args = fixture()
        args[4] = args[0] < np.datetime64("2013-01-01")
        args[5][:] = np.datetime64("2014-01-31")
        three_blocks = annual_block_forecasts(*args)
        self.assertEqual(three_blocks["audit"]["annual"][0]["training_months"], 36)
        self.assertEqual(len(three_blocks["audit"]["annual"][0]["signal_year_blocks"]), 3)
        np.testing.assert_array_equal(three_blocks["available"], False)

    def test_invalid_events_and_missing_features_are_unavailable(self):
        args = fixture()
        args[7][0] = False
        args[6][1, 0] = np.nan
        result = annual_block_forecasts(*args)
        np.testing.assert_array_equal(result["available"][:2], False)
        np.testing.assert_array_equal(result["fold_available"][:, :2], False)
        np.testing.assert_array_equal(result["full"][:2], 0.)
        self.assertTrue(np.isfinite(result["folds"]).all())

    def test_forecast_scaling_is_not_fit_on_future_events(self):
        args = fixture()
        original = annual_block_forecasts(*args)
        args[6][-1] = 1e8
        changed = annual_block_forecasts(*args)
        np.testing.assert_array_equal(original["full"][:-1], changed["full"][:-1])
        np.testing.assert_array_equal(original["folds"][:, :-1], changed["folds"][:, :-1])

    def test_fitted_numerical_zero_remains_available(self):
        args = fixture()
        args[3][:] = 0.
        result = annual_block_forecasts(*args)
        np.testing.assert_array_equal(result["full"], 0.)
        np.testing.assert_array_equal(result["available"], True)


class TestPairEvidence(unittest.TestCase):
    def test_common_prediction_noise_cancels_in_pair_difference(self):
        common = np.array([-.5, -.2, .1, .4, .7])
        folds = np.column_stack((common, common + .02))
        result = pair_evidence(np.array([.01, .03]), folds, np.ones(2, bool),
                               np.ones(folds.shape, bool), 0, 1, 2.)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["delta_full"], .02)
        self.assertLess(result["se"], 1e-14)
        self.assertAlmostEqual(result["lower"], .02)
        self.assertGreater(np.std(folds[:, 0]), .1)

    def test_stability_penalty_and_k_zero_control(self):
        differences = np.array([-.01, .01, .03, .05, .07])
        folds = np.column_stack((np.zeros(5), differences))
        args = (np.array([0., .03]), folds, np.ones(2, bool), np.ones(folds.shape, bool), 0, 1)
        neutral, penalized = pair_evidence(*args, k=0.), pair_evidence(*args, k=1.)
        self.assertEqual(neutral["lower"], neutral["delta_full"])
        expected_se = np.sqrt(4. / 5. * np.sum((differences - differences.mean()) ** 2))
        self.assertAlmostEqual(penalized["se"], expected_se)
        self.assertLess(penalized["lower"], 0.)
        reverse = pair_evidence(args[0], folds, args[2], args[3], 1, 0, 0.)
        self.assertAlmostEqual(reverse["delta_full"], -neutral["delta_full"])
        self.assertAlmostEqual(reverse["se"], neutral["se"])

    def test_missing_or_mismatched_fold_evidence_strictly_blocks_change(self):
        full = np.array([0., .1])
        folds = np.tile(full, (5, 1))
        valid = np.ones(2, bool)
        mask = np.ones(folds.shape, bool)
        mask[0, 0] = False
        result = pair_evidence(full, folds, valid, mask, 0, 1)
        self.assertFalse(result["available"])
        self.assertEqual(result["lower"], 0.)
        mask[:] = True
        folds[0, 0] = np.nan
        self.assertFalse(pair_evidence(full, folds, valid, mask, 0, 1)["available"])
        self.assertFalse(pair_evidence(full, folds[1:4], valid, mask[1:4], 0, 1)["available"])

    def test_padding_slots_are_not_counted_as_year_blocks(self):
        full = np.array([0., .02])
        folds = np.tile(full, (6, 1))
        masks = np.ones(folds.shape, bool)
        masks[-1] = False
        result = pair_evidence(full, folds, np.ones(2, bool), masks, 0, 1)
        self.assertEqual(result["fold_count"], 5)
        self.assertTrue(result["available"])


if __name__ == "__main__":
    unittest.main()
