"""Causality, paired-sample, and missing-prediction tests for earnings research."""
import json
import unittest
from unittest.mock import patch

import numpy as np

import earnings_model
from earnings_model import annual_forecasts, month_balanced_weights


def sample_inputs():
    months = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2019-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 4)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 4)
    rng = np.random.default_rng(71)
    plain = rng.normal(size=(len(dates), 2))
    earn = rng.normal(size=(len(dates), 3))
    y = .03 * plain[:, 0] - .01 * plain[:, 1] + .015 * earn[:, 0]
    y += rng.normal(scale=.002, size=len(y))
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-01-31"],
                                    dtype="datetime64[D]"), 4)
    return [dates, ends, plain, earn, y, np.ones(len(y), dtype=bool), event_dates,
            rng.normal(size=(len(event_dates), 2)), rng.normal(size=(len(event_dates), 3)),
            np.ones(len(event_dates), dtype=bool)]


class TestAnnualEarningsForecasts(unittest.TestCase):
    def test_future_unmatured_labels_and_current_year_features_cannot_leak(self):
        args = sample_inputs()
        before = annual_forecasts(*args)
        changed = [a.copy() for a in args]
        future = changed[1] > np.datetime64("2015-12-31")
        changed[4][future] = 500. + changed[4][future] * 100.
        changed[2][future] *= 1000.
        changed[3][future] *= 1000.
        after = annual_forecasts(*changed)
        in_2016 = args[6] < np.datetime64("2017-01-01")
        for key in ("plain", "earn"):
            np.testing.assert_array_equal(before[key][in_2016], after[key][in_2016])
            self.assertGreater(float(np.abs(before[key][in_2016]).sum()), 0)
        np.testing.assert_array_equal(before["available"], after["available"])

    def test_models_are_frozen_within_year_and_refit_next_year(self):
        args = sample_inputs()
        args[7][4:8] = args[7][:4]
        args[7][8:12] = args[7][:4]
        args[8][4:8] = args[8][:4]
        args[8][8:12] = args[8][:4]
        result = annual_forecasts(*args)
        for key in ("plain", "earn"):
            np.testing.assert_array_equal(result[key][:4], result[key][4:8])
            self.assertFalse(np.array_equal(result[key][:4], result[key][8:12]))
        self.assertEqual([a["cutoff"] for a in result["audit"]["annual"]],
                         ["2015-12-31", "2016-12-31"])

    def test_both_fits_use_exact_same_complete_training_rows_and_weights(self):
        args = sample_inputs()
        rows = np.flatnonzero(args[0] == np.datetime64("2014-05-31"))
        args[2][rows[0], 0] = np.nan
        args[3][rows[1], 0] = np.inf
        args[4][rows[2]] = np.nan
        args[5][rows[3]] = False
        with patch.object(earnings_model, "_fit_ridge", wraps=earnings_model._fit_ridge) as fit:
            result = annual_forecasts(*args)
        self.assertEqual(fit.call_count, 4)
        for i, row in enumerate(result["audit"]["annual"]):
            left, right = fit.call_args_list[2 * i].args, fit.call_args_list[2 * i + 1].args
            cutoff = np.datetime64(row["cutoff"])
            mask = (args[5] & np.isfinite(args[4]) & np.isfinite(args[2]).all(axis=1)
                    & np.isfinite(args[3]).all(axis=1) & (args[1] <= cutoff)
                    & (args[0] > np.datetime64(row["lookback_exclusive_start"])))
            np.testing.assert_array_equal(left[0], args[2][mask])
            np.testing.assert_array_equal(right[0], np.column_stack((args[2][mask], args[3][mask])))
            np.testing.assert_array_equal(left[1], args[4][mask])
            np.testing.assert_array_equal(right[1], left[1])
            np.testing.assert_array_equal(right[2], left[2])
            np.testing.assert_array_equal(left[2], month_balanced_weights(args[0][mask], cutoff, 24.))
            self.assertEqual(left[3], 10.)
            self.assertEqual(right[3], 10.)
            self.assertEqual(row["training_rows"], int(mask.sum()))
            self.assertTrue(row["plain_and_earn_share_rows"])

    def test_five_year_start_is_exclusive_and_label_cutoff_inclusive(self):
        args = sample_inputs()
        result = annual_forecasts(*args)
        row = result["audit"]["annual"][0]
        self.assertEqual(row["lookback_exclusive_start"], "2010-12-31")
        self.assertEqual(row["first_training_signal"], "2011-01-31")
        self.assertEqual(row["latest_training_signal"], "2015-11-30")
        self.assertEqual(row["latest_label_end"], "2015-12-31")
        self.assertEqual(row["training_months"], 59)
        changed = [a.copy() for a in args]
        changed[4][changed[0] <= np.datetime64("2010-12-31")] += 1000.
        after = annual_forecasts(*changed)
        for key in ("plain", "earn"):
            np.testing.assert_array_equal(result[key], after[key])

    def test_invalid_events_are_nan_in_both_arms(self):
        args = sample_inputs()
        args[9][0] = False
        args[8][1, 0] = np.nan
        args[7][2, 0] = np.inf
        result = annual_forecasts(*args)
        for key in ("plain", "earn"):
            self.assertTrue(np.isnan(result[key][:3]).all())
            self.assertTrue(np.isfinite(result[key][3:]).all())
            np.testing.assert_array_equal(np.isfinite(result[key]), result["available"])
        self.assertEqual(result["available"].dtype, np.dtype(bool))
        row = result["audit"]["annual"][0]
        self.assertEqual(row["invalid_or_ineligible_event_rows"], 3)
        self.assertEqual(row["prediction_rows"], 5)

    def test_minimum_history_counts_months_after_common_filter(self):
        args = sample_inputs()
        args[6][:] = np.datetime64("2012-01-31")
        args[5][:] = args[0] >= np.datetime64("2010-06-01")
        # June 2010 through November 2011: exactly 18 completed signal months.
        enough = annual_forecasts(*args)
        self.assertTrue(enough["available"].all())
        self.assertEqual(enough["audit"]["annual"][0]["training_months"], 18)
        args[3][args[0].astype("datetime64[M]") == np.datetime64("2010-06"), 0] = np.nan
        short = annual_forecasts(*args)
        self.assertEqual(short["audit"]["annual"][0]["training_months"], 17)
        self.assertGreater(short["audit"]["annual"][0]["training_rows"], 18)
        self.assertEqual(short["audit"]["annual"][0]["status"], "cold_start")
        self.assertFalse(short["available"].any())
        self.assertTrue(np.isnan(short["plain"]).all())
        self.assertTrue(np.isnan(short["earn"]).all())

    def test_no_training_rows_and_no_events_are_supported(self):
        args = sample_inputs()
        no_training = [a[:0] if i < 6 else a for i, a in enumerate(args)]
        result = annual_forecasts(*no_training)
        self.assertFalse(result["available"].any())
        self.assertTrue(np.isnan(result["plain"]).all())
        self.assertTrue(np.isnan(result["earn"]).all())
        no_events = [a if i < 6 else a[:0] for i, a in enumerate(args)]
        result = annual_forecasts(*no_events)
        for key in ("plain", "earn", "available"):
            self.assertEqual(result[key].shape, (0,))
        self.assertEqual(result["audit"]["annual"], [])

    def test_event_features_cannot_influence_scaling_or_other_predictions(self):
        args = sample_inputs()
        before = annual_forecasts(*args)
        changed = [a.copy() for a in args]
        changed[7][-1] = 1e12
        changed[8][-1] = -1e12
        after = annual_forecasts(*changed)
        for key in ("plain", "earn"):
            np.testing.assert_array_equal(before[key][:-1], after[key][:-1])

    def test_constant_features_and_tied_predictions_are_finite(self):
        args = sample_inputs()
        for i in (2, 3, 7, 8):
            args[i][:] = 1.
        args[4][:] = .02
        result = annual_forecasts(*args)
        self.assertTrue(result["available"].all())
        for key in ("plain", "earn"):
            np.testing.assert_allclose(result[key], .02, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(result[key][:4], result[key][4:8])
        np.testing.assert_array_equal(result["plain"], result["earn"])

    def test_added_constant_feature_does_not_change_plain_prediction(self):
        args = sample_inputs()
        args[3][:] = 3.
        args[8][:] = 3.
        result = annual_forecasts(*args)
        np.testing.assert_allclose(result["plain"], result["earn"], rtol=0, atol=1e-15)

    def test_month_balancing_and_24_month_half_life(self):
        dates = np.array(["2020-01-31"] * 2 + ["2022-01-31"] * 7, dtype="datetime64[D]")
        uniform = month_balanced_weights(dates, "2022-12-31", 0.)
        self.assertAlmostEqual(uniform[:2].sum(), uniform[2:].sum())
        decayed = month_balanced_weights(dates, "2022-12-31", 24.)
        self.assertAlmostEqual(decayed[:2].sum() / decayed[2:].sum(), .5)
        self.assertAlmostEqual(decayed.sum(), len(dates))

    def test_audit_is_json_serializable_and_forecasts_keep_event_order(self):
        args = sample_inputs()
        original = annual_forecasts(*args)
        json.dumps(original["audit"], allow_nan=False)
        order = np.array([11, 0, 7, 9, 2, 6, 4, 10, 8, 1, 5, 3])
        shuffled = args[:6] + [a[order] for a in args[6:]]
        result = annual_forecasts(*shuffled)
        for key in ("plain", "earn", "available"):
            np.testing.assert_array_equal(result[key], original[key][order])

    def test_invalid_shapes_horizons_dates_and_parameters_are_rejected(self):
        for index, replacement in ((1, np.array(["2010-01-31"])),
                                    (2, np.zeros(3)), (3, np.zeros((432, 0))),
                                    (7, np.zeros((12, 5))), (9, np.ones(11))):
            with self.subTest(index=index):
                args = sample_inputs()
                args[index] = replacement
                with self.assertRaises(ValueError):
                    annual_forecasts(*args)
        for index in (0, 1, 6):
            args = sample_inputs()
            args[index][0] = np.datetime64("NaT")
            with self.assertRaises(ValueError):
                annual_forecasts(*args)
        args = sample_inputs()
        args[1][0] = args[0][0]
        with self.assertRaises(ValueError):
            annual_forecasts(*args)
        for kwargs in ({"alpha": -1.}, {"alpha": np.inf}, {"half_life_months": -1.},
                       {"half_life_months": np.nan}, {"min_months": 0},
                       {"min_months": 1.5}, {"min_months": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                annual_forecasts(*sample_inputs(), **kwargs)


if __name__ == "__main__":
    unittest.main()
