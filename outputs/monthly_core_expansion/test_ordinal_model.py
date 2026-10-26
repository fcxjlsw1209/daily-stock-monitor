import json
import unittest

import numpy as np
import pandas as pd

from ordinal_model import build_targets, annual_forecasts, _NUM


def one_month():
    prices = pd.DataFrame([[10., 10., 10., 10.], [11., 30., 40., 11.]],
        index=pd.to_datetime(["2020-01-31", "2020-02-28"]), columns=["A", "B", "C", "SPY"])
    dates = np.array(["2020-01-31"] * 3, dtype="datetime64[D]")
    ends = np.array(["2020-02-28"] * 3, dtype="datetime64[D]")
    return prices, dates, ends, np.array(["A", "B", "C"]), np.ones(3, dtype=bool)


def forecast_fixture():
    rng = np.random.default_rng(2241)
    months = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2018-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 4)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 4)
    X = rng.normal(size=(len(dates), 10))
    y = .03 * X[:, 0] - .02 * X[:, 1] + rng.normal(0, .005, len(dates))
    targets = {"return": y, "ordinal": .8 * np.tanh(y / .05)}
    good = np.ones(len(dates), dtype=bool)
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-03-31"], dtype="datetime64[D]"), 4)
    event_X = rng.normal(size=(len(event_dates), 10))
    event_good = np.ones(len(event_dates), dtype=bool)
    return [dates, ends, X, targets, good, event_dates, event_X, event_good]


class BuildTargetsTests(unittest.TestCase):
    def test_unclipped_ordinal_distinguishes_magnitude_clipped_ties(self):
        targets, audit = build_targets(*one_month())
        np.testing.assert_allclose(targets["return"], [-2./3, 1./3, 1./3])
        np.testing.assert_allclose(targets["ordinal"], [-2./3, 0., 2./3])
        self.assertEqual(audit["months"][0]["magnitude_clipped_rows"], 2)
        self.assertEqual(audit["months"][0]["distinct_unclipped_payoffs"], 3)
        self.assertEqual(audit["complete_months"], 1)
        json.dumps(audit, allow_nan=False)

    def test_missing_one_eligible_label_excludes_whole_month_both_arms(self):
        args = list(one_month())
        args[0].loc[pd.Timestamp("2020-02-28"), "B"] = np.nan
        targets, audit = build_targets(*args)
        for arm in ("return", "ordinal"):
            self.assertTrue(np.isnan(targets[arm]).all())
        self.assertEqual(audit["dropped_months"], 1)
        self.assertEqual(audit["months"][0]["issues"][0]["symbols"], ["B"])
        np.testing.assert_array_equal(args[4], True)  # eligibility is never mutated

    def test_missing_ineligible_security_does_not_change_target_universe(self):
        args = list(one_month())
        args[0].loc[pd.Timestamp("2020-02-28"), "B"] = np.nan
        args[4][1] = False
        targets, audit = build_targets(*args)
        np.testing.assert_allclose(targets["ordinal"][[0, 2]], [-.5, .5])
        self.assertTrue(np.isnan(targets["ordinal"][1]))
        self.assertEqual(audit["matched_valid_rows"], 2)

    def test_missing_or_nonpositive_interior_price_including_spy_drops_entire_month(self):
        for symbol in ("B", "SPY"):
            for invalid in (np.nan, 0., -1., np.inf):
                args = list(one_month())
                args[0].loc[pd.Timestamp("2020-02-14")] = [10.5, 15., 20., 10.5]
                args[0] = args[0].sort_index()
                args[0].loc[pd.Timestamp("2020-02-14"), symbol] = invalid
                targets, audit = build_targets(*args)
                self.assertTrue(np.isnan(targets["return"]).all())
                self.assertTrue(np.isnan(targets["ordinal"]).all())
                self.assertEqual(audit["dropped_months"], 1)
                issue = audit["months"][0]["issues"][0]
                self.assertEqual(issue["symbols"], [symbol])
                self.assertEqual(issue["first_invalid_session_by_symbol"][symbol], "2020-02-14")

    def test_ties_use_midrank_and_all_ties_are_exact_zero(self):
        args = list(one_month())
        args[0].iloc[1, :3] = [10., 10., 20.]
        targets, _ = build_targets(*args)
        np.testing.assert_allclose(targets["ordinal"], [-1./3, -1./3, 2./3])
        args[0].iloc[1, :3] = 15.
        targets, _ = build_targets(*args)
        np.testing.assert_array_equal(targets["ordinal"], 0.)
        np.testing.assert_array_equal(targets["return"], 0.)

    def test_unsupported_prices_and_overflow_are_not_zero_imputed(self):
        for invalid in [0., -1., np.inf, np.nan]:
            args = list(one_month())
            args[0].iloc[1, 0] = invalid
            targets, audit = build_targets(*args)
            self.assertTrue(np.isnan(targets["return"]).all())
            self.assertTrue(np.isnan(targets["ordinal"]).all())
            self.assertEqual(audit["dropped_months"], 1)
        args = list(one_month())
        args[0].iloc[0, 0], args[0].iloc[1, 0] = 1e-308, 1e308
        targets, audit = build_targets(*args)
        self.assertTrue(np.isnan(targets["return"]).all())
        self.assertEqual(audit["months"][0]["issues"][0]["kind"], "nonfinite_simple_payoff")

    def test_missing_spy_or_endpoint_excludes_month_for_both(self):
        for missing in ("SPY", "endpoint"):
            args = list(one_month())
            args[0] = args[0].drop(columns="SPY") if missing == "SPY" else args[0].iloc[:1]
            targets, audit = build_targets(*args)
            self.assertTrue(np.isnan(targets["ordinal"]).all())
            self.assertEqual(audit["dropped_months"], 1)

    def test_target_rows_are_permutation_equivariant(self):
        args = list(one_month())
        original, _ = build_targets(*args)
        order = np.array([2, 0, 1])
        changed, _ = build_targets(args[0], *(v[order] for v in args[1:]))
        for arm in ("return", "ordinal"):
            np.testing.assert_allclose(changed[arm], original[arm][order])

    def test_bad_shapes_horizon_duplicate_security_and_time_index_rejected(self):
        args = list(one_month())
        with self.assertRaisesRegex(ValueError, "flattened"):
            build_targets(*args[:3], args[3][:2], args[4])
        args = list(one_month())
        args[2][0] = np.datetime64("2020-03-31")
        with self.assertRaisesRegex(ValueError, "next calendar month"):
            build_targets(*args)
        args = list(one_month())
        args[3][1] = "A"
        with self.assertRaisesRegex(ValueError, "duplicate eligible"):
            build_targets(*args)
        args = list(one_month())
        args[0].index = args[0].index.tz_localize("UTC")
        with self.assertRaisesRegex(ValueError, "timezone-naive"):
            build_targets(*args)


class AnnualForecastTests(unittest.TestCase):
    def test_future_labels_features_and_availability_do_not_change_earlier_forecasts(self):
        args = forecast_fixture()
        original, audit = annual_forecasts(*args)
        later = args[1] > np.datetime64("2015-12-31")
        args[2][later] *= 1000.
        args[3]["return"][later] = .9
        args[3]["ordinal"][later] = -.9
        args[4][later] = False
        modified, _ = annual_forecasts(*args)
        earlier = args[5] < np.datetime64("2017-01-01")
        for key in ("return", "ordinal", "available"):
            np.testing.assert_array_equal(original[key][earlier], modified[key][earlier])
        for row in audit["annual"]:
            self.assertLessEqual(row["latest_label_end"], row["cutoff"])
            self.assertEqual(row["cutoff"], "%04d-12-31" % (row["year"] - 1))
        json.dumps(audit, allow_nan=False)

    def test_scaler_is_past_only_and_magnitude_arm_matches_existing_ridge(self):
        args = forecast_fixture()
        pred, audit = annual_forecasts(*args)
        cutoff = np.datetime64("2015-12-31")
        ix = (args[1] <= cutoff) & (args[0] > _NUM._years_before(cutoff))
        weights = _NUM.month_balanced_weights(args[0][ix], cutoff, 24.)
        model = _NUM._fit_ridge(args[2][ix], args[3]["return"][ix], weights, 10.)
        ev = args[5] < np.datetime64("2017-01-01")
        np.testing.assert_allclose(pred["return"][ev], _NUM._predict(model, args[6][ev]))
        np.testing.assert_array_equal(audit["annual"][0]["training_feature_mean"], model[0])
        np.testing.assert_array_equal(audit["annual"][0]["training_feature_scale"], model[1])
        args[6][-1] *= 1e8
        changed, audit2 = annual_forecasts(*args)
        for key in ("return", "ordinal"):
            np.testing.assert_array_equal(pred[key][:-1], changed[key][:-1])
        self.assertEqual(audit["annual"][0]["training_feature_scale"], audit2["annual"][0]["training_feature_scale"])

    def test_minimum18_months_and_neutral_fitted_zero(self):
        for number in (17, 18):
            args = forecast_fixture()
            keep = args[0].astype("datetime64[M]") < np.datetime64("2010-01", "M") + number
            args[4] = keep
            for arm in ("return", "ordinal"):
                args[3][arm][:] = np.nan
                args[3][arm][keep] = 0.
            args[5][:] = np.datetime64("2012-01-31")
            pred, audit = annual_forecasts(*args)
            np.testing.assert_array_equal(pred["return"], 0.)
            np.testing.assert_array_equal(pred["ordinal"], 0.)
            np.testing.assert_array_equal(pred["available"], number == 18)
            self.assertEqual(audit["annual"][0]["training_months"], number)

    def test_invalid_training_label_drops_entire_month_for_both_arms(self):
        args = forecast_fixture()
        _, original_audit = annual_forecasts(*args)
        ix = np.flatnonzero(args[0] == np.datetime64("2015-06-30"))
        args[3]["ordinal"][ix[0]] = np.nan
        _, audit = annual_forecasts(*args)
        self.assertIn("2015-06", audit["whole_month_exclusions_on_input_validation"])
        self.assertEqual(audit["annual"][0]["training_rows"], original_audit["annual"][0]["training_rows"] - 4)
        self.assertTrue(audit["matched_training_rows"])

    def test_current_event_unavailability_is_neutral_without_future_trade_filter(self):
        args = forecast_fixture()
        args[6][0, 0] = np.nan
        args[7][1] = False
        pred, _ = annual_forecasts(*args)
        for arm in ("return", "ordinal"):
            np.testing.assert_array_equal(pred[arm][:2], 0.)
            self.assertTrue(np.isfinite(pred[arm]).all())
        np.testing.assert_array_equal(pred["available"][:2], False)
        np.testing.assert_array_equal(pred["available"][2:], True)

    def test_shapes_and_unsupported_target_ranges_rejected(self):
        args = forecast_fixture()
        args[6] = args[6][:, :9]
        with self.assertRaisesRegex(ValueError, "10 features"):
            annual_forecasts(*args)
        args = forecast_fixture()
        args[3]["ordinal"][0] = 1.1
        with self.assertRaisesRegex(ValueError, "ordinal target"):
            annual_forecasts(*args)
        args = forecast_fixture()
        args[3]["return"][0] = 2.1
        with self.assertRaisesRegex(ValueError, "centered clipped return"):
            annual_forecasts(*args)
        args = forecast_fixture()
        args[4] = args[4].astype(int)
        with self.assertRaisesRegex(ValueError, "boolean"):
            annual_forecasts(*args)


if __name__ == "__main__":
    unittest.main()
