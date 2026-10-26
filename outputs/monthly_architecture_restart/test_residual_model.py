"""Causality and weighting tests for the isolated monthly experiment."""
import json
import unittest

import numpy as np

from residual_model import (_fit_ridge, _oof_base_predictions, _predict,
                            annual_forecasts, month_balanced_weights)


def sample_inputs():
    months = np.arange(np.datetime64("2010-01", "M"),
                       np.datetime64("2019-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 4)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 4)
    rng = np.random.default_rng(71)
    X_base = rng.normal(size=(len(dates), 2))
    X_aux = rng.normal(size=(len(dates), 3))
    y = .03 * X_base[:, 0] - .01 * X_base[:, 1] + .015 * X_aux[:, 0]
    y += rng.normal(scale=.002, size=len(y))
    eligible = np.ones(len(y), dtype=bool)
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-01-31"],
                                    dtype="datetime64[D]"), 4)
    event_base = rng.normal(size=(len(event_dates), 2))
    event_aux = rng.normal(size=(len(event_dates), 3))
    event_eligible = np.ones(len(event_dates), dtype=bool)
    return [dates, ends, X_base, X_aux, y, eligible,
            event_dates, event_base, event_aux, event_eligible]


class TestCausalResidualLearning(unittest.TestCase):
    def test_months_have_equal_weight_before_decay(self):
        dates = np.array(["2020-01-31"] * 2 + ["2020-02-28"] * 7,
                         dtype="datetime64[D]")
        weights = month_balanced_weights(dates, "2020-03-31", 0)
        self.assertAlmostEqual(weights[:2].sum(), weights[2:].sum())
        self.assertAlmostEqual(weights.sum(), len(dates))

    def test_age_decay_is_applied_to_month_totals(self):
        dates = np.array(["2020-01-31"] * 3 + ["2020-02-28"] * 8,
                         dtype="datetime64[D]")
        weights = month_balanced_weights(dates, "2020-03-31", 1)
        self.assertAlmostEqual(weights[:3].sum() / weights[3:].sum(), .5)
        with self.assertRaises(ValueError):
            month_balanced_weights(dates, "2019-12-31")

    def test_oof_never_sees_its_own_label_or_later_labels(self):
        args = sample_inputs()
        dates, ends, base, _, y, eligible = args[:6]
        before, audit = _oof_base_predictions(dates, ends, base, y, eligible,
                                               10., 24., 18)
        target = np.datetime64("2014-05-31")
        changed = y.copy()
        changed[dates >= target] += 1000.
        after, _ = _oof_base_predictions(dates, ends, base, changed, eligible,
                                          10., 24., 18)
        np.testing.assert_array_equal(before[dates <= target], after[dates <= target])
        self.assertTrue(np.isfinite(before[dates == target]).all())
        row = next(row for row in audit if row["signal_date"] == str(target))
        self.assertLess(row["latest_training_signal"][:7], str(target)[:7])
        self.assertLessEqual(row["latest_training_end"], str(target))

    def test_future_labels_cannot_change_frozen_annual_forecast(self):
        args = sample_inputs()
        before = annual_forecasts(*args)
        altered = [a.copy() for a in args]
        future = altered[1] > np.datetime64("2015-12-31")
        altered[4][future] = 500. + altered[4][future] * 100.
        altered[2][altered[0] > np.datetime64("2015-12-31")] *= 1000.
        altered[3][altered[0] > np.datetime64("2015-12-31")] *= 1000.
        after = annual_forecasts(*altered)
        compare = args[6] < np.datetime64("2017-01-01")
        for key in ("direct", "residual"):
            np.testing.assert_array_equal(before[key][compare], after[key][compare])
            self.assertGreater(float(np.abs(before[key][compare]).sum()), 0)

    def test_annual_audit_and_training_set_are_shared(self):
        args = sample_inputs()
        result = annual_forecasts(*args)
        audit = result["audit"]
        json.dumps(audit, allow_nan=False)
        for row in audit["annual"]:
            self.assertEqual(row["cutoff"], "%04d-12-31" % (row["year"] - 1))
            self.assertLessEqual(row["latest_label_end"], row["cutoff"])
            self.assertTrue(row["direct_and_residual_share_rows"])
            self.assertGreaterEqual(row["residual_months"], 18)
        # Independently construct both fits, including exactly the same subset.
        dates, ends, base, aux, y, eligible, event_dates, _, event_aux, _ = args
        oof, _ = _oof_base_predictions(dates, ends, base, y, eligible, 10., 24., 18)
        cutoff = np.datetime64("2015-12-31")
        mask = (eligible & np.isfinite(oof) & (ends <= cutoff)
                & (dates > np.datetime64("2010-12-31")))
        weights = month_balanced_weights(dates[mask], cutoff, 24.)
        event_mask = event_dates < np.datetime64("2017-01-01")
        for key, target in (("direct", y), ("residual", y - oof)):
            model = _fit_ridge(aux[mask], target[mask], weights, 10.)
            expected = _predict(model, event_aux[event_mask])
            np.testing.assert_allclose(result[key][event_mask], expected,
                                       rtol=0., atol=1e-15)

    def test_cold_start_is_zero_not_in_sample_prediction(self):
        args = sample_inputs()
        args[6][:] = np.datetime64("2011-01-31")
        result = annual_forecasts(*args)
        np.testing.assert_array_equal(result["direct"], 0.)
        np.testing.assert_array_equal(result["residual"], 0.)
        self.assertEqual(result["audit"]["annual"][0]["status"], "cold_start")

    def test_nonfinite_and_ineligible_events_get_zero_correction(self):
        args = sample_inputs()
        args[9][0] = False
        args[8][1, 0] = np.nan
        args[7][2, 0] = np.inf
        result = annual_forecasts(*args)
        for key in ("direct", "residual"):
            np.testing.assert_array_equal(result[key][:3], 0.)
            self.assertTrue(np.isfinite(result[key]).all())
        row = result["audit"]["annual"][0]
        self.assertEqual(row["invalid_or_ineligible_event_rows"], 3)

    def test_invalid_target_horizon_rejected(self):
        args = sample_inputs()
        args[1][0] = args[0][0]
        with self.assertRaises(ValueError):
            annual_forecasts(*args)

    def test_scales_fit_training_rows_only(self):
        args = sample_inputs()
        original = annual_forecasts(*args)
        changed = [a.copy() for a in args]
        changed[8][-1] = 1e12
        other = annual_forecasts(*changed)
        for key in ("direct", "residual"):
            np.testing.assert_array_equal(original[key][:-1], other[key][:-1])

    def test_einsum_fit_matches_augmented_least_squares_reference(self):
        rng = np.random.default_rng(89)
        X = rng.normal(size=(43, 5))
        X[:, 4] = 1.  # Constant columns must also agree.
        y = rng.normal(scale=.1, size=43)
        weights = rng.uniform(.1, 2., size=43)
        alpha = 10.
        model = _fit_ridge(X, y, weights, alpha)
        mean = np.average(X, axis=0, weights=weights)
        scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.)
        z = (X - mean) / scale
        intercept = np.average(y, weights=weights)
        design = np.vstack((np.sqrt(weights)[:, None] * z, np.sqrt(alpha) * np.eye(5)))
        target = np.r_[np.sqrt(weights) * (y - intercept), np.zeros(5)]
        expected_coef = np.linalg.lstsq(design, target, rcond=None)[0]
        expected_prediction = np.sum(z * expected_coef, axis=1) + intercept
        np.testing.assert_allclose(model[2], expected_coef, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(_predict(model, X), expected_prediction,
                                   atol=1e-14, rtol=1e-12)

    def test_nonfinite_linear_algebra_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "nonfinite features"):
            _fit_ridge(np.array([[1.], [np.inf]]), np.ones(2), np.ones(2), 10.)
        with self.assertRaisesRegex(ValueError, "prediction coefficients"):
            _predict((np.zeros(1), np.ones(1), np.array([np.nan]), 0.), np.ones((2, 1)))


if __name__ == "__main__":
    unittest.main()
