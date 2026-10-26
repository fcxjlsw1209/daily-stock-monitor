"""Causal timing, partial pooling, pairwise symmetry and cold-start tests."""
import json
import unittest

import numpy as np

from models import (annual_forecasts, _fit_partial, _predict_partial, _fit_pairwise,
                    _pairwise_examples, _fit_ridge, _predict, month_balanced_weights)

KEYS = ("pooled", "partial_10", "partial_100", "independent", "pairwise")


def fixture():
    rng = np.random.default_rng(421)
    months = np.arange(np.datetime64("2010-01", "M"), np.datetime64("2018-01", "M"))
    dates = np.repeat((months + 1).astype("datetime64[D]") - 1, 4)
    ends = np.repeat((months + 2).astype("datetime64[D]") - 1, 4)
    X = rng.normal(size=(len(dates), 3))
    kind = np.tile([0, 0, 1, 1], len(months))
    y = .03 * X[:, 0] + .02 * X[:, 1] * (2 * kind - 1)
    y += rng.normal(0., .005, len(y))
    good = np.ones(len(y), dtype=bool)
    event_dates = np.repeat(np.array(["2016-01-29", "2016-12-30", "2017-03-31"], dtype="datetime64[D]"), 4)
    event_X = rng.normal(size=(len(event_dates), 3))
    event_good = np.ones(len(event_dates), dtype=bool)
    event_type = np.tile([0, 0, 1, 1], 3)
    return [dates, ends, X, y, good, kind, event_dates, event_X, event_good, event_type]


class TestAnnualModels(unittest.TestCase):
    def test_future_labels_and_features_do_not_change_past_annual_predictions(self):
        args = fixture()
        original = annual_forecasts(*args)
        changed = [a.copy() for a in args]
        later = changed[1] > np.datetime64("2015-12-31")
        changed[3][later] += 1000.
        changed[2][changed[0] > np.datetime64("2015-12-31")] *= 1000.
        modified = annual_forecasts(*changed)
        old = args[6] < np.datetime64("2017-01-01")
        for key in KEYS:
            np.testing.assert_array_equal(original[key][old], modified[key][old])
            np.testing.assert_array_equal(original["available"][key][old], True)
        for record in original["audit"]["annual"]:
            self.assertEqual(record["cutoff"], "%04d-12-31" % (record["year"] - 1))
            self.assertLessEqual(record["latest_label_end"], record["cutoff"])
        json.dumps(original["audit"], allow_nan=False)

    def test_future_events_do_not_change_past_scaling(self):
        args = fixture()
        original = annual_forecasts(*args)
        args[7][-1] = 1e8
        changed = annual_forecasts(*args)
        for key in KEYS:
            np.testing.assert_array_equal(original[key][:-1], changed[key][:-1])

    def test_cold_start_zero_is_explicitly_unavailable(self):
        args = fixture()
        args[6][:] = np.datetime64("2011-01-31")
        result = annual_forecasts(*args)
        for key in KEYS:
            np.testing.assert_array_equal(result[key], 0.)
            np.testing.assert_array_equal(result["available"][key], False)

    def test_fitted_zero_is_available(self):
        args = fixture()
        args[3][:] = 0.
        result = annual_forecasts(*args)
        for key in KEYS:
            np.testing.assert_array_equal(result[key], 0.)
            np.testing.assert_array_equal(result["available"][key], True)

    def test_independent_type_requires_its_own_18_months(self):
        args = fixture()
        args[5][:] = 0
        late_type = (args[0] >= np.datetime64("2015-01-01")) & (np.arange(len(args[0])) % 4 >= 2)
        args[5][late_type] = 1
        result = annual_forecasts(*args)
        before = args[6] < np.datetime64("2017-01-01")
        etf = before & (args[9] == 1)
        stock = before & (args[9] == 0)
        np.testing.assert_array_equal(result["available"]["independent"][etf], False)
        np.testing.assert_array_equal(result["available"]["independent"][stock], True)
        np.testing.assert_array_equal(result["available"]["pooled"][etf], True)
        self.assertLess(result["audit"]["annual"][0]["independent"]["etf"]["training_months"], 18)

    def test_nonfinite_or_ineligible_events_never_get_correction(self):
        args = fixture()
        args[7][0, 0] = np.nan
        args[8][1] = False
        result = annual_forecasts(*args)
        for key in KEYS:
            np.testing.assert_array_equal(result[key][:2], 0.)
            np.testing.assert_array_equal(result["available"][key][:2], False)
            self.assertTrue(np.isfinite(result[key]).all())

    def test_label_horizon_and_type_are_validated(self):
        args = fixture()
        args[1][0] = np.datetime64("2010-03-31")
        with self.assertRaisesRegex(ValueError, "next calendar month"):
            annual_forecasts(*args)
        args = fixture()
        args[5][0] = 2
        with self.assertRaisesRegex(ValueError, "0=stock"):
            annual_forecasts(*args)

    def test_fixed_flags_take_priority_over_legacy_subset_scores(self):
        args = fixture()
        flags = np.tile([True, True, True, False], len(args[0]) // 4)
        original = annual_forecasts(*args, boundary_top3=flags)
        changed = annual_forecasts(*args, boundary_top3=flags,
                                   boundary_scores=np.full(len(flags), np.nan))
        for key in KEYS:
            np.testing.assert_array_equal(original[key], changed[key])
            np.testing.assert_array_equal(original["available"][key], changed["available"][key])
        self.assertEqual(original["audit"]["boundary_weighting"],
                         "fixed_signal_time_top3_flags_crossing_2x_no_rerank")
        with self.assertRaisesRegex(ValueError, "boolean flag"):
            annual_forecasts(*args, boundary_top3=flags.astype(int))


class TestPartialPooling(unittest.TestCase):
    def test_deviation_shrinks_and_converges_to_pooled_model(self):
        rng = np.random.default_rng(541)
        X = rng.normal(size=(500, 2))
        kind = np.tile([0, 1], 250)
        y = .02 * X[:, 0] + .08 * X[:, 0] * (2 * kind - 1)
        w = np.ones(500)
        low = _fit_partial(X, y, kind, w, 10., 10.)
        high = _fit_partial(X, y, kind, w, 10., 100.)
        near_pool = _fit_partial(X, y, kind, w, 10., 1e10)
        pooled = _fit_ridge(X, y, w, 10.)
        self.assertLess(np.linalg.norm(high["coefficient"][2:]),
                        np.linalg.norm(low["coefficient"][2:]))
        np.testing.assert_allclose(_predict_partial(near_pool, X, kind), _predict(pooled, X),
                                   atol=3e-9, rtol=1e-6)

    def test_effect_code_makes_type_specific_slopes_not_ticker_labels(self):
        x = np.tile(np.linspace(-2., 2., 101), 2)[:, None]
        kind = np.repeat([0, 1], 101)
        y = x[:, 0] * np.where(kind == 0, -.05, .05)
        model = _fit_partial(x, y, kind, np.ones(len(y)), 10., 10.)
        forecast = _predict_partial(model, np.ones((2, 1)), np.array([0, 1]))
        self.assertLess(forecast[0], 0.)
        self.assertGreater(forecast[1], 0.)
        self.assertAlmostEqual(forecast[0], -forecast[1], places=12)


class TestPairwiseModels(unittest.TestCase):
    def test_all_within_month_pairs_and_month_balancing(self):
        dates = np.array(["2020-01-31"] * 3 + ["2020-02-28"] * 4, dtype="datetime64[D]")
        X = np.arange(14, dtype=float).reshape(7, 2)
        y = np.linspace(-.1, .1, 7)
        dx, dy, w, months = _pairwise_examples(dates, X, y, "2020-03-31", 0.)
        self.assertEqual(len(dy), 3 + 6)
        self.assertEqual(dx.shape, (9, 2))
        self.assertAlmostEqual(w[months == np.datetime64("2020-01", "M")].sum(),
                               w[months == np.datetime64("2020-02", "M")].sum())
        self.assertAlmostEqual(w.sum(), 7.)

    def test_boundary_weights_depend_only_on_past_baseline(self):
        dates = np.repeat(np.datetime64("2020-01-31"), 4)
        X = np.arange(8, dtype=float).reshape(4, 2)
        score = np.array([4., 3., 2., 1.])
        original = _pairwise_examples(dates, X, np.arange(4.), "2020-02-29", 0., score)
        altered = _pairwise_examples(dates, X, np.array([100., -5., 12., -100.]), "2020-02-29", 0., score)
        np.testing.assert_array_equal(original[2], altered[2])
        np.testing.assert_allclose(original[2] / original[2][0], [1., 1., 2., 1., 2., 2.])
        self.assertFalse(np.array_equal(original[1], altered[1]))

    def test_future_label_filter_cannot_promote_fourth_into_top_three(self):
        # The full signal-time basket was 0,1,2. Security 0 subsequently lacks
        # a completed label; security 3 must remain outside the old top three.
        dates = np.repeat(np.datetime64("2020-01-31"), 5)
        X = np.arange(10, dtype=float).reshape(5, 2)
        y = np.arange(5, dtype=float)
        flags = np.array([True, True, True, False, False])
        full = _pairwise_examples(dates, X, y, "2020-02-29", 0., boundary_top3=flags)
        valid_label = np.array([False, True, True, True, True])
        subset = _pairwise_examples(dates[valid_label], X[valid_label], y[valid_label],
                                    "2020-02-29", 0., boundary_top3=flags[valid_label])
        left, right = np.triu_indices(5, 1)
        remaining_pairs = valid_label[left] & valid_label[right]
        full_relative = full[2][remaining_pairs] / full[2][remaining_pairs].min()
        subset_relative = subset[2] / subset[2].min()
        np.testing.assert_allclose(full_relative, subset_relative)
        np.testing.assert_allclose(subset_relative, [1., 2., 2., 2., 2., 1.])
        # Both target perturbation and contradictory fallback scores leave
        # the precomputed flags, hence relative pair weights, unchanged.
        changed = _pairwise_examples(dates[valid_label], X[valid_label],
            y[valid_label] * -1000., "2020-02-29", 0.,
            boundary_scores=np.array([1., 2., 3., 4.]), boundary_top3=flags[valid_label])
        np.testing.assert_array_equal(subset[2], changed[2])

    def test_pairwise_fit_is_permutation_invariant_and_antisymmetric(self):
        rng = np.random.default_rng(889)
        dates = np.repeat(np.array(["2020-01-31", "2020-02-28", "2020-03-31"], dtype="datetime64[D]"), 4)
        X = rng.normal(size=(12, 3))
        y = .1 * X[:, 0] - .05 * X[:, 1]
        weights = month_balanced_weights(dates, "2020-04-30", 24.)
        model, audit = _fit_pairwise(dates, X, y, weights, "2020-04-30", 10., 24.)
        perm = rng.permutation(len(y))
        reordered, _ = _fit_pairwise(dates[perm], X[perm], y[perm], weights[perm],
                                     "2020-04-30", 10., 24.)
        np.testing.assert_allclose(_predict(model, X), _predict(reordered, X), atol=1e-13, rtol=1e-12)
        self.assertEqual(audit["pair_rows"], 18)
        self.assertTrue(audit["cross_asset_type_pairs_included"])
        zero = model[0]
        displacement = np.array([[.2, -.1, .3]])
        positive = _predict(model, zero + displacement)
        negative = _predict(model, zero - displacement)
        np.testing.assert_allclose(positive, -negative, atol=1e-14, rtol=1e-12)

    def test_pairwise_constant_target_within_month_has_zero_slopes(self):
        rng = np.random.default_rng(990)
        dates = np.repeat(np.array(["2020-01-31", "2020-02-28"], dtype="datetime64[D]"), 4)
        X = rng.normal(size=(8, 2))
        y = np.repeat([.15, -.35], 4)
        weights = month_balanced_weights(dates, "2020-03-31", 24.)
        model, _ = _fit_pairwise(dates, X, y, weights, "2020-03-31", 10., 24.)
        np.testing.assert_array_equal(_predict(model, X), 0.)


if __name__ == "__main__":
    unittest.main()
