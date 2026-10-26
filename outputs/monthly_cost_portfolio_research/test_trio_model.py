import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("trio_under_test", Path(__file__).with_name("trio_model.py"))
model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model)


class TrioFeaturesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.index = pd.bdate_range("2019-01-01", "2020-03-31")
        self.symbols = np.array(["S%d" % i for i in range(9)])
        returns = rng.normal(.0004, .017, (len(self.index), 10))
        self.adj = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=self.index,
                                columns=list(self.symbols) + ["SPY"])
        self.day, self.end = "2020-01-31", "2020-02-28"
        self.eligible = np.ones(9, dtype=bool)
        self.score = np.arange(9., 0., -1.)
        self.plain = rng.uniform(-.5, .5, (9, 10))

    def build(self, adj=None, end="2020-02-28", eligible=None, score=None, plain=None):
        return model.build_month(self.adj if adj is None else adj, self.day, end,
            self.symbols, self.eligible if eligible is None else eligible,
            self.score if score is None else score, self.plain if plain is None else plain)

    def test_top8_all_slots_compete_baseline_first_and_centered_zero(self):
        out = self.build()
        self.assertEqual(out["indices"].shape, (56, 3))
        np.testing.assert_array_equal(out["indices"][0], [0, 1, 2])
        np.testing.assert_array_equal(out["baseline_indices"], [0, 1, 2])
        self.assertIn([5, 6, 7], out["indices"].tolist())
        self.assertFalse((out["indices"] == 8).any())
        np.testing.assert_array_equal(out["X_additive"][0], np.zeros(10))
        np.testing.assert_array_equal(out["X_joint"][0], np.zeros(13))
        self.assertEqual(out["y"][0], 0.)
        self.assertTrue(out["train_good"].all())
        self.assertEqual(out["audit"]["label"]["status"], "complete")
        np.testing.assert_allclose(out["base_score"], self.score[out["indices"]].mean(axis=1))

    def test_labels_are_equal_weight_log_not_average_individual_logs(self):
        out = self.build()
        r = self.adj.loc[self.end, self.symbols] / self.adj.loc[self.day, self.symbols] - 1.
        logs = np.log1p(r.to_numpy()[out["indices"]].mean(axis=1))
        np.testing.assert_allclose(out["y"], logs - logs[0], atol=1e-15)
        stock_logs = np.log1p(r.to_numpy()[out["indices"]]).mean(axis=1)
        self.assertGreater(np.max(np.abs(out["y"] - (stock_logs - stock_logs[0]))), 1e-6)

    def test_future_price_perturbation_and_event_without_end_cannot_change_features(self):
        base = self.build(end=None)
        changed = self.adj.copy()
        changed.loc[changed.index > self.day] = np.nan
        other = self.build(adj=changed, end=None)
        for key in ["indices", "base_score", "X_additive", "X_joint", "event_good"]:
            np.testing.assert_array_equal(base[key], other[key])
        self.assertTrue(np.isnan(other["y"]).all())
        self.assertFalse(other["label_good"].any())
        self.assertEqual(other["audit"]["label"]["status"], "not_requested")
        prefix = self.build(adj=self.adj.loc[:self.day], end=None)
        np.testing.assert_array_equal(base["X_joint"], prefix["X_joint"])

    def test_full_eligible_path_gap_outside_top8_drops_both_targets_not_candidates(self):
        old = self.build()
        for symbol in ["S8", "SPY"]:
            changed = self.adj.copy()
            changed.loc["2020-02-14", symbol] = np.nan
            other = self.build(adj=changed)
            self.assertFalse(other["label_good"].any())
            self.assertTrue(np.isnan(other["y"]).all())
            self.assertEqual(other["audit"]["label"]["status"], "whole_month_excluded")
            for key in ["indices", "X_additive", "X_joint", "event_good"]:
                np.testing.assert_array_equal(old[key], other[key])

    def test_ineligible_missing_future_is_not_required_label(self):
        eligible = self.eligible.copy(); eligible[8] = False
        changed = self.adj.copy(); changed.loc["2020-02-14", "S8"] = np.nan
        out = self.build(adj=changed, eligible=eligible)
        self.assertTrue(out["label_good"].all())

    def test_joint_risk_definition_matches_independent_three_asset_calculation(self):
        out = self.build()
        r = self.adj.loc[:self.day, self.symbols].tail(127).pct_change(fill_method=None).iloc[1:]
        def risk(row):
            z = r.iloc[:, row].to_numpy()
            corr = np.corrcoef(z.T)
            neg = np.minimum(z, 0.)
            cosine = [np.mean(neg[:, i] * neg[:, j]) /
                      np.sqrt(np.mean(neg[:, i] ** 2) * np.mean(neg[:, j] ** 2))
                      for i, j in [(0, 1), (0, 2), (1, 2)]]
            return np.array([np.mean(corr[np.triu_indices(3, 1)]), np.mean(cosine),
                             z.std(axis=0, ddof=1).std(ddof=0)])
        expected = np.stack([risk(row) for row in out["indices"]])
        np.testing.assert_allclose(out["X_joint"][:, 10:], expected - expected[0], atol=1e-15)
        np.testing.assert_allclose(out["X_joint"][:, :10], out["X_additive"])

    def test_missing_risk_keeps_combos_and_common_feature_availability(self):
        changed = self.adj.copy(); changed.loc["2020-01-15", "S7"] = np.nan
        out = self.build(adj=changed)
        self.assertEqual(len(out["indices"]), 56)
        contains7 = (out["indices"] == 7).any(axis=1)
        np.testing.assert_array_equal(out["event_good"], ~contains7)
        self.assertTrue(out["label_good"].all())
        self.assertTrue(np.isfinite(out["X_additive"]).all())
        short = self.build(adj=self.adj.loc[:self.day].tail(126), end=None)
        self.assertEqual(len(short["indices"]), 56)
        self.assertFalse(short["event_good"].any())

    def test_baseline_missing_feature_disables_all_forecasts_without_candidate_change(self):
        for missing in [np.nan, np.inf, -np.inf]:
            plain = self.plain.copy(); plain[0, 2] = missing
            out = self.build(plain=plain)
            self.assertEqual(len(out["indices"]), 56)
            self.assertFalse(out["event_good"].any())
            self.assertTrue(out["label_good"].all())

    def test_tie_order_fewer_than3_and_nonfinite_score_rejected(self):
        out = self.build(score=np.ones(9))
        np.testing.assert_array_equal(out["indices"][0], [0, 1, 2])
        order = np.array([8, 3, 6, 1, 0, 7, 5, 2, 4])
        tied = model.build_month(self.adj, self.day, self.end, self.symbols[order],
            self.eligible[order], np.ones(9), self.plain[order])
        self.assertEqual(tied["symbols"][0], ["S0", "S1", "S2"])
        self.assertEqual(out["symbols"], tied["symbols"])
        np.testing.assert_allclose(out["X_joint"], tied["X_joint"], atol=1e-15)
        eligible = np.zeros(9, bool); eligible[:2] = True
        out = self.build(eligible=eligible)
        self.assertEqual(out["indices"].shape, (0, 3))
        self.assertEqual(out["X_joint"].shape, (0, 13))
        scores = self.score.copy(); scores[2] = np.nan
        with self.assertRaises(ValueError): self.build(score=scores)
        with self.assertRaises(ValueError): self.build(end="2020-03-31")

    def test_symbol_input_permutation_keeps_same_ranked_triples_and_features(self):
        original = self.build()
        order = np.array([8, 3, 6, 1, 0, 7, 5, 2, 4])
        permuted = model.build_month(self.adj, self.day, self.end, self.symbols[order],
            self.eligible[order], self.score[order], self.plain[order])
        self.assertEqual(original["symbols"], permuted["symbols"])
        np.testing.assert_allclose(original["X_joint"], permuted["X_joint"], atol=1e-15)
        np.testing.assert_allclose(original["y"], permuted["y"], atol=1e-15)


class AnnualModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(89)
        signals = pd.date_range("2016-01-31", periods=60, freq=pd.offsets.MonthEnd())
        self.dates = np.repeat(signals.to_numpy(dtype="datetime64[D]"), 4)
        self.ends = np.repeat((signals + pd.offsets.MonthEnd(1)).to_numpy(dtype="datetime64[D]"), 4)
        self.X = rng.normal(size=(len(self.dates), 13))
        self.X[::4] = 0.
        self.y = np.einsum("ni,i->n", self.X, np.linspace(-.01, .02, 13), optimize=False)
        self.good = np.ones(len(self.dates), bool)
        self.event_dates = np.array(["2019-02-28", "2019-02-28", "2020-02-28", "2020-02-28"], dtype="datetime64[D]")
        self.event_X = rng.normal(size=(4, 13)); self.event_X[::2] = 0.
        self.event_good = np.ones(4, bool)

    def predict(self, **kw):
        args = dict(dates=self.dates, ends=self.ends, X=self.X, y=self.y, train_good=self.good,
                    event_dates=self.event_dates, event_X=self.event_X, event_good=self.event_good)
        args.update(kw)
        return model.annual_forecasts(**args)

    def test_year_cutoff_and_future_label_feature_availability_invariance(self):
        original = self.predict()
        future = self.ends > np.datetime64("2018-12-31")
        y, X, good = self.y.copy(), self.X.copy(), self.good.copy()
        y[future] += 1234.; X[future] *= 100.; good[future] = False
        changed = self.predict(y=y, X=X, train_good=good)
        np.testing.assert_array_equal(original["pred"][:2], changed["pred"][:2])
        np.testing.assert_array_equal(original["available"][:2], changed["available"][:2])
        a = original["audit"]["annual"][0]
        self.assertEqual(a["cutoff"], "2018-12-31")
        self.assertEqual(a["latest_label_end"], "2018-12-31")
        self.assertEqual(a["training_months"], 35)

    def test_baseline_and_cold_start_neutral_zero_are_distinguished(self):
        out = self.predict()
        np.testing.assert_array_equal(out["pred"][[0, 2]], np.zeros(2))
        self.assertTrue(out["available"].all())
        for months, expected in [(17, False), (18, True)]:
            good = np.zeros(len(self.good), bool); good[:months * 4] = True
            short = self.predict(train_good=good)
            self.assertEqual(bool(short["available"][0]), expected)
            if not expected: np.testing.assert_array_equal(short["pred"], np.zeros(4))

    def test_rms_scale_training_only_and_reference_numeric_fit(self):
        out = self.predict()
        select = self.good & (self.ends <= np.datetime64("2018-12-31")) & (self.dates > np.datetime64("2013-12-31"))
        weights = model._NUM.month_balanced_weights(self.dates[select], "2018-12-31", 24.)
        scale = np.sqrt(np.average(self.X[select] ** 2, axis=0, weights=weights))
        np.testing.assert_allclose(out["audit"]["annual"][0]["training_feature_rms"], scale)
        fit = model._NUM._fit_rms_ridge(self.X[select], self.y[select], weights)
        np.testing.assert_allclose(out["pred"][:2], model._NUM._predict(fit, self.event_X[:2]))
        changed_event = self.event_X.copy(); changed_event[2:] *= 1000
        other = self.predict(event_X=changed_event)
        self.assertEqual(out["audit"]["annual"][0]["training_feature_rms"], other["audit"]["annual"][0]["training_feature_rms"])

    def test_month_total_weights_are_balanced_then_age_decayed(self):
        dates = np.array(["2018-01-31"] * 2 + ["2018-02-28"] * 8 + ["2018-03-31"] * 5, dtype="datetime64[D]")
        w = model._NUM.month_balanced_weights(dates, "2018-12-31", 24.)
        sums = np.array([w[dates == d].sum() for d in np.unique(dates)])
        np.testing.assert_allclose(sums[1:] / sums[:-1], np.repeat(2 ** (1 / 24), 2))
        self.assertAlmostEqual(w.sum(), len(dates))

    def test_common_masks_preserve_additive_joint_training_counts(self):
        good = self.good.copy(); good[1::4] = False
        event_good = self.event_good.copy(); event_good[1] = False
        joint = self.predict(train_good=good, event_good=event_good)
        additive = self.predict(X=self.X[:, :10], event_X=self.event_X[:, :10],
                                train_good=good, event_good=event_good)
        np.testing.assert_array_equal(joint["available"], additive["available"])
        self.assertEqual([x["training_combinations"] for x in joint["audit"]["annual"]],
                         [x["training_combinations"] for x in additive["audit"]["annual"]])
        self.assertEqual(joint["pred"][1], 0.)

    def test_row_permutation_and_bad_input_handling(self):
        original = self.predict()
        perm = np.random.default_rng(5).permutation(len(self.y))
        other = self.predict(dates=self.dates[perm], ends=self.ends[perm], X=self.X[perm],
                             y=self.y[perm], train_good=self.good[perm])
        np.testing.assert_allclose(original["pred"], other["pred"], atol=1e-14)
        with self.assertRaises(ValueError): self.predict(X=self.X[:, :9])
        with self.assertRaises(ValueError): self.predict(train_good=self.good.astype(int))
        bad = self.X.copy(); bad[3, 0] = np.nan
        with self.assertRaises(ValueError): self.predict(X=bad)
        event = self.event_X.copy(); event[1, 0] = np.nan
        out = self.predict(event_X=event)
        self.assertFalse(out["available"][1]); self.assertEqual(out["pred"][1], 0.)


if __name__ == "__main__":
    unittest.main()
