"""Synthetic timing/normalization tests only; no strategy return evaluation."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


_SPEC = importlib.util.spec_from_file_location("core_timing70_features_for_tests", Path(__file__).with_name("features.py"))
features = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(features)
CAL = features._CAL.market_calendar("1998-10-01", "2035-12-31")


def panel():
    dates = CAL.sessions_in_range("2014-01-02", "2026-09-24")
    rng = np.random.default_rng(124252)
    returns = rng.normal([.0004, .0001, .0008], [.012, .009, .017], (len(dates)-1, 3))
    values = np.vstack((np.ones(3), np.exp(np.cumsum(returns, axis=0))))*[100., 57., 201.]
    return pd.DataFrame(values, index=dates, columns=["A", "B", "ETF"])


def metadata(*dates, symbols=("A", "B", "ETF")):
    return dict(dates=list(dates), symbols=[list(symbols) for _ in dates])


def raw_values(adj, meta, calm):
    prices, dates, pools, flags = features._validate(adj, meta, np.asarray(calm, bool))
    return features._raw_features(prices, dates, pools, flags, features._calendar(dates, prices))[0]


def centered(values):
    return pd.Series(values).rank(pct=True, method="average").to_numpy()-.5


class CoreTimingTests(unittest.TestCase):
    def test_calendar_formulas_and_current_sample_std_match_independent_math(self):
        adj = panel()
        meta = metadata("2024-05-31")
        anchors = [pd.Timestamp(x) for x in ("2024-03-28", "2023-05-31", "2023-03-31")]
        t = adj.index.get_loc(pd.Timestamp(meta["dates"][0]))
        p = adj.to_numpy()
        returns = p[t-62:t+1]/p[t-63:t]-1.
        sigma = np.std(returns, axis=0, ddof=1)
        twelve = adj.loc[anchors[0]].to_numpy()/adj.loc[anchors[1]].to_numpy()-1.
        fourteen = adj.loc[anchors[0]].to_numpy()/adj.loc[anchors[2]].to_numpy()-1.
        for flag in (True, False):
            raw = raw_values(adj, meta, [flag])
            expected = {"calendar_mixed": twelve if flag else fourteen/sigma,
                        "calendar12_all": twelve if flag else twelve/sigma}
            ranked, _ = features.build_features(adj, meta, np.array([flag]))
            for name, value in expected.items():
                np.testing.assert_allclose(raw[name][0], value, atol=1e-11, rtol=1e-11)
                np.testing.assert_array_equal(ranked[name][0], centered(value))

    def test_smoothing_uses_complete_raw_values_and_each_observation_risk(self):
        adj = panel()
        meta = metadata("2024-05-31")
        t = adj.index.get_loc(pd.Timestamp(meta["dates"][0]))
        p = adj.to_numpy()
        for flag in (True, False):
            raw = raw_values(adj, meta, [flag])
            for window in (5, 21):
                observations = []
                for u in range(t-window+1, t+1):
                    if flag:
                        observations.append(p[u-42]/p[u-252]-1.)
                    else:
                        daily = p[u-62:u+1]/p[u-63:u]-1.
                        observations.append((p[u-42]/p[u-294]-1.)/np.std(daily, axis=0, ddof=1))
                np.testing.assert_allclose(raw["trading_smooth"+str(window)][0], np.mean(observations, axis=0), atol=1e-11, rtol=1e-11)

    def test_average_raw_then_rank_is_not_average_historical_ranks(self):
        dates = CAL.sessions_in_range("2019-01-02", "2020-09-30")
        adj = pd.DataFrame(100., index=dates, columns=["A", "B", "ETF"])
        t = len(adj)-1
        raw = np.array([[.1, .3, .2], [.4, .1, .2], [.4, .1, .2], [.4, .1, .2], [.1, .3, .2]])
        for k, u in enumerate(range(t-4, t+1)):
            adj.iloc[u-42] = 100.*(1.+raw[k])
        ranked, _ = features.build_features(adj, metadata("2020-09-30"), np.array([True]))
        expected = centered(raw.mean(axis=0))
        np.testing.assert_array_equal(ranked["trading_smooth5"][0], expected)
        wrong = np.mean([centered(row) for row in raw], axis=0)
        self.assertFalse(np.allclose(expected, wrong))

    def test_real_monthends_handle_good_friday_leap_day_and_weekends(self):
        adj = panel()
        meta = metadata("2024-04-30", "2024-05-31")
        _, audit = features.build_features(adj, meta, np.array([True, False]))
        leap, friday = audit["anchors"]
        self.assertEqual(leap["actual_anchors"]["2"], "2024-02-29")
        self.assertEqual(leap["actual_anchors"]["12"], "2023-04-28")
        self.assertEqual(friday["actual_anchors"]["2"], "2024-03-28")
        self.assertEqual(friday["requested_anchors"]["2"], "2024-03-31")
        self.assertTrue(all(row["signal_is_official_month_end"] for row in audit["anchors"]))

    def test_non_monthend_uses_dateoffset_previous_session_and_official_closure(self):
        adj = panel()
        meta = metadata("2016-08-17", "2017-08-17", "2026-01-09")
        _, audit = features.build_features(adj, meta, np.array([True, True, True]))
        initial, weekend, closure = audit["anchors"]
        self.assertEqual(initial["actual_anchors"], {"2": "2016-06-17", "12": "2015-08-17", "14": "2015-06-17"})
        self.assertEqual(weekend["requested_anchors"]["2"], "2017-06-17")
        self.assertEqual(weekend["actual_anchors"]["2"], "2017-06-16")
        self.assertEqual(closure["requested_anchors"]["12"], "2025-01-09")
        self.assertEqual(closure["actual_anchors"]["12"], "2025-01-08")
        self.assertTrue(all(not row["signal_is_official_month_end"] for row in audit["anchors"]))
        for row in audit["anchors"]:
            self.assertTrue(all(value < row["signal_date"] for value in row["actual_anchors"].values()))

    def test_partial_current_month_is_not_promoted_to_monthend(self):
        adj = panel()
        _, audit = features.build_features(adj, metadata("2026-09-24"), np.array([True]))
        row = audit["anchors"][0]
        self.assertEqual(row["official_month_end"], "2026-09-30")
        self.assertFalse(row["signal_is_official_month_end"])
        self.assertEqual(row["actual_anchors"]["2"], "2026-07-24")

    def test_absent_scheduled_anchor_is_nan_without_searching_earlier_price(self):
        adj = panel().drop(pd.Timestamp("2024-03-28"))
        ranked, audit = features.build_features(adj, metadata("2024-05-31"), np.array([True]))
        self.assertEqual(audit["anchors"][0]["actual_anchors"]["2"], "2024-03-28")
        self.assertFalse(audit["anchors"][0]["anchor_sessions_present"]["2"])
        for name in ("calendar_mixed", "calendar12_all"):
            self.assertTrue(np.isnan(ranked[name]).all())

    def test_future_extension_prices_and_future_states_cannot_change_prefix(self):
        adj = panel()
        meta = metadata("2023-05-31", "2024-05-31")
        calm = np.array([True, False])
        full, _ = features.build_features(adj, meta, calm)
        cutoff = pd.Timestamp("2023-05-31")
        prefix, _ = features.build_features(adj.loc[:cutoff], metadata("2023-05-31"), calm[:1])
        changed = adj.copy()
        changed.loc[changed.index > cutoff] = np.nan
        altered_meta = copy.deepcopy(meta)
        altered_meta["symbols"][1].reverse()
        altered, _ = features.build_features(changed, altered_meta, np.array([True, True]))
        for name in features.FEATURE_NAMES:
            np.testing.assert_array_equal(full[name][:1], prefix[name])
            np.testing.assert_array_equal(full[name][:1], altered[name][:1])

    def test_missing_old_raw_observation_breaks21_not5_and_recent_nan_does_not_break_calm(self):
        adj = panel()
        meta = metadata("2024-05-31")
        t = adj.index.get_loc(pd.Timestamp(meta["dates"][0]))
        expected = raw_values(adj, meta, [True])
        changed = adj.copy()
        changed.iloc[t-57, 0] = np.nan  # Raw numerator at observation t-15.
        raw = raw_values(changed, meta, [True])
        self.assertTrue(np.isnan(raw["trading_smooth21"][0, 0]))
        np.testing.assert_array_equal(raw["trading_smooth5"], expected["trading_smooth5"])
        changed = adj.copy()
        changed.iloc[t-41:t+1] = np.nan
        raw = raw_values(changed, meta, [True])
        for name in ("trading_smooth5", "trading_smooth21"):
            np.testing.assert_array_equal(raw[name], expected[name])
        risk = raw_values(changed, meta, [False])
        self.assertTrue(np.isnan(risk["trading_smooth5"]).all())

    def test_zero_risk_missing_anchors_and_nonpositive_prices_remain_nan(self):
        adj = panel()
        adj["A"] = 100.
        meta = metadata("2024-05-31")
        risk, _ = features.build_features(adj, meta, np.array([False]))
        for name in features.FEATURE_NAMES:
            self.assertTrue(np.isnan(risk[name][0, 0]))
        calm, _ = features.build_features(adj, meta, np.array([True]))
        self.assertTrue(all(np.isfinite(value[0, 0]) for value in calm.values()))
        for invalid in (np.nan, np.inf, 0., -1.):
            changed = adj.copy()
            changed.loc[pd.Timestamp("2024-03-28"), "B"] = invalid
            actual, _ = features.build_features(changed, meta, np.array([True]))
            self.assertTrue(np.isnan(actual["calendar_mixed"][0, 1]))
            self.assertTrue(np.isnan(actual["calendar12_all"][0, 1]))

    def test_all49_finite_ranking_ignores_eligibility_and_preserves_order(self):
        adj = panel()
        all_prices = pd.DataFrame({"S%02d" % j: adj.iloc[:, j % 3]*(j+1) for j in range(49)})
        symbols = list(all_prices.columns)
        meta = metadata("2024-05-31", symbols=symbols)
        raw = raw_values(all_prices, meta, [False])
        ranked, _ = features.build_features(all_prices, meta, np.array([False]))
        meta["eligible"] = np.zeros((1, 49), bool)
        ignored, _ = features.build_features(all_prices, meta, np.array([False]))
        reverse, _ = features.build_features(all_prices, metadata("2024-05-31", symbols=symbols[::-1]), np.array([False]))
        for name in features.FEATURE_NAMES:
            np.testing.assert_array_equal(ranked[name][0], centered(raw[name][0]))
            np.testing.assert_array_equal(ranked[name], ignored[name])
            np.testing.assert_array_equal(ranked[name], reverse[name][:, ::-1])

    def test_shape_input_invariants_unknown_symbols_and_source_audit(self):
        adj = panel()
        saved = adj.copy(deep=True)
        meta = metadata("2024-05-31", symbols=("A", "UNKNOWN", "ETF"))
        result, audit = features.build_features(adj, meta, np.array([True]))
        pd.testing.assert_frame_equal(adj, saved)
        self.assertTrue(all(value.shape == (1, 3) and np.isnan(value[0, 1]) for value in result.values()))
        self.assertFalse(audit["trading_eligibility_modified"])
        self.assertFalse(audit["current_market_state_smoothed"])
        json.dumps(audit, allow_nan=False)
        self.assertIn(str(Path(features.__file__).resolve()), audit["source_hashes"])
        self.assertIn(str(features.CALENDAR_PATH.resolve()), audit["source_hashes"])
        self.assertIn(audit["calendar"]["manifest"], audit["source_hashes"])
        empty, _ = features.build_features(adj, metadata(), np.array([], bool))
        self.assertTrue(all(value.shape == (0, 0) for value in empty.values()))
        with self.assertRaises(ValueError):
            features.build_features(adj, meta, np.array([1]))
        with self.assertRaises(ValueError):
            features.build_features(adj, metadata("2024-03-29"), np.array([True]))
        with self.assertRaises(ValueError):
            features.build_features(adj.iloc[::-1], meta, np.array([True]))


if __name__ == "__main__":
    unittest.main()
