"""Synthetic tests of ranking history; no performance, holdings or outcomes."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location("residence_for_tests", Path(__file__).with_name("residence.py"))
feature = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature)


def fixture():
    dates = pd.bdate_range("2014-01-01", periods=1050)
    prices = pd.DataFrame(100., index=dates, columns=["A", "B", "C", "D"])
    ends = dates.to_series().groupby(dates.to_period("M")).last().tolist()
    chosen = ends[20:26]
    # Six distinct ordered ranks; window endpoints do not overlap one another.
    gains = [[.4, .3, .2, .1], [.1, .4, .2, .3], [.3, .2, .1, .4],
             [.2, .3, .4, .1], [.4, .1, .3, .2], [.3, .2, .4, .1]]
    for day, row in zip(chosen, gains):
        t = dates.get_loc(day)
        prices.iloc[t - 42] = 100. * (1. + np.asarray(row))
    meta = {"dates": [str(day.date()) for day in chosen], "symbols": [list(prices.columns) for _ in chosen]}
    return {"adj": prices}, meta, chosen


def run(p, meta, days=None, symbols=None):
    if days is None:
        days = [pd.Timestamp(meta["dates"][-1])]
    if symbols is None:
        symbols = [list(p["adj"].columns) for _ in days]
    return feature.build_features(p, meta, list(zip(days, symbols)))


def manual_ranks(p, day, symbols):
    t = p["adj"].index.get_loc(day)
    ratio = p["adj"].iloc[t - 42][symbols] / p["adj"].iloc[t - 252][symbols] - 1.
    return ratio.rank(pct=True) - .5


class TestResidence(unittest.TestCase):
    def test_exact_mean_minus_population_std_of_six_historical_ranks(self):
        p, meta, dates = fixture()
        result, audit = run(p, meta)
        ranks = np.stack([manual_ranks(p, d, list(p["adj"].columns)).to_numpy() for d in dates])
        expected = ranks.mean(axis=0) - ranks.std(axis=0, ddof=0)
        np.testing.assert_allclose(result["rank_residence"], expected[None, :], atol=1e-15)
        self.assertFalse(result["rank_residence_fallback"].any())
        self.assertEqual(audit["event_history"][0]["usable_observations"], [6] * 4)
        self.assertFalse(audit["trading_eligibility_modified"])

    def test_missing_months_are_not_replaced_by_older_months(self):
        p, meta, dates = fixture()
        monthends = p["adj"].index.to_series().groupby(p["adj"].index.to_period("M")).last()
        old = [monthends.loc[dates[0].to_period("M") - offset] for offset in [2, 1]]
        reduced = {"dates": old + [dates[0], dates[1], dates[-1]], "symbols": [list(p["adj"].columns)] * 5}
        result, audit = run(p, reduced, [dates[-1]])
        np.testing.assert_allclose(result["rank_residence"][0], manual_ranks(p, dates[-1], list(p["adj"].columns)))
        self.assertTrue(result["rank_residence_fallback"].all())
        self.assertEqual(audit["event_history"][0]["usable_observations"], [3] * 4)
        self.assertEqual(audit["event_history"][0]["snapshots"].count(None), 3)

    def test_four_observations_use_residence_three_fall_back(self):
        p, meta, dates = fixture()
        for count in [3, 4]:
            small = {"dates": dates[-count:], "symbols": [list(p["adj"].columns)] * count}
            result, audit = run(p, small, [dates[-1]])
            self.assertTrue((result["rank_residence_fallback"] == (count < 4)).all())
            self.assertEqual(audit["event_history"][0]["usable_observations"], [count] * 4)

    def test_union_events_supplies_history_after_training_metadata_ends(self):
        p, meta, dates = fixture()
        truncated = {"dates": meta["dates"][:2], "symbols": meta["symbols"][:2]}
        expected, _ = run(p, meta, dates)
        result, audit = run(p, truncated, dates)
        np.testing.assert_array_equal(expected["rank_residence"], result["rank_residence"])
        self.assertEqual(audit["event_history"][-1]["usable_observations"], [6] * 4)

    def test_historical_membership_is_used_instead_of_current_symbols(self):
        p, meta, dates = fixture()
        # A is absent in three past snapshots, even though its prices exist.
        for i in range(3):
            meta["symbols"][i] = ["B", "C", "D"]
        result, audit = run(p, meta)
        self.assertTrue(result["rank_residence_fallback"][0, 0])
        self.assertFalse(result["rank_residence_fallback"][0, 1:].any())
        self.assertEqual(audit["event_history"][0]["usable_observations"], [3, 6, 6, 6])
        expected = []
        for d, symbols in zip(dates, meta["symbols"]):
            expected.append(manual_ranks(p, d, symbols)["B"])
        self.assertAlmostEqual(result["rank_residence"][0, 1], np.mean(expected) - np.std(expected))

    def test_no_future_snapshot_or_price_use_and_prefix_invariance(self):
        p, meta, dates = fixture()
        signal = dates[3]
        expected, _ = run(p, meta, [signal])
        prefix = {"adj": p["adj"].loc[:signal].copy()}
        short, _ = run(prefix, meta, [signal])
        changed = {"adj": p["adj"].copy()}
        changed["adj"].loc[changed["adj"].index > signal] = 1e200
        future, _ = run(changed, meta, [signal])
        for key in expected:
            np.testing.assert_array_equal(expected[key], short[key])
            np.testing.assert_array_equal(expected[key], future[key])

    def test_price_return_skips_exact_last_42_sessions(self):
        p, meta, dates = fixture()
        expected, _ = run(p, meta)
        t = p["adj"].index.get_loc(dates[-1])
        p["adj"].iloc[t - 41:t + 1] *= [1e6, 7., .01, 100.]
        skipped, _ = run(p, meta)
        np.testing.assert_array_equal(expected["rank_residence"], skipped["rank_residence"])
        # t-42 is used; its alteration can change the latest cross-sectional rank.
        p["adj"].iloc[t - 42, 0] *= 100.
        included, _ = run(p, meta)
        self.assertFalse(np.array_equal(expected["rank_residence"], included["rank_residence"]))

    def test_positive_momentum_gate_and_complete_295_price_gate(self):
        p, meta, dates = fixture()
        for day in dates[:3]:
            t = p["adj"].index.get_loc(day)
            p["adj"].iloc[t - 42, 0] = 90.
        result, audit = run(p, meta)
        self.assertEqual(audit["event_history"][0]["usable_observations"][0], 3)
        self.assertTrue(result["rank_residence_fallback"][0, 0])
        # A non-endpoint missing price invalidates historical availability, never
        # turns that missing observation into a centered zero rank.
        for bad in [np.nan, 0., -1., np.inf]:
            changed = {"adj": p["adj"].copy()}
            t = changed["adj"].index.get_loc(dates[-1])
            changed["adj"].iloc[t - 100, 1] = bad
            invalid, aa = run(changed, meta)
            self.assertTrue(invalid["rank_residence_fallback"][0, 1])
            self.assertLess(aa["event_history"][0]["usable_observations"][1], 4)
            self.assertEqual(invalid["rank_residence"][0, 1], manual_ranks(changed, dates[-1], list(p["adj"].columns))["B"])

    def test_current_missing_momentum_remains_nan(self):
        p, meta, dates = fixture()
        t = p["adj"].index.get_loc(dates[-1])
        p["adj"].iloc[t - 42, 0] = np.nan
        result, audit = run(p, meta)
        # The same missing day also invalidates some prior 295-price checks.
        self.assertEqual(audit["event_history"][0]["usable_observations"][0], 3)
        self.assertTrue(np.isnan(result["rank_residence"][0, 0]))

    def test_295_complete_price_boundary_is_inclusive(self):
        p, _, _ = fixture()
        p["adj"] = pd.DataFrame(100 * np.exp(np.arange(len(p["adj"]))[:, None] * np.array([.001, .002, .003, .004])), index=p["adj"].index, columns=p["adj"].columns)
        for t, count in [(293, 0), (294, 1)]:
            result, audit = run(p, {"dates": [], "symbols": []}, [p["adj"].index[t]])
            self.assertEqual(audit["event_history"][0]["usable_observations"], [count] * 4)
            self.assertTrue(np.isfinite(result["rank_residence"]).all())
            self.assertTrue(result["rank_residence_fallback"].all())

    def test_scale_order_and_input_immutability(self):
        p, meta, dates = fixture()
        saved = p["adj"].copy(deep=True)
        expected, _ = run(p, meta)
        pd.testing.assert_frame_equal(saved, p["adj"])
        p["adj"] *= [31., .25, 128., 7.]
        scaled, _ = run(p, meta, symbols=[["D", "C", "B", "A"]])
        np.testing.assert_allclose(expected["rank_residence"], scaled["rank_residence"][:, ::-1])
        np.testing.assert_array_equal(expected["rank_residence_fallback"], scaled["rank_residence_fallback"][:, ::-1])

    def test_initial_partial_month_is_never_later_history(self):
        p, _, _ = fixture()
        dates = [pd.Timestamp(d) for d in ["2016-05-31", "2016-06-30", "2016-07-29", "2016-08-17", "2016-08-31", "2016-09-30"]]
        p["adj"] = pd.DataFrame(100 * np.exp(np.arange(len(p["adj"]))[:, None] * np.array([.001, .002, .003, .004])), index=p["adj"].index, columns=p["adj"].columns)
        meta = {"dates": dates, "symbols": [list(p["adj"].columns)] * len(dates)}
        _, aa = run(p, meta, [dates[3], dates[-1]])
        self.assertIn("2016-08-17", aa["event_history"][0]["snapshots"])
        self.assertNotIn("2016-08-31", aa["event_history"][0]["snapshots"])
        self.assertNotIn("2016-08-17", aa["event_history"][1]["snapshots"])
        self.assertIn("2016-08-31", aa["event_history"][1]["snapshots"])

    def test_unknown_current_symbol_missing_not_zero(self):
        p, meta, dates = fixture()
        meta["symbols"][-1] = ["A", "B", "C", "UNKNOWN"]
        result, _ = run(p, meta, symbols=[meta["symbols"][-1]])
        self.assertTrue(np.isnan(result["rank_residence"][0, -1]))
        self.assertTrue(result["rank_residence_fallback"][0, -1])

    def test_conflicting_duplicate_dates_and_invalid_event_schema_raise(self):
        p, meta, dates = fixture()
        with self.assertRaisesRegex(ValueError, "Conflicting same-date"):
            run(p, meta, symbols=[["A", "B", "C", "UNKNOWN"]])
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            run(p, meta, symbols=[["A", "A", "B", "C"]])
        with self.assertRaisesRegex(ValueError, "same number"):
            run(p, meta, dates[-2:], [["A"], ["A", "B"]])
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            run(p, {"dates": [], "symbols": []}, [pd.Timestamp("2016-01-02")])

    def test_flat_returns_are_not_positive_history_and_ties_use_average_rank(self):
        p, meta, dates = fixture()
        p["adj"].iloc[:, :] = 100.
        result, audit = run(p, meta)
        np.testing.assert_allclose(result["rank_residence"], .125)
        self.assertTrue(result["rank_residence_fallback"].all())
        self.assertEqual(audit["event_history"][0]["usable_observations"], [0] * 4)


if __name__ == "__main__":
    unittest.main()
