"""Synthetic-only coskewness tests; no market files or strategy execution."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np
import pandas as pd

import coskew162_kernel as kernel


def from_returns(returns, initial=100.):
    return initial * np.r_[1., np.cumprod(1. + np.asarray(returns))]


def fixture():
    sessions = pd.bdate_range("2015-01-01", "2017-04-05")
    length = len(sessions) - 1
    m = np.resize(np.arange(-3., 4.), length) * .004
    q = m * m - np.mean(m * m)
    noise = np.resize([-2., 3., -1., 2., -3., 1., 0.], length) * .0003
    returns = {
        "A": .0005 + .8 * m + 25. * q + noise,
        "B": .0004 + 1.4 * m - 15. * q + .5 * noise,
        "C": .0003 - .3 * m + 3. * q + 5. * noise,
        "SPY": m + .0002,
    }
    prices = pd.DataFrame({name: from_returns(values) for name, values in returns.items()}, index=sessions)
    events = [("2016-12-30", ["A", "B", "C", "SPY"])]
    return prices, events, [str(day.date()) for day in sessions]


def run(prices, events, calendar, limit=None, metadata=None):
    return kernel.build_features(prices, metadata, events, calendar, limit)


def scalar_coskew(prices, symbol, day, calendar):
    """Independent fsum calculation, without using kernel helpers/ranking."""
    t = calendar.index(day)
    dates = [pd.Timestamp(value) for value in calendar[t - 252:t - 41]]
    stock = list(prices.loc[dates, symbol])
    spy = list(prices.loc[dates, "SPY"])
    a = [stock[i + 1] / stock[i] - 1. for i in range(210)]
    b = [spy[i + 1] / spy[i] - 1. for i in range(210)]
    av, bv = math.fsum(a) / 210., math.fsum(b) / 210.
    y, m = [v - av for v in a], [v - bv for v in b]
    m2 = math.fsum(v * v for v in m)
    beta = math.fsum(x * z for x, z in zip(y, m)) / m2
    e = [x - beta * z for x, z in zip(y, m)]
    sd = math.sqrt(math.fsum(v * v for v in e) / 210.)
    if sd <= 1e-12:
        return None
    return (math.fsum(x * z * z for x, z in zip(e, m)) / 210.) / (sd * m2 / 210.)


class Coskew162Tests(unittest.TestCase):
    def test_formula_independent_ranks_mirror_and_json_audit(self):
        prices, events, calendar = fixture()
        delta, audit = run(prices, events, calendar)
        row = audit["event_history"][0]
        expected = [scalar_coskew(prices, name, events[0][0], calendar) for name in events[0][1]]
        np.testing.assert_allclose(row["raw_coskewness"][:3], expected[:3], rtol=0, atol=1e-14)
        self.assertIsNone(row["raw_coskewness"][3])
        valid = expected[:3]
        ranks = [(sum(x < value for x in valid) + (sum(x == value for x in valid) + 1) / 2.) / 3. - .5 for value in valid]
        np.testing.assert_array_equal(row["ranks"][:3], ranks)
        np.testing.assert_array_equal(delta["delta_positive"][0, :3], .02 * np.asarray(ranks))
        np.testing.assert_array_equal(delta["delta_negative"], -delta["delta_positive"])
        self.assertEqual(row["common_support"], [True, True, True, False])
        self.assertEqual(audit["common_support_cells"], 3)
        self.assertEqual(audit["neutral_cells"], 1)
        self.assertEqual(audit["window_returns"], 210)
        self.assertEqual(audit["skip_sessions"], 42)
        self.assertFalse(audit["trading_eligibility_modified"])
        self.assertFalse(audit["outcomes_used"])
        t = calendar.index(events[0][0])
        self.assertEqual(row["window_start"], calendar[t - 252])
        self.assertEqual(row["window_end"], calendar[t - 42])
        json.dumps(audit, allow_nan=False)
        for values in delta.values():
            self.assertTrue(np.isfinite(values).all())
            self.assertLessEqual(np.max(np.abs(values)), .01)

    def test_ties_are_average_rank_and_missing_not_in_denominator(self):
        prices, events, calendar = fixture()
        prices["B"] = prices["A"]
        prices["C"] = np.nan
        delta, audit = run(prices, events, calendar)
        row = audit["event_history"][0]
        self.assertEqual(row["ranks"], [.25, .25, None, None])
        np.testing.assert_array_equal(delta["delta_positive"], [[.005, .005, 0., 0.]])
        self.assertEqual(row["raw_coskewness"][0], row["raw_coskewness"][1])

    def test_single_valid_value_uses_percentile_not_minmax_normalization(self):
        prices, events, calendar = fixture()
        events[0][1][:] = ["A", "UNKNOWN", "SPY"]
        delta, audit = run(prices, events, calendar)
        row = audit["event_history"][0]
        self.assertEqual(row["ranks"], [.5, None, None])
        np.testing.assert_array_equal(delta["delta_positive"], [[.01, 0., 0.]])
        np.testing.assert_array_equal(delta["delta_negative"], [[-.01, 0., 0.]])

    def test_price_scaling_and_column_membership_order_invariance(self):
        prices, events, calendar = fixture()
        expected, audit = run(prices, events, calendar)
        changed = (prices * [8., .25, 32., .5]).loc[:, ["SPY", "C", "A", "B"]]
        actual, second = run(changed, [(events[0][0], list(reversed(events[0][1])))], calendar)
        for key in expected:
            np.testing.assert_array_equal(actual[key][:, ::-1], expected[key])
        np.testing.assert_allclose(second["event_history"][0]["raw_coskewness"][1:][::-1],
                                   audit["event_history"][0]["raw_coskewness"][:3], atol=1e-14)

    def test_ols_removes_added_market_exposure_and_return_intercept(self):
        prices, events, calendar = fixture()
        _, before = run(prices, events, calendar)
        changed = prices.copy()
        original_returns = prices["A"].to_numpy()[1:] / prices["A"].to_numpy()[:-1] - 1.
        market_returns = prices["SPY"].to_numpy()[1:] / prices["SPY"].to_numpy()[:-1] - 1.
        changed["A"] = from_returns(original_returns + 2.3 * market_returns + .002)
        _, after = run(changed, events, calendar)
        self.assertAlmostEqual(before["event_history"][0]["raw_coskewness"][0],
                               after["event_history"][0]["raw_coskewness"][0], places=12)

    def test_all_symbols_including_etfs_have_same_rule_and_spy_is_degenerate(self):
        prices, events, calendar = fixture()
        prices["XLK"] = prices["A"]
        events[0][1][:] = ["A", "XLK", "B", "SPY"]
        _, audit = run(prices, events, calendar)
        row = audit["event_history"][0]
        self.assertEqual(row["raw_coskewness"][0], row["raw_coskewness"][1])
        self.assertEqual(row["ranks"][0], row["ranks"][1])
        self.assertEqual(row["common_support"], [True, True, True, False])

    def test_missing_bad_internal_prices_and_unknown_symbols_are_neutral(self):
        prices, events, calendar = fixture()
        day = pd.Timestamp(calendar[calendar.index(events[0][0]) - 90])
        for invalid in [np.nan, np.inf, -np.inf, 0., -1.]:
            with self.subTest(invalid=invalid):
                changed = prices.copy()
                changed.at[day, "A"] = invalid
                delta, audit = run(changed, events, calendar)
                row = audit["event_history"][0]
                self.assertIsNone(row["raw_coskewness"][0])
                self.assertIsNone(row["ranks"][0])
                self.assertFalse(row["common_support"][0])
                for values in delta.values():
                    self.assertEqual(values[0, 0], 0.)
                    self.assertFalse(np.signbit(values[0, 0]))
        _, audit = run(prices.drop(columns="A"), events, calendar)
        self.assertIsNone(audit["event_history"][0]["raw_coskewness"][0])

    def test_missing_or_invalid_market_neutralizes_all_candidates(self):
        prices, events, calendar = fixture()
        day = pd.Timestamp(calendar[calendar.index(events[0][0]) - 60])
        missing = prices.drop(columns="SPY")
        bad = prices.copy()
        bad.at[day, "SPY"] = np.nan
        for changed in [missing, bad]:
            delta, audit = run(changed, events, calendar)
            self.assertEqual(audit["event_history"][0]["raw_coskewness"], [None] * 4)
            for values in delta.values():
                np.testing.assert_array_equal(values, np.zeros((1, 4)))

    def test_flat_tiny_market_and_linear_or_tiny_residual_are_missing(self):
        prices, events, calendar = fixture()
        for amplitude in [0., 1e-13]:
            changed = prices.copy()
            changed["SPY"] = from_returns(np.resize([-amplitude, amplitude], len(prices) - 1))
            delta, audit = run(changed, events, calendar)
            self.assertEqual(audit["common_support_cells"], 0)
            self.assertEqual(np.count_nonzero(delta["delta_positive"]), 0)
        market = prices["SPY"].to_numpy()[1:] / prices["SPY"].to_numpy()[:-1] - 1.
        changed = prices.copy()
        changed["A"] = from_returns(.001 + 2. * market)
        changed["B"] = from_returns(.001 + 2. * market + np.resize([-1., 1.], len(market)) * 1e-13)
        changed["C"] = 100.
        delta, audit = run(changed, events, calendar)
        self.assertEqual(audit["event_history"][0]["raw_coskewness"], [None] * 4)
        np.testing.assert_array_equal(delta["delta_positive"], np.zeros((1, 4)))

    def test_nonfinite_return_computation_is_missing(self):
        prices, events, calendar = fixture()
        t = calendar.index(events[0][0])
        changed = prices.copy()
        changed.loc[pd.Timestamp(calendar[t - 100]), "A"] = 1e-300
        changed.loc[pd.Timestamp(calendar[t - 99]), "A"] = 1e300
        _, audit = run(changed, events, calendar)
        self.assertIsNone(audit["event_history"][0]["raw_coskewness"][0])
        changed.loc[pd.Timestamp(calendar[t - 100]), "SPY"] = 1e-300
        changed.loc[pd.Timestamp(calendar[t - 99]), "SPY"] = 1e300
        _, audit = run(changed, events, calendar)
        self.assertEqual(audit["common_support_cells"], 0)

    def test_exact_prefixes_and_empty_prefix_do_not_use_training_metadata(self):
        prices, _, calendar = fixture()
        events = [(day, ["A", "B", "C", "SPY"]) for day in ["2016-10-31", "2016-11-30", "2016-12-30"]]
        full, audit = run(prices, events, calendar, metadata=object())
        for n in range(4):
            part, prefix = run(prices, events, calendar, limit=n, metadata={"garbage": "unused"})
            for key in full:
                self.assertEqual(part[key].tobytes(), full[key][:n].tobytes())
            self.assertEqual(prefix["event_history"], audit["event_history"][:n])
            self.assertEqual(prefix["events"], n)
            self.assertEqual(prefix["symbols_per_event"], 4)

    def test_future_skipped_and_preformation_prices_are_not_read(self):
        prices, events, calendar = fixture()
        expected, audit = run(prices, events, calendar)
        t = calendar.index(events[0][0])
        changed = prices.astype(object)
        changed.iloc[:t - 252] = "not a price"
        changed.iloc[t - 41:] = "future poison"
        actual, second = run(changed, events, calendar)
        self.assertEqual(audit, second)
        for key in expected:
            self.assertEqual(expected[key].tobytes(), actual[key].tobytes())
        truncated = prices.loc[:pd.Timestamp(calendar[t - 42])]
        actual, second = run(truncated, events, calendar)
        self.assertEqual(audit, second)
        for key in expected:
            self.assertEqual(expected[key].tobytes(), actual[key].tobytes())

    def test_official_calendar_missing_session_not_replaced_by_extra_row(self):
        prices, events, calendar = fixture()
        t = calendar.index(events[0][0])
        missing = pd.Timestamp(calendar[t - 100])
        changed = prices.drop(index=missing)
        extra = pd.Timestamp("2016-11-05")
        changed.loc[extra] = 100.
        changed = changed.sort_index()
        _, audit = run(changed, events, calendar)
        self.assertEqual(audit["common_support_cells"], 0)
        extra_only = prices.copy()
        extra_only.loc[extra] = np.nan
        expected, first = run(prices, events, calendar)
        actual, second = run(extra_only.sort_index(), events, calendar)
        self.assertEqual(first, second)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_exact_history_boundary_211_prices_and_insufficient_calendar(self):
        prices, events, calendar = fixture()
        t = calendar.index(events[0][0])
        full, audit = run(prices, events, calendar[t - 252:])
        self.assertEqual(audit["common_support_cells"], 3)
        short, audit = run(prices, events, calendar[t - 251:])
        self.assertEqual(audit["common_support_cells"], 0)
        self.assertIsNone(audit["event_history"][0]["window_start"])
        np.testing.assert_array_equal(short["delta_positive"], np.zeros((1, 4)))
        boundary = prices.drop(index=pd.Timestamp(calendar[t - 252]))
        _, audit = run(boundary, events, calendar)
        self.assertEqual(audit["common_support_cells"], 0)
        boundary = prices.drop(index=pd.Timestamp(calendar[t - 42]))
        _, audit = run(boundary, events, calendar)
        self.assertEqual(audit["common_support_cells"], 0)

    def test_initial_forced_neutral_preserves_observed_raw_values(self):
        prices, _, calendar = fixture()
        events = [("2016-08-17", ["A", "B", "C", "SPY"])]
        delta, audit = run(prices, events, calendar)
        row = audit["event_history"][0]
        self.assertTrue(row["initial_forced_neutral"])
        self.assertTrue(all(value is not None for value in row["raw_coskewness"][:3]))
        self.assertEqual(row["common_support"], [False] * 4)
        self.assertEqual(audit["common_support_cells"], 0)
        self.assertEqual(audit["neutral_cells"], 4)
        for values in delta.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))

    def test_no_input_mutation_and_empty_events(self):
        prices, events, calendar = fixture()
        saved_prices, saved_events, saved_calendar = prices.copy(), deepcopy(events), list(calendar)
        run(prices, events, calendar)
        pd.testing.assert_frame_equal(prices, saved_prices)
        self.assertEqual(events, saved_events)
        self.assertEqual(calendar, saved_calendar)
        delta, audit = run(prices, [], calendar)
        self.assertEqual(delta["delta_positive"].shape, (0, 0))
        self.assertEqual(audit["event_history"], [])

    def test_reject_price_axis_schema(self):
        prices, events, calendar = fixture()
        invalid = [prices.iloc[::-1], pd.concat([prices, prices.iloc[-1:]]),
                   prices.rename(columns={"B": "A"}), prices.rename(columns={"B": ""}),
                   prices.set_axis(prices.index.tz_localize("UTC")),
                   prices.set_axis(prices.index + pd.Timedelta(hours=1)),
                   prices.set_axis(list(range(len(prices))))]
        for value in invalid:
            with self.subTest(case=repr(value.index[:1])):
                with self.assertRaises((ValueError, TypeError)):
                    run(value, events, calendar)

    def test_reject_calendar_schema_and_missing_monthend_proof(self):
        prices, events, calendar = fixture()
        invalid = [[], calendar[::-1], calendar + [calendar[-1]],
                   [pd.Timestamp(day) for day in calendar],
                   ["2015-1-1"] + calendar[1:],
                   calendar[:2] + ["2015-01-03"] + calendar[2:]]
        for value in invalid:
            with self.assertRaises((ValueError, TypeError)):
                run(prices, events, value)
        with self.assertRaisesRegex(ValueError, "next session"):
            run(prices, events, calendar[:calendar.index(events[0][0]) + 1])
        with self.assertRaisesRegex(ValueError, "official calendar"):
            run(prices, events, [day for day in calendar if day != events[0][0]])

    def test_reject_event_schema_and_future_invalid_events_even_for_prefix(self):
        prices, events, calendar = fixture()
        names = events[0][1]
        invalid = [[("2016-12-29", names)], [("2016-12-31", names)],
                   [("2016-12-30 01:00:00", names)],
                   [("2016-12-30T00:00:00Z", names)],
                   [("2016-12-30", ["A", "A"])], [("2016-12-30", [])],
                   [("2016-12-30", ["A", ""])], [("2016-12-30", "SPY")],
                   events * 2,
                   events + [("2016-11-30", names)],
                   [("2016-11-30", names), ("2016-12-30", ["A"])],
                   events + [("2017-01-30", names)]]
        for value in invalid:
            with self.subTest(events=value):
                with self.assertRaises((ValueError, TypeError)):
                    run(prices, value, calendar, limit=0)

    def test_reject_invalid_prefix_lengths(self):
        prices, events, calendar = fixture()
        for limit in [-1, 2, True, 1., "1", np.int64(1)]:
            with self.assertRaisesRegex(ValueError, "limit"):
                run(prices, events, calendar, limit)


if __name__ == "__main__":
    unittest.main()
