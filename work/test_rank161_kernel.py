"""Synthetic-only tests; no real factors, orders, returns or registration."""
from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd

import rank161_kernel as kernel


GAINS = [
    [.4, .3, .2, .1], [.1, .4, .2, .3], [.3, .2, .1, .4],
    [.2, .3, .4, .1], [.4, .1, .3, .2], [.3, .2, .4, .1],
    [.2, .1, .3, .4], [.1, .3, .4, .2],
]


def fixture(count=6):
    sessions = pd.bdate_range("2014-01-01", "2018-03-05")
    prices = pd.DataFrame(100., index=sessions, columns=["A", "B", "C", "D"])
    ends = sessions.to_series().groupby(sessions.to_period("M")).last().tolist()
    days = ends[20:20 + count]
    for day, gains in zip(days, GAINS):
        t = sessions.get_loc(day)
        prices.iloc[t - 42] = 100. * (1. + np.asarray(gains))
    meta = {"dates": [str(day.date()) for day in days],
            "symbols": [list(prices.columns) for _ in days]}
    calendar = [str(day.date()) for day in sessions]
    return prices, meta, days, calendar


def run(prices, meta, calendar, days=None, symbols=None, limit=None):
    if days is None:
        days = [pd.Timestamp(meta["dates"][-1])]
    if symbols is None:
        symbols = [list(prices.columns) for _ in days]
    return kernel.build_features(prices, meta, list(zip(days, symbols)), calendar, limit)


def independent_ranks(prices, day, symbols, calendar):
    """Scalar tie ranks independent of pandas rank and kernel helpers."""
    position = calendar.index(str(day.date()))
    momentum = {}
    for symbol in symbols:
        if symbol not in prices.columns or position < 252:
            continue
        start, end = [pd.Timestamp(calendar[position - offset]) for offset in [252, 42]]
        if start not in prices.index or end not in prices.index:
            continue
        a, b = float(prices.at[start, symbol]), float(prices.at[end, symbol])
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            value = b / a - 1.
            if np.isfinite(value):
                momentum[symbol] = value
    result = {}
    for symbol, value in momentum.items():
        less = sum(other < value for other in momentum.values())
        equal = sum(other == value for other in momentum.values())
        result[symbol] = (less + (equal + 1.) / 2.) / len(momentum) - .5
    return result


class Rank161Tests(unittest.TestCase):
    def test_exact_six_rank_formulas_and_json_audit(self):
        prices, meta, days, calendar = fixture()
        actual, audit = run(prices, meta, calendar)
        ranks = np.asarray([[independent_ranks(prices, day, names, calendar)[symbol]
                             for symbol in prices.columns]
                            for day, names in zip(days, meta["symbols"])])
        expected_control = .02 * np.clip(np.mean(ranks, axis=0) - np.std(ranks, axis=0, ddof=0), -.5, .5)
        expected_trend = np.asarray([.01 * sum((k - 2.5) * ranks[k, j] for k in range(6)) / 4.5
                                     for j in range(4)])
        np.testing.assert_allclose(actual["delta_residence"][0], expected_control, atol=1e-15)
        np.testing.assert_allclose(actual["delta_rank_trend"][0], expected_trend, atol=1e-15)
        row = audit["event_history"][0]
        self.assertEqual(row["usable_observations"], [6] * 4)
        self.assertEqual(row["common_support"], [True] * 4)
        np.testing.assert_array_equal(row["rank_trajectories"], ranks)
        self.assertEqual(audit["common_support_cells"], 4)
        self.assertEqual(audit["neutral_cells"], 0)
        self.assertFalse(audit["trading_eligibility_modified"])
        self.assertFalse(audit["outcomes_used"])
        json.dumps(audit, allow_nan=False)
        for values in actual.values():
            self.assertTrue(np.isfinite(values).all())
            self.assertLessEqual(float(np.abs(values).max()), .01)

    def test_same_current_and_rank_multiset_different_order_changes_only_trend(self):
        prices, meta, days, calendar = fixture()
        expected, first = run(prices, meta, calendar)
        changed = prices.copy()
        for day, gains in zip(days, list(reversed(GAINS[:5])) + GAINS[5:6]):
            changed.iloc[changed.index.get_loc(day) - 42] = 100. * (1. + np.asarray(gains))
        actual, second = run(changed, meta, calendar)
        np.testing.assert_array_equal(first["event_history"][0]["rank_trajectories"][-1],
                                      second["event_history"][0]["rank_trajectories"][-1])
        np.testing.assert_allclose(expected["delta_residence"], actual["delta_residence"], atol=1e-15)
        self.assertFalse(np.allclose(expected["delta_rank_trend"], actual["delta_rank_trend"]))

    def test_extreme_rank_paths_respect_trend_bound_and_residence_clip(self):
        base, _, days, calendar = fixture()
        symbols = ["S" + str(index) for index in range(40)]
        prices = pd.DataFrame(100., index=base.index, columns=symbols)
        meta = {"dates": [str(day.date()) for day in days], "symbols": [symbols[:] for _ in days]}
        for positive_direction in [True, False]:
            for index, day in enumerate(days):
                values = np.linspace(.1, .49, 40)
                top = (index >= 3) == positive_direction
                values[0] = .6 if top else .01
                prices.iloc[prices.index.get_loc(day) - 42] = 100. * (1. + values)
            actual, _ = run(prices, meta, calendar)
            expected = .01 * (1. - 1. / 40) * (1. if positive_direction else -1.)
            self.assertAlmostEqual(actual["delta_rank_trend"][0, 0], expected, places=15)
            for values in actual.values():
                self.assertLessEqual(float(np.abs(values).max()), .01)
        for index, day in enumerate(days):
            values = np.linspace(.1, .49, 40)
            values[0] = .6 if index == 5 else .01
            prices.iloc[prices.index.get_loc(day) - 42] = 100. * (1. + values)
        actual, _ = run(prices, meta, calendar)
        self.assertEqual(actual["delta_residence"][0, 0], -.01)

    def test_constant_rank_paths_have_exact_zero_trend(self):
        base, _, days, calendar = fixture()
        # A width with non-binary percentile fractions catches dot-product
        # residuals that ordinary four-member rank fixtures cannot expose.
        symbols = ["S" + str(index) for index in range(49)]
        prices = pd.DataFrame(100., index=base.index, columns=symbols)
        for day in days:
            prices.iloc[prices.index.get_loc(day) - 42] = 100. + np.arange(1, 50)
        meta = {"dates": [str(day.date()) for day in days], "symbols": [symbols[:] for _ in days]}
        actual, audit = run(prices, meta, calendar)
        self.assertEqual(audit["event_history"][0]["common_support"], [True] * 49)
        np.testing.assert_array_equal(actual["delta_rank_trend"], np.zeros((1, 49)))

    def test_future_prices_memberships_and_event_price_prefixes_do_not_change_past(self):
        prices, meta, days, calendar = fixture(8)
        queries = days[5:]
        expected, full_audit = run(prices, meta, calendar, queries)
        prefix, prefix_audit = run(prices, meta, calendar, queries, limit=1)
        truncated, _ = run(prices.loc[:queries[0]], meta, calendar, queries[:1])
        changed = prices.copy()
        changed.loc[changed.index > queries[0]] = np.nan
        future_meta = deepcopy(meta)
        for index in [6, 7]:
            future_meta["symbols"][index] = ["B", "C", "D", "FUTURE"]
        future, future_audit = run(changed, future_meta, calendar, queries,
                                   [meta["symbols"][5]] + future_meta["symbols"][6:], limit=1)
        for key in expected:
            np.testing.assert_array_equal(expected[key][:1], prefix[key])
            np.testing.assert_array_equal(expected[key][:1], truncated[key])
            np.testing.assert_array_equal(expected[key][:1], future[key])
        self.assertEqual(full_audit["event_history"][:1], prefix_audit["event_history"])
        self.assertEqual(prefix_audit, future_audit)
        self.assertTrue(all(day <= str(queries[0].date()) for day in prefix_audit["snapshot_audits"]))

    def test_event_union_supplies_history_after_training_ends(self):
        prices, meta, days, calendar = fixture()
        expected, _ = run(prices, meta, calendar, days)
        short_meta = {"dates": meta["dates"][:2], "symbols": meta["symbols"][:2]}
        actual, audit = run(prices, short_meta, calendar, days)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        self.assertEqual(audit["event_history"][-1]["common_support"], [True] * 4)

    def test_missing_month_is_not_filled_from_older_history(self):
        prices, meta, days, calendar = fixture()
        ends = prices.index.to_series().groupby(prices.index.to_period("M")).last()
        old = ends.loc[days[0].to_period("M") - 1]
        reduced = {"dates": [str(old.date())] + meta["dates"][:2] + meta["dates"][3:],
                   "symbols": [list(prices.columns) for _ in range(6)]}
        actual, audit = run(prices, reduced, calendar)
        for values in actual.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))
        row = audit["event_history"][0]
        self.assertEqual(row["usable_observations"], [5] * 4)
        self.assertEqual(row["common_support"], [False] * 4)
        self.assertIsNone(row["snapshots"][2])
        self.assertEqual(row["missing_snapshot_months"], [str(days[2].to_period("M"))])
        self.assertEqual(row["rank_trajectories"][2], [None] * 4)
        self.assertNotIn(str(old.date()), audit["snapshot_audits"])
        json.dumps(audit, allow_nan=False)

    def test_five_usable_observations_force_both_deltas_to_zero(self):
        prices, meta, days, calendar = fixture()
        meta["symbols"][0] = ["B", "C", "D"]
        actual, audit = run(prices, meta, calendar)
        row = audit["event_history"][0]
        self.assertEqual(row["usable_observations"], [5, 6, 6, 6])
        self.assertEqual(row["common_support"], [False, True, True, True])
        for values in actual.values():
            self.assertEqual(values[0, 0], 0.)
        self.assertIsNone(row["raw_rank_trajectories"][0][0])

    def test_old_pool_members_and_gate_are_not_replaced_by_current_pool(self):
        prices, meta, days, calendar = fixture()
        # D ranked in all old pools but is absent from the current pool. Its
        # old rank still changes B's percentile. A was never in one old pool.
        meta["symbols"][0] = ["B", "C", "D"]
        meta["symbols"][-1] = ["A", "B", "C"]
        actual, audit = run(prices, meta, calendar, symbols=[meta["symbols"][-1]])
        row = audit["event_history"][0]
        self.assertEqual(row["usable_observations"], [5, 6, 6])
        self.assertIsNone(row["rank_trajectories"][0][0])
        for index, day in enumerate(days):
            expected = independent_ranks(prices, day, meta["symbols"][index], calendar)
            self.assertEqual(row["raw_rank_trajectories"][index][1], expected["B"])
        self.assertEqual(audit["snapshot_audits"][meta["dates"][0]]["symbols"], ["B", "C", "D"])
        self.assertEqual(actual["delta_residence"][0, 0], 0.)

    def test_endpoint_valid_member_is_ranked_even_when_not_individually_usable(self):
        prices, meta, days, calendar = fixture()
        original, audit = run(prices, meta, calendar)
        t = prices.index.get_loc(days[0])
        prices.iloc[t - 294, 0] = 0.
        actual, changed = run(prices, meta, calendar)
        first = changed["snapshot_audits"][meta["dates"][0]]
        self.assertEqual(first["endpoint_valid_count"], 4)
        self.assertFalse(first["usable"][0])
        self.assertEqual(first["centered_percentile_ranks"],
                         audit["snapshot_audits"][meta["dates"][0]]["centered_percentile_ranks"])
        for key in actual:
            self.assertEqual(actual[key][0, 0], 0.)
            np.testing.assert_array_equal(actual[key][0, 1:], original[key][0, 1:])

    def test_zero_negative_nan_inf_nonendpoint_prices_fail_exact_warmup(self):
        prices, meta, days, calendar = fixture()
        t = prices.index.get_loc(days[0])
        for invalid in [0., -1., np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid):
                changed = prices.copy()
                changed.iloc[t - 294, 0] = invalid
                actual, audit = run(changed, meta, calendar)
                self.assertEqual(audit["event_history"][0]["usable_observations"], [5, 6, 6, 6])
                for values in actual.values():
                    self.assertEqual(values[0, 0], 0.)
                json.dumps(audit, allow_nan=False)

    def test_nonpositive_momentum_is_ranked_but_unusable(self):
        prices, meta, days, calendar = fixture()
        for value in [100., 90.]:
            with self.subTest(endpoint=value):
                changed = prices.copy()
                changed.iloc[prices.index.get_loc(days[0]) - 42, 0] = value
                actual, audit = run(changed, meta, calendar)
                row = audit["event_history"][0]
                self.assertEqual(row["usable_observations"][0], 5)
                self.assertIsNone(row["rank_trajectories"][0][0])
                self.assertEqual(row["raw_rank_trajectories"][0][0], -.25)
                first = audit["snapshot_audits"][meta["dates"][0]]
                self.assertEqual(first["endpoint_valid_count"], 4)
                self.assertFalse(first["positive_momentum"][0])
                for values in actual.values():
                    self.assertEqual(values[0, 0], 0.)

    def test_invalid_endpoints_are_excluded_from_percentile_denominator(self):
        prices, meta, days, calendar = fixture()
        for invalid in [0., -1., np.nan, np.inf]:
            with self.subTest(invalid=invalid):
                changed = prices.copy()
                changed.iloc[prices.index.get_loc(days[0]) - 42, 0] = invalid
                _, audit = run(changed, meta, calendar)
                first = audit["snapshot_audits"][meta["dates"][0]]
                self.assertEqual(first["endpoint_valid_count"], 3)
                self.assertIsNone(first["centered_percentile_ranks"][0])
                expected = independent_ranks(changed, days[0], meta["symbols"][0], calendar)
                self.assertEqual(first["centered_percentile_ranks"][1:], [expected[x] for x in ["B", "C", "D"]])

    def test_calendar_session_gap_cannot_be_filled_by_an_extra_weekend_row(self):
        prices, meta, days, calendar = fixture()
        missing = prices.index[prices.index.get_loc(days[0]) - 294]
        changed = prices.drop(index=missing)
        weekend = missing + pd.Timedelta(days=(5 - missing.weekday()) % 7)
        changed.loc[weekend] = 100.
        changed = changed.sort_index()
        actual, audit = run(changed, meta, calendar)
        self.assertEqual(audit["event_history"][0]["usable_observations"], [5] * 4)
        self.assertEqual(audit["snapshot_audits"][meta["dates"][0]]["missing_price_sessions"], [str(missing.date())])
        for values in actual.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))

    def test_extra_noncalendar_prices_do_not_shift_session_offsets(self):
        prices, meta, days, calendar = fixture()
        expected, _ = run(prices, meta, calendar)
        changed = prices.copy()
        weekend = days[0] - pd.Timedelta(days=(days[0].weekday() - 5) % 7)
        changed.loc[weekend] = [np.nan, -1., np.inf, 1e100]
        changed = changed.sort_index()
        actual, _ = run(changed, meta, calendar)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_missing_endpoint_session_never_shifts_offsets(self):
        prices, meta, days, calendar = fixture()
        missing = prices.index[prices.index.get_loc(days[0]) - 252]
        _, audit = run(prices.drop(index=missing), meta, calendar)
        first = audit["snapshot_audits"][meta["dates"][0]]
        self.assertEqual(first["endpoint_valid_count"], 0)
        self.assertEqual(first["centered_percentile_ranks"], [None] * 4)

    def test_295_inclusive_boundary_and_insufficient_calendar_history(self):
        prices, meta, days, calendar = fixture()
        t = prices.index.get_loc(days[0])
        for length, count in [(294, 5), (295, 6)]:
            with self.subTest(length=length):
                shortened_calendar = calendar[t - length + 1:]
                _, audit = run(prices, meta, shortened_calendar)
                self.assertEqual(audit["event_history"][0]["usable_observations"], [count] * 4)
        baseline, _ = run(prices, meta, calendar)
        prices.iloc[t - 295] = np.nan
        changed, _ = run(prices, meta, calendar)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], changed[key])

    def test_missing_current_price_row_is_neutral_without_changing_shape(self):
        prices, meta, days, calendar = fixture()
        actual, audit = run(prices.drop(index=days[-1]), meta, calendar)
        self.assertEqual(audit["event_history"][0]["usable_observations"], [5] * 4)
        for values in actual.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))

    def test_last_42_sessions_do_not_change_ranks_when_prices_remain_valid(self):
        prices, meta, days, calendar = fixture()
        expected, _ = run(prices, meta, calendar)
        t = prices.index.get_loc(days[-1])
        changed = prices.copy()
        changed.iloc[t - 41:t + 1] *= [100., .1, 3., 1e6]
        actual, _ = run(changed, meta, calendar)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        changed.iloc[t - 42, 0] *= 2.
        endpoint, _ = run(changed, meta, calendar)
        self.assertFalse(np.array_equal(actual["delta_rank_trend"], endpoint["delta_rank_trend"]))

    def test_initial_is_forced_neutral_even_with_six_ranks_and_never_later_history(self):
        prices, _, _, calendar = fixture()
        prices.iloc[:, :] = 100. * np.exp(np.arange(len(prices))[:, None] * np.asarray([.001, .002, .003, .004]))
        dates = ["2016-03-31", "2016-04-29", "2016-05-31", "2016-06-30", "2016-07-29",
                 "2016-08-17", "2016-08-31", "2016-09-30"]
        meta = {"dates": dates, "symbols": [list(prices.columns) for _ in dates]}
        actual, audit = run(prices, meta, calendar, [pd.Timestamp(d) for d in dates[-3:]])
        first = audit["event_history"][0]
        self.assertEqual(first["usable_observations"], [6] * 4)
        self.assertEqual(first["common_support"], [False] * 4)
        self.assertEqual(first["snapshots"][-1], "2016-08-17")
        for values in actual.values():
            np.testing.assert_array_equal(values[0], np.zeros(4))
        for row in audit["event_history"][1:]:
            self.assertNotIn("2016-08-17", row["snapshots"])
            self.assertIn("2016-08-31", row["snapshots"])
            self.assertEqual(row["common_support"], [True] * 4)

    def test_initial_does_not_substitute_for_missing_august_month_end(self):
        prices, _, _, calendar = fixture()
        prices.iloc[:, :] = 100. * np.exp(np.arange(len(prices))[:, None] * np.asarray([.001, .002, .003, .004]))
        dates = ["2016-04-29", "2016-05-31", "2016-06-30", "2016-07-29", "2016-08-17", "2016-09-30"]
        meta = {"dates": dates, "symbols": [list(prices.columns) for _ in dates]}
        actual, audit = run(prices, meta, calendar)
        self.assertEqual(audit["event_history"][0]["snapshots"][-2:], [None, "2016-09-30"])
        self.assertEqual(audit["event_history"][0]["usable_observations"], [5] * 4)
        for values in actual.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))

    def test_ties_scale_column_and_same_date_membership_order_invariance(self):
        prices, meta, days, calendar = fixture()
        prices["B"] = prices["A"]
        saved = prices.copy(deep=True)
        saved_meta = deepcopy(meta)
        expected, audit = run(prices, meta, calendar)
        pd.testing.assert_frame_equal(prices, saved)
        self.assertEqual(meta, saved_meta)
        first = audit["snapshot_audits"][meta["dates"][0]]["centered_percentile_ranks"]
        self.assertEqual(first[:2], [.375, .375])
        changed = (prices * [32., .25, 128., 8.]).loc[:, ["C", "A", "D", "B"]]
        actual, _ = run(changed, meta, calendar, symbols=[["D", "C", "B", "A"]])
        for key in expected:
            np.testing.assert_allclose(expected[key], actual[key][:, ::-1], atol=1e-15)

    def test_flat_momentum_and_unknown_symbols_are_neutral_and_not_zero_rank_filled(self):
        prices, meta, days, calendar = fixture()
        prices.iloc[:, :] = 100.
        actual, audit = run(prices, meta, calendar)
        self.assertEqual(audit["event_history"][0]["usable_observations"], [0] * 4)
        self.assertEqual(audit["event_history"][0]["raw_rank_trajectories"][0], [.125] * 4)
        self.assertEqual(audit["event_history"][0]["rank_trajectories"][0], [None] * 4)
        for values in actual.values():
            np.testing.assert_array_equal(values, np.zeros((1, 4)))
        meta["symbols"][-1][-1] = "UNKNOWN"
        _, audit = run(prices, meta, calendar, symbols=[meta["symbols"][-1]])
        self.assertEqual(audit["event_history"][0]["usable_observations"][-1], 0)
        self.assertTrue(all(row[-1] is None for row in audit["event_history"][0]["raw_rank_trajectories"]))

    def test_reject_conflicting_dates_duplicate_symbols_and_mixed_width(self):
        prices, meta, days, calendar = fixture()
        with self.assertRaisesRegex(ValueError, "Conflicting same-date"):
            run(prices, meta, calendar, symbols=[["A", "B", "C", "OTHER"]])
        with self.assertRaisesRegex(ValueError, "Duplicate symbols"):
            run(prices, meta, calendar, symbols=[["A", "A", "B", "C"]])
        with self.assertRaisesRegex(ValueError, "same number"):
            run(prices, meta, calendar, days[-2:], [["A"], ["A", "B"]])
        bad = deepcopy(meta)
        bad["dates"] = bad["dates"][:-1]
        with self.assertRaisesRegex(ValueError, "different lengths"):
            run(prices, bad, calendar)

    def test_same_date_equal_sets_are_allowed_and_future_conflicts_fail_schema(self):
        prices, meta, days, calendar = fixture(8)
        expected, _ = run(prices, meta, calendar, days[5:])
        duplicate = deepcopy(meta)
        duplicate["dates"].insert(1, meta["dates"][0])
        duplicate["symbols"].insert(1, list(reversed(meta["symbols"][0])))
        actual, _ = run(prices, duplicate, calendar, days[5:])
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        conflicting_queries = [list(prices.columns), list(prices.columns), ["A", "B", "C", "FUTURE"]]
        with self.assertRaisesRegex(ValueError, "Conflicting same-date"):
            run(prices, meta, calendar, days[5:], conflicting_queries, limit=1)

    def test_reject_off_calendar_nonmonthend_and_multiple_monthly_snapshots(self):
        prices, meta, days, calendar = fixture()
        for day, message in [(pd.Timestamp("2016-01-02"), "official calendar session"),
                             (pd.Timestamp("2016-01-04"), "official month end")]:
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, message):
                run(prices, {"dates": [], "symbols": []}, calendar, [day])
        bad = deepcopy(meta)
        bad["dates"].insert(1, str((days[0] - pd.offsets.BDay()).date()))
        bad["symbols"].insert(1, list(prices.columns))
        paired = sorted(zip(bad["dates"], bad["symbols"]))
        bad = {"dates": [x[0] for x in paired], "symbols": [x[1] for x in paired]}
        with self.assertRaisesRegex(ValueError, "Multiple noninitial"):
            run(prices, bad, calendar)
        truncated = calendar[:calendar.index(meta["dates"][-1]) + 1]
        with self.assertRaisesRegex(ValueError, "needs the next session"):
            run(prices, meta, truncated)

    def test_reject_price_and_calendar_duplicate_or_disordered_axes(self):
        prices, meta, days, calendar = fixture()
        bad_prices = [prices.iloc[::-1], pd.concat([prices.iloc[:1], prices]),
                      prices.rename(columns={"D": "A"})]
        for changed in bad_prices:
            with self.subTest(kind=str(changed.shape)), self.assertRaisesRegex(ValueError, "unique"):
                run(changed, meta, calendar)
        for changed in [calendar[::-1], calendar[:1] + calendar, calendar[:1] + ["2014-01-04"] + calendar[1:]]:
            with self.subTest(calendar=changed[:3]), self.assertRaises(ValueError):
                run(prices, meta, changed)
        with self.assertRaisesRegex(TypeError, "DatetimeIndex"):
            run(prices.reset_index(drop=True), meta, calendar)

    def test_reject_invalid_event_chronology_date_and_symbol_types(self):
        prices, meta, days, calendar = fixture()
        with self.assertRaisesRegex(ValueError, "chronological"):
            run(prices, meta, calendar, list(reversed(days)))
        with self.assertRaisesRegex(ValueError, "chronological"):
            run(prices, {"dates": meta["dates"][::-1], "symbols": meta["symbols"]}, calendar)
        for day in [pd.NaT, days[-1] + pd.Timedelta(hours=1), days[-1].tz_localize("UTC")]:
            with self.subTest(day=day), self.assertRaises(ValueError):
                run(prices, meta, calendar, [day])
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            run(prices, meta, calendar, symbols=[["A", "B", "C", None]])

    def test_limit_zero_empty_events_and_invalid_limits(self):
        prices, meta, days, calendar = fixture()
        for count in [-1, 2, True, 1.0]:
            with self.subTest(limit=count), self.assertRaisesRegex(ValueError, "limit"):
                run(prices, meta, calendar, limit=count)
        actual, audit = run(prices, meta, calendar, limit=0)
        for values in actual.values():
            self.assertEqual(values.shape, (0, 4))
        self.assertEqual(audit["snapshots_computed"], 0)
        actual, audit = run(prices, meta, calendar, days=[])
        for values in actual.values():
            self.assertEqual(values.shape, (0, 0))
        self.assertEqual(audit["event_history"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
