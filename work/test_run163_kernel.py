"""Synthetic-only terminal run tests; no market files or strategy execution."""
from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd

import run163_kernel as kernel


def from_signs(signs, initial=100.):
    """Independent linear interpolation for ten block endpoints."""
    if len(signs) != 10:
        raise ValueError('Fixture needs ten signs')
    points = initial + np.r_[0., np.cumsum(signs)]
    return np.interp(np.arange(211), np.arange(11)*21, points)


def fixture():
    sessions = pd.bdate_range('2015-01-01', '2017-04-05')
    events = [('2016-12-30', ['A', 'B', 'C', 'SPY'])]
    t = sessions.get_loc(pd.Timestamp(events[0][0]))
    prices = pd.DataFrame(100., index=sessions, columns=events[0][1])
    patterns = ([1,-1,1,-1,1,-1,1,1,1,1], [-1,1,-1,1,-1,1,-1,-1,-1,-1],
                [1,-1,1,-1,1,-1,1,1,1,0], [1]*10)
    for name, signs in zip(events[0][1], patterns):
        prices.iloc[t-252:t-41, prices.columns.get_loc(name)] = from_signs(signs)
    return prices, events, [str(day.date()) for day in sessions]


def run(prices, events, calendar, limit=None, metadata=None):
    return kernel.build_features(prices, metadata, events, calendar, limit)


class Run163Tests(unittest.TestCase):
    def test_formula_independent_ranks_mirror_and_json_audit(self):
        prices, events, calendar = fixture()
        delta, audit = run(prices, events, calendar)
        row = audit['event_history'][0]
        self.assertEqual(row['raw_run'], [.4, -.4, 0., 1.])
        self.assertEqual(row['ranks'], [.25, -.25, 0., .5])
        np.testing.assert_array_equal(delta['delta_positive'], [[.005, -.005, 0., .01]])
        np.testing.assert_array_equal(delta['delta_negative'], -delta['delta_positive'])
        self.assertEqual(row['common_support'], [True]*4)
        self.assertEqual(audit['common_support_cells'], 4)
        self.assertEqual(audit['neutral_cells'], 0)
        self.assertEqual(audit['window_returns'], 210)
        self.assertEqual(audit['skip_sessions'], 42)
        self.assertFalse(audit['trading_eligibility_modified'])
        self.assertFalse(audit['outcomes_used'])
        t = calendar.index(events[0][0])
        self.assertEqual(row['window_start'], calendar[t-252])
        self.assertEqual(row['window_end'], calendar[t-42])
        json.dumps(audit, allow_nan=False)
        self.assertNotIn('block_signs', row)

    def test_every_terminal_length_both_signs_and_zero(self):
        for sign in (-1, 1):
            for length in range(1, 11):
                pattern = [-sign]*(10-length)+[sign]*length
                with self.subTest(sign=sign, length=length):
                    self.assertEqual(kernel._terminal_run(from_signs(pattern)), sign*length/10.)
        for pattern in ([0]*10, [1]*9+[0], [-1]*9+[0]):
            self.assertEqual(kernel._terminal_run(from_signs(pattern)), 0.)

    def test_same_direction_frequency_different_terminal_run(self):
        early = [1,1,1,1,1,-1,-1,-1,-1,-1]
        alternating = [1,-1]*5
        self.assertEqual(sorted(early), sorted(alternating))
        self.assertEqual(kernel._terminal_run(from_signs(early)), -.5)
        self.assertEqual(kernel._terminal_run(from_signs(alternating)), -.1)

    def test_zero_breaks_terminal_run_and_zero_run_remains_ranked(self):
        self.assertEqual(kernel._terminal_run(from_signs([1]*7+[0,1,1])), .2)
        prices, events, calendar = fixture()
        prices[:] = 100.
        delta, audit = run(prices, events, calendar)
        self.assertEqual(audit['event_history'][0]['raw_run'], [0.]*4)
        self.assertEqual(audit['event_history'][0]['ranks'], [.125]*4)
        self.assertEqual(audit['common_support_cells'], 4)
        np.testing.assert_array_equal(delta['delta_positive'], [[.0025]*4])

    def test_block_boundaries_exact_21_sessions_not_20_or_22(self):
        values = from_signs([1]*10)
        # Arbitrary finite-positive interior excursions never define a block.
        values[np.arange(211) % 21 != 0] = 1e100
        self.assertEqual(kernel._terminal_run(values), 1.)
        # The shared endpoint at 189 changes the final block direction.
        values[189] = 1000.
        self.assertEqual(kernel._terminal_run(values), -.1)
        values[210] = values[189]
        self.assertEqual(kernel._terminal_run(values), 0.)

    def test_exact_endpoint_sign_handles_ulp_difference_and_extreme_scales(self):
        values = np.ones(211)
        values[210] = np.nextafter(1., 2.)
        self.assertEqual(kernel._terminal_run(values), .1)
        values[210] = np.nextafter(1., 0.)
        self.assertEqual(kernel._terminal_run(values), -.1)
        values[:] = np.finfo(float).tiny
        values[210] = np.finfo(float).max
        self.assertEqual(kernel._terminal_run(values), .1)

    def test_ties_average_rank_and_missing_excluded_from_denominator(self):
        prices, events, calendar = fixture()
        prices['B'] = prices['A']; prices['C'] = np.nan; prices['SPY'] = np.nan
        delta, audit = run(prices, events, calendar)
        row = audit['event_history'][0]
        self.assertEqual(row['raw_run'], [.4, .4, None, None])
        self.assertEqual(row['ranks'], [.25, .25, None, None])
        np.testing.assert_array_equal(delta['delta_positive'], [[.005,.005,0.,0.]])

    def test_single_value_uses_percentile_rank(self):
        prices, events, calendar = fixture()
        events[0][1][:] = ['A', 'UNKNOWN']
        delta, audit = run(prices, events, calendar)
        self.assertEqual(audit['event_history'][0]['ranks'], [.5, None])
        np.testing.assert_array_equal(delta['delta_positive'], [[.01,0.]])

    def test_price_scaling_and_column_membership_order_invariance(self):
        prices, events, calendar = fixture()
        expected, first = run(prices, events, calendar)
        changed = (prices*[8.,.25,32.,.5]).loc[:, ['SPY','C','A','B']]
        actual, second = run(changed, [(events[0][0], list(reversed(events[0][1])))], calendar)
        for key in expected:
            np.testing.assert_array_equal(actual[key][:, ::-1], expected[key])
        self.assertEqual(second['event_history'][0]['raw_run'][::-1], first['event_history'][0]['raw_run'])

    def test_etfs_and_spy_same_rule_and_no_external_market_dependency(self):
        prices, events, calendar = fixture()
        prices['XLK'] = prices['A']; events[0][1][:] = ['A','XLK','B']
        expected, first = run(prices, events, calendar)
        actual, second = run(prices.drop(columns='SPY'), events, calendar)
        self.assertEqual(first, second)
        self.assertEqual(first['event_history'][0]['raw_run'][:2], [.4,.4])
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_missing_invalid_interior_prices_or_symbols_are_neutral(self):
        prices, events, calendar = fixture()
        t = calendar.index(events[0][0]); day = pd.Timestamp(calendar[t-100])
        for invalid in (np.nan, np.inf, -np.inf, 0., -1.):
            changed = prices.copy(); changed.at[day, 'A'] = invalid
            delta, audit = run(changed, events, calendar)
            row = audit['event_history'][0]
            with self.subTest(invalid=invalid):
                self.assertIsNone(row['raw_run'][0]); self.assertIsNone(row['ranks'][0])
                self.assertFalse(row['common_support'][0])
                for values in delta.values():
                    self.assertEqual(values[0,0], 0.); self.assertFalse(np.signbit(values[0,0]))
        _, audit = run(prices.drop(columns='A'), events, calendar)
        self.assertIsNone(audit['event_history'][0]['raw_run'][0])

    def test_scalar_shape_and_missing_validation(self):
        for size in (0, 210, 212):
            with self.assertRaisesRegex(ValueError, '211'):
                kernel._terminal_run(np.ones(size))
        for position in (0, 1, 20, 21, 100, 209, 210):
            values = from_signs([1]*10); values[position] = np.nan
            self.assertTrue(np.isnan(kernel._terminal_run(values)))

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

    def test_all123_prefixes_match_full122_synthetic_features(self):
        sessions = pd.bdate_range('2013-01-01', '2027-01-31')
        calendar = [str(day.date()) for day in sessions]
        dates = ['2016-08-17'] + [str(day.date()) for day in pd.date_range('2016-08-31', periods=121, freq='BME')]
        x = np.arange(len(sessions))
        prices = pd.DataFrame({'A': 100.+.01*x+np.sin(x*.1),
                               'ETF': 100.+.01*x+np.cos(x*.1)}, index=sessions)
        events = [(day, ['A', 'ETF']) for day in dates]
        full, audit = run(prices, events, calendar)
        for n in range(123):
            part, prefix = run(prices, events, calendar, limit=n)
            self.assertEqual(prefix['event_history'], audit['event_history'][:n])
            for key in full:
                self.assertEqual(part[key].tobytes(), full[key][:n].tobytes())

    def test_irregular_official_calendar_controls_block_and_skip_offsets(self):
        prices, events, calendar = fixture()
        # Removing declared closures shifts official offsets, not dataframe positions.
        closures = [calendar[20], calendar[45], calendar[101], calendar[312]]
        calendar = [day for day in calendar if day not in closures]
        t = calendar.index(events[0][0])
        dates = [pd.Timestamp(day) for day in calendar[t-252:t-41]]
        prices.loc[dates, 'A'] = from_signs([1,-1,1,-1,1,-1,1,1,1,1])
        # These extra dataframe rows are deliberately unusable and must not matter.
        prices = prices.astype(object)
        prices.loc[[pd.Timestamp(day) for day in closures], :] = 'closed-session poison'
        _, audit = run(prices, events, calendar)
        row = audit['event_history'][0]
        self.assertEqual(row['raw_run'][0], .4)
        self.assertEqual(row['window_start'], calendar[t-252])
        self.assertEqual(row['window_end'], calendar[t-42])

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
        self.assertEqual(audit["common_support_cells"], 4)
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
        self.assertTrue(all(value is not None for value in row["raw_run"][:3]))
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
