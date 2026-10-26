"""Only synthetic prices, permutations and independent integer counting."""
from copy import deepcopy
from fractions import Fraction
import json
import unittest
import numpy as np
import pandas as pd
import monthly_session_cluster148_factor as f


def prices(events):
    # Every close-to-close return is exactly zero; only the same-day session
    # decomposition changes. Reordering these pairs preserves the close path.
    close = np.full(211, 100.)
    opening = np.where(np.asarray(events, dtype=bool), 101., 100.)
    return close, opening


def fixture(rows=300):
    dates = pd.bdate_range('2020-01-01', periods=rows)
    symbols = ['S'+str(i) for i in range(32)]+['ETF'+str(i) for i in range(17)]
    close = pd.DataFrame(100., index=dates, columns=symbols)
    opening = pd.DataFrame(100., index=dates, columns=symbols)
    for j, symbol in enumerate(symbols):
        opening.loc[dates[np.arange(rows) % (j+2) == 0], symbol] = 101.
    return dates, close, opening, [dates[252], dates[-1]], [symbols[:], symbols[:]]


class WindowTests(unittest.TestCase):
    def test_integer_counts_match_independent_covariance(self):
        events = np.random.default_rng(1201).integers(0, 2, 210).astype(bool)
        got = f.event_statistics(*prices(events))
        x, y = [int(v) for v in events[:-1]], [int(v) for v in events[1:]]
        joint = sum(a*b for a, b in zip(x, y))
        exact = Fraction(joint, 209)-Fraction(sum(x)*sum(y), 209**2)
        self.assertEqual(got['frequency'], float(Fraction(int(sum(events)), 210)))
        self.assertEqual(got['cluster'], float(exact))
        self.assertEqual(got['left_ones'], sum(x))
        self.assertEqual(got['right_ones'], sum(y))

    def test_covariance_does_not_subtract_the210_frequency_squared(self):
        events = [True]*55+[False]*155
        got = f.event_statistics(*prices(events))
        wrong = got['adjacent_one_pairs']/209-got['frequency']**2
        self.assertNotEqual(got['cluster'], wrong)

    def test_same_marginals_different_internal_order(self):
        clustered = [False]+[True]*104+[False]*105
        alternating = [False]+[True, False]*104+[False]
        one, two = (f.event_statistics(*prices(v)) for v in (clustered, alternating))
        for key in ('event_ones', 'left_ones', 'right_ones', 'frequency'):
            self.assertEqual(one[key], two[key])
        self.assertGreater(one['cluster'], 0)
        self.assertLess(two['cluster'], 0)
        np.testing.assert_array_equal(prices(clustered)[0], prices(alternating)[0])
        np.testing.assert_array_equal(np.sort(prices(clustered)[1]), np.sort(prices(alternating)[1]))

    def test_reversing_sequence_preserves_covariance(self):
        events = np.random.default_rng(17).integers(0, 2, 210)
        a, b = (f.event_statistics(*prices(v)) for v in (events, events[::-1]))
        self.assertEqual(a['cluster'], b['cluster'])
        self.assertEqual(a['frequency'], b['frequency'])

    def test_all_zero_and_all_one_sequences_are_valid(self):
        for flag in (False, True):
            got = f.event_statistics(*prices([flag]*210))
            self.assertTrue(got['available'])
            self.assertEqual(got['cluster'], 0)
            self.assertEqual(got['frequency'], float(flag))

    def test_both_legs_required_and_numerical_zero(self):
        close = np.full(211, 100.)
        opening = np.full(210, 100.)
        opening[:4] = [100., 100.*np.exp(.5e-12), 100.*np.exp(2e-12), 99.]
        got = f.event_statistics(close, opening)
        self.assertEqual(got['event_ones'], 1)
        close[3] = opening[2]  # Positive overnight followed by flat intraday.
        self.assertEqual(f.event_statistics(close, opening)['event_ones'], 0)

    def test_invalid_prices_make_whole_window_unavailable(self):
        for leg in (0, 1):
            for value in (0., -1., np.nan, np.inf, -np.inf):
                arrays = list(prices([True]*210))
                arrays[leg][100] = value
                got = f.event_statistics(*arrays)
                self.assertFalse(got['available'])
                self.assertIsNone(got['frequency'])
                self.assertIsNone(got['cluster'])

    def test_shapes_and_nonnumeric_prices_rejected(self):
        with self.assertRaises(ValueError):
            f.event_statistics(np.ones(210), np.ones(210))
        for bad in (np.ones(211, dtype=bool), np.full(211, '100')):
            with self.assertRaises(ValueError):
                f.event_statistics(bad, np.ones(210))

    def test_nullable_missing_values_are_unavailable_not_events(self):
        for value in (None, pd.NA):
            close, opening = prices([False]*210)
            close = close.astype(object)
            close[100] = value
            self.assertFalse(f.event_statistics(close, opening)['available'])

    def test_unrepresentable_ratios_are_explicitly_unavailable(self):
        got = f.event_statistics(np.full(211, 1e-300), np.full(210, 1e300))
        self.assertFalse(got['available'])
        self.assertEqual(got['reason'], 'unrepresentable_log_ratio')

    def test_common_scale_and_consistent_split_adjustment(self):
        events = np.arange(210) % 3 == 0
        close, opening = prices(events)
        expected = f.event_statistics(close, opening)
        for scale in (1e-100, 1e100, 7.):
            self.assertEqual(f.event_statistics(close*scale, opening*scale), expected)
        # Synthesize a2:1 split in raw quotes, then express all observations in
        # the identical post-split basis before passing them into this API.
        units = np.where(np.arange(211) < 105, 1., .5)
        raw_close, raw_open = close*units, opening*units[1:]
        adjustment = np.where(np.arange(211) < 105, .5, 1.)
        self.assertEqual(f.event_statistics(raw_close*adjustment,
                                           raw_open*adjustment[1:]), expected)

    def test_dividend_adjustment_can_change_overnight_event(self):
        close, opening = prices([False]*210)
        opening[100], close[101] = 99., 98.
        raw = f.event_statistics(close, opening)
        # A stylized adjustment shifts the prior close basis below the open.
        adjusted = close.copy()
        adjusted[:101] *= .98
        adjusted_open = opening.copy()
        adjusted_open[:100] *= .98
        got = f.event_statistics(adjusted, adjusted_open)
        self.assertEqual(got['event_ones'], raw['event_ones']+1)


class RankTests(unittest.TestCase):
    def test_average_ties_lower_is_better_and_missing_zero(self):
        got = f.centered_lower_rank(np.array([1., 1., 3., 4., np.nan]),
                                    np.array([True, True, True, True, False]))
        np.testing.assert_allclose(got, [1/3, 1/3, -1/6, -.5, 0], rtol=0, atol=1e-16)

    def test_singleton_empty_and_all_tied_are_zero(self):
        for values, mask in [([2.], [True]), ([np.nan], [False]),
                              ([2., 2., 2.], [True]*3)]:
            self.assertTrue((f.centered_lower_rank(np.array(values), np.array(mask)) == 0).all())

    def test_rank_permutation_invariant_and_available_centered(self):
        values = np.array([2., 1., 3., 1., 4., 2.])
        mask = np.ones(6, dtype=bool)
        got = f.centered_lower_rank(values, mask)
        perm = np.array([3, 2, 4, 1, 5, 0])
        np.testing.assert_array_equal(f.centered_lower_rank(values[perm], mask), got[perm])
        self.assertAlmostEqual(got.sum(), 0)
        self.assertLessEqual(np.max(np.abs(.01*got)), .005)


class PanelTests(unittest.TestCase):
    def test_shape_etfs_audit_and_no_mutation(self):
        args = fixture()
        before = deepcopy(args)
        got = f.compute_features(*args)
        for key in f.FIELDS:
            self.assertEqual(got[key].shape, (2, 49))
        self.assertTrue(got['available'].all())
        self.assertTrue(got['available'][:, 32:].all())
        self.assertEqual(got['available'].dtype, bool)
        pd.testing.assert_frame_equal(args[1], before[1])
        pd.testing.assert_frame_equal(args[2], before[2])
        json.dumps(got['audit'], allow_nan=False)

    def test_exact_window_and_first_available_position(self):
        dates, close, opening, _, pools = fixture()
        got = f.compute_features(dates, close, opening, dates[[251, 252]], pools)
        self.assertFalse(got['available'][0].any())
        self.assertTrue(got['available'][1].all())
        month = got['audit']['months'][1]
        self.assertEqual(month['close_window_start'], str(dates[0].date()))
        self.assertEqual(month['event_window_start'], str(dates[1].date()))
        self.assertEqual(month['window_end'], str(dates[210].date()))

    def test_recent42_and_future_values_not_inspected(self):
        dates, close, opening, _, pools = fixture()
        signals, pools = [dates[252]], pools[:1]
        expected = f.compute_features(dates, close, opening, signals, pools)
        close.iloc[211:] = np.nan
        opening.iloc[211:] = np.inf
        got = f.compute_features(dates, close, opening, signals, pools)
        for key in f.FIELDS:
            np.testing.assert_array_equal(got[key], expected[key])

    def test_pre_window_values_not_inspected(self):
        dates, close, opening, _, pools = fixture()
        signals, pools = [dates[-1]], pools[:1]
        expected = f.compute_features(dates, close, opening, signals, pools)
        start = len(dates)-1-252
        close.iloc[:start] = np.nan
        opening.iloc[:start+1] = np.nan
        got = f.compute_features(dates, close, opening, signals, pools)
        for key in f.FIELDS:
            np.testing.assert_array_equal(got[key], expected[key])

    def test_missing_row_not_compressed_and_both_ranks_zero(self):
        dates, close, opening, signals, pools = fixture()
        opening = opening.drop(dates[100])
        got = f.compute_features(dates, close, opening, signals[:1], pools[:1])
        self.assertFalse(got['available'].any())
        self.assertTrue(np.isnan(got['frequency']).all())
        self.assertTrue(np.isnan(got['cluster']).all())
        self.assertTrue((got['frequency_rank'] == 0).all())
        self.assertTrue((got['cluster_rank'] == 0).all())

    def test_missing_column_excluded_from_both_rank_universes(self):
        dates, close, opening, signals, pools = fixture()
        opening = opening.drop(columns=pools[0][0])
        got = f.compute_features(dates, close, opening, signals, pools)
        self.assertFalse(got['available'][:, 0].any())
        self.assertTrue((got['frequency_rank'][:, 0] == 0).all())
        self.assertTrue((got['cluster_rank'][:, 0] == 0).all())
        np.testing.assert_allclose(got['frequency_rank'].sum(axis=1), 0, atol=1e-14)

    def test_every_signal_prefix_and_price_prefix_is_identical(self):
        dates, close, opening, _, pools = fixture()
        signals = dates[252::3]
        pools = [pools[0][:] for _ in signals]
        full = f.compute_features(dates, close, opening, signals, pools)
        for n, signal in enumerate(signals, 1):
            position = dates.get_loc(signal)
            prefix = f.compute_features(dates[:position+1], close.iloc[:position+1],
                                        opening.iloc[:position+1], signals[:n], pools[:n])
            for key in f.FIELDS:
                np.testing.assert_array_equal(prefix[key], full[key][:n])

    def test_pool_reordering_preserves_named_results(self):
        dates, close, opening, signals, pools = fixture()
        expected = f.compute_features(dates, close, opening, signals, pools)
        reversed_pools = [names[::-1] for names in pools]
        got = f.compute_features(dates, close, opening, signals, reversed_pools)
        for key in f.FIELDS:
            np.testing.assert_array_equal(got[key][:, ::-1], expected[key])

    def test_invalid_axes_or_pool_rejected(self):
        dates, close, opening, signals, pools = fixture()
        with self.assertRaises(ValueError):
            f.compute_features(dates[::-1], close, opening, signals, pools)
        with self.assertRaises(ValueError):
            f.compute_features(dates, close, opening, [pd.Timestamp('1900-01-01')], pools[:1])
        with self.assertRaises(ValueError):
            f.compute_features(dates, close, opening, signals, [pools[0][:-1]]*2)


if __name__ == '__main__':
    unittest.main()
