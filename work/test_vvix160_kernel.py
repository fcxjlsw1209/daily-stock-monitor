"""Synthetic-only kernel checks; no source files, prices or results are read."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
import unittest

import numpy as np

import vvix160_kernel as k


def fixture():
    day = date(2015, 1, 1)
    calendar = []
    while len(calendar) < 800:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)
    start = calendar.index(k.INITIAL)
    signals = [calendar[i] for i in (start, start + 10, start + 31, start + 53)]
    sources = {
        'VIX': {day: float(i + 10) for i, day in enumerate(calendar)},
        'VVIX': {day: float(2000 - i) for i, day in enumerate(calendar)},
    }
    vol = np.tile(np.linspace(-0.5, 0.5, k.WIDTH), (len(signals), 1))
    return signals, calendar, sources, vol


def window_for(calendar, signal):
    end = calendar.index(signal)
    return calendar[end - 252:end]


def sorted_rank_oracle(values):
    ordered = sorted(values)
    ranks = [i + 1 for i, value in enumerate(ordered) if value == values[-1]]
    rank = Fraction(sum(ranks), len(ranks))
    return float(rank), float(2 * (rank - 1) / 251 - 1)


class GuardedValues(Mapping):
    """Reject value enumeration as well as any out-of-window lookup."""
    def __init__(self, values, allowed):
        self.values = values
        self.allowed = set(allowed)
        self.reads = []

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError('Read outside fixed window: ' + str(key))
        self.reads.append(key)
        return self.values[key]

    def __contains__(self, key):
        if key not in self.allowed:
            raise AssertionError('Membership outside fixed window')
        return key in self.values

    def __iter__(self):
        raise AssertionError('Source value enumeration forbidden')

    def __len__(self):
        raise AssertionError('Source value enumeration forbidden')


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.signals, self.calendar, self.sources, self.vol = fixture()

    def build(self, **kwargs):
        return k.build_states(self.signals, self.calendar, self.sources, **kwargs)

    def test_monotone_extremes_and_exact_formula(self):
        states = self.build()
        np.testing.assert_array_equal(states['states']['VIX'], [0, 1, 1, 1])
        np.testing.assert_array_equal(states['states']['VVIX'], [0, -1, -1, -1])
        delta = k.build_deltas(states, self.vol)
        np.testing.assert_array_equal(delta['delta_vix'][1:], -0.02 * self.vol[1:])
        np.testing.assert_array_equal(delta['delta_vvix'][1:], 0.02 * self.vol[1:])
        self.assertEqual(delta['delta_vix'][1, 0], 0.01)
        self.assertEqual(delta['delta_vix'][1, -1], -0.01)

    def test_interior_midrank_uses_all_ties(self):
        window = window_for(self.calendar, self.signals[1])
        values = [1.] * 120 + [2.] * 19 + [3.] * 112 + [2.]
        self.sources['VIX'].update(zip(window, values))
        result = self.build(limit=2)
        self.assertEqual(result['audit']['monthly'][1]['VIX']['latest_rank'], 130.5)
        self.assertEqual(result['states']['VIX'][1], float(Fraction(8, 251)))

    def test_top_ties_are_not_forced_to_one(self):
        window = window_for(self.calendar, self.signals[1])
        self.sources['VIX'].update(zip(window, [1.] * 249 + [2.] * 3))
        result = self.build(limit=2)
        self.assertEqual(result['audit']['monthly'][1]['VIX']['latest_rank'], 251.)
        self.assertEqual(result['states']['VIX'][1], float(Fraction(249, 251)))

    def test_seeded_midrank_matches_independent_sorted_oracle(self):
        random = np.random.RandomState(160)
        window = window_for(self.calendar, self.signals[1])
        for unused in range(80):
            values = (random.randint(1, 18, 252) / 4.).tolist()
            rank, state = sorted_rank_oracle(values)
            self.sources['VIX'].update(zip(window, values))
            result = self.build(limit=2)
            self.assertEqual(result['audit']['monthly'][1]['VIX']['latest_rank'], rank)
            self.assertEqual(result['states']['VIX'][1], state)

    def test_all_equal_observed_state_is_zero_and_available(self):
        for name in k.SOURCES:
            self.sources[name] = {day: 13.5 for day in self.calendar}
        states = self.build()
        for name in k.SOURCES:
            np.testing.assert_array_equal(states['states'][name], np.zeros(4))
        np.testing.assert_array_equal(states['common_available'], [False, True, True, True])
        self.assertEqual(states['audit']['monthly'][1]['VIX']['latest_rank'], 126.5)
        for delta in k.build_deltas(states, self.vol).values():
            np.testing.assert_array_equal(delta, np.zeros((4, 49)))

    def test_initial_is_neutral_without_reading_either_source(self):
        poison = GuardedValues({}, [])
        result = k.build_states(self.signals, self.calendar,
                                {'VIX': poison, 'VVIX': poison}, limit=1)
        row = result['audit']['monthly'][0]
        self.assertIs(row['initial'], True)
        self.assertIs(row['initial_partial'], True)
        self.assertEqual((row['window_size'], row['window_start'], row['window_end']),
                         (0, None, None))
        for name in k.SOURCES:
            self.assertEqual(row[name], dict(passed=True, observations=0, latest_rank=None, m=0.))
        for delta in k.build_deltas(result, self.vol).values():
            self.assertTrue(np.array_equal(delta, np.zeros((1, 49))))

    def test_required_values_only_no_enumeration(self):
        window = window_for(self.calendar, self.signals[1])
        guarded = {name: GuardedValues(source, window) for name, source in self.sources.items()}
        k.build_states(self.signals, self.calendar, guarded, limit=2)
        for source in guarded.values():
            self.assertEqual(source.reads, window)

    def test_current_and_future_poison_ignored(self):
        expected = self.build(limit=2)
        end = self.calendar.index(self.signals[1])
        for name in k.SOURCES:
            for day in self.calendar[end:]:
                self.sources[name][day] = object()
        actual = self.build(limit=2)
        for name in k.SOURCES:
            np.testing.assert_array_equal(actual['states'][name], expected['states'][name])
        self.assertEqual(actual['audit'], expected['audit'])

    def test_later_bad_value_fails_full_but_not_prior_prefix(self):
        prior = self.calendar[self.calendar.index(self.signals[2]) - 1]
        self.sources['VVIX'][prior] = float('nan')
        self.build(limit=2)
        with self.assertRaisesRegex(ValueError, 'VVIX close must be finite'):
            self.build()

    def test_missing_immediately_prior_session_is_not_backfilled(self):
        prior = self.calendar[self.calendar.index(self.signals[1]) - 1]
        del self.sources['VVIX'][prior]
        with self.assertRaisesRegex(ValueError, 'VVIX missing close at ' + prior):
            self.build(limit=2)

    def test_oldest_of_252_required_but_253rd_is_not_read(self):
        end = self.calendar.index(self.signals[1])
        self.sources['VIX'][self.calendar[end - 253]] = object()
        self.build(limit=2)
        del self.sources['VIX'][self.calendar[end - 252]]
        with self.assertRaisesRegex(ValueError, 'VIX missing close'):
            self.build(limit=2)

    def test_invalid_required_values_fail_for_either_source(self):
        prior = self.calendar[self.calendar.index(self.signals[1]) - 1]
        for name in k.SOURCES:
            for invalid in (None, True, np.bool_(True), '25', 0, -2,
                            float('nan'), float('inf'), -float('inf'), 1 + 2j):
                with self.subTest(source=name, invalid=repr(invalid)):
                    original = self.sources[name][prior]
                    self.sources[name][prior] = invalid
                    with self.assertRaises(ValueError):
                        self.build(limit=2)
                    self.sources[name][prior] = original

    def test_both_sources_required_with_no_unilateral_fallback(self):
        for name in k.SOURCES:
            with self.assertRaisesRegex(ValueError, 'Exactly VIX and VVIX'):
                k.build_states(self.signals, self.calendar, {name: self.sources[name]})
        states = self.build()
        states['common_available'][2] = False
        with self.assertRaisesRegex(ValueError, 'common complete windows'):
            k.build_deltas(states, self.vol)

    def test_monday_window_ends_friday_and_noncalendar_rows_unread(self):
        monday = next(day for day in self.calendar
                      if day > self.signals[1] and date.fromisoformat(day).weekday() == 0)
        self.signals = [k.INITIAL, monday]
        saturday = (date.fromisoformat(monday) - timedelta(days=2)).isoformat()
        for name in k.SOURCES:
            self.sources[name][saturday] = object()
        row = self.build()['audit']['monthly'][1]
        self.assertEqual(row['window_end'],
                         (date.fromisoformat(monday) - timedelta(days=3)).isoformat())
        self.assertGreater((date.fromisoformat(monday) - date.fromisoformat(row['window_start'])).days, 252)
        self.assertEqual(row['window_size'], 252)

    def test_supplied_holiday_exclusion_is_respected(self):
        prior = self.calendar[self.calendar.index(self.signals[1]) - 1]
        self.calendar.remove(prior)
        for name in k.SOURCES:
            self.sources[name][prior] = object()
        row = self.build(limit=2)['audit']['monthly'][1]
        self.assertEqual(row['window_end'], self.calendar[self.calendar.index(self.signals[1]) - 1])
        self.assertNotEqual(row['window_end'], prior)

    def test_invalid_calendar_axes_rejected(self):
        alternatives = [self.calendar + [self.calendar[-1]],
                        list(reversed(self.calendar)),
                        sorted(self.calendar + ['2015-01-17'])]
        for calendar in alternatives:
            with self.assertRaises(ValueError):
                k.build_states(self.signals, calendar, self.sources)
        absent = [day for day in self.calendar if day != self.signals[1]]
        with self.assertRaisesRegex(ValueError, 'Signal missing from calendar'):
            k.build_states(self.signals, absent, self.sources)

    def test_positive_source_unit_scaling_does_not_change_state(self):
        before = self.build()
        for name in k.SOURCES:
            self.sources[name] = {day: value * 1000 for day, value in self.sources[name].items()}
        after = self.build()
        for name in k.SOURCES:
            np.testing.assert_array_equal(before['states'][name], after['states'][name])
        self.assertEqual(before['audit'], after['audit'])

    def test_noncanonical_or_wrong_initial_and_duplicate_signals_rejected(self):
        for signals in (['2016-08-18', self.signals[1]],
                        [k.INITIAL, self.signals[1], self.signals[1]],
                        [k.INITIAL, '2016-9-1']):
            with self.assertRaises(ValueError):
                k.build_states(signals, self.calendar, self.sources)

    def test_insufficient_warmup_cannot_be_padded(self):
        calendar = self.calendar[self.calendar.index(k.INITIAL) - 10:]
        with self.assertRaisesRegex(ValueError, 'Insufficient calendar warmup'):
            k.build_states(self.signals, calendar, self.sources)

    def test_every_prefix_equals_full_build_and_full_delta_slice(self):
        full = self.build()
        full_delta = k.build_deltas(full, self.vol)
        for length in range(len(self.signals) + 1):
            prefix = self.build(limit=length)
            self.assertEqual(prefix['audit']['monthly'], full['audit']['monthly'][:length])
            np.testing.assert_array_equal(prefix['common_available'], full['common_available'][:length])
            for name in k.SOURCES:
                np.testing.assert_array_equal(prefix['states'][name], full['states'][name][:length])
            from_prefix = k.build_deltas(prefix, self.vol)
            from_full = k.build_deltas(full, self.vol, limit=length)
            for field in k.EXTRA:
                np.testing.assert_array_equal(from_prefix[field], full_delta[field][:length])
                np.testing.assert_array_equal(from_full[field], full_delta[field][:length])

    def test_delta_prefix_does_not_validate_future_numeric_values(self):
        states = self.build()
        expected = k.build_deltas(states, self.vol, limit=2)
        self.vol[2:] = np.nan
        states['states']['VVIX'][2:] = np.inf
        states['common_available'][2:] = False
        actual = k.build_deltas(states, self.vol, limit=2)
        for field in k.EXTRA:
            np.testing.assert_array_equal(actual[field], expected[field])
        with self.assertRaises(ValueError):
            k.build_deltas(states, self.vol)

    def test_volatility_not_reranked_recentered_or_eligibility_filtered(self):
        # Nonzero mean and repeated, irregularly spaced ranks deliberately
        # distinguish direct multiplication from recentering or re-ranking.
        self.vol[:] = 0.1
        self.vol[:, 0] = -0.5
        self.vol[:, 1] = 0.5
        self.vol[:, 2] = 0.0
        self.vol[:, 3] = 0.123456789
        result = k.build_deltas(self.build(), self.vol)
        np.testing.assert_array_equal(result['delta_vix'][1:], -0.02 * self.vol[1:])
        self.assertNotEqual(float(result['delta_vix'][1].mean()), 0.)
        self.assertEqual(result['delta_vix'][1, 2], 0.)

    def test_delta_bounds_hold_for_all_rank_states(self):
        states = self.build()
        for state in np.linspace(-1., 1., 257):
            states['states']['VIX'][1:] = state
            states['states']['VVIX'][1:] = -state
            deltas = k.build_deltas(states, self.vol)
            self.assertTrue(all((np.abs(delta) <= 0.01).all() for delta in deltas.values()))

    def test_invalid_consumed_volatility_fails_without_clipping(self):
        states = self.build()
        for invalid in (np.nan, np.inf, -np.inf, 0.5000001, -0.5000001):
            vol = self.vol.copy()
            vol[1, 5] = invalid
            with self.assertRaisesRegex(ValueError, 'Consumed original vol126'):
                k.build_deltas(states, vol)
        for vol in (self.vol[:, :48], self.vol[0], self.vol[:1],
                    self.vol.astype(object), self.vol.astype(bool)):
            with self.assertRaises(ValueError):
                k.build_deltas(states, vol)

    def test_invalid_state_arrays_or_initial_are_rejected(self):
        states = self.build()
        for invalid in (np.nan, np.inf, 1.00001, -1.00001):
            bad = deepcopy(states)
            bad['states']['VIX'][1] = invalid
            with self.assertRaises(ValueError):
                k.build_deltas(bad, self.vol)
        bad = deepcopy(states)
        bad['states']['VIX'][0] = 0.1
        with self.assertRaisesRegex(ValueError, 'Initial state'):
            k.build_deltas(bad, self.vol)
        bad = deepcopy(states)
        bad['states']['VVIX'] = bad['states']['VVIX'][:-1]
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            k.build_deltas(bad, self.vol)

    def test_integer_minimum_cannot_overflow_absolute_bound_check(self):
        states = self.build()
        vol = np.zeros(self.vol.shape, dtype=np.int64)
        vol[0, 0] = np.iinfo(np.int64).min
        with self.assertRaisesRegex(ValueError, 'Consumed original vol126'):
            k.build_deltas(states, vol)
        states['states']['VIX'] = np.zeros(len(self.signals), dtype=np.int64)
        states['states']['VIX'][1] = np.iinfo(np.int64).min
        with self.assertRaisesRegex(ValueError, 'state outside finite'):
            k.build_deltas(states, np.zeros(self.vol.shape))

    def test_invalid_limit_rejected(self):
        states = self.build()
        for limit in (-1, 5, True, 1.0, '2'):
            with self.assertRaises(ValueError):
                self.build(limit=limit)
            with self.assertRaises(ValueError):
                k.build_deltas(states, self.vol, limit=limit)

    def test_inputs_unchanged_and_outputs_do_not_alias_inputs(self):
        original_signals, original_calendar = self.signals[:], self.calendar[:]
        original_sources, original_vol = deepcopy(self.sources), self.vol.copy()
        states = self.build()
        before_states = deepcopy(states)
        deltas = k.build_deltas(states, self.vol)
        self.assertEqual(self.signals, original_signals)
        self.assertEqual(self.calendar, original_calendar)
        self.assertEqual(self.sources, original_sources)
        np.testing.assert_array_equal(self.vol, original_vol)
        for name in k.SOURCES:
            np.testing.assert_array_equal(states['states'][name], before_states['states'][name])
        self.assertEqual(states['audit'], before_states['audit'])
        deltas['delta_vix'][1, 0] = 123
        np.testing.assert_array_equal(self.vol, original_vol)
        np.testing.assert_array_equal(states['states']['VIX'], before_states['states']['VIX'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
