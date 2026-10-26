"""Only fabricated volume, dates and accounts; never load actual research inputs."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_volume_cluster195_independent as independent


def event_path(events):
    values = [16.] * 63
    for event in events:
        median = sorted(values[-63:])[31]
        values.append(median * (2. if event else .5))
    return np.asarray(values)


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index))
    frame = pd.DataFrame({'A': 100. + (x * 17 % 53),
        'B': 200. + 15. * np.sin(x / 7.) + 5. * np.cos(x / 23.),
        'FUND': 80. + (x // 14 % 3) * 20., 'D': np.full(len(index), 8.)}, index=index)
    names = ['A', 'B', 'FUND', 'D'][:width]
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'volume': frame}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


class ScalarTests(unittest.TestCase):
    def test_hand_counted_covariance_and_frequency(self):
        events = [True, True, False, True] + [False] * 206
        out = independent.measure_oracle(event_path(events))
        self.assertEqual(out['high_volume_events'], events)
        self.assertEqual([out[k] for k in ('event_ones', 'left_ones', 'right_ones', 'adjacent_one_pairs')], [3, 3, 2, 1])
        self.assertEqual(out['covariance_numerator'], 203)
        self.assertEqual(out['candidate'], 203 / 43681)
        self.assertEqual(out['control'], 3 / 210)
        self.assertEqual((out['event_observations'], out['adjacent_pair_observations'], out['rolling_medians_computed']), (210, 209, 210))

    def test_matched_frequency_different_serial_structure(self):
        cluster = independent.measure_oracle(event_path([True] * 105 + [False] * 105))
        alternate = independent.measure_oracle(event_path([True, False] * 105))
        self.assertEqual(cluster['control'], alternate['control'])
        self.assertEqual(cluster['candidate'], (209 * 104 - 105 * 104) / 43681)
        self.assertEqual(alternate['candidate'], -105 * 104 / 43681)
        self.assertGreater(cluster['candidate'], 0)
        self.assertLess(alternate['candidate'], 0)

    def test_odd_median_strictly_previous_and_equality(self):
        values = np.array([1.] * 32 + [3.] * 31 + [1.] * 210)
        a = independent.measure_oracle(values)
        self.assertEqual(a['median_values'][0], 1.)
        self.assertFalse(a['high_volume_events'][0])
        values[63] = float(np.nextafter(1., math.inf))
        b = independent.measure_oracle(values)
        self.assertEqual(b['median_values'][0], 1.)
        self.assertTrue(b['high_volume_events'][0])
        values[63] = 1e300
        self.assertEqual(independent.measure_oracle(values)['median_values'][0], 1.)

    def test_all_zero_or_one_events_are_valid_covariance_zero(self):
        for values, frequency in ((np.ones(273), 0.), (2. ** np.arange(273), 1.), (2. ** -np.arange(273), 0.)):
            r = independent.measure_oracle(values)
            self.assertTrue(r['available'])
            self.assertEqual((r['candidate'], r['control']), (0., frequency))

    def test_support_failures_and_nonnumeric_types(self):
        for bad in (0., -1., float('nan'), float('inf')):
            for place in (0, 62, 63, 271, 272):
                v = np.ones(273); v[place] = bad
                out = independent.measure_oracle(v)
                self.assertEqual(out['reason'], 'invalid_volume')
                self.assertEqual(out['rolling_medians_computed'], 0)
                self.assertEqual(out['high_volume_events'], [])
                self.assertIsNone(out['event_ones'])
        for bad in (True, None, '3', complex(3, 0)):
            v = np.ones(273, object); v[17] = bad
            self.assertEqual(independent.measure_oracle(v)['reason'], 'nonnumeric_volume')
        with self.assertRaises(ValueError): independent.measure_oracle(np.ones(272))

    def test_constant_scale_not_time_varying_split_invariance(self):
        v = event_path(([True] * 7 + [False] * 8) * 14)
        before = v.copy(); original = independent.measure_oracle(v)
        for multiplier in (2. ** -200, 2. ** 200):
            changed = independent.measure_oracle(v * multiplier)
            for key in original:
                if key != 'median_values': self.assertEqual(changed[key], original[key])
        np.testing.assert_array_equal(v, before)
        flat = np.ones(273) * 10; jump = flat.copy(); jump[100:] *= 2
        self.assertEqual(independent.measure_oracle(flat)['event_ones'], 0)
        self.assertEqual(independent.measure_oracle(jump)['event_ones'], 32)
        self.assertGreater(independent.measure_oracle(jump)['candidate'], 0)


class BuildTests(unittest.TestCase):
    def test_source_windows_audit_etf_and_immutability(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        values, audit = independent.independent_features(p, m, c)
        receipt = independent.verify_features(p, m, c, values, audit)
        self.assertEqual((receipt['checked_cells'], receipt['available_cells']), (12, 12))
        self.assertEqual(receipt['rolling_medians_computed'], 2520)
        self.assertFalse(receipt['computes_returns'])
        self.assertTrue(values['available'][:, 2].all())
        for row in audit['event_history']:
            t = c.index(row['date'])
            self.assertEqual([row[k] for k in ('volume_start', 'volume_end', 'event_start', 'event_end')],
                             [c[t - 314], c[t - 42], c[t - 251], c[t - 42]])
        pd.testing.assert_frame_equal(p['volume'], before['volume'])

    def test_prefix_future_poison_and_physical_truncation(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for n in range(4):
            q = copy.deepcopy(p); q['volume'] = q['volume'].astype(object)
            stop = pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n - 1]) - 42])
            q['volume'].loc[q['volume'].index > stop, :] = 'unread future'
            v, a = independent.independent_features(q, m, c, limit=n)
            independent.verify_features(q, m, c, v, a, limit=n)
            for key in values: np.testing.assert_array_equal(v[key], values[key][:n])
            self.assertEqual(a['event_history'], audit['event_history'][:n])
            q['volume'] = q['volume'].loc[:stop]
            vv, aa = independent.independent_features(q, m, c, limit=n)
            for key in v: np.testing.assert_array_equal(vv[key], v[key])
            self.assertEqual(aa, a)
        self.assertEqual(independent.independent_features(p, m, c, 0)[0]['available'].shape, (0, 4))

    def test_missing_support_no_filling_and_failure_priority(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        q = {'volume': p['volume'].drop(pd.Timestamp(c[t - 313]))}
        v, a = independent.independent_features(q, m, c)
        self.assertFalse(v['available'].any())
        self.assertEqual(a['reason_counts'], {'missing_official_session': 4})
        q['volume'] = q['volume'].drop(columns='A')
        _, a = independent.independent_features(q, m, c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'missing_symbol')
        short = [x for x in c if x >= '2016-01-01']
        _, a = independent.independent_features(q, m, short)
        self.assertEqual(a['reason_counts'], {'insufficient_history': 4})
        p['volume'].loc[c[t - 314], 'A'] = 0
        v, a = independent.independent_features(p, m, c)
        self.assertFalse(v['available'][0, 0]); self.assertTrue(v['available'][0, 1:].all())
        self.assertEqual(a['event_observations'], 630)

    def test_monthends_nextsession_and_schema_rejection(self):
        p, m, c = fixture()
        for bad in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError): independent.independent_features(p, m, c, bad)
        q = copy.deepcopy(m); q['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        last = c.index(m['dates'][-1])
        with self.assertRaises(ValueError): independent.independent_features(p, m, c[:last + 1])
        q = dict(m); q.pop('etfs')
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        with self.assertRaises(ValueError): independent.independent_features({'adj': p['volume']}, m, c)
        with self.assertRaises(ValueError): independent.independent_features(p, m, c + [c[-1]])
        q = copy.deepcopy(p); q['volume'] = q['volume'].iloc[::-1]
        with self.assertRaises(ValueError): independent.independent_features(q, m, c)

    def test_hand_midrank_domain_ties_and_degenerate_population(self):
        raw = np.array([[8., 3., 3., np.nan], [4., 4., 4., 4.], [9., np.nan, np.nan, np.nan]])
        expected = np.array([[.5, -.25, -.25, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(independent.centered_ranks(raw, np.isfinite(raw)), expected)
        p, m, c = fixture(); p['volume'].loc[:, :] = 1.
        values, audit = independent.independent_features(p, m, c)
        self.assertTrue(values['available'].all())
        self.assertTrue((values['candidate_rank'] == 0).all())
        self.assertTrue((values['control_rank'] == 0).all())
        independent.verify_features(p, m, c, values, audit)

    def test_corrupt_value_rank_mask_counter_event_and_window_rejected(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for kind in ('raw', 'rank', 'mask', 'counter', 'event', 'date', 'type'):
            v, a = copy.deepcopy(values), copy.deepcopy(audit)
            if kind == 'raw': v['candidate'][0, 0] += 1e-10
            elif kind == 'rank': v['candidate_rank'][0, 0] += .001
            elif kind == 'mask': v['available'][0, 0] = False
            elif kind == 'counter': a['rolling_medians_computed'] -= 1
            elif kind == 'event': a['event_history'][0]['cells'][0]['high_volume_events'][0] ^= True
            elif kind == 'date': a['event_history'][0]['event_start'] = a['event_history'][0]['volume_start']
            else: a['checked_cells'] = float(a['checked_cells'])
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, v, a)

    def test_fixed_delta_initial_neutral_and_corruption_rejection(self):
        p, m, c = fixture(); values, _ = independent.independent_features(p, m, c)
        deltas = independent.independent_deltas(values)
        receipt = independent.verify_deltas(values, deltas)
        self.assertEqual(receipt['max_absolute_delta'], .005)
        self.assertTrue(receipt['centered_cross_sectional_ranking'])
        for arm, raw in independent.RAW.items():
            np.testing.assert_array_equal(deltas[arm][0], np.zeros(4))
            np.testing.assert_array_equal(deltas[arm][1:], .01 * values[raw + '_rank'][1:])
        for where in ((0, 0), (1, 1)):
            q = copy.deepcopy(deltas); q['volume_cluster'][where] += .0001
            with self.assertRaises(ValueError): independent.verify_deltas(values, q)
        with self.assertRaises(ValueError): independent.verify_deltas(values, deltas, dose=.02)
        v = copy.deepcopy(values); v['candidate_rank'][1] *= -1
        with self.assertRaises(ValueError): independent.independent_deltas(v)

    def test_accounting_dependency_pin_and_no_production_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(), independent.FROZEN_HELPER_SHA)
        self.assertIn(independent.FROZEN_HELPER, independent.dependency_files())
        self.assertTrue(callable(independent.accounting().audit_path))
        source = inspect.getsource(independent)
        self.assertNotIn('import monthly_volume_cluster195_kernel', source)
        with patch.object(independent, 'FROZEN_HELPER_SHA', '0' * 64):
            with self.assertRaises(ValueError): independent.accounting()

    def test_synthetic_producer_cross_both_fixtures_and_prefixes(self):
        import monthly_volume_cluster195_kernel as producer
        import test_monthly_volume_cluster195_kernel as producer_tests
        for p, m, c in (fixture(), producer_tests.fixture()):
            for n in range(len(m['dates']) + 1):
                stop = pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n - 1])
                q = {'volume': p['volume'].loc[:stop]}
                values, audit = producer.build_features(q, m, c, limit=n)
                receipt = independent.verify_features(q, m, c, values, audit, limit=n)
                self.assertTrue(receipt['passed'])
                independent.verify_deltas(values, independent.independent_deltas(values))
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        for bad in (0., np.nan, True, 'bad'):
            q = {'volume': p['volume'].astype(object).copy()}; q['volume'].at[pd.Timestamp(c[t - 314]), 'A'] = bad
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
