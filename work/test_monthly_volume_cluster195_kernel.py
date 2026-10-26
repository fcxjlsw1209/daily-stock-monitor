"""Synthetic-only tests; no registered panels, features, orders or returns."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

import monthly_volume_cluster195_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    volume = pd.DataFrame({'A': 100.+x, 'B': 100.+20.*np.sin(x*.39),
                           'FUND': 200.+30.*np.cos(x*.17)}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
                symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'volume': volume}, meta, dates.strftime('%Y-%m-%d').tolist()


class MeasureTests(unittest.TestCase):
    def test_constant_and_increasing_sequences(self):
        constant = kernel.measure(np.ones(273))
        growing = kernel.measure(np.arange(1., 274.))
        self.assertTrue(constant['available'])
        self.assertEqual((constant['candidate'], constant['control']), (0., 0.))
        self.assertEqual((growing['candidate'], growing['control']), (0., 1.))
        self.assertEqual(growing['rolling_medians_computed'], 210)
        self.assertEqual(growing['adjacent_one_pairs'], 209)

    def test_alternating_exact_integer_marginals(self):
        detail = kernel.measure(1.+np.arange(273)%2)
        self.assertEqual(detail['event_ones'], 105)
        self.assertEqual((detail['left_ones'], detail['right_ones']), (105, 104))
        self.assertEqual(detail['adjacent_one_pairs'], 0)
        self.assertEqual(detail['covariance_numerator'], -105*104)
        self.assertEqual(detail['candidate'], -105*104/(209*209))
        self.assertEqual(detail['control'], .5)

    def test_strict_previous_median_and_ties(self):
        v = np.ones(273)
        v[:63] = [1.]*32+[3.]*31
        v[63] = 2.
        detail = kernel.measure(v)
        self.assertEqual(detail['median_values'][0], 1.)
        self.assertTrue(detail['high_volume_events'][0])
        v[63] = 1.
        self.assertFalse(kernel.measure(v)['high_volume_events'][0])

    def test_every_rolling_window_and_overlap_boundaries(self):
        v = np.random.default_rng(9).integers(1, 20, 273).astype(float)
        result = kernel.measure(v)
        medians = [float(sorted(v[i-63:i])[31]) for i in range(63, 273)]
        events = [v[i] > medians[i-63] for i in range(63, 273)]
        self.assertEqual(result['median_values'], medians)
        self.assertEqual(result['high_volume_events'], events)
        self.assertEqual(result['median_values'][-1], sorted(v[209:272])[31])
        v[-1] = 1e300
        changed = kernel.measure(v)
        self.assertEqual(result['median_values'], changed['median_values'])
        self.assertEqual(result['high_volume_events'][:-1], changed['high_volume_events'][:-1])
        self.assertTrue(changed['high_volume_events'][-1])

    def test_power_two_scale_invariance_extreme_finite(self):
        v = np.random.default_rng(3).integers(1, 40, 273).astype(float)
        base = kernel.measure(v)
        for scale in (2.**-900, 2.**900):
            other = kernel.measure(v*scale)
            for key in ('high_volume_events', 'event_ones', 'covariance_numerator', 'candidate', 'control'):
                self.assertEqual(base[key], other[key])

    def test_invalid_numeric_and_nonnumeric_are_common_unavailable(self):
        for item, reason in [(0., 'invalid_volume'), (-1., 'invalid_volume'),
                (np.nan, 'invalid_volume'), (np.inf, 'invalid_volume'),
                (True, 'nonnumeric_volume'), (None, 'nonnumeric_volume'),
                ('2', 'nonnumeric_volume'), (1+0j, 'nonnumeric_volume')]:
            v = [1.]*273
            v[123] = item
            result = kernel.measure(v)
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], reason)
            self.assertTrue(np.isnan(result['candidate']) and np.isnan(result['control']))
            self.assertEqual(result['rolling_medians_computed'], 0)
            self.assertEqual(result['high_volume_events'], [])

    def test_shape_rejected(self):
        for v in (np.ones(272), np.ones(274), np.ones((273, 1)), 1.):
            with self.assertRaises(ValueError):
                kernel.measure(v)

    def test_input_not_mutated(self):
        v = np.random.default_rng(22).uniform(1., 20., 273)
        before = v.copy()
        kernel.measure(v)
        np.testing.assert_array_equal(v, before)


class RankTests(unittest.TestCase):
    def test_average_ties_missing_and_small_domains(self):
        np.testing.assert_array_equal(kernel.centered_rank(np.array([1., 2., 2., 4., np.nan]),
            np.array([1, 1, 1, 1, 0], bool)), [-.5, 0., 0., .5, 0.])
        for a in ([False, False], [True, False], [True, True]):
            np.testing.assert_array_equal(kernel.centered_rank(np.ones(2), np.array(a)), [0., 0.])

    def test_permutation_and_invalid_inputs(self):
        raw = np.array([5., 2., 2., 9.])
        perm = [2, 0, 3, 1]
        base = kernel.centered_rank(raw, np.ones(4, bool))
        np.testing.assert_array_equal(kernel.centered_rank(raw[perm], np.ones(4, bool)), base[perm])
        with self.assertRaises(ValueError):
            kernel.centered_rank(raw, np.ones(4, int))
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.array([np.nan]), np.array([True]))


class BuildTests(unittest.TestCase):
    def test_schema_dates_counts_and_etfs_included(self):
        panel, meta, cal = fixture()
        values, audit = kernel.build_features(panel, meta, cal)
        self.assertEqual(set(values), {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'})
        self.assertTrue(values['available'].all())
        self.assertEqual(audit['event_observations'], 9*210)
        self.assertEqual(audit['adjacent_pair_observations'], 9*209)
        self.assertEqual(audit['rolling_medians_computed'], 9*210)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertEqual(audit['new_model_fits'], 0)
        row = audit['event_history'][0]
        t = cal.index(meta['dates'][0])
        self.assertEqual((row['volume_start'], row['volume_end']), (cal[t-314], cal[t-42]))
        self.assertEqual((row['event_start'], row['event_end']), (cal[t-251], cal[t-42]))
        self.assertTrue(row['cells'][2]['available'])
        json.dumps(audit, allow_nan=False)

    def test_initial_feature_not_artificially_suppressed(self):
        values, _ = kernel.build_features(*fixture())
        self.assertTrue(values['available'][0].all())
        self.assertEqual(values['control'][0, 0], 1.)
        self.assertEqual(values['control_rank'][0, 0], .5)

    def test_prefix_all_and_zero(self):
        args = fixture()
        full, audit = kernel.build_features(*args)
        for n in range(4):
            values, partial = kernel.build_features(*args, limit=n)
            for key in full:
                np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(partial['event_history'], audit['event_history'][:n])
        values, audit = kernel.build_features(*args, limit=0)
        self.assertEqual(values['candidate'].shape, (0, 3))
        self.assertEqual(audit['event_observations'], 0)

    def test_future_values_never_converted(self):
        p, m, c = fixture()
        full, audit = kernel.build_features(p, m, c, limit=1)
        bad = p['volume'].astype(object)
        cut = c.index(m['dates'][0])-42
        bad.loc[pd.Timestamp(c[cut+1]):, :] = 'future poison'
        actual, observed = kernel.build_features({'volume': bad}, m, c, limit=1)
        for key in full:
            np.testing.assert_array_equal(actual[key], full[key])
        self.assertEqual(observed, audit)
        kernel.build_features({'volume': bad}, m, c, limit=0)

    def test_future_symbol_values_not_read(self):
        p, m, c = fixture()
        m['symbols'][1:] = [['A', 'B', 'FUTURE']]*2
        p['volume']['FUTURE'] = 'poison'
        values, _ = kernel.build_features(p, m, c, limit=1)
        self.assertTrue(values['available'].all())

    def test_no_source_or_metadata_mutation(self):
        p, m, c = fixture()
        before, metadata = p['volume'].copy(deep=True), copy.deepcopy(m)
        kernel.build_features(p, m, c)
        pd.testing.assert_frame_equal(p['volume'], before)
        self.assertEqual(m, metadata)

    def test_missing_rows_symbols_and_values(self):
        p, m, c = fixture()
        drop = c[c.index(m['dates'][0])-100]
        p['volume'] = p['volume'].drop(pd.Timestamp(drop))
        v, a = kernel.build_features(p, m, c, limit=1)
        self.assertFalse(v['available'].any())
        self.assertEqual(a['reason_counts'], {'missing_official_session': 3})
        self.assertEqual(a['event_history'][0]['missing_official_sessions'], [drop])
        p, m, c = fixture()
        p['volume'] = p['volume'].drop(columns=['B'])
        v, a = kernel.build_features(p, m, c, limit=1)
        self.assertEqual(a['reason_counts'], {'available': 2, 'missing_symbol': 1})
        self.assertEqual(v['candidate_rank'][0, 1], 0.)
        p, m, c = fixture()
        p['volume'].loc[pd.Timestamp(c[c.index(m['dates'][0])-50]), 'B'] = 0.
        v, a = kernel.build_features(p, m, c, limit=1)
        self.assertEqual(a['reason_counts'], {'available': 2, 'invalid_volume': 1})
        self.assertEqual(a['rolling_medians_computed'], 420)

    def test_insufficient_history_and_single_valid_symbol(self):
        p, m, c = fixture()
        m = dict(dates=['2014-01-31'], symbols=[['A']], etfs=[])
        v, a = kernel.build_features(p, m, c)
        self.assertEqual(a['reason_counts'], {'insufficient_history': 1})
        self.assertFalse(v['available'].any())
        p, m, c = fixture()
        m['symbols'] = [['A']]*3
        v, a = kernel.build_features(p, m, c)
        self.assertTrue(v['available'].all())
        self.assertTrue((v['candidate_rank'] == 0.).all())
        self.assertTrue((v['control_rank'] == 0.).all())

    def test_symbol_permutation(self):
        p, m, c = fixture()
        original, _ = kernel.build_features(p, m, c)
        perm = [2, 0, 1]
        m['symbols'] = [[r[j] for j in perm] for r in m['symbols']]
        actual, _ = kernel.build_features(p, m, c)
        for key in original:
            np.testing.assert_array_equal(actual[key], original[key][:, perm])

    def test_monthend_and_initial_exception(self):
        p, m, c = fixture()
        m['dates'][1] = '2016-08-30'
        with self.assertRaisesRegex(ValueError, 'month-end'):
            kernel.build_features(p, m, c)
        p, m, c = fixture()
        c = c[:c.index(m['dates'][-1])+1]
        with self.assertRaisesRegex(ValueError, 'month-end'):
            kernel.build_features(p, m, c)

    def test_invalid_axes_metadata_and_limit(self):
        p, m, c = fixture()
        for limit in (-1, 4, True, 1.):
            with self.assertRaises(ValueError):
                kernel.build_features(p, m, c, limit=limit)
        with self.assertRaises(ValueError):
            kernel.build_features(dict(p, price=p['volume']), m, c)
        for field, value in [('symbols', [['A', 'A']]*3), ('etfs', ['FUND', 'FUND']),
                              ('symbols', [['A'], ['A', 'B'], ['A']])]:
            bad = copy.deepcopy(m)
            bad[field] = value
            with self.assertRaises(ValueError):
                kernel.build_features(p, bad, c)
        bad = p['volume'].copy()
        bad.index = bad.index.tz_localize('UTC')
        with self.assertRaises(ValueError):
            kernel.build_features({'volume': bad}, m, c)
        with self.assertRaises(ValueError):
            kernel.build_features(p, m, c[:20]+[c[19]]+c[20:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
