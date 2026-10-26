"""Synthetic-only recovery/durability checks; no market panels or portfolio I/O."""
import copy
import unittest

import numpy as np
import pandas as pd

import monthly_recovery_durability197_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    adj = pd.DataFrame({'A': 100.+.03*x+3.*np.sin(x*.13),
        'B': 200.+.02*x+5.*np.cos(x*.23), 'FUND': 80.+.02*x+4.*np.sin(x*.37)}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
                symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'adj': adj}, meta, dates.strftime('%Y-%m-%d').tolist()


class MeasureTests(unittest.TestCase):
    def test_preflight_same_recovery_different_durability(self):
        a = kernel.measure([100, 90, 100, 105, 101]+[105]*248)
        b = kernel.measure([100, 90, 100, 105, 99]+[105]*248)
        for result in (a, b):
            self.assertEqual([e['anchor_index'] for e in result['anchor_events']], [1, 4])
            self.assertEqual([e['recovery_lag'] for e in result['anchor_events']], [1, 1])
            self.assertEqual(result['control'], 1.)
            self.assertEqual(result['negative_anchors'], 2)
            self.assertEqual(result['recovered_anchors'], 2)
        self.assertEqual((a['candidate'], b['candidate']), (1., .5))
        self.assertEqual((a['durable_anchors'], b['durable_anchors']), (2, 1))

    def test_no_negative_anchors_missing_not_zero(self):
        for path in (np.ones(253), np.arange(1., 254.)):
            result = kernel.measure(path)
            self.assertTrue(result['source_available'])
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'no_negative_anchors')
            self.assertEqual(result['anchor_events'], [])
            self.assertTrue(np.isnan(result['candidate']) and np.isnan(result['control']))

    def test_negative_without_recovery_is_real_zero(self):
        result = kernel.measure(np.arange(253., 0., -1))
        self.assertTrue(result['available'])
        self.assertEqual(result['negative_anchors'], 210)
        self.assertEqual((result['candidate'], result['control']), (0., 0.))
        self.assertEqual((result['recovered_anchors'], result['durable_anchors']), (0, 0))
        for event in result['anchor_events']:
            self.assertFalse(event['recovered'] or event['durable'])
            self.assertIsNone(event['recovery_lag'])
            self.assertIsNone(event['post_minimum'])

    def test_last_anchor_last_recovery_and_last_post_close(self):
        path = np.full(253, 100.)
        path[210:231] = 90.
        result = kernel.measure(path)
        self.assertEqual(result['negative_anchors'], 1)
        event = result['anchor_events'][0]
        self.assertEqual((event['anchor_index'], event['reference_index']), (210, 209))
        self.assertEqual((event['recovery_lag'], event['recovery_index']), (21, 231))
        self.assertEqual((event['post_start_index'], event['post_end_index']), (232, 252))
        self.assertEqual(event['post_minimum'], 100.)
        self.assertEqual((result['candidate'], result['control']), (1., 1.))
        path[-1] = np.nextafter(100., 0.)
        changed = kernel.measure(path)
        self.assertEqual(changed['negative_anchors'], 1)
        self.assertEqual((changed['candidate'], changed['control']), (0., 1.))

    def test_first_crossing_not_later_recovery(self):
        path = np.full(253, 100.)
        path[1] = 90.
        path[3] = 99.
        event = kernel.measure(path)['anchor_events'][0]
        self.assertEqual(event['recovery_lag'], 1)
        self.assertEqual((event['post_start_index'], event['post_end_index']), (3, 23))
        self.assertFalse(event['durable'])
        self.assertEqual(event['post_minimum'], 99.)

    def test_recovery_outside21_is_not_counted(self):
        path = np.full(253, 100.)
        path[1:23] = 90.
        result = kernel.measure(path)
        self.assertEqual(result['negative_anchors'], 1)
        self.assertEqual((result['candidate'], result['control']), (0., 0.))
        self.assertIsNone(result['anchor_events'][0]['recovery_index'])

    def test_strict_negative_equality_and_post_window_exclusion(self):
        path = np.full(253, 100.)
        path[210] = 90.
        path[211] = 110.
        # Recovery close is excluded; the following21 closes all equal reference.
        result = kernel.measure(path)
        event = result['anchor_events'][0]
        self.assertEqual(event['recovery_price'], 110.)
        self.assertEqual(event['post_minimum'], 100.)
        self.assertTrue(event['durable'])
        self.assertEqual(event['post_end_index'], 232)
        path[233] = 90.  # Outside the fixed phase, and outside eligible anchor range.
        changed = kernel.measure(path)
        self.assertEqual(changed['candidate'], 1.)
        self.assertEqual(changed['negative_anchors'], 1)

    def test_scale_and_extreme_finite_values(self):
        path = np.array([100, 90, 100, 105, 99]+[105]*248, dtype=float)
        expected = kernel.measure(path)
        for scale in (2.**-900, 2.**900):
            other = kernel.measure(path*scale)
            for key in ('candidate', 'control', 'negative_anchors', 'recovered_anchors', 'durable_anchors'):
                self.assertEqual(other[key], expected[key])
            for a, b in zip(other['anchor_events'], expected['anchor_events']):
                for key in ('anchor_index', 'recovery_lag', 'post_start_index', 'post_end_index', 'durable'):
                    self.assertEqual(a[key], b[key])
        extreme = kernel.measure([1e300, 1e-300]+[1e300]*251)
        self.assertEqual((extreme['candidate'], extreme['control']), (1., 1.))

    def test_full253_invalid_types_shapes_and_no_mutation(self):
        original = np.array([100, 90, 100]+[100]*250, dtype=float)
        before = original.copy()
        kernel.measure(original)
        np.testing.assert_array_equal(original, before)
        for value, reason in [(0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                (np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price'),
                (True, 'nonnumeric_adjusted_price'), (None, 'nonnumeric_adjusted_price'),
                ('2', 'nonnumeric_adjusted_price'), (1+0j, 'nonnumeric_adjusted_price')]:
            path = list(original)
            path[250] = value  # Even after this path's recovered anchors mature.
            result = kernel.measure(path)
            self.assertFalse(result['source_available'] or result['available'])
            self.assertEqual(result['reason'], reason)
            self.assertEqual(result['negative_anchors'], 0)
            self.assertEqual(result['anchor_events'], [])
        for path in (np.ones(252), np.ones((253, 1)), 1.):
            with self.assertRaises(ValueError):
                kernel.measure(path)


class RankTests(unittest.TestCase):
    def test_common_ties_zero_and_singleton(self):
        np.testing.assert_array_equal(kernel.centered_rank(np.array([0., .5, .5, 1., np.nan]),
            np.array([True]*4+[False])), [-.5, 0., 0., .5, 0.])
        for mask in ([False, False], [True, False], [True, True]):
            np.testing.assert_array_equal(kernel.centered_rank(np.zeros(2), np.array(mask)), [0., 0.])
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.array([np.nan]), np.array([True]))


class BuilderTests(unittest.TestCase):
    def test_schema_official_windows_and_descriptive_counts(self):
        panel, meta, calendar = fixture()
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(set(values), {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'})
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [3]*3)
        self.assertEqual(audit['valid_price_observations'], 253*9)
        self.assertEqual(audit['new_model_fits'], 0)
        self.assertTrue(values['available'].all())
        self.assertTrue(np.isfinite(values['candidate'][0]).all())
        self.assertTrue((values['candidate'] <= values['control']).all())
        for key in ('negative_anchors', 'recovered_anchors', 'durable_anchors'):
            self.assertEqual(audit[key], sum(audit[key+'_by_event']))
            self.assertEqual(audit[key], sum(c[key] for e in audit['event_history'] for c in e['cells']))
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual((event['price_start'], event['price_end']), (calendar[t-294], calendar[t-42]))
            self.assertEqual((event['anchor_first_date'], event['anchor_last_date']), (calendar[t-293], calendar[t-84]))
            self.assertTrue(event['cells'][2]['available'])  # ETF treated identically.
            for cell in event['cells']:
                for anchor in cell['anchor_events']:
                    self.assertLessEqual(anchor['anchor_index'], 210)
                    if anchor['recovered']:
                        self.assertEqual(anchor['post_end_index']-anchor['post_start_index']+1, 21)
                        self.assertLessEqual(anchor['post_end_index'], 252)

    def test_no_anchor_source_valid_but_feature_missing_and_singleton_valid(self):
        panel, meta, calendar = fixture()
        panel['adj']['B'] = 100.
        panel['adj']['FUND'] = 200.
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [1]*3)
        self.assertEqual(audit['source_available_cells'], 9)
        self.assertEqual(audit['available_cells'], 3)
        np.testing.assert_array_equal(values['available'], [[True, False, False]]*3)
        np.testing.assert_array_equal(values['candidate_rank'], np.zeros((3, 3)))
        self.assertTrue(np.isfinite(values['candidate'][:, 0]).all())
        self.assertTrue(np.isnan(values['candidate'][:, 1:]).all())

    def test_middle_missing_price_row_symbol_and_history_priorities(self):
        panel, meta, calendar = fixture()
        t = calendar.index(meta['dates'][0])
        bad = copy.deepcopy(panel)
        bad['adj'].loc[calendar[t-294+17], 'B'] = np.nan
        values, audit = kernel.build_features(bad, meta, calendar, limit=1)
        np.testing.assert_array_equal(values['available'], [[True, False, True]])
        self.assertEqual(audit['source_available_by_event'], [2])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'invalid_adjusted_price')
        bad['adj'] = bad['adj'].drop(pd.Timestamp(calendar[t-294+17])).drop(columns='B')
        _, audit = kernel.build_features(bad, meta, calendar, limit=1)
        self.assertEqual(audit['reason_counts'], {'missing_official_session': 2, 'missing_symbol': 1})
        short_meta = dict(dates=['2016-08-31'], symbols=[['UNKNOWN']], etfs=[])
        _, audit = kernel.build_features(panel, short_meta, calendar[-200:])
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 1})
        self.assertIsNone(audit['event_history'][0]['price_start'])

    def test_prefixes_future_poison_and_source_truncation(self):
        panel, meta, calendar = fixture()
        full, audit = kernel.build_features(panel, meta, calendar)
        for n in range(4):
            cut = {'adj': panel['adj'].iloc[:0].copy() if n == 0 else panel['adj'].loc[:meta['dates'][n-1]].copy()}
            small, check = kernel.build_features(cut, meta, calendar, limit=n)
            for key in full:
                np.testing.assert_array_equal(small[key], full[key][:n])
            self.assertEqual(check['event_history'], audit['event_history'][:n])
            for key in ('source_available', 'available', 'negative_anchors', 'recovered_anchors', 'durable_anchors'):
                self.assertEqual(check[key+'_by_event'], audit[key+'_by_event'][:n])
        t = calendar.index(meta['dates'][0])
        poison = {'adj': panel['adj'].astype(object)}
        poison['adj'].loc[calendar[t-41]:, :] = 'future'
        poison['adj']['UNUSED'] = 'unrequested source'
        small, check = kernel.build_features(poison, meta, calendar, limit=1)
        self.assertEqual(check['event_history'], audit['event_history'][:1])
        for key in full:
            np.testing.assert_array_equal(small[key], full[key][:1])

    def test_permutation_and_immutability(self):
        panel, meta, calendar = fixture()
        old_frame, old_meta, old_calendar = panel['adj'].copy(deep=True), copy.deepcopy(meta), list(calendar)
        full, _ = kernel.build_features(panel, meta, calendar)
        other_meta = copy.deepcopy(meta)
        order = [2, 0, 1]
        other_meta['symbols'] = [[row[j] for j in order] for row in meta['symbols']]
        other, _ = kernel.build_features(panel, other_meta, calendar)
        for key in full:
            np.testing.assert_array_equal(other[key], full[key][:, order])
        pd.testing.assert_frame_equal(panel['adj'], old_frame)
        self.assertEqual(meta, old_meta)
        self.assertEqual(calendar, old_calendar)

    def test_validation_and_monthend_next_session(self):
        panel, meta, calendar = fixture()
        for change in ({'etfs': ['FUND', 'FUND']}, {'dates': ['2016-08-18']},
                       {'symbols': [['A'], ['A', 'B'], ['A']]}):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, dict(meta, **change), calendar)
        for limit in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, meta, calendar, limit=limit)
        with self.assertRaises(ValueError):
            kernel.build_features(dict(panel, volume=panel['adj']), meta, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features({'adj': panel['adj'].iloc[::-1]}, meta, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, meta, calendar[:calendar.index('2016-09-30')+1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
