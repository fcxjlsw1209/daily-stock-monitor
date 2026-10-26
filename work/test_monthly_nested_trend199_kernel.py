"""Synthetic-only nested-horizon common support and original-rank validation."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_nested_trend199_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({'A': 100.+.03*x+3.*np.sin(x*.13),
        'B': 200.+.02*x+5.*np.cos(x*.23), 'FUND': 80.+.02*x+4.*np.sin(x*.37)}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
        symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


class MeasureTests(unittest.TestCase):
    def test_actual_horizon_endpoints_and_constant_valid(self):
        p = np.ones(253)*100.
        p[0], p[63], p[126], p[252] = 30., 40., 60., 120.
        result = kernel.measure(p)
        self.assertEqual([result['gain'+str(h)] for h in (126, 189, 252)], [1., 2., 3.])
        self.assertEqual(result['start_prices'], {'126': 60., '189': 40., '252': 30.})
        self.assertEqual(result['end_price'], 120.)
        result = kernel.measure(np.ones(253))
        self.assertTrue(result['source_available'] and result['available'])
        self.assertEqual([result['gain'+str(h)] for h in (126, 189, 252)], [0.]*3)

    def test_every_intermediate_source_required_and_invalid_types(self):
        for value, reason in [(0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                (np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price'),
                (True, 'nonnumeric_adjusted_price'), (None, 'nonnumeric_adjusted_price'),
                ('2', 'nonnumeric_adjusted_price'), (1+0j, 'nonnumeric_adjusted_price')]:
            p = [100.]*253; p[17] = value
            result = kernel.measure(p)
            self.assertFalse(result['source_available'] or result['available'])
            self.assertEqual(result['reason'], reason)
            self.assertIsNone(result['start_prices'])
            self.assertTrue(all(np.isnan(result['gain'+str(h)]) for h in (126, 189, 252)))
        for p in (np.ones(252), np.ones((253, 1)), 1.):
            with self.assertRaises(ValueError): kernel.measure(p)

    def test_finite_source_arithmetic_overflow_and_underflow(self):
        p = np.ones(253); p[0], p[252] = 1e-308, 1e308
        with self.assertRaises(FloatingPointError): kernel.measure(p)
        p[0], p[252] = 1e308, 1e-308
        self.assertEqual(kernel.measure(p)['gain252'], -1.)

    def test_power_two_scale_invariance_and_immutability(self):
        p = np.arange(100., 353.)
        before = p.copy(); expected = kernel.measure(p)
        for factor in (2.**-500, 2.**500):
            result = kernel.measure(p*factor)
            for h in (126, 189, 252): self.assertEqual(result['gain'+str(h)], expected['gain'+str(h)])
        np.testing.assert_array_equal(p, before)


class RankTests(unittest.TestCase):
    def test_original_average_percentiles_ties_and_missing(self):
        actual = kernel.percentile_rank(np.array([1., 2., 2., 4., np.nan]), np.array([True]*4+[False]))
        np.testing.assert_array_equal(actual, [-.25, .125, .125, .5, np.nan])
        np.testing.assert_array_equal(kernel.percentile_rank(np.ones(3), np.ones(3, bool)), [2/3-.5]*3)
        np.testing.assert_array_equal(kernel.percentile_rank(np.array([5., np.nan]), np.array([True, False])), [.5, np.nan])
        self.assertTrue(np.isnan(kernel.percentile_rank(np.array([np.nan]), np.array([False]))).all())
        for raw, available in [(np.ones(2), np.ones(3, bool)), (np.ones(2), np.ones(2)),
                               (np.array([np.inf]), np.array([True]))]:
            with self.assertRaises(ValueError): kernel.percentile_rank(raw, available)

    def test_direct_min_mean_and_exact_equal_copy(self):
        r = [.5, 2/3-.5, 1/3-.5]
        q, c, equal = kernel.aggregate(r)
        self.assertEqual(q, min(r)); self.assertEqual(c, math.fsum(r)/3); self.assertFalse(equal)
        for rank in (.1, .5, -.25):
            q, c, equal = kernel.aggregate([rank]*3)
            self.assertTrue(equal); self.assertEqual(q.hex(), rank.hex()); self.assertEqual(c.hex(), rank.hex())
        self.assertFalse(kernel.aggregate([.1, .1, np.nextafter(.1, 1.)])[2])
        for r in ([1., 0., 0.], [np.nan]*3, [.1]*2, [True, 0., 0.]):
            with self.assertRaises(ValueError): kernel.aggregate(r)


class BuilderTests(unittest.TestCase):
    def test_schema_counts_horizon_dates_initial_and_etfs(self):
        panel, meta, calendar = fixture()
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(set(values), {'candidate', 'control', 'available', 'gain126', 'gain189', 'gain252',
                                      'rank126', 'rank189', 'rank252'})
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [3]*3)
        self.assertEqual(audit['valid_price_observations'], 253*9)
        self.assertEqual(audit['horizon_returns'], 27)
        self.assertEqual(audit['ranked_horizon_observations'], 27)
        self.assertEqual(audit['aggregated_scores'], 18)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertEqual(audit['new_model_fits'], 0)
        self.assertFalse(audit['ranking_eligibility_filtered'])
        self.assertTrue(np.isfinite(values['candidate'][0]).all())
        for row in audit['event_history']:
            t = row['session_position']
            self.assertEqual((row['price_start'], row['price_end']), (calendar[t-294], calendar[t-42]))
            self.assertEqual(row['horizon_start_dates'], {str(h): calendar[t-42-h] for h in (126, 189, 252)})
            self.assertTrue(row['cells'][2]['available'])

    def test_direct_targets_not_ranked_a_second_time(self):
        panel, meta, calendar = fixture(); meta = dict(meta, dates=meta['dates'][:1], symbols=meta['symbols'][:1])
        t = calendar.index(meta['dates'][0]); dates = pd.to_datetime(calendar[t-294:t-41])
        # Horizon order A>B>C, A>C>B, B>C>A produces identical minima but distinct means.
        gains = {'A': [3., 3., 1.], 'B': [2., 1., 3.], 'FUND': [1., 2., 2.]}
        for symbol, gs in gains.items():
            panel['adj'].loc[dates, symbol] = 100.
            panel['adj'].loc[dates[-1], symbol] = 120.
            for h, g in zip((126, 189, 252), gs): panel['adj'].loc[dates[252-h], symbol] = 120/(1+g)
        values, audit = kernel.build_features(panel, meta, calendar)
        np.testing.assert_array_equal(values['candidate'][0], [1/3-.5]*3)
        for j in range(3):
            ranks = [values['rank'+str(h)][0, j] for h in (126, 189, 252)]
            self.assertEqual(values['control'][0, j], math.fsum(ranks)/3)
            self.assertFalse(audit['event_history'][0]['cells'][j]['exact_equal_ranks'])
        self.assertEqual(len(set(values['control'][0])), 3)

    def test_common_support_all_float_missing_and_singleton_positive_half(self):
        panel, meta, calendar = fixture(); t = calendar.index(meta['dates'][0])
        panel['adj'].loc[pd.Timestamp(calendar[t-100]), ['B', 'FUND']] = np.nan
        values, audit = kernel.build_features(panel, meta, calendar, limit=1)
        self.assertEqual(audit['source_available_by_event'], [1])
        self.assertEqual(audit['available_by_event'], [1])
        for key, value in values.items():
            if key != 'available': self.assertTrue(np.isnan(value[0, 1:]).all())
        for key in ('rank126', 'rank189', 'rank252', 'candidate', 'control'):
            self.assertEqual(values[key][0, 0], .5)
        self.assertTrue(audit['event_history'][0]['cells'][0]['exact_equal_ranks'])

    def test_constant_common_rows_all_tie_offset_and_empty(self):
        panel, meta, calendar = fixture(); panel['adj'].loc[:, :] = 100.
        values, audit = kernel.build_features(panel, meta, calendar)
        for key in ('rank126', 'rank189', 'rank252', 'candidate', 'control'):
            np.testing.assert_array_equal(values[key], np.full((3, 3), 2/3-.5))
        self.assertTrue(all(c['exact_equal_ranks'] for r in audit['event_history'] for c in r['cells']))
        panel['adj'].loc[:, :] = np.nan
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(audit['available_cells'], 0)
        self.assertEqual(audit['aggregated_scores'], 0)
        self.assertTrue(all(np.isnan(v).all() for k, v in values.items() if k != 'available'))

    def test_structural_missing_priority_and_no_compressed_calendar(self):
        panel, meta, calendar = fixture(); t = calendar.index(meta['dates'][0])
        panel['adj'] = panel['adj'].drop(index=pd.Timestamp(calendar[t-100]), columns='B')
        values, audit = kernel.build_features(panel, meta, calendar, limit=1)
        self.assertEqual([c['reason'] for c in audit['event_history'][0]['cells']],
                         ['missing_official_session', 'missing_symbol', 'missing_official_session'])
        self.assertEqual(audit['event_history'][0]['missing_official_sessions'], [calendar[t-100]])
        short = calendar[t-100:]
        _, audit = kernel.build_features(panel, meta, short, limit=1)
        self.assertEqual([c['reason'] for c in audit['event_history'][0]['cells']], ['insufficient_history']*3)

    def test_all_prefixes_physically_truncated_equal_and_empty_safe(self):
        panel, meta, calendar = fixture(); full, audit = kernel.build_features(panel, meta, calendar)
        for n in range(4):
            truncated = {'adj': panel['adj'].loc[:meta['dates'][n-1]].copy() if n else panel['adj'].iloc[:0].copy()}
            values, other = kernel.build_features(truncated, meta, calendar, limit=n)
            for key in full: np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(other['event_history'], audit['event_history'][:n])
            self.assertEqual(other['source_available_by_event'], audit['source_available_by_event'][:n])
            self.assertEqual(other['symbols_per_event'], 3)

    def test_skip42_future_poison_and_unused_symbol_never_converted(self):
        panel, meta, calendar = fixture(); expected, audit = kernel.build_features(panel, meta, calendar, limit=1)
        t = calendar.index(meta['dates'][0]); panel['adj'] = panel['adj'].astype(object)
        panel['adj'].loc[pd.Timestamp(calendar[t-41]):] = 'future poison'
        panel['adj']['UNUSED'] = 'unread poison'
        actual, other = kernel.build_features(panel, meta, calendar, limit=1)
        for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(other, audit)
        values, empty = kernel.build_features(panel, meta, calendar, limit=0)
        self.assertEqual(values['candidate'].shape, (0, 3)); self.assertEqual(empty['checked_cells'], 0)

    def test_permutation_and_no_input_mutation(self):
        panel, meta, calendar = fixture(); source = panel['adj'].copy(deep=True); original_meta = copy.deepcopy(meta)
        values, _ = kernel.build_features(panel, meta, calendar)
        permutation = [2, 0, 1]
        changed_meta = dict(meta, symbols=[[s[j] for j in permutation] for s in meta['symbols']])
        changed, _ = kernel.build_features(panel, changed_meta, calendar)
        for key in values: np.testing.assert_array_equal(changed[key], values[key][:, permutation])
        pd.testing.assert_frame_equal(source, panel['adj']); self.assertEqual(meta, original_meta)

    def test_metadata_and_source_axis_validation(self):
        panel, meta, calendar = fixture()
        for limit in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError): kernel.build_features(panel, meta, calendar, limit=limit)
        bad = copy.deepcopy(meta); bad['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): kernel.build_features(panel, bad, calendar)
        bad = copy.deepcopy(meta); bad['symbols'][1] = ['A']
        with self.assertRaises(ValueError): kernel.build_features(panel, bad, calendar)
        bad = copy.deepcopy(meta); bad['etfs'] = ['FUND', 'FUND']
        with self.assertRaises(ValueError): kernel.build_features(panel, bad, calendar)
        with self.assertRaises(ValueError): kernel.build_features({'adj': panel['adj'], 'volume': panel['adj']}, meta, calendar)
        with self.assertRaises(ValueError): kernel.build_features({'adj': panel['adj'].iloc[::-1]}, meta, calendar)
        with self.assertRaises(ValueError): kernel.build_features(panel, meta, calendar[:calendar.index(meta['dates'][-1])+1])
        with self.assertRaises(ValueError): kernel.build_features(panel, meta, calendar+[calendar[-1]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
