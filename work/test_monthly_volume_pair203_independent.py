"""Synthetic-only strict prior-volume clock and paired excess audits."""
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_volume_pair203_independent as independent


def fixture(count=3, width=5):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    adj = pd.DataFrame({'A': np.exp(.0004*x+.13*np.cos(x/17.)),
        'B': np.exp(.0002*x+.10*np.sin(x/27.)), 'ASML': np.exp(.0001*x+.12*np.sin(x/8.)),
        'XLK': np.exp(.0002*x+.04*np.cos(x/13.)), 'XLC': np.exp(.0002*x+.05*np.cos(x/23.)),
        'FUND': np.ones(len(index)), 'UNKNOWN': np.ones(len(index)), 'UNUSED': np.full(len(index), np.nan)}, index=index)
    volume = pd.DataFrame({'A': np.exp(.002*x+.3*np.sin(x/8.)), 'B': np.exp(.3*np.cos(x/19.)),
        'ASML': np.exp(.005*x+.1*np.sin(x/2.)), 'FUND': np.ones(len(index)), 'UNKNOWN': np.ones(len(index))}, index=index)
    sectors = pd.DataFrame({'A': 'Technology', 'B': 'Communication Services', 'ASML': 'Energy',
        'FUND': 'Technology', 'UNKNOWN': 'Unknown'}, index=index)
    names = ['A', 'B', 'ASML', 'FUND', 'UNKNOWN'][:width]
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': adj, 'volume': volume, 'sectors': sectors}, dict(dates=dates,
        symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def mapper(day, symbols, sectors, etfs):
    mapping = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Energy': 'XLE'}
    return {s: None if s in etfs else 'XLK' if s == 'ASML' else
        mapping.get(sectors.at[pd.Timestamp(day), s]) if s in sectors.columns and
        pd.Timestamp(day) in sectors.index else None for s in symbols}


def volume_for_flags(flags):
    values = [1.] * 63
    for flag in flags:
        median = sorted(values[-63:])[31]
        values.append(median * (2. if flag else .5))
    return np.array(values)


def scalar_fixture():
    x = np.arange(253, dtype=float)
    s = np.exp(.002*x+.03*np.sin(x/9.))
    b = np.exp(.001*x+.02*np.cos(x/13.))
    v = np.exp(.003*np.arange(315)+.1*np.sin(np.arange(315)/13.))
    return s, b, v


class ScalarTests(unittest.TestCase):
    def test_strict_previous63_odd_median_and_ties(self):
        v = np.ones(315); v[:63] = [1.] * 31 + [10.] + [20.] * 31; v[63] = 10.
        a = independent.measure_oracle(np.ones(253), np.ones(253), v)
        self.assertEqual(a['preceding_medians'][0], 10.)
        self.assertEqual(a['high_volume_flags'][0], 0)
        v[63] = np.nextafter(10., np.inf)
        b = independent.measure_oracle(np.ones(253), np.ones(253), v)
        self.assertEqual(b['preceding_medians'][0], 10.)
        self.assertEqual(b['high_volume_flags'][0], 1)
        self.assertEqual(len(b['preceding_medians']), 252)
        self.assertEqual(b['preceding_medians'][-1], sorted(v[251:314])[31])
        self.assertEqual(b['high_volume_flags'][-1], int(v[314] > sorted(v[251:314])[31]))

    def test_block_alignment_and_pairing_permuted_activity(self):
        s = np.ones(253); endpoints = [1., 2.] * 6 + [1.]
        s[::21] = endpoints
        v = volume_for_flags(([1] * 21 + [0] * 21) * 6)
        reverse = volume_for_flags(([0] * 21 + [1] * 21) * 6)
        a = independent.measure_oracle(s, np.ones(253), v)
        b = independent.measure_oracle(s, np.ones(253), reverse)
        self.assertEqual(a['high_counts'], [21, 0] * 6)
        self.assertEqual(b['high_counts'], [0, 21] * 6)
        self.assertEqual((a['control'], b['control']), (0., 0.))
        self.assertEqual(a['candidate'], math.log(2.) / 2)
        self.assertEqual(b['candidate'], -math.log(2.) / 2)
        self.assertEqual(a['volume_block_start_indices'], list(range(63, 315, 21)))
        self.assertEqual(a['volume_block_end_indices'], list(range(83, 315, 21)))
        self.assertEqual(a['block_end_indices'][-1], 252)

    def test_constant_branches_valid_zero_and_no_epsilon(self):
        s, b, _ = scalar_fixture()
        a = independent.measure_oracle(s, b, np.ones(315))
        self.assertTrue(a['available']); self.assertEqual(a['constant_branch'], 'all_activity_equal')
        self.assertEqual(a['candidate'].hex(), a['control'].hex())
        a = independent.measure_oracle(s, s.copy(), volume_for_flags(([1] * 21 + [0] * 21) * 6))
        self.assertEqual(a['constant_branch'], 'all_excess_equal')
        self.assertEqual((a['candidate'], a['control']), (0., 0.))
        e = [.1] * 12; e[1] = float(np.nextafter(.1, np.inf))
        self.assertIsNone(independent.paired_summary(e, [1., 0.] * 6)['constant_branch'])
        self.assertEqual(independent.paired_summary([0.] * 12, [0.] * 12)['constant_branch'], 'all_activity_equal')

    def test_all_sources_including_prior_volume_and_nonendpoint_prices(self):
        for axis, name in enumerate(('stock_price', 'benchmark_price', 'volume')):
            for position in (0, 11, 62, 63, 252):
                for bad in (0., -1., np.nan, np.inf):
                    args = list(scalar_fixture()); args[axis][position] = bad
                    a = independent.measure_oracle(*args)
                    self.assertEqual(a['reason'], 'invalid_' + name)
                    self.assertFalse(a['available']); self.assertIsNone(a['candidate'])
                    self.assertEqual(a['high_volume_flags'], [])
                    self.assertTrue(all(a[k] == 0 for k in independent.COUNTERS))
            for bad in (True, None, '1', complex(1, 0)):
                args = [x.astype(object) for x in scalar_fixture()]; args[axis][31] = bad
                self.assertEqual(independent.measure_oracle(*args)['reason'], 'nonnumeric_' + name)
        s, b, v = scalar_fixture(); v[-1] = 0
        self.assertEqual(independent.measure_oracle(s, b, v)['reason'], 'invalid_volume')
        with self.assertRaises(ValueError): independent.measure_oracle(s[:-1], b, v)
        with self.assertRaises(ValueError): independent.measure_oracle(s, b, v[:-1])

    def test_unit_invariance_not_corporate_action_correction(self):
        s, b, v = scalar_fixture(); before = tuple(x.copy() for x in (s, b, v))
        a = independent.measure_oracle(s, b, v)
        for scale in (2. ** -400, 2. ** 400):
            z = independent.measure_oracle(s, b, v * scale)
            for key in a:
                if key != 'preceding_medians': self.assertEqual(a[key], z[key])
            np.testing.assert_array_equal(np.array(a['preceding_medians']) * scale, z['preceding_medians'])
        jump = v.copy(); jump[100:] *= 100
        z = independent.measure_oracle(s, b, jump)
        self.assertNotEqual(a['high_volume_flags'], z['high_volume_flags'])
        for x, y in zip(before, (s, b, v)): np.testing.assert_array_equal(x, y)

    def test_extreme_logs_and_arithmetic_failure(self):
        s = np.ones(253); s[0] = 5e-324; s[21] = 1e308
        a = independent.measure_oracle(s, np.ones(253), np.ones(315))
        self.assertTrue(a['available']); self.assertTrue(math.isfinite(a['candidate']))
        with patch.object(independent.math, 'log', return_value=np.inf):
            with self.assertRaises(FloatingPointError): independent.measure_oracle(*scalar_fixture())


class BuildTests(unittest.TestCase):
    def test_clock_counters_mapping_and_no_mutation(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        v, a = independent.independent_features(p, m, c, mapper)
        r = independent.verify_features(p, m, c, mapper, v, a)
        self.assertTrue(all(r[k] is True for k in independent.FEATURE_FLAGS))
        self.assertEqual(r['available_by_event'], [3] * 3)
        for key, multiple in zip(independent.COUNTERS, (506, 315, 252, 252, 12, 12, 1, 2, 2)):
            self.assertEqual(r[key], 9 * multiple)
        for row in a['event_history']:
            t = c.index(row['date'])
            self.assertEqual(row['price_dates'], c[t - 294:t - 41])
            self.assertEqual(row['volume_dates'], c[t - 356:t - 41])
            self.assertEqual(row['activity_dates'], c[t - 293:t - 41])
            self.assertEqual(row['median_start_dates'][0], c[t - 356])
            self.assertEqual(row['median_end_dates'][0], c[t - 294])
            self.assertEqual(row['median_end_dates'][-1], c[t - 43])
            self.assertEqual(row['volume_block_end_dates'], row['endpoint_dates'][1:])
            self.assertEqual(row['cells'][2]['benchmark'], 'XLK')
        for key in p: pd.testing.assert_frame_equal(p[key], before[key])

    def test_unused_etf_unmapped_sources_singleton_and_tie_ranks(self):
        p, m, c = fixture(); base, audit = independent.independent_features(p, m, c, mapper)
        p['adj']['UNUSED'] = 'unconsumed'; p['volume']['FUND'] = 'ETF'; p['adj']['UNKNOWN'] = 'unmapped'
        independent.verify_features(p, m, c, mapper, base, audit)
        p, m, c = fixture(width=1); v, _ = independent.independent_features(p, m, c, mapper)
        self.assertTrue(v['available'].all()); self.assertTrue((v['candidate_rank'] == 0).all())
        raw = np.array([[2., 2., 6., np.nan], [3., 3., 3., np.nan]])
        np.testing.assert_array_equal(independent.centered_ranks(raw, np.isfinite(raw)),
                                      [[-.25, -.25, .5, 0.], [0., 0., 0., 0.]])

    def test_missing_sources_complete_calendars_and_reason_order(self):
        for frame, col, name in (('adj', 'A', 'missing_symbol'), ('adj', 'XLC', 'missing_benchmark_symbol'),
                                  ('volume', 'A', 'missing_volume_symbol')):
            p, m, c = fixture(); p[frame] = p[frame].drop(columns=col)
            _, a = independent.independent_features(p, m, c, mapper)
            self.assertEqual(a['event_history'][0]['cells'][1 if col == 'XLC' else 0]['reason'], name)
        p, m, c = fixture(); t = c.index(m['dates'][0]); bad = c[t - 294 + 11]
        p['adj'] = p['adj'].drop(pd.Timestamp(bad))
        _, a = independent.independent_features(p, m, c, mapper)
        self.assertEqual(a['event_history'][0]['missing_official_price_sessions'], [bad])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'missing_official_price_session')
        p, m, c = fixture(); bad = c[t - 356 + 11]
        p['volume'] = p['volume'].drop(pd.Timestamp(bad))
        _, a = independent.independent_features(p, m, c, mapper)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'missing_official_volume_session')
        self.assertEqual(a['event_history'][0]['missing_official_volume_sessions'], [bad])

    def test_source_mapping_as_query_and_false_callback_rejected(self):
        p, m, c = fixture(); observed = []
        def checked(day, symbols, sectors, etfs):
            self.assertLessEqual(sectors.index.max(), pd.Timestamp(day)); observed.append(day)
            return mapper(day, symbols, sectors, etfs)
        independent.independent_features(p, m, c, checked)
        self.assertEqual(observed, m['dates'])
        def wrong(day, symbols, sectors, etfs):
            out = mapper(day, symbols, sectors, etfs); out['A'] = 'XLF'; return out
        with self.assertRaises(ValueError): independent.independent_features(p, m, c, wrong)

    def test_physical_prefixes_numeric_future_poison_and_skip42(self):
        p, m, c = fixture(); full, audit = independent.independent_features(p, m, c, mapper)
        for n in range(4):
            q = {k: f.loc[:m['dates'][n - 1]].copy() if n else f.iloc[:0].copy() for k, f in p.items()}
            v, a = independent.independent_features(q, m, c, mapper, n)
            independent.verify_features(q, m, c, mapper, v, a, n)
            self.assertEqual(a['event_history'], audit['event_history'][:n])
            for key in full: np.testing.assert_array_equal(v[key], full[key][:n])
        q = copy.deepcopy(p)
        for key in ('adj', 'volume'):
            q[key] = q[key].astype(object)
            q[key].loc[q[key].index > pd.Timestamp(m['dates'][0]), :] = 'future poison'
        v, _ = independent.independent_features(q, m, c, mapper, 1)
        for key in full: np.testing.assert_array_equal(v[key], full[key][:1])
        q = copy.deepcopy(p); t = c.index(m['dates'][0])
        for key in ('adj', 'volume'):
            q[key] = q[key].astype(object); q[key].loc[c[t - 41]:m['dates'][0], :] = 'recent unused'
        v, _ = independent.independent_features(q, m, c, mapper, 1)
        for key in full: np.testing.assert_array_equal(v[key], full[key][:1])

    def test_schema_history_and_monthends(self):
        p, m, c = fixture()
        for limit in (True, -1, 4, 1.5):
            with self.assertRaises(ValueError): independent.independent_features(p, m, c, mapper, limit)
        bad = copy.deepcopy(m); bad['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p, bad, c, mapper)
        with self.assertRaises(ValueError): independent.independent_features(p, m, c[:c.index(m['dates'][-1]) + 1], mapper)
        short = c[c.index('2016-01-01'):]
        _, a = independent.independent_features(p, m, short, mapper)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        self.assertEqual(a['event_history'][0]['volume_dates'], [])
        t = c.index(m['dates'][0])
        _, a = independent.independent_features(p, m, c[t - 320:], mapper, 1)
        self.assertEqual(len(a['event_history'][0]['price_dates']), 253)
        self.assertEqual(a['event_history'][0]['volume_dates'], [])
        self.assertEqual(a['event_history'][0]['cells'][0]['endpoint_indices'], [])
        with self.assertRaises(ValueError): independent.independent_features({k: v for k, v in p.items() if k != 'volume'}, m, c, mapper)

    def test_whole_artifact_corruption_rejected(self):
        p, m, c = fixture(); v, a = independent.independent_features(p, m, c, mapper)
        for mutate in (lambda z: z['event_history'][0]['cells'][0]['high_volume_flags'].__setitem__(0, 99),
                       lambda z: z['event_history'][0]['cells'][0]['preceding_medians'].__setitem__(0, 99.),
                       lambda z: z.__setitem__('valid_volume_observations', 0),
                       lambda z: z['event_history'][0]['median_end_dates'].__setitem__(0, m['dates'][0])):
            bad = copy.deepcopy(a); mutate(bad)
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, mapper, v, bad)
        for key in ('candidate', 'control', 'candidate_rank', 'control_rank'):
            bad = {k: x.copy() for k, x in v.items()}; bad[key][0, 0] += .001
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, mapper, bad, a)

    def test_deltas_fixed_linear_initial_zero_and_bound(self):
        p, m, c = fixture(); v, _ = independent.independent_features(p, m, c, mapper)
        delta = independent.independent_deltas(v); r = independent.verify_deltas(v, delta)
        self.assertEqual(r['max_absolute_delta'], .005)
        for arm, raw in independent.RAW.items():
            self.assertTrue((delta[arm][0] == 0).all())
            np.testing.assert_array_equal(delta[arm][1:], .01 * v[raw + '_rank'][1:])
            bad = {k: x.copy() for k, x in delta.items()}; bad[arm][1, 0] += 1e-16
            with self.assertRaises(ValueError): independent.verify_deltas(v, bad)
        with self.assertRaises(ValueError): independent.verify_deltas(v, delta, .02)

    def test_frozen_helper_and_production_independence(self):
        self.assertNotIn('import monthly_volume_pair203_kernel', inspect.getsource(independent))
        self.assertTrue(independent.FROZEN_HELPER in independent.dependency_files())
        self.assertTrue(callable(independent.accounting().audit_path))
        with patch.object(independent, 'FROZEN_HELPER_SHA', '0' * 64):
            with self.assertRaises(ValueError): independent.frozen_helper()

    def test_both_producer_fixtures_and_all_prefix_full_audit(self):
        import monthly_volume_pair203_kernel as production
        import test_monthly_volume_pair203_kernel as producer_test
        for fixture_fn, callback in ((fixture, mapper), (producer_test.fixture, producer_test.mapper)):
            p, m, c = fixture_fn()
            for n in range(len(m['dates']) + 1):
                q = {k: f.loc[:m['dates'][n - 1]].copy() if n else f.iloc[:0].copy() for k, f in p.items()}
                v, a = production.build_features(q, m, c, callback, n)
                independent.verify_features(q, m, c, callback, v, a, n)
                independent.verify_deltas(v, independent.independent_deltas(v))


if __name__ == '__main__':
    unittest.main(verbosity=2)
