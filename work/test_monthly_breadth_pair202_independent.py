"""Synthetic-only independent breadth/excess pairing tests; no market inputs."""
import copy
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_breadth_pair202_independent as independent


def fixture(count=3, width=5):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    frames = {s: np.exp(.0001 * (j - 3) * x + .09 * np.sin(x / (13. + j)))
              for j, s in enumerate(independent.SECTORS)}
    frames.update(A=np.exp(.0004 * x + .13 * np.cos(x / 17.)),
                  B=np.exp(.0002 * x + .10 * np.sin(x / 27.)),
                  ASML=np.exp(.0001 * x + .12 * np.sin(x / 8.)),
                  XLC=np.exp(.0002 * x + .05 * np.cos(x / 23.)),
                  FUND=np.ones(len(index)), UNKNOWN=np.ones(len(index)),
                  XLRE=np.full(len(index), np.nan))
    sectors = pd.DataFrame({'A': 'Technology', 'B': 'Communication Services',
                           'ASML': 'Energy', 'FUND': 'Technology', 'UNKNOWN': 'Unknown'}, index=index)
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    names = ['A', 'B', 'ASML', 'FUND', 'UNKNOWN'][:width]
    return {'adj': pd.DataFrame(frames, index=index), 'sectors': sectors}, dict(
        dates=dates, symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def mapper(day, symbols, sectors, etfs):
    mapping = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Energy': 'XLE'}
    return {s: None if s in etfs else 'XLK' if s == 'ASML' else
            mapping.get(sectors.at[pd.Timestamp(day), s]) if s in sectors.columns and
            pd.Timestamp(day) in sectors.index else None for s in symbols}


def scalar_fixture(m=8):
    points = np.arange(253, dtype=float)
    stock = np.exp(.002 * points + .03 * np.sin(points / 9.))
    benchmark = np.exp(.001 * points + .02 * np.cos(points / 13.))
    breadth = np.column_stack([np.exp(.001 * (j - 3) * points + .1 * np.sin(points / (8 + j)))
                               for j in range(m)])
    return stock, benchmark, breadth


class ScalarTests(unittest.TestCase):
    def test_hand_covariance_pairing_and_matched_mass(self):
        e, w = [1., -1.] * 6, [1., 0.] * 6
        a = independent.paired_summary(e, w)
        b = independent.paired_summary(e, w[::-1])
        self.assertEqual((a['control'], b['control']), (0., 0.))
        self.assertEqual((a['candidate'], b['candidate']), (.5, -.5))
        self.assertEqual(math.fsum(1 + x - a['mean_breadth'] for x in w), 12.)
        self.assertEqual(a['covariance'], .5)
        self.assertIsNone(a['constant_branch'])

    def test_constant_identity_exact_and_no_near_constant_rule(self):
        a = independent.paired_summary([1., -1.] * 6, [1 / 9] * 12)
        self.assertEqual(a['constant_branch'], 'all_breadth_equal')
        self.assertEqual(a['candidate'].hex(), a['control'].hex())
        a = independent.paired_summary([.1] * 12, [1., 0.] * 6)
        self.assertEqual(a['constant_branch'], 'all_excess_equal')
        self.assertEqual(a['candidate'].hex(), a['control'].hex())
        e = [.1] * 12; e[1] = float(np.nextafter(.1, np.inf))
        a = independent.paired_summary(e, [1., 0.] * 6)
        self.assertIsNone(a['constant_branch'])
        a = independent.paired_summary([0.] * 12, [0.] * 12)
        self.assertEqual(a['constant_branch'], 'all_breadth_equal')

    def test_scalar_windows_strict_down_ties_and_counts(self):
        s, b, p = scalar_fixture()
        p[:22, 0] = 1.; p[21, 0] = np.nextafter(1., 0.)
        p[:22, 1] = 1.
        a = independent.measure_oracle(s, b, p)
        self.assertEqual(a['endpoint_indices'], list(range(0, 253, 21)))
        self.assertEqual(a['stock_log_gains'][0], math.log(s[21]) - math.log(s[0]))
        self.assertEqual(a['excess_log_returns'][-1],
                         (math.log(s[252]) - math.log(s[231])) - (math.log(b[252]) - math.log(b[231])))
        self.assertEqual(a['down_counts'][0], sum(p[21, j] < p[0, j] for j in range(8)))
        self.assertEqual(a['valid_price_observations'], 2530)
        self.assertEqual(a['down_sign_comparisons'], 96)
        self.assertEqual((a['covariance_estimates'], a['raw_score_observations']), (1, 2))

    def test_complete_support_and_source_reason_precedence(self):
        for source, name in enumerate(('stock', 'benchmark', 'breadth')):
            for bad in (0., -1., np.nan, np.inf):
                args = list(scalar_fixture())
                if source == 2: args[source][11, 3] = bad
                else: args[source][11] = bad
                a = independent.measure_oracle(*args)
                self.assertFalse(a['available']); self.assertEqual(a['reason'], 'invalid_' + name + '_price')
                self.assertEqual(a['endpoint_indices'], [])
                self.assertTrue(all(a[k] == 0 for k in independent.COUNTERS))
            for bad in (True, None, '1', complex(1, 0)):
                args = [x.astype(object) for x in scalar_fixture()]
                if source == 2: args[source][11, 3] = bad
                else: args[source][11] = bad
                self.assertEqual(independent.measure_oracle(*args)['reason'], 'nonnumeric_' + name + '_price')
        with self.assertRaises(ValueError): independent.measure_oracle(np.ones(252), np.ones(253), np.ones((253, 8)))
        with self.assertRaises(ValueError): independent.measure_oracle(np.ones(253), np.ones(253), np.ones((253, 7)))

    def test_extremes_no_ratio_overflow_and_scale_evidence(self):
        s, b, p = scalar_fixture()
        before = tuple(x.copy() for x in (s, b, p))
        a = independent.measure_oracle(s, b, p)
        for scale in (2. ** -400, 2. ** 400):
            z = independent.measure_oracle(s * scale, b * scale, p * scale)
            self.assertEqual(a['down_counts'], z['down_counts'])
            for key in ('candidate', 'control', 'covariance'):
                self.assertAlmostEqual(a[key], z[key], delta=1e-12)
        for x, y in zip(before, (s, b, p)): np.testing.assert_array_equal(x, y)
        s = np.ones(253); s[0] = 5e-324; s[21] = 1e308
        z = independent.measure_oracle(s, np.ones(253), np.ones((253, 8)))
        self.assertTrue(z['available']); self.assertTrue(math.isfinite(z['candidate']))
        with patch.object(independent.math, 'log', return_value=np.inf):
            with self.assertRaises(FloatingPointError): independent.measure_oracle(*scalar_fixture())


class BuildTests(unittest.TestCase):
    def test_mapping_baskets_all_counts_and_dates(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        v, a = independent.independent_features(p, m, c, mapper)
        r = independent.verify_features(p, m, c, mapper, v, a)
        self.assertTrue(all(r[k] is True for k in independent.FEATURE_FLAGS))
        self.assertEqual(r['available_by_event'], [3] * 3)
        self.assertEqual(r['eight_sector_cells_by_event'], [2] * 3)
        self.assertEqual(r['nine_sector_cells_by_event'], [1] * 3)
        self.assertEqual(r['valid_price_observations'], 3 * 253 * 31)
        self.assertEqual(r['down_sign_comparisons'], 3 * 12 * 25)
        self.assertEqual(r['excess_block_observations'], 9 * 12)
        for row in a['event_history']:
            t = c.index(row['date'])
            self.assertEqual((row['price_start'], row['price_end']), (c[t - 294], c[t - 42]))
            self.assertEqual(row['endpoint_dates'], c[t - 294:t - 41:21])
            self.assertNotIn('XLK', row['cells'][0]['breadth_etfs'])
            self.assertEqual(row['cells'][1]['breadth_etfs'], list(independent.SECTORS))
            self.assertEqual(row['cells'][2]['benchmark'], 'XLK')
        for key in p: pd.testing.assert_frame_equal(p[key], before[key])

    def test_unused_sources_neutral_etfs_and_constant_singleton(self):
        p, m, c = fixture()
        base, audit = independent.independent_features(p, m, c, mapper)
        p['adj']['XLRE'] = 'unconsumed'
        p['adj']['FUND'] = 'unused etf candidate'
        p['adj']['UNKNOWN'] = 'unmapped candidate'
        independent.verify_features(p, m, c, mapper, base, audit)
        p, m, c = fixture(width=1)
        v, _ = independent.independent_features(p, m, c, mapper)
        self.assertTrue(v['available'].all()); self.assertTrue((v['candidate_rank'] == 0).all())
        p['adj'].loc[:, :] = 1.
        v, a = independent.independent_features(p, m, c, mapper)
        self.assertTrue(v['available'].all()); self.assertTrue((v['candidate'] == 0).all())
        self.assertTrue(all(row['cells'][0]['constant_branch'] == 'all_breadth_equal' for row in a['event_history']))

    def test_missing_and_invalid_consumed_sources_and_nonendpoint_session(self):
        for column, reason in (('A', 'missing_symbol'), ('XLC', 'missing_benchmark_symbol'),
                               ('XLB', 'missing_breadth_symbol')):
            p, m, c = fixture(); p['adj'] = p['adj'].drop(columns=column)
            v, a = independent.independent_features(p, m, c, mapper)
            cell = a['event_history'][0]['cells'][1 if column == 'XLC' else 0]
            self.assertEqual(cell['reason'], reason); self.assertFalse(cell['available'])
        p, m, c = fixture(); t = c.index(m['dates'][0]); bad = c[t - 294 + 11]
        p['adj'].at[pd.Timestamp(bad), 'XLK'] = np.nan
        v, a = independent.independent_features(p, m, c, mapper)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'invalid_benchmark_price')
        self.assertEqual(a['event_history'][0]['cells'][1]['reason'], 'invalid_breadth_price')
        p, m, c = fixture(); p['adj'] = p['adj'].drop(pd.Timestamp(bad))
        v, a = independent.independent_features(p, m, c, mapper)
        self.assertEqual(a['event_history'][0]['missing_official_sessions'], [bad])
        self.assertEqual(a['available_by_event'][0], 0)

    def test_mapping_clock_foreign_precedence_and_false_callback_rejected(self):
        p, m, c = fixture()
        observed = []
        def checked(day, symbols, sectors, etfs):
            self.assertLessEqual(sectors.index.max(), pd.Timestamp(day))
            observed.append(day)
            return mapper(day, symbols, sectors, etfs)
        independent.independent_features(p, m, c, checked)
        self.assertEqual(observed, m['dates'])
        p['sectors'].loc[pd.Timestamp(m['dates'][1]):, 'A'] = 'Energy'
        _, a = independent.independent_features(p, m, c, mapper)
        self.assertEqual([r['benchmarks']['A'] for r in a['event_history']], ['XLK', 'XLE', 'XLE'])
        def wrong(day, symbols, sectors, etfs):
            q = mapper(day, symbols, sectors, etfs); q['A'] = 'XLF'; return q
        with self.assertRaises(ValueError): independent.independent_features(p, m, c, wrong)

    def test_all_prefixes_future_poison_and_late42_irrelevance(self):
        p, m, c = fixture()
        full, full_audit = independent.independent_features(p, m, c, mapper)
        for n in range(4):
            q = copy.deepcopy(p)
            for key in q:
                q[key] = q[key].loc[:m['dates'][n - 1]] if n else q[key].iloc[:0]
            values, audit = independent.independent_features(q, m, c, mapper, n)
            independent.verify_features(q, m, c, mapper, values, audit, n)
            for key in full: np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(audit['event_history'], full_audit['event_history'][:n])
        q = copy.deepcopy(p)
        q['adj'] = q['adj'].astype(object)
        q['adj'].loc[q['adj'].index > pd.Timestamp(m['dates'][0]), :] = 'future poison'
        independent.verify_features(q, m, c, mapper, {k: v[:1] for k, v in full.items()},
                                    independent.independent_features(p, m, c, mapper, 1)[1], 1)
        q = copy.deepcopy(p); t = c.index(m['dates'][0])
        q['adj'] = q['adj'].astype(object); q['adj'].loc[c[t - 41]:m['dates'][0], :] = 'recent unused'
        z, _ = independent.independent_features(q, m, c, mapper, 1)
        for key in full: np.testing.assert_array_equal(z[key], full[key][:1])

    def test_schema_calendar_history_and_empty_prefix(self):
        p, m, c = fixture()
        z, a = independent.independent_features(p, m, c, mapper, 0)
        self.assertEqual(z['available'].shape, (0, 5)); self.assertEqual(a['checked_cells'], 0)
        short = c[c.index('2016-01-01'):]
        z, a = independent.independent_features(p, m, short, mapper)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        for limit in (True, -1, 4, 1.5):
            with self.assertRaises(ValueError): independent.independent_features(p, m, c, mapper, limit)
        bad = copy.deepcopy(m); bad['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p, bad, c, mapper)
        with self.assertRaises(ValueError): independent.independent_features(p, m, c[:c.index(m['dates'][-1]) + 1], mapper)
        with self.assertRaises(ValueError): independent.independent_features(p, {k: v for k, v in m.items() if k != 'etfs'}, c, mapper)

    def test_audit_array_and_rank_corruption_rejected(self):
        p, m, c = fixture(); v, a = independent.independent_features(p, m, c, mapper)
        for mutate in (lambda z: z['event_history'][0]['cells'][0]['down_counts'].__setitem__(0, 99),
                       lambda z: z.__setitem__('valid_price_observations', 0),
                       lambda z: z['event_history'][0]['cells'][1]['breadth_etfs'].pop(),
                       lambda z: z['event_history'][0]['cells'][0].__setitem__('covariance', 99.)):
            bad = copy.deepcopy(a); mutate(bad)
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, mapper, v, bad)
        for key in ('candidate', 'control', 'candidate_rank', 'control_rank'):
            bad = {k: x.copy() for k, x in v.items()}; bad[key][0, 0] += .001
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, mapper, bad, a)
        raw = np.array([[2., 2., 6., np.nan], [3., 3., 3., np.nan]])
        rank = independent.centered_ranks(raw, np.isfinite(raw))
        np.testing.assert_array_equal(rank, [[-.25, -.25, .5, 0.], [0., 0., 0., 0.]])

    def test_deltas_exact_initial_zero_neutral_mask_and_bounds(self):
        p, m, c = fixture(); v, _ = independent.independent_features(p, m, c, mapper)
        delta = independent.independent_deltas(v)
        r = independent.verify_deltas(v, delta)
        self.assertEqual(r['max_absolute_delta'], .005)
        for arm, raw in independent.RAW.items():
            self.assertTrue((delta[arm][0] == 0).all())
            self.assertTrue((delta[arm][~v['available']] == 0).all())
            np.testing.assert_array_equal(delta[arm][1:], .01 * v[raw + '_rank'][1:])
            wrong = {k: x.copy() for k, x in delta.items()}; wrong[arm][1, 0] += 1e-16
            with self.assertRaises(ValueError): independent.verify_deltas(v, wrong)
        with self.assertRaises(ValueError): independent.verify_deltas(v, delta, .02)

    def test_pinned_mapping_accounting_no_production_imports(self):
        code = inspect.getsource(independent)
        self.assertNotIn('import monthly_breadth_pair202_kernel', code)
        self.assertTrue(independent.FROZEN_HELPER in independent.dependency_files())
        account = independent.accounting()
        self.assertTrue(callable(account.audit_path))
        with patch.object(independent, 'FROZEN_HELPER_SHA', '0' * 64):
            with self.assertRaises(ValueError): independent.frozen_helper()

    def test_both_production_fixtures_full_audit_all_prefixes(self):
        import monthly_breadth_pair202_kernel as production
        import test_monthly_breadth_pair202_kernel as production_test
        for fixture_fn, callback in ((fixture, mapper), (production_test.fixture, production_test.mapper)):
            p, m, c = fixture_fn()
            for n in range(len(m['dates']) + 1):
                q = {k: x.loc[:m['dates'][n - 1]].copy() if n else x.iloc[:0].copy() for k, x in p.items()}
                values, audit = production.build_features(q, m, c, callback, n)
                independent.verify_features(q, m, c, callback, values, audit, n)
                independent.verify_deltas(values, independent.independent_deltas(values))


if __name__ == '__main__':
    unittest.main(verbosity=2)
