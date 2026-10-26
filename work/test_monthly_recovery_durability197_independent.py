"""Synthetic-only independent ordered recovery/survival tests."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_recovery_durability197_independent as independent


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({'A': np.exp(.0001 * x + .03 * np.sin(x / 11.)),
        'B': np.exp(-.0003 * x), 'FUND': 5. + np.sin(x / 19.), 'D': np.full(len(index), 8.)}, index=index)
    names = ['A', 'B', 'FUND', 'D'][:width]
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': frame}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def event(result, index):
    return next(x for x in result['anchor_events'] if x['anchor_index'] == index)


class ScalarTests(unittest.TestCase):
    def test_same_recovery_and_endpoint_but_different_survival(self):
        a = independent.measure_oracle([100, 90, 100, 105, 101] + [105] * 248)
        b = independent.measure_oracle([100, 90, 100, 105, 99] + [105] * 248)
        self.assertEqual([x['anchor_index'] for x in a['anchor_events']], [1, 4])
        self.assertEqual([x['anchor_index'] for x in b['anchor_events']], [1, 4])
        self.assertEqual([x['recovery_lag'] for x in a['anchor_events']], [1, 1])
        self.assertEqual([x['recovery_lag'] for x in b['anchor_events']], [1, 1])
        self.assertEqual((a['control'], b['control']), (1., 1.))
        self.assertEqual((a['candidate'], b['candidate']), (1., .5))
        self.assertEqual((event(a, 1)['post_minimum'], event(b, 1)['post_minimum']), (101., 99.))

    def test_early_late_equal_boundary_and_equal_post_exposure(self):
        early = np.full(253, 100.); early[1] = 90.
        late = np.full(253, 100.); late[1:22] = 90.
        a, b = independent.measure_oracle(early), independent.measure_oracle(late)
        self.assertEqual((a['candidate'], b['candidate']), (1., 1.))
        e, l = event(a, 1), event(b, 1)
        self.assertEqual((e['recovery_lag'], l['recovery_lag']), (1, 21))
        self.assertEqual((e['post_start_index'], e['post_end_index']), (3, 23))
        self.assertEqual((l['post_start_index'], l['post_end_index']), (23, 43))
        self.assertEqual(e['post_end_index'] - e['post_start_index'] + 1, 21)
        self.assertEqual(l['post_end_index'] - l['post_start_index'] + 1, 21)
        self.assertEqual(e['post_minimum'], e['reference_price'])
        self.assertTrue(e['durable'])

    def test_final_anchor_maturity_day252_and_day22_not_recovery(self):
        p = np.full(253, 100.); p[210:231] = 90.
        a = independent.measure_oracle(p); e = event(a, 210)
        self.assertEqual(a['negative_anchors'], 1)
        self.assertEqual((e['recovery_index'], e['post_start_index'], e['post_end_index']), (231, 232, 252))
        self.assertEqual((a['candidate'], a['control']), (1., 1.))
        p[252] = float(np.nextafter(100., 0.))
        b = independent.measure_oracle(p)
        self.assertEqual((b['candidate'], b['control']), (0., 1.))
        p = np.full(253, 100.); p[210:232] = 90.
        b = independent.measure_oracle(p)
        self.assertEqual((b['candidate'], b['control']), (0., 0.))
        self.assertIsNone(event(b, 210)['recovery_index'])
        self.assertIsNone(event(b, 210)['post_minimum'])

    def test_first_crossing_not_last_crossing_and_recovery_day_excluded(self):
        p = np.full(253, 105.); p[:4] = [100, 90, 100, 99]
        result = independent.measure_oracle(p); e = event(result, 1)
        self.assertEqual((e['recovery_index'], e['recovery_price']), (2, 100.))
        self.assertEqual(e['post_minimum'], 99.); self.assertFalse(e['durable'])
        p[3] = 105.; e = event(independent.measure_oracle(p), 1)
        self.assertEqual(e['post_minimum'], 105.)
        self.assertNotEqual(e['post_minimum'], e['recovery_price'])

    def test_no_negative_is_missing_no_recovery_is_valid_zero(self):
        for p in (np.ones(253), np.arange(1, 254, dtype=float)):
            r = independent.measure_oracle(p)
            self.assertTrue(r['source_available']); self.assertFalse(r['available'])
            self.assertEqual(r['reason'], 'no_negative_anchors')
            self.assertIsNone(r['candidate']); self.assertEqual(r['anchor_events'], [])
        r = independent.measure_oracle(np.arange(253, 0, -1, dtype=float))
        self.assertTrue(r['source_available']); self.assertTrue(r['available'])
        self.assertEqual((r['candidate'], r['control']), (0., 0.))
        self.assertEqual((r['negative_anchors'], r['recovered_anchors'], r['durable_anchors']), (210, 0, 0))
        self.assertTrue(all(x['post_end_index'] is None for x in r['anchor_events']))
        p = np.ones(253); p[211] = .5
        self.assertEqual(independent.measure_oracle(p)['negative_anchors'], 0)

    def test_strict_binary_comparisons_and_safe_extreme_prices(self):
        p = np.ones(253); p[1] = np.nextafter(1., 0.)
        r = independent.measure_oracle(p)
        self.assertEqual((r['negative_anchors'], r['recovered_anchors'], r['durable_anchors']), (1, 1, 1))
        self.assertEqual(event(r, 1)['recovery_lag'], 1)
        p = np.full(253, 1e308); p[1] = 5e-324
        r = independent.measure_oracle(p)
        self.assertEqual((r['candidate'], r['control']), (1., 1.))
        p = np.array([100, 90, 100, 105, 99] + [105] * 248, float)
        before = p.copy(); a = independent.measure_oracle(p)
        for scale in (2. ** -400, 2. ** 400):
            b = independent.measure_oracle(p * scale)
            for key in ('candidate', 'control', 'negative_anchors', 'recovered_anchors', 'durable_anchors'):
                self.assertEqual(a[key], b[key])
            for x, y in zip(a['anchor_events'], b['anchor_events']):
                for key in ('anchor_index', 'recovery_index', 'recovery_lag', 'post_start_index', 'post_end_index', 'durable'):
                    self.assertEqual(x[key], y[key])
        np.testing.assert_array_equal(p, before)

    def test_all253_support_including_unused_or_post_horizon_days(self):
        for position in (0, 1, 20, 210, 211, 251, 252):
            for bad in (0., -1., np.nan, np.inf):
                p = np.ones(253); p[position] = bad
                r = independent.measure_oracle(p)
                self.assertFalse(r['source_available']); self.assertFalse(r['available'])
                self.assertEqual(r['reason'], 'invalid_adjusted_price')
                self.assertEqual(r['negative_anchors'], 0)
        for bad in (True, None, '1', complex(1, 0)):
            p = np.ones(253, object); p[30] = bad
            self.assertEqual(independent.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError): independent.measure_oracle(np.ones(252))


class BuildTests(unittest.TestCase):
    def test_full_source_and_anchor_audit_count_geometry(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        values, audit = independent.independent_features(p, m, c)
        receipt = independent.verify_features(p, m, c, values, audit)
        self.assertEqual(receipt['source_available_by_event'], [4, 4, 4])
        self.assertEqual(receipt['available_by_event'], [3, 3, 3])
        self.assertEqual(receipt['source_available_cells'], 12)
        self.assertEqual(receipt['available_cells'], 9)
        self.assertEqual(receipt['valid_price_observations'], 3036)
        for key in ('negative_anchors', 'recovered_anchors', 'durable_anchors'):
            self.assertEqual(receipt[key], sum(receipt[key + '_by_event']))
        self.assertTrue(values['available'][:, 2].all())
        for row in audit['event_history']:
            t = c.index(row['date'])
            self.assertEqual((row['price_start'], row['price_end']), (c[t - 294], c[t - 42]))
            self.assertEqual((row['anchor_first_date'], row['anchor_last_date']), (c[t - 293], c[t - 84]))
            for cell in row['cells']:
                for e in cell['anchor_events']:
                    if e['recovered']:
                        self.assertEqual(e['post_end_index'] - e['post_start_index'] + 1, 21)
                        self.assertLessEqual(e['post_end_index'], 252)
        pd.testing.assert_frame_equal(p['adj'], before['adj'])

    def test_singleton_valid_raw_rank_zero_and_empty_source(self):
        p, m, c = fixture(count=1, width=1)
        values, audit = independent.independent_features(p, m, c)
        independent.verify_features(p, m, c, values, audit)
        self.assertTrue(values['available'][0, 0])
        self.assertEqual(values['candidate_rank'][0, 0], 0.)
        self.assertEqual(audit['source_available_cells'], 1)
        self.assertEqual(audit['available_cells'], 1)
        p['adj'] = p['adj'].drop(columns='A')
        values, audit = independent.independent_features(p, m, c)
        self.assertEqual((audit['source_available_cells'], audit['available_cells']), (0, 0))

    def test_missing_source_and_no_negative_reasons_are_distinct(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        p['adj'].loc[c[t - 293], 'B'] = 0.
        values, audit = independent.independent_features(p, m, c)
        self.assertEqual(values['available'][0].tolist(), [True, False, True, False])
        self.assertEqual(audit['event_history'][0]['source_available'], [True, False, True, True])
        cells = audit['event_history'][0]['cells']
        self.assertEqual(cells[1]['reason'], 'invalid_adjusted_price')
        self.assertEqual(cells[3]['reason'], 'no_negative_anchors')
        p['adj'] = p['adj'].drop(pd.Timestamp(c[t - 293])).drop(columns='A')
        _, audit = independent.independent_features(p, m, c)
        self.assertEqual(audit['reason_counts'], {'missing_symbol': 1, 'missing_official_session': 3})
        _, audit = independent.independent_features(p, m, [x for x in c if x >= '2016-01-01'])
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 4})

    def test_all_prefixes_truncation_and_future_poison(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for n in range(4):
            q = {'adj': p['adj'].astype(object).copy()}
            stop = pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n - 1]) - 42])
            q['adj'].loc[q['adj'].index > stop, :] = 'unread future'
            v, a = independent.independent_features(q, m, c, limit=n)
            independent.verify_features(q, m, c, v, a, limit=n)
            for key in values: np.testing.assert_array_equal(v[key], values[key][:n])
            self.assertEqual(a['event_history'], audit['event_history'][:n])
            q['adj'] = q['adj'].loc[:stop]
            vv, aa = independent.independent_features(q, m, c, limit=n)
            for key in v: np.testing.assert_array_equal(vv[key], v[key])
            self.assertEqual(aa, a)
        self.assertEqual(independent.independent_features(p, m, c, 0)[0]['available'].shape, (0, 4))

    def test_official_monthend_and_schema_contract(self):
        p, m, c = fixture()
        for bad in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError): independent.independent_features(p, m, c, bad)
        q = copy.deepcopy(m); q['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        with self.assertRaises(ValueError): independent.independent_features(p, m, c[:c.index(m['dates'][-1]) + 1])
        with self.assertRaises(ValueError): independent.independent_features({'volume': p['adj']}, m, c)
        q = dict(m); q.pop('etfs')
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        with self.assertRaises(ValueError): independent.independent_features({'adj': p['adj'].iloc[::-1]}, m, c)

    def test_corrupt_complete_audit_paths_counts_and_values_rejected(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for kind in ('raw', 'rank', 'mask', 'first', 'survival', 'source', 'counter', 'date', 'type'):
            v, a = copy.deepcopy(values), copy.deepcopy(audit)
            if kind == 'raw': v['control'][0, 0] += 1e-10
            elif kind == 'rank': v['candidate_rank'][0, 0] += .001
            elif kind == 'mask': v['available'][0, 0] = False
            elif kind == 'first': a['event_history'][0]['cells'][0]['anchor_events'][0]['recovery_index'] = 251
            elif kind == 'survival': a['event_history'][0]['cells'][0]['anchor_events'][0]['durable'] ^= True
            elif kind == 'source': a['source_available_by_event'][0] -= 1
            elif kind == 'counter': a['negative_anchors_by_event'][0] -= 1
            elif kind == 'date': a['event_history'][0]['anchor_last_date'] = a['event_history'][0]['price_end']
            else: a['checked_cells'] = float(a['checked_cells'])
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, v, a)

    def test_tie_ranks_bounds_delta_and_initial_only_zero(self):
        raw = np.array([[1., .5, .5, 0.], [0., 0., 0., 0.]])
        np.testing.assert_array_equal(independent.centered_ranks(raw, np.ones_like(raw, bool)), [[.5, 0, 0, -.5], [0, 0, 0, 0]])
        p, m, c = fixture(); m['eligibility'] = 'not consumed'
        values, _ = independent.independent_features(p, m, c)
        deltas = independent.independent_deltas(values)
        result = independent.verify_deltas(values, deltas)
        self.assertEqual(result['max_absolute_delta'], .005)
        for arm, raw in independent.RAW.items():
            np.testing.assert_array_equal(deltas[arm][0], np.zeros(4))
            np.testing.assert_array_equal(deltas[arm][1:], .01 * values[raw + '_rank'][1:])
        self.assertGreater(sum(np.count_nonzero(x) for x in deltas.values()), 0)
        q = copy.deepcopy(deltas); q['recovery_durability'][0, 0] = .001
        with self.assertRaises(ValueError): independent.verify_deltas(values, q)
        with self.assertRaises(ValueError): independent.verify_deltas(values, deltas, .02)
        q = copy.deepcopy(values); q['candidate'][0, 1] = .5
        with self.assertRaises(ValueError): independent.arrays(q)

    def test_frozen_accounting_pin_and_independence(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(), independent.FROZEN_HELPER_SHA)
        self.assertIn(independent.FROZEN_HELPER, independent.dependency_files())
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_recovery_durability197_kernel', inspect.getsource(independent))
        with patch.object(independent, 'FROZEN_HELPER_SHA', '0' * 64):
            with self.assertRaises(ValueError): independent.accounting()

    def test_producer_full_audit_both_fixtures_all_prefixes_and_special_support(self):
        import monthly_recovery_durability197_kernel as producer
        import test_monthly_recovery_durability197_kernel as producer_tests
        for p, m, c in (fixture(), producer_tests.fixture()):
            for n in range(len(m['dates']) + 1):
                stop = pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n - 1])
                q = {'adj': p['adj'].loc[:stop]}
                values, audit = producer.build_features(q, m, c, limit=n)
                independent.verify_features(q, m, c, values, audit, limit=n)
                independent.verify_deltas(values, independent.independent_deltas(values))
        p, m, c = fixture(count=1); t = c.index(m['dates'][0]); dates = c[t - 294:t - 41]
        for path in (np.ones(253), np.arange(253, 0, -1, dtype=float),
                     np.array([100, 90, 100, 105, 99] + [105] * 248, float)):
            q = copy.deepcopy(p)
            for name in q['adj']: q['adj'].loc[dates, name] = path
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)
        for n in (0, 1):
            q = {'adj': p['adj'][m['symbols'][0][:n]]}
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)
        for bad in (0., np.nan, True, 'bad'):
            q = {'adj': p['adj'].astype(object).copy()}; q['adj'].at[pd.Timestamp(c[t - 293]), 'A'] = bad
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
