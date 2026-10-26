"""Synthetic-only independent tournament proofs and full producer audit checks."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_pairwise_majority196_independent as independent


def endpoint_paths(exponents):
    blocks = np.asarray(exponents, dtype=int)
    return (2. ** np.vstack((np.zeros(blocks.shape[1], int), np.cumsum(blocks, axis=0)))).T.tolist()


def daily_prices(endpoints):
    return np.array([endpoints[min(k // 21, 12)] for k in range(253)], float)


def hand_blocks():
    # A/B have equal pooled margin but different net opponent majorities.
    return [[3, 2, 1, 1]] * 5 + [[1, 3, 2, 2]] * 3 + [[2, 1, 3, 3]] * 4


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({'A': np.exp(.0004 * x),
        'B': np.exp(.0002 * x + .04 * np.sin(x / 43.)),
        'FUND': np.exp(.0001 * x + .07 * np.cos(x / 57.)), 'D': np.full(len(index), 8.)}, index=index)
    names = ['A', 'B', 'FUND', 'D'][:width]
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': frame}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


class ScalarTests(unittest.TestCase):
    def test_equal_borda_different_copeland_hand_counts(self):
        q, f, margins, blocks = independent.tournament_oracle(endpoint_paths(hand_blocks()))
        self.assertEqual(margins, [[0, 6, -2, -2], [-6, 0, 4, 4], [2, -4, 0, 0], [2, -4, 0, 0]])
        self.assertEqual(f[0], f[1]); self.assertEqual(f[0], 2 / 36)
        self.assertEqual(q, [-1 / 3, 1 / 3, 0., 0.])
        self.assertEqual(blocks[(0, 1)], [1] * 5 + [-1] * 3 + [1] * 4)
        self.assertEqual(blocks[(1, 0)], [-x for x in blocks[(0, 1)]])
        self.assertEqual(blocks[(2, 3)], [0] * 12)

    def test_majority_cycle_and_zero_outputs(self):
        paths = endpoint_paths([[3, 2, 1]] * 4 + [[1, 3, 2]] * 4 + [[2, 1, 3]] * 4)
        q, f, m, _ = independent.tournament_oracle(paths)
        self.assertEqual(m, [[0, 4, -4], [-4, 0, 4], [4, -4, 0]])
        self.assertEqual(q, [0., 0., 0.]); self.assertEqual(f, [0., 0., 0.])

    def test_time_permutation_and_independent_unit_scaling(self):
        blocks = hand_blocks(); paths = endpoint_paths(blocks)
        q, f, matrix, _ = independent.tournament_oracle(paths)
        shuffled = blocks[::2] + blocks[1::2]
        qq, ff, mm, _ = independent.tournament_oracle(endpoint_paths(shuffled))
        self.assertEqual((qq, ff, mm), (q, f, matrix))
        scaled = [[x * 2. ** (200 * (j - 2)) for x in row] for j, row in enumerate(paths)]
        qq, ff, mm, _ = independent.tournament_oracle(scaled)
        self.assertEqual((qq, ff, mm), (q, f, matrix))
        common_shift = [[x + 1 for x in row] for row in blocks]
        qq, ff, mm, _ = independent.tournament_oracle(endpoint_paths(common_shift))
        self.assertEqual((qq, ff, mm), (q, f, matrix))

    def test_integer_products_avoid_overflow_underflow_and_near_ties(self):
        lo = 1e308; hi = float(np.nextafter(lo, math.inf))
        self.assertTrue(math.isinf(lo * lo))
        self.assertEqual(independent.compare_ratios(hi, lo, lo, lo), 1)
        self.assertEqual(independent.compare_ratios(lo, lo, hi, lo), -1)
        self.assertEqual(independent.compare_ratios(hi, lo, hi, lo), 0)
        self.assertEqual(independent.compare_ratios(5e-324, 1e308, 5e-324, 1e308), 0)
        self.assertEqual(independent.compare_ratios(1e308, 5e-324, 1., 1.), 1)
        self.assertEqual(independent.compare_ratios(5e-324, 1e308, 1., 1.), -1)
        paths = [[1e308 if k % 2 else 5e-324 for k in range(13)], [1.] * 13]
        q, f, m, _ = independent.tournament_oracle(paths)
        self.assertEqual((q, f, m), ([0., 0.], [0., 0.], [[0, 0], [0, 0]]))
        for bad in (0., -1., np.nan, np.inf, True, '1'):
            with self.assertRaises(ValueError): independent.compare_ratios(bad, 1., 1., 1.)

    def test_all253_prices_required_not_only_endpoints(self):
        base = np.ones(253)
        for position in (0, 1, 20, 21, 251, 252):
            for value in (0., -1., np.nan, np.inf):
                p = base.copy(); p[position] = value
                self.assertEqual(independent.source_oracle(p), ('invalid_adjusted_price', []))
        for value in (True, None, '1', complex(1, 0)):
            p = base.astype(object); p[30] = value
            self.assertEqual(independent.source_oracle(p), ('nonnumeric_adjusted_price', []))
        self.assertEqual(independent.source_oracle(base), ('available', [1.] * 13))
        with self.assertRaises(ValueError): independent.source_oracle(base[:-1])

    def test_empty_singleton_ties_and_two_names(self):
        q, f, m, signs = independent.tournament_oracle([])
        self.assertEqual((q, f, m, signs), ([], [], [], {}))
        q, f, m, signs = independent.tournament_oracle([[1.] * 13])
        self.assertEqual((q, f, m, signs), ([None], [None], [[0]], {}))
        q, f, m, _ = independent.tournament_oracle([[1.] * 13] * 3)
        self.assertEqual((q, f), ([0.] * 3, [0.] * 3))
        self.assertEqual(m, [[0] * 3 for _ in range(3)])
        q, f, m, _ = independent.tournament_oracle(endpoint_paths([[1, 0]] * 12))
        self.assertEqual((q, f, m), ([1., -1.], [1., -1.], [[0, 12], [-12, 0]]))


class BuildTests(unittest.TestCase):
    def test_full_audit_counts_dates_and_unfiltered_etf(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        values, audit = independent.independent_features(p, m, c)
        receipt = independent.verify_features(p, m, c, values, audit)
        self.assertEqual(receipt['source_available_by_event'], [4, 4, 4])
        self.assertEqual((receipt['source_available_cells'], receipt['available_cells'], receipt['checked_cells']), (12, 12, 12))
        self.assertEqual((receipt['valid_price_observations'], receipt['block_comparisons'], receipt['pair_majority_decisions']), (3036, 216, 18))
        self.assertFalse(receipt['computes_returns'])
        self.assertTrue(values['available'][:, 2].all())
        for row in audit['event_history']:
            t = c.index(row['date'])
            self.assertEqual((row['price_start'], row['price_end']), (c[t - 294], c[t - 42]))
            self.assertEqual(row['endpoint_dates'], [c[t - 294 + 21 * k] for k in range(13)])
            self.assertEqual(row['source_indices'], [0, 1, 2, 3])
            for pair in row['pairs']:
                a, b = pair['left_index'], pair['right_index']
                self.assertEqual(sum(pair['block_signs']), row['pair_margin_matrix'][a][b])
                self.assertEqual(row['pair_margin_matrix'][a][b], -row['pair_margin_matrix'][b][a])
        pd.testing.assert_frame_equal(p['adj'], before['adj'])

    def test_actual_cells_match_hand_tournament_and_ranks(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0]); dates = c[t - 294:t - 41]
        for name, path in zip(m['symbols'][0], endpoint_paths(hand_blocks())):
            p['adj'].loc[dates, name] = daily_prices(path)
        values, audit = independent.independent_features(p, m, c)
        independent.verify_features(p, m, c, values, audit)
        np.testing.assert_array_equal(values['candidate_rank'][0], [-.5, .5, 0., 0.])
        np.testing.assert_array_equal(values['control_rank'][0], [2.5 / 3 - .5, 2.5 / 3 - .5, .5 / 3 - .5, .5 / 3 - .5])
        self.assertEqual(audit['event_history'][0]['cells'][0]['majority_sum'], -1)
        self.assertEqual(audit['event_history'][0]['cells'][1]['margin_sum'], 2)

    def test_n_less_two_retains_source_but_neutral_features(self):
        p, m, c = fixture(count=1, width=1)
        values, audit = independent.independent_features(p, m, c)
        independent.verify_features(p, m, c, values, audit)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['source_available_cells'], 1)
        self.assertEqual(audit['valid_price_observations'], 253)
        self.assertEqual((audit['block_comparisons'], audit['pair_majority_decisions']), (0, 0))
        row = audit['event_history'][0]
        self.assertEqual(row['pair_margin_matrix'], [[0]])
        self.assertEqual(row['cells'][0]['reason'], 'insufficient_comparison_domain')
        self.assertEqual(row['cells'][0]['source_reason'], 'available')
        self.assertEqual(len(row['cells'][0]['endpoint_prices']), 13)
        p['adj'] = p['adj'].drop(columns='A')
        values, audit = independent.independent_features(p, m, c)
        self.assertEqual(audit['source_available_cells'], 0)
        self.assertEqual(audit['event_history'][0]['pair_margin_matrix'], [])

    def test_invalid_member_common_domain_not_pair_specific_deletion(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        p['adj'].loc[c[t - 293], 'B'] = 0.
        values, audit = independent.independent_features(p, m, c)
        self.assertEqual(values['available'][0].tolist(), [True, False, True, True])
        row = audit['event_history'][0]
        self.assertEqual(row['source_indices'], [0, 2, 3])
        self.assertEqual([(x['left_index'], x['right_index']) for x in row['pairs']], [(0, 2), (0, 3), (2, 3)])
        self.assertEqual((audit['valid_price_observations'], audit['block_comparisons']), (759, 36))
        self.assertEqual(row['cells'][1]['reason'], 'invalid_adjusted_price')
        self.assertEqual(row['cells'][1]['candidate_rank'], 0.)
        for cell in (row['cells'][0], row['cells'][2], row['cells'][3]): self.assertEqual(cell['opponent_count'], 2)

    def test_missing_official_session_history_and_failure_priority(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        p['adj'] = p['adj'].drop(pd.Timestamp(c[t - 293])).drop(columns='A')
        values, audit = independent.independent_features(p, m, c)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reason_counts'], {'missing_symbol': 1, 'missing_official_session': 3})
        self.assertEqual(audit['source_available_cells'], 0)
        _, audit = independent.independent_features(p, m, [x for x in c if x >= '2016-01-01'])
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 4})

    def test_all_prefixes_source_truncation_and_future_poison(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for n in range(4):
            q = {'adj': p['adj'].astype(object).copy()}
            stop = pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n - 1]) - 42])
            q['adj'].loc[q['adj'].index > stop, :] = 'unread future'
            vv, aa = independent.independent_features(q, m, c, limit=n)
            independent.verify_features(q, m, c, vv, aa, limit=n)
            for key in values: np.testing.assert_array_equal(vv[key], values[key][:n])
            self.assertEqual(aa['event_history'], audit['event_history'][:n])
            q['adj'] = q['adj'].loc[:stop]
            v2, a2 = independent.independent_features(q, m, c, limit=n)
            for key in vv: np.testing.assert_array_equal(v2[key], vv[key])
            self.assertEqual(a2, aa)
        self.assertEqual(independent.independent_features(p, m, c, 0)[0]['available'].shape, (0, 4))

    def test_query_schema_and_nextsession_checks(self):
        p, m, c = fixture()
        for bad in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError): independent.independent_features(p, m, c, bad)
        q = copy.deepcopy(m); q['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        with self.assertRaises(ValueError): independent.independent_features(p, m, c[:c.index(m['dates'][-1]) + 1])
        with self.assertRaises(ValueError): independent.independent_features({'volume': p['adj']}, m, c)
        q = dict(m); q.pop('etfs')
        with self.assertRaises(ValueError): independent.independent_features(p, q, c)
        q = {'adj': p['adj'].iloc[::-1]}
        with self.assertRaises(ValueError): independent.independent_features(q, m, c)

    def test_exact_corrupt_comparison_margin_raw_rank_counts_rejected(self):
        p, m, c = fixture(); values, audit = independent.independent_features(p, m, c)
        for kind in ('raw', 'rank', 'mask', 'margin', 'block', 'source', 'counter', 'date', 'type'):
            v, a = copy.deepcopy(values), copy.deepcopy(audit)
            if kind == 'raw': v['control'][0, 0] += 1e-10
            elif kind == 'rank': v['candidate_rank'][0, 0] += .001
            elif kind == 'mask': v['available'][0, 0] = False
            elif kind == 'margin': a['event_history'][0]['pair_margin_matrix'][0][1] += 1
            elif kind == 'block': a['event_history'][0]['pairs'][0]['block_signs'][0] *= -1
            elif kind == 'source': a['source_available_by_event'][0] -= 1
            elif kind == 'counter': a['block_comparisons'] -= 1
            elif kind == 'date': a['event_history'][0]['endpoint_dates'][1] = a['event_history'][0]['endpoint_dates'][0]
            else: a['checked_cells'] = float(a['checked_cells'])
            with self.assertRaises(ValueError): independent.verify_features(p, m, c, v, a)

    def test_centered_ranks_ties_noop_and_no_eligibility_filter(self):
        raw = np.array([[1., 0., 0., -1.], [0., 0., 0., 0.]])
        np.testing.assert_array_equal(independent.centered_ranks(raw, np.ones_like(raw, bool)), [[.5, 0, 0, -.5], [0, 0, 0, 0]])
        p, m, c = fixture(); m['eligibility'] = 'must not be consumed'
        p['adj'].loc[:, :] = 1.
        values, audit = independent.independent_features(p, m, c)
        self.assertTrue(values['available'].all()); self.assertTrue((values['candidate_rank'] == 0).all())
        self.assertTrue((values['control_rank'] == 0).all())
        independent.verify_features(p, m, c, values, audit)

    def test_fixed_deltas_initial_row_and_bad_artifacts(self):
        p, m, c = fixture(); values, _ = independent.independent_features(p, m, c)
        deltas = independent.independent_deltas(values)
        receipt = independent.verify_deltas(values, deltas)
        self.assertEqual(receipt['max_absolute_delta'], .005)
        for arm, raw in independent.RAW.items():
            np.testing.assert_array_equal(deltas[arm][0], np.zeros(4))
            np.testing.assert_array_equal(deltas[arm][1:], .01 * values[raw + '_rank'][1:])
        for place in ((0, 0), (1, 1)):
            d = copy.deepcopy(deltas); d['tournament_majority'][place] += .0001
            with self.assertRaises(ValueError): independent.verify_deltas(values, d)
        with self.assertRaises(ValueError): independent.verify_deltas(values, deltas, .02)
        v = copy.deepcopy(values); v['candidate_rank'][1] *= -1
        with self.assertRaises(ValueError): independent.independent_deltas(v)

    def test_frozen_accounting_dependency_and_no_production_oracle_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(), independent.FROZEN_HELPER_SHA)
        self.assertIn(independent.FROZEN_HELPER, independent.dependency_files())
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_pairwise_majority196_kernel', inspect.getsource(independent))
        with patch.object(independent, 'FROZEN_HELPER_SHA', '0' * 64):
            with self.assertRaises(ValueError): independent.accounting()

    def test_producer_full_audit_both_fixtures_and_prefixes(self):
        import monthly_pairwise_majority196_kernel as producer
        import test_monthly_pairwise_majority196_kernel as producer_tests
        for p, m, c in (fixture(), producer_tests.fixture()):
            for n in range(len(m['dates']) + 1):
                stop = pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n - 1])
                q = {'adj': p['adj'].loc[:stop]}
                values, audit = producer.build_features(q, m, c, limit=n)
                independent.verify_features(q, m, c, values, audit, limit=n)
                independent.verify_deltas(values, independent.independent_deltas(values))
        p, m, c = fixture(count=1)
        for n in (0, 1, 2):
            q = {'adj': p['adj'][m['symbols'][0][:n]]}
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)
        t = c.index(m['dates'][0])
        for bad in (0., np.nan, True, 'bad'):
            q = {'adj': p['adj'].astype(object).copy()}
            q['adj'].at[pd.Timestamp(c[t - 293]), 'A'] = bad
            values, audit = producer.build_features(q, m, c)
            independent.verify_features(q, m, c, values, audit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
