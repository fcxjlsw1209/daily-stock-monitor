"""Pure synthetic validation of the frozen prospective round196 definition."""
import copy
import unittest

import numpy as np
import pandas as pd

import monthly_pairwise_majority196_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    adj = pd.DataFrame({'A': 100.*np.exp(.001*x+.03*np.sin(x*.1)),
        'B': 200.*np.exp(.0008*x+.02*np.cos(x*.13)),
        'FUND': 80.*np.exp(.0007*x+.04*np.sin(x*.05))}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
        symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'adj': adj}, meta, dates.strftime('%Y-%m-%d').tolist()


def endpoints(patterns):
    """Three names with exact powers-of-two block gross returns."""
    rows = [[1.] for _ in range(3)]
    for order in patterns:
        for j, symbol in enumerate('ABC'):
            rows[j].append(rows[j][-1]*2.**(2-order.index(symbol)))
    return np.array(rows)


class TournamentTests(unittest.TestCase):
    def test_preflight_equal_balance_different_majorities(self):
        prices = endpoints(['ABC']*4+['ACB']+['BAC']*2+['BCA']*3+['CAB']*2)
        result = kernel.tournament(prices)
        self.assertEqual(result['pair_margin_matrix'], [[0, 2, 2], [-2, 0, 6], [-2, -6, 0]])
        np.testing.assert_array_equal(result['candidate'], [1., 0., -1.])
        np.testing.assert_array_equal(result['control'], [1/6, 1/6, -1/3])
        np.testing.assert_array_equal(kernel.centered_rank(result['candidate'], np.ones(3, bool)), [.5, 0, -.5])
        np.testing.assert_array_equal(kernel.centered_rank(result['control'], np.ones(3, bool)), [.25, .25, -.5])

    def test_majority_cycle_and_block_order_invariance(self):
        result = kernel.tournament(endpoints(['ABC']*4+['BCA']*4+['CAB']*4))
        self.assertEqual(result['pair_margin_matrix'], [[0, 4, -4], [-4, 0, 4], [4, -4, 0]])
        np.testing.assert_array_equal(result['candidate'], [0., 0., 0.])
        np.testing.assert_array_equal(result['control'], [0., 0., 0.])
        base = kernel.tournament(endpoints(['ABC']*7+['BCA']*5))
        other = kernel.tournament(endpoints(['BCA']*5+['ABC']*7))
        self.assertEqual(base['pair_margin_matrix'], other['pair_margin_matrix'])

    def test_exact_ties_and_even_vote_ties(self):
        tied = kernel.tournament(np.ones((4, 13)))
        np.testing.assert_array_equal(tied['candidate'], np.zeros(4))
        self.assertTrue(all(p['block_signs'] == [0]*12 for p in tied['pairs']))
        split = kernel.tournament(endpoints(['ABC']*6+['CBA']*6))
        self.assertEqual(split['pair_margin_matrix'], [[0]*3 for _ in range(3)])

    def test_exact_extreme_products_and_near_rounding_ties(self):
        prices = np.array([[2.**900 if k%2 else 2.**-900 for k in range(13)],
                           [2.**-900 if k%2 else 2.**900 for k in range(13)]])
        result = kernel.tournament(prices)
        self.assertEqual(result['pairs'][0]['block_signs'], [1, -1]*6)
        # Both floating divisions round to the same value, but exact gross returns differ.
        x = np.nextafter(1., 2.)
        first = [1., x]*6+[1.]
        second = [x, np.nextafter(x, 2.)]*6+[x]
        exact = kernel.tournament(np.array([first, second]))
        self.assertEqual(exact['pairs'][0]['block_signs'], [1, -1]*6)

    def test_per_security_power_two_scale_invariance(self):
        base = endpoints(['ABC']*7+['BCA']*5)
        original = kernel.tournament(base)
        scaled = kernel.tournament(base*np.array([2.**800, 2.**-800, 2.**300])[:, None])
        self.assertEqual(original['pairs'], scaled['pairs'])
        np.testing.assert_array_equal(original['candidate'], scaled['candidate'])

    def test_permutation_and_small_domains(self):
        base = endpoints(['ABC']*7+['BCA']*5)
        original = kernel.tournament(base)
        order = [2, 0, 1]
        other = kernel.tournament(base[order])
        np.testing.assert_array_equal(other['candidate'], original['candidate'][order])
        np.testing.assert_array_equal(other['pair_margin_matrix'], np.array(original['pair_margin_matrix'])[np.ix_(order, order)])
        for n in (0, 1):
            small = kernel.tournament(np.ones((n, 13)))
            self.assertEqual(small['pairs'], [])
            self.assertTrue(np.isnan(small['candidate']).all())
            self.assertEqual(small['pair_margin_matrix'], [] if n == 0 else [[0]])

    def test_bad_shapes_and_endpoints_raise(self):
        for prices in (np.ones(13), np.ones((2, 12)), np.zeros((2, 13)), np.full((2, 13), np.inf)):
            with self.assertRaises(ValueError):
                kernel.tournament(prices)
        for prices in (np.full((2, 13), True), np.full((2, 13), '2')):
            with self.assertRaises(ValueError):
                kernel.tournament(prices)

    def test_every_price_validated_and_no_mutation(self):
        source = np.ones(253)
        before = source.copy()
        result, reason = kernel.source_prices(source)
        self.assertEqual(reason, 'available')
        np.testing.assert_array_equal(result, source)
        np.testing.assert_array_equal(source, before)
        for value, expected in [(0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                (np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price'),
                (True, 'nonnumeric_adjusted_price'), (None, 'nonnumeric_adjusted_price'),
                ('3', 'nonnumeric_adjusted_price'), (1+0j, 'nonnumeric_adjusted_price')]:
            array = [1.]*253
            array[17] = value
            parsed, reason = kernel.source_prices(array)
            self.assertIsNone(parsed)
            self.assertEqual(reason, expected)
        for array in (np.ones(252), np.ones((253, 1)), 1.):
            with self.assertRaises(ValueError):
                kernel.source_prices(array)


class RankTests(unittest.TestCase):
    def test_average_ties_common_mask_and_small_domains(self):
        raw = np.array([1., 2., 2., 4., np.nan])
        valid = np.array([True]*4+[False])
        np.testing.assert_array_equal(kernel.centered_rank(raw, valid), [-.5, 0., 0., .5, 0.])
        for mask in ([False, False], [True, False], [True, True]):
            np.testing.assert_array_equal(kernel.centered_rank(np.ones(2), np.array(mask)), [0., 0.])

    def test_mask_shape_numeric_invalid(self):
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.ones(3), np.ones(3, int))
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.array([np.nan]), np.array([True]))
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.ones((2, 2)), np.ones((2, 2), bool))


class BuilderTests(unittest.TestCase):
    def test_schema_counts_official_endpoints_and_initial_raw(self):
        panel, meta, calendar = fixture()
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(set(values), {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'})
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['source_available_cells'], 9)
        self.assertEqual(audit['valid_price_observations'], 253*9)
        self.assertEqual(audit['pair_majority_decisions'], 9)
        self.assertEqual(audit['block_comparisons'], 108)
        self.assertEqual(audit['new_model_fits'], 0)
        self.assertTrue(values['available'].all())
        self.assertTrue(np.isfinite(values['candidate'][0]).all())
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual(event['price_start'], calendar[t-294])
            self.assertEqual(event['price_end'], calendar[t-42])
            self.assertEqual(event['endpoint_dates'], [calendar[t-294+21*k] for k in range(13)])
            self.assertEqual(event['source_indices'], [0, 1, 2])
            self.assertEqual(event['source_available'], [True]*3)
            self.assertTrue(event['cells'][2]['available'])  # ETFs participate.

    def test_source_domain_separate_from_feature_singleton(self):
        panel, meta, calendar = fixture()
        panel['adj'] = panel['adj'][['A']]
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(audit['source_available_by_event'], [1]*3)
        self.assertEqual(audit['source_available_cells'], 3)
        self.assertEqual(audit['available_cells'], 0)
        self.assertEqual(audit['valid_price_observations'], 759)
        self.assertEqual(audit['pair_majority_decisions'], 0)
        self.assertTrue(np.isnan(values['candidate']).all())
        np.testing.assert_array_equal(values['candidate_rank'], np.zeros((3, 3)))
        for event in audit['event_history']:
            self.assertEqual(event['pair_margin_matrix'], [[0]])
            self.assertEqual(event['cells'][0]['reason'], 'insufficient_comparison_domain')
            self.assertEqual(event['cells'][0]['source_reason'], 'available')
            self.assertEqual(event['cells'][1]['reason'], 'missing_symbol')

    def test_missing_official_row_and_intermediate_bad_price(self):
        panel, meta, calendar = fixture()
        t = calendar.index(meta['dates'][0])
        broken = copy.deepcopy(panel)
        broken['adj'] = broken['adj'].drop(pd.Timestamp(calendar[t-294+17]))
        _, audit = kernel.build_features(broken, meta, calendar, limit=1)
        self.assertEqual(audit['source_available_cells'], 0)
        self.assertEqual(audit['reason_counts'], {'missing_official_session': 3})
        panel['adj'].loc[calendar[t-294+17], 'B'] = np.nan
        values, audit = kernel.build_features(panel, meta, calendar, limit=1)
        self.assertEqual(audit['source_available_by_event'], [2])
        np.testing.assert_array_equal(values['available'], [[True, False, True]])
        event = audit['event_history'][0]
        self.assertEqual(event['source_indices'], [0, 2])
        self.assertEqual(event['pairs'][0]['right_index'], 2)
        self.assertEqual(event['cells'][1]['reason'], 'invalid_adjusted_price')

    def test_insufficient_history_priority_and_zero_domain(self):
        panel, meta, calendar = fixture()
        calendar = calendar[-200:]
        meta = dict(dates=['2016-08-31'], symbols=[['UNKNOWN', 'B']], etfs=[])
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 2})
        self.assertEqual(audit['event_history'][0]['pair_margin_matrix'], [])
        self.assertEqual(audit['source_available_by_event'], [0])
        self.assertFalse(values['available'].any())

    def test_all_prefixes_truncated_sources_and_future_poison(self):
        panel, meta, calendar = fixture()
        full, full_audit = kernel.build_features(panel, meta, calendar)
        for n in range(4):
            cut = {'adj': panel['adj'].iloc[:0].copy() if n == 0 else panel['adj'].loc[:meta['dates'][n-1]].copy()}
            values, audit = kernel.build_features(cut, meta, calendar, limit=n)
            for key in full:
                np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(audit['event_history'], full_audit['event_history'][:n])
            self.assertEqual(audit['source_available_by_event'], full_audit['source_available_by_event'][:n])
        t = calendar.index(meta['dates'][0])
        poisoned = {'adj': panel['adj'].astype(object)}
        poisoned['adj'].loc[calendar[t-41]:, :] = 'future poison'
        poisoned['adj']['FUTURE_ONLY'] = 'never requested'
        values, audit = kernel.build_features(poisoned, meta, calendar, limit=1)
        for key in full:
            np.testing.assert_array_equal(values[key], full[key][:1])
        self.assertEqual(audit['event_history'], full_audit['event_history'][:1])

    def test_permutation_and_source_immutability(self):
        panel, meta, calendar = fixture()
        original_frame = panel['adj'].copy(deep=True)
        original_meta = copy.deepcopy(meta)
        original_calendar = list(calendar)
        values, _ = kernel.build_features(panel, meta, calendar)
        order = [2, 0, 1]
        permuted = copy.deepcopy(meta)
        permuted['symbols'] = [[row[j] for j in order] for row in meta['symbols']]
        other, _ = kernel.build_features(panel, permuted, calendar)
        for key in values:
            np.testing.assert_array_equal(other[key], values[key][:, order])
        pd.testing.assert_frame_equal(panel['adj'], original_frame)
        self.assertEqual(meta, original_meta)
        self.assertEqual(calendar, original_calendar)

    def test_metadata_calendar_panel_validation(self):
        panel, meta, calendar = fixture()
        for changes in ({'etfs': ['A', 'A']}, {'dates': ['2016-08-18']},
                        {'symbols': [['A'], ['A', 'B'], ['A']]}):
            other = dict(meta, **changes)
            with self.assertRaises(ValueError):
                kernel.build_features(panel, other, calendar)
        for limit in (-1, 4, True, .5):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, meta, calendar, limit=limit)
        with self.assertRaises(ValueError):
            kernel.build_features(dict(panel, volume=panel['adj']), meta, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, meta, calendar[:-1]+[calendar[-1], calendar[-1]])
        with self.assertRaises(ValueError):
            kernel.build_features({'adj': panel['adj'].iloc[::-1]}, meta, calendar)
        missing_next = calendar[:calendar.index('2016-09-30')+1]
        with self.assertRaises(ValueError):
            kernel.build_features(panel, meta, missing_next)


if __name__ == '__main__':
    unittest.main(verbosity=2)
