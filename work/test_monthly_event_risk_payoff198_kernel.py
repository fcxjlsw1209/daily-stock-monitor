"""Synthetic-only causal event-risk and matched-mass payoff validation."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_event_risk_payoff198_kernel as kernel


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04')
    x = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({'A': 100.+.03*x+3.*np.sin(x*.13),
        'B': 200.+.02*x+5.*np.cos(x*.23), 'FUND': 80.+.02*x+4.*np.sin(x*.37)}, index=dates)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
        symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


def single_event(anchor=64):
    source = np.ones(253)*100.
    for i in range(1, 253):
        source[i] = source[i-1]*(1.+(.001 if i%2 else .003))
    source[anchor] = source[anchor-1]*.9
    for i in range(anchor+1, 253):
        source[i] = source[i-1]*1.002
    return source


class AggregateTests(unittest.TestCase):
    def test_pairing_vs_identical_inverse_mass(self):
        paired, inverse, standardized = kernel.aggregate_payoffs([1., 2.], [2., -1.])
        swapped, _, _ = kernel.aggregate_payoffs([2., 1.], [2., -1.])
        denominator = 22*math.sqrt(21)
        self.assertEqual(inverse, [1., .5])
        self.assertEqual(standardized, [2., -.5])
        self.assertEqual(paired['candidate'], 1.5/denominator)
        self.assertEqual(paired['control'], .75/denominator)
        self.assertEqual(swapped['control'], paired['control'])
        self.assertEqual(swapped['candidate'], 0.)
        covariance = ((2.-.5)*(1.-.75)+(-1.-.5)*(.5-.75))/2
        self.assertAlmostEqual(paired['candidate']-paired['control'], 2/22/math.sqrt(21)*covariance)

    def test_exact_identity_branches_and_near_equality_not_snapped(self):
        cases = [([.2], [-.3], 'single_event'), ([.2, .2], [.3, -.1], 'equal_risks'),
                 ([.2, .4], [.25, .25], 'equal_payoffs'), ([.2, .4], [0., 0.], 'equal_payoffs')]
        for risks, labels, branch in cases:
            result, _, _ = kernel.aggregate_payoffs(risks, labels)
            self.assertEqual(result['identity_branch'], branch)
            self.assertEqual(result['candidate'].hex(), result['control'].hex())
        result, _, _ = kernel.aggregate_payoffs([.2, np.nextafter(.2, 1.)], [.3, -.1])
        self.assertIsNone(result['identity_branch'])

    def test_zero_risk_preserves_all_labels_but_invalidates_both(self):
        result, inverse, standardized = kernel.aggregate_payoffs([0., .2], [.3, -.1])
        self.assertFalse(result['available'])
        self.assertTrue(np.isnan(result['candidate']) and np.isnan(result['control']))
        self.assertEqual(result['payoff_sum'], math.fsum([.3, -.1]))
        self.assertEqual(inverse, [None, 5.])
        self.assertEqual(standardized, [None, -.5])
        self.assertIsNone(result['inverse_risk_sum'])

    def test_invalid_arithmetic_hard_fails(self):
        for risks, labels in [([1e-320], [1.]), ([1.], [np.inf]), ([np.inf], [1.]),
                              ([1e-308, 1e-308], [1., 1.])]:
            with self.assertRaises(FloatingPointError):
                kernel.aggregate_payoffs(risks, labels)
        with self.assertRaises(ValueError):
            kernel.aggregate_payoffs([], [])
        with self.assertRaises(ValueError):
            kernel.aggregate_payoffs([-1.], [1.])


class MeasureTests(unittest.TestCase):
    def test_first_anchor_risk_before_loss_and_label(self):
        source = single_event()
        result = kernel.measure(source)
        self.assertTrue(result['available'])
        self.assertEqual(result['anchor_indices'], [64])
        event = result['anchor_events'][0]
        self.assertEqual((event['risk_price_start_index'], event['risk_price_end_index']), (0, 63))
        self.assertEqual((event['risk_return_first_index'], event['risk_return_last_index']), (1, 63))
        expected = source[1:64]/source[:63]-1.
        np.testing.assert_array_equal(event['pre_event_simple_returns'], expected)
        self.assertEqual(event['sigma'], float(np.std(expected, ddof=1)))
        self.assertEqual(event['payoff'], math.log(source[85])-math.log(source[64]))
        self.assertEqual(event['maturity_index'], 85)
        self.assertEqual(result['candidate'], result['control'])
        changed = source.copy()
        changed[64] *= .7
        other = kernel.measure(changed)
        self.assertEqual(other['anchor_events'][0]['sigma'], event['sigma'])
        self.assertNotEqual(other['anchor_events'][0]['payoff'], event['payoff'])

    def test_last_anchor_matures_at252(self):
        result = kernel.measure(single_event(231))
        self.assertEqual(result['anchor_indices'], [231])
        event = result['anchor_events'][0]
        self.assertEqual((event['risk_price_start_index'], event['risk_price_end_index']), (167, 230))
        self.assertEqual(event['maturity_index'], 252)
        self.assertEqual(result['selected_anchors'], result['valid_event_payoffs'])

    def test_common_greedy_cooldown_and_eight_maximum(self):
        source = np.arange(100., 353.)
        for s in (64, 65, 84, 85, 106, 127, 148, 169, 190, 211, 232):
            source[s] = source[s-1]*.9
        result = kernel.measure(source)
        self.assertEqual(result['anchor_indices'], list(range(64, 212, 21)))
        self.assertEqual(result['selected_anchors'], 8)
        self.assertEqual(len(result['anchor_events']), 8)
        self.assertTrue(all(e['maturity_index'] <= 252 for e in result['anchor_events']))

    def test_no_events_missing_and_negative_payoffs_not_filtered(self):
        for source in (np.ones(253), np.arange(1., 254.)):
            result = kernel.measure(source)
            self.assertTrue(result['source_available'])
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'no_selected_anchors')
            self.assertEqual(result['anchor_events'], [])
        losses = kernel.measure(np.arange(1000., 747., -1.))
        self.assertTrue(losses['available'])
        self.assertEqual(losses['selected_anchors'], 8)
        self.assertLess(losses['candidate'], 0.)
        self.assertLess(losses['control'], 0.)
        self.assertTrue(all(e['payoff'] < 0 for e in losses['anchor_events']))

    def test_zero_sigma_invalidates_whole_cell_without_event_deletion(self):
        source = np.ones(253)*100.
        source[64] = 90.
        source[65:] = np.arange(100., 288.)
        source[85] = source[84]*.9
        result = kernel.measure(source)
        self.assertEqual(result['anchor_indices'], [64, 85])
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], 'zero_pre_event_sigma')
        self.assertEqual(result['selected_anchors'], 2)
        self.assertEqual(result['valid_event_payoffs'], 0)
        self.assertEqual(result['anchor_events'][0]['sigma'], 0.)
        self.assertGreater(result['anchor_events'][1]['sigma'], 0.)
        self.assertIsNone(result['anchor_events'][0]['inverse_sigma'])
        self.assertTrue(all(math.isfinite(e['payoff']) for e in result['anchor_events']))
        self.assertIsNotNone(result['anchor_events'][1]['inverse_sigma'])

    def test_finite_price_arithmetic_overflow_raises_even_after_zero_sigma(self):
        source = single_event()
        source[1], source[2] = 1e-300, 1e300
        with self.assertRaises(FloatingPointError):
            kernel.measure(source)
        source = np.ones(253)*100.
        source[64], source[70], source[71], source[85] = 90., 1e-300, 1e300, 90.
        with self.assertRaises(FloatingPointError):
            kernel.measure(source)
        source = single_event()
        source[1], source[2] = 1., 1e200  # Finite return; its squared deviation overflows.
        with self.assertRaises(FloatingPointError):
            kernel.measure(source)

    def test_power_two_scale_invariance_and_source_immutability(self):
        source = single_event()
        before = source.copy()
        result = kernel.measure(source)
        for scale in (2.**-500, 2.**500):
            other = kernel.measure(source*scale)
            self.assertEqual(other['anchor_indices'], result['anchor_indices'])
            self.assertEqual(other['anchor_events'][0]['sigma'], result['anchor_events'][0]['sigma'])
            self.assertAlmostEqual(other['candidate'], result['candidate'], delta=1e-10)
        np.testing.assert_array_equal(source, before)

    def test_full253_invalid_source_and_shapes(self):
        for value, reason in [(0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                (np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price'),
                (True, 'nonnumeric_adjusted_price'), (None, 'nonnumeric_adjusted_price'),
                ('2', 'nonnumeric_adjusted_price'), (1+0j, 'nonnumeric_adjusted_price')]:
            source = list(single_event())
            source[250] = value
            result = kernel.measure(source)
            self.assertFalse(result['source_available'] or result['available'])
            self.assertEqual(result['reason'], reason)
            self.assertEqual(result['selected_anchors'], 0)
        for source in (np.ones(252), np.ones((253, 1)), 1.):
            with self.assertRaises(ValueError):
                kernel.measure(source)


class BuilderTests(unittest.TestCase):
    def test_schema_counts_windows_and_initial_raw(self):
        panel, meta, calendar = fixture()
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(set(values), {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'})
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [3]*3)
        self.assertEqual(audit['valid_price_observations'], 253*9)
        self.assertEqual(audit['pre_event_return_observations'], 63*audit['selected_anchors'])
        self.assertEqual(audit['matured_payoffs'], audit['selected_anchors'])
        self.assertEqual(audit['selected_anchors'], sum(audit['selected_anchors_by_event']))
        self.assertEqual(audit['valid_event_payoffs'], sum(audit['valid_event_payoffs_by_event']))
        self.assertEqual(audit['new_model_fits'], 0)
        self.assertTrue(np.isfinite(values['candidate'][0]).all())
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual((event['price_start'], event['price_end']), (calendar[t-294], calendar[t-42]))
            self.assertEqual((event['anchor_first_date'], event['anchor_last_date']), (calendar[t-230], calendar[t-63]))
            self.assertTrue(event['cells'][2]['available'])

    def test_singleton_common_rank_and_equal_ties(self):
        panel, meta, calendar = fixture()
        panel['adj']['B'] = panel['adj']['FUND'] = 100.
        values, audit = kernel.build_features(panel, meta, calendar)
        self.assertEqual(audit['source_available_by_event'], [3]*3)
        self.assertEqual(audit['available_by_event'], [1]*3)
        self.assertTrue(np.isfinite(values['candidate'][:, 0]).all())
        self.assertTrue(np.isnan(values['candidate'][:, 1:]).all())
        np.testing.assert_array_equal(values['candidate_rank'], np.zeros((3, 3)))
        np.testing.assert_array_equal(kernel.centered_rank(np.array([1., 2., 2., 4., np.nan]),
            np.array([True]*4+[False])), [-.5, 0., 0., .5, 0.])

    def test_source_missing_and_structural_priorities(self):
        panel, meta, calendar = fixture()
        t = calendar.index(meta['dates'][0])
        panel['adj'].loc[calendar[t-294+17], 'B'] = np.nan
        values, audit = kernel.build_features(panel, meta, calendar, limit=1)
        self.assertEqual(audit['source_available_by_event'], [2])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'invalid_adjusted_price')
        panel['adj'] = panel['adj'].drop(pd.Timestamp(calendar[t-294+17])).drop(columns='B')
        _, audit = kernel.build_features(panel, meta, calendar, limit=1)
        self.assertEqual(audit['reason_counts'], {'missing_official_session': 2, 'missing_symbol': 1})
        short_meta = dict(dates=['2016-08-31'], symbols=[['UNKNOWN']], etfs=[])
        _, audit = kernel.build_features(panel, short_meta, calendar[-200:])
        self.assertEqual(audit['reason_counts'], {'insufficient_history': 1})

    def test_all_prefixes_truncated_sources_and_future_poison(self):
        panel, meta, calendar = fixture()
        full, audit = kernel.build_features(panel, meta, calendar)
        for n in range(4):
            cut = {'adj': panel['adj'].iloc[:0].copy() if n == 0 else panel['adj'].loc[:meta['dates'][n-1]].copy()}
            small, check = kernel.build_features(cut, meta, calendar, limit=n)
            for key in full:
                np.testing.assert_array_equal(small[key], full[key][:n])
            self.assertEqual(check['event_history'], audit['event_history'][:n])
            for key in ('source_available', 'available', 'selected_anchors', 'valid_event_payoffs'):
                self.assertEqual(check[key+'_by_event'], audit[key+'_by_event'][:n])
        t = calendar.index(meta['dates'][0])
        poison = {'adj': panel['adj'].astype(object)}
        poison['adj'].loc[calendar[t-41]:, :] = 'future'
        poison['adj']['UNUSED'] = 'never requested'
        small, check = kernel.build_features(poison, meta, calendar, limit=1)
        self.assertEqual(check['event_history'], audit['event_history'][:1])
        for key in full:
            np.testing.assert_array_equal(small[key], full[key][:1])

    def test_permutation_source_immutability_and_validation(self):
        panel, meta, calendar = fixture()
        before, oldmeta, oldcal = panel['adj'].copy(deep=True), copy.deepcopy(meta), list(calendar)
        full, _ = kernel.build_features(panel, meta, calendar)
        order = [2, 0, 1]
        permuted = dict(meta, symbols=[[r[j] for j in order] for r in meta['symbols']])
        other, _ = kernel.build_features(panel, permuted, calendar)
        for key in full:
            np.testing.assert_array_equal(other[key], full[key][:, order])
        pd.testing.assert_frame_equal(panel['adj'], before)
        self.assertEqual(meta, oldmeta)
        self.assertEqual(calendar, oldcal)
        for limit in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                kernel.build_features(panel, meta, calendar, limit=limit)
        with self.assertRaises(ValueError):
            kernel.build_features(dict(panel, volume=panel['adj']), meta, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features({'adj': panel['adj'].iloc[::-1]}, meta, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, meta, calendar[:calendar.index('2016-09-30')+1])
        with self.assertRaises(ValueError):
            kernel.build_features(panel, dict(meta, etfs=['FUND', 'FUND']), calendar)


if __name__ == '__main__':
    unittest.main(verbosity=2)
