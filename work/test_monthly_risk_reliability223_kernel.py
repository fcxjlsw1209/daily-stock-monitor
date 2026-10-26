"""Synthetic annual clock, risk-error arithmetic and immutable-prefix contracts."""
from copy import deepcopy
from unittest import mock
import math
import unittest

import numpy as np
import pandas as pd
import monthly_risk_reliability223_kernel as k


def prices(returns, start=64.):
    out = [float(start)]
    for r in returns:
        out.append(out[-1]*(1.+r))
    return np.asarray(out)


def fixture():
    dates = pd.bdate_range('2010-08-02', '2019-02-05')
    t = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({'A': 100*np.exp(.0002*t+.035*np.sin(t/7)),
        'B': 70*np.exp(.0004*t+.025*np.sin(t/11)+.01*np.cos(t/3)),
        'FLAT': np.ones(len(dates))*100.}, index=dates)
    query = ['2016-08-17', '2016-08-31', '2017-01-31', '2017-12-29', '2018-12-31']
    meta = {'dates': query, 'symbols': [['A', 'B', 'FLAT'] for _ in query], 'etfs': ['B']}
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


def first_query():
    panel, meta, calendar = fixture()
    return panel, {**meta, 'dates': meta['dates'][:1], 'symbols': meta['symbols'][:1]}, calendar


def group(audit, year=2016, symbol='A'):
    return next(g for g in audit['group_history'] if g['year'] == year and g['symbol'] == symbol)


class ArithmeticTests(unittest.TestCase):
    def test_analytic_noncentered_second_moments_and_squared_errors(self):
        f = prices([1., -.5, 0.]*21)
        p = prices([1., -.5])
        r = prices([0., 1., -.5, 0.])
        actual = k.measure(f, p, r)
        self.assertTrue(actual['available'])
        self.assertEqual(actual['forecast_second_moment'], 1.25/3)
        self.assertEqual(actual['prior_second_moment'], 1.25/2)
        self.assertEqual(actual['realized_second_moment'], 1.25/4)
        e = ((1.25/4-1.25/3)/(1.25/4+1.25/3))
        self.assertEqual(actual['candidate_error'], e*e)
        self.assertAlmostEqual(actual['control_error'], 1/9, places=16)

    def test_zero_denominator_either_leg_excludes_both(self):
        f = np.full(64, 64.); p = prices([1., -.5]); r = np.full(5, 64.)
        actual = k.measure(f, p, r)
        self.assertTrue(actual['source_available']); self.assertFalse(actual['available'])
        self.assertEqual(actual['reason'], 'zero_pair_denominator')
        self.assertEqual(actual['forecast_second_moment'], 0.)
        self.assertGreater(actual['prior_second_moment'], 0.)
        self.assertIsNone(actual['candidate_error']); self.assertIsNone(actual['control_error'])
        other = k.measure(prices([1., -.5, 0.]*21), np.ones(4), np.ones(5))
        self.assertFalse(other['available'])

    def test_zero_forecast_with_positive_realized_is_valid(self):
        actual = k.measure(np.ones(64), np.ones(4), prices([1., -.5]))
        self.assertTrue(actual['available'])
        self.assertEqual(actual['candidate_error'], 1.); self.assertEqual(actual['control_error'], 1.)

    def test_all_zero_pair_is_source_valid(self):
        actual = k.measure(np.ones(64), np.ones(4), np.ones(5))
        self.assertTrue(actual['source_available']); self.assertFalse(actual['available'])
        for key in ('forecast_second_moment', 'prior_second_moment', 'realized_second_moment'):
            self.assertEqual(actual[key], 0.)

    def test_matching_forecasts_have_zero_error(self):
        actual = k.measure(prices([1., -.5, 0.]*21), prices([1., -.5, 0.]*2), prices([1., -.5, 0.]))
        self.assertEqual(actual['candidate_error'], 0.); self.assertEqual(actual['control_error'], 0.)

    def test_binary_scale_invariance(self):
        args = (prices([1., -.5, 0.]*21), prices([1., -.5]), prices([0., 1., -.5, 0.]))
        self.assertEqual(k.measure(*args), k.measure(args[0]*2., args[1]*4., args[2]*8.))

    def test_invalid_prices_joint_missing_and_priority(self):
        for position, label in enumerate(('forecast', 'prior', 'realized')):
            for value in (0., -1., np.nan, np.inf, True, '64', None):
                args = [list(np.ones(64)), list(np.ones(4)), list(np.ones(5))]
                args[position][1] = value; actual = k.measure(*args)
                self.assertFalse(actual['source_available']); self.assertFalse(actual['available'])
                self.assertIn(label, actual['reason'])
                for key in k.PAIR_FLOAT_FIELDS:self.assertIsNone(actual[key])

    def test_all_shapes_checked_before_invalid_values(self):
        bad = [None]*64
        for args in ((np.ones(63), np.ones(3), np.ones(3)),
                     (bad, np.ones(1), np.ones(3)), (bad, np.ones(3), np.ones((3,1)))):
            with self.assertRaises(ValueError):k.measure(*args)

    def test_overflow_and_underflow_hard_fail(self):
        p = np.ones(64); p[0] = 1e-150; p[1] = 1e100
        with self.assertRaises(FloatingPointError):k.measure(p, np.ones(4), np.ones(5))
        with self.assertRaises(FloatingPointError):k._square(1e-200)
        with self.assertRaises(FloatingPointError):k._negative_mean([float(np.nextafter(0., 1.))]+[0.]*59)
        with self.assertRaises(FloatingPointError):k.error(1e308, 1e308)

    def test_month_equal_calibration_is_not_session_weighted(self):
        self.assertEqual(k._negative_mean([.25, 1.]), -.625)
        self.assertEqual(k._negative_mean([0.]*18), 0.)

    def test_original_percentile_convention_ties_and_singleton(self):
        actual = k.percentile_rank(np.array([-.1, -.1, -.9, np.nan]), np.array([1,1,1,0], bool))
        np.testing.assert_array_equal(actual[:3], [2.5/3-.5, 2.5/3-.5, 1/3-.5])
        self.assertTrue(np.isnan(actual[3]))
        self.assertEqual(k.percentile_rank(np.array([-.5]), np.array([True]))[0], .5)


class ClockAndGroupTests(unittest.TestCase):
    def test_exact_sixty_month_clocks_counts_and_annual_reuse(self):
        panel, meta, calendar = fixture(); values, audit = k.build_features(panel, meta, calendar)
        self.assertEqual(set(values), set(k.FEATURE_KEYS)); self.assertEqual(values['available'].shape, (5,3))
        self.assertEqual(audit['year_security_groups'], 9); self.assertEqual(audit['historical_month_pairs'], 540)
        self.assertEqual(audit['supported_groups'], 6); self.assertEqual(audit['cold_start_groups'], 3)
        self.assertEqual(audit['calibration_estimates'], 12); self.assertEqual(audit['new_model_fits'], 12)
        self.assertEqual(audit['predictive_model_fits'], 12); self.assertEqual(audit['new_ridge_fits'], 0)
        self.assertEqual(audit['available_by_event'], [2]*5)
        np.testing.assert_array_equal(values['raw_candidate'][0], values['raw_candidate'][1])
        g = group(audit); self.assertEqual(g['cutoff'], '2015-12-31')
        self.assertEqual([p['month'] for p in g['pairs']], pd.period_range('2011-01','2015-12',freq='M').astype(str).tolist())
        pair = g['pairs'][0]
        self.assertEqual(pair['forecast_date'], '2010-12-31')
        self.assertEqual(pair['prior_price_start'], '2010-11-30')
        self.assertEqual(pair['realized_price_start'], '2010-12-31')
        self.assertEqual(pair['realized_price_end'], '2011-01-31')
        self.assertEqual(audit['second_moment_estimates'], 3*audit['source_available_pairs'])
        self.assertEqual(audit['error_evaluations'], 2*audit['available_pairs'])
        self.assertEqual(audit['valid_price_observations'], audit['simple_return_observations']+3*audit['source_available_pairs'])

    def test_future_and_same_year_prices_cannot_change_frozen_profile(self):
        panel, meta, calendar = first_query(); before, audit = k.build_features(panel, meta, calendar)
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].loc['2016-01-01':, :] = 'future poison'
        after, other = k.build_features(panel, meta, calendar)
        for key in before:np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(audit, other)

    def test_exact18_support_threshold_no_missing_month_backfill(self):
        for count in (17, 18):
            panel, meta, calendar = first_query(); _, initial = k.build_features(panel, meta, calendar)
            first = group(initial)['pairs'][-count]['forecast_price_start']
            panel['adj'].loc[panel['adj'].index < first, 'A'] = np.nan
            values, audit = k.build_features(panel, meta, calendar)
            g = group(audit); self.assertEqual(g['available_pairs'], count)
            self.assertEqual(g['status'], 'supported' if count==18 else 'cold_start')
            self.assertEqual(bool(values['available'][0,0]), count==18)
            self.assertEqual(len(g['pairs']), 60)
            self.assertEqual(g['history_start_month'], '2011-01')

    def test_pre_window_source_values_cannot_change_profile(self):
        panel, meta, calendar = first_query(); before, audit = k.build_features(panel, meta, calendar)
        start = min(p['forecast_price_start'] for p in group(audit)['pairs'])
        panel['adj'].loc[panel['adj'].index < start, :] = np.nan
        after, other = k.build_features(panel, meta, calendar)
        for key in before:np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(audit, other)

    def test_source_missing_price_is_pair_invalid_not_calendar_failure(self):
        panel, meta, calendar = first_query(); panel['adj'] = panel['adj'].drop(pd.Timestamp('2013-04-15'))
        _, audit = k.build_features(panel, meta, calendar)
        self.assertGreater(sum(n for reason,n in audit['pair_reason_counts'].items() if reason.startswith('missing_')), 0)
        self.assertEqual(len(group(audit)['pairs']), 60)

    def test_calendar_missing_month_or_warmup_is_hard_error(self):
        panel, meta, calendar = first_query()
        incomplete = [d for d in calendar if not d.startswith('2013-04')]
        with self.assertRaises(ValueError):k.build_features(panel, meta, incomplete)
        with self.assertRaises(ValueError):k.build_features(panel, meta, [d for d in calendar if d >= '2010-12-01'])

    def test_missing_symbol_no_alias_or_membership_substitution(self):
        panel, meta, calendar = first_query(); meta['symbols'][0][0] = 'A_ALIAS'
        values, audit = k.build_features(panel, meta, calendar)
        self.assertFalse(values['available'][0,0])
        g = group(audit, symbol='A_ALIAS'); self.assertEqual(g['pair_reason_counts'], {'missing_symbol':60})
        self.assertTrue(values['available'][0,1])

    def test_all_historical_prices_used_despite_absent_prior_query_membership(self):
        panel, meta, calendar = fixture()
        meta['symbols'][2] = ['NEW', 'B', 'FLAT']; panel['adj']['NEW'] = panel['adj']['A']
        values, audit = k.build_features(panel, meta, calendar)
        self.assertTrue(values['available'][2,0]); self.assertEqual(group(audit,2017,'NEW')['available_pairs'],60)
        self.assertEqual(audit['history_membership_filtered'], False)

    def test_unique_profile_computed_once_per_year_security(self):
        panel, meta, calendar = fixture()
        with mock.patch.object(k, '_group', wraps=k._group) as spy:
            k.build_features(panel, meta, calendar)
        self.assertEqual(spy.call_count, 9)

    def test_every_logical_and_physical_prefix_group_identity(self):
        panel, meta, calendar = fixture(); values, audit = k.build_features(panel, meta, calendar)
        groups = {g['group_id']:g for g in audit['group_history']}
        for n in range(6):
            sm = {**meta, 'dates':meta['dates'][:n], 'symbols':meta['symbols'][:n]}
            frame = panel['adj'].loc[:meta['dates'][n-1]] if n else panel['adj'].iloc[:0]
            physical, pa = k.build_features({'adj':frame}, sm, calendar)
            logical, la = k.build_features(panel, meta, calendar, limit=n)
            self.assertEqual(pa['event_history'], audit['event_history'][:n])
            self.assertEqual(la['event_history'], audit['event_history'][:n])
            for g in pa['group_history']:self.assertEqual(g, groups[g['group_id']])
            self.assertEqual(pa['group_history'],la['group_history'])
            for key in values:
                np.testing.assert_array_equal(logical[key],values[key][:n])
                if n:np.testing.assert_array_equal(physical[key],values[key][:n])
                else:self.assertEqual(physical[key].shape,(0,0))

    def test_all_zero_history_is_source_complete_but_unavailable(self):
        panel, meta, calendar = first_query()
        values, audit = k.build_features(panel, meta, calendar)
        self.assertFalse(values['available'][0,2]); self.assertTrue(np.isnan(values['raw_candidate'][0,2]))
        g = group(audit,symbol='FLAT'); self.assertEqual(g['source_available_pairs'],60)
        self.assertEqual(g['available_pairs'],0); self.assertEqual(g['pair_reason_counts'],{'zero_pair_denominator':60})

    def test_no_mutation_invalid_limits_and_initial_partial(self):
        panel, meta, calendar = first_query(); before=panel['adj'].copy(); original=deepcopy(meta)
        k.build_features(panel,meta,calendar)
        pd.testing.assert_frame_equal(before,panel['adj']);self.assertEqual(meta,original)
        for n in (-1,True,2):
            with self.assertRaises(ValueError):k.build_features(panel,meta,calendar,limit=n)
        meta['dates'][0]='2016-08-18'
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar)


if __name__ == '__main__':unittest.main(verbosity=2)
