"""Synthetic-only round206 model tests; no research cache or market data reads."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_identity206', P/'monthly_identity_transfer206_model.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def fixture():
    days = pd.date_range('2013-12-31', '2022-12-31', freq='ME')
    dates = np.repeat(days.values.astype('datetime64[D]'), 4)
    ends = np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 4)
    symbols = np.tile(['A', 'B', 'C', 'D'], len(days))
    rng = np.random.default_rng(206)
    X = rng.uniform(-.5, .5, (len(dates), 10))
    y = .08*X[:, 0]+.19*X[:, 2]**2+np.tile([-.04, .06, -.03, .01], len(days))
    y = (y.reshape(-1, 4)-y.reshape(-1, 4).mean(axis=1, keepdims=True)).reshape(-1)
    targets = {key: y*(i+1) for i, key in enumerate(m.TARGETS)}
    events = np.array(['2016-08-17', '2018-01-31', '2019-01-31',
                       '2019-11-29', '2021-02-26', '2021-12-31'], dtype='datetime64[D]')
    return [dates, ends, symbols, X, targets, np.ones(len(dates), bool), events,
            np.array(['A', 'D', 'B', 'C', 'A', 'D']),
            rng.uniform(-.5, .5, (len(events), 10)), np.ones(len(events), bool)]


def run(args, minimum=18, own_minimum=18):
    return m.annual_forecasts(*args, min_months=minimum, min_own_months=own_minimum)


def prefix(args, n, physical=True):
    a = copy.deepcopy(args)
    for i in (6, 7, 8, 9):
        a[i] = a[i][:n]
    if physical:
        cutoff = np.datetime64('%04d-12-31' % (int(a[6].max().astype('datetime64[Y]').astype(int))+1969)) if n else None
        use = a[1] <= cutoff if n else np.zeros(len(a[0]), bool)
        for i in (0, 1, 2, 3, 5):
            a[i] = a[i][use]
        a[4] = {key: value[use] for key, value in a[4].items()}
    return a


class PeerTests(unittest.TestCase):
    def test_uniform_kernel_is_same_month_mean(self):
        value, audit = m.peer_month_average(np.zeros((3, 10)), np.zeros(10), np.array([2., -1., 5.]))
        self.assertEqual(value, 2.)
        self.assertEqual(audit, dict(peer_count=3, peer_weight_mass=3., peer_weighted_residual=6., peer_mean=2.))

    def test_gaussian_weights_not_equal_or_age_weighted(self):
        x = np.array([[0.]*10, [.5]*10])
        value, audit = m.peer_month_average(x, np.zeros(10), np.array([1., -1.]))
        expected = (1.-np.exp(-.5))/(1.+np.exp(-.5))
        self.assertEqual(value, expected)
        self.assertGreater(value, 0.)
        self.assertEqual(audit['peer_weight_mass'], 1.+np.exp(-.5))

    def test_duplicate_identical_peers_do_not_increase_month_mass(self):
        x = np.array([[0.]*10, [.2]*10]); r = np.array([-.3, .5]); q = np.zeros(10)
        first, _ = m.peer_month_average(x, q, r)
        second, _ = m.peer_month_average(np.repeat(x, 2, axis=0), q, np.repeat(r, 2))
        self.assertAlmostEqual(first, second, places=15)

    def test_empty_nonfinite_bounds_and_shapes_rejected(self):
        bad = [(np.empty((0, 10)), np.zeros(10), np.empty(0)),
               (np.zeros((2, 9)), np.zeros(10), np.ones(2)),
               (np.ones((2, 10)), np.zeros(10), np.ones(2)),
               (np.zeros((2, 10)), np.zeros(10), np.array([np.inf, 0.]))]
        for args in bad:
            with self.subTest(args=args), self.assertRaises((ValueError, FloatingPointError)):
                m.peer_month_average(*args)

    def test_finite_arithmetic_overflow_raises(self):
        with self.assertRaises(FloatingPointError):
            m.peer_month_average(np.zeros((4, 10)), np.zeros(10), np.full(4, 1e308))


class AnnualTests(unittest.TestCase):
    def test_original_three_forecasts_and_constant_control_exact(self):
        a = fixture(); z = run(a); ref = m.local.annual_forecasts(*a)
        self.assertEqual(set(z), {'raw', 'available', 'local_available', 'audit'})
        self.assertEqual(set(z['raw']), set(m.RAW_KEYS))
        for key in ('source_log', 'local_linear', 'local_constant'):
            np.testing.assert_array_equal(z['raw'][key], ref['raw'][key])
        np.testing.assert_array_equal(z['raw']['control'], ref['raw']['local_constant'])
        for key in ('available', 'local_available'):
            np.testing.assert_array_equal(z[key], ref[key])
        self.assertEqual(z['audit']['source'], ref['audit']['source'])
        self.assertTrue(np.any(z['raw']['candidate'] != z['raw']['control']))

    def test_peer_exclusion_same_month_same_mass_and_formula(self):
        a = fixture(); z = run(a)
        for detail in z['audit']['local']:
            self.assertEqual(detail['status'], 'fit')
            self.assertEqual(len(detail['peer_months']), detail['own_months'])
            self.assertEqual(detail['correction_denominator'], detail['weight_mass']+12.)
            for entry, weight, own_id in zip(detail['peer_months'], detail['own_weights'], detail['own_row_ids']):
                self.assertEqual(entry['own_row_id'], own_id)
                self.assertEqual(entry['own_weight'], weight)
                self.assertNotIn(own_id, entry['peer_row_ids'])
                self.assertEqual(entry['peer_count'], 3)
                for peer in entry['peer_row_ids']:
                    self.assertNotEqual(a[2][peer], detail['symbol'])
                    self.assertEqual(str(a[0][peer].astype('datetime64[M]')), entry['month'])
                    self.assertLessEqual(str(a[1][peer]), detail['cutoff'])
            numerator = float(np.einsum('n,n->', np.array(detail['own_weights']),
                np.array([x['peer_mean'] for x in detail['peer_months']]), optimize=False))
            self.assertEqual(numerator, detail['peer_numerator'])
            self.assertEqual(detail['peer_correction'], numerator/detail['correction_denominator'])
            j = detail['event_row']
            self.assertEqual(z['raw']['candidate'][j], z['raw']['source_log'][j]+detail['peer_correction'])

    def test_actual_counts_not_fixed_years_or_zero_model_claim(self):
        z = run(fixture()); c = z['audit']['counts']; rows = z['audit']['local']
        self.assertEqual(set(c), set(m.COUNT_FIELDS))
        annual_fits = sum(x['prediction_rows'] > 0 for x in z['audit']['annual'])
        self.assertEqual(c['source_target_fits'], 4*annual_fits)
        self.assertEqual(c['reference_pooled_log_refits'], annual_fits)
        self.assertEqual(c['transfer_pooled_log_refits'], annual_fits)
        self.assertEqual(c['peer_query_evaluations'], len(rows))
        self.assertEqual(c['own_constant_evaluations'], len(rows))
        self.assertEqual(c['peer_month_evaluations'], sum(len(x['peer_months']) for x in rows))
        self.assertEqual(c['peer_row_observations'], sum(e['peer_count'] for x in rows for e in x['peer_months']))
        self.assertEqual(c['actual_regression_solves'], 6*annual_fits+c['reference_local_solves'])

    def test_original_exclusive_five_year_window_and_maturity(self):
        a = prefix(fixture(), 1, physical=False); a[6][0] = np.datetime64('2021-01-04')
        z = run(a); annual = z['audit']['annual'][0]
        self.assertEqual(annual['lookback_exclusive_start'], '2015-12-31')
        self.assertEqual(annual['first_training_signal'], '2016-01-31')
        self.assertEqual(annual['latest_label_end'], '2020-12-31')
        for row in z['audit']['local']:
            self.assertLessEqual(row['latest_label_end'], row['cutoff'])

    def test_all_physical_prefixes_exact_and_inputs_unchanged(self):
        a = fixture(); before = copy.deepcopy(a); full = run(a)
        for n in range(len(a[6])+1):
            z = run(prefix(a, n))
            for key in m.RAW_KEYS:
                np.testing.assert_array_equal(z['raw'][key], full['raw'][key][:n])
            for key in ('available', 'local_available'):
                np.testing.assert_array_equal(z[key], full[key][:n])
        for i, value in enumerate(a):
            if isinstance(value, dict):
                for k in value:
                    np.testing.assert_array_equal(value[k], before[i][k])
            else:
                np.testing.assert_array_equal(value, before[i])

    def test_future_numeric_and_boolean_object_poison_ignored(self):
        a = prefix(fixture(), 2, physical=False); expected = run(a)
        future = a[1] > np.datetime64('2017-12-31')
        a[3] = a[3].astype(object); a[3][future] = 'future poison'
        a[5] = a[5].astype(object); a[5][future] = 'future poison'
        for key in a[4]:
            a[4][key] = a[4][key].astype(object); a[4][key][future] = 'future poison'
        z = run(a)
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], expected['raw'][key])
        self.assertEqual(z['audit'], expected['audit'])
        zero = prefix(a, 0, physical=False)
        z = run(zero)
        self.assertEqual(z['audit']['local'], [])
        self.assertTrue(all(x == 0 for x in z['audit']['counts'].values()))

    def test_own17_no_correction_and18_preserves_support(self):
        a = prefix(fixture(), 1, physical=False)
        own = np.flatnonzero((a[2] == 'A') & (a[1] <= np.datetime64('2015-12-31')))
        a[5][own[:-18]] = False
        z = run(a); self.assertTrue(z['local_available'][0])
        self.assertEqual(z['audit']['local'][0]['own_months'], 18)
        a[5][own[-18]] = False
        z = run(a); self.assertTrue(z['available'][0]); self.assertFalse(z['local_available'][0])
        self.assertEqual(z['audit']['local'][0]['peer_months'], [])
        for key in ('candidate', 'control'):
            np.testing.assert_array_equal(z['raw'][key], z['raw']['source_log'])

    def test_all_four_target_cohort_and_no_future_query_endpoint_filter(self):
        a = fixture(); a[4]['loss_utility'][0] = np.nan
        z = run(a)
        for row in z['audit']['annual']:
            self.assertNotIn(0, row['training_row_ids'])
        a[7][-1] = 'UNKNOWN'
        z = run(a)
        self.assertTrue(z['available'][-1]); self.assertFalse(z['local_available'][-1])
        self.assertEqual(z['raw']['candidate'][-1], z['raw']['source_log'][-1])

    def test_cold_start_and_ineligible_query_no_estimates(self):
        a = fixture(); a[9][:] = False; z = run(a)
        self.assertFalse(z['available'].any())
        self.assertTrue(all(c == 0 for c in z['audit']['counts'].values()))
        a = prefix(fixture(), 1); a[5][:] = False; z = run(a)
        self.assertFalse(z['available'].any()); self.assertEqual(z['audit']['local'], [])

    def test_zero_residual_exact_pooled_and_no_linear_solve(self):
        a = fixture()
        for y in a[4].values():
            y[:] = 0.
        z = run(a)
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], np.zeros(len(a[6])))
        self.assertEqual(z['audit']['counts']['reference_local_solves'], 0)
        self.assertGreater(z['audit']['counts']['peer_month_evaluations'], 0)

    def test_duplicate_ids_shapes_bad_boolean_and_minimum_rejected(self):
        for change in ('duplicate', 'shape', 'bool', 'minimum'):
            a = fixture()
            if change == 'duplicate': a[2][1] = a[2][0]
            if change == 'shape': a[3] = a[3][:, :9]
            if change == 'bool': a[5] = a[5].astype(int)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(a, minimum=True if change == 'minimum' else 18)

    def test_original_control_parity_failure_is_not_silently_copied(self):
        original = m.local.annual_forecasts
        def altered(*args, **kwargs):
            z = original(*args, **kwargs)
            z['audit']['local'][0]['constant_correction'] += .01
            return z
        with patch.object(m.local, 'annual_forecasts', side_effect=altered), self.assertRaises(AssertionError):
            run(fixture())


if __name__ == '__main__':
    unittest.main(verbosity=2)
