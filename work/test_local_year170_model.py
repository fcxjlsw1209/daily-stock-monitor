"""Synthetic-only checks for local annual-block sensitivity; no source data."""
from pathlib import Path
import copy
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_local_year170', P/'local_year170_model.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def fixture():
    days = pd.date_range('2015-12-31', '2021-12-31', freq='ME')
    dates = np.repeat(days.values.astype('datetime64[D]'), 3)
    ends = np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 3)
    symbols = np.tile(['A', 'B', 'C'], len(days))
    rng = np.random.default_rng(170)
    X = rng.uniform(-.5, .5, (len(dates), 10))
    y = .1*X[:, 0]+np.tile([-.03, .05, -.02], len(days))+.15*X[:, 2]**2
    y = (y.reshape(-1, 3)-y.reshape(-1, 3).mean(axis=1, keepdims=True)).reshape(-1)
    targets = {key: (i+1)*y for i, key in enumerate(m.TARGETS)}
    events = np.array(['2017-06-30', '2019-01-31', '2019-11-29', '2021-02-26'], dtype='datetime64[D]')
    return [dates, ends, symbols, X, targets, np.ones(len(dates), bool), events,
            np.array(['A', 'B', 'C', 'A']), rng.uniform(-.5, .5, (4, 10)), np.ones(4, bool)]


def run(args, min_months=3, min_own_months=3):
    return m.annual_forecasts(*args, min_months=min_months, min_own_months=min_own_months)


def slice_queries(args, n):
    args = copy.deepcopy(args)
    for i in (6, 7, 8, 9):
        args[i] = args[i][:n]
    return args


class AgreementTests(unittest.TestCase):
    def test_weighted_not_unweighted_and_strict_sign(self):
        a = m.weighted_sign_shrink(2., [8., 1., 1.], [1., -1., 0.])
        self.assertAlmostEqual(a['p'], .8)
        self.assertAlmostEqual(a['shrink'], .6)
        self.assertEqual(a['same_sign'], [True, False, False])
        b = m.weighted_sign_shrink(-2., [8., 1., 1.], [1., -1., -1.])
        self.assertAlmostEqual(b['p'], .2)
        self.assertEqual(b['shrink'], 0.)
        self.assertEqual(m.weighted_sign_shrink(1., [1., 1.], [1., -1.])['shrink'], 0.)
        self.assertEqual(m.weighted_sign_shrink(1., [1., 1.], [1., 1.])['shrink'], 1.)

    def test_zero_mass_and_fewer_two_positive_years(self):
        a = m.weighted_sign_shrink(1., [1., 0., 0.], [1., 1., 1.])
        self.assertEqual(a['p'], 1.)
        self.assertEqual(a['positive_mass_years'], 1)
        self.assertEqual(a['shrink'], 0.)
        b = m.weighted_sign_shrink(1., [0., 0.], [-1., 1.])
        self.assertIsNone(b['p'])
        self.assertEqual(b['shrink'], 0.)

    def test_zero_full_correction_override_including_signed_zero(self):
        for zero in (0., -0.):
            a = m.weighted_sign_shrink(zero, [0.], [-2.])
            self.assertEqual(a['shrink'], 1.)
            self.assertIsNone(a['p'])
        a = m.weighted_sign_shrink(0., [1., 1.], [1., -1.])
        self.assertEqual(a['p'], 0.)
        self.assertEqual(a['shrink'], 1.)

    def test_no_product_underflow_or_overflow_for_sign_comparison(self):
        for value in (np.nextafter(0., 1.), 1e308):
            a = m.weighted_sign_shrink(value, [1., 1.], [value, value])
            self.assertEqual(a['shrink'], 1.)

    def test_invalid_numeric_inputs_raise(self):
        cases = [(np.nan, [1.], [1.]), (1., [np.inf], [1.]),
                 (1., [1.], [np.nan]), (1., [-1.], [1.]),
                 (1., [[1.]], [[1.]]), (1., [1., 2.], [1.])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                m.weighted_sign_shrink(*args)


class FoldTests(unittest.TestCase):
    def test_scalar_zero_design_independent_formula_and_removed_rows(self):
        dates = np.array(['2018-01-31', '2018-02-28', '2019-01-31',
                          '2019-02-28', '2020-01-31', '2020-02-29'], dtype='datetime64[D]')
        ends = (pd.DatetimeIndex(dates)+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]')
        ids = np.array([7, 10, 43, 46, 79, 82])
        X = np.zeros((6, 10)); query = np.zeros(10); scale = np.ones(10)
        r = np.array([2., 2., 1., 1., -1., -1.]); w = np.arange(1., 7.)
        original = [x.copy() for x in (dates, ends, ids, X, query, scale, r, w)]
        c0 = sum(float(a)*float(b) for a, b in zip(w, r))/(sum(w)+12.)
        folds, agreement = m._year_folds(dates, ends, ids, X, query, scale, r, w, c0)
        self.assertEqual([f['removed_year'] for f in folds], [2018, 2019, 2020])
        self.assertEqual([f['removed_row_ids'] for f in folds], [[7, 10], [43, 46], [79, 82]])
        self.assertEqual([f['weight_mass'] for f in folds], [3., 7., 11.])
        for i, f in enumerate(folds):
            keep = np.ones(6, bool); keep[2*i:2*i+2] = False
            expected = sum(float(a)*float(b) for a, b in zip(w[keep], r[keep]))/(sum(w[keep])+12.)
            self.assertAlmostEqual(f['correction'], expected, places=15)
            self.assertEqual(f['retained_rows'], 4)
            self.assertEqual(f['retained_weight_mass'], float(sum(w[keep])))
            self.assertEqual(f['latest_label_end'], str(ends[keep].max()))
        self.assertAlmostEqual(agreement['p'], 11./21.)
        self.assertAlmostEqual(agreement['shrink'], 1./21.)
        for actual, before in zip((dates, ends, ids, X, query, scale, r, w), original):
            np.testing.assert_array_equal(actual, before)

    def test_one_year_empty_deletion_is_neutral_but_not_new_availability(self):
        dates = np.array(['2020-01-31', '2020-02-29'], dtype='datetime64[D]')
        folds, g = m._year_folds(dates, dates+30, np.array([0, 1]),
            np.zeros((2, 10)), np.zeros(10), np.ones(10), np.ones(2), np.ones(2), .1)
        self.assertEqual(folds[0]['retained_rows'], 0)
        self.assertEqual(folds[0]['status'], 'zero_weight')
        self.assertIsNone(folds[0]['latest_label_end'])
        self.assertEqual(g['shrink'], 0.)

    def test_folds_reuse_exact_raw_weights_scales_and_residuals(self):
        a = fixture(); calls = []; correct = m.local.local_corrections
        def spy(*args):
            calls.append([np.array(x, copy=True) for x in args])
            return correct(*args)
        with patch.object(m.local, 'local_corrections', side_effect=spy):
            z = run(a)
        offset = 0
        for detail in z['audit']['local']:
            full = calls[offset]; offset += 1
            ids = np.array(detail['own_row_ids'])
            np.testing.assert_array_equal(full[0], a[3][ids])
            np.testing.assert_array_equal(full[4], detail['own_weights'])
            for fold in detail['folds']:
                part = calls[offset]; offset += 1
                keep = ~np.isin(ids, fold['removed_row_ids'])
                for position in (0, 3, 4):
                    np.testing.assert_array_equal(part[position], full[position][keep])
                for position in (1, 2):
                    np.testing.assert_array_equal(part[position], full[position])
        self.assertEqual(offset, len(calls))


class AnnualTests(unittest.TestCase):
    def test_frozen_full_prediction_exact_and_expected_raw_formulas(self):
        a = fixture(); z = run(a)
        ref = m.local.annual_forecasts(*a, min_months=3, min_own_months=3)
        self.assertEqual(set(z['raw']), set(m.RAW_KEYS))
        for key in ('source_log', 'local_linear'):
            np.testing.assert_array_equal(z['raw'][key], ref['raw'][key])
        for key in ('available', 'local_available'):
            np.testing.assert_array_equal(z[key], ref[key])
        self.assertEqual(z['audit']['source'], ref['audit']['source'])
        for detail, old in zip(z['audit']['local'], ref['audit']['local']):
            j = detail['event_row']; c0 = detail['full_correction']
            self.assertEqual(c0, old['linear_correction'])
            self.assertEqual(detail['own_weights'], m.local.local_weights(
                a[0][detail['own_row_ids']], detail['cutoff'], a[3][detail['own_row_ids']], a[8][j]).tolist())
            self.assertEqual(z['raw']['local_year_stability'][j], z['raw']['source_log'][j]+c0*detail['shrink'])
            self.assertEqual(z['raw']['local_half'][j], z['raw']['source_log'][j]+.5*c0)
        for value in z['raw'].values():
            np.testing.assert_array_equal(np.isfinite(value), z['available'])

    def test_five_calendar_years_and_mature_cutoff_exact(self):
        a = slice_queries(fixture(), 1); a[6][0] = np.datetime64('2021-01-04')
        z = run(a); record = z['audit']['annual'][0]; detail = z['audit']['local'][0]
        self.assertEqual(record['lookback_exclusive_start'], '2015-12-31')
        self.assertEqual(record['first_training_signal'], '2016-01-31')
        self.assertEqual(record['latest_label_end'], '2020-12-31')
        self.assertEqual([f['removed_year'] for f in detail['folds']], list(range(2016, 2021)))
        for detail in z['audit']['local']:
            self.assertLessEqual(detail['latest_label_end'], detail['cutoff'])
            self.assertEqual(sorted(x for f in detail['folds'] for x in f['removed_row_ids']), detail['own_row_ids'])
            for f in detail['folds']:
                self.assertLessEqual(f['latest_label_end'], detail['cutoff'])
        unused = (a[0] <= np.datetime64('2015-12-31')) | (a[1] > np.datetime64('2020-12-31'))
        a[3][unused] = 999.
        for value in a[4].values():
            value[unused] = -999.
        got = run(a)
        for key in z['raw']:
            np.testing.assert_array_equal(got['raw'][key], z['raw'][key])

    def test_folds_below18_still_fit_original_own18_kept(self):
        a = slice_queries(fixture(), 1); a[6][0] = np.datetime64('2019-01-31'); a[7][0] = 'A'
        own = np.flatnonzero((a[2] == 'A') & (a[1] <= np.datetime64('2018-12-31')))
        a[5][own[:-18]] = False
        z = run(a, min_own_months=18); detail = z['audit']['local'][0]
        self.assertTrue(z['local_available'][0]); self.assertEqual(detail['own_months'], 18)
        self.assertTrue(all(f['retained_rows'] < 18 for f in detail['folds']))
        self.assertTrue(all(f['status'] == 'fit' for f in detail['folds']))
        a[5][own[-18]] = False
        got = run(a, min_own_months=18)
        self.assertTrue(got['available'][0]); self.assertFalse(got['local_available'][0])
        self.assertEqual(got['audit']['local'][0]['folds'], [])
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(got['raw'][key], got['raw']['source_log'])

    def test_absent_own_history_not_removed_from_original_availability(self):
        a = fixture(); a[7] = np.array(['NO_HISTORY']*4)
        z = run(a)
        self.assertTrue(z['available'].all()); self.assertFalse(z['local_available'].any())
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], z['raw']['source_log'])
        self.assertEqual(z['audit']['counts']['full_local_evaluations'], 0)

    def test_default_coldstarts_and_invalid_queries_preserve_source_mask(self):
        a = fixture(); a[9][1] = False; a[8][2, 9] = np.inf
        z = m.annual_forecasts(*a)
        ref = m.local.annual_forecasts(*a)
        np.testing.assert_array_equal(z['available'], ref['available'])
        np.testing.assert_array_equal(z['local_available'], ref['local_available'])
        for value in z['raw'].values():
            np.testing.assert_array_equal(np.isfinite(value), z['available'])

    def test_future_poison_and_every_query_prefix_nonmutating(self):
        a = fixture(); before = copy.deepcopy(a); z = run(a)
        for n in range(5):
            small = run(slice_queries(a, n))
            for key in m.RAW_KEYS:
                np.testing.assert_array_equal(small['raw'][key], z['raw'][key][:n])
            np.testing.assert_array_equal(small['available'], z['available'][:n])
            np.testing.assert_array_equal(small['local_available'], z['local_available'][:n])
        for actual, original in zip(a, before):
            if isinstance(actual, dict):
                for key in actual:
                    np.testing.assert_array_equal(actual[key], original[key])
            else:
                np.testing.assert_array_equal(actual, original)
        future = a[1] > np.datetime64('2018-12-31')
        a[3][future] = np.inf
        for value in a[4].values():
            value[future] = np.nan
        a[6][-1] = np.datetime64('2035-01-31'); a[8][-1] = np.nan
        got = run(a)
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(got['raw'][key][:3], z['raw'][key][:3])

    def test_zero_full_correction_and_zero_weight_retains_exact_source(self):
        a = fixture()
        for value in a[4].values():
            value[:] = 0.
        z = run(a)
        self.assertTrue(z['local_available'].all())
        self.assertTrue(all(d['shrink'] == 1. for d in z['audit']['local']))
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], z['raw']['source_log'])
        with patch.object(m.local, 'local_weights', side_effect=lambda dates, *args: np.zeros(len(dates))):
            got = run(fixture())
        self.assertFalse(got['local_available'].any()); self.assertTrue(got['available'].all())
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(got['raw'][key], got['raw']['source_log'])

    def test_common_finite_all_targets_singleton_months_and_duplicate_ids(self):
        a = fixture(); a[4]['vol_scaled'][:2] = np.nan; a[3][3, 9] = np.inf; a[5][6:9] = False
        z = run(a); ref = m.local.annual_forecasts(*a, min_months=3, min_own_months=3)
        np.testing.assert_array_equal(z['raw']['local_linear'], ref['raw']['local_linear'])
        for annual in z['audit']['annual']:
            self.assertTrue(set(annual['training_row_ids']).isdisjoint({0, 1, 2, 3, 6, 7, 8}))
        a = fixture(); a[2][4] = a[2][3]
        with self.assertRaisesRegex(ValueError, 'one row'):
            run(a)

    def test_audit_fitcounts_pins_json_deterministic(self):
        z = run(fixture()); again = run(fixture())
        self.assertEqual(z['audit'], again['audit']); json.dumps(z['audit'], allow_nan=False)
        counts = z['audit']['counts']; details = z['audit']['local']
        self.assertEqual(counts['pooled_log_refits'], 3)
        self.assertEqual(counts['source_target_fits'], 12)
        self.assertEqual(counts['full_local_evaluations'], 4)
        self.assertEqual(counts['full_local_solves'], sum(d['status'] == 'fit' for d in details))
        self.assertEqual(counts['deletion_evaluations'], sum(len(d['folds']) for d in details))
        self.assertEqual(counts['deletion_solves'], sum(f['status'] == 'fit' for d in details for f in d['folds']))
        deps = set(m.dependency_files())
        self.assertIn(P/'local_year170_model.py', deps); self.assertIn(m.LOCAL_PATH, deps)
        self.assertTrue(set(m.local.dependency_files()) <= deps)
        self.assertEqual((m.MIN_MONTHS, m.MIN_OWN_MONTHS), (18, 18))

    def test_numeric_failure_is_not_a_missing_fold(self):
        original = m.local.local_corrections; calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise np.linalg.LinAlgError('synthetic deletion failure')
            return original(*args)
        with patch.object(m.local, 'local_corrections', side_effect=fail_second):
            with self.assertRaisesRegex(np.linalg.LinAlgError, 'deletion failure'):
                run(fixture())


if __name__ == '__main__':
    unittest.main()
