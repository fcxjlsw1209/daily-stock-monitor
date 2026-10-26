"""Invented early monthly queries only; no real cache or price input is opened."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_early229', Path(__file__).with_name('monthly_early_history229_model.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rng = np.random.default_rng(229)
    days = pd.date_range('2011-01-31', '2018-12-31', freq='ME')
    dates = np.repeat(days.to_numpy(dtype='datetime64[D]'), 4)
    ends = np.repeat((days + pd.offsets.MonthEnd(1)).to_numpy(dtype='datetime64[D]'), 4)
    symbols = np.tile(['A', 'B', 'C', 'D'], len(days))
    X = rng.uniform(-.5, .5, (len(dates), 10))
    y = .1 * X[:, 0] + .2 * X[:, 1] ** 2 + np.tile([-.04, .01, .05, -.02], len(days))
    y = (y.reshape(-1, 4) - y.reshape(-1, 4).mean(axis=1, keepdims=True)).ravel()
    targets = {key: y * (j + 1) for j, key in enumerate(m.TARGETS)}
    events = np.array(['2011-01-31', '2012-02-29', '2013-01-31', '2013-06-30',
                       '2014-01-31', '2014-06-30', '2016-07-31'], dtype='datetime64[D]')
    ed = np.repeat(events, 4)
    args = [dates, ends, symbols, X, targets, np.ones(len(dates), bool), ed,
            np.tile(['A', 'B', 'C', 'D'], len(events)),
            rng.uniform(-.5, .5, (len(ed), 10)),
            np.tile([True, False, False, True], len(events))]
    return args, np.tile([False, True, False, True], len(events))


def physical_prefix(value, n):
    args, allowed = copy.deepcopy(value)
    if type(n) is not int or not 0 <= n <= len(args[6]):
        raise ValueError('Invalid synthetic query prefix')
    for k in (6, 7, 8, 9):
        args[k] = args[k][:n]
    allowed = allowed[:n]
    cutoff = np.datetime64('%04d-12-31' % (int(str(args[6][-1])[:4]) - 1), 'D') if n else None
    use = args[1] <= cutoff if n else np.zeros(len(args[0]), bool)
    for k in (0, 1, 2, 3, 5):
        args[k] = args[k][use]
    args[4] = {k: v[use] for k, v in args[4].items()}
    return args, allowed


def run(value):
    args, allowed = value
    return m.forecasts(*args, expansion_allowed=allowed)


def array_state(value):
    if isinstance(value, dict):
        return {k: array_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [array_state(v) for v in value]
    a = np.asarray(value)
    return (a.dtype.str, a.shape, a.tobytes())


class ForecastTests(unittest.TestCase):
    def test_union_exactly_matches_one_frozen_call(self):
        f = fixture(); args, allowed = f
        expected = m.local.annual_forecasts(*args[:9], args[9] | allowed)
        with patch.object(m.local, 'annual_forecasts', wraps=m.local.annual_forecasts) as replay:
            actual = run(f)
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(set(actual), {'raw', 'available', 'local_available', 'audit'})
        self.assertEqual(set(actual['raw']), set(m.RAW_KEYS))
        for key in m.RAW_KEYS:
            self.assertEqual(array_state(actual['raw'][key]), array_state(expected['raw'][key]))
        self.assertEqual(actual['audit']['replay_audit'], expected['audit'])
        np.testing.assert_array_equal(actual['available'], expected['available'])

    def test_original_and_added_masks_disjoint_and_counts_truthful(self):
        z = run(fixture()); a = z['audit']; c = a['counts']
        self.assertEqual(tuple(c), m.COUNT_FIELDS)
        old = np.asarray(a['original_query_mask']); new = np.asarray(a['expansion_query_mask'])
        self.assertFalse((old & new).any())
        np.testing.assert_array_equal(old | new, a['union_query_mask'])
        self.assertEqual(c['union_requested_rows'], c['original_query_rows'] + c['expansion_requested_rows'])
        self.assertEqual(c['available_rows'], c['original_available_rows'] + c['added_available_rows'])
        self.assertEqual(c['local_evaluations'], c['available_rows'])
        self.assertEqual(c['actual_regression_solves'], c['source_target_fits'] + c['pooled_log_refits'] + c['local_solves'])

    def test_actual_fit_counts_match_called_primitives(self):
        with patch.object(m.local.numeric, '_fit_ridge', wraps=m.local.numeric._fit_ridge) as fits:
            z = run(fixture())
        c = z['audit']['counts']
        self.assertEqual(fits.call_count, c['source_target_fits'] + c['pooled_log_refits'])
        self.assertEqual(c['source_target_fits'], 4 * c['pooled_log_refits'])
        self.assertGreater(c['local_solves'], 0)

    def test_original_eligible_does_not_require_expansion_warmup(self):
        f = fixture(); f[1][:] = False; z = run(f)
        self.assertEqual(z['audit']['counts']['added_available_rows'], 0)
        self.assertTrue(z['available'][8])
        self.assertTrue(z['available'][11])
        self.assertFalse(z['available'][9])

    def test_ineligible_warmup_can_predict_without_changing_training(self):
        f = fixture(); before = array_state(f); z = run(f)
        self.assertTrue(z['available'][9])
        self.assertFalse(f[0][9][9])
        self.assertFalse(z['available'][10])
        self.assertEqual(before, array_state(f))

    def test_current_future_label_not_used_for_query_admission(self):
        f = physical_prefix(fixture(), 12); a = f[0]
        # All training has matured by 2012-12-31, so 2013 query labels are absent.
        self.assertTrue((a[1] <= np.datetime64('2012-12-31')).all())
        self.assertFalse((a[0] == np.datetime64('2013-01-31')).any())
        z = run(f)
        self.assertTrue(z['available'][8:12][[0, 1, 3]].all())
        self.assertFalse(z['audit']['query_future_labels_used'])

    def test_future_training_values_cannot_change_earlier_predictions(self):
        f = physical_prefix(fixture(), 16); full = fixture()
        for k in (6, 7, 8, 9): full[0][k] = full[0][k][:16]
        full = (full[0], full[1][:16])
        future = full[0][1] > np.datetime64('2012-12-31')
        full[0][3][future] = 1e100
        for v in full[0][4].values(): v[future] = np.inf
        got, expected = run(full), run(f)
        for key in m.RAW_KEYS: np.testing.assert_array_equal(got['raw'][key], expected['raw'][key])
        self.assertEqual(got['audit'], expected['audit'])

    def test_every_physical_prefix_matches_full_predictions(self):
        f = fixture(); full = run(f)
        for n in range(len(f[0][6]) + 1):
            small = run(physical_prefix(f, n))
            for key in m.RAW_KEYS:
                np.testing.assert_array_equal(small['raw'][key], full['raw'][key][:n])
            for key in ('available', 'local_available'):
                np.testing.assert_array_equal(small[key], full[key][:n])

    def test_2011_and_2012_are_cold_without_fabricated_training(self):
        z = run(physical_prefix(fixture(), 8))
        self.assertFalse(z['available'].any())
        self.assertEqual(z['audit']['counts']['actual_regression_solves'], 0)
        self.assertTrue(all(a['status'] == 'cold_start' for a in z['audit']['replay_audit']['annual']))

    def test_exact_eighteen_month_boundary_and_singleton_exclusion(self):
        f = physical_prefix(fixture(), 12); a = f[0]
        keep = (a[0] >= np.datetime64('2011-06-01'))
        for k in (0, 1, 2, 3, 5): a[k] = a[k][keep]
        a[4] = {k: v[keep] for k, v in a[4].items()}
        self.assertEqual(len(np.unique(a[0].astype('datetime64[M]'))), 18)
        self.assertTrue(run(f)['available'][8])
        a[5][:3] = False  # One remaining row in the first month cannot train.
        z = run(f)
        self.assertFalse(z['available'][8])
        self.assertEqual(z['audit']['replay_audit']['annual'][-1]['training_months'], 17)

    def test_absent_own_identity_retains_exact_pooled_prediction(self):
        f = fixture(); f[0][7][9] = 'Z'; z = run(f)
        self.assertTrue(z['available'][9]); self.assertFalse(z['local_available'][9])
        self.assertEqual(z['raw']['local_linear'][9], z['raw']['source_log'][9])
        self.assertEqual(z['raw']['local_constant'][9], z['raw']['source_log'][9])

    def test_nonfinite_query_features_neutralize_original_and_added(self):
        f = fixture(); f[0][8][8, 0] = np.nan; f[0][8][9, 0] = np.inf
        z = run(f)
        self.assertFalse(z['available'][8:10].any())
        self.assertFalse(any(z['audit']['union_query_mask'][8:10]))

    def test_zero_targets_count_pooled_fits_but_no_local_solve(self):
        f = fixture()
        for v in f[0][4].values(): v[:] = 0.
        z = run(f); c = z['audit']['counts']
        self.assertGreater(c['pooled_log_refits'], 0); self.assertEqual(c['local_solves'], 0)
        self.assertTrue(all(r['status'] == 'zero_residual' for r in z['audit']['replay_audit']['local']))
        for key in m.RAW_KEYS: self.assertTrue((z['raw'][key][z['available']] == 0.).all())

    def test_no_requested_queries_counts_no_model_fits(self):
        f = fixture(); f[0][9][:] = False; f[1][:] = False
        z = run(f); self.assertFalse(z['available'].any())
        self.assertEqual(z['audit']['counts']['actual_regression_solves'], 0)

    def test_empty_physical_query_has_empty_training_and_zero_counts(self):
        f = physical_prefix(fixture(), 0); self.assertEqual(len(f[0][0]), 0)
        z = run(f)
        self.assertEqual(z['available'].shape, (0,))
        self.assertTrue(all(v == 0 for v in z['audit']['counts'].values()))
        self.assertEqual(z['audit']['replay_audit']['annual'], [])

    def test_prior_december_and_five_year_cutoffs_are_unchanged(self):
        z = run(fixture()); a = z['audit']['replay_audit']
        for row in a['annual']:
            self.assertEqual(row['cutoff'], '%04d-12-31' % (row['year'] - 1))
            self.assertEqual(row['lookback_exclusive_start'], '%04d-12-31' % (row['year'] - 6))
            if row['latest_label_end']: self.assertLessEqual(row['latest_label_end'], row['cutoff'])
        self.assertEqual(a['min_months'], 18); self.assertEqual(a['min_own_months'], 18)
        self.assertEqual(a['alpha'], 10.); self.assertEqual(a['half_life_months'], 12.)

    def test_inputs_unchanged_and_outputs_do_not_share_input_arrays(self):
        f = fixture(); before = array_state(f); z = run(f)
        self.assertEqual(before, array_state(f))
        for value in list(z['raw'].values()) + [z['available'], z['local_available']]:
            for arg in f[0]:
                if isinstance(arg, np.ndarray): self.assertFalse(np.shares_memory(value, arg))

    def test_canonical_string_dates_are_accepted(self):
        f = fixture(); expected = run(f)
        for k in (0, 1, 6): f[0][k] = f[0][k].astype(str)
        got = run(f)
        self.assertEqual(got['audit'], expected['audit'])
        for k in m.RAW_KEYS: np.testing.assert_array_equal(got['raw'][k], expected['raw'][k])

    def test_duplicate_training_and_query_identity_rejected(self):
        for key in (2, 7):
            f = fixture(); f[0][key][1] = f[0][key][0]
            with self.assertRaises(ValueError): run(f)

    def test_bad_dates_boundary_order_or_within_month_signal_rejected(self):
        for which in ('boundary', 'after', 'nat', 'numeric', 'time', 'order', 'split_month'):
            f = fixture(); a = f[0]
            if which == 'boundary': a[6][-1] = np.datetime64('2016-08-17')
            elif which == 'after': a[6][-1] = np.datetime64('2017-01-31')
            elif which == 'nat': a[6][-1] = np.datetime64('NaT')
            elif which == 'numeric': a[6] = np.arange(len(a[6]))
            elif which == 'time': a[6] = a[6].astype('datetime64[ns]')
            elif which == 'order': a[6] = a[6][::-1]
            else: a[6][3] = np.datetime64('2011-01-30')
            with self.subTest(which=which), self.assertRaises(ValueError): run(f)

    def test_malformed_shapes_types_ids_targets_and_ends_fail(self):
        for which in range(10):
            f = fixture(); a = f[0]
            if which == 0: a[3] = a[3][:, :9]
            elif which == 1: a[8] = a[8].astype(complex)
            elif which == 2: a[3] = a[3].astype(object)
            elif which == 3: a[9] = a[9].astype(int)
            elif which == 4: f = (a, f[1].astype(int))
            elif which == 5: a[7] = np.array([' A'] * len(a[7]))
            elif which == 6: a[4].pop(m.TARGETS[0])
            elif which == 7: a[4][m.TARGETS[0]] = np.zeros(len(a[0]), bool)
            elif which == 8: a[1][0] = a[0][0]
            else: a[1][0] = np.datetime64('2011-02-27')
            with self.subTest(which=which), self.assertRaises(ValueError): run(f)

    def test_consumed_rank_overflow_not_silently_clipped(self):
        f = fixture(); f[0][8][9, 0] = .5001
        with self.assertRaises(ValueError): run(f)

    def test_expansion_mask_is_mandatory_and_defaults_are_not_tunable(self):
        a, allowed = fixture()
        with self.assertRaises(TypeError): m.forecasts(*a)
        with self.assertRaises(TypeError): m.forecasts(*a, expansion_allowed=allowed, min_months=1)

    def test_dependency_closure_contains_only_existing_frozen_numeric_files_and_self(self):
        paths = set(m.dependency_files())
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertIn(m.LOCAL_PATH, paths)
        self.assertTrue(set(m.local.dependency_files()).issubset(paths))
        self.assertFalse(any('227' in p.name for p in paths))


if __name__ == '__main__':
    unittest.main(verbosity=2)
