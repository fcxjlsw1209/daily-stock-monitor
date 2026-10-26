"""Synthetic local-model algebra, source parity and causal annual-cohort tests."""
from pathlib import Path
import copy
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('local_linear70_model_test', P/'model.py')
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)


def fixture():
    days = pd.date_range('2015-12-31', '2021-12-31', freq='ME')
    dates = np.repeat(days.values.astype('datetime64[D]'), 3)
    ends = np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 3)
    symbols = np.tile(['A', 'B', 'C'], len(days))
    rng = np.random.default_rng(81823)
    X = rng.uniform(-.5, .5, (len(dates), 10))
    y = .08*X[:, 0]+np.tile([-.02, .04, -.02], len(days))+.12*X[:, 2]**2
    y = (y.reshape(-1, 3)-y.reshape(-1, 3).mean(axis=1, keepdims=True)).reshape(-1)
    targets = {key: (i+1)*y for i, key in enumerate(m.TARGETS)}
    ed = np.array(['2017-06-30', '2019-01-31', '2019-11-29', '2021-02-26'], dtype='datetime64[D]')
    return [dates, ends, symbols, X, targets, np.ones(len(dates), bool), ed,
            np.array(['A', 'B', 'C', 'A']), rng.uniform(-.5, .5, (4, 10)), np.ones(4, bool)]


def run(args, min_months=3, min_own_months=3):
    return m.annual_forecasts(*args, min_months=min_months, min_own_months=min_own_months)


def query_slice(args, n):
    result = copy.deepcopy(args)
    for i in (6, 7, 8, 9):
        result[i] = result[i][:n]
    return result


class FormulaTests(unittest.TestCase):
    def test_kernel_decay_is_original_rank_distance_cutoff_clock_and_unnormalized(self):
        dates = np.array(['2019-12-01', '2020-12-01', '2020-12-31'], dtype='datetime64[D]')
        X = np.array([[0.]*10, [.5]*10, [-.5]*10])
        query = np.zeros(10)
        got = m.local_weights(dates, '2020-12-31', X, query)
        np.testing.assert_allclose(got, [.5, np.exp(-.5), np.exp(-.5)], rtol=1e-15, atol=0.)
        self.assertNotAlmostEqual(got.sum(), 1.)
        self.assertNotAlmostEqual(got.sum(), len(got))
        # Month age, not day age; changing cutoff day within the same month
        # leaves weights alone when it still follows every used signal.
        np.testing.assert_array_equal(m.local_weights(dates[:2], '2020-12-02', X[:2], query), got[:2])

    def test_independent_augmented_least_squares_and_normal_equations(self):
        rng = np.random.default_rng(871)
        X = rng.uniform(-.5, .5, (27, 10)); query = rng.uniform(-.5, .5, 10)
        scale = np.linspace(.1, 1., 10); residual = rng.normal(0., .08, len(X))
        weights = np.linspace(.05, .9, len(X))
        constant, linear, info = m.local_corrections(X, query, scale, residual, weights)
        Z = np.column_stack([np.ones(len(X)), (X-query)/scale])
        penalty = np.array([12.]+[10.]*10)
        augmented = np.vstack([np.sqrt(weights)[:, None]*Z, np.diag(np.sqrt(penalty))])
        response = np.r_[np.sqrt(weights)*residual, np.zeros(11)]
        theta = np.linalg.lstsq(augmented, response, rcond=None)[0]
        self.assertAlmostEqual(linear, theta[0], places=14)
        self.assertAlmostEqual(constant, sum(float(w)*float(r) for w, r in zip(weights, residual))/(sum(weights)+12), places=15)
        self.assertAlmostEqual(info['weight_mass'], weights.sum(), places=14)
        self.assertAlmostEqual(info['effective_samples'], sum(weights)**2/sum(weights**2), places=13)
        self.assertGreater(abs(constant-linear), 1e-5)

    def test_query_centering_and_intercept_penalty_not_unpenalized_intercept(self):
        X = np.zeros((4, 10)); X[:, 0] = [-.4, -.1, .2, .5]
        query = np.zeros(10); query[0] = .3
        residual = np.array([-.2, .1, .2, .5]); w = np.array([.2, .4, .8, 1.])
        scale = np.ones(10); scale[0] = .2
        c, line, _ = m.local_corrections(X, query, scale, residual, w)
        d = (X[:, 0]-query[0])/scale[0]
        lhs = np.array([[sum(w)+12, sum(w*d)], [sum(w*d), sum(w*d*d)+10]])
        rhs = np.array([sum(w*residual), sum(w*d*residual)])
        self.assertAlmostEqual(line, np.linalg.solve(lhs, rhs)[0], places=15)
        self.assertNotAlmostEqual(c, np.average(residual, weights=w))
        # A second local standardization would remove this deliberate scale.
        other = m.local_corrections(X, query, np.ones(10), residual, w)[1]
        self.assertGreater(abs(line-other), 1e-5)

    def test_all_zero_design_exact_or_close_to_constant_without_special_fit(self):
        query = np.linspace(-.4, .4, 10)
        X = np.tile(query, (19, 1)); residual = np.linspace(-.2, .3, 19); w = np.linspace(.1, 1., 19)
        c, line, _ = m.local_corrections(X, query, np.linspace(.01, 2., 10), residual, w)
        self.assertAlmostEqual(c, line, places=15)

    def test_zero_weight_and_zero_residual_do_not_solve(self):
        X = np.zeros((18, 10)); query = np.ones(10)*.2
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('must not solve zero correction')):
            for r, w, status in ((np.ones(18), np.zeros(18), 'zero_weight'),
                                 (np.zeros(18), np.ones(18), 'zero_residual')):
                c, line, audit = m.local_corrections(X, query, np.ones(10), r, w)
                self.assertEqual((c, line), (0., 0.)); self.assertEqual(audit['status'], status)

    def test_no_normalization_prior_strength_changes_with_observation_mass(self):
        X = np.zeros((6, 10)); r = np.full(6, .1); w = np.ones(6)
        first = m.local_corrections(X, np.zeros(10), np.ones(10), r, w)
        second = m.local_corrections(X, np.zeros(10), np.ones(10), r, .5*w)
        self.assertAlmostEqual(first[0], .1*6/(6+12))
        self.assertAlmostEqual(second[0], .1*3/(3+12))
        self.assertGreater(first[0], second[0])

    def test_used_numeric_failures_raise_instead_of_missing_or_clipping(self):
        X = np.zeros((18, 10)); q = np.zeros(10); scale = np.ones(10); r = np.ones(18); w = np.ones(18)
        with patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('bad solve')):
            with self.assertRaises(np.linalg.LinAlgError):m.local_corrections(X, q, scale, r, w)
        with patch.object(np.linalg, 'solve', return_value=np.full(11, np.nan)):
            with self.assertRaisesRegex(ValueError, 'coefficients'):m.local_corrections(X, q, scale, r, w)
        for name, bad in [('scale', np.zeros(10)), ('residual', np.full(18, np.nan)), ('weights', -np.ones(18))]:
            args = [X, q, scale, r, w]; args[{'scale': 2, 'residual': 3, 'weights': 4}[name]] = bad
            with self.assertRaises(ValueError):m.local_corrections(*args)
        with self.assertRaises(ValueError):m.local_weights(np.array(['2020-01-31'], dtype='datetime64[D]'), '2019-12-31', X[:1], q)
        for bad in (np.nan, np.inf, .5001):
            xx = X[:1].copy(); xx[0, 0] = bad
            with self.assertRaises(ValueError):m.local_weights(np.array(['2019-01-31'], dtype='datetime64[D]'), '2019-12-31', xx, q)


class AnnualTests(unittest.TestCase):
    def test_exact_original_raw_available_audit_and_fixed_parameters(self):
        a = fixture(); z = run(a)
        ref = m.original.annual_forecasts(a[0], a[1], a[3], a[4], a[5], a[6], a[8], a[9], min_months=3)
        np.testing.assert_array_equal(z['raw']['source_log'], ref['log_excess'])
        np.testing.assert_array_equal(z['available'], ref['available'])
        self.assertEqual(z['audit']['source'], ref['audit'])
        self.assertEqual(set(z['raw']), {'source_log', 'local_constant', 'local_linear'})
        self.assertEqual((m.ALPHA, m.HALF_LIFE_MONTHS, m.MIN_MONTHS, m.MIN_OWN_MONTHS,
                          m.BANDWIDTH, m.INTERCEPT_PENALTY, m.SLOPE_PENALTY), (10., 12., 18, 18, .5, 12., 10.))
        self.assertTrue(z['local_available'].all())
        for raw in z['raw'].values():np.testing.assert_array_equal(np.isfinite(raw), z['available'])

    def test_exact_annual_boundary_maturity_and_unused_future_values(self):
        a = fixture()
        a[6] = np.array(['2021-01-04'], dtype='datetime64[D]')
        for i in (7, 8, 9):a[i] = a[i][:1]
        z = run(a); audit = z['audit']['annual'][0]
        self.assertEqual(audit['lookback_exclusive_start'], '2015-12-31')
        self.assertEqual(audit['first_training_signal'], '2016-01-31')
        self.assertEqual(audit['latest_label_end'], '2020-12-31')
        unused = (a[0] <= np.datetime64('2015-12-31')) | (a[1] > np.datetime64('2020-12-31'))
        a[3][unused] = 999.
        for y in a[4].values():y[unused] = -999.
        got = run(a)
        for key in z['raw']:np.testing.assert_array_equal(got['raw'][key], z['raw'][key])

    def test_pooled_rows_scaler_and_local_shared_inputs_match_independent_formula(self):
        a = fixture(); pooled_seen = []; local_seen = []
        fit, correct = m._fit_pooled, m.local_corrections
        def capture_fit(X, y, w):
            result = fit(X, y, w); pooled_seen.append((X.copy(), y.copy(), w.copy(), result)); return result
        def capture_local(X, query, scale, residual, weights):
            local_seen.append(tuple(x.copy() for x in (X, query, scale, residual, weights)))
            return correct(X, query, scale, residual, weights)
        with patch.object(m, '_fit_pooled', side_effect=capture_fit), patch.object(m, 'local_corrections', side_effect=capture_local):
            z = run(a)
        self.assertEqual(len(pooled_seen), 3); self.assertEqual(len(local_seen), 4)
        by_year = {}
        for row, observed in zip(z['audit']['annual'], pooled_seen):
            cutoff = np.datetime64(row['cutoff']); start = np.datetime64(row['lookback_exclusive_start'])
            use = (a[0] > start) & (a[1] <= cutoff)
            np.testing.assert_array_equal(observed[0], a[3][use])
            np.testing.assert_array_equal(observed[1], a[4]['log_excess'][use])
            np.testing.assert_array_equal(observed[2], m.numeric.month_balanced_weights(a[0][use], cutoff, 12.))
            np.testing.assert_array_equal(observed[3][0], np.average(a[3][use], axis=0, weights=observed[2]))
            by_year[row['year']] = (use, observed[3])
        for j, seen in enumerate(local_seen):
            year = int(str(a[6][j])[:4]); train, pooled = by_year[year]
            own = train & (a[2] == a[7][j]); cutoff = np.datetime64(str(year-1)+'-12-31')
            np.testing.assert_array_equal(seen[0], a[3][own])
            np.testing.assert_array_equal(seen[1], a[8][j]); np.testing.assert_array_equal(seen[2], pooled[1])
            np.testing.assert_array_equal(seen[3], a[4]['log_excess'][own]-m.numeric._predict(pooled, a[3][own]))
            expected_w = np.exp2(-(cutoff.astype('datetime64[M]').astype(int)-a[0][own].astype('datetime64[M]').astype(int))/12)
            expected_w *= np.exp(-np.mean((a[3][own]-a[8][j])**2, axis=1)/(2*.5**2))
            np.testing.assert_array_equal(seen[4], expected_w)
            c, line, _ = correct(*seen)
            self.assertAlmostEqual(z['raw']['local_constant'][j], z['raw']['source_log'][j]+c, places=15)
            self.assertAlmostEqual(z['raw']['local_linear'][j], z['raw']['source_log'][j]+line, places=15)

    def test_query_changes_never_fit_pooled_scaler_or_include_other_security(self):
        a = fixture(); fits = []; local_rows = []
        fitting, correcting = m._fit_pooled, m.local_corrections
        def fit(X, y, w):
            result = fitting(X, y, w); fits.append(result); return result
        def local(X, *args):local_rows.append(X.copy()); return correcting(X, *args)
        with patch.object(m, '_fit_pooled', side_effect=fit), patch.object(m, 'local_corrections', side_effect=local):run(a)
        first_fits = fits.copy(); first_rows = local_rows.copy(); fits.clear(); local_rows.clear()
        a[8] *= -.5
        with patch.object(m, '_fit_pooled', side_effect=fit), patch.object(m, 'local_corrections', side_effect=local):run(a)
        for left, right in zip(first_fits, fits):
            for x, y in zip(left, right):np.testing.assert_array_equal(x, y)
        for left, right in zip(first_rows, local_rows):np.testing.assert_array_equal(left, right)

    def test_all4_targets_all10_features_and_singleton_month_filter_shared(self):
        a = fixture(); a[4]['vol_scaled'][:2] = np.nan
        a[3][3, 9] = np.inf; a[5][6:9] = False
        seen = []; fit = m._fit_pooled
        def capture(X, y, w):seen.append(X.copy()); return fit(X, y, w)
        with patch.object(m, '_fit_pooled', side_effect=capture):z = run(a)
        for audit, rows in zip(z['audit']['annual'], seen):
            train = (a[0] > np.datetime64(audit['lookback_exclusive_start'])) & (a[1] <= np.datetime64(audit['cutoff']))
            train[:3] = False; train[3] = False; train[6:9] = False
            np.testing.assert_array_equal(rows, a[3][train])
            self.assertEqual(len(rows), audit['training_rows'])
        self.assertEqual(z['audit']['annual'][0]['training_rows'], z['audit']['source']['annual'][0]['training_rows'])

    def test_own18_threshold_changes_only_correction_not_event_availability(self):
        a = fixture(); a[6] = np.array(['2017-02-01', '2018-02-01'], dtype='datetime64[D]')
        for i in (7, 8, 9):a[i] = a[i][:2]
        a[7][:] = 'A'; z = run(a, min_own_months=18)
        np.testing.assert_array_equal(z['available'], [True, True])
        np.testing.assert_array_equal(z['local_available'], [False, True])
        for key in m.MODES:self.assertEqual(z['raw'][key][0], z['raw']['source_log'][0])
        a[7][:] = 'NO_HISTORY'; got = run(a, min_own_months=18)
        self.assertTrue(got['available'].all()); self.assertFalse(got['local_available'].any())
        for key in m.MODES:np.testing.assert_array_equal(got['raw'][key], got['raw']['source_log'])

    def test_exact_seventeen_eighteen_own_months_boundary(self):
        for own_count in (17, 18):
            a = fixture()
            a[6] = np.array(['2019-01-31'], dtype='datetime64[D]')
            for i in (7, 8, 9):a[i] = a[i][:1]
            a[7][:] = 'A'
            own = np.flatnonzero((a[2] == 'A') & (a[1] <= np.datetime64('2018-12-31')))
            a[5][own[:-own_count]] = False
            z = run(a, min_own_months=18)
            self.assertEqual(z['audit']['local'][0]['own_months'], own_count)
            self.assertTrue(z['available'][0])
            self.assertEqual(bool(z['local_available'][0]), own_count == 18)
            if own_count == 17:
                for key in m.MODES:self.assertEqual(z['raw'][key][0], z['raw']['source_log'][0])

    def test_default18_global_coldstart_and_bad_queries_retain_original_nan_mask(self):
        a = fixture(); a[9][1] = False; a[8][2, 9] = np.inf
        z = m.annual_forecasts(*a)
        np.testing.assert_array_equal(z['available'], [False, False, False, True])
        np.testing.assert_array_equal(z['local_available'], [False, False, False, True])
        for raw in z['raw'].values():np.testing.assert_array_equal(np.isfinite(raw), z['available'])

    def test_zero_residual_and_zero_local_weight_copy_source_exact(self):
        a = fixture()
        for target in a[4].values():target[:] = 0.
        z = run(a)
        self.assertTrue(z['local_available'].all())
        self.assertEqual({row['status'] for row in z['audit']['local']}, {'zero_residual'})
        for key in m.MODES:np.testing.assert_array_equal(z['raw'][key], z['raw']['source_log'])
        a = fixture()
        with patch.object(m, 'local_weights', side_effect=lambda dates, *args: np.zeros(len(dates))):got = run(a)
        self.assertFalse(got['local_available'].any())
        self.assertTrue(got['available'].all())
        for key in m.MODES:np.testing.assert_array_equal(got['raw'][key], got['raw']['source_log'])

    def test_future_poison_labels_features_and_queries_do_not_change_past(self):
        a = fixture(); z = run(a)
        future = a[1] > np.datetime64('2018-12-31')
        a[3][future] = np.inf
        for target in a[4].values():target[future] = np.nan
        a[6][-1] = np.datetime64('2035-01-31'); a[8][-1] = np.nan
        got = run(a)
        for key in z['raw']:np.testing.assert_array_equal(got['raw'][key][:3], z['raw'][key][:3])
        np.testing.assert_array_equal(got['available'][:3], z['available'][:3])

    def test_unused_out_of_bounds_ranks_ignore_but_actual_used_fail(self):
        a = query_slice(fixture(), 3); z = run(a)
        future = a[1] > np.datetime64('2018-12-31'); a[3][future, 0] = 999.
        got = run(a)
        for key in z['raw']:np.testing.assert_array_equal(got['raw'][key], z['raw'][key])
        a = fixture(); a[3][3, 0] = .50001
        with self.assertRaisesRegex(ValueError, 'percentile'):run(a)
        a[5][3] = False; run(a)
        a = fixture(); a[8][0, 0] = .50001
        with self.assertRaisesRegex(ValueError, 'percentile'):run(a)
        a[9][0] = False; run(a)

    def test_every_query_prefix_deterministic_nonmutating_and_json_audit(self):
        a = fixture(); before = copy.deepcopy(a); z = run(a); repeat = run(a)
        self.assertEqual(z['audit'], repeat['audit']); json.dumps(z['audit'], allow_nan=False)
        for n in range(5):
            got = run(query_slice(a, n))
            for key in z['raw']:np.testing.assert_array_equal(got['raw'][key], z['raw'][key][:n])
            np.testing.assert_array_equal(got['available'], z['available'][:n])
            np.testing.assert_array_equal(got['local_available'], z['local_available'][:n])
        for i in range(len(a)):
            if isinstance(a[i], dict):
                for key in a[i]:np.testing.assert_array_equal(a[i][key], before[i][key])
            else:np.testing.assert_array_equal(a[i], before[i])

    def test_local_failure_aborts_and_never_discards_used_history(self):
        a = fixture()
        with patch.object(m, 'local_corrections', side_effect=FloatingPointError('local failure')):
            with self.assertRaises(FloatingPointError):run(a)
        with patch.object(m, 'local_corrections', return_value=(np.inf, 0., {'weight_mass': 1.})):
            with self.assertRaises(ValueError):run(a)

    def test_actual_month_security_identity_duplicates_rejected(self):
        a = fixture(); a[2][1] = 'A'
        with self.assertRaisesRegex(ValueError, 'one row per signal month'):run(a)
        # A duplicate in a later, unqueried annual cohort cannot spoil the past.
        a = query_slice(fixture(), 3); a[2][-1] = a[2][-2]; run(a)

    def test_strict_shapes_ids_targets_and_positive_integer_minima(self):
        for option in ('min_months', 'min_own_months'):
            for value in (0, True, 1.5):
                with self.assertRaises(ValueError):m.annual_forecasts(*fixture(), **{option: value})
        for index, change in ((3, lambda x: x[:, :9]), (8, lambda x: x[:, :9]),
                              (9, lambda x: x.astype(int)), (2, lambda x: x[:-1]),
                              (7, lambda x: np.array(['']*len(x))), (5, lambda x: x[:, None])):
            a = fixture(); a[index] = change(a[index])
            with self.assertRaises(ValueError):run(a)
        a = fixture(); del a[4]['vol_scaled']
        with self.assertRaises(ValueError):run(a)
        a = fixture(); a[1][0] = a[0][0]
        with self.assertRaises(ValueError):run(a)
        for path in m.dependency_files():self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
