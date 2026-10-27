"""Synthetic-only clock replication, maturity, parity and prefix tests."""
from copy import deepcopy
import hashlib
import importlib.util
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import monthly_matured_clock273_kernel as k

_spec = importlib.util.spec_from_file_location('clock273_synthetic_original',
    k.ROOT / 'outputs/monthly_target_definition70/model.py')
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)


def fixture():
    names = list('ABCD'); width = len(names)
    days = pd.date_range('2010-12-31', '2017-12-31', freq='ME')
    dates = np.repeat(days.values.astype('datetime64[D]'), width)
    ends = np.repeat((days + pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), width)
    rng = np.random.default_rng(273)
    X = rng.uniform(-.48, .48, (len(dates), 10))
    y = .18 * X[:, 0] - .11 * X[:, 1] + .06 * X[:, 5] ** 2
    # Deliberate 2016 drift makes the update meaningful without an outcome test.
    recent = dates >= np.datetime64('2016-01-01')
    y[recent] += .35 * X[recent, 4]
    y = (y.reshape(-1, width) - y.reshape(-1, width).mean(axis=1)[:, None]).reshape(-1)
    targets = {key: (j + 1) * y.copy() for j, key in enumerate(k.TARGETS)}
    good = np.ones(len(dates), bool); good[5] = False
    for values in targets.values():
        values[5] = np.nan
    q = ['2016-08-17', '2016-08-31', '2016-09-30', '2017-01-31', '2017-02-28', '2017-03-31']
    qe = ['2016-08-31', '2016-09-30', '2016-10-31', '2017-02-28', '2017-03-31', '2017-04-28']
    Q = rng.uniform(-.48, .48, (len(q) * width, 10))
    E = np.ones((len(q), width), bool); E[3, 1] = False
    meta = dict(monthly=dict(dates=q, ends=qe, symbols=[names.copy() for _ in q], etfs=['D']),
                training=dict(dates=days.strftime('%Y-%m-%d').tolist(),
                              ends=(days + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d').tolist(),
                              symbols=[names.copy() for _ in days]))
    train = dict(dates=dates, ends=ends, train_plain=X, common_good=good,
                 event_dates=np.repeat(np.asarray(q, dtype='datetime64[D]'), width), event_plain=Q,
                 offsets=np.arange(len(days) + 1) * width,
                 **{'target_' + key: value for key, value in targets.items()})
    arrays = dict(baseline=np.zeros(E.shape), eligible=E, state_weak_narrow=np.array([True, True, False, True, True, True]))
    refresh(train, meta, arrays)
    # One incomplete local pair must retain the original pooled slot unchanged.
    arrays['raw_local_linear'][4, 0] = np.nan
    arrays['paired_month_ready'][4] = False
    return train, meta, arrays


def refresh(train, meta, arrays):
    n, width = arrays['baseline'].shape
    with threadpool_limits(limits=1):
        out = original.annual_forecasts(train['dates'], train['ends'], train['train_plain'],
            {key: train['target_' + key] for key in k.TARGETS}, train['common_good'],
            train['event_dates'], train['event_plain'], arrays['eligible'].reshape(-1))
    raw = out['log_excess'].reshape(n, width)
    av = out['available'].reshape(n, width)
    arrays.update(raw_source_log=raw.copy(), forecast_available=av.copy(),
                  rank_log_excess=k._rank(raw, av & arrays['eligible'], np.zeros((n, width))),
                  model_available=av.copy(), raw_local_linear=raw.copy(), raw_local_constant=raw.copy(),
                  paired_month_ready=(av & arrays['eligible']).any(axis=1))


def scalar_oracle(X, y, dates, cutoff, Q):
    """Scalar weight/statistic accumulation plus augmented least squares.

    Independent of producer normal equations. Fixed comparison atol/rtol2e-10
    covers alternate floating operation order; rank and annual parity stay exact.
    """
    months = [str(d)[:7] for d in dates]
    counts = {month: months.count(month) for month in set(months)}
    cy, cm = map(int, str(cutoff)[:7].split('-'))
    weights = []
    for month in months:
        year, m = map(int, month.split('-'))
        weights.append((2. ** (-((cy - year) * 12 + cm - m) / 12.)) / counts[month])
    total = math.fsum(weights)
    w = np.asarray([value * len(weights) / total for value in weights])
    mass = math.fsum(w)
    mean = np.asarray([math.fsum(float(w[i]) * float(X[i, j]) for i in range(len(X))) / mass for j in range(10)])
    scale = np.asarray([math.sqrt(math.fsum(float(w[i]) * (float(X[i, j]) - mean[j]) ** 2
                                          for i in range(len(X))) / mass) for j in range(10)])
    scale[scale <= 1e-12] = 1.
    a = math.fsum(float(w[i]) * float(y[i]) for i in range(len(y))) / mass
    Z = (X - mean) / scale
    lhs = np.vstack([np.sqrt(w)[:, None] * Z, math.sqrt(10.) * np.eye(10)])
    rhs = np.r_[np.sqrt(w) * (y - a), np.zeros(10)]
    beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    pred = np.asarray([math.fsum(((Q[j] - mean) / scale * beta).tolist()) + a for j in range(len(Q))])
    return w, mean, scale, beta, a, pred


class KernelTests(unittest.TestCase):
    def test_exact_original_annual_and_active_only_monthly(self):
        t, m, a = fixture(); b = k.fit_models(t, m, a)
        self.assertEqual(tuple(b), k.BUNDLE_KEYS)
        for key, expected in [('raw', 'raw_source_log'), ('available', 'forecast_available'), ('rank', 'rank_log_excess')]:
            np.testing.assert_array_equal(b['control'][key], a[expected])
        self.assertEqual([r['query_indices'] for r in b['candidate']['audit']['fits']], [[1], [3], [5]])
        self.assertEqual(b['audit']['counts']['annual_ridge_fits'], 2)
        self.assertEqual(b['audit']['counts']['monthly_ridge_fits'], 3)
        self.assertEqual(b['audit']['counts']['local_model_fits'], 0)
        rank, delta, audit = k.rank_and_delta(a, m['monthly'], b)
        self.assertEqual(tuple(audit), k.RANK_AUDIT_KEYS)
        for i in (0, 2, 4):
            np.testing.assert_array_equal(rank[i], a['rank_log_excess'][i]); self.assertTrue((delta[i] == 0).all())
        self.assertEqual(audit['active_indices'][3], [0, 2, 3])
        active = delta != 0
        np.testing.assert_array_equal(delta[active], .0375 * (rank - a['rank_log_excess'])[active])

    def test_maturity_boundary_window_and_full_clock(self):
        t, m, a = fixture(); b = k.fit_models(t, m, a)
        for r in b['candidate']['audit']['fits']:
            self.assertEqual(tuple(r), k.FIT_KEYS)
            cutoff = np.datetime64(r['cutoff']); lower = np.datetime64(r['lower_bound'])
            ids = np.asarray(r['training_row_ids'])
            self.assertTrue((t['ends'][ids] <= cutoff).all())
            self.assertTrue((t['dates'][ids] > lower).all())
            self.assertIn(str(cutoff), r['label_end_dates'])
            self.assertEqual(r['weights_sha256'], hashlib.sha256(np.asarray(r['weights'], float).tobytes()).hexdigest())
        self.assertTrue(b['audit']['full_clock_change']); self.assertFalse(b['audit']['strict_pit'])

    def test_scalar_weight_and_augmented_oracle(self):
        t, m, a = fixture(); b = k.fit_models(t, m, a)
        for r in b['control']['audit']['fits'] + b['candidate']['audit']['fits']:
            ids, qi = r['training_row_ids'], r['query_row_ids']
            w, mean, scale, beta, intercept, pred = scalar_oracle(t['train_plain'][ids], t['target_log_excess'][ids],
                t['dates'][ids], r['cutoff'], t['event_plain'][qi])
            for expected, actual in ((w,r['weights']), (mean,r['mean']), (scale,r['scale']), (beta,r['beta']), (pred,r['raw_predictions'])):
                np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
            self.assertAlmostEqual(r['intercept'], intercept, places=12)

    def test_control_gate_precedes_every_candidate_fit(self):
        t, m, a = fixture(); seen = []
        real = k._record
        def record(s, kind, *args):
            seen.append(kind); return real(s, kind, *args)
        def gate(view):
            self.assertIsNone(view['candidate']); self.assertIsNone(view['audit']['control_gate'])
            self.assertEqual(seen, ['annual', 'annual'])
            self.assertEqual(view['audit']['counts']['monthly_ridge_fits'], 0)
            return dict(passed=True)
        with patch.object(k, '_record', side_effect=record):
            k.fit_models(t, m, a, control_validator=gate)
        self.assertEqual(seen, ['annual', 'annual', 'monthly', 'monthly', 'monthly'])

    def test_bad_raw_blocks_gate_and_candidates(self):
        t, m, a = fixture(); a['raw_source_log'][1, 0] += .01
        a['rank_log_excess'] = k._rank(a['raw_source_log'], a['eligible'] & a['forecast_available'], a['rank_log_excess'])
        with patch.object(k, '_record', wraps=k._record) as record, patch('builtins.print'):
            with self.assertRaises(AssertionError): k.fit_models(t, m, a, control_validator=lambda _: self.fail('gate invoked'))
            self.assertTrue(all(call.args[1] == 'annual' for call in record.call_args_list))

    def test_bad_availability_blocks_candidate(self):
        t, m, a = fixture(); a['forecast_available'][1, 0] = False
        a['rank_log_excess'] = k._rank(a['raw_source_log'], a['eligible'] & a['forecast_available'], a['rank_log_excess'])
        with self.assertRaises(AssertionError): k.fit_models(t, m, a)

    def test_gate_false_and_mutation_fail(self):
        t, m, a = fixture()
        with patch.object(k, '_record', wraps=k._record) as record:
            with self.assertRaises(ValueError): k.fit_models(t, m, a, control_validator=lambda _: dict(passed=False))
            self.assertTrue(all(call.args[1] == 'annual' for call in record.call_args_list))
        def mutate(_):
            t['target_log_excess'][50] += .1
            return dict(passed=True)
        with self.assertRaisesRegex(ValueError, 'changed by control gate'): k.fit_models(t, m, a, control_validator=mutate)

    def test_monthly_insufficient_history_has_no_annual_fallback(self):
        t, m, a = fixture()
        # Annual2015 contains18months, but queryAug2016 rolls out seven of them.
        keep = (t['dates'] >= np.datetime64('2011-01-01')) & (t['dates'] <= np.datetime64('2012-06-30'))
        t['common_good'] &= keep
        refresh(t, m, a)
        with self.assertRaisesRegex(ValueError, 'no fallback'): k.fit_models(t, m, a, n=2)

    def test_solver_failure_no_lstsq(self):
        t, m, a = fixture()
        with patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('synthetic solve failure')):
            with patch.object(np.linalg, 'lstsq', side_effect=AssertionError('forbidden fallback')):
                with self.assertRaises(np.linalg.LinAlgError): k.fit_models(t, m, a)

    def test_nonfinite_solver_and_overflow_fail(self):
        t, m, a = fixture()
        with patch.object(np.linalg, 'solve', return_value=np.full(10, np.nan)):
            with self.assertRaisesRegex(ValueError, 'coefficients'): k.fit_models(t, m, a)
        with self.assertRaises((ValueError, FloatingPointError)):
            k.fit_ridge(np.full((20, 10), 1e308), np.ones(20), np.ones(20))

    def test_future_labels_do_not_affect_prefix(self):
        t, m, a = fixture(); first = k.fit_models(t, m, a, n=2)
        future = t['ends'] > np.datetime64('2016-08-31')
        for key in k.TARGETS: t['target_' + key][future] = 987654.
        t['train_plain'][future] = np.inf
        changed = k.fit_models(t, m, a, n=2)
        self.assertEqual(first['candidate']['raw'].tobytes(), changed['candidate']['raw'].tobytes())
        self.assertEqual(first['candidate']['audit'], changed['candidate']['audit'])

    def test_physical_prefix_matches_cached_and_logical(self):
        t, m, a = fixture(); full = k.fit_models(t, m, a)
        for n in (0, 1, 2, 4, 6):
            logical = k.fit_models(t, m, a, n=n)
            pt, pm, pa, scope = k.physical_inputs(t, m, a, n)
            small = k.fit_models(pt, pm, pa)
            for role in ('control', 'candidate'):
                for field in ('raw', 'available', 'rank'):
                    self.assertEqual(logical[role][field].tobytes(), small[role][field].tobytes())
                    self.assertEqual(logical[role][field].tobytes(), full[role][field][:n].tobytes())
            left = k.rank_and_delta(a, m['monthly'], full, n=n)
            right = k.rank_and_delta(pa, pm['monthly'], small)
            np.testing.assert_array_equal(left[1], right[1])
            self.assertEqual(left[2], right[2])
            self.assertEqual(scope['consumed_training_rows'], len(pt['dates']))

    def test_empty_and_initial_only_no_fit_or_numeric_conversion(self):
        t, m, a = fixture()
        t['train_plain'] = np.full(t['train_plain'].shape, 'future-poison', object)
        t['event_plain'] = np.full(t['event_plain'].shape, 'future-poison', object)
        for key in k.TARGETS: t['target_' + key] = np.full(t['target_' + key].shape, 'future-poison', object)
        with patch.object(k, 'fit_ridge', side_effect=AssertionError('no fit')):
            for n in (0, 1):
                out = k.fit_models(t, m, a, n=n)
                self.assertEqual(out['audit']['counts']['actual_model_fits'], 0)
                self.assertEqual(out['candidate']['raw'].shape, (n, 4))

    def test_common_allfour_and_minimum_two_cohort(self):
        t, m, a = fixture()
        cohort = np.flatnonzero(t['dates'] == np.datetime64('2015-06-30'))
        t['target_loss_utility'][cohort[:3]] = np.nan
        refresh(t, m, a)
        b = k.fit_models(t, m, a)
        for r in b['control']['audit']['fits'] + b['candidate']['audit']['fits']:
            self.assertTrue(set(cohort).isdisjoint(r['training_row_ids']))
        self.assertTrue(b['audit']['label_preservation']['labels_unchanged'])
        self.assertFalse(b['audit']['label_preservation']['recentered'])

    def test_no_availability_expansion_from_cold_annual(self):
        t, m, a = fixture(); t['common_good'][t['dates'] < np.datetime64('2014-08-01')] = False
        refresh(t, m, a); b = k.fit_models(t, m, a)
        self.assertFalse(b['candidate']['available'][1].any())
        self.assertNotIn([1], [r['query_indices'] for r in b['candidate']['audit']['fits']])

    def test_no_weak_candidates_uses_only_annual_maturity(self):
        t, m, a = fixture(); a['state_weak_narrow'][:] = False
        b = k.fit_models(t, m, a)
        self.assertEqual(b['candidate']['audit']['fits'], [])
        self.assertEqual(b['audit']['source_scope']['latest_needed_cutoff'], '2016-12-31')
        self.assertEqual(b['candidate']['raw'].tobytes(), b['control']['raw'].tobytes())

    def test_exact_rank_ties_and_constant_features(self):
        t, m, a = fixture(); t['train_plain'][:, 3] = .1
        t['event_plain'][:] = .05
        refresh(t, m, a); b = k.fit_models(t, m, a)
        np.testing.assert_array_equal(b['candidate']['rank'][1], np.full(4, .125))
        for r in b['candidate']['audit']['fits']: self.assertEqual(r['scale'][3], 1.)

    def test_rank_and_delta_rejects_inactive_raw_or_support_mutation(self):
        t, m, a = fixture(); b = k.fit_models(t, m, a)
        b['candidate']['raw'][2, 0] += .1
        b['candidate']['rank'] = k._rank(b['candidate']['raw'], a['eligible'] & a['forecast_available'], a['rank_log_excess'])
        with self.assertRaises(AssertionError): k.rank_and_delta(a, m['monthly'], b)
        b = k.fit_models(t, m, a); b['candidate']['available'][3, 1] = True
        with self.assertRaises(AssertionError): k.rank_and_delta(a, m['monthly'], b)

    def test_paired_flag_must_match_original_local_support(self):
        t, m, a = fixture(); a['paired_month_ready'][4] = True
        with self.assertRaises(AssertionError): k.fit_models(t, m, a)

    def test_metadata_identity_offsets_and_boolean_validation(self):
        for problem in ('ids', 'offset', 'bool', 'querydate'):
            t, m, a = fixture()
            if problem == 'ids': m['training']['symbols'][30][0] = 'B'
            if problem == 'offset': t['offsets'][30] += 1
            if problem == 'bool': a['eligible'] = a['eligible'].astype(int)
            if problem == 'querydate': t['event_dates'][4] += np.timedelta64(1, 'D')
            with self.assertRaises(ValueError): k.fit_models(t, m, a)
        t, m, a = fixture()
        for n in (True, -1, 7, 1.5):
            with self.assertRaises(ValueError): k.fit_models(t, m, a, n=n)

    def test_consumed_rank_bounds_fail_not_clip(self):
        t, m, a = fixture(); t['event_plain'][4, 1] = .50001
        with self.assertRaisesRegex(ValueError, 'rank bounds'): k.fit_models(t, m, a)

    def test_years_before_leap_and_lower_exact_exclusion(self):
        self.assertEqual(str(k.numeric._years_before(np.datetime64('2020-02-29'))), '2015-02-28')
        t, m, a = fixture(); b = k.fit_models(t, m, a)
        r = b['candidate']['audit']['fits'][0]
        boundary = np.flatnonzero(t['dates'] == np.datetime64(r['lower_bound']))
        self.assertTrue(len(boundary)); self.assertTrue(set(boundary).isdisjoint(r['training_row_ids']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
