import copy
import hashlib
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import monthly_cmf_state200_independent as oracle


def fixture():
    dates = pd.bdate_range('2013-09-02', '2016-11-30')
    t = np.arange(len(dates), dtype=float)
    spy = 90 * np.cumprod(1 + .0003 + (.004 + .008 * (t / len(t))) * np.sin(.41 * t))
    adj = pd.DataFrame({'SPY': spy, 'A': 40 + t*.021, 'B': 50 + t*.013,
                        'FUND': 70 + t*.009}, index=dates)
    low = adj * .98
    high = adj * 1.02
    close = low + (high-low) * pd.Series({'SPY': .5, 'A': .8, 'B': .2, 'FUND': .5})
    volume = pd.DataFrame(100 + (t[:, None] % 31) * np.array([[1., 2., 3., 4.]]),
                          index=dates, columns=adj.columns)
    panel = dict(adj=adj, close=close, high=high, low=low, volume=volume)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30'],
                symbols=[['A', 'B', 'FUND'] for _ in range(3)], etfs=['FUND'])
    return panel, meta, dates.strftime('%Y-%m-%d').tolist()


def arrays_for(values, future=0):
    n, width = values['cmf_rank'].shape
    full = n + future
    calm = np.r_[values['calm'], np.zeros(future, bool)]
    return dict(state_calm10=calm, state_weak_narrow=np.zeros(full, bool),
                core_weight=np.where(calm, .5, .52)[:, None],
                extension_available=np.ones((full, width), bool),
                forecast_available=np.ones((full, width), bool), eligible=np.ones((full, width), bool),
                rank_log_excess=np.zeros((full, width), float))


def regime_values():
    windows = [np.r_[np.full(251, .004), .005], np.r_[np.full(251, .004), .005],
               np.r_[np.full(251, .004), .005], np.r_[np.full(251, .004), .005],
               np.r_[np.full(251, .025), .02], np.r_[np.full(251, .015), .02],
               np.r_[np.full(251, .015), .02], np.r_[np.full(251, .025), .02]]
    n, width = len(windows), 3
    values = {k: np.empty(n) for k in ('state', 'gate_candidate', 'gate_control', 'coefficient', 'sigma', 'median')}
    values.update(cmf_rank=np.tile(np.array([-.5, 0., .5]), (n, 1)),
                  calm=np.empty(n, bool), sigma_window=np.asarray(windows))
    for i, window in enumerate(windows):
        g = oracle.gate_oracle(window)
        for key in ('state', 'gate_candidate', 'gate_control', 'sigma', 'median'):
            values[key][i] = g[key]
        calm = g['sigma'] * np.sqrt(252) < .10
        values['calm'][i] = calm
        values['coefficient'][i] = .075 if calm else .075 * (1 - (.10 if g['state'] > 0 else 0.) / .48)
    arrays = arrays_for(values)
    arrays['state_weak_narrow'][3] = True
    arrays['state_weak_narrow'][6:8] = True
    arrays['extension_available'][1] = False
    return values, arrays


class Tests(unittest.TestCase):
    def test_scalar_gate_hand_counts_and_median_ties(self):
        for window in (np.r_[np.zeros(50), np.ones(202)],
                       np.r_[np.full(50, 2.), np.ones(202)], np.zeros(252)):
            g = oracle.gate_oracle(window)
            self.assertEqual(g['gate_candidate'], 0.)
            self.assertEqual(g['state'], -1.)
            self.assertEqual(g['gate_control'], -.5)
        g = oracle.gate_oracle(np.r_[np.zeros(126), np.ones(126)])
        self.assertEqual((g['less'], g['equal'], g['median'], g['gate_candidate']), (126, 126, .5, .5))

    def test_gate_extremes_and_invalid(self):
        for x, sign in ((np.arange(252.), 1.), (np.arange(252.)[::-1], -1.)):
            result = oracle.gate_oracle(x)
            self.assertAlmostEqual(result['gate_candidate'], sign*251/252)
            self.assertLess(abs(result['gate_candidate']), 1)
        for x in (np.zeros(251), np.full(252, np.nan), np.full(252, -1.), np.full(252, np.inf)):
            with self.assertRaises(ValueError):
                oracle.gate_oracle(x)

    def test_original_percentile_missing_ties(self):
        np.testing.assert_array_equal(oracle.rank_oracle([1., 1., 2., np.nan]), [0., 0., .5, 0.])
        np.testing.assert_array_equal(oracle.rank_oracle([2., 2., 2.]), [2/3-.5] * 3)
        np.testing.assert_array_equal(oracle.rank_oracle([np.nan, 2.]), [0., .5])
        np.testing.assert_array_equal(oracle.rank_oracle([np.nan, np.nan]), [0., 0.])

    def test_source_windows_and_raw_cmf_direction(self):
        p, m, c = fixture()
        v, a = oracle.independent_features(p, m, c)
        for i, row in enumerate(a['event_history']):
            t = c.index(row['date'])
            self.assertEqual(row['spy_window_start'], c[t-314])
            self.assertEqual(row['sigma_window_start'], c[t-251])
            self.assertEqual(row['spy_window_end'], c[t])
            self.assertLess(v['cmf_rank'][i, 0], v['cmf_rank'][i, 2])
            self.assertLess(v['cmf_rank'][i, 2], v['cmf_rank'][i, 1])
        self.assertEqual(a['spy_price_observations'], 945)
        self.assertEqual(a['volatility_rank_observations'], 756)
        receipt = oracle.verify_features(p, m, c, v, a)
        self.assertTrue(receipt['canonical_sigma_checked'])
        self.assertFalse(receipt['computes_returns'])

    def test_cmf_zero_range_and_missing_neutral(self):
        p, m, c = fixture()
        p['volume']['A'] = 0.
        p['high']['B'] = p['low']['B']
        p['close']['B'] = np.nan  # Original where(spread!=0,0) behavior is retained.
        p['close']['FUND'] = np.nan
        v, a = oracle.independent_features(p, m, c)
        np.testing.assert_array_equal(v['cmf_rank'], [[0., .5, 0.]] * 3)
        self.assertEqual(a['finite_cmf_cells'], 3)
        self.assertEqual(a['neutral_cmf_cells'], 6)

    def test_constant_spy_calm_and_no_implicit_missing(self):
        p, m, c = fixture()
        p['adj']['SPY'] = 100.
        v, a = oracle.independent_features(p, m, c)
        np.testing.assert_array_equal(v['sigma_window'], np.zeros((3, 252)))
        np.testing.assert_array_equal(v['gate_candidate'], np.zeros(3))
        np.testing.assert_array_equal(v['state'], -np.ones(3))
        self.assertTrue(v['calm'].all())

    def test_prefix_truncation_and_future_poison(self):
        p, m, c = fixture()
        v, a = oracle.independent_features(p, m, c)
        for n in range(4):
            small = {k: f.iloc[:0] if n == 0 else f.loc[:m['dates'][n-1]] for k, f in p.items()}
            z, b = oracle.independent_features(small, m, c, n)
            for key in v:
                np.testing.assert_array_equal(z[key], v[key][:n])
            self.assertEqual(b['event_history'], a['event_history'][:n])
        for key in p:
            p[key] = p[key].astype(object)
            p[key].loc[p[key].index > m['dates'][0], :] = 'UNREAD_FUTURE'
        z, b = oracle.independent_features(p, m, c, 1)
        for key in v:
            np.testing.assert_array_equal(z[key], v[key][:1])
        p = {k: f.astype(object).map(lambda x: 'UNREAD') for k, f in p.items()}
        z, b = oracle.independent_features(p, m, c, 0)
        self.assertEqual(z['cmf_rank'].shape, (0, 3))

    def test_missing_extra_official_axis_and_panel_axes(self):
        for mode in ('missing', 'extra', 'axes'):
            p, m, c = fixture()
            if mode == 'missing':
                p = {k: f.drop(pd.Timestamp('2016-08-01')) for k, f in p.items()}
            elif mode == 'extra':
                p = {k: pd.concat([f, f.iloc[[-1]].set_axis(pd.DatetimeIndex(['2016-08-06']))]).sort_index()
                     for k, f in p.items()}
            else:
                p['volume'] = p['volume'].iloc[1:]
            with self.assertRaises(ValueError):
                oracle.independent_features(p, m, c)

    def test_invalid_spy_and_nonnumeric_prefix(self):
        for x in (0., -1., np.nan, np.inf, 'bad'):
            p, m, c = fixture()
            if isinstance(x, str):
                p['adj'] = p['adj'].astype(object)
            p['adj'].loc['2016-08-01', 'SPY'] = x
            with self.assertRaises((ValueError, FloatingPointError)):
                oracle.independent_features(p, m, c)

    def test_missing_symbols_monthend_and_insufficient_history(self):
        p, m, c = fixture()
        with self.assertRaises(ValueError):
            oracle.independent_features(dict(p, adj=p['adj'].drop(columns='SPY')), m, c)
        with self.assertRaises(ValueError):
            oracle.independent_features(dict(p, high=p['high'].drop(columns='A')), m, c)
        bad = copy.deepcopy(m)
        bad['dates'][1] = '2016-08-30'
        with self.assertRaises(ValueError):
            oracle.independent_features(p, bad, c)
        young = dict(dates=['2013-09-30'], symbols=[['A', 'B', 'FUND']], etfs=['FUND'])
        with self.assertRaises(ValueError):
            oracle.independent_features(p, young, c)

    def test_corrupt_artifact_rejected_exactly(self):
        p, m, c = fixture()
        v, a = oracle.independent_features(p, m, c)
        for key in ('cmf_rank', 'coefficient', 'sigma', 'sigma_window', 'gate_candidate'):
            bad = copy.deepcopy(v)
            flat = bad[key].reshape(-1)
            flat[-1] = np.nextafter(flat[-1], np.inf)
            with self.assertRaises(ValueError):
                oracle.verify_features(p, m, c, bad, a)
        bad = copy.deepcopy(a)
        bad['event_history'][0]['less'] += 1
        with self.assertRaises(ValueError):
            oracle.verify_features(p, m, c, v, bad)

    def test_input_immutability(self):
        p, m, c = fixture()
        before = {k: v.copy(deep=True) for k, v in p.items()}
        v, a = oracle.independent_features(p, m, c)
        oracle.verify_features(p, m, c, v, a)
        for key in p:
            pd.testing.assert_frame_equal(p[key], before[key])

    def test_seven_states_original_support_subtractions(self):
        v, a = regime_values()
        d, evidence = oracle.independent_deltas(v, a)
        expected_scale = [None, .95, .93, .85, 1., 1., 1-.05/.48, 1-.05/.48]
        for i in range(1, 8):
            # Isolated original arithmetic: support subtraction is a linear share
            # of the CMF contribution; no source-derived data are loaded.
            original_cmf = v['coefficient'][i] * v['cmf_rank'][i, 2] * v['state'][i]
            retained_cmf = original_cmf * expected_scale[i]
            new_cmf = retained_cmf * v['gate_control'][i] / v['state'][i]
            self.assertAlmostEqual(d['cmf_half_sign_control'][i, 2], new_cmf-retained_cmf, places=17)
        self.assertEqual(evidence['effective_support_scale_min'], .85)
        self.assertEqual(evidence['effective_support_scale_max'], 1.)
        receipt = oracle.verify_deltas(v, a, d)
        self.assertTrue(receipt['effective_support_scale_checked'])
        for arm in d:
            np.testing.assert_array_equal(d[arm][0], np.zeros(3))
            np.testing.assert_array_equal(d[arm][:, 1], np.zeros(8))
            self.assertLessEqual(np.max(np.abs(d[arm])), .0375)

    def test_active_mask_variations_and_no_outer_eligibility_filter(self):
        v, a = regime_values()
        a['extension_available'][2, 0] = False
        a['forecast_available'][2, 1] = False
        a['eligible'][2, 2] = False
        a['rank_log_excess'][1, 0] = np.nan
        d, e = oracle.independent_deltas(v, a)
        # Eligibility controls only inherited model-support subtraction, not CMF correction admission.
        self.assertNotEqual(d['cmf_half_sign_control'][2, 2], 0.)
        for bad in (np.nan, np.inf, -np.inf):
            a['rank_log_excess'][2, 2] = bad
            oracle.independent_deltas(v, a)

    def test_delta_weight_shapes_prefix_future_and_zero_prefix(self):
        p, m, c = fixture()
        v, _ = oracle.independent_features(p, m, c)
        a = arrays_for(v)
        d, _ = oracle.independent_deltas(v, a)
        for cw in (a['core_weight'][:, 0], a['core_weight'], np.repeat(a['core_weight'], 3, axis=1)):
            aa = dict(a, core_weight=cw)
            oracle.verify_deltas(v, aa, d)
        vv = {key: value[:1] for key, value in v.items()}
        aa = copy.deepcopy(a)
        aa['core_weight'] = aa['core_weight'].astype(object)
        aa['core_weight'][1:] = 'UNREAD_FUTURE'
        aa['rank_log_excess'] = aa['rank_log_excess'].astype(object)
        aa['rank_log_excess'][1:] = 'UNREAD_FUTURE'
        oracle.verify_deltas(vv, aa, {arm: z[:1] for arm, z in d.items()})
        vv = {key: value[:0] for key, value in v.items()}
        dd, _ = oracle.independent_deltas(vv, aa)
        oracle.verify_deltas(vv, aa, dd)

    def test_bad_original_arrays_and_delta_mutation(self):
        v, a = regime_values()
        d, _ = oracle.independent_deltas(v, a)
        before = {k: z.copy() for k, z in a.items()}
        oracle.verify_deltas(v, a, d)
        for key in a:
            np.testing.assert_array_equal(a[key], before[key])
        for key, value in (('core_weight', np.full((8, 1), .51)),
                           ('eligible', np.ones((8, 3), float)),
                           ('state_calm10', ~a['state_calm10'])):
            with self.assertRaises(ValueError):
                oracle.verify_deltas(v, dict(a, **{key: value}), d)
        bad = copy.deepcopy(d)
        bad['cmf_empirical_state'][1, 0] = np.nextafter(bad['cmf_empirical_state'][1, 0], np.inf)
        with self.assertRaises(ValueError):
            oracle.verify_deltas(v, a, bad)

    def test_pinned_accounting_dependency_and_no_producer_import(self):
        self.assertEqual(hashlib.sha256(oracle.FROZEN_HELPER.read_bytes()).hexdigest(), oracle.FROZEN_HELPER_SHA)
        self.assertTrue(callable(oracle.accounting().audit_path))
        source = Path(oracle.__file__).read_text()
        self.assertNotIn('import monthly_cmf_state200_kernel', source)

    def test_two_fixture_production_full_audit_and_all_prefix_cross(self):
        # Producer imports only in this synthetic test, never in the oracle.
        import monthly_cmf_state200_kernel as production
        from test_monthly_cmf_state200_kernel import Tests as ProducerTests
        for make in (fixture, ProducerTests().fixture):
            p, m, c = make()
            for n in range(len(m['dates'])+1):
                small = {key: f.iloc[:0].copy() if n == 0 else f.loc[:m['dates'][n-1]].copy()
                         for key, f in p.items()}
                v, a = production.build_features(small, m, c, n)
                receipt = oracle.verify_features(small, m, c, v, a, n)
                self.assertEqual(receipt['events'], n)


if __name__ == '__main__':
    unittest.main()
