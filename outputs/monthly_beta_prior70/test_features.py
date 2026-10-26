"""Synthetic-only beta-prior numerics, causal windows, masks and source contracts."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('beta_prior_features_test', P / 'features.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
ETFS = ['ARKK', 'GLD', 'IGV', 'IWM', 'MTUM', 'QQQ', 'SMH', 'SOXX', 'SPY',
        'TLT', 'VGT', 'XLE', 'XLF', 'XLI', 'XLK', 'XLV', 'XLY']
SS = sorted(['S%02d' % i for i in range(32)] + ETFS)


def prices(returns):
    return np.r_[100., 100. * np.cumprod(1. + returns)]


def sample_returns():
    rng = np.random.default_rng(810081)
    market = rng.normal(.0003, .012, 378)
    sector = .4 * market + rng.normal(.0001, .009, 378)
    own = .0008 + .65 * market + 1.3 * sector + rng.normal(0., .008, 378)
    return own, market, sector


def fixture():
    import exchange_calendars as xcals
    index = xcals.get_calendar('XNYS', start='2010-01-01', end='2013-01-01').sessions.tz_localize(None)[:650]
    names = sorted(set(SS + ['XLB', 'XLC', 'XLRE', 'XLU', 'XLP']))
    rng = np.random.default_rng(81001)
    px = pd.DataFrame(40. * np.cumprod(1. + .0007 + rng.normal(0., .011, (len(index), len(names))), axis=0),
                      index=index, columns=names)
    close = px * 1.17
    panel = dict(adj=px, close=close, open=close * 1.001, high=close * 1.03,
        low=close * .97, volume=pd.DataFrame(1e6, index=index, columns=names),
        sectors=pd.DataFrame('Technology', index=index, columns=names))
    dates = [str(index[i].date()) for i in (420, 480, 560)]
    meta = dict(dates=dates, symbols=[SS.copy() for _ in dates], etfs=ETFS.copy(),
                ends=dates[1:] + [str(index[580].date())])
    return panel, meta


def event(panel, t):
    return dict(dates=[str(panel['adj'].index[t].date())], symbols=[SS.copy()], etfs=ETFS.copy())


class WindowTests(unittest.TestCase):
    def test_independent_lstsq_and_posterior_all252_windows(self):
        path = [prices(value) for value in sample_returns()]
        raw, diag = f.window_features(*path)
        own, market, sector = [value[1:] / value[:-1] - 1. for value in path]
        expected = np.empty((252, 3))
        for k in range(252):
            X = np.column_stack((market[k:k + 126], sector[k:k + 126]))
            y = own[k:k + 126]
            design = np.column_stack((np.ones(126), X))
            coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
            centered_x, centered_y = X - X.mean(axis=0), y - y.mean()
            scatter = np.einsum('ni,nj->ij', centered_x, centered_x, optimize=False)
            cross = np.einsum('ni,n->i', centered_x, centered_y, optimize=False)
            error = y - np.einsum('ni,i->n', design, coefficient, optimize=False)
            noise = np.sum(error ** 2) / 123.
            posterior = np.linalg.solve(scatter + 4. * noise * np.eye(2), cross + 4. * noise * np.array([0., 1.]))
            np.testing.assert_allclose(diag['beta_ols'][k], coefficient[1:], atol=5e-14, rtol=0.)
            np.testing.assert_allclose(diag['beta_shrunk'][k], posterior, atol=2e-14, rtol=0.)
            self.assertAlmostEqual(diag['noise_variance'][k], noise, delta=1e-18)
            m, s, y = market[k + 126], sector[k + 126], own[k + 126]
            expected[k] = [y - coefficient[1] * m - coefficient[2] * s,
                           y - s, y - posterior[0] * m - posterior[1] * s]
        self.assertEqual(diag['status'], 'available')
        np.testing.assert_allclose(diag['residuals'], expected, atol=2e-15, rtol=0.)
        for j, arm in enumerate(f.ARMS):
            self.assertAlmostEqual(raw[arm], expected[:, j].sum() / expected[:, j].std(ddof=1), delta=2e-10)

    def test_known_betas_keep_drift_not_fitted_intercept(self):
        _, market, sector = sample_returns()
        own = .002 + .7 * market + 1.4 * sector
        raw, diag = f.window_features(prices(own), prices(market), prices(sector))
        np.testing.assert_allclose(diag['beta_ols'], np.tile([.7, 1.4], (252, 1)), atol=1e-13, rtol=0.)
        np.testing.assert_allclose(diag['residuals'][:, 0], .002, atol=1e-15, rtol=0.)
        self.assertEqual(diag['status'], 'degenerate_residual')
        self.assertTrue(all(np.isnan(value) for value in raw.values()))

    def test_exact_zero_noise_posterior_is_ols(self):
        _, market, sector = sample_returns()
        _, diag = f.window_features(np.full(379, 100.), prices(market), prices(sector))
        np.testing.assert_array_equal(diag['noise_variance'], np.zeros(252))
        np.testing.assert_array_equal(diag['beta_ols'], diag['beta_shrunk'])

    def test_fixed_prior_limit_and_zero_noise_matrix_reference(self):
        scatter = np.array([[[2., .3], [.3, 1.]]])
        cross = np.array([[4., -1.]])
        expected = np.linalg.solve(scatter[0], cross[0])
        np.testing.assert_allclose(f.posterior_betas(scatter, cross, [0.])[0], expected, atol=5e-16)
        np.testing.assert_allclose(f.posterior_betas(scatter, cross, [1e12])[0], [0., 1.], atol=2e-12, rtol=0.)
        with self.assertRaises(ValueError):
            f.posterior_betas(scatter, cross, [-1.])

    def test_collinear_or_zero_factor_invalidates_all_arms(self):
        own, market, sector = sample_returns()
        for same in (market, np.zeros(378)):
            raw, diag = f.window_features(prices(own), prices(market), prices(same))
            self.assertEqual(diag['status'], 'invalid_joint_fit')
            self.assertTrue(all(np.isnan(value) for value in raw.values()))

    def test_any_one_arm_constant_invalidates_all(self):
        _, market, sector = sample_returns()
        raw, diag = f.window_features(prices(sector), prices(market), prices(sector))
        self.assertEqual(diag['status'], 'degenerate_residual')
        self.assertEqual(diag['residual_std'][1], 0.)
        self.assertTrue(all(np.isnan(value) for value in raw.values()))

    def test_each_path_endpoint_and_nonfinite_is_required(self):
        for which in range(3):
            for where, value in ((0, np.nan), (126, 0.), (127, -1.), (378, np.inf)):
                paths = [prices(x) for x in sample_returns()]
                paths[which][where] = value
                raw, diag = f.window_features(*paths)
                self.assertEqual(diag['status'], 'invalid_' + ('stock', 'spy', 'sector')[which] + '_path')
                self.assertTrue(all(np.isnan(v) for v in raw.values()))
        with self.assertRaises(ValueError):
            f.window_features(*[prices(x)[:-1] for x in sample_returns()])

    def test_fit_lag_excludes_current_return(self):
        own, market, sector = sample_returns()
        _, before = f.window_features(prices(own), prices(market), prices(sector))
        own[126] += .2
        _, after = f.window_features(prices(own), prices(market), prices(sector))
        np.testing.assert_array_equal(before['beta_ols'][0], after['beta_ols'][0])
        np.testing.assert_array_equal(before['beta_shrunk'][0], after['beta_shrunk'][0])
        self.assertGreater(np.max(np.abs(before['beta_ols'][1] - after['beta_ols'][1])), .001)

    def test_fixed_price_scales_and_sign_reversal_about_sector_prior(self):
        own, market, sector = sample_returns()
        paths = [prices(value) for value in (own, market, sector)]
        raw, diag = f.window_features(*paths)
        scaled, other = f.window_features(*(p * s for p, s in zip(paths, (1e60, .013, 1e-60))))
        for arm in f.ARMS:
            self.assertAlmostEqual(raw[arm], scaled[arm], delta=1e-10)
        opposite, flipped = f.window_features(prices(2. * sector - own), prices(market), prices(sector))
        for arm in f.ARMS:
            self.assertAlmostEqual(opposite[arm], -raw[arm], delta=1e-10)
        np.testing.assert_allclose(flipped['noise_variance'], diag['noise_variance'], atol=1e-17, rtol=0.)


class PanelTests(unittest.TestCase):
    def test_exact379prices_offsets_and_fullstock_ranks(self):
        panel, meta = fixture()
        arrays, audit = f.build_features(panel, meta)
        self.assertEqual(audit['path_prices'], 379)
        self.assertEqual(audit['beta_returns'], 126)
        self.assertEqual(audit['formation_returns'], 252)
        for i, day in enumerate(meta['dates']):
            t = panel['adj'].index.get_loc(pd.Timestamp(day))
            row = audit['records'][i]
            for field, position in [('first_price', t - 420), ('last_price', t - 42),
                    ('first_fit_last_return', t - 294), ('first_formation_return', t - 293),
                    ('last_fit_last_return', t - 43)]:
                self.assertEqual(row[field], str(panel['adj'].index[position].date()))
            self.assertEqual(row['available'], 32)
            j = SS.index('S00')
            paths = [panel['adj'][name].iloc[t - 420:t - 41].to_numpy() for name in ('S00', 'SPY', 'XLK')]
            expected, _ = f.window_features(*paths)
            for arm in f.ARMS:
                self.assertEqual(arrays['raw_' + arm][i, j], expected[arm])
                ranks = pd.Series(arrays['raw_' + arm][i]).rank(method='average', pct=True).to_numpy() - .5
                np.testing.assert_array_equal(arrays['rank_' + arm][i], ranks)
        for prefix in ('raw_', 'rank_'):
            for arm in f.ARMS:
                np.testing.assert_array_equal(np.isfinite(arrays[prefix + arm]), arrays['available'])
                self.assertTrue(np.isnan(arrays[prefix + arm][~arrays['available']]).all())

    def test_first_valid_offset_and_one_day_short(self):
        panel, _ = fixture()
        short, _ = f.build_features(panel, event(panel, 419))
        full, _ = f.build_features(panel, event(panel, 420))
        self.assertFalse(short['available'].any())
        self.assertEqual(int(full['available'].sum()), 32)

    def test_recent42_future_and_unused_old_values_are_not_read(self):
        panel, _ = fixture()
        meta = event(panel, 480)
        ref, _ = f.build_features(panel, meta)
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].iloc[:60] = 'unused old'
        panel['adj'].iloc[439:] = 'unused recent and future'
        got, _ = f.build_features(panel, meta)
        for key in got:
            self.assertTrue(f.same_bytes(got[key], ref[key]), key)

    def test_signal_date_mapping_is_fixed_over_whole_path(self):
        panel, meta = fixture()
        base, _ = f.build_features(panel, meta)
        panel['sectors'][:] = 'Healthcare'
        for day in meta['dates']:
            panel['sectors'].loc[day] = 'Technology'
        got, audit = f.build_features(panel, meta)
        for key in got:
            self.assertTrue(f.same_bytes(got[key], base[key]), key)
        self.assertTrue(all(row['mapping']['S00'] == 'XLK' for row in audit['records']))

    def test_unknown_etf_missing_and_shared_failure_masks(self):
        panel, _ = fixture()
        meta = event(panel, 480)
        panel['sectors'].loc[meta['dates'][0], 'S00'] = 'Unknown'
        panel['adj'].iloc[200, panel['adj'].columns.get_loc('S01')] = np.nan
        got, audit = f.build_features(panel, meta)
        self.assertEqual(audit['records'][0]['status']['S00'], 'unknown_sector')
        self.assertEqual(audit['records'][0]['status']['S01'], 'invalid_stock_path')
        for symbol in ETFS + ['S00', 'S01']:
            j = SS.index(symbol)
            self.assertFalse(got['available'][0, j])
            for prefix in ('raw_', 'rank_'):
                for arm in f.ARMS:
                    self.assertTrue(np.isnan(got[prefix + arm][0, j]))

    def test_strict_mapper_foreign_priority_and_etf_override(self):
        mapper = f.benchmark_mapper()
        frame = pd.DataFrame({'ASML': ['Healthcare'], 'S00': ['Technology']}, index=pd.to_datetime(['2020-01-31']))
        self.assertEqual(mapper('2020-01-31', ['ASML', 'S00', 'OTHER'], frame, set()),
                         {'ASML': 'XLK', 'S00': 'XLK', 'OTHER': None})
        self.assertIsNone(mapper('2020-01-31', ['ASML'], frame, {'ASML'})['ASML'])

    def test_invalid_pool_axes_and_observed_objects_fail(self):
        panel, meta = fixture()
        meta['symbols'][0][0] = meta['symbols'][0][1]
        with self.assertRaises(ValueError):
            f.build_features(panel, meta)
        panel, meta = fixture()
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].iloc[100, panel['adj'].columns.get_loc('S00')] = 'invalid observed'
        with self.assertRaises(ValueError):
            f.build_features(panel, meta)


class CalendarTests(unittest.TestCase):
    def fixture(self):
        import exchange_calendars as xcals
        sessions = xcals.get_calendar('XNYS', start='2024-12-01', end='2025-02-28').sessions.tz_localize(None)
        sessions = sessions[sessions != pd.Timestamp('2025-01-09')]
        frame = pd.DataFrame({'A': 1.}, index=sessions)
        return frame, dict(dates=['2024-12-31', '2025-02-28'])

    def test_official_closure_and_half_session_accepted_without_global_mutation(self):
        import exchange_calendars as xcals
        before = xcals.get_calendar('XNYS', start='2024-12-01', end='2025-02-28').sessions.copy()
        frame, meta = self.fixture()
        self.assertIn(pd.Timestamp('2024-12-24'), frame.index)
        audit = f.validate_calendar(frame, meta)
        self.assertTrue(audit['complete_session_axis'])
        self.assertFalse(audit['global_registry_modified'])
        after = xcals.get_calendar('XNYS', start='2024-12-01', end='2025-02-28').sessions
        self.assertTrue(before.equals(after))

    def test_missing_true_session_and_extra_closure_rejected(self):
        frame, meta = self.fixture()
        with self.assertRaisesRegex(ValueError, 'complete XNYS'):
            f.validate_calendar(frame.drop(pd.Timestamp('2025-01-10')), meta)
        frame.loc[pd.Timestamp('2025-01-09')] = 1.
        with self.assertRaisesRegex(ValueError, 'complete XNYS'):
            f.validate_calendar(frame.sort_index(), meta)

    def test_future_index_is_irrelevant_to_prefix_validation(self):
        frame, meta = self.fixture()
        prefix = dict(dates=['2024-12-31'])
        frame = frame.drop(frame.index[frame.index > '2025-01-15'])
        self.assertTrue(f.validate_calendar(frame, prefix)['complete_session_axis'])


class IntegrationTests(unittest.TestCase):
    def test_prepare_exact11keys_cached_baseline_and_no_mutation(self):
        panel, meta = fixture()
        before = {k: value.copy(deep=True) for k, value in panel.items()}
        adapter = f._module('beta_prior_synthetic_baseline', f.BASELINE_PATH)
        original = adapter.prepare(panel, list(zip(meta['dates'], meta['symbols'])), ETFS)
        cache = dict(baseline=original['score'].copy(), core=original['core'].copy(),
            core_weight=original['cw'][:, :1].copy(), eligible=original['eligible'].copy())
        cache['baseline'][0, 0] = np.nextafter(cache['baseline'][0, 0], np.inf)
        arrays, audit = f.prepare(panel, meta, baseline_cache=cache)
        self.assertEqual(set(arrays), set(f.DATA_KEYS))
        self.assertEqual(len(arrays), 11)
        self.assertTrue(f.same_bytes(arrays['baseline'], cache['baseline']))
        for key in ('core', 'cw', 'eligible'):
            self.assertTrue(f.same_bytes(arrays[key], original[key]))
        self.assertGreater(audit['baseline_max_abs_error'], 0.)
        for key in panel:
            pd.testing.assert_frame_equal(panel[key], before[key])

    def test_prepare_prefix_and_future_object_poison(self):
        panel, meta = fixture()
        full, _ = f.prepare(panel, meta)
        for n in (1, 2, 3):
            poison = {key: value.copy(deep=True) for key, value in panel.items()}
            for key in ('adj', 'open', 'close', 'high', 'low', 'volume'):
                poison[key] = poison[key].astype(object)
                poison[key].loc[poison[key].index > meta['dates'][n - 1]] = 'future'
            prefix = dict(meta, dates=meta['dates'][:n], symbols=meta['symbols'][:n], ends=meta['ends'][:n])
            got, _ = f.prepare(poison, prefix)
            for key in full:
                self.assertTrue(f.same_bytes(got[key], full[key][:n]), (key, n))

    def test_invalid_new_path_does_not_change_original_eligibility(self):
        panel, meta = fixture()
        panel['adj'].iloc[5, panel['adj'].columns.get_loc('S00')] = np.nan
        adapter = f._module('beta_prior_missing_synthetic_baseline', f.BASELINE_PATH)
        source = adapter.prepare(panel, list(zip(meta['dates'], meta['symbols'])), ETFS)
        arrays, _ = f.prepare(panel, meta)
        self.assertTrue(f.same_bytes(arrays['eligible'], source['eligible']))
        self.assertFalse(arrays['available'][0, SS.index('S00')])

    def test_source_metadata_bindings_verify_and_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / 'outputs'
            current, source = output / 'monthly_beta_prior70', output / 'monthly_local_linear70'
            current.mkdir(parents=True)
            source.mkdir(parents=True)
            raw = {}
            for i in range(64):
                path = source / ('raw_%d' % i)
                path.write_text('synthetic input %d' % i)
                raw[str(path)] = f.sha(path)
            review = root / 'review.txt'
            review.write_text('synthetic review')
            (current / 'DESIGN.json').write_text(json.dumps(dict(source_reviews={'review.txt': f.sha(review)})))
            (source / 'PLAN.json').write_text(json.dumps(dict(raw_source_hashes=raw)))
            with patch.multiple(f, P=current, O=output, SOURCE=source, DESIGN_SHA=f.sha(current / 'DESIGN.json')):
                self.assertEqual(f._verified_sources(), dict(raw, **{str(review): f.sha(review)}))
                review.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Immutable feature source'):
                    f._verified_sources()


if __name__ == '__main__':
    unittest.main()
