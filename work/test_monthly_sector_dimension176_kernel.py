"""Synthetic-only checks for round176; never reads research market inputs."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_dimension176_kernel as kernel


def market_fixture():
    rng = np.random.default_rng(176)
    common = rng.normal(0., .008, 1512)
    noise = rng.normal(0., .009, (1512, 9))
    # Deterministic change in shared movement, without any payoff targeting.
    loading = np.where(np.arange(1512) < 1260, .2, 1.5)
    returns = common[:, None] * loading[:, None] + noise
    return np.exp(np.vstack([np.zeros(9), np.cumsum(returns, axis=0)])) * 100.


def relative_fixture():
    x = np.arange(210.)
    sector = .0002 + .005 * np.sin(x * .23)
    relative = .0004 + .007 * np.cos(x * .17)
    base = 100. * np.exp(np.r_[0., np.cumsum(sector)])
    stock = 100. * np.exp(np.r_[0., np.cumsum(sector + relative)])
    return stock, base, relative


def panel_fixture():
    days = pd.bdate_range('2000-01-03', periods=1650)
    frame = pd.DataFrame(100., index=days,
                         columns=list(kernel.SECTOR_ETFS) + ['A', 'B', 'XLC', 'XLRE', 'FUND'])
    # Event1600 needs the inclusive span46..1558, exactly1513 prices.
    frame.loc[days[46:1559], list(kernel.SECTOR_ETFS)] = market_fixture()
    stock, sector, _ = relative_fixture()
    frame.loc[days[1348:1559], 'A'] = stock
    frame.loc[days[1348:1559], 'B'] = stock
    frame.loc[days[1348:1559], 'XLC'] = sector
    frame.loc[days[1348:1559], 'XLRE'] = sector
    sectors = pd.DataFrame('Communication Services', index=days, columns=['A', 'B'])
    sectors['B'] = 'Real Estate'
    meta = dict(dates=[str(days[1600].date())], symbols=[['A', 'B', 'FUND']], etfs=['FUND'])
    return {'adj': frame, 'sectors': sectors}, meta, days.strftime('%Y-%m-%d').tolist()


def mapper(day, names, sectors, etfs):
    labels = {'Communication Services': 'XLC', 'Real Estate': 'XLRE', 'Technology': 'XLK'}
    return {s: None if s in etfs or s not in sectors else labels.get(sectors.at[pd.Timestamp(day), s])
            for s in names}


class MarketTests(unittest.TestCase):
    def test_exact_windows_pair_order_and_historical_mean(self):
        out = kernel.measure_market(market_fixture())
        self.assertTrue(out['available'])
        self.assertEqual(len(out['windows']), 61)
        self.assertEqual((out['windows'][0]['start_index'], out['windows'][0]['end_index']), (1260, 1512))
        self.assertEqual((out['windows'][60]['start_index'], out['windows'][60]['end_index']), (0, 252))
        self.assertEqual(len(out['windows'][0]['pair_correlations']), 36)
        self.assertEqual(kernel.PAIRS[:3], ((0, 1), (0, 2), (0, 3)))
        self.assertAlmostEqual(out['historical_dispersion_mean'], np.mean(out['dispersion_values'][1:]), 14)
        self.assertNotAlmostEqual(out['historical_dispersion_mean'], np.mean(out['dispersion_values']), 7)
        self.assertEqual(out['correlation_window_estimates'], 61)
        self.assertEqual(out['pair_estimates'], 2196)

    def test_matches_independent_numpy_correlation_formula(self):
        prices = market_fixture()
        out = kernel.measure_market(prices)
        for k in [0, 1, 30, 60]:
            a, b = 1260 - 21*k, 1512 - 21*k
            returns = np.diff(np.log(prices[a:b+1]), axis=0)
            correlation = np.corrcoef(returns, rowvar=False)
            expected = 1. - sum(correlation[i, j]**2 for i, j in kernel.PAIRS)/36.
            self.assertAlmostEqual(out['dispersion_values'][k], expected, 13)
            for actual, (i, j) in zip(out['windows'][k]['pair_correlations'], kernel.PAIRS):
                self.assertAlmostEqual(actual, correlation[i, j], 13)

    def test_identical_or_opposite_series_have_zero_dispersion(self):
        p = market_fixture()[:, 0]
        for signs in [np.ones(9), np.array([1., -1.]*4 + [1.])]:
            prices = np.exp(np.log(p)[:, None] * signs[None, :])
            out = kernel.measure_market(prices)
            self.assertTrue(out['available'])
            self.assertLess(max(out['dispersion_values']), 2e-14)

    def test_orthogonal_periodic_returns_have_unit_dispersion(self):
        t = np.arange(1512)
        returns = np.column_stack([.01*np.sin(2*np.pi*k*t/252.) for k in range(1, 10)])
        prices = np.exp(np.vstack([np.zeros(9), np.cumsum(returns, axis=0)]))
        out = kernel.measure_market(prices)
        self.assertTrue(out['available'])
        for value in out['dispersion_values']:
            self.assertAlmostEqual(value, 1., 13)

    def test_one_invalid_window_keeps_other_diagnostics(self):
        prices = market_fixture()
        prices[:253, 0] = 100.
        out = kernel.measure_market(prices)
        self.assertFalse(out['available'])
        self.assertIn(60, out['invalid_window_indices'])
        self.assertTrue(out['windows'][0]['available'])
        self.assertEqual(out['windows'][60]['pair_correlations'], [])
        self.assertIsNone(out['current_dispersion'])
        self.assertIsNone(out['historical_dispersion_mean'])
        self.assertEqual(out['correlation_window_estimates'], 61-len(out['invalid_window_indices']))

    def test_all_daily_prices_required_not_only_window_endpoints(self):
        for idx in [0, 1, 251, 900, 1512]:
            p = market_fixture()
            p[idx, 4] = np.nan
            out = kernel.measure_market(p)
            self.assertEqual(out['reason'], 'invalid_market_price')
            self.assertEqual(out['windows'], [])

    def test_zero_negative_infinite_and_nonnumeric_sources(self):
        for bad in [0., -1., np.inf, -np.inf]:
            p = market_fixture()
            p[3, 2] = bad
            self.assertEqual(kernel.measure_market(p)['reason'], 'invalid_market_price')
        p = market_fixture().astype(object)
        p[1, 1] = 'bad'
        self.assertEqual(kernel.measure_market(p)['reason'], 'nonnumeric_market_price')

    def test_shape_is_contract_failure(self):
        for shape in [(1512, 9), (1513, 8), (9, 1513)]:
            with self.assertRaises(ValueError):
                kernel.measure_market(np.ones(shape))

    def test_strict_standard_deviation_floor(self):
        for level, allowed in [(kernel.STD_FLOOR, False), (kernel.STD_FLOOR*1.001, True)]:
            def center_override(x):
                mean, residuals, sd, squares = original(x)
                return mean, residuals, level, squares
            original = kernel._center
            with patch.object(kernel, '_center', side_effect=center_override):
                self.assertEqual(kernel.measure_market(market_fixture())['available'], allowed)

    def test_roundoff_clamps_only_within_fixed_tolerance(self):
        self.assertEqual(kernel._bounded(1.+5e-13, -1., 1., 'rho'), 1.)
        self.assertEqual(kernel._bounded(-5e-13, 0., 1., 'D'), 0.)
        for value in [1.+2e-12, -1.-2e-12, np.inf, np.nan]:
            with self.assertRaises(ArithmeticError):
                kernel._bounded(value, -1., 1., 'rho')

    def test_unit_changes_preserve_correlations(self):
        p = market_fixture()
        expected = kernel.measure_market(p)
        scales = np.logspace(-250., 250., 9)
        observed = kernel.measure_market(p * scales[None, :])
        np.testing.assert_allclose(observed['dispersion_values'], expected['dispersion_values'], atol=1e-13)


class RelativeTests(unittest.TestCase):
    def test_quality_uses_sample_std_and_annualization(self):
        stock, sector, relative = relative_fixture()
        out = kernel.measure_relative(stock, sector)
        self.assertTrue(out['available'])
        self.assertAlmostEqual(out['relative_mean'], np.mean(relative), 15)
        self.assertAlmostEqual(out['relative_standard_deviation'], np.std(relative, ddof=1), 15)
        expected = math.sqrt(252.)*np.mean(relative)/np.std(relative, ddof=1)
        self.assertAlmostEqual(out['relative_sharpe'], expected, 12)
        self.assertAlmostEqual(out['quality'], math.tanh(expected), 12)

    def test_reversing_stock_and_sector_reverses_quality(self):
        stock, sector, _ = relative_fixture()
        a, b = kernel.measure_relative(stock, sector), kernel.measure_relative(sector, stock)
        self.assertAlmostEqual(a['quality'], -b['quality'], 14)

    def test_identical_series_and_tiny_variance_unavailable(self):
        stock, sector, _ = relative_fixture()
        self.assertEqual(kernel.measure_relative(stock, stock)['reason'], 'relative_volatility_floor')
        for size, expected in [(1e-13, False), (1e-10, True)]:
            relative = size*np.sin(np.arange(210.))
            prices = np.exp(np.r_[0., np.cumsum(relative)])
            self.assertEqual(kernel.measure_relative(prices, np.ones(211))['available'], expected)

    def test_relative_exact_floor_is_not_valid(self):
        stock, sector, _ = relative_fixture()
        original = kernel._center
        def on_floor(x):
            mean, centered, _, ss = original(x)
            return mean, centered, kernel.STD_FLOOR, ss
        with patch.object(kernel, '_center', side_effect=on_floor):
            self.assertFalse(kernel.measure_relative(stock, sector)['available'])

    def test_all_relative_daily_prices_required(self):
        stock, sector, _ = relative_fixture()
        for idx, name in [(1, 'stock'), (80, 'benchmark'), (210, 'stock')]:
            a, b = stock.copy(), sector.copy()
            (a if name == 'stock' else b)[idx] = np.nan
            self.assertEqual(kernel.measure_relative(a, b)['reason'], 'invalid_'+name+'_price')

    def test_relative_nonnumeric_and_shape_errors(self):
        stock, sector, _ = relative_fixture()
        a = stock.astype(object)
        a[50] = 'bad'
        self.assertEqual(kernel.measure_relative(a, sector)['reason'], 'nonnumeric_stock_price')
        self.assertEqual(kernel.measure_relative(sector, a)['reason'], 'nonnumeric_benchmark_price')
        with self.assertRaises(ValueError):
            kernel.measure_relative(stock[:-1], sector)

    def test_logarithm_safe_across_finite_float_range(self):
        for start, end in [(1e-308, 1e308), (1e308, 1e-308), (1., np.nextafter(1., 2.))]:
            result = kernel._log_ratio(end, start)
            self.assertTrue(math.isfinite(result))
            self.assertAlmostEqual(result, math.log(end)-math.log(start), 12)


class BuilderTests(unittest.TestCase):
    def build(self, panel=None, meta=None, calendar=None, mapping=mapper, limit=None):
        p, m, c = panel_fixture()
        return kernel.build_features(p if panel is None else panel, m if meta is None else meta,
                                     c if calendar is None else calendar, mapping, limit=limit)

    def test_mapping_xlc_xlre_are_valid_and_etfs_neutral(self):
        values, audit = self.build()
        self.assertEqual(values['available'].tolist(), [[True, True, False]])
        event = audit['event_history'][0]
        self.assertEqual(event['benchmarks'], {'A': 'XLC', 'B': 'XLRE', 'FUND': None})
        self.assertEqual(event['cells'][2]['reason'], 'etf_neutral')
        self.assertEqual(audit['relative_quality_estimates'], 2)
        self.assertEqual(audit['pair_estimates'], 2196)
        for j in [0, 1]:
            cell = event['cells'][j]
            self.assertEqual(values['candidate'][0, j], cell['quality']*event['market']['current_dispersion'])
            self.assertEqual(values['control'][0, j], cell['quality']*event['market']['historical_dispersion_mean'])

    def test_date_positions_and_skip42_boundaries(self):
        p, m, c = panel_fixture()
        _, audit = self.build(p, m, c)
        event = audit['event_history'][0]
        self.assertEqual((event['market_start'], event['market_end']), (c[46], c[1558]))
        self.assertEqual((event['relative_start'], event['relative_end']), (c[1348], c[1558]))
        self.assertEqual(event['market']['windows'][60]['start_date'], c[46])
        self.assertEqual(event['market']['windows'][60]['end_date'], c[298])
        self.assertEqual(event['market']['windows'][0]['start_date'], c[1306])

    def test_missing_market_symbol_or_session_invalidates_common_support(self):
        for kind, reason in [('symbol', 'missing_market_symbol'), ('session', 'missing_market_session')]:
            p, m, c = panel_fixture()
            p['adj'] = p['adj'].drop(columns='XLE') if kind == 'symbol' else p['adj'].drop(index=p['adj'].index[100])
            values, audit = self.build(p, m, c)
            self.assertFalse(values['available'].any())
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], reason)
            self.assertEqual(audit['correlation_window_estimates'], 0)

    def test_missing_stock_or_benchmark_never_falls_back(self):
        for dropped, expected in [('A', 'missing_symbol'), ('XLC', 'missing_benchmark_symbol')]:
            p, m, c = panel_fixture()
            p['adj'] = p['adj'].drop(columns=dropped)
            values, audit = self.build(p, m, c)
            self.assertEqual(values['available'].tolist(), [[False, True, False]])
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], expected)

    def test_nonnumeric_own_sector_only_invalidates_affected_cell(self):
        p, m, c = panel_fixture()
        p['adj'] = p['adj'].astype(object)
        p['adj'].at[pd.Timestamp(c[1400]), 'XLC'] = 'bad'
        v, a = self.build(p, m, c)
        self.assertEqual(v['available'].tolist(), [[False, True, False]])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'nonnumeric_benchmark_price')

    def test_missing_classification_and_no_backfill(self):
        p, m, c = panel_fixture()
        p['sectors'].at[pd.Timestamp(m['dates'][0]), 'A'] = 'Unknown'
        v, a = self.build(p, m, c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'missing_benchmark')
        self.assertFalse(v['available'][0, 0])

    def test_all_global_inputs_required_even_for_single_stock(self):
        p, m, c = panel_fixture()
        p['adj'].at[pd.Timestamp(c[47]), 'XLB'] = np.nan
        v, a = self.build(p, m, c)
        self.assertFalse(v['available'].any())
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'invalid_market_price')
        self.assertIsNone(a['event_history'][0]['cells'][0]['quality'])

    def test_insufficient_history_and_etf_reason_precedence(self):
        p, m, c = panel_fixture()
        m['dates'] = [c[1500]]
        v, a = self.build(p, m, c)
        self.assertFalse(v['available'].any())
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells']],
                         ['insufficient_history', 'insufficient_history', 'etf_neutral'])

    def test_future_numeric_poison_and_truncation_are_invariant(self):
        p, m, c = panel_fixture()
        v, a = self.build(p, m, c)
        p['adj'] = p['adj'].astype(object)
        p['adj'].iloc[1559:] = 'future poison'
        vv, aa = self.build(p, m, c)
        self.assertEqual(a, aa)
        np.testing.assert_equal(v['candidate'], vv['candidate'])
        p['adj'] = p['adj'].iloc[:1559].copy()
        p['sectors'] = p['sectors'].loc[:m['dates'][0]].copy()
        vvv, aaa = self.build(p, m, c)
        self.assertEqual(a, aaa)
        np.testing.assert_equal(v['control'], vvv['control'])

    def test_mapper_never_sees_future_classifications(self):
        p, m, c = panel_fixture()
        calls = []
        def checked(day, symbols, sectors, etfs):
            self.assertLessEqual(sectors.index[-1], pd.Timestamp(day))
            calls.append(day)
            return mapper(day, symbols, sectors, etfs)
        self.build(p, m, c, mapping=checked)
        self.assertEqual(calls, m['dates'])

    def test_prefixes_and_counting_do_not_accumulate_calls(self):
        p, m, c = panel_fixture()
        m['dates'].append(c[1601])
        m['symbols'].append(m['symbols'][0][:])
        full, fa = self.build(p, m, c)
        short, sa = self.build(p, m, c, limit=1)
        np.testing.assert_equal(short['candidate'], full['candidate'][:1])
        self.assertEqual(sa['event_history'], fa['event_history'][:1])
        self.assertEqual(sa['correlation_window_estimates'], 61)
        self.assertEqual(fa['correlation_window_estimates'], 122)
        zero, za = self.build(p, m, c, limit=0)
        self.assertEqual(zero['candidate'].shape, (0, 3))
        self.assertEqual(za['event_history'], [])
        self.assertEqual(za['pair_estimates'], 0)

    def test_preserves_all_source_objects(self):
        p, m, c = panel_fixture()
        before, bm, bc = copy.deepcopy(p), copy.deepcopy(m), c[:]
        self.build(p, m, c)
        for key in p:
            pd.testing.assert_frame_equal(p[key], before[key])
        self.assertEqual(m, bm)
        self.assertEqual(c, bc)

    def test_rejects_ambiguous_or_changed_mapping_identity(self):
        for bad in [lambda *x: {}, lambda *x: {'A': 'XLC', 'B': 'XLRE', 'FUND': 'XLK'},
                    lambda *x: {'A': 5, 'B': 'XLRE', 'FUND': None}]:
            with self.assertRaises(ValueError):
                self.build(mapping=bad)

    def test_rejects_axis_metadata_and_calendar_errors(self):
        p, m, c = panel_fixture()
        p['adj'] = p['adj'].iloc[::-1]
        with self.assertRaises(ValueError):
            self.build(p, m, c)
        p, m, c = panel_fixture()
        m['symbols'] = [['A', 'A', 'FUND']]
        with self.assertRaises(ValueError):
            self.build(p, m, c)
        with self.assertRaises(ValueError):
            self.build(calendar=c[::-1])
        with self.assertRaises(ValueError):
            self.build(limit=True)

    def test_independent_scalar_oracle_checks_every_synthetic_audit_field(self):
        import monthly_sector_dimension176_independent as independent
        p, m, c = panel_fixture()
        for case in ('valid', 'missing_sector_history', 'invalid_market_window'):
            source = copy.deepcopy(p)
            if case == 'missing_sector_history':
                source['adj'].at[pd.Timestamp(c[1350]), 'XLC'] = np.nan
            elif case == 'invalid_market_window':
                source['adj'].loc[source['adj'].index[46:299], 'XLB'] = 100.
            values, audit = self.build(source, m, c)
            receipt = independent.verify_features(source, m, c, values, audit)
            self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['checked_cells'], 3)
            self.assertEqual(receipt['pair_estimates'], audit['pair_estimates'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
