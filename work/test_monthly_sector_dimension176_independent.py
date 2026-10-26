"""Synthetic-only sector-dimension and relative-price independent checks."""
import ast
import copy
from datetime import date
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_dimension176_independent as audit


def orthogonal_prices(count=1513):
    time = np.arange(count - 1)
    returns = np.array([.0002 + .003 * np.cos(2 * np.pi * k * time / 252.) for k in range(1, 10)]).T
    return np.exp(np.vstack([np.zeros(9), np.cumsum(returns, axis=0)]))


def fixture(width=2, count=2):
    index = pd.bdate_range('2008-01-01', periods=1650)
    calendar = index.strftime('%Y-%m-%d').tolist()
    symbols = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    etf_prices = orthogonal_prices(len(index))
    frame = pd.DataFrame(etf_prices, index=index, columns=audit.ETF9)
    for symbol in symbols:
        frame[symbol] = frame['XLK'] * np.exp(.0003 * np.arange(len(index)) + .003 * np.sin(np.arange(len(index)) / 5.))
    frame['XLC'] = frame['XLK'] * 1.1
    frame['XLRE'] = frame['XLK'] * .9
    dates = [calendar[t] for t in (1554, 1575)[:count]]
    sectors = pd.DataFrame('Technology', index=pd.to_datetime(dates), columns=symbols)
    meta = dict(dates=dates, symbols=[symbols[:] for _ in dates], etfs=['QQQ'])
    return {'adj': frame, 'sectors': sectors}, meta, calendar


def metrics(days, values):
    x = np.asarray(values)
    r = x[1:] / x[:-1] - 1.
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    sd = float(np.std(r, ddof=1))
    return dict(start=days[0], end=days[-1], years=years, final_equity=float(x[-1]),
        cagr=float((x[-1] / 100000.) ** (1 / years) - 1),
        sharpe=float(np.mean(r) / sd * np.sqrt(252.)) if sd > 0 else None,
        max_drawdown=float(np.min(x / np.maximum.accumulate(np.r_[100000., x])[1:] - 1)),
        trading_sessions=len(days), total_return=float(x[-1] / 100000. - 1))


def account_fixture():
    """Synthetic flat prices and three explicit fills; no engine invoked."""
    idx = pd.bdate_range('2016-08-17', '2026-09-24')
    days = idx.strftime('%Y-%m-%d').tolist()
    dates = ['2016-08-17'] + [days[i] for i in range(len(days) - 1) if days[i][:7] != days[i + 1][:7]]
    ends = dates[1:] + ['2026-09-24']
    meta = dict(dates=dates, ends=ends)
    px = pd.DataFrame(1., index=idx, columns=['S0', 'S1', 'S2'])
    orders = {arm: [dict(date=d, signal_date=d, end=e, symbols=['S0', 'S1', 'S2'], scores=[.3, .2, .1])
                   for d, e in zip(dates, ends)] for arm in audit.ARMS}
    exports, results = {}, {}
    for arm in audit.ARMS:
        exports[arm], results[arm] = {}, {}
        for path, execution, cost in audit.PATHS:
            executions = dates if execution == 'close' else [days[days.index(d) + 1] for d in dates]
            units = 100000. / (3. * (1 + cost))
            fee, cash = 3 * units * cost, 100000.
            for _ in range(3): cash -= units + units * cost
            nav = cash + math.fsum([units] * 3)
            trades = [dict(date=executions[0], side='buy', symbol=s, adjusted_units=units,
                           adjusted_price=1., dollars=units, cost=units * cost) for s in px.columns]
            holdings = [dict(execution_date=d, signal_date=o['date'], symbols=o['symbols'], scores=o['scores'],
                            fee=fee if i == 0 else 0.) for i, (d, o) in enumerate(zip(executions, orders[arm]))]
            daily, positions = [], []
            for d in days:
                active = d >= executions[0]
                equity = nav if active else 100000.
                daily.append(dict(date=d, equity=equity))
                positions.append(dict(date=d, cash=cash if active else 100000., equity=equity,
                    adjusted_units={s: units for s in px.columns} if active else {},
                    market_values={s: units for s in px.columns} if active else {}))
            vals = [r['equity'] for r in daily]
            slices = {'full122': metrics(days, vals)}
            for key, left, right in [('early53', 0, 52), ('recent35', 53, 87), ('later34', 88, 121), ('first88', 0, 87)]:
                lo, hi = days.index(dates[left]), days.index(ends[right])
                slices[key] = metrics(days[lo:hi + 1], [v / vals[lo] * 100000. for v in vals[lo:hi + 1]])
            annual = []
            for year in sorted({d[:4] for d in days}):
                ids = [i for i, d in enumerate(days) if d.startswith(year)]
                prior = vals[ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=vals[ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            result = dict(slices['full122'], fees_paid=fee, dollar_turnover=3 * units, trade_rows=3, annual_returns=annual)
            exports[arm][path] = dict(DAILY=daily, TRADES=trades, HOLDINGS=holdings, DAILY_POSITIONS=positions)
            results[arm][path] = dict(metrics=result, slices=slices,
                accounting=dict(exported_fill_ledger_matches=True, terminal_liquidation=False))
    return px, px.copy(), orders, exports, results, meta


class NumericTests(unittest.TestCase):
    def test_orthogonal_sectors_dimension_one_all61(self):
        r = audit.dimension_oracle(orthogonal_prices())
        self.assertTrue(r['available'])
        self.assertEqual(len(r['windows']), 61)
        self.assertEqual(r['correlation_window_estimates'], 61)
        self.assertEqual(r['pair_estimates'], 2196)
        for k, row in enumerate(r['windows']):
            self.assertEqual((row['start_index'], row['end_index']), (1260 - 21 * k, 1512 - 21 * k))
            self.assertEqual(len(row['pair_correlations']), 36)
            self.assertLess(max(abs(x) for x in row['pair_correlations']), 1e-12)
            self.assertAlmostEqual(row['dispersion'], 1., places=14)
        self.assertAlmostEqual(r['historical_dispersion_mean'], 1., places=14)

    def test_perfect_positive_and_negative_correlation_redundant(self):
        line = np.exp(.003 * np.sin(np.arange(1513) / 5.))
        prices = np.array([line if i % 2 else 1. / line for i in range(9)]).T
        r = audit.dimension_oracle(prices)
        self.assertTrue(r['available'])
        self.assertLess(max(r['dispersion_values']), 2e-14)
        self.assertTrue(any(x < -.99 for x in r['windows'][0]['pair_correlations']))

    def test_historical_mean_excludes_current_window(self):
        prices = orthogonal_prices(); original = audit.dimension_oracle(prices)
        prices[-21:, 1] = prices[-21:, 0]
        changed = audit.dimension_oracle(prices)
        self.assertNotEqual(original['current_dispersion'], changed['current_dispersion'])
        self.assertEqual(original['dispersion_values'][1:], changed['dispersion_values'][1:])
        self.assertEqual(original['historical_dispersion_mean'], changed['historical_dispersion_mean'])

    def test_lowvol_window_blocks_all_but_records_successful_windows(self):
        prices = orthogonal_prices(); prices[:253, 0] = 1.
        r = audit.dimension_oracle(prices)
        self.assertFalse(r['available']); self.assertEqual(r['reason'], 'market_volatility_floor')
        self.assertIn(60, r['invalid_window_indices'])
        self.assertEqual(len(r['windows']), 61)
        self.assertIsNone(r['current_dispersion']); self.assertIsNone(r['historical_dispersion_mean'])
        self.assertGreater(r['correlation_window_estimates'], 0)
        self.assertEqual(r['pair_estimates'], 36 * r['correlation_window_estimates'])

    def test_all_market_source_prices_required(self):
        for pos in ((0, 0), (1512, 8), (101, 5)):
            for bad in (np.nan, np.inf, 0., -1.):
                p = orthogonal_prices(); p[pos] = bad
                r = audit.dimension_oracle(p)
                self.assertFalse(r['available']); self.assertEqual(r['windows'], [])
        p = orthogonal_prices().astype(object); p[12, 5] = 'MISSING'
        self.assertEqual(audit.dimension_oracle(p)['reason'], 'nonnumeric_market_price')
        with self.assertRaises(ValueError): audit.dimension_oracle(np.ones((1512, 9)))

    def test_ddof1_relative_quality_and_zero_mean(self):
        ret = np.array([-.001, .003] * 105)
        stock = np.exp(np.r_[0., np.cumsum(ret)]); sector = np.ones(211)
        r = audit.quality_oracle(stock, sector)
        self.assertTrue(r['available'])
        expected_sd = math.sqrt(210 * .002 ** 2 / 209.)
        self.assertAlmostEqual(r['relative_mean'], .001, places=14)
        self.assertAlmostEqual(r['relative_standard_deviation'], expected_sd, places=14)
        self.assertAlmostEqual(r['quality'], math.tanh(math.sqrt(252.) * .001 / expected_sd), places=13)
        stock = np.exp(np.r_[0., np.cumsum(np.array([-.001, .001] * 105))])
        r = audit.quality_oracle(stock, sector)
        self.assertTrue(r['available']); self.assertAlmostEqual(r['quality'], 0., places=13)
        self.assertFalse(audit.quality_oracle(sector, sector)['available'])

    def test_relative_std_floor_and_sources(self):
        for scale, expected in ((1e-13, False), (1e-9, True)):
            stock = np.exp(np.r_[0., np.cumsum(np.array([-scale, scale] * 105))])
            self.assertEqual(audit.quality_oracle(stock, np.ones(211))['available'], expected)
        for col in (0, 1):
            args = [np.ones(211), np.ones(211)]; args[col][10] = np.nan
            r = audit.quality_oracle(*args)
            self.assertEqual(r['reason'], 'invalid_' + ('stock' if col == 0 else 'benchmark') + '_price')
        with self.assertRaises(ValueError): audit.quality_oracle(np.ones(212), np.ones(211))

    def test_unit_invariance(self):
        prices = orthogonal_prices(); a = audit.dimension_oracle(prices)
        b = audit.dimension_oracle(prices * np.array([2. ** (i * 100 - 400) for i in range(9)]))
        audit.close(a['dispersion_values'], b['dispersion_values'], 'units')
        stock = np.exp(np.r_[0., np.cumsum(np.array([-.001, .003] * 105))])
        a = audit.quality_oracle(stock, np.ones(211))
        b = audit.quality_oracle(stock * 2. ** 800, np.ones(211) * 2. ** -800)
        self.assertAlmostEqual(a['quality'], b['quality'], places=13)

    def test_roundoff_enforcement(self):
        self.assertEqual(audit.unit_interval(-5e-13, 'D'), 0.)
        self.assertEqual(audit.unit_interval(1. + 5e-13, 'D'), 1.)
        for value in (-2e-12, 1. + 2e-12, np.inf):
            with self.assertRaises(ValueError): audit.unit_interval(value, 'D')
        with patch.object(audit.math, 'sqrt', return_value=.01), self.assertRaises(ValueError):
            audit.correlation([1., 2., 3.], [1., 2., 3.])


class MappingAndPanelTests(unittest.TestCase):
    def test_frozen_constants_and_precedence(self):
        self.assertEqual(audit.mapping_constants()['sha256'], audit.MAPPING_SHA)
        with patch.object(audit, 'MAPPING_SHA', '0' * 64), self.assertRaises(ValueError): audit.mapping_constants()
        day = pd.Timestamp('2020-01-31')
        sector = pd.DataFrame({'SHEL': ['Technology'], 'A': ['Communication Services'], 'B': ['Real Estate']}, index=[day])
        self.assertEqual(audit.benchmark_for(day, 'SHEL', sector, []), 'XLE')
        self.assertIsNone(audit.benchmark_for(day, 'SHEL', sector, ['SHEL']))
        self.assertEqual(audit.benchmark_for(day, 'A', sector, []), 'XLC')
        self.assertEqual(audit.benchmark_for(day, 'B', sector, []), 'XLRE')
        self.assertIsNone(audit.benchmark_for(day, 'Z', sector, []))
        self.assertIsNone(audit.benchmark_for(day + pd.Timedelta(days=1), 'A', sector, []))

    def test_dates_counts_and_full_receipt(self):
        panel, meta, calendar = fixture(count=1)
        values, details = audit.independent_features(panel, meta, calendar)
        check = audit.verify_features(panel, meta, calendar, values, details)
        self.assertTrue(check['frozen_mapping_checked']); self.assertFalse(check['computes_returns'])
        self.assertEqual(check['correlation_window_estimates'], 61); self.assertEqual(check['pair_estimates'], 2196)
        self.assertEqual(check['relative_quality_estimates'], 1)
        row = details['event_history'][0]; t = calendar.index(meta['dates'][0])
        self.assertEqual((row['market_start'], row['market_end']), (calendar[t - 1554], calendar[t - 42]))
        self.assertEqual((row['relative_start'], row['relative_end']), (calendar[t - 252], calendar[t - 42]))
        for k, window in enumerate(row['market']['windows']):
            self.assertEqual(window['start_date'], calendar[t - 294 - 21 * k])
            self.assertEqual(window['end_date'], calendar[t - 42 - 21 * k])
        json.dumps(details, allow_nan=False); json.dumps(check, allow_nan=False)

    def test_common_market_failure_etf_and_unknown_neutral(self):
        panel, meta, calendar = fixture(count=1)
        panel['adj'].iloc[0, 0] = np.nan
        values, d = audit.independent_features(panel, meta, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(d['event_history'][0]['cells'][0]['reason'], 'invalid_market_price')
        self.assertEqual(d['event_history'][0]['cells'][1]['reason'], 'etf_neutral')
        panel['sectors'].iloc[0, 0] = 'UNKNOWN'
        _, d = audit.independent_features(panel, meta, calendar)
        self.assertEqual(d['event_history'][0]['cells'][0]['reason'], 'missing_benchmark')

    def test_stock_only_needs211_and_XLC_not_in_global9(self):
        panel, meta, calendar = fixture(count=1)
        panel['adj'].loc[:calendar[1301], 'S0'] = np.nan
        panel['sectors'].loc[meta['dates'][0], 'S0'] = 'Communication Services'
        values, d = audit.independent_features(panel, meta, calendar)
        self.assertTrue(values['available'][0, 0])
        self.assertEqual(d['event_history'][0]['benchmarks']['S0'], 'XLC')
        panel['adj'] = panel['adj'].drop(columns='XLC')
        values, d = audit.independent_features(panel, meta, calendar)
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(d['event_history'][0]['cells'][0]['reason'], 'missing_benchmark_symbol')
        self.assertTrue(d['event_history'][0]['market']['available'])

    def test_missing_global_symbol_session_and_stock(self):
        panel, meta, calendar = fixture(count=1)
        changed = dict(panel, adj=panel['adj'].drop(columns='XLE'))
        values, d = audit.independent_features(changed, meta, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(d['event_history'][0]['market']['reason'], 'missing_market_symbol')
        changed = dict(panel, adj=panel['adj'].drop(index=calendar[0]))
        _, d = audit.independent_features(changed, meta, calendar)
        self.assertEqual(d['event_history'][0]['market']['reason'], 'missing_market_session')
        changed = dict(panel, adj=panel['adj'].drop(columns='S0'))
        _, d = audit.independent_features(changed, meta, calendar)
        self.assertEqual(d['event_history'][0]['cells'][0]['reason'], 'missing_symbol')

    def test_prefixes_price_future_poison_and_map_future_poison(self):
        panel, meta, calendar = fixture()
        full, details = audit.independent_features(panel, meta, calendar)
        for n in range(3):
            part, audit_part = audit.independent_features(panel, meta, calendar, n)
            for key in full: np.testing.assert_array_equal(part[key], full[key][:n])
            self.assertEqual(audit_part['event_history'], details['event_history'][:n])
        changed = {'adj': panel['adj'].astype(object), 'sectors': panel['sectors'].copy()}
        changed['adj'].iloc[1513:] = 'FUTURE_POISON'
        changed['sectors'].iloc[1] = 'UNKNOWN'
        part, _ = audit.independent_features(changed, meta, calendar, 1)
        for key in full: np.testing.assert_array_equal(part[key], full[key][:1])

    def test_bad_audit_windows_mapping_quality_rejected(self):
        panel, meta, calendar = fixture(count=1)
        values, d = audit.independent_features(panel, meta, calendar)
        for key in ('dispersion', 'start_index', 'standard_deviations', 'pair_correlations'):
            bad = copy.deepcopy(d)
            bad['event_history'][0]['market']['windows'][0][key] = []
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(panel, meta, calendar, values, bad)
        bad = copy.deepcopy(d); bad['event_history'][0]['benchmarks']['S0'] = 'XLE'
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, calendar, values, bad)
        bad = copy.deepcopy(d); bad['event_history'][0]['cells'][0]['quality'] += .1
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, calendar, values, bad)
        bad = {k: v.copy() for k, v in values.items()}; bad['control'][0, 0] += .1
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, calendar, bad, d)

    def test_49_universe_and_schema(self):
        panel, meta, calendar = fixture(width=49, count=1)
        values, _ = audit.independent_features(panel, meta, calendar)
        self.assertEqual(values['available'].shape, (1, 49)); self.assertEqual(values['available'].sum(), 48)
        for bad in (-1, True, 2):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, calendar, bad)
        with self.assertRaises(ValueError): audit.independent_features(panel, meta, calendar[::-1])
        with self.assertRaises(ValueError): audit.independent_features(dict(panel, extra=None), meta, calendar)
        short = dict(meta, dates=[calendar[1553]])
        values, _ = audit.independent_features(panel, short, calendar)
        self.assertFalse(values['available'].any())

    def test_direct_delta_no_tanh_and_initial_and_missing_zero(self):
        values = dict(candidate=np.array([[.1, np.nan], [-.5, 0.]]), control=np.array([[.2, np.nan], [.4, 0.]]),
                      available=np.array([[True, False], [True, True]]))
        delta = audit.independent_deltas(values); r = audit.verify_deltas(values, delta)
        self.assertTrue(r['direct_linear_mapping']); self.assertEqual(delta[audit.ARMS[1]][1, 0], -.01)
        self.assertEqual(delta[audit.ARMS[1]][1, 1], 0.)
        self.assertTrue(values['available'][1, 1])
        for i, j in ((0, 0), (0, 1)):
            bad = copy.deepcopy(delta); bad[audit.ARMS[1]][i, j] = 1e-30
            with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, delta, dose=.01)
        values['candidate'][1, 0] = 1.1
        with self.assertRaises(ValueError): audit.independent_deltas(values)

    def test_no_production_import_or_entrypoint(self):
        source = Path(audit.__file__).read_text(); tree = ast.parse(source)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in n or 'host' in n for n in imports))
        self.assertNotIn('__main__', source)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.args = account_fixture()

    def test_nine_paths45_slices_and_public_accounting(self):
        result = audit.audit_outcomes(*self.args)
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(result['outcome_helper_sha256'], audit.OUTCOME_SHA)
        metric = self.args[4][audit.ARMS[0]]['same_close_zero']['slices']['first88']
        audit.accounting().check_metrics(metric, metric.copy(), 'synthetic training')
        json.dumps(result, allow_nan=False)

    def test_hash_binding_and_incorrect_fills_fail(self):
        with patch.object(audit, 'ACCOUNTING_SHA', '0' * 64), self.assertRaises(ValueError): audit.accounting()
        with patch.object(audit, 'OUTCOME_SHA', '0' * 64), self.assertRaises(ValueError): audit.audit_outcomes(*self.args)
        args = list(self.args); args[3] = copy.deepcopy(args[3])
        args[3][audit.ARMS[1]]['same_close_zero']['TRADES'][0]['adjusted_units'] += 1
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)
        args = list(self.args); args[4] = copy.deepcopy(args[4])
        args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['cagr'] += .1
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)


if __name__ == '__main__':
    unittest.main()
