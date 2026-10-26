"""Synthetic-only independent market adjustment, event, and accounting checks."""
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

import market_absorption174_independent as audit


def sparse_sources():
    prices = np.full(1513, 100.)
    market = np.exp(.001 * np.arange(1513) + .03 * np.sin(np.arange(1513) / 13.))
    close, volume = np.ones(1259), np.ones(1259)
    anchors = [1 + 42 * k for k in range(6)]
    for k, s in enumerate(anchors):
        prices[s + 252:s + 273] = 99.
        if k < 3:
            volume[s + 19] = 2.
    return prices, market, close, volume, anchors


def fixture(width=2, count=2):
    index = pd.bdate_range('2008-01-01', periods=1650)
    calendar = index.strftime('%Y-%m-%d').tolist()
    symbols = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    columns = symbols + ['SPY']
    panel = {key: pd.DataFrame(1., index=index, columns=columns) for key in ('adj', 'close', 'volume')}
    p, market, c, v, _ = sparse_sources()
    panel['adj'].iloc[:1513, -1] = market
    for s in symbols:
        panel['adj'].iloc[:1513, panel['adj'].columns.get_loc(s)] = p
        panel['close'].iloc[233:233 + 1259, panel['close'].columns.get_loc(s)] = c
        panel['volume'].iloc[233:233 + 1259, panel['volume'].columns.get_loc(s)] = v
    events = [(calendar[t], symbols[:]) for t in (1554, 1575)[:count]]
    return panel, events, calendar


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


class EventTests(unittest.TestCase):
    def test_six_events_offsets_and_raw_contrast(self):
        p, m, c, v, anchors = sparse_sources()
        result = audit.measure_oracle(p, m, c, v)
        self.assertTrue(result['available'])
        self.assertEqual(result['selected_anchors'], anchors)
        self.assertEqual(result['high_anchors'], anchors[:3])
        self.assertEqual(result['low_anchors'], anchors[3:])
        self.assertEqual(result['extended_anchor_indices'], [s + 252 for s in anchors])
        self.assertEqual(result['beta_first_indices'], [s - 1 for s in anchors])
        self.assertEqual(result['beta_last_indices'], [s + 251 for s in anchors])
        self.assertEqual(result['beta_attempts_count'], 6)
        self.assertEqual(result['beta_estimates_count'], 6)
        self.assertEqual(result['betas'][0], 0.)
        self.assertAlmostEqual(result['control'], 0., places=14)
        self.assertEqual(result['labels'], result['raw_labels'])
        for raw, beta, market, residual in zip(result['raw_labels'], result['betas'], result['market_labels'], result['residual_labels']):
            self.assertAlmostEqual(residual, raw - beta * market, places=14)

    def test_OLS_demeaned_intercept_not_uncentered(self):
        market = np.exp(np.cumsum(np.r_[0., .002 * np.sin(np.arange(252)) + .001]))
        stock = market ** 3 * np.exp(.0007 * np.arange(253))
        beta, variance, why = audit.beta_oracle(stock, market)
        self.assertEqual(why, 'available')
        self.assertGreater(variance, 0)
        self.assertAlmostEqual(beta, 3., places=12)
        beta, _, _ = audit.beta_oracle(market ** -2, market)
        self.assertAlmostEqual(beta, -2., places=12)
        beta, _, _ = audit.beta_oracle(np.ones(253), market)
        self.assertEqual(beta, 0.)

    def test_beta2_removes_market_move(self):
        increments = np.where(np.arange(1512) % 2 == 0, -.002, .003)
        m = np.exp(np.r_[0., np.cumsum(increments)])
        p = m * m
        c, v = np.ones(1259), np.ones(1259)
        anchors = list(range(1, 1240, 22))
        v[np.array(anchors[::2]) + 19] = 2.
        result = audit.measure_oracle(p, m, c, v)
        self.assertEqual(result['selected_anchors'], anchors)
        self.assertTrue(result['available'])
        for beta in result['betas']: self.assertAlmostEqual(beta, 2., places=12)
        self.assertLess(max(abs(x) for x in result['residual_labels']), 1e-13)
        self.assertAlmostEqual(result['candidate'], 0., places=13)

    def test_beta_excludes_current_anchor_and_all_future_prices(self):
        p, m, c, v, _ = sparse_sources()
        original = audit.measure_oracle(p, m, c, v)
        a = original['extended_anchor_indices'][0]
        p[a:] *= .9; m[a:] *= 1.2
        modified = audit.measure_oracle(p, m, c, v)
        self.assertEqual(modified['selected_anchors'][0], 1)
        self.assertEqual(modified['betas'][0], original['betas'][0])
        self.assertEqual(modified['market_variances'][0], original['market_variances'][0])
        # Last pre-anchor return matters: the window is not accidentally lagged two sessions.
        p[a - 1] = 101.
        self.assertNotEqual(audit.measure_oracle(p, m, c, v)['betas'][0], original['betas'][0])

    def test_common_cooldown_group_after_event_selection(self):
        p, m, c, v, _ = sparse_sources()
        p = np.exp(-.001 * np.arange(1513))
        anchors = list(range(1, 1240, 21))
        v[:] = 1.; v[np.array(anchors[::2]) + 19] = 2.
        result = audit.measure_oracle(p, m, c, v)
        self.assertEqual(result['selected_anchors'], anchors)
        self.assertEqual(result['high_anchors'], anchors[::2])
        self.assertEqual(result['low_anchors'], anchors[1::2])
        self.assertEqual(result['n_all'], 59)
        self.assertAlmostEqual(result['control'], -.021 * (30 / 50 - 29 / 49), places=13)

    def test_final_anchor_label_and_beta_boundaries(self):
        p, m, c, v, _ = sparse_sources()
        p[1491:1512] = 90.
        r = audit.measure_oracle(p, m, c, v)
        self.assertEqual(r['selected_anchors'][-1], 1239)
        self.assertEqual(r['beta_first_indices'][-1], 1238)
        self.assertEqual(r['beta_last_indices'][-1], 1490)
        self.assertAlmostEqual(r['raw_labels'][-1], math.log(100 / 90), places=13)

    def test_zero_market_variance_shared_unavailable(self):
        p, m, c, v, anchors = sparse_sources()
        m[:253] = 1.
        result = audit.measure_oracle(p, m, c, v)
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], 'invalid_beta')
        self.assertIn(1, result['invalid_beta_anchors'])
        self.assertIsNone(result['betas'][0]); self.assertIsNone(result['residual_labels'][0])
        self.assertEqual(result['beta_reasons'][0], 'zero_or_nonfinite_market_variance')
        self.assertEqual(result['selected_anchors'], anchors)
        self.assertEqual(result['beta_attempts_count'], 6)
        self.assertIsNone(result['candidate']); self.assertIsNone(result['control'])

    def test_no_epsilon_variance_threshold(self):
        x = np.arange(253)
        market = 1. + np.where(x % 2, 1e-10, 0.)
        beta, variance, why = audit.beta_oracle(market, market)
        self.assertEqual(why, 'available')
        self.assertGreater(variance, 0.); self.assertLess(variance, 1e-16)
        self.assertAlmostEqual(beta, 1., places=14)

    def test_identical_nonzero_market_returns_are_exactly_degenerate(self):
        market = 2. ** np.arange(253)
        beta, variance, why = audit.beta_oracle(market, market)
        self.assertIsNone(beta); self.assertEqual(variance, 0.)
        self.assertEqual(why, 'zero_or_nonfinite_market_variance')

    def test_price_units_leave_beta_labels_and_feature_unchanged(self):
        p, m, c, v, _ = sparse_sources()
        expected = audit.measure_oracle(p, m, c, v)
        actual = audit.measure_oracle(p * 2. ** 600, m * 2. ** -600, c, v)
        for key in ('betas', 'raw_labels', 'market_labels', 'residual_labels'):
            audit.close(actual[key], expected[key], key)
        for key in ('candidate', 'control'):
            self.assertAlmostEqual(actual[key], expected[key], places=14)

    def test_strict_activity_exact_products_and_overflow_units(self):
        p, m, c, v, anchors = sparse_sources()
        eps = 2. ** -52
        c[:] = 1. + eps; v[:] = 1. - eps
        for s in anchors[:3]: c[s + 19], v[s + 19] = 1., 1.
        self.assertTrue(np.all(c * v == 1.))
        expected = audit.measure_oracle(p, m, c, v)
        self.assertEqual(expected['high_anchors'], anchors[:3])
        self.assertEqual(expected['low_anchors'], anchors[3:])
        for power in (-900, 900):
            got = audit.measure_oracle(p, m, c * 2. ** power, v * 2. ** power)
            self.assertEqual(got['classifications'], expected['classifications'])
            self.assertEqual(got['candidate'], expected['candidate'])
        v[:] = c[:] = 1.
        tied = audit.measure_oracle(p, m, c, v)
        self.assertEqual(tied['high_anchors'], [])
        self.assertEqual(tied['low_anchors'], anchors)

    def test_zero_prior_wins_failure_priority_and_all_samples_preserved(self):
        p, m, c, v, anchors = sparse_sources()
        v[:20] = 0.; m[:] = 1.
        result = audit.measure_oracle(p, m, c, v)
        self.assertEqual(result['reason'], 'zero_prior_activity')
        self.assertEqual(result['invalid_activity_anchors'], [1])
        self.assertEqual(result['invalid_beta_anchors'], anchors)
        self.assertEqual(result['n_all'], 6)
        self.assertEqual(len(result['raw_labels']), 6)

    def test_minimum_three_each_no_events(self):
        p, m, c, v, _ = sparse_sources()
        v[85 + 19] = 1.
        r = audit.measure_oracle(p, m, c, v)
        self.assertEqual(r['reason'], 'insufficient_groups')
        self.assertEqual((r['n_high'], r['n_low']), (2, 4))
        r = audit.measure_oracle(np.ones(1513), m, c, v)
        self.assertEqual(r['n_all'], 0); self.assertFalse(r['available'])

    def test_full_four_source_windows_required(self):
        args = sparse_sources()[:4]
        for column in range(4):
            for pos in (0, 300, len(args[column]) - 1):
                changed = [a.copy() for a in args]; changed[column][pos] = np.nan
                self.assertFalse(audit.measure_oracle(*changed)['available'])
            changed = [a.copy() for a in args]; changed[column] = changed[column][:-1]
            with self.assertRaises(ValueError): audit.measure_oracle(*changed)
            if column != 3:
                changed = [a.copy() for a in args]; changed[column][0] = 0.
                self.assertFalse(audit.measure_oracle(*changed)['available'])
        changed = [a.copy().astype(object) for a in args]; changed[1][0] = 'MISSING'
        self.assertEqual(audit.measure_oracle(*changed)['reason'], 'nonnumeric_market_price')


class PanelTests(unittest.TestCase):
    def test_full_window_beta_dates_and_receipts(self):
        panel, events, calendar = fixture()
        values, details = audit.independent_features(panel, events, calendar)
        receipt = audit.verify_features(panel, events, calendar, values, details)
        self.assertTrue(receipt['pre_anchor_beta_checked']); self.assertFalse(receipt['computes_returns'])
        t = calendar.index(events[0][0]); row = details['event_history'][0]
        self.assertEqual((row['adjusted_start'], row['adjusted_end'], row['activity_start'], row['activity_end']),
                         (calendar[t - 1554], calendar[t - 42], calendar[t - 1321], calendar[t - 63]))
        for cell in row['cells']:
            self.assertEqual(cell['selected_anchor_dates'], [calendar[t - 1302 + s] for s in cell['selected_anchors']])
            self.assertEqual(cell['beta_first_dates'], [calendar[t - 1555 + s] for s in cell['selected_anchors']])
            self.assertEqual(cell['beta_last_dates'], [calendar[t - 1303 + s] for s in cell['selected_anchors']])
            self.assertEqual(cell['label_end_dates'], [calendar[t - 1281 + s] for s in cell['selected_anchors']])
            self.assertTrue(all(e < a for e, a in zip(cell['beta_last_dates'], cell['selected_anchor_dates'])))
        json.dumps(details, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_future_poison_and_all_prefixes(self):
        panel, events, calendar = fixture()
        expected, details = audit.independent_features(panel, events, calendar)
        for n in range(3):
            values, notes = audit.independent_features(panel, events, calendar, n)
            for key in expected: np.testing.assert_array_equal(values[key], expected[key][:n])
            self.assertEqual(notes['event_history'], details['event_history'][:n])
        t = calendar.index(events[0][0])
        for key in panel:
            changed = copy.deepcopy(panel); changed[key] = changed[key].astype(object)
            end = t - (42 if key == 'adj' else 63)
            changed[key].iloc[end + 1:] = 'MUST_NOT_CONVERT'
            values, _ = audit.independent_features(changed, events, calendar, 1)
            for key in expected: np.testing.assert_array_equal(values[key], expected[key][:1])

    def test_missing_market_symbol_day_and_market_cell(self):
        panel, events, calendar = fixture()
        no_market = {key: frame.drop(columns='SPY') for key, frame in panel.items()}
        values, details = audit.independent_features(no_market, events[:1], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(details['reason_counts'], {'missing_market_symbol': 2})
        missing = {key: frame.drop(index=calendar[0]) for key, frame in panel.items()}
        values, details = audit.independent_features(missing, events[:1], calendar)
        self.assertFalse(values['available'].any())
        panel['adj'].loc[calendar[0], 'SPY'] = np.nan
        values, details = audit.independent_features(panel, events[:1], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(details['reason_counts'], {'invalid_market_price': 2})

    def test_all49_candidates_ETF_and_one_stock_missing(self):
        panel, events, calendar = fixture(width=49, count=1)
        values, details = audit.independent_features(panel, events, calendar)
        self.assertEqual(values['available'].shape, (1, 49)); self.assertTrue(values['available'].all())
        self.assertEqual(details['event_history'][0]['cells'][-1]['symbol'], 'QQQ')
        panel['adj'].iat[100, 0] = np.nan
        values, _ = audit.independent_features(panel, events, calendar)
        self.assertFalse(values['available'][0, 0]); self.assertTrue(values['available'][0, 1:].all())

    def test_short_history_and_bad_schema(self):
        panel, events, calendar = fixture()
        values, _ = audit.independent_features(panel, [(calendar[1553], events[0][1])], calendar)
        self.assertFalse(values['available'].any())
        for n in (-1, True, 3):
            with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar, n)
        with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar[::-1])
        with self.assertRaises(ValueError): audit.independent_features(dict(panel, volume=panel['volume'].iloc[:-1]), events, calendar)

    def test_tampered_beta_anchor_label_and_raw_rejected(self):
        panel, events, calendar = fixture(count=1)
        values, details = audit.independent_features(panel, events, calendar)
        keys = ('betas', 'residual_labels', 'raw_labels', 'market_labels', 'market_variances',
                'beta_first_indices', 'beta_last_indices', 'beta_first_dates', 'beta_last_dates',
                'selected_anchors', 'high_anchors', 'classifications', 'label_end_dates')
        for key in keys:
            bad = copy.deepcopy(details); bad['event_history'][0]['cells'][0][key] = []
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, values, bad)
        bad = {k: v.copy() for k, v in values.items()}; bad['control'][0, 0] += .01
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, bad, details)

    def test_direct_tanh_and_missing_exactzero(self):
        values = dict(candidate=np.array([[.1, np.nan], [-1., np.nan]]),
                      control=np.array([[-.2, np.nan], [.01, np.nan]]),
                      available=np.array([[True, False], [True, False]]))
        deltas = audit.independent_deltas(values)
        result = audit.verify_deltas(values, deltas)
        self.assertTrue(result['no_cross_sectional_ranking'])
        self.assertEqual(deltas[audit.ARMS[1]][1, 0], .01 * math.tanh(-20.))
        for arm in audit.RAW:
            np.testing.assert_array_equal(deltas[arm][0], [0., 0.])
            self.assertLessEqual(np.max(abs(deltas[arm])), .01)
        bad = copy.deepcopy(deltas); bad[audit.ARMS[1]][0, 0] = 1e-30
        with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        bad = copy.deepcopy(deltas); bad[audit.ARMS[1]][1, 1] = 1e-30
        with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, dose=.02)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, scale=.1)

    def test_no_producer_imports_or_real_entrypoint(self):
        source = Path(audit.__file__).read_text(); tree = ast.parse(source)
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in imports))
        self.assertNotIn('market_absorption174_kernel', source); self.assertNotIn('__main__', source)


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
