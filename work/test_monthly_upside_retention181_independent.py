"""Synthetic-only point-in-time retention, score, and ledger verification."""
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

import monthly_upside_retention181_independent as audit


def price_path(log_returns):
    return np.exp(np.r_[0., np.cumsum(log_returns)])


def example_prices():
    t = np.arange(252)
    return price_path(.0005 + .012 * np.sin(t * .63))


def fixture(width=2, count=3):
    index = pd.bdate_range('2014-01-02', periods=400)
    calendar = index.strftime('%Y-%m-%d').tolist()
    names = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    t = np.arange(len(index) - 1)
    frame = pd.DataFrame({s: price_path(.0002 + .015 * np.sin(t * (.59 + j * .007)))
                          for j, s in enumerate(names)}, index=index)
    dates = [calendar[t] for t in (294, 315, 336)[:count]]
    return {'adj': frame}, dict(dates=dates, symbols=[names[:] for _ in dates], etfs=['QQQ']), calendar


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
    """Construct explicit constant-price units, cash, and all 45 metric slices."""
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
            for _ in range(3):
                cash -= units + units * cost
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


class ScalarTests(unittest.TestCase):
    def test_single_anchor_partial_full_and_zero_retention(self):
        for end, q, c in ((2., .5, 0.), (4., 1., 1.), (.5, 0., 0.)):
            p = np.full(253, end); p[0:2] = (1., 4.)
            result = audit.measure_oracle(p)
            self.assertEqual(result['anchor_indices'], [1])
            self.assertAlmostEqual(result['retention_share'], q)
            self.assertEqual(result['binary_retention_share'], c)
            self.assertEqual(result['candidate'], q - .5)
            self.assertEqual(result['control'], c - .5)
            self.assertEqual(result['retention_share_estimates'], 2)

    def test_weighting_is_by_log_upmove_not_unweighted_win_rate(self):
        p = np.full(253, 4.); p[0] = 1.; p[2:50] = 2.
        result = audit.measure_oracle(p)
        self.assertEqual(result['anchor_indices'], [1, 50])
        self.assertAlmostEqual(result['retention_share'], 2. / 3.)
        self.assertAlmostEqual(result['binary_retention_share'], 1. / 3.)
        self.assertEqual(result['binary_recovered'], [False, True])

    def test_first_last_matured_anchor_and_late_ups_excluded(self):
        p = np.ones(253); p[231] = 4.; p[232:] = 2.
        result = audit.measure_oracle(p)
        self.assertEqual(result['anchor_indices'], [231])
        self.assertEqual(result['retention_share'], .5)
        p = np.ones(253); p[232:] = np.arange(2., 23.)
        result = audit.measure_oracle(p)
        self.assertEqual(result['reason'], 'no_positive_anchors')

    def test_all_up_has_231_overlapping_anchors_no_cooldown(self):
        result = audit.measure_oracle(price_path(np.repeat(.001, 252)))
        self.assertEqual(result['anchor_indices'], list(range(1, 232)))
        self.assertEqual(result['positive_anchor_count'], 231)
        self.assertEqual((result['candidate'], result['control']), (.5, .5))

    def test_constant_and_falling_sources_unavailable_but_two_estimates(self):
        for p in (np.ones(253), price_path(np.repeat(-.001, 252))):
            result = audit.measure_oracle(p)
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'no_positive_anchors')
            self.assertEqual(result['retention_share_estimates'], 2)
            self.assertEqual(result['positive_return_sum'], 0.)
            self.assertIsNone(result['retention_share']); self.assertIsNone(result['candidate'])

    def test_single_tiny_positive_anchor_no_minimum_or_floor(self):
        p = np.ones(253); p[1:] = np.nextafter(1., 2.)
        result = audit.measure_oracle(p)
        self.assertTrue(result['available'])
        self.assertEqual(result['positive_anchor_count'], 1)
        self.assertGreater(result['positive_return_sum'], 0.)

    def test_full_source_validation_even_nonanchor_tail(self):
        for pos in (0, 1, 21, 22, 230, 231, 232, 251, 252):
            for bad in (0., -1., np.nan, np.inf):
                p = example_prices(); p[pos] = bad
                result = audit.measure_oracle(p)
                self.assertEqual(result['reason'], 'invalid_adjusted_price')
                self.assertEqual(result['retention_share_estimates'], 0)
        p = example_prices().astype(object); p[40] = 'bad'
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        for p in (np.ones(252), np.ones(254), np.ones((253, 1))):
            with self.assertRaises(ValueError): audit.measure_oracle(p)

    def test_log_differences_avoid_ratio_overflow_and_arithmetic_failures_raise(self):
        p = np.tile([5e-324, 1e308], 127)[:253]
        result = audit.measure_oracle(p)
        self.assertTrue(result['available'])
        self.assertTrue(math.isfinite(result['positive_return_sum']))
        with patch.object(audit.math, 'log', return_value=math.inf):
            with self.assertRaises(FloatingPointError): audit.measure_oracle(example_prices())
        with self.assertRaises(FloatingPointError): audit.finite_sum([1e308, 1e308])

    def test_scale_invariance_and_no_source_mutation(self):
        p = example_prices(); before = p.copy()
        wanted = audit.measure_oracle(p)
        for scale in (1e-150, 1e150, 100., .01):
            audit.record_check(audit.measure_oracle(p * scale), wanted)
        np.testing.assert_array_equal(p, before)


class FeatureTests(unittest.TestCase):
    def test_official_windows_complete_audit_and_etfs_treated_equally(self):
        panel, meta, cal = fixture()
        values, record = audit.independent_features(panel, meta, cal)
        receipt = audit.verify_features(panel, meta, cal, values, record)
        self.assertEqual(receipt['checked_cells'], 6)
        self.assertEqual(receipt['retention_share_estimates'], 12)
        self.assertTrue(values['available'].all())
        for t, event in zip((294, 315, 336), record['event_history']):
            self.assertEqual((event['price_start'], event['price_end']), (cal[t - 294], cal[t - 42]))
            self.assertEqual(event['windows']['anchors'], dict(first_date=cal[t-293], last_date=cal[t-63], count=231))
            self.assertEqual(event['windows']['maturities'], dict(first_date=cal[t-272], last_date=cal[t-42], horizon_sessions=21))
        for flag in ('full_253_price_support_checked', 'matured_21_session_labels_checked',
                     'positive_log_return_anchors_checked', 'two_retention_shares_checked'):
            self.assertTrue(receipt[flag])
        json.dumps(record, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_every_prefix_and_latest42_future_poison_never_converted(self):
        panel, meta, cal = fixture(); full, record = audit.independent_features(panel, meta, cal)
        for n in range(4):
            a, rec = audit.independent_features(panel, meta, cal, limit=n)
            for key in a: np.testing.assert_array_equal(a[key], full[key][:n])
            self.assertEqual(rec['event_history'], record['event_history'][:n])
        class Poison:
            def __float__(self): raise AssertionError('Post-cutoff source converted')
        poisoned = {'adj': panel['adj'].astype(object)}
        poisoned['adj'].loc[poisoned['adj'].index > pd.Timestamp(cal[252]), :] = Poison()
        actual, rec = audit.independent_features(poisoned, meta, cal, limit=1)
        for key in actual: np.testing.assert_array_equal(actual[key], full[key][:1])
        self.assertEqual(rec['event_history'], record['event_history'][:1])
        truncated = {'adj': panel['adj'].loc[:cal[252]]}
        got, _ = audit.independent_features(truncated, meta, cal, limit=1)
        for key in got: np.testing.assert_array_equal(got[key], full[key][:1])

    def test_missing_session_symbol_history_and_no_positive_count(self):
        panel, meta, cal = fixture()
        panel['adj'] = panel['adj'].drop(index=pd.Timestamp(cal[10]), columns=['S0'])
        _, record = audit.independent_features(panel, meta, cal)
        self.assertEqual([c['reason'] for c in record['event_history'][0]['cells']],
                         ['missing_symbol', 'missing_official_session'])
        meta['dates'][0] = cal[293]
        _, early = audit.independent_features(panel, meta, cal)
        self.assertIsNone(early['event_history'][0]['windows'])
        self.assertTrue(all(c['reason'] == 'insufficient_history' for c in early['event_history'][0]['cells']))
        panel, meta, cal = fixture(); panel['adj'][:] = 1.
        values, record = audit.independent_features(panel, meta, cal)
        self.assertFalse(values['available'].any())
        self.assertEqual(record['retention_share_estimates'], 12)
        self.assertEqual(record['positive_anchor_count'], 0)

    def test_official_holiday_not_calendar_days_or_sparse_rows(self):
        panel, meta, cal = fixture(); removed = cal.pop(150)
        panel['adj'] = panel['adj'].drop(index=pd.Timestamp(removed)); meta['dates'][0] = cal[294]
        _, record = audit.independent_features(panel, meta, cal)
        event = record['event_history'][0]
        self.assertEqual(event['price_end'], cal[252])
        self.assertEqual(event['missing_official_sessions'], [])

    def test_full49_and_input_immutability(self):
        panel, meta, cal = fixture(width=49)
        before, oldmeta = panel['adj'].copy(deep=True), copy.deepcopy(meta)
        values, record = audit.independent_features(panel, meta, cal)
        self.assertEqual(values['available'].shape, (3, 49))
        self.assertEqual(record['retention_share_estimates'], 294)
        pd.testing.assert_frame_equal(panel['adj'], before); self.assertEqual(meta, oldmeta)

    def test_tampered_anchor_values_dates_counts_and_raw_rejected(self):
        panel, meta, cal = fixture(); values, record = audit.independent_features(panel, meta, cal)
        for field in ('positive_return_sum', 'retained_return_sum', 'binary_weighted_sum', 'retention_share', 'control'):
            bad = copy.deepcopy(record); bad['event_history'][0]['cells'][0][field] += .01
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        for field in ('anchor_indices', 'positive_log_returns', 'retained_log_moves', 'capped_retained_log_moves'):
            bad = copy.deepcopy(record); bad['event_history'][0]['cells'][0][field][0] += 1
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(record); bad['event_history'][0]['windows']['maturities']['last_date'] = cal[251]
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        for field in ('retention_share_estimates', 'positive_anchor_count', 'anchor_count', 'horizon_sessions'):
            bad = copy.deepcopy(record); bad[field] += 1
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(values); bad['candidate'][1, 0] += .001
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, bad, record)

    def test_schema_range_and_common_mask(self):
        panel, meta, cal = fixture()
        for limit in (-1, 4, True, 1.1):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, cal, limit=limit)
        for calendar in (cal[::-1], cal[:2] + cal[1:], ['2014-01-04'] + cal):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, calendar)
        m = copy.deepcopy(meta); del m['etfs']
        with self.assertRaises(ValueError): audit.independent_features(panel, m, cal)
        values, _ = audit.independent_features(panel, meta, cal)
        for key in ('candidate', 'control'):
            for value in (-.51, .51, np.inf, np.nan):
                bad = copy.deepcopy(values); bad[key][0, 0] = value
                with self.assertRaises(ValueError): audit.arrays(bad)
        bad = copy.deepcopy(values); bad['available'][0, 0] = False
        with self.assertRaises(ValueError): audit.arrays(bad)

    def test_direct_deltas_bounds_initial_and_missing_exactly_zero(self):
        values = dict(candidate=np.array([[.5, np.nan], [-.5, 0.]]),
                      control=np.array([[.5, np.nan], [.5, 0.]]), available=np.array([[True, False], [True, True]]))
        delta = audit.independent_deltas(values); receipt = audit.verify_deltas(values, delta)
        self.assertEqual(delta[audit.ARMS[1]][1, 0], -.025)
        self.assertEqual(delta[audit.ARMS[2]][1, 0], .025)
        self.assertEqual(receipt['max_absolute_delta'], .025)
        for j in (0, 1):
            bad = copy.deepcopy(delta); bad[audit.ARMS[1]][0, j] = -1e-30
            with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        bad = copy.deepcopy(delta); bad[audit.ARMS[2]][1, 0] = .024
        with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, delta, dose=.10)

    def test_no_production_import_or_execution_entry(self):
        source = Path(audit.__file__).read_text(); tree = ast.parse(source)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in n or 'host' in n for n in imports))
        self.assertNotIn('__main__', source)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.args = account_fixture()

    def test_nine_paths45_slices(self):
        result = audit.audit_outcomes(*self.args)
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(result['outcome_helper_sha256'], audit.OUTCOME_SHA)
        metric = self.args[4][audit.ARMS[0]]['same_close_zero']['slices']['first88']
        audit.accounting().check_metrics(metric, metric.copy(), 'synthetic training')
        json.dumps(result, allow_nan=False)

    def test_hashes_fills_and_metrics_fail(self):
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
