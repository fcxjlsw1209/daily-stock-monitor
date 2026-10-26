"""Synthetic-only source, window, delta, and accounting tests for round179."""
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

import monthly_downside_mix179_independent as audit


def price_path(returns):
    return np.r_[1., np.cumprod(1. + np.asarray(returns))]


def example_prices():
    time = np.arange(294)
    return price_path(.0005 + (.003 + .00002 * time) * np.sin(time * .73))


def fixture(width=2, count=3):
    index = pd.bdate_range('2014-01-02', periods=400)
    calendar = index.strftime('%Y-%m-%d').tolist()
    names = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    time = np.arange(len(index) - 1)
    frame = pd.DataFrame({s: price_path(.0001 + (.003 + .00002 * time) * np.sin(time * (.71 + j * .003)))
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
    """Explicit synthetic fixed units and constant quotes; no engine invoked."""
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
    def test_manual_uncentered_simple_square_windows_and_formula(self):
        p = example_prices()
        r = p[1:] / p[:-1] - 1.
        result = audit.measure_oracle(p)
        self.assertTrue(result['available'])
        shares = []
        for name, values in (('old252', r[:252]), ('current63', r[231:])):
            total, down = np.sum(values ** 2), np.sum(values[values < 0.] ** 2)
            self.assertAlmostEqual(result['sum_squares_' + name], total, places=15)
            self.assertAlmostEqual(result['downside_sum_squares_' + name], down, places=15)
            self.assertAlmostEqual(result['share_' + name], down / total, places=15)
            shares.append(down / total)
        self.assertAlmostEqual(result['candidate'], shares[0] - shares[1])
        self.assertAlmostEqual(result['control'], .5 - shares[1])
        centered = r[:252] - r[:252].mean()
        wrong = np.sum(centered[centered < 0.] ** 2) / np.sum(centered ** 2)
        self.assertNotAlmostEqual(result['share_old252'], wrong, places=4)
        self.assertEqual(result['downside_share_estimates'], 2)

    def test_simple_not_log_and_strict_negative_or_zero(self):
        p = price_path(np.tile([.2, -.1, 0.], 98))
        result = audit.measure_oracle(p)
        self.assertAlmostEqual(result['share_old252'], .2)
        self.assertAlmostEqual(result['share_current63'], .2)
        self.assertAlmostEqual(result['candidate'], 0.)
        self.assertAlmostEqual(result['control'], .3)
        log_share = math.log(.9) ** 2 / (math.log(1.2) ** 2 + math.log(.9) ** 2)
        self.assertNotAlmostEqual(result['share_old252'], log_share, places=3)

    def test_latest42_changes_current_but_not_old252(self):
        p = example_prices()
        changed = p.copy()
        changed[253:] *= np.exp(.05 * np.sin(np.arange(42)))
        a, b = audit.measure_oracle(p), audit.measure_oracle(changed)
        self.assertEqual(a['share_old252'], b['share_old252'])
        self.assertNotEqual(a['share_current63'], b['share_current63'])

    def test_all_positive_or_negative_nonconstant_price_are_available(self):
        for value, share in ((.001, 0.), (-.001, 1.)):
            result = audit.measure_oracle(price_path(np.repeat(value, 294)))
            self.assertTrue(result['available'])
            self.assertEqual(result['share_old252'], share)
            self.assertEqual(result['share_current63'], share)
            self.assertEqual(result['candidate'], 0.)
            self.assertEqual(result['control'], .5 - share)

    def test_asymmetric_historical_and_current_downside_move_in_both_directions(self):
        a = audit.measure_oracle(price_path(np.r_[np.repeat(-.01, 231), np.repeat(.01, 63)]))
        b = audit.measure_oracle(price_path(np.r_[np.repeat(.01, 231), np.repeat(-.01, 63)]))
        self.assertGreater(a['candidate'], .9)
        self.assertLess(b['candidate'], -.9)
        self.assertEqual(a['control'], .5); self.assertEqual(b['control'], -.5)

    def test_both_denominators_share_floor_availability(self):
        for p in (np.ones(295), np.r_[np.ones(253), price_path(.01 * np.sin(np.arange(42)))[1:]],
                  np.r_[price_path(.01 * np.sin(np.arange(231))), np.ones(63)]):
            if not np.all(p == 1.):
                p = p.copy()
                if len(p) == 295 and np.all(p[-63:] == 1.):
                    p[-64:] = p[-64]
            out = audit.measure_oracle(p)
            self.assertFalse(out['available'])
            self.assertEqual(out['reason'], 'downside_denominator_floor')
            self.assertEqual(out['downside_share_estimates'], 2)
            self.assertIsNone(out['candidate']); self.assertIsNone(out['share_old252'])
            self.assertIsNone(out['share_current63'])
            self.assertIsNotNone(out['sum_squares_old252'])

    def test_exact_and_near_denominator_floor(self):
        for r in (np.zeros(294), np.tile([1e-14, -1e-14], 147)):
            out = audit.measure_oracle(price_path(r))
            self.assertEqual(out['reason'], 'downside_denominator_floor')
        self.assertTrue(audit.measure_oracle(price_path(np.tile([2e-13, -2e-13], 147)))['available'])
        with patch.object(audit, 'squared_returns', return_value=(1e-24, .5e-24)):
            self.assertFalse(audit.measure_oracle(np.ones(295))['available'])

    def test_every_required_source_position_is_validated(self):
        for position in (0, 1, 41, 42, 230, 231, 251, 252, 253, 293, 294):
            for bad in (0., -1., np.nan, np.inf):
                p = example_prices(); p[position] = bad
                out = audit.measure_oracle(p)
                self.assertEqual(out['reason'], 'invalid_adjusted_price')
                self.assertEqual(out['downside_share_estimates'], 0)
                self.assertTrue(all(out[k] is None for k in ('share_old252', 'candidate', 'control')))

    def test_nonnumeric_shape_and_conversion(self):
        p = example_prices().astype(object); p[17] = 'bad'
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        for p in (np.ones(294), np.ones(296), np.ones((295, 1))):
            with self.assertRaises(ValueError): audit.measure_oracle(p)

    def test_numeric_overflow_and_ratio_underflow_fail_explicitly(self):
        for a, b in ((5e-324, 1e308), (1e308, 5e-324), (1., 1e200)):
            p = np.ones(295); p[20:22] = (a, b)
            with self.assertRaises(FloatingPointError): audit.measure_oracle(p)
        p = np.tile([1e-154, 1.], 148)[:295]
        with self.assertRaises(FloatingPointError): audit.measure_oracle(p)

    def test_scale_invariance_and_no_source_mutation(self):
        p = example_prices(); before = p.copy()
        wanted = audit.measure_oracle(p)
        for scale in (1e-150, 1e150, 100., .01):
            audit.record_check(audit.measure_oracle(p * scale), wanted)
        np.testing.assert_array_equal(p, before)


class FeatureTests(unittest.TestCase):
    def test_dates_counts_receipt_and_etf_not_neutral(self):
        panel, meta, cal = fixture()
        values, record = audit.independent_features(panel, meta, cal)
        result = audit.verify_features(panel, meta, cal, values, record)
        self.assertEqual(record['downside_share_estimates'], 12)
        self.assertEqual(result['checked_cells'], 6)
        self.assertTrue(values['available'].all())
        for t, event in zip((294, 315, 336), record['event_history']):
            self.assertEqual((event['price_start'], event['price_end']), (cal[t - 294], cal[t]))
            for key, start, end, count in (('old252', t - 294, t - 42, 252),
                                            ('current63', t - 63, t, 63)):
                self.assertEqual(event['windows'][key], dict(price_start=cal[start], price_end=cal[end], return_count=count))
                self.assertEqual(event['cells'][0]['windows'], event['windows'])
        for k in ('two_downside_shares_checked', 'strict_negative_simple_returns_checked',
                  'official_session_windows_checked', 'full_295_price_support_checked'):
            self.assertTrue(result[k])
        json.dumps(record, allow_nan=False); json.dumps(result, allow_nan=False)

    def test_prefixes_and_future_poison_not_converted(self):
        panel, meta, cal = fixture()
        full, full_record = audit.independent_features(panel, meta, cal)
        for n in range(4):
            a, record = audit.independent_features(panel, meta, cal, limit=n)
            for key in a: np.testing.assert_array_equal(a[key], full[key][:n])
            self.assertEqual(record['event_history'], full_record['event_history'][:n])
        class Poison:
            def __float__(self): raise AssertionError('Future conversion')
        poisoned = {'adj': panel['adj'].astype(object)}
        poisoned['adj'].loc[poisoned['adj'].index > pd.Timestamp(meta['dates'][0]), :] = Poison()
        actual, rec = audit.independent_features(poisoned, meta, cal, limit=1)
        for key in actual: np.testing.assert_array_equal(actual[key], full[key][:1])
        self.assertEqual(rec['event_history'], full_record['event_history'][:1])

    def test_missing_session_symbol_and_history_precedence(self):
        panel, meta, cal = fixture()
        panel['adj'] = panel['adj'].drop(index=pd.Timestamp(cal[10]), columns=['S0'])
        _, record = audit.independent_features(panel, meta, cal)
        cells = record['event_history'][0]['cells']
        self.assertEqual([c['reason'] for c in cells], ['missing_symbol', 'missing_official_session'])
        meta['dates'][0] = cal[293]
        _, early = audit.independent_features(panel, meta, cal)
        self.assertEqual(early['event_history'][0]['windows'], None)
        self.assertTrue(all(c['reason'] == 'insufficient_history' for c in early['event_history'][0]['cells']))

    def test_official_holiday_calendar_not_calendar_days_or_sparse_rows(self):
        panel, meta, cal = fixture()
        removed = cal[150]
        cal.remove(removed); panel['adj'] = panel['adj'].drop(index=pd.Timestamp(removed))
        meta['dates'][0] = cal[294]
        _, record = audit.independent_features(panel, meta, cal)
        event = record['event_history'][0]
        self.assertEqual(event['windows']['old252']['price_end'], cal[252])
        self.assertEqual(event['missing_official_sessions'], [])
        self.assertTrue(event['cells'][0]['available'])

    def test_49_universe_and_no_data_mutation(self):
        panel, meta, cal = fixture(width=49)
        before, oldmeta = panel['adj'].copy(deep=True), copy.deepcopy(meta)
        values, record = audit.independent_features(panel, meta, cal)
        self.assertEqual(values['available'].shape, (3, 49))
        self.assertEqual(record['downside_share_estimates'], 294)
        pd.testing.assert_frame_equal(panel['adj'], before); self.assertEqual(meta, oldmeta)

    def test_bad_audit_and_numbers_are_rejected(self):
        panel, meta, cal = fixture(); values, record = audit.independent_features(panel, meta, cal)
        for field in ('sum_squares_old252', 'downside_sum_squares_old252', 'share_old252',
                      'sum_squares_current63', 'downside_sum_squares_current63', 'share_current63', 'control'):
            bad = copy.deepcopy(record); bad['event_history'][0]['cells'][0][field] += .01
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(record); bad['event_history'][0]['windows']['old252']['price_end'] = cal[251]
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        for key in ('downside_share_estimates', 'denominator_floor', 'price_count'):
            bad = copy.deepcopy(record); bad[key] += 1
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(record); bad['denominator_floor'] = 0.
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(values); bad['candidate'][1, 0] += .01
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, bad, record)

    def test_schema_and_limit_failures(self):
        panel, meta, cal = fixture()
        for limit in (-1, 4, True, 1.1):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, cal, limit=limit)
        for calendar in (cal[::-1], cal[:2] + cal[1:], ['2014-01-04'] + cal):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, calendar)
        for mutate in ('duplicate', 'width', 'event', 'etfs'):
            m = copy.deepcopy(meta)
            if mutate == 'duplicate': m['symbols'][0][0] = m['symbols'][0][1]
            elif mutate == 'width': m['symbols'][1] = ['QQQ']
            elif mutate == 'event': m['dates'][0] = '2014-01-04'
            else: m['etfs'] = ['QQQ', 'QQQ']
            with self.assertRaises(ValueError): audit.independent_features(panel, m, cal)
        m = dict(meta); del m['etfs']
        with self.assertRaises(ValueError): audit.independent_features(panel, m, cal)
        with self.assertRaises(ValueError): audit.independent_features(dict(panel, close=panel['adj']), meta, cal)

    def test_common_mask_and_raw_range_enforced(self):
        panel, meta, cal = fixture(); values, _ = audit.independent_features(panel, meta, cal)
        for mutate in ('mask', 'positive', 'low', 'infinite', 'shape', 'integer'):
            a = copy.deepcopy(values)
            if mutate == 'mask': a['available'][0, 0] = False
            elif mutate == 'positive': a['candidate'][0, 0] = 1.01
            elif mutate == 'low': a['control'][0, 0] = -.51
            elif mutate == 'infinite': a['candidate'][0, 0] = np.inf
            elif mutate == 'shape': a['control'] = a['control'][:1]
            else: a['available'] = a['available'].astype(int)
            with self.assertRaises(ValueError): audit.arrays(a)

    def test_direct_dose_and_exact_initial_and_missing_zeros(self):
        values = dict(candidate=np.array([[-.5, np.nan], [-.5, 0.]]),
                      control=np.array([[-.2, np.nan], [-.2, 0.]]), available=np.array([[True, False], [True, True]]))
        delta = audit.independent_deltas(values)
        result = audit.verify_deltas(values, delta)
        self.assertEqual(delta[audit.ARMS[1]][1, 0], -.025)
        self.assertTrue(result['direct_linear_mapping']); self.assertTrue(result['no_eligibility_or_model_filter'])
        for i, j in ((0, 0), (0, 1)):
            bad = copy.deepcopy(delta); bad[audit.ARMS[1]][i, j] = -1e-30
            with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        bad = copy.deepcopy(delta); bad[audit.ARMS[2]][1, 0] -= .001
        with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, delta, dose=.02)

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

    def test_hashes_fills_and_slice_errors_fail(self):
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
