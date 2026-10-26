"""Synthetic-only round171 independent audit tests; no historical inputs read."""
import ast
import copy
from datetime import date
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import price_distribution171_independent as audit


def producer():
    path = Path(__file__).with_name('price_distribution171_kernel.py')
    spec = importlib.util.spec_from_file_location('distribution171_synthetic_producer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(width=3, count=3):
    index = pd.bdate_range('2014-01-01', '2017-12-29')
    calendar = index.strftime('%Y-%m-%d').tolist()
    days = ['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'][:count]
    symbols = ['S%d' % n for n in range(width)]
    rng = np.random.default_rng(171)
    panel = {key: pd.DataFrame(np.exp(rng.normal(0, .2, (len(index), width))) * scale,
                              index=index, columns=symbols)
             for key, scale in (('adj', 100.), ('close', 90.), ('volume', 1e6))}
    return panel, [(day, symbols[:]) for day in days], calendar


def metric_fixture(days, values):
    values = np.array(values)
    returns = values[1:] / values[:-1] - 1.
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    peak = np.maximum.accumulate(np.r_[100000., values])[1:]
    sd = np.std(returns, ddof=1)
    return dict(start=days[0], end=days[-1], years=years, final_equity=float(values[-1]),
        cagr=float((values[-1] / 100000.) ** (1. / years) - 1.),
        sharpe=float(np.mean(returns) / sd * np.sqrt(252.)) if sd > 0 else None,
        max_drawdown=float(np.min(values / peak - 1.)), trading_sessions=len(days),
        total_return=float(values[-1] / 100000. - 1.))


def outcome_fixture():
    """Flat synthetic prices and one explicit three-stock allocation, no engine."""
    idx = pd.bdate_range('2016-08-17', '2026-09-24')
    days = idx.strftime('%Y-%m-%d').tolist()
    dates = ['2016-08-17'] + [days[i] for i in range(len(days) - 1)
                              if days[i][:7] != days[i + 1][:7]]
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
            units = 100000. / (3. * (1. + cost))
            fee = 3. * units * cost
            residual_cash = 100000.
            for _ in range(3):
                residual_cash -= units + units * cost
            nav = residual_cash + math.fsum([units] * 3)
            trades = [dict(date=executions[0], side='buy', symbol=s, adjusted_units=units,
                           adjusted_price=1., dollars=units, cost=units * cost) for s in px.columns]
            holdings = [dict(execution_date=d, signal_date=o['date'], symbols=o['symbols'],
                             scores=o['scores'], fee=fee if i == 0 else 0.)
                        for i, (d, o) in enumerate(zip(executions, orders[arm]))]
            daily, positions = [], []
            for d in days:
                active = d >= executions[0]
                equity = nav if active else 100000.
                daily.append(dict(date=d, equity=equity))
                positions.append(dict(date=d, cash=residual_cash if active else 100000., equity=equity,
                    adjusted_units={s: units for s in px.columns} if active else {},
                    market_values={s: units for s in px.columns} if active else {}))
            vals = [row['equity'] for row in daily]
            slices = {'full122': metric_fixture(days, vals)}
            for key, left, right in (('early53', 0, 52), ('recent35', 53, 87), ('later34', 88, 121), ('first88', 0, 87)):
                lo, hi = days.index(dates[left]), days.index(ends[right])
                slices[key] = metric_fixture(days[lo:hi + 1], [x / vals[lo] * 100000. for x in vals[lo:hi + 1]])
            annual = []
            for year in sorted({d[:4] for d in days}):
                rows = [i for i, day in enumerate(days) if day.startswith(year)]
                before = vals[rows[0] - 1] if rows[0] else 100000.
                annual.append(dict(year=int(year), return_=vals[rows[-1]] / before - 1., partial=int(year) in (2016, 2026)))
            metrics = dict(slices['full122'], fees_paid=fee, dollar_turnover=3. * units, trade_rows=3, annual_returns=annual)
            exports[arm][path] = dict(DAILY=daily, TRADES=trades, HOLDINGS=holdings, DAILY_POSITIONS=positions)
            results[arm][path] = dict(metrics=metrics, slices=slices,
                accounting=dict(exported_fill_ledger_matches=True, terminal_liquidation=False))
    return px, px.copy(), orders, exports, results, meta


class OracleTests(unittest.TestCase):
    def test_closed_form_fraction_and_zero_volume(self):
        prices = np.array([1.] * 105 + [3.] * 105 + [2.])
        close = np.ones(210)
        volume = np.array([1.] * 105 + [3.] * 105)
        candidate, control, reason, count = audit.measure_oracle(prices, close, volume)
        g1, g2 = Fraction(1, 3), Fraction(-1, 5)
        mean = (g1 + g2) / 2
        weighted = (g1 + 3 * g2) / 4
        self.assertEqual((reason, count), ('available', 210))
        self.assertEqual(control, float(mean)); self.assertEqual(candidate, float(weighted - mean))
        volume[:105] = 0.
        got = audit.measure_oracle(prices, close, volume)
        self.assertEqual(got[3], 105); self.assertEqual(got[0], float(g2 - mean))

    def test_exact_counts_and_invalid_sources(self):
        args = [np.ones(211), np.ones(210), np.ones(210)]
        for key in range(3):
            bad = copy.deepcopy(args); bad[key] = bad[key][:-1]
            with self.assertRaisesRegex(ValueError, '211/210/210'):
                audit.measure_oracle(*bad)
            for index in (0, len(args[key]) - 1):
                for value in (np.nan, np.inf, -np.inf, -1.):
                    bad = copy.deepcopy(args); bad[key][index] = value
                    self.assertEqual(audit.measure_oracle(*bad)[:2], (None, None))
        for key in (0, 1):
            bad = copy.deepcopy(args); bad[key][0] = 0.
            self.assertEqual(audit.measure_oracle(*bad)[:2], (None, None))
        self.assertEqual(audit.measure_oracle(args[0], args[1], np.zeros(210))[2], 'zero_total_activity')

    def test_exact_degeneracies(self):
        history = np.linspace(1., 10., 211)
        self.assertEqual(audit.measure_oracle(history, np.ones(210), np.ones(210))[0], 0.)
        self.assertEqual(audit.measure_oracle(np.full(211, 7.), np.arange(1., 211.), np.ones(210))[:2], (0., 0.))
        # Constant historical g with a different reference is still exactly centered.
        a = np.r_[np.ones(210), 2.]
        self.assertEqual(audit.measure_oracle(a, np.arange(1., 211.), np.ones(210))[0], 0.)

    def test_extreme_prices_and_activity_cannot_overflow(self):
        a = np.r_[np.full(105, 1e-300), np.full(105, 1e300), 1.]
        c = np.full(210, 1e300); v = np.r_[np.full(105, 1e300), np.full(105, 1e-300)]
        got = audit.measure_oracle(a, c, v)
        self.assertTrue(all(math.isfinite(x) for x in got[:2])); self.assertAlmostEqual(got[0], 1.)
        actual = producer().measure(a, c, v)
        audit.close([actual['candidate'], actual['control']], got[:2], 'extreme')

    def test_scale_and_reciprocal_split_unit_invariance(self):
        rng = np.random.default_rng(9)
        a, c, v = np.exp(rng.normal(size=211)), np.exp(rng.normal(size=210)), np.exp(rng.normal(size=210))
        base = audit.measure_oracle(a, c, v)
        scaled = audit.measure_oracle(a * 1e50, c * 1e150, v * 1e150)
        audit.close(scaled[:2], base[:2], 'scale')
        split = np.r_[np.ones(105), np.full(105, 4.)]
        audit.close(audit.measure_oracle(a, c / split, v * split)[:2], base[:2], 'reciprocal')

    def test_producer_full_source_audit_and_no_mutation(self):
        panel, events, calendar = fixture(); before = copy.deepcopy(panel)
        values, details = producer().build_features(panel, events, calendar)
        result = audit.verify_features(panel, events, calendar, values, details)
        self.assertTrue(result['passed']); self.assertEqual(result['checked_cells'], 9)
        json.dumps(result, allow_nan=False)
        for key in panel:
            pd.testing.assert_frame_equal(panel[key], before[key])

    def test_exact_offsets_and_recent_future_poison(self):
        panel, events, calendar = fixture()
        values, details = audit.independent_features(panel, events, calendar)
        row = details['event_history'][1]; t = calendar.index(events[1][0])
        self.assertEqual((row['window_start'], row['history_end'], row['reference_date']),
                         (calendar[t - 252], calendar[t - 43], calendar[t - 42]))
        for key in panel:
            changed = copy.deepcopy(panel); end = t - 42 if key == 'adj' else t - 43
            changed[key] = changed[key].astype(object)
            changed[key].loc[changed[key].index > calendar[end]] = 'future-not-a-number'
            got, _ = audit.independent_features(changed, events, calendar, limit=2)
            for name in ('candidate', 'control', 'available'):
                np.testing.assert_array_equal(got[name], values[name][:2])

    def test_prefix_and_future_membership_invariance(self):
        panel, events, calendar = fixture(count=4)
        values, details = audit.independent_features(panel, events, calendar)
        for n in range(5):
            part, notes = audit.independent_features(panel, events, calendar, limit=n)
            for key in values:
                np.testing.assert_array_equal(part[key], values[key][:n])
            self.assertEqual(notes['event_history'], details['event_history'][:n])
        changed = copy.deepcopy(events); changed[-1] = (changed[-1][0], ['X', 'Y', 'Z'])
        partial, _ = audit.independent_features(panel, changed, calendar, limit=3)
        for key in values:
            np.testing.assert_array_equal(partial[key], values[key][:3])

    def test_missing_axis_day_and_symbol_are_explicitly_unavailable(self):
        panel, events, calendar = fixture()
        t = calendar.index(events[1][0]); missing = calendar[t - 100]
        removed = {key: frame.drop(index=missing) for key, frame in panel.items()}
        values, details = producer().build_features(removed, events, calendar)
        audit.verify_features(removed, events, calendar, values, details)
        self.assertFalse(values['available'][1].any())
        self.assertEqual(details['event_history'][1]['missing_official_sessions'], [missing])
        absent = {key: frame.drop(columns='S0') for key, frame in panel.items()}
        values, details = producer().build_features(absent, events, calendar)
        audit.verify_features(absent, events, calendar, values, details)
        self.assertEqual(details['event_history'][1]['reasons'][0], 'missing_symbol')

    def test_bad_cell_does_not_spoil_other_symbols(self):
        panel, events, calendar = fixture(); t = calendar.index(events[1][0])
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].at[calendar[t - 100], 'S0'] = 'bad'
        values, details = producer().build_features(panel, events, calendar)
        audit.verify_features(panel, events, calendar, values, details)
        np.testing.assert_array_equal(values['available'][1], [False, True, True])

    def test_schema_and_calendar_fail_closed(self):
        panel, events, calendar = fixture()
        changed = copy.deepcopy(panel); changed['volume'] = changed['volume'].iloc[:-1]
        with self.assertRaises(ValueError): audit.independent_features(changed, events, calendar)
        for wrong in (calendar[::-1], calendar + [calendar[-1]], ['2016-08-20']):
            with self.assertRaises(ValueError): audit.independent_features(panel, events, wrong)
        with self.assertRaises(ValueError): audit.independent_features(panel, [('2016-08-20', ['S0'])], calendar)
        with self.assertRaises(ValueError): audit.independent_features(panel, [(events[0][0], ['S0', 'S0'])], calendar)
        changed = {key: frame.tz_localize('UTC') for key, frame in panel.items()}
        with self.assertRaises(ValueError): audit.independent_features(changed, events, calendar)

    def test_insufficient_history_and_mask_numeric_audit_corruption(self):
        panel, events, calendar = fixture(); events = [(calendar[20], events[0][1])] + events
        values, details = producer().build_features(panel, events, calendar)
        self.assertFalse(values['available'][0].any())
        audit.verify_features(panel, events, calendar, values, details)
        bad = copy.deepcopy(values); bad['candidate'][1, 1] += .0001
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, bad, details)
        bad = copy.deepcopy(values); bad['available'][1, 1] = False
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, bad, details)
        bad = copy.deepcopy(details); bad['event_history'][1]['reference_date'] = events[1][0]
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, values, bad)

    def test_near_zero_rank_change_is_not_hidden_by_numeric_tolerance(self):
        panel, events, calendar = fixture(width=2, count=1)
        panel['volume'][:] = 1.; panel['close'][:] = 1.
        values, details = audit.independent_features(panel, events, calendar)
        self.assertTrue(np.all(values['candidate'] == 0.))
        values['candidate'][0, 0] = 1e-16
        with self.assertRaisesRegex(ValueError, 'raw rank'):
            audit.verify_features(panel, events, calendar, values)


class RankTests(unittest.TestCase):
    def values(self, row):
        arr = np.asarray([row, row], float)
        return dict(candidate=arr, control=-arr, available=np.isfinite(arr))

    def test_all49_common_pool_average_ties_and_initial_delta_only(self):
        values = self.values(list(range(49)))
        ranks, deltas = audit.independent_rank_deltas(values)
        expected = (np.arange(1, 50) / 49. - .5)
        np.testing.assert_array_equal(ranks[audit.ARMS[1]][0], expected)
        np.testing.assert_array_equal(deltas[audit.ARMS[1]][1], .02 * expected)
        self.assertTrue(np.all(deltas[audit.ARMS[1]][0] == 0.))
        self.assertFalse(np.signbit(deltas[audit.ARMS[1]][0]).any())
        self.assertTrue(audit.verify_rank_deltas(values, ranks, deltas)['passed'])

    def test_ties_missing_singleton_and_no_support(self):
        vals = self.values([1., 1., 2., np.nan]); ranks, deltas = audit.independent_rank_deltas(vals)
        np.testing.assert_array_equal(ranks[audit.ARMS[1]][1], [0., 0., .5, 0.])
        for row in ([np.nan, 1., np.nan], [np.nan, np.nan, np.nan]):
            ranks, deltas = audit.independent_rank_deltas(self.values(row))
            for arm in audit.RAW:
                self.assertTrue(np.all(ranks[arm] == 0.)); self.assertTrue(np.all(deltas[arm] == 0.))

    def test_mask_inf_filter_and_delta_corruption_rejected(self):
        values = self.values([1., 2., 3.]); ranks, deltas = audit.independent_rank_deltas(values)
        bad = copy.deepcopy(deltas); bad[audit.ARMS[1]][1, 0] *= 2
        with self.assertRaises(ValueError): audit.verify_rank_deltas(values, ranks, bad)
        bad = copy.deepcopy(ranks); bad[audit.ARMS[1]][1] = [0., 0., 0.]
        with self.assertRaises(ValueError): audit.verify_rank_deltas(values, bad, deltas)
        values['control'][0, 0] = np.inf
        with self.assertRaises(ValueError): audit.independent_rank_deltas(values)

    def test_oracle_does_not_import_production(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        self.assertFalse(any('kernel' in (n or '') or 'engine' in (n or '') or 'runner' in (n or '') for n in names))


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = outcome_fixture()

    def test_nine_accounts_and45_metric_slices(self):
        result = audit.audit_outcomes(*self.args)
        self.assertTrue(result['passed']); self.assertEqual(result['checked_paths'], 9)
        self.assertEqual(result['checked_slices'], 45)
        json.dumps(result, allow_nan=False)

    def test_price_quantity_fee_and_extra_day_trades_fail(self):
        for kind in ('price', 'units', 'fee', 'date'):
            args = list(self.args); args[3] = copy.deepcopy(args[3])
            row = args[3][audit.ARMS[0]]['same_close_zero']['TRADES'][0]
            if kind == 'price': row['adjusted_price'] = 2.
            elif kind == 'units': row['adjusted_units'] += 1.
            elif kind == 'fee': row['cost'] = 1.
            else: row['date'] = '2016-08-19'
            with self.assertRaises(ValueError): audit.audit_outcomes(*args)

    def test_nav_cash_holdings_statistics_and_arm_omission_fail(self):
        for kind in ('nav', 'cash', 'units', 'stats', 'arm'):
            args = list(self.args); args[3] = copy.deepcopy(args[3]); args[4] = copy.deepcopy(args[4])
            data = args[3][audit.ARMS[0]]['same_close_zero']
            if kind == 'nav': data['DAILY'][3]['equity'] += 1.
            elif kind == 'cash': data['DAILY_POSITIONS'][3]['cash'] += 1.
            elif kind == 'units': data['DAILY_POSITIONS'][3]['adjusted_units']['S0'] += 1.
            elif kind == 'stats': args[4][audit.ARMS[0]]['same_close_zero']['slices']['first88']['cagr'] += .01
            else: args[4].pop(audit.ARMS[-1])
            with self.assertRaises(ValueError): audit.audit_outcomes(*args)


if __name__ == '__main__':
    unittest.main()
