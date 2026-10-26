"""Synthetic only: independent monthly block transitions and fixed accounting."""
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

import monthly_transition175_independent as audit


def price_path(q=None):
    if q is None:
        endpoints = np.where(np.arange(61) % 2, 2., 1.)
    else:
        endpoints = np.exp(np.r_[0., np.cumsum(q)])
    return np.interp(np.arange(1261), np.arange(0, 1261, 21), endpoints)


def fixture(width=2, count=2):
    index = pd.bdate_range('2008-01-01', periods=1400)
    calendar = index.strftime('%Y-%m-%d').tolist()
    symbols = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    frame = pd.DataFrame(1., index=index, columns=symbols)
    for symbol in symbols: frame.iloc[:1261, frame.columns.get_loc(symbol)] = price_path()
    return {'adj': frame}, [(calendar[t], symbols[:]) for t in (1302, 1323)[:count]], calendar


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


class MeasurementTests(unittest.TestCase):
    def test_closed_form_alternating_signs(self):
        result = audit.measure_oracle(price_path())
        logtwo = math.log(2.)
        self.assertTrue(result['available'])
        self.assertEqual(result['endpoint_indices'], list(range(0, 1261, 21)))
        self.assertEqual(result['states'], [1, -1] * 30)
        self.assertEqual(result['current_state'], -1)
        self.assertEqual(result['matching_transition_indices'], list(range(2, 60, 2)))
        self.assertEqual(result['matching_count'], 29)
        self.assertAlmostEqual(result['matching_sum'], 29 * logtwo, places=13)
        self.assertAlmostEqual(result['unconditional_mean'], -logtwo / 59, places=14)
        self.assertAlmostEqual(result['candidate'], (29 * logtwo - 20 * logtwo / 59) / 49, places=14)
        self.assertEqual(result['candidate'], result['shrunken_payoff'])
        self.assertEqual(result['control'], result['unconditional_mean'])
        self.assertEqual(result['training_labels'], result['log_returns'][1:])
        self.assertEqual(len(result['training_labels']), 59)

    def test_positive_trend_all_matches_equal_unconditional(self):
        result = audit.measure_oracle(price_path([.01] * 60))
        self.assertEqual(result['matching_count'], 59)
        self.assertAlmostEqual(result['candidate'], result['control'], places=15)
        self.assertAlmostEqual(result['candidate'], .01, places=14)
        self.assertEqual(result['matching_transition_indices'], list(range(1, 60)))

    def test_flat_is_real_state_not_missing(self):
        r = audit.measure_oracle(np.full(1261, 50.))
        self.assertTrue(r['available']); self.assertEqual(r['current_state'], 0)
        self.assertEqual(r['matching_count'], 59)
        self.assertEqual(r['states'], [0] * 60)
        self.assertEqual(r['candidate'], 0.); self.assertEqual(r['control'], 0.)

    def test_minimum_ten_boundary_and_no_fallback(self):
        for n in (0, 9, 10):
            q = [0.] * n + [.01] * (59 - n) + [0.]
            r = audit.measure_oracle(price_path(q))
            self.assertEqual(r['current_state'], 0)
            self.assertEqual(r['matching_count'], n)
            self.assertEqual(r['available'], n >= 10)
            if n < 10:
                self.assertEqual(r['reason'], 'insufficient_matches')
                self.assertIsNone(r['candidate']); self.assertIsNone(r['control'])
                self.assertIsNone(r['shrunken_payoff'])
                self.assertEqual(len(r['training_labels']), 59)
            else:
                self.assertAlmostEqual(r['candidate'], (r['matching_sum'] + 20 * r['control']) / (n + 20), places=14)

    def test_first_return_is_not_training_label(self):
        p = price_path(); r = audit.measure_oracle(p)
        p[0] = .5
        changed = audit.measure_oracle(p)
        self.assertNotEqual(changed['log_returns'][0], r['log_returns'][0])
        self.assertEqual(changed['training_labels'], r['training_labels'])
        self.assertEqual(changed['candidate'], r['candidate'])
        self.assertEqual(changed['control'], r['control'])

    def test_query_state_not_extra_training_predictor(self):
        r = audit.measure_oracle(price_path())
        self.assertNotIn(60, r['matching_transition_indices'])
        self.assertEqual(r['matching_count'], r['states'][:-1].count(r['current_state']))
        self.assertEqual(r['training_labels'][-1], r['log_returns'][-1])

    def test_tie_and_adjacent_float_have_different_states(self):
        p = np.ones(1261)
        p[21] = np.nextafter(1., 2.)
        r = audit.measure_oracle(p)
        self.assertEqual(r['states'][:3], [1, -1, 0])
        self.assertGreater(r['log_returns'][0], 0.)
        self.assertLess(r['log_returns'][1], 0.)
        self.assertEqual(r['matching_count'], 57)

    def test_nonendpoint_path_not_measure_but_must_be_valid(self):
        p = price_path(); r = audit.measure_oracle(p)
        p[1:21] = 1000.
        self.assertEqual(audit.measure_oracle(p), r)
        p[2] = np.nan
        broken = audit.measure_oracle(p)
        self.assertFalse(broken['available'])
        self.assertEqual(broken['endpoint_indices'], [])

    def test_scale_and_extreme_prices(self):
        p = price_path(); r = audit.measure_oracle(p)
        for multiplier in (2. ** -900, 2. ** 900):
            got = audit.measure_oracle(p * multiplier)
            self.assertEqual(got['states'], r['states'])
            for key in ('candidate', 'control', 'matching_sum'):
                self.assertAlmostEqual(got[key], r[key], places=13)
        p = np.ones(1261)
        p[::21] = np.where(np.arange(61) % 2, 1e300, 1e-300)
        r = audit.measure_oracle(p)
        self.assertTrue(r['available'])
        self.assertTrue(all(math.isfinite(x) for x in r['log_returns']))
        self.assertAlmostEqual(r['log_returns'][0], 600 * math.log(10.), places=10)

    def test_all_source_values_and_shape_required(self):
        for position in (0, 1, 630, 1260):
            for missing in (0., -1., np.nan, np.inf):
                p = price_path(); p[position] = missing
                self.assertFalse(audit.measure_oracle(p)['available'])
        p = price_path().astype(object); p[10] = 'MISSING'
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        for p in (np.ones(1260), np.ones((1, 1261))):
            with self.assertRaises(ValueError): audit.measure_oracle(p)


class PanelTests(unittest.TestCase):
    def test_dates_and_independent_receipt(self):
        panel, events, calendar = fixture()
        values, details = audit.independent_features(panel, events, calendar)
        receipt = audit.verify_features(panel, events, calendar, values, details)
        self.assertTrue(receipt['block_partition_checked'])
        self.assertTrue(receipt['query_and_transition_alignment_checked'])
        self.assertFalse(receipt['computes_returns'])
        row = details['event_history'][0]; t = calendar.index(events[0][0])
        self.assertEqual(row['adjusted_start'], calendar[t - 1302])
        self.assertEqual(row['adjusted_end'], calendar[t - 42])
        for cell in row['cells']:
            self.assertEqual(cell['endpoint_dates'], [calendar[t - 1302 + k] for k in range(0, 1261, 21)])
            self.assertEqual(cell['matching_predictor_end_dates'], [calendar[t - 1302 + k * 21] for k in cell['matching_transition_indices']])
            self.assertEqual(cell['matching_label_end_dates'], [calendar[t - 1302 + (k + 1) * 21] for k in cell['matching_transition_indices']])
            self.assertTrue(all(day <= calendar[t - 42] for day in cell['matching_label_end_dates']))
        json.dumps(details, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_future_poison_and_every_prefix(self):
        panel, events, calendar = fixture()
        full, full_audit = audit.independent_features(panel, events, calendar)
        for n in range(3):
            part, note = audit.independent_features(panel, events, calendar, n)
            for key in full: np.testing.assert_array_equal(part[key], full[key][:n])
            self.assertEqual(note['event_history'], full_audit['event_history'][:n])
        changed = {'adj': panel['adj'].astype(object)}
        changed['adj'].iloc[1261:] = 'FUTURE_POISON'
        got, _ = audit.independent_features(changed, events, calendar, 1)
        for key in full: np.testing.assert_array_equal(got[key], full[key][:1])

    def test_missing_date_and_symbol_and_onebadcell(self):
        panel, events, calendar = fixture()
        absent = {'adj': panel['adj'].drop(index=calendar[630])}
        values, notes = audit.independent_features(absent, events[:1], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(notes['reason_counts'], {'missing_official_session': 2})
        values, notes = audit.independent_features(panel, [(events[0][0], ['UNKNOWN', 'QQQ'])], calendar)
        self.assertFalse(values['available'][0, 0]); self.assertTrue(values['available'][0, 1])
        panel['adj'].iat[1, 0] = np.nan
        values, _ = audit.independent_features(panel, events[:1], calendar)
        np.testing.assert_array_equal(values['available'], [[False, True]])

    def test_full49_and_nofilter(self):
        panel, events, calendar = fixture(width=49, count=1)
        values, notes = audit.independent_features(panel, events, calendar)
        self.assertEqual(values['available'].shape, (1, 49)); self.assertTrue(values['available'].all())
        self.assertEqual(notes['event_history'][0]['cells'][-1]['symbol'], 'QQQ')

    def test_shortcalendar_and_malformed_axes(self):
        panel, events, calendar = fixture()
        values, _ = audit.independent_features(panel, [(calendar[1301], events[0][1])], calendar)
        self.assertFalse(values['available'].any())
        for n in (-1, True, 3):
            with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar, n)
        with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar[::-1])
        with self.assertRaises(ValueError): audit.independent_features(dict(panel, volume=panel['adj']), events, calendar)
        with self.assertRaises(ValueError): audit.independent_features(panel, events[::-1], calendar)

    def test_altered_samples_dates_sums_or_signs_rejected(self):
        panel, events, calendar = fixture(count=1)
        values, notes = audit.independent_features(panel, events, calendar)
        for key in ('endpoint_indices', 'states', 'log_returns', 'training_labels', 'matching_transition_indices',
                    'matching_labels', 'endpoint_dates', 'matching_predictor_end_dates', 'matching_label_end_dates'):
            bad = copy.deepcopy(notes); bad['event_history'][0]['cells'][0][key] = []
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, values, bad)
        for key in ('current_state', 'matching_count', 'matching_sum', 'unconditional_mean', 'shrunken_payoff'):
            bad = copy.deepcopy(notes); bad['event_history'][0]['cells'][0][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, values, bad)
        bad = {k: v.copy() for k, v in values.items()}; bad['candidate'][0, 0] += .1
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, bad, notes)
        bad = {k: v.copy() for k, v in values.items()}; bad['available'][0, 0] = False
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, bad, notes)

    def test_dose_scale_missing_and_first_row(self):
        values = dict(candidate=np.array([[.1, np.nan], [-1., np.nan]]),
                      control=np.array([[-.2, np.nan], [.01, np.nan]]),
                      available=np.array([[True, False], [True, False]]))
        deltas = audit.independent_deltas(values)
        receipt = audit.verify_deltas(values, deltas)
        self.assertEqual(receipt['dose'], .02); self.assertEqual(receipt['scale'], .05)
        self.assertTrue(receipt['no_cross_sectional_ranking'])
        self.assertEqual(deltas[audit.ARMS[1]][1, 0], .02 * math.tanh(-20.))
        for arm in audit.RAW:
            np.testing.assert_array_equal(deltas[arm][0], [0., 0.])
            self.assertLessEqual(np.max(abs(deltas[arm])), .02)
        for i, j in ((0, 0), (1, 1)):
            bad = copy.deepcopy(deltas); bad[audit.ARMS[1]][i, j] = 1e-30
            with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, dose=.01)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, scale=.1)

    def test_no_producer_imports_or_realentrypoint(self):
        source = Path(audit.__file__).read_text(); tree = ast.parse(source)
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in imports))
        self.assertNotIn('monthly_transition175_kernel', source); self.assertNotIn('__main__', source)


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
