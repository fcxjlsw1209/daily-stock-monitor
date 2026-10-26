"""Fabricated event, date, and account checks; no actual source data loaded."""
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

import shock_absorption173_independent as audit


def sparse_sources():
    prices = np.full(1261, 100.)
    close, volume = np.ones(1259), np.ones(1259)
    anchors = [1 + 42 * k for k in range(6)]
    for k, s in enumerate(anchors):
        prices[s:s + 21] = 99.
        if k < 3:
            volume[s + 19] = 2.
    return prices, close, volume, anchors


def fixture(width=2, count=2):
    index = pd.bdate_range('2009-01-01', periods=1400)
    calendar = index.strftime('%Y-%m-%d').tolist()
    symbols = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    panel = {key: pd.DataFrame(1., index=index, columns=symbols) for key in ('adj', 'close', 'volume')}
    p, c, v, _ = sparse_sources()
    for s in symbols:
        panel['adj'].iloc[19:19 + 1261, panel['adj'].columns.get_loc(s)] = p
        panel['close'].iloc[:1259, panel['close'].columns.get_loc(s)] = c
        panel['volume'].iloc[:1259, panel['volume'].columns.get_loc(s)] = v
    events = [(calendar[t], symbols[:]) for t in (1321, 1342)[:count]]
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
    def test_six_events_three_each_closed_form(self):
        p, c, v, anchors = sparse_sources()
        result = audit.measure_oracle(p, c, v)
        self.assertTrue(result['available'])
        self.assertEqual(result['selected_anchors'], anchors)
        self.assertEqual(result['high_anchors'], anchors[:3])
        self.assertEqual(result['low_anchors'], anchors[3:])
        self.assertEqual((result['n_high'], result['n_low']), (3, 3))
        self.assertEqual(result['candidate'], 0.)
        self.assertAlmostEqual(result['control'], 6 * math.log(100 / 99) / 26, places=13)

    def test_common_cooldown_is_selected_before_groups(self):
        p = np.exp(-.001 * np.arange(1261))
        c, v = np.ones(1259), np.ones(1259)
        wanted = list(range(1, 1240, 21))
        v[np.array(wanted[::2]) + 19] = 2.
        result = audit.measure_oracle(p, c, v)
        self.assertEqual(result['selected_anchors'], wanted)
        self.assertEqual(result['high_anchors'], wanted[::2])
        self.assertEqual(result['low_anchors'], wanted[1::2])
        self.assertEqual(result['n_all'], 59)
        self.assertAlmostEqual(result['candidate'], -.021 * (30 / 50 - 29 / 49), places=13)
        self.assertAlmostEqual(result['control'], -.021 * 59 / 79, places=13)

    def test_final_anchor_and_label_have_exact_boundaries(self):
        p, c, v, _ = sparse_sources()
        p[1239:1260] = 90.
        result = audit.measure_oracle(p, c, v)
        self.assertEqual(result['selected_anchors'][-1], 1239)
        self.assertAlmostEqual(result['labels'][-1], math.log(100 / 90), places=13)
        p[1240:1261] = 80.
        changed = audit.measure_oracle(p, c, v)
        self.assertEqual(changed['selected_anchors'], result['selected_anchors'])

    def test_strict_activity_tie_and_single_ulp(self):
        p, c, v, anchors = sparse_sources()
        v[:] = 1.
        tied = audit.measure_oracle(p, c, v)
        self.assertEqual(tied['high_anchors'], [])
        self.assertEqual(tied['low_anchors'], anchors)
        for s in anchors[:3]: v[s + 19] = np.nextafter(1., 2.)
        got = audit.measure_oracle(p, c, v)
        self.assertEqual(got['high_anchors'], anchors[:3])
        self.assertTrue(got['available'])

    def test_exact_product_not_rounded_product(self):
        p, c, v, anchors = sparse_sources()
        eps = 2. ** -52
        c[:] = 1. + eps; v[:] = 1. - eps
        for s in anchors[:3]:
            c[s + 19], v[s + 19] = 1., 1.
        # All products round to one, but exact anchor activity exceeds its
        # baseline in the first three cases. Rounded-product code misses H.
        self.assertTrue(np.all(c * v == 1.))
        result = audit.measure_oracle(p, c, v)
        self.assertEqual(result['high_anchors'], anchors[:3])
        self.assertEqual(result['low_anchors'], anchors[3:])
        self.assertTrue(result['available'])

    def test_scaled_and_reciprocal_activity_units(self):
        p, c, v, _ = sparse_sources()
        expected = audit.measure_oracle(p, c, v)
        for cp, vp in ((c * 2. ** 900, v * 2. ** 900), (c * 2. ** -900, v * 2. ** -900)):
            got = audit.measure_oracle(p * 2. ** 100, cp, vp)
            self.assertEqual(got['classifications'], expected['classifications'])
            audit.close(got['labels'], expected['labels'], 'unit scaling')
        split = np.where(np.arange(1259) < 100, 1., 4.)
        self.assertEqual(audit.measure_oracle(p, c / split, v * split)['classifications'], expected['classifications'])

    def test_zero_prior_invalidates_whole_cell_retains_full_audit(self):
        p, c, v, anchors = sparse_sources()
        v[:20] = 0.
        got = audit.measure_oracle(p, c, v)
        self.assertEqual(got['reason'], 'zero_prior_activity')
        self.assertFalse(got['available']); self.assertIsNone(got['candidate'])
        self.assertEqual(got['selected_anchors'], anchors)
        self.assertEqual(got['invalid_activity_anchors'], [1])
        self.assertIsNone(got['classifications'][0])
        self.assertEqual(len(got['labels']), 6)

    def test_log_ratio_extremes_remain_finite(self):
        p, c, v, anchors = sparse_sources()
        for s in anchors:
            p[s:s + 21] = 1e-300
        p[anchors[0] + 21] = 1e300
        result = audit.measure_oracle(p, c, v)
        self.assertTrue(all(math.isfinite(y) for y in result['labels']))
        self.assertAlmostEqual(result['labels'][0], 600 * math.log(10), places=10)

    def test_minimum_three_each_and_no_events(self):
        p, c, v, anchors = sparse_sources()
        v[anchors[2] + 19] = 1.
        got = audit.measure_oracle(p, c, v)
        self.assertEqual(got['reason'], 'insufficient_groups')
        self.assertEqual((got['n_high'], got['n_low']), (2, 4))
        self.assertFalse(got['available'])
        self.assertEqual(audit.measure_oracle(np.ones(1261), c, v)['n_all'], 0)

    def test_all_source_window_required_not_only_selected_days(self):
        values = sparse_sources()[:3]
        for column in range(3):
            for pos in (0, len(values[column]) - 1, 300):
                for invalid in (np.nan, np.inf, -1.):
                    args = [a.copy() for a in values]; args[column][pos] = invalid
                    self.assertFalse(audit.measure_oracle(*args)['available'])
            args = [a.copy() for a in values]; args[column] = args[column][:-1]
            with self.assertRaises(ValueError): audit.measure_oracle(*args)
        for column in (0, 1):
            args = [a.copy() for a in values]; args[column][300] = 0.
            self.assertFalse(audit.measure_oracle(*args)['available'])
        args = [a.copy().astype(object) for a in values]; args[2][300] = 'missing'
        self.assertEqual(audit.measure_oracle(*args)['reason'], 'nonnumeric_stored_volume')


class PanelTests(unittest.TestCase):
    def test_true_calendar_offsets_and_all_anchor_dates(self):
        panel, events, calendar = fixture()
        values, details = audit.independent_features(panel, events, calendar)
        receipt = audit.verify_features(panel, events, calendar, values, details)
        self.assertTrue(receipt['independent_scalar_rebuild'])
        self.assertFalse(receipt['computes_returns'])
        row = details['event_history'][0]; t = calendar.index(events[0][0])
        self.assertEqual((row['adjusted_start'], row['adjusted_end'], row['activity_start'], row['activity_end']),
                         (calendar[t - 1302], calendar[t - 42], calendar[t - 1321], calendar[t - 63]))
        for cell in row['cells']:
            self.assertEqual(cell['selected_anchor_dates'], [calendar[t - 1302 + s] for s in cell['selected_anchors']])
            self.assertEqual(cell['label_end_dates'], [calendar[t - 1302 + s + 21] for s in cell['selected_anchors']])
        json.dumps(details, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_recent_future_poison_unread_and_prefixes(self):
        panel, events, calendar = fixture()
        expected, details = audit.independent_features(panel, events, calendar)
        for n in range(3):
            part, notes = audit.independent_features(panel, events, calendar, n)
            for key in expected: np.testing.assert_array_equal(part[key], expected[key][:n])
            self.assertEqual(notes['event_history'], details['event_history'][:n])
        first = calendar.index(events[0][0])
        for key in panel:
            changed = copy.deepcopy(panel)
            changed[key] = changed[key].astype(object)
            end = first - (42 if key == 'adj' else 63)
            changed[key].iloc[end + 1:] = 'MUST_NOT_CONVERT'
            got, _ = audit.independent_features(changed, events, calendar, 1)
            for name in expected: np.testing.assert_array_equal(got[name], expected[name][:1])

    def test_missing_calendar_day_name_and_bad_value(self):
        panel, events, calendar = fixture()
        removed = {key: frame.drop(index=calendar[10]) for key, frame in panel.items()}
        values, note = audit.independent_features(removed, events[:1], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(note['event_history'][0]['missing_official_sessions'], [calendar[10]])
        names = [(events[0][0], ['UNKNOWN', 'QQQ'])]
        values, note = audit.independent_features(panel, names, calendar)
        self.assertEqual(note['event_history'][0]['cells'][0]['reason'], 'missing_symbol')
        self.assertTrue(values['available'][0, 1])
        panel['adj'].iat[30, 0] = np.nan
        values, _ = audit.independent_features(panel, events[:1], calendar)
        np.testing.assert_array_equal(values['available'], [[False, True]])

    def test_short_history_and_schema(self):
        panel, events, calendar = fixture()
        values, note = audit.independent_features(panel, [(calendar[1320], events[0][1])], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(note['reason_counts'], {'insufficient_history': 2})
        for wrong in (-1, True, 3):
            with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar, wrong)
        with self.assertRaises(ValueError): audit.independent_features(panel, events, calendar[::-1])
        changed = dict(panel, volume=panel['volume'].iloc[:-1])
        with self.assertRaises(ValueError): audit.independent_features(changed, events, calendar)

    def test_wrong_audit_fields_and_values_rejected(self):
        panel, events, calendar = fixture()
        values, details = audit.independent_features(panel, events, calendar)
        for key in ('selected_anchors', 'high_anchors', 'low_anchors', 'selected_anchor_dates', 'label_end_dates', 'labels', 'classifications'):
            bad = copy.deepcopy(details)
            bad['event_history'][0]['cells'][0][key] = []
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.verify_features(panel, events, calendar, values, bad)
        wrong = {k: v.copy() for k, v in values.items()}; wrong['control'][0, 0] += .1
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, wrong, details)
        wrong = {k: v.copy() for k, v in values.items()}; wrong['available'][0, 0] = False
        with self.assertRaises(ValueError): audit.verify_features(panel, events, calendar, wrong, details)

    def test_all49_candidates_including_etf_no_eligibility_filter(self):
        panel, events, calendar = fixture(width=49, count=1)
        values, details = audit.independent_features(panel, events, calendar)
        self.assertEqual(values['available'].shape, (1, 49))
        self.assertTrue(values['available'].all())
        self.assertEqual(details['event_history'][0]['cells'][-1]['symbol'], 'QQQ')

    def test_direct_tanh_bounded_and_initial_exact_zero(self):
        values = dict(candidate=np.array([[.1, np.nan], [-1., 100.]]),
                      control=np.array([[-.2, np.nan], [.01, -100.]]),
                      available=np.array([[True, False], [True, True]]))
        deltas = audit.independent_deltas(values)
        receipt = audit.verify_deltas(values, deltas)
        self.assertTrue(receipt['no_cross_sectional_ranking'])
        for arm in audit.RAW:
            np.testing.assert_array_equal(deltas[arm][0], [0., 0.])
            self.assertLessEqual(np.max(abs(deltas[arm])), .01)
        self.assertEqual(deltas[audit.ARMS[1]][1, 0], .01 * math.tanh(-20.))
        bad = copy.deepcopy(deltas); bad[audit.ARMS[1]][0, 0] = 1e-30
        with self.assertRaises(ValueError): audit.verify_deltas(values, bad)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, dose=.02)
        with self.assertRaises(ValueError): audit.verify_deltas(values, deltas, scale=.1)

    def test_no_producer_imports_or_real_entrypoint(self):
        source = Path(audit.__file__).read_text()
        tree = ast.parse(source)
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in imports))
        self.assertNotIn('shock_absorption173_kernel', source)
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
