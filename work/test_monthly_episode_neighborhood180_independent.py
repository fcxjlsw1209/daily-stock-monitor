"""Synthetic-only independent episode-selector and accounting tests for round180."""
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

import monthly_episode_neighborhood180_independent as audit


def fixture(rows=4, width=6):
    pre = np.tile(np.linspace(.12, .07, width), (rows, 1))
    arrays = dict(baseline=np.zeros((rows, width)), core=np.zeros((rows, width)),
        core_weight=np.full((rows, 1), .52), eligible=np.ones((rows, width), bool),
        effects=np.tile(np.linspace(-.5, .5, width), (rows, 1)),
        state_calm10=np.zeros(rows, bool), state_weak_narrow=np.zeros(rows, bool))
    meta = dict(dates=pd.bdate_range('2020-01-02', periods=rows).strftime('%Y-%m-%d').tolist(),
                symbols=[['S%d' % j for j in range(width)] for _ in range(rows)])
    return pre, arrays, meta


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


class SelectorTests(unittest.TestCase):
    def test_exact_formulas_and_json_receipt(self):
        pre, a, m = fixture(); a['baseline'][:] = .048
        result, record = audit.independent_selection(pre, a, m)
        q = pre[1, 2]
        for j in range(6):
            support = .048 / .48
            d = .05 * (min(.5, max(-.5, support + a['effects'][1, j])) - support)
            w = math.exp(-.5 * ((pre[1, j] - q) / .02) ** 2)
            self.assertAlmostEqual(result['corrections']['control'][1, j], d)
            self.assertAlmostEqual(result['corrections']['candidate'][1, j], d * w)
        rec = audit.verify_selection(pre, a, m, result, record)
        self.assertEqual(rec['episode_applications'], 1)
        self.assertTrue(rec['global_top3_checked'])
        json.dumps(record, allow_nan=False); json.dumps(rec, allow_nan=False)

    def test_budget_cap_calm_weak_and_clipping(self):
        pre, a, m = fixture()
        a['state_calm10'][:] = [False, True, False, True]
        a['state_weak_narrow'][:] = [False, False, True, True]
        a['effects'][:] = 1e20
        _, record = audit.independent_selection(pre, a, m)
        np.testing.assert_allclose([r['budget'] for r in record['event_history']], [.05, .05, .05, .025])
        self.assertAlmostEqual(record['event_history'][3]['cells'][2]['unweighted_correction'], .0125)
        a['effects'][:] *= -1
        result, _ = audit.independent_selection(pre, a, m)
        self.assertAlmostEqual(result['corrections']['control'][3, 2], -.0125)

    def test_initial_neutral_and_cardinality(self):
        pre, a, m = fixture(rows=5)
        a['eligible'][1] = False
        a['eligible'][2] = [True, False, False, False, False, False]
        a['eligible'][3] = [True, False, False, True, False, False]
        result, record = audit.independent_selection(pre, a, m)
        for key in audit.KEYS:
            np.testing.assert_array_equal(result['corrections'][key][:4], np.zeros((4, 6)))
            self.assertEqual(result['choices'][key][:4], [[0, 1, 2], [], [], [0, 3]])
        self.assertEqual(record['initial_rows_suppressed'], 1)
        self.assertEqual(record['low_cardinality_rows'], 3)

    def test_all_eligible_global_selection_no_top_two_lock(self):
        pre, a, m = fixture()
        pre[:] = [.101, .1005, .1, .099, .098, .097]
        a['effects'][:] = [-.5, -.5, 0., .5, .5, .5]
        result, _ = audit.independent_selection(pre, a, m)
        for key in audit.KEYS:
            self.assertNotIn(0, result['choices'][key][1])
            self.assertNotIn(1, result['choices'][key][1])
            expected = sorted(range(6), key=lambda j: (-result['scores'][key][1, j], m['symbols'][1][j]))[:3]
            self.assertEqual(result['choices'][key][1], expected)

    def test_no_hard_rank_five_or_gap_point_zero_two_boundary(self):
        pre, a, m = fixture()
        a['effects'][:] = .2
        result, _ = audit.independent_selection(pre, a, m)
        self.assertGreater(result['corrections']['candidate'][1, 5], 0.)
        for epsilon in (-1e-9, 1e-9):
            altered = pre.copy(); altered[1, 4] = pre[1, 2] - .02 + epsilon
            got, _ = audit.independent_selection(altered, a, m)
            self.assertAlmostEqual(got['corrections']['candidate'][1, 4], .01 * math.exp(-.5), places=8)

    def test_uniform_and_neighborhood_are_distinct(self):
        pre, a, m = fixture(); a['effects'][:] = .3
        result, record = audit.independent_selection(pre, a, m)
        self.assertEqual(result['corrections']['candidate'][1, 2], result['corrections']['control'][1, 2])
        self.assertLess(result['corrections']['candidate'][1, 5], result['corrections']['control'][1, 5])
        self.assertEqual(record['event_history'][1]['third_score'], pre[1, 2])

    def test_nan_effect_is_neutral_without_changing_eligibility(self):
        pre, a, m = fixture(); a['effects'][1, 2] = np.nan
        result, record = audit.independent_selection(pre, a, m)
        for key in audit.KEYS:
            self.assertEqual(result['corrections'][key][1, 2], 0.)
            self.assertEqual(result['scores'][key][1, 2], pre[1, 2])
        self.assertTrue(result['eligible'][1, 2])
        self.assertEqual(record['missing_effect_eligible_cells'], 1)
        self.assertEqual(record['event_history'][1]['cells'][2]['neighborhood_weight'], 1.)

    def test_ineligible_nonfinite_unchanged_but_eligible_invalid_rejected(self):
        pre, a, m = fixture(); a['eligible'][:, 0] = False
        pre[1, 0] = np.inf; a['baseline'][1, 0] = np.nan; a['core'][1, 0] = -np.inf
        result, record = audit.independent_selection(pre, a, m)
        audit.verify_selection(pre, a, m, result, record)
        self.assertEqual(result['scores']['candidate'][1, 0], np.inf)
        self.assertIsNone(record['event_history'][1]['cells'][0]['support'])
        for key in ('baseline', 'core'):
            _, bad, _ = fixture(); bad[key][1, 2] = np.nan
            with self.assertRaises(ValueError): audit.independent_selection(np.nan_to_num(pre), bad, m)
        a['eligible'][1, 0] = True
        with self.assertRaises(ValueError): audit.independent_selection(pre, a, m)

    def test_infinite_effect_anywhere_is_not_missing(self):
        pre, a, m = fixture(); a['eligible'][:, 0] = False
        for bad in (np.inf, -np.inf):
            a['effects'][1, 0] = bad
            with self.assertRaises(ValueError): audit.independent_selection(pre, a, m)

    def test_weight_shapes_and_no_new_regime_inference(self):
        pre, a, m = fixture(); original, _ = audit.independent_selection(pre, a, m)
        for shape in ((4,), (4, 1), (4, 6)):
            a['core_weight'] = np.full(shape, .52)
            got, _ = audit.independent_selection(pre, a, m)
            for key in audit.KEYS: np.testing.assert_array_equal(got['scores'][key], original['scores'][key])
        a['core_weight'][:] = .7
        audit.independent_selection(pre, a, m)  # Source hashes, not this helper, establish actual .50/.52.
        for value in (0., 1., np.nan, np.inf):
            a['core_weight'][:] = value
            with self.assertRaises(ValueError): audit.independent_selection(pre, a, m)

    def test_lexicographic_ties(self):
        pre, a, m = fixture(); pre[:] = .1; a['effects'][:] = 0.
        m['symbols'] = [['Z', 'Y', 'X', 'B', 'A', 'C'] for _ in range(4)]
        result, _ = audit.independent_selection(pre, a, m)
        for key in audit.KEYS: self.assertEqual(result['choices'][key][1], [4, 3, 5])

    def test_prefix_future_poison_and_no_mutation(self):
        pre, a, m = fixture(); full, rec = audit.independent_selection(pre, a, m)
        before = copy.deepcopy((pre, a, m))
        for n in range(5):
            p = pre.astype(object)
            altered = copy.deepcopy(a)
            class Poison:
                def __float__(self): raise AssertionError('future converted')
            p[n:] = Poison()
            for key in ('baseline', 'core', 'effects', 'core_weight'):
                altered[key] = altered[key].astype(object); altered[key][n:] = Poison()
            result, record = audit.independent_selection(p, altered, m, limit=n)
            for key in audit.KEYS:
                np.testing.assert_array_equal(result['scores'][key], full['scores'][key][:n])
                self.assertEqual(result['choices'][key], full['choices'][key][:n])
            self.assertEqual(record['event_history'], rec['event_history'][:n])
        np.testing.assert_array_equal(pre, before[0])
        for key in a: np.testing.assert_array_equal(a[key], before[1][key])
        self.assertEqual(m, before[2])

    def test_extreme_finite_distance_and_strict_overflow_before_clip(self):
        pre, a, m = fixture(); pre[1] = [1e308, 1e307, 0., -1e307, -1e308, -1.5e308]
        a['effects'][1] = 1e308
        result, record = audit.independent_selection(pre, a, m)
        self.assertTrue(np.isfinite(result['scores']['candidate']).all())
        self.assertEqual(record['event_history'][1]['cells'][0]['neighborhood_weight'], 0.)
        a['baseline'][1, 0] = 4.8e307
        with self.assertRaises(FloatingPointError): audit.independent_selection(pre, a, m)
        a['baseline'][1, 0] = 1e308; a['core'][1, 0] = -1e308
        with self.assertRaises(FloatingPointError): audit.independent_selection(pre, a, m)

    def test_bad_result_and_audit_rejected(self):
        pre, a, m = fixture(); result, record = audit.independent_selection(pre, a, m)
        for category in ('scores', 'corrections'):
            altered = copy.deepcopy(result); altered[category]['candidate'][1, 2] += .001
            with self.assertRaises(ValueError): audit.verify_selection(pre, a, m, altered, record)
        altered = copy.deepcopy(result); altered['choices']['candidate'][1].reverse()
        with self.assertRaises(ValueError): audit.verify_selection(pre, a, m, altered, record)
        for name in ('support', 'third_score', 'budget'):
            altered = copy.deepcopy(record)
            row = altered['event_history'][1]
            if name == 'support': row['cells'][0][name] += .01
            else: row[name] += .01
            with self.assertRaises(ValueError): audit.verify_selection(pre, a, m, result, altered)
        altered = copy.deepcopy(result); altered['corrections']['candidate'][0, 0] = 1e-30
        with self.assertRaises(ValueError): audit.verify_selection(pre, a, m, altered, record)

    def test_structure_validation_and_no_production_import(self):
        pre, a, m = fixture()
        for n in (-1, 5, True, 1.5):
            with self.assertRaises(ValueError): audit.independent_selection(pre, a, m, limit=n)
        for change in ('shape', 'state', 'eligibility', 'symbol', 'date'):
            bad, meta = copy.deepcopy(a), copy.deepcopy(m)
            if change == 'shape': bad['effects'] = bad['effects'][:, :-1]
            elif change == 'state': bad['state_calm10'] = bad['state_calm10'].astype(int)
            elif change == 'eligibility': bad['eligible'] = bad['eligible'].astype(int)
            elif change == 'symbol': meta['symbols'][1][0] = meta['symbols'][1][1]
            else: meta['dates'][1] = meta['dates'][0]
            with self.assertRaises(ValueError): audit.independent_selection(pre, bad, meta)
        tree = ast.parse(Path(audit.__file__).read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
        self.assertFalse(any('kernel' in s or 'host' in s for s in imports))
        self.assertNotIn('__main__', Path(audit.__file__).read_text())


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
