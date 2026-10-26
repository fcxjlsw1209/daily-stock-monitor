"""Synthetic-only tests; no real market data or producer implementation imported."""
import ast
from datetime import date
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_breakpoint177_independent as audit


def prices(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def current_prices(scale=.01):
    return prices(scale * np.sin(np.arange(63) * .73))


def fixture(count=2, width=3):
    index = pd.bdate_range('2008-01-01', periods=360)
    names = ['S' + str(i) for i in range(width)]
    frame = pd.DataFrame(index=index)
    for i, name in enumerate(names):
        frame[name] = prices(.0001 * (i + 1) + .006 * np.sin(np.arange(len(index) - 1) / (4. + i)))
    ds = index[[310, 331]][:count].strftime('%Y-%m-%d').tolist()
    meta = dict(dates=ds, symbols=[names[:] for _ in ds], etfs=[names[-1]])
    return dict(adj=frame), meta, index.strftime('%Y-%m-%d').tolist()


def score_fixture():
    baseline = np.array([[4., 2., 2., np.nan], [4., 2., 2., np.nan],
                         [3., 2., 1., np.nan], [3., 2., 1., np.nan]])
    candidate = baseline.copy(); candidate[:, :3] = candidate[:, [2, 1, 0]]
    control = baseline.copy()
    available = np.isfinite(baseline)
    original = dict(core=np.array([.52 * audit.ranks(r, np.isfinite(r)) for r in baseline]),
        core_weight=np.array([.52, .52, .5, .52]), state_calm10=np.array([0., 0., 1., 0.]),
        eligible=np.array([[True, False, True, False]] * 4), forecast_available=np.zeros((4, 4), dtype=bool))
    # Calm uses a different, untouched definition: parity is only required outside calm.
    original['core'][2] = [.1, -.3, .4, -.2]
    return dict(baseline=baseline, candidate=candidate, control=control, available=available), original


class MeasurementTests(unittest.TestCase):
    def test_constant_flat_series_exact_tie_earliest_and_no_break(self):
        r = audit.measure_oracle(np.ones(253), current_prices())
        self.assertTrue(r['available']); self.assertEqual(r['sse0'], 0.)
        self.assertEqual(r['candidate_fit']['k'], 63)
        self.assertFalse(r['candidate_fit']['accepted']); self.assertFalse(r['control_fit']['accepted'])
        self.assertEqual(r['candidate'], r['baseline']); self.assertEqual(r['control'], r['baseline'])
        self.assertEqual([v['k'] for v in r['splits']], list(range(63, 190)))

    def test_clear_known_break_all127_squared_errors(self):
        f = prices(np.r_[np.full(90, -.001), np.full(162, .003)])
        r = audit.measure_oracle(f, current_prices())
        self.assertTrue(r['candidate_fit']['accepted']); self.assertEqual(r['candidate_fit']['k'], 90)
        logs = [audit.scalar_log(a, b) for a, b in zip(f[:-1], f[1:])]
        for s in r['splits']:
            a, b = logs[:s['k']], logs[s['k']:]
            expected = sum((v - sum(a) / len(a)) ** 2 for v in a) + sum((v - sum(b) / len(b)) ** 2 for v in b)
            self.assertAlmostEqual(s['sse'], expected, places=15)
        self.assertAlmostEqual(r['candidate'], math.expm1(252 * .003) / r['recent_volatility'], places=10)

    def test_midpoint_has_own_gate(self):
        f = prices(np.r_[np.full(126, .002), np.full(126, -.003)])
        r = audit.measure_oracle(f, current_prices())
        self.assertEqual(r['candidate_fit']['k'], 126)
        self.assertEqual(r['candidate'], r['control'])
        self.assertTrue(r['candidate_fit']['accepted']); self.assertTrue(r['control_fit']['accepted'])

    def test_no_break_exact_endpoint_fallback(self):
        f = prices(np.array([.001, -.001] * 126))
        r = audit.measure_oracle(f, current_prices())
        self.assertTrue(r['available']); self.assertFalse(r['candidate_fit']['accepted'])
        self.assertEqual(r['candidate'], r['baseline']); self.assertEqual(r['control'], r['baseline'])
        direct = (f[-1] / f[0] - 1.) / r['recent_volatility']
        self.assertEqual(r['baseline'], direct)

    def test_gate_strict_and_perfect_fit_finite_audit(self):
        s = dict(k=63, pre_mean=0., post_mean=.01, sse=1.)
        fit, value = audit.fit_oracle(s, 1., 0., 0., 0., .1)
        self.assertFalse(fit['accepted']); self.assertEqual(fit['statistic'], 0.)
        s['sse'] = 0.
        fit, value = audit.fit_oracle(s, 1., 0., 0., 0., .1)
        self.assertTrue(fit['perfect_fit']); self.assertTrue(fit['accepted']); self.assertIsNone(fit['statistic'])
        json.dumps(fit, allow_nan=False)
        fit, value = audit.fit_oracle(s, 0., 0., 0., 0., .1)
        self.assertFalse(fit['accepted']); self.assertFalse(fit['perfect_fit'])

    def test_numerically_constant_formation_suppresses_pseudo_break(self):
        f = prices(np.full(252, .001))
        r = audit.measure_oracle(f, current_prices())
        self.assertTrue(r['numerically_constant']); self.assertTrue(r['available'])
        self.assertEqual(r['candidate_fit']['k'], 63)
        self.assertEqual(r['candidate_fit']['selection_rule'], 'canonical_inactive_k63')
        self.assertIsNone(r['candidate_fit']['statistic'])
        self.assertFalse(r['candidate_fit']['accepted']); self.assertFalse(r['control_fit']['accepted'])
        self.assertEqual(r['candidate'], r['baseline']); self.assertEqual(r['control'], r['baseline'])
        self.assertEqual(r['split_mean_estimates'], 127)

    def test_penalty_equality_is_rejected(self):
        s = dict(k=63, pre_mean=0., post_mean=.01, sse=1.)
        with patch.object(audit, 'PENALTY', 252. * math.log(2.)):
            fit, _ = audit.fit_oracle(s, 2., 0., .1, 1., .1)
        self.assertFalse(fit['accepted'])

    def test_symmetric_breaks_use_registered_relative_tie_earliest(self):
        f = prices(np.r_[np.full(63, .01), np.full(126, -.01), np.full(63, .01)])
        r = audit.measure_oracle(f, current_prices())
        self.assertFalse(r['numerically_constant'])
        self.assertEqual(r['candidate_fit']['k'], 63)
        self.assertEqual(r['candidate_fit']['selection_rule'], 'minimum_sse_relative_tie_earliest')
        self.assertEqual(r['split_tie_tolerance'], 1e-12 * r['sse0'])

    def test_current_volatility_sample_std(self):
        p = prices(np.array([.01, -.005] * 31 + [.01]))
        r = audit.measure_oracle(prices(np.zeros(252)), p)
        expected = np.std(p[1:] / p[:-1] - 1., ddof=1)
        self.assertAlmostEqual(r['recent_volatility'], expected, places=15)

    def test_baseline_survives_full_window_gap(self):
        f = prices(np.full(252, .001)); f[15] = np.nan
        r = audit.measure_oracle(f, current_prices())
        self.assertTrue(r['baseline_available']); self.assertFalse(r['available'])
        self.assertEqual(r['reason'], 'invalid_formation_price'); self.assertEqual(r['splits'], [])
        f = f.astype(object); f[15] = 'gap'
        self.assertEqual(audit.measure_oracle(f, current_prices())['reason'], 'nonnumeric_formation_price')

    def test_minute_volatility_baseline_valid_but_new_unavailable(self):
        r = audit.measure_oracle(prices(np.full(252, .001)), current_prices(1e-13))
        self.assertTrue(r['baseline_available']); self.assertFalse(r['available'])
        self.assertEqual(r['reason'], 'volatility_floor')
        self.assertEqual(audit.measure_oracle(np.ones(253), np.ones(64))['reason'], 'zero_current_volatility')

    def test_forecast_overflow_blocks_both_keeps_baseline_and_counts(self):
        f = prices(np.r_[np.full(126, -4.), np.full(126, 4.)])
        r = audit.measure_oracle(f, current_prices())
        self.assertEqual(r['reason'], 'nonfinite_forecast'); self.assertTrue(r['baseline_available'])
        self.assertTrue(np.isnan(r['candidate'])); self.assertTrue(np.isnan(r['control']))
        self.assertEqual(r['split_mean_estimates'], 127)

    def test_validation_of_shapes_and_raw_prices(self):
        with self.assertRaises(ValueError): audit.measure_oracle(np.ones(252), current_prices())
        for bad in (0., -1., np.nan, np.inf):
            p = current_prices(); p[12] = bad
            self.assertEqual(audit.measure_oracle(np.ones(253), p)['reason'], 'invalid_current_price')
            f = np.ones(253); f[-1] = bad
            self.assertEqual(audit.measure_oracle(f, current_prices())['reason'], 'invalid_endpoint_price')

    def test_currency_scale_invariance(self):
        f = prices(np.r_[np.full(100, .001), np.full(152, .003)])
        r = audit.measure_oracle(f, current_prices())
        s = audit.measure_oracle(f * 2. ** 800, current_prices() * 2. ** -800)
        audit.record_check(s, r)


class SourceTests(unittest.TestCase):
    def test_all_cells_full_audit_and_counts(self):
        panel, meta, cal = fixture(); values, rec = audit.independent_features(panel, meta, cal)
        receipt = audit.verify_features(panel, meta, cal, values, rec)
        self.assertEqual(receipt['checked_cells'], 6); self.assertEqual(receipt['split_mean_estimates'], 6 * 127)
        self.assertTrue(receipt['breakpoint_search_checked']); self.assertFalse(receipt['computes_returns'])
        e = rec['event_history'][0]
        self.assertEqual(e['formation_start'], cal[310 - 294]); self.assertEqual(e['formation_end'], cal[310 - 42])
        self.assertEqual(e['volatility_start'], cal[310 - 63]); self.assertEqual(e['volatility_end'], cal[310])
        s = e['cells'][0]['splits'][0]
        self.assertEqual(s['break_after_date'], cal[310 - 294 + 63])

    def test_future_poison_cannot_change_previous_features(self):
        panel, meta, cal = fixture(count=1); v, r = audit.independent_features(panel, meta, cal)
        panel['adj'].loc[panel['adj'].index > meta['dates'][0]] = np.nan
        audit.verify_features(panel, meta, cal, v, r)

    def test_current_prices_change_vol_not_formation_search(self):
        panel, meta, cal = fixture(count=1); v, r = audit.independent_features(panel, meta, cal)
        panel['adj'].loc[cal[310 - 20], 'S0'] *= 1.02
        _, s = audit.independent_features(panel, meta, cal)
        self.assertEqual(r['event_history'][0]['cells'][0]['splits'], s['event_history'][0]['cells'][0]['splits'])
        self.assertNotEqual(r['event_history'][0]['cells'][0]['recent_volatility'], s['event_history'][0]['cells'][0]['recent_volatility'])

    def test_missing_session_retains_only_endpoint_baseline(self):
        panel, meta, cal = fixture(count=1); panel['adj'] = panel['adj'].drop(pd.Timestamp(cal[50]))
        v, r = audit.independent_features(panel, meta, cal)
        self.assertTrue(np.isfinite(v['baseline']).all()); self.assertFalse(v['available'].any())
        self.assertEqual(r['event_history'][0]['cells'][0]['reason'], 'missing_formation_session')
        audit.verify_features(panel, meta, cal, v, r)

    def test_audit_tampering_rejected_for_nonselected_split_and_dates(self):
        panel, meta, cal = fixture(count=1); v, r = audit.independent_features(panel, meta, cal)
        for key in ('sse', 'break_after_date'):
            wrong = copy.deepcopy(r)
            wrong['event_history'][0]['cells'][0]['splits'][2][key] = 999. if key == 'sse' else '2000-01-01'
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, v, wrong)

    def test_no_data_and_metadata_validation(self):
        panel, meta, cal = fixture(); meta['dates'] = []; meta['symbols'] = []
        v, r = audit.independent_features(panel, meta, cal)
        self.assertEqual(v['available'].shape, (0, 0)); audit.verify_features(panel, meta, cal, v, r)
        with self.assertRaises(ValueError): audit.independent_features(panel, meta, cal[::-1])


class DeltaTests(unittest.TestCase):
    def test_average_tie_ranks_half_core_and_no_model_filter(self):
        values, original = score_fixture(); d = audit.independent_deltas(values, original)
        r = audit.verify_deltas(values, d, original)
        self.assertEqual(audit.ranks(values['baseline'][1], np.isfinite(values['baseline'][1])).tolist(), [.5, 0., 0., 0.])
        self.assertTrue(r['no_eligibility_or_model_filter']); self.assertTrue(np.any(d['breakpoint_core'][1]))
        self.assertEqual(d['midpoint_core'][1].tolist(), [0.] * 4)
        self.assertTrue(np.all(d['breakpoint_core'][0] == 0)); self.assertTrue(np.all(d['breakpoint_core'][2] == 0))

    def test_one_missing_new_cell_makes_entire_row_neutral(self):
        values, original = score_fixture(); values['available'][3, 1] = False
        values['candidate'][3, 1] = np.nan; values['control'][3, 1] = np.nan
        d = audit.independent_deltas(values, original)
        self.assertTrue(np.all(d['breakpoint_core'][3] == 0)); audit.verify_deltas(values, d, original)

    def test_wrong_original_core_and_wrong_shared_mask_fail(self):
        values, original = score_fixture(); bad = copy.deepcopy(original); bad['core'][1, 0] += .001
        with self.assertRaises(ValueError): audit.independent_deltas(values, bad)
        values['control'][1, 0] = np.nan
        with self.assertRaises(ValueError): audit.independent_deltas(values, original)

    def test_wrong_delta_and_dose_rejected(self):
        values, original = score_fixture(); d = audit.independent_deltas(values, original)
        d['breakpoint_core'][0, 0] = .01
        with self.assertRaises(ValueError): audit.verify_deltas(values, d, original)
        with self.assertRaises(ValueError): audit.independent_deltas(values, original, .25)

    def test_module_has_no_producer_import_or_execution(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [v.name for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names]
        self.assertFalse(any('kernel' in (s or '') or 'host' in (s or '') for s in names))
        self.assertEqual(audit.sha(audit.ACCOUNTING), audit.ACCOUNTING_SHA)
        self.assertEqual(audit.sha(audit.OUTCOME_HELPER), audit.OUTCOME_SHA)
        self.assertTrue(callable(audit.accounting().audit_path))
        with patch.object(audit, 'ACCOUNTING_SHA', 'bad'):
            with self.assertRaises(ValueError): audit.accounting()


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

class OutcomeTests(unittest.TestCase):
    def test_nine_paths_and_fortyfive_slices(self):
        data = account_fixture()
        receipt = audit.audit_outcomes(*data)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['checked_paths'], 9)
        self.assertEqual(receipt['checked_slices'], 45)

if __name__ == '__main__':
    unittest.main()
