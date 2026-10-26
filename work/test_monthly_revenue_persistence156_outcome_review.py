"""Synthetic accounting, source-mask, chronology and tamper tests only."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_revenue_persistence156_outcome_review as m


def order(day, end, symbols):
    return dict(date=str(day.date()), signal_date=str(day.date()), end=str(end.date()),
                symbols=list(symbols), scores=[.2]*len(symbols))


def synthetic_outputs():
    dates = pd.bdate_range('2016-01-04', periods=245); x = np.arange(245)
    close = pd.DataFrame({s: price*np.exp(rate*x+.007*np.sin(x/period)) for s, price, rate, period in
                         [('A', 10, .001, 4), ('B', 20, .0007, 7), ('C', 30, .0013, 6)]}, index=dates)
    opening = close*np.array([.999, 1.001, 1.0002])
    meta = dict(dates=[str(d.date()) for d in dates[:-1:2]], ends=[str(d.date()) for d in dates[2::2]],
                symbols=[['A', 'B', 'C']+[f'S{i}' for i in range(3, 49)] for _ in range(122)], etfs=[])
    orders = {n: [order(d, e, ['A', 'C'] if k and i % (k+2) == 0 else ['A', 'B'])
                  for i, (d, e) in enumerate(zip(dates[:-1:2], dates[2::2]))] for k, n in enumerate(m.NAMES)}
    data, results, navs, opens = {}, {}, {}, {}
    bounds = dict(early2016_2020=(meta['dates'][0], meta['ends'][52]), original2021_2026=(meta['dates'][53], meta['ends'][-1]),
                  training2016_2023=(meta['dates'][0], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    for name in m.NAMES:
        results[name] = {}
        for execution, cost, key in m.PATHS:
            met, nav, trades, holdings = m.replay(close, opening, orders[name], dates, execution, cost)
            prefix = name+'_'+key
            results[name][key] = dict(metrics=met, slices={k: m.h.slice_stat(dates, nav, *span) for k, span in bounds.items()})
            data[prefix+'_DAILY.json'] = [dict(date=str(d.date()), equity=float(v)) for d, v in zip(dates, nav)]
            data[prefix+'_TRADES.json'] = trades; data[prefix+'_HOLDINGS.json'] = holdings
            if key == 'same_close': navs[name] = nav
            if key == 'next_open_cost': opens[name] = nav
    rows = [dict(name=n, **m.h.measures(dates, navs[n], meta)) for n in m.NAMES]
    opened = [dict(name=n, **m.h.measures(dates, opens[n], meta)) for n in m.NAMES]
    choice = m.select(rows, opened)
    data.update({'RESULTS.json': results, 'TRAIN_TRIALS.json': rows, 'TRAIN_OPEN_TRIALS.json': opened,
        'FROZEN.json': dict(selection=choice), 'WALKFORWARD.json': m.walkforward(dates, navs, opens, meta),
        'VALIDATION.json': dict(selection=choice, pristine_oos=False, independent_outcome_review_pending=True,
                                no_observer_writes=True, no_intramonth_trades=True, new_model_fits=0)})
    return close, opening, orders, meta, data


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.original = synthetic_outputs()

    def setUp(self): self.close, self.opening, self.orders, self.meta, self.data = deepcopy(self.original)
    def check(self): return m.check_outcomes(self.close, self.opening, self.orders, self.meta, self.data.__getitem__)

    def test_twelve_paths_and_488_orders(self):
        report = self.check()
        self.assertTrue(report['passed']); self.assertEqual(report['execution_paths_verified'], 12)
        self.assertEqual(report['total_monthly_orders_verified'], 488)
        self.assertEqual(len(report['independent_monthly_comparison']), 122)

    def test_tampered_nav_rejected(self):
        self.data[m.NAMES[3]+'_next_open_cost_DAILY.json'][5]['equity'] += 10
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_trade_cost_rejected(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost'] += 10
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_holdings_rejected(self):
        self.data[m.NAMES[1]+'_same_close_HOLDINGS.json'][0]['symbols'] = []
        with self.assertRaises((AssertionError, ValueError)): self.check()

    def test_wrong_next_open_execution_day_rejected(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date'] = self.meta['dates'][0]
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_frozen_choice_rejected(self):
        self.data['FROZEN.json']['selection']['primary'] = 'forged'
        with self.assertRaises(AssertionError): self.check()

    def test_tampered_training_measure_rejected(self):
        self.data['TRAIN_TRIALS.json'][2]['training']['sharpe'] += .1
        with self.assertRaises(ValueError): self.check()

    def test_tampered_walkforward_rejected(self):
        self.data['WALKFORWARD.json']['folds'][1]['selection']['admitted'] = not self.data['WALKFORWARD.json']['folds'][1]['selection']['admitted']
        with self.assertRaises(ValueError): self.check()

    def test_single_holding_not_original_rule(self):
        self.orders[m.NAMES[0]][0].update(symbols=['A'], scores=[.1])
        with self.assertRaises(ValueError): m.validate_orders(self.close, self.opening, self.orders, self.meta)

    def test_later34_cannot_promote_challenger(self):
        rows = [dict(name=n, training=dict(cagr=.5, sharpe=1., max_drawdown=-.2), early={'cagr': .4}, recent={'cagr': .5}) for n in m.NAMES]
        rows[3]['later34'] = {'cagr': 1000.}
        self.assertFalse(m.select(rows, deepcopy(rows))['admitted'])

    def test_no_new_producer_imports(self):
        self.assertNotIn('monthly_revenue_persistence156_runner', sys.modules)
        self.assertNotIn('after155_revenue_pilot_factor', sys.modules)
        self.assertNotIn('monthly_revenue_persistence156_lifecycle', sys.modules)


class FixedUnitTests(unittest.TestCase):
    def fixture(self):
        days = pd.bdate_range('2016-01-04', periods=4)
        close = pd.DataFrame(dict(A=[10., 20., 30., 40.], B=[20., 20., 20., 20.], C=[10., 10., 10., 10.]), index=days)
        opening = close.copy(); opening.iloc[1] = [10., 10., 10.]
        return days, close, opening

    def test_fixed_units_not_daily_rebalance(self):
        days, close, opening = self.fixture()
        _, nav, _, _ = m.replay(close, opening, [order(days[0], days[-1], ['A', 'B'])], days)
        np.testing.assert_allclose(nav, [100000, 150000, 200000, 250000])

    def test_entry_and_exit_fees(self):
        days, close, opening = self.fixture()
        met, nav, _, _ = m.replay(close, opening, [order(days[0], days[2], ['B', 'C']), order(days[2], days[-1], [])], days, 'close', .0005)
        self.assertAlmostEqual(nav[-1], 100000/1.0005*(1-.0005))
        self.assertAlmostEqual(met['fees_paid'], 100000-nav[-1])

    def test_next_open_lag(self):
        days, close, opening = self.fixture()
        _, nav, trades, _ = m.replay(close, opening, [order(days[0], days[-1], ['A', 'B'])], days, 'open', 0.)
        self.assertEqual(nav[0], 100000); self.assertEqual(nav[1], 200000)
        self.assertEqual(trades[0]['date'], str(days[1].date()))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(dates=['2016-01-04']*122, symbols=[['MSFT']+[f'S{i}' for i in range(48)] for _ in range(122)], etfs=[])
        self.reports = {'MSFT': [dict(report_id=f'r{i}', fiscal_year=2010+i, current=v, scale='1000000', known_ns=1)
                                 for i, v in enumerate(['100', '200', '100', '200'])], 'ABBV': [], 'AAPL': []}
        self.rows, cells = [], []
        self.arrays = {key: np.full((122, 49), np.nan) for key in m.HANDS}
        self.arrays['common_available'] = np.zeros((122, 49), dtype=bool)
        values = m.independent_features(['100', '200', '100', '200'])
        for i in range(122):
            for j in range(49):
                row = dict(signal_index=i, candidate_index=j, decision_date=self.meta['dates'][i], symbol=self.meta['symbols'][i][j],
                           status=m.USABLE if j == 0 else 'held', reasons=[] if j == 0 else ['missing'])
                cell = dict(row, available=j == 0)
                if j == 0:
                    row.update(report_ids=['r0', 'r1', 'r2', 'r3'], edge_ids=['e0', 'e1', 'e2'], latest_report_id='r3', decision_cutoff_ns=2)
                    cell.update({k: row[k] for k in ['report_ids', 'edge_ids', 'latest_report_id', 'decision_cutoff_ns']}, features=values)
                    for key in m.HANDS: self.arrays[key][i, j] = values[key]
                    self.arrays['common_available'][i, j] = True
                self.rows.append(row); cells.append(cell)
        self.feature = dict(cells=cells, new_model_fits=0, source_holds_preserved=True, usable_rows=122, rows=5978,
                            financial_source_scope=['MSFT', 'AAPL', 'ABBV'], **{k: True for k in m.SOURCE_CHECKS})

    def check(self): return m.verify_features(self.arrays, self.feature, self.meta, self.rows, self.reports)

    def test_original_ratio_and_weakest_year(self):
        values = m.independent_features(['100', '200', '100', '200'])
        self.assertEqual(values[m.HANDS[0]], .5)
        self.assertAlmostEqual(values[m.HANDS[2]], -1/3)
        self.assertAlmostEqual(values[m.HANDS[1]], 1-2**(-1/3))
        self.assertEqual(self.check(), 122)

    def test_extreme_finite_ratios_converge(self):
        for last in ('1e300', '1e-300'):
            measured = m.independent_features(['1', '1', '1', last])
            self.assertTrue(all(np.isfinite(v) and abs(v) <= 1 for v in measured.values()))

    def test_wrong_common_mask_rejected(self):
        self.arrays['common_available'][0, 1] = True
        with self.assertRaises(AssertionError): self.check()

    def test_wrong_feature_value_rejected(self):
        self.arrays[m.HANDS[0]][0, 0] += .1
        with self.assertRaises(AssertionError): self.check()

    def test_future_original_rejected(self):
        self.reports['MSFT'][0]['known_ns'] = 3
        with self.assertRaises(ValueError): self.check()

    def test_unavailable_row_cannot_supply_features(self):
        self.feature['cells'][1]['features'] = {key: 0 for key in m.HANDS}
        with self.assertRaises(ValueError): self.check()

    def test_source_status_cannot_be_promoted(self):
        self.feature['cells'][1]['available'] = True
        with self.assertRaises(ValueError): self.check()


class ChronologyTests(unittest.TestCase):
    def test_diagnostic_binds_frozen_and_precedes_audit(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as root:
            root = Path(root); (root/'FROZEN.json').write_text('{}')
            sha = hashlib_sha(root/'FROZEN.json')
            marker = dict(plan_sha256='a'*64, frozen_sha256=sha, started_utc='2020-01-02T00:00:00+00:00')
            (root/'DIAGNOSE_STARTED.json').write_text(json.dumps(marker))
            plan = dict(registered_utc='2020-01-01T00:00:00+00:00')
            m.verify_diagnostic_start(m.Reader(root), root, plan, 'a'*64, datetime(2020, 1, 3, tzinfo=timezone.utc))
            marker['frozen_sha256'] = 'b'*64
            (root/'DIAGNOSE_STARTED.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):
                m.verify_diagnostic_start(m.Reader(root), root, plan, 'a'*64, datetime(2020, 1, 3, tzinfo=timezone.utc))


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    unittest.main()
