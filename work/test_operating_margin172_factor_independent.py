"""Fabricated source and accounting checks; no original financial data loaded."""
import ast
import copy
from datetime import date, timedelta
import importlib.util
import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import operating_margin172_factor_independent as audit


def producer():
    path = Path(__file__).with_name('operating_margin172_kernel.py')
    spec = importlib.util.spec_from_file_location('margin172_synthetic_kernel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(symbol='AAPL', year=2019, known='2020-01-02', rid=None):
    return dict(symbol=symbol, fiscal_year=year, report_id=rid or symbol + str(year),
        known_session=known, release_date=(date.fromisoformat(known) - timedelta(days=1)).isoformat(),
        current=dict(revenue='200', operating_income='30', days=366),
        comparative=dict(revenue='100', operating_income='20', days=365),
        usable=True, hold_reasons=['reported_scope_only'])


class Poison(dict):
    def get(self, key, *args):
        if key in ('current', 'comparative'):
            raise AssertionError('Unavailable financial values were inspected')
        return super().get(key, *args)
    def __getitem__(self, key):
        if key in ('current', 'comparative'):
            raise AssertionError('Unavailable financial values were inspected')
        return super().__getitem__(key)


def source_fixture():
    """35 artificial reports in the registered structural scope, not real data."""
    records, html, apple, reviewed = [], [], [], []
    for symbol, years in [('MSFT', range(2015, 2027)), ('ABBV', range(2013, 2026)), ('AAPL', range(2016, 2026))]:
        for year in years:
            release = f'{year+1}-02-01'
            known = f'{year+1}-02-02'
            r = record(symbol, year, known, f'{symbol}_FY{year}')
            r['release_date'] = release
            r['source_sha256'] = 'b' * 64
            roles = {}
            fields = dict(revenue={}, operating_income={})
            for role, fy in [('current', year), ('comparative', year - 1)]:
                start = None if (symbol, year, role) == ('AAPL', 2016, 'comparative') else f'{fy}-01-01'
                end = f'{fy}-12-31'
                duration = None if start is None else (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
                period = dict(start=start, end=end, days_inclusive=duration)
                roles[role] = period
                r[role]['days'] = duration
                for field in fields:
                    value = r[role][field]
                    fields[field][role] = dict(amount_decimal=value, period=copy.deepcopy(period),
                        period_start=start, period_end=end, known_at_session=known,
                        known_by=known + 'T20:00:00+00:00', release_date=release)
            r['usable'] = r['comparative']['days'] is not None
            src = dict(issuer=symbol, symbol=symbol, fiscal_year=year, report_id=r['report_id'],
                cells=fields, fields=fields, periods=roles, holds=r['hold_reasons'][:],
                publication=dict(release_date=release),
                source={'sha256': 'b'*64} if symbol == 'AAPL' else {'raw': {'sha256': 'b'*64}})
            (apple if symbol == 'AAPL' else html).append(src)
            reviewed.append(dict(issuer=symbol, fiscal_year=year, source_sha256='b'*64,
                passed=True, checked_annual_cells=4, next_session_clock_rederived=True,
                release_date=release, known_session=known, known_by=known + 'T20:00:00+00:00'))
            records.append(r)
    review = dict(passed=True, source_reports=35, checked_annual_cells=140, reports=reviewed)
    return records, html, apple, review


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
    """Flat made-up prices and one explicit EW3 allocation, no strategy engine."""
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
            units = 100000. / (3. * (1. + cost))
            fee, cash = 3. * units * cost, 100000.
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
            vals = [row['equity'] for row in daily]
            slices = {'full122': metric_fixture(days, vals)}
            for key, left, right in [('early53', 0, 52), ('recent35', 53, 87), ('later34', 88, 121), ('first88', 0, 87)]:
                lo, hi = days.index(dates[left]), days.index(ends[right])
                slices[key] = metric_fixture(days[lo:hi + 1], [x / vals[lo] * 100000. for x in vals[lo:hi + 1]])
            annual = []
            for year in sorted({d[:4] for d in days}):
                ids = [i for i, d in enumerate(days) if d.startswith(year)]
                before = vals[ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=vals[ids[-1]] / before - 1., partial=int(year) in (2016, 2026)))
            metrics = dict(slices['full122'], fees_paid=fee, dollar_turnover=3. * units, trade_rows=3, annual_returns=annual)
            exports[arm][path] = dict(DAILY=daily, TRADES=trades, HOLDINGS=holdings, DAILY_POSITIONS=positions)
            results[arm][path] = dict(metrics=metrics, slices=slices,
                accounting=dict(exported_fill_ledger_matches=True, terminal_liquidation=False))
    return px, px.copy(), orders, exports, results, meta


class FactorTests(unittest.TestCase):
    def verify(self, records, events):
        values, details = producer().build_features(records, events)
        receipt = audit.verify_features(records, events, values, details)
        self.assertTrue(receipt['passed'])
        json.dumps(receipt, allow_nan=False)
        return values, details

    def test_exact_candidate_control_diverge(self):
        values, _ = self.verify([record()], [('2020-01-02', ['AAPL', 'QQQ'])])
        np.testing.assert_array_equal(values['candidate'], [[-1., np.nan]])
        np.testing.assert_array_equal(values['control'], [[1., np.nan]])

    def test_exact_near_cancellation_and_negative_values(self):
        r = record()
        r['current'].update(revenue='10000000000000000000000000000000000000000',
                            operating_income='10000000000000000000000000000000000000001', days=365)
        r['comparative'].update(revenue='1', operating_income='1', days=365)
        values, _ = self.verify([r], [('2020-01-02', ['AAPL'])])
        self.assertEqual(values['candidate'][0, 0], 1.)
        r['current'].update(revenue='100', operating_income='-10')
        r['comparative'].update(revenue='100', operating_income='-20')
        self.assertEqual(self.verify([r], [('2020-01-02', ['AAPL'])])[0]['candidate'][0, 0], 1.)

    def test_exact_ties(self):
        r = record(); r['current'] = copy.deepcopy(r['comparative'])
        values, _ = self.verify([r], [('2020-01-02', ['AAPL'])])
        self.assertEqual(values['candidate'][0, 0], 0.)
        self.assertEqual(values['control'][0, 0], 0.)

    def test_common_unit_invariance(self):
        r = record(); scaled = copy.deepcopy(r)
        for role in ('current', 'comparative'):
            for key in ('revenue', 'operating_income'):
                scaled[role][key] = r[role][key] + 'e120'
        events = [('2020-01-02', ['AAPL'])]
        one, _ = self.verify([r], events); two, _ = self.verify([scaled], events)
        for key in one:
            np.testing.assert_array_equal(one[key], two[key])

    def test_known_clock_and_inclusive_age400(self):
        r = record()
        known = date.fromisoformat(r['known_session'])
        events = [((known + timedelta(days=n)).isoformat(), ['AAPL']) for n in (-1, 0, 400, 401)]
        values, _ = self.verify([r], events)
        np.testing.assert_array_equal(values['available'][:, 0], [False, True, True, False])

    def test_future_stale_unusable_poison_never_read(self):
        future = Poison(record(known='2030-01-02'))
        stale = Poison(record(known='2010-01-02'))
        unusable = Poison(record()); unusable['usable'] = False
        for r in (future, stale, unusable):
            self.assertFalse(self.verify([r], [('2020-01-02', ['AAPL'])])[0]['available'][0, 0])

    def test_latest_unusable_blocks_old_valid_and_holds_are_descriptive(self):
        old = record(known='2019-12-02', rid='old')
        latest = record(rid='new'); latest['usable'] = False
        values, details = self.verify([latest, old], [('2020-01-02', ['AAPL'])])
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(details['event_history'][0]['cells'][0]['selected_report_id'], 'new')
        latest['usable'] = True
        self.assertTrue(self.verify([latest, old], [('2020-01-02', ['AAPL'])])[0]['available'][0, 0])

    def test_latest_clock_not_highest_fiscal_year(self):
        old = record(year=2025, known='2019-12-02', rid='future_year_label')
        new = record(year=2018, rid='new_clock')
        _, details = self.verify([old, new], [('2020-01-02', ['AAPL'])])
        self.assertEqual(details['event_history'][0]['cells'][0]['selected_report_id'], 'new_clock')

    def test_invalid_values_and_days_common_mask(self):
        for role in ('current', 'comparative'):
            for field, bads in [('revenue', ['0', '-1', 'NaN', 'Infinity', None, 1.]),
                                ('operating_income', ['NaN', 'Infinity', '1,000', None, 1.]),
                                ('days', [None, True, 329, 401, 365.])]:
                for bad in bads:
                    r = record(); r[role][field] = bad
                    with self.subTest(role=role, field=field, bad=bad):
                        self.assertFalse(self.verify([r], [('2020-01-02', ['AAPL'])])[0]['available'][0, 0])
        for duration in (330, 400):
            r = record(); r['current']['days'] = duration
            self.assertTrue(self.verify([r], [('2020-01-02', ['AAPL'])])[0]['available'][0, 0])

    def test_all_failure_reasons_are_checked(self):
        r = record(); r['current'] = dict(revenue='0', operating_income='NaN', days=True); r['comparative'] = None
        _, details = self.verify([r], [('2020-01-02', ['AAPL'])])
        self.assertEqual(len(details['event_history'][0]['cells'][0]['failure_reasons']), 4)

    def test_schema_duplicates_prefix_and_empty(self):
        events = [('2020-01-02', ['AAPL']), ('2020-02-02', ['AAPL'])]
        for limit in (0, 1, 2):
            a, b = producer().build_features([record()], events, limit)
            expected, details = audit.independent_features([record()], events, limit)
            for key in a:
                np.testing.assert_array_equal(a[key], expected[key])
            audit.record_check(b, details)
        self.assertEqual(audit.independent_features([], [])[0]['candidate'].shape, (0, 0))
        for records in ([record(), record()], [record(), record(rid='other')]):
            with self.assertRaises(ValueError):
                audit.independent_features(records, events)
        for bad in (True, -1, 3):
            with self.assertRaises(ValueError):
                audit.independent_features([record()], events, bad)

    def test_full49_direct_score_and_initial_zero(self):
        names = ['AAPL', 'MSFT', 'ABBV'] + [f'E{i}' for i in range(46)]
        rows = [record(symbol=s) for s in names[:3]]
        values, _ = self.verify(rows, [('2020-01-02', names), ('2020-01-31', names)])
        scores, deltas = audit.independent_rank_deltas(values)
        self.assertTrue(audit.verify_rank_deltas(values, scores, deltas)['no_cross_sectional_ranking'])
        self.assertEqual(scores[audit.ARMS[1]][0, 0], -1.)
        self.assertTrue(np.all(deltas[audit.ARMS[1]][0] == 0.))
        self.assertEqual(deltas[audit.ARMS[1]][1, 0], -.01)
        self.assertTrue(np.all(deltas[audit.ARMS[1]][:, 3:] == 0.))
        deltas[audit.ARMS[1]][1, 0] = -.02
        with self.assertRaises(ValueError):
            audit.verify_rank_deltas(values, scores, deltas)

    def test_random_fabricated_rational_parity(self):
        rng = random.Random(172)
        for _ in range(60):
            r = record()
            for role in ('current', 'comparative'):
                r[role] = dict(revenue=str(rng.randint(1, 100000)),
                    operating_income=str(rng.randint(-50000, 100000)), days=rng.randint(330, 400))
            self.verify([r], [('2020-01-02', ['AAPL'])])

    def test_wrong_sign_audit_and_mask_rejected(self):
        rows, events = [record()], [('2020-01-02', ['AAPL'])]
        values, details = producer().build_features(rows, events)
        wrong = copy.deepcopy(values); wrong['candidate'][0, 0] = 1.
        with self.assertRaises(ValueError): audit.verify_features(rows, events, wrong, details)
        wrong = copy.deepcopy(values); wrong['control'][0, 0] = np.nan
        with self.assertRaises(ValueError): audit.verify_features(rows, events, wrong, details)
        details['event_history'][0]['cells'][0]['age_days'] = 3
        with self.assertRaises(ValueError): audit.verify_features(rows, events, values, details)

    def test_independent_module_does_not_import_producer(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in (n or '') or 'host' in (n or '') for n in names))


class SourceTests(unittest.TestCase):
    def test_exact35_fourcells_and_single_missing_period(self):
        args = source_fixture()
        result = audit.verify_source_records(*args)
        self.assertEqual(result['checked_annual_cells'], 140)
        self.assertEqual(result['unavailable_records'], [['AAPL', 2016]])
        self.assertFalse(result['financial_ratios_computed'])

    def test_source_value_clock_digest_period_and_usable_tampering(self):
        for kind in ('value', 'clock', 'digest', 'days', 'usable', 'hold', 'scope', 'review'):
            args = source_fixture()
            r = args[0][0]
            if kind == 'value': r['current']['operating_income'] = '999'
            elif kind == 'clock': r['known_session'] = '2016-02-03'
            elif kind == 'digest': r['source_sha256'] = 'c'*64
            elif kind == 'days': r['current']['days'] = 360
            elif kind == 'usable': r['usable'] = False
            elif kind == 'hold': r['hold_reasons'] = []
            elif kind == 'scope': args[0].pop()
            else: args[3]['reports'][0]['passed'] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.verify_source_records(*args)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = outcome_fixture()

    def test_original_nine_paths_and45_slices(self):
        result = audit.audit_outcomes(*self.args)
        self.assertTrue(result['passed'])
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(result['outcome_helper_sha256'], audit.OUTCOME_SHA)
        json.dumps(result, allow_nan=False)

    def test_public_accounting_checks_frozen_training_metrics(self):
        helper = audit.accounting()
        metrics = self.args[4][audit.ARMS[0]]['same_close_zero']['slices']['first88']
        helper.check_metrics(metrics, metrics.copy(), 'synthetic frozen training')
        changed = dict(metrics, cagr=metrics['cagr'] + .01)
        with self.assertRaises(ValueError):
            helper.check_metrics(metrics, changed, 'synthetic changed training')
        with patch.object(audit, 'ACCOUNTING_SHA', '0'*64), self.assertRaises(ValueError):
            audit.accounting()

    def test_frozen_dependency_tampering_rejected(self):
        with patch.object(audit, 'OUTCOME_SHA', '0'*64), self.assertRaises(ValueError):
            audit.outcome_helper()
        with patch.object(audit, 'ACCOUNTING_SHA', '0'*64), self.assertRaises(ValueError):
            audit.outcome_helper()

    def test_fake_fill_and_slice_rejected(self):
        args = list(self.args); args[3] = copy.deepcopy(args[3])
        args[3][audit.ARMS[1]]['same_close_zero']['TRADES'][0]['adjusted_units'] += 1.
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)
        args = list(self.args); args[4] = copy.deepcopy(args[4])
        args[4][audit.ARMS[2]]['next_open_cost']['slices']['first88']['cagr'] = .1
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)


if __name__ == '__main__':
    unittest.main()
