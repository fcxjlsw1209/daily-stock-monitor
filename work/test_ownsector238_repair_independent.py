"""Synthetic-only independent repair checks; no historical prices are loaded."""
import ast
import copy
from datetime import timedelta
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import ownsector238_repair_independent as ind
import ownsector238_source_independent as original


def policy():
    return dict(schema='ownsector238_repair_source_policy_v1',
        missing_mapping_rules=[dict(symbol=s, benchmark=x[0], public_date=x[1],
            valid_from=x[2], valid_until_exclusive=x[3], source_file='synthetic.json',
            source_ids=['synthetic_' + s]) for s, x in ind.RULES.items()],
        communication_benchmark=dict(old='XLC', new='VOX', price_field='adjClose', fill_or_splice=False))


def mapping(d, names, sectors, etfs):
    return {s: original.benchmark_for(pd.Timestamp(d), s, sectors, etfs) for s in names}


def fixture(long=False):
    calendar = pd.bdate_range('2014-01-01', '2026-09-03' if long else '2016-10-04')
    names = ['T', 'BUD', 'GOOGL', 'FUND', 'UNKNOWN']
    dates = ['2016-08-17']
    if long:
        months = pd.period_range('2016-08', '2026-08', freq='M')
        dates += [str(calendar[calendar.to_period('M') == month][-1].date()) for month in months]
    else:
        dates += ['2016-08-31', '2016-09-30']
    adj = pd.DataFrame(100., index=calendar, columns=names + ['SPY', 'XLC', 'XLK', 'XLP'])
    sectors = pd.DataFrame(None, index=pd.to_datetime(dates), columns=names, dtype=object)
    sectors['T'] = 'Communication Services'
    if long:
        sectors.loc['2018-09-01':, 'GOOGL'] = 'Communication Services'
    eligible = np.ones((len(dates), len(names)), dtype=bool)
    eligible[:, -1] = False
    meta = dict(dates=dates, symbols=[names.copy() for _ in dates], etfs=['FUND'])
    report = original.independent_sources({'adj': adj, 'sectors': sectors}, meta,
        [str(d.date()) for d in calendar], eligible, mapping)
    vox = pd.DataFrame({'VOX': np.full(len(calendar), 75.)}, index=calendar)
    return report, adj, vox, policy()


def physical(data, n):
    old, adj, vox, p = data
    short = copy.deepcopy(old)
    short['events'] = short['events'][:n]
    short['records'] = [r for r in short['records'] if r['query_index'] < n]
    short['gaps'] = [g for g in short['gaps'] if g['query_index'] < n]
    if n:
        cutoff = short['events'][-1]['date']
        adj, vox = adj.loc[:cutoff].copy(), vox.loc[:cutoff].copy()
    else:
        adj, vox = adj.iloc[:0].copy(), vox.iloc[:0].copy()
    return short, adj, vox, copy.deepcopy(p)


def run(data, limit=None):
    return ind.repair_sources(*data, limit=limit)


def raw_rows():
    return [dict(date='2015-01-02T00:00:00.000Z', adjClose=71.25),
            dict(date='2015-01-05T00:00:00Z', adjClose=72.)]


def expected_vox():
    return pd.DataFrame({'VOX': [71.25, 72.]}, index=pd.to_datetime(['2015-01-02', '2015-01-05']))


class RepairIndependentTests(unittest.TestCase):
    def test_explicit_mapping_expected_counts_and_scope(self):
        data = fixture(); report = run(data)
        self.assertTrue(report['admitted'])
        self.assertEqual(report['repair_counts'], dict(unchanged=2,
            full_period_communication_proxy=2, dated_business_proxy=4, unresolved=0))
        self.assertEqual([r['benchmark'] for r in report['records'][:4]], ['VOX', 'XLP', 'XLK', 'SPY'])
        self.assertEqual(report['counts']['required_cells'], 8)
        self.assertEqual(report['counts']['valid_endpoints'], 36)
        receipt = ind.check_repaired(*data, report)
        self.assertTrue(all(receipt[k] is True for k in ind.FLAGS))
        self.assertEqual((receipt['cell_count'], receipt['endpoint_count']), (8, 36))
        self.assertFalse(receipt['official_clock_reconstructed_from_calendar'])
        self.assertFalse(receipt['producer_modules_imported'])
        for key in ('momentum_computed', 'eligibility_changed', 'rankings_computed', 'orders_computed', 'returns_computed'):
            self.assertIs(report[key], False)

    def test_all_six_publication_boundaries_and_google_expiry(self):
        p = policy()
        for symbol, (benchmark, public, start, expiry) in ind.RULES.items():
            row = dict(symbol=symbol, date=public, is_etf=False, benchmark=None)
            self.assertEqual(ind.resolve(row, p), (None, 'unresolved', None))
            row['date'] = start
            self.assertEqual(ind.resolve(row, p), (benchmark, 'dated_business_proxy', ['synthetic_' + symbol]))
            if expiry:
                row['date'] = str(ind.day(expiry) - timedelta(days=1))
                self.assertEqual(ind.resolve(row, p)[0], benchmark)
                row['date'] = expiry
                self.assertEqual(ind.resolve(row, p), (None, 'unresolved', None))

    def test_etf_and_existing_mapping_precedence(self):
        p = policy()
        for symbol in ['GOOGL', 'CHL', 'XLC', 'VOX']:
            row = dict(symbol=symbol, date='2020-05-29', is_etf=True, benchmark='SPY')
            self.assertEqual(ind.resolve(row, p), ('SPY', 'unchanged', None))
            row['benchmark'] = 'XLC'
            with self.assertRaises(ValueError): ind.resolve(row, p)
        for old in ['XLK', 'XLE', 'XLP']:
            row = dict(symbol='BUD', date='2020-05-29', is_etf=False, benchmark=old)
            self.assertEqual(ind.resolve(row, p), (old, 'unchanged', None))

    def test_xlc_full_period_replacement_not_missing_value_fallback(self):
        data = fixture()
        # The original XLC is valid throughout, but it must still be replaced.
        self.assertTrue(all(r['admitted'] for r in data[0]['records'] if r['symbol'] == 'T'))
        report = run(data)
        self.assertTrue(all(r['benchmark'] == 'VOX' for r in report['records'] if r['symbol'] == 'T'))
        data[2]['VOX'] = np.nan
        report = run(data)
        self.assertEqual(report['counts']['gap_cells'], 2)
        self.assertTrue(all(r['benchmark'] == 'VOX' for r in report['records'] if r['symbol'] == 'T'))

    def test_unresolved_never_falls_back_to_spy(self):
        p = policy(); row = dict(symbol='UNKNOWN', date='2026-01-30', is_etf=False, benchmark=None)
        self.assertEqual(ind.resolve(row, p), (None, 'unresolved', None))
        p['missing_mapping_rules'][0]['symbol'] = 'UNKNOWN'
        with self.assertRaises(ValueError): ind.resolve(row, p)

    def test_policy_publication_and_rule_tampering(self):
        for change in ('public', 'start', 'end', 'duplicate', 'price_field', 'fill'):
            p = policy()
            if change == 'public': p['missing_mapping_rules'][0]['public_date'] = '2023-03-09'
            elif change == 'start': p['missing_mapping_rules'][0]['valid_from'] = '2023-03-10'
            elif change == 'end': p['missing_mapping_rules'][3]['valid_until_exclusive'] = None
            elif change == 'duplicate': p['missing_mapping_rules'][0] = p['missing_mapping_rules'][1]
            elif change == 'price_field': p['communication_benchmark']['price_field'] = 'close'
            else: p['communication_benchmark']['fill_or_splice'] = True
            with self.subTest(change=change), self.assertRaises(ValueError): ind.policy_rules(p)

    def test_missing_and_invalid_vox_are_distinct(self):
        for value, status, reason in [(None, 'missing', 'missing_value'),
                (np.nan, 'missing', 'missing_value'), (pd.NA, 'missing', 'missing_value'),
                (True, 'invalid', 'invalid_boolean'), ('75', 'invalid', 'invalid_nonreal'),
                (math.inf, 'invalid', 'invalid_nonfinite'), (0., 'invalid', 'invalid_nonpositive')]:
            data = fixture(); data[2]['VOX'] = pd.Series([value] * len(data[2]), index=data[2].index, dtype=object)
            point = run(data)['records'][0]['endpoints'][2]
            self.assertEqual((point['status'], point['reason']), (status, reason))

    def test_missing_row_not_filled_or_spliced_from_adj(self):
        old, adj, vox, p = fixture()
        missing = old['events'][1]['source_dates'][0]
        vox = vox.drop(pd.Timestamp(missing))
        adj['VOX'] = 99.
        report = run((old, adj, vox, p))
        self.assertEqual(report['records'][0]['endpoints'][2]['reason'], 'missing_row')
        self.assertFalse(report['admitted'])

    def test_market_gap_collected_even_without_eligible(self):
        old, adj, vox, p = fixture()
        for e in old['events'][1:]:
            e.update(original_eligible_indices=[], required_indices=[], required_count=0,
                     admitted_count=0, gap_count=0)
        old['records'] = []
        report = run((old, adj.drop(columns='SPY'), vox, p))
        self.assertEqual(report['counts']['required_cells'], 0)
        self.assertEqual(report['counts']['market_gap_events'], 2)
        self.assertFalse(report['admitted'])

    def test_partial_and_empty_prefix_no_numeric_reads(self):
        data = fixture()
        data[1].iloc[:, :] = np.nan; data[2].iloc[:, :] = np.nan
        for n in (0, 1):
            result = run(data, n)
            self.assertTrue(result['admitted'])
            self.assertEqual(result['counts']['valid_endpoints'], 0)
            self.assertEqual(result['counts']['required_cells'], 0)

    def test_sealed_clock_and_identity_tamper_rejected(self):
        for change in ('positions', 'endpoint_date', 'endpoint_symbol', 'required', 'query', 'active'):
            data = fixture(); old = data[0]
            if change == 'positions': old['events'][1]['source_positions'][0] += 1
            elif change == 'endpoint_date': old['records'][0]['endpoints'][0]['date'] = '2015-01-02'
            elif change == 'endpoint_symbol': old['records'][0]['endpoints'][0]['symbol'] = 'OTHER'
            elif change == 'required': old['events'][1]['required_indices'] = [0, 1]
            elif change == 'query': old['records'][0]['date'] = '2016-07-29'
            else: old['events'][1]['active'] = False
            with self.subTest(change=change), self.assertRaises(ValueError): run(data)

    def test_exact_report_tampering_detected(self):
        data = fixture(); expected = run(data)
        for change in ('count', 'repair', 'status', 'reason', 'type', 'extra'):
            changed = copy.deepcopy(expected)
            if change == 'count': changed['counts']['valid_endpoints'] += 1
            elif change == 'repair': changed['records'][0]['mapping_repair'] = 'unchanged'
            elif change == 'status': changed['admitted'] = False
            elif change == 'reason': changed['records'][0]['gap_reasons'] = ['made_up']
            elif change == 'type': changed['counts']['required_cells'] = float(changed['counts']['required_cells'])
            else: changed['extra'] = True
            with self.subTest(change=change), self.assertRaises(ValueError): ind.check_repaired(*data, changed)

    def test_no_input_mutation_or_output_aliasing(self):
        data = fixture(); before = copy.deepcopy(data); out = run(data)
        self.assertEqual(data[0], before[0]); self.assertEqual(data[3], before[3])
        pd.testing.assert_frame_equal(data[1], before[1]); pd.testing.assert_frame_equal(data[2], before[2])
        out['records'][0]['endpoints'][0]['reason'] = 'changed'
        out['events'][1]['required_indices'].clear()
        self.assertEqual(data[0], before[0])

    def test_future_numeric_poison_and_all_small_prefixes(self):
        data = fixture(); full = run(data)
        for n in range(4):
            short = run(data, n)
            self.assertEqual(short, run(physical(data, n)))
            self.assertEqual(short['events'], full['events'][:n])
        poisoned = list(copy.deepcopy(data))
        last = data[0]['events'][2]['source_dates'][1]
        poisoned[2] = poisoned[2].astype(object)
        poisoned[2].loc[last:, 'VOX'] = 'future poison'
        self.assertEqual(run(poisoned, 2), run(data, 2))
        self.assertFalse(run(poisoned)['admitted'])

    def test_122_events_required_physical_prefixes(self):
        data = fixture(long=True)
        self.assertEqual(len(data[0]['events']), 122)
        full = run(data)
        for n in (0, 1, 53, 88, 122):
            logical = run(data, n); physical_result = run(physical(data, n))
            self.assertEqual(logical, physical_result)
            self.assertEqual(logical['events'], full['events'][:n])
            self.assertTrue(ind.check_repaired(*physical(data, n), physical_result)['passed'])

    def test_unique_bilateral_fixtures_full_audit_all_prefixes(self):
        import ownsector238_repair_sources as producer
        from test_ownsector238_repair_sources import fixture as producer_fixture
        for data in (fixture(), producer_fixture()):
            for n in range(len(data[0]['events']) + 1):
                expected = run(data, n)
                actual = producer.repair_sources(*data, limit=n)
                self.assertEqual(expected, actual)
                self.assertTrue(ind.check_repaired(*data, actual, limit=n)['passed'])
                physical_data = physical(data, n)
                self.assertEqual(expected, producer.repair_sources(*physical_data))
        rows = raw_rows(); calendar = ['2015-01-02', '2015-01-05']
        self.assertTrue(ind.check_vox(rows, calendar, producer.parse_vox(rows, calendar))['passed'])

    def test_checker_has_no_new_producer_imports(self):
        tree = ast.parse(Path(ind.__file__).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn('ownsector238_repair_sources', names)
        self.assertNotIn('ownsector238_source_audit', names)
        self.assertIn(Path(original.__file__).resolve(), ind.dependency_files())


class VoxIndependentTests(unittest.TestCase):
    def test_valid_exact_raw_frame_contract(self):
        result = ind.check_vox(raw_rows(), ['2015-01-02', '2015-01-05'], expected_vox())
        self.assertTrue(result['passed']); self.assertEqual(result['row_count'], 2)
        self.assertFalse(result['missing_rows_filled']); self.assertFalse(result['returns_computed'])

    def test_raw_date_rejections(self):
        stamps = ['2014-12-31T00:00:00Z', '2026-09-25T00:00:00Z',
            '2015-01-03T00:00:00Z', '2015-01-02', '2015-01-02T01:00:00Z',
            '2015-01-02T00:00:00-05:00', 'garbage']
        for stamp in stamps:
            rows = raw_rows(); rows[0]['date'] = stamp
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                ind.check_vox(rows, ['2015-01-02', '2015-01-05'], expected_vox())

    def test_raw_value_rejections(self):
        for value in [None, True, np.bool_(True), '71.25', complex(71.25, 0),
                      0., -1., float('inf'), float('nan'), 10 ** 10000]:
            rows = raw_rows(); rows[0]['adjClose'] = value
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                ind.check_vox(rows, ['2015-01-02', '2015-01-05'], expected_vox())

    def test_empty_duplicate_reordered_rows_rejected(self):
        for rows in [[], raw_rows()[::-1], [raw_rows()[0], raw_rows()[0]]]:
            with self.assertRaises(ValueError):
                ind.check_vox(rows, ['2015-01-02', '2015-01-05'], expected_vox())

    def test_actual_session_calendar_and_bounds(self):
        for calendar in [[], ['2015-01-02'], ['2015-01-05', '2015-01-02'],
                         ['2015-01-02', '2015-01-02'], ['2015-1-2', '2015-01-05'],
                         ['2015-01-02', '2015-01-03', '2015-01-05']]:
            with self.assertRaises(ValueError): ind.check_vox(raw_rows(), calendar, expected_vox())

    def test_imported_dtype_values_dates_identity_exact(self):
        for kind in ('value', 'dtype', 'date', 'column', 'extra'):
            frame = expected_vox()
            if kind == 'value': frame.iloc[0, 0] += .01
            elif kind == 'dtype': frame = frame.astype(np.float32)
            elif kind == 'date': frame.index = pd.to_datetime(['2015-01-01', '2015-01-05'])
            elif kind == 'column': frame.columns = ['XLC']
            else: frame['EXTRA'] = 1.
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ind.check_vox(raw_rows(), ['2015-01-02', '2015-01-05'], frame)


if __name__ == '__main__':
    unittest.main(verbosity=2)
