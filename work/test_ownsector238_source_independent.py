"""Invented source-classification tests; no returns, strategy or real panel."""
import ast
import copy
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import ownsector238_source_independent as ind


def mapper(day, symbols, sectors, etfs):
    result = {}
    for symbol in symbols:
        if symbol in etfs:
            result[symbol] = None
        elif symbol in ind.FOREIGN:
            result[symbol] = ind.FOREIGN[symbol]
        elif pd.Timestamp(day) in sectors.index and symbol in sectors.columns:
            result[symbol] = ind.SECTOR.get(sectors.at[pd.Timestamp(day), symbol])
        else:
            result[symbol] = None
    return result


def fixture():
    axis = pd.bdate_range('2014-01-01', '2017-02-03')
    names = ['AAPL', 'TSM', 'FUND', 'UNKNOWN']
    dates = ['2016-08-17', '2016-08-31', '2016-09-30']
    adj = pd.DataFrame(100., index=axis,
                       columns=names + ['SPY', 'XLK', 'XLV', 'XLE'])
    sectors = pd.DataFrame(index=pd.to_datetime(dates), columns=names,
                           data=[['Technology', 'Energy', 'Energy', 'Unknown']] * 3)
    sectors.at[pd.Timestamp(dates[-1]), 'AAPL'] = 'Healthcare'
    meta = dict(dates=dates, symbols=[names.copy() for _ in dates], etfs=['FUND'])
    eligible = np.array([[True] * 4, [True, True, True, False],
                         [True, True, True, False]], dtype=bool)
    return {'adj': adj, 'sectors': sectors}, meta, [str(x.date()) for x in axis], eligible


def physical(data, n):
    panel, meta, calendar, eligible = data
    mm = copy.deepcopy(meta)
    mm['dates'] = mm['dates'][:n]
    mm['symbols'] = mm['symbols'][:n]
    if n:
        cutoff = mm['dates'][-1]
        end = calendar.index(cutoff) + 2
        pp = {key: frame.loc[:cutoff].copy(deep=True) for key, frame in panel.items()}
        cc = calendar[:end]
    else:
        pp = {key: frame.iloc[:0].copy(deep=True) for key, frame in panel.items()}
        cc = []
    return pp, mm, cc, eligible[:n].copy()


def run(data, mapping=mapper, limit=None):
    return ind.independent_sources(*data, mapping, limit=limit)


def slot(data, value, symbol='AAPL', role='security_start'):
    panel, meta, calendar, _ = data
    t = calendar.index(meta['dates'][1])
    day = calendar[t - (252 if role.endswith('start') else 42)]
    frame = panel['adj'].astype(object)
    frame.at[pd.Timestamp(day), symbol] = value
    panel['adj'] = frame
    return pd.Timestamp(day)


class SourceClassificationTests(unittest.TestCase):
    def test_admitted_exact_counts_and_no_strategy_scope(self):
        data = fixture()
        report = run(data)
        expected = dict(events=3, symbols_per_event=4, checked_cells=12,
            initial_partial_events=1, regular_events=2, original_eligible_cells=10,
            required_cells=6, stock_required_cells=4, etf_required_cells=2,
            mapping_available_cells=6, mapping_missing_cells=0,
            security_benchmark_endpoint_checks=24, market_endpoint_checks=4,
            valid_endpoints=28, missing_endpoints=0, invalid_endpoints=0,
            admitted_cells=6, gap_cells=0, market_gap_events=0)
        self.assertEqual(report['counts'], expected)
        self.assertEqual(report['status'], 'ADMITTED')
        self.assertEqual(report['gaps'], [])
        self.assertTrue(report['source_only'])
        for name in ('momentum_computed', 'eligibility_changed', 'rankings_computed',
                     'orders_computed', 'returns_computed', 'strict_pit'):
            self.assertIs(report[name], False)
        self.assertEqual(report['model_fits'], 0)
        receipt = ind.check_sources(*data, mapper, report)
        self.assertTrue(all(receipt[key] is True for key in ind.FLAGS))
        self.assertEqual(receipt['checked_endpoint_records'], 28)
        self.assertEqual(receipt['numerical_tolerance'], 0.)
        self.assertIs(receipt['producer_modules_imported'], False)

    def test_foreign_and_etf_precedence(self):
        data = fixture()
        report = run(data)
        self.assertEqual(report['records'][0]['benchmark'], 'XLK')
        self.assertEqual(report['records'][1]['benchmark'], 'XLK')
        self.assertEqual(report['records'][1]['mapping_source'], 'frozen_FOREIGN_override')
        self.assertEqual(report['records'][2]['benchmark'], 'SPY')
        data[1]['etfs'].append('TSM')
        report = run(data)
        self.assertEqual(report['records'][1]['benchmark'], 'SPY')
        self.assertEqual(report['records'][1]['mapping_source'], 'original_META.etfs_SPY')

    def test_query_classification_transition_no_forward_fill(self):
        data = fixture()
        self.assertEqual([r['benchmark'] for r in run(data)['records'] if r['symbol'] == 'AAPL'],
                         ['XLK', 'XLV'])
        data[0]['sectors'] = data[0]['sectors'].drop(pd.Timestamp(data[1]['dates'][1]))
        report = run(data)
        first = report['records'][0]
        self.assertIsNone(first['benchmark'])
        self.assertEqual(first['gap_reasons'], ['missing_benchmark_mapping'])
        self.assertEqual([x['status'] for x in first['endpoints']],
                         ['valid', 'valid', 'missing', 'missing'])
        self.assertEqual(report['status'], 'NOT_ADMITTED')

    def test_unknown_mapping_is_not_spy_fallback(self):
        data = fixture()
        data[3][1, 3] = True
        report = run(data)
        unknown = report['records'][3]
        self.assertIsNone(unknown['benchmark'])
        self.assertEqual(unknown['mapping_status'], 'missing')
        self.assertEqual(unknown['gap_reasons'], ['missing_benchmark_mapping'])
        self.assertEqual(report['counts']['mapping_missing_cells'], 1)
        self.assertEqual(report['counts']['missing_endpoints'], 2)

    def test_missing_vs_invalid_endpoint_classification(self):
        cases = [(None, 'missing', 'missing_value'),
                 (pd.NA, 'missing', 'missing_value'),
                 (float('nan'), 'missing', 'missing_value'),
                 (True, 'invalid', 'invalid_boolean'),
                 (np.bool_(False), 'invalid', 'invalid_boolean'),
                 ('100', 'invalid', 'invalid_nonreal'),
                 (complex(100, 0), 'invalid', 'invalid_nonreal'),
                 (math.inf, 'invalid', 'invalid_nonfinite'),
                 (-math.inf, 'invalid', 'invalid_nonfinite'),
                 (0., 'invalid', 'invalid_nonpositive'),
                 (-100., 'invalid', 'invalid_nonpositive'),
                 (10 ** 10000, 'invalid', 'invalid_numeric_conversion')]
        for value, status, reason in cases:
            with self.subTest(reason=reason, kind=type(value).__name__):
                data = fixture()
                slot(data, value)
                first = run(data)['records'][0]['endpoints'][0]
                self.assertEqual((first['status'], first['reason']), (status, reason))

    def test_missing_row_precedes_missing_column_and_missing_mapping(self):
        data = fixture()
        day = slot(data, 100.)
        data[0]['adj'] = data[0]['adj'].drop(index=day, columns='AAPL')
        first = run(data)['records'][0]
        self.assertEqual(first['endpoints'][0]['reason'], 'missing_row')
        self.assertEqual(first['endpoints'][1]['reason'], 'missing_column')
        self.assertEqual(first['gap_reasons'], ['missing_row', 'missing_column'])
        self.assertEqual(ind.endpoint(data[0]['adj'], 'benchmark_start', None, day)['reason'],
                         'missing_benchmark_mapping')

    def test_all_gaps_collected_market_first_then_cell_order(self):
        data = fixture()
        slot(data, None, 'SPY')
        slot(data, False, 'AAPL')
        slot(data, 'bad', 'TSM')
        report = run(data)
        gaps = [g for g in report['gaps'] if g['query_index'] == 1]
        self.assertEqual([g['scope'] for g in gaps], ['market', 'cell', 'cell', 'cell'])
        self.assertEqual([g['column_index'] for g in gaps], [None, 0, 1, 2])
        self.assertEqual(gaps[0]['benchmark'], 'SPY')
        self.assertEqual(report['counts']['market_gap_events'], 1)
        self.assertEqual(report['counts']['gap_cells'], 3)

    def test_empty_eligibility_still_requires_each_regular_spy(self):
        data = fixture()
        data[3][:] = False
        data[0]['adj'] = data[0]['adj'].drop(columns='SPY')
        report = run(data)
        self.assertEqual(report['records'], [])
        self.assertEqual(report['counts']['market_endpoint_checks'], 4)
        self.assertEqual(report['counts']['market_gap_events'], 2)
        self.assertEqual(report['counts']['required_cells'], 0)
        self.assertEqual(report['status'], 'NOT_ADMITTED')

    def test_initial_partial_bypasses_sources_and_mapper(self):
        data = physical(fixture(), 1)
        data[0]['adj'].iloc[:, :] = float('nan')
        def forbidden(*args):
            raise AssertionError('Partial must not invoke mapper')
        report = run(data, forbidden)
        self.assertTrue(report['admitted'])
        self.assertEqual(report['counts']['original_eligible_cells'], 4)
        self.assertEqual(report['counts']['required_cells'], 0)
        self.assertEqual(report['events'][0]['source_dates'], [])

    def test_only_exact_endpoint_dates_are_consumed(self):
        data = fixture()
        before = run(data)
        keep = set(x['date'] for r in before['records'] for x in r['endpoints'])
        data[0]['adj'] = data[0]['adj'].astype(object)
        for day in data[0]['adj'].index:
            if str(day.date()) not in keep:
                data[0]['adj'].loc[day, :] = 'not consumed'
        self.assertEqual(run(data), before)
        event = before['events'][1]
        t = data[2].index(data[1]['dates'][1])
        self.assertEqual(event['source_positions'], [t - 252, t - 42])
        self.assertEqual(event['source_dates'], [data[2][t - 252], data[2][t - 42]])

    def test_noneligible_prices_unread_but_current_mapping_schema_checked(self):
        data = fixture()
        data[0]['adj'] = data[0]['adj'].astype(object)
        data[0]['adj']['UNKNOWN'] = 'invalid but unused'
        self.assertTrue(run(data)['admitted'])
        data[0]['sectors'].at[pd.Timestamp(data[1]['dates'][1]), 'UNKNOWN'] = ['bad schema']
        with self.assertRaises((TypeError, ValueError)):
            run(data)

    def test_mapper_exact_keys_types_and_precedence_checked(self):
        data = fixture()
        for mutation in ('drop', 'extra', 'bad_type', 'foreign', 'etf'):
            def changed(day, symbols, sectors, etfs, mutation=mutation):
                result = mapper(day, symbols, sectors, etfs)
                if mutation == 'drop':result.pop('UNKNOWN')
                elif mutation == 'extra':result['EXTRA'] = None
                elif mutation == 'bad_type':result['UNKNOWN'] = ['XLK']
                elif mutation == 'foreign':result['TSM'] = 'XLE'
                else:result['FUND'] = 'SPY'
                return result
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run(data, changed)

    def test_future_sector_rows_are_physically_hidden_from_mapper(self):
        data = fixture()
        data[0]['sectors'].loc[pd.Timestamp('2017-01-31')] = [['unhashable']] * 4
        calls = []
        def watched(day, symbols, sectors, etfs):
            self.assertTrue(all(sectors.index <= pd.Timestamp(day)))
            calls.append(day)
            return mapper(day, symbols, sectors, etfs)
        self.assertTrue(run(data, watched)['admitted'])
        self.assertEqual(calls, data[1]['dates'][1:])

    def test_no_input_mutation_and_mapper_receives_detached_sectors(self):
        data = fixture()
        before = copy.deepcopy(data)
        def mutating(day, symbols, sectors, etfs):
            result = mapper(day, symbols, sectors, etfs)
            sectors.iloc[:, :] = 'changed'
            symbols.clear();etfs.clear()
            return result
        run(data, mutating)
        for key in data[0]:pd.testing.assert_frame_equal(data[0][key], before[0][key])
        self.assertEqual(data[1], before[1]);self.assertEqual(data[2], before[2])
        np.testing.assert_array_equal(data[3], before[3])


class StructuralAndReceiptTests(unittest.TestCase):
    def test_invalid_source_axes_are_hard_failures(self):
        for mutation in ('duplicate', 'reverse', 'timezone', 'time', 'duplicate_column'):
            data = fixture();frame = data[0]['adj']
            if mutation == 'duplicate':frame = pd.concat([frame.iloc[:1], frame])
            elif mutation == 'reverse':frame = frame.iloc[::-1]
            elif mutation == 'timezone':frame.index = frame.index.tz_localize('UTC')
            elif mutation == 'time':frame.index = frame.index + pd.Timedelta(hours=1)
            else:frame.columns = [frame.columns[0]] * len(frame.columns)
            data[0]['adj'] = frame
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):run(data)

    def test_calendar_and_query_clock_errors_are_structural(self):
        for mutation in ('noncanonical', 'weekend', 'duplicate', 'not_monthend',
                         'no_witness', 'short_history'):
            data = list(fixture())
            if mutation == 'noncanonical':data[2][0] = '2014-1-1'
            elif mutation == 'weekend':data[2][0] = '2013-12-29'
            elif mutation == 'duplicate':data[2][1] = data[2][0]
            elif mutation == 'not_monthend':data[1]['dates'][1] = '2016-08-30'
            elif mutation == 'no_witness':data[2] = data[2][:data[2].index(data[1]['dates'][-1]) + 1]
            else:data[2] = data[2][data[2].index('2016-01-01'):]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):run(data)

    def test_meta_boolean_and_panel_schemas_rejected(self):
        for mutation in ('int_e', 'duplicate_id', 'bad_width', 'missing_etfs',
                         'extra_panel', 'partial_not_first'):
            data = list(fixture())
            if mutation == 'int_e':data[3] = data[3].astype(int)
            elif mutation == 'duplicate_id':data[1]['symbols'][1][1] = 'AAPL'
            elif mutation == 'bad_width':data[3] = data[3][:, :3]
            elif mutation == 'missing_etfs':del data[1]['etfs']
            elif mutation == 'extra_panel':data[0]['other'] = data[0]['adj']
            else:data[1]['dates'] = ['2016-07-29', '2016-08-17', '2016-09-30']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):run(data)

    def test_logical_and_physical_prefixes_and_empty_calendar(self):
        data = fixture();full = run(data)
        for n in range(4):
            with self.subTest(n=n):
                logical = run(data, limit=n)
                physical_report = run(physical(data, n))
                self.assertEqual(logical, physical_report)
                self.assertEqual(logical['events'], full['events'][:n])
                self.assertEqual(logical['records'], [r for r in full['records'] if r['query_index'] < n])
                self.assertEqual(logical['counts']['events'], n)
        self.assertEqual(run(physical(data, 0))['counts']['symbols_per_event'], 4)
        for limit in (-1, 4, True, 1.0):
            with self.assertRaises(ValueError):run(data, limit=limit)

    def test_future_numeric_and_metadata_poison_does_not_leak(self):
        data = list(fixture());expected = run(data, limit=2)
        data[3] = data[3].astype(object);data[3][2, :] = 'future bool poison'
        data[1]['dates'][2] = 'future date poison'
        data[1]['symbols'][2] = ['future identity poison']
        data[0]['adj'] = data[0]['adj'].astype(object)
        data[0]['adj'].loc['2016-09-01':] = 'future prices'
        data[0]['sectors'].loc['2016-09-01':] = 'future labels'
        self.assertEqual(run(data, limit=2), expected)
        with self.assertRaises(ValueError):run(data)

    def test_all_five_registered_prefix_sizes_on_invented_122_rows(self):
        axis = pd.bdate_range('2014-01-01', '2027-01-04')
        dates = ['2016-08-17'] + [str(d.date()) for d in
                pd.date_range('2016-08-31', periods=121, freq='BME')]
        names = ['AAPL', 'TSM', 'FUND', 'UNKNOWN']
        meta = dict(dates=dates, symbols=[names.copy() for _ in dates], etfs=['FUND'])
        adj = pd.DataFrame(100., index=axis,
                           columns=names + ['SPY', 'XLK', 'XLC'])
        sectors = pd.DataFrame(index=pd.to_datetime(dates), columns=names,
            data=[['Technology', 'Energy', 'Energy', 'Unknown']] * 122)
        sectors.loc['2018-09-01':, 'AAPL'] = 'Communication Services'
        eligible = np.tile([True, True, True, False], (122, 1))
        data = ({'adj': adj, 'sectors': sectors}, meta,
                [str(d.date()) for d in axis], eligible)
        full = run(data)
        self.assertEqual(full['counts']['regular_events'], 121)
        for n in (0, 1, 53, 88, 122):
            with self.subTest(n=n):
                limited = run(data, limit=n)
                self.assertEqual(run(physical(data, n)), limited)
                self.assertEqual(limited['events'], full['events'][:n])
                self.assertEqual(limited['records'],
                                 [r for r in full['records'] if r['query_index'] < n])

    def test_frozen_mapping_literal_and_digest_are_verified(self):
        constants = ind.mapping_constants()
        self.assertEqual(len(constants['SECTOR']), 11)
        self.assertEqual(len(constants['FOREIGN']), 14)
        self.assertIn(ind.MAPPING, ind.dependency_files())
        with mock.patch.object(ind, 'MAPPING_SHA', '0' * 64), self.assertRaises(ValueError):
            run(fixture())
        with mock.patch.dict(ind.FOREIGN, {'TSM': 'XLE'}), self.assertRaises(ValueError):
            run(fixture())

    def test_complete_report_tamper_and_type_tamper_rejected(self):
        data = fixture();report = run(data)
        mutations = [lambda r:r['counts'].__setitem__('valid_endpoints', 27),
                     lambda r:r['events'][1].__setitem__('source_dates', ['2000-01-03'] * 2),
                     lambda r:r['records'][0]['endpoints'][0].__setitem__('reason', 'invented'),
                     lambda r:r.__setitem__('admitted', 1),
                     lambda r:r.__setitem__('model_fits', 1),
                     lambda r:r.__setitem__('extra', None)]
        for mutation in mutations:
            bad = copy.deepcopy(report);mutation(bad)
            with self.assertRaises(ValueError):ind.check_sources(*data, mapper, bad)

    def test_source_module_has_no_producer_or_numeric_strategy_imports(self):
        tree = ast.parse(Path(ind.__file__).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):names.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):names.append(node.module)
        self.assertFalse(any(name and ('producer' in name or 'model' in name or 'engine' in name
                                      or 'ownsector238_source_audit' in name) for name in names))


class BilateralCrossTests(unittest.TestCase):
    def test_unique_both_fixtures_every_prefix_full_report(self):
        import ownsector238_source_audit as producer
        import test_ownsector238_source_audit as production_tests
        for data in (fixture(), production_tests.fixture()):
            for n in range(len(data[1]['dates']) + 1):
                with self.subTest(width=data[3].shape[1], n=n):
                    expected = run(data, limit=n)
                    produced = producer.inspect_sources(*data, mapper, limit=n)
                    self.assertEqual(produced, expected)
                    self.assertTrue(ind.check_sources(*data, mapper, produced, limit=n)['passed'])
                    small = physical(data, n)
                    self.assertEqual(producer.inspect_sources(*small, mapper), expected)
                    self.assertEqual(run(small), expected)

    def test_shared_missing_invalid_and_mapping_gap_cross(self):
        import ownsector238_source_audit as producer
        for value in (None, pd.NA, float('nan'), True, 'bad', math.inf, 0., -1., 10 ** 10000):
            data = fixture();slot(data, value)
            with self.subTest(kind=type(value).__name__):
                self.assertEqual(producer.inspect_sources(*data, mapper), run(data))
        data = fixture();data[3][1, 3] = True
        data[0]['adj'] = data[0]['adj'].drop(columns='SPY')
        self.assertEqual(producer.inspect_sources(*data, mapper), run(data))


if __name__ == '__main__':
    unittest.main()
