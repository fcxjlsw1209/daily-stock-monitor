"""Round233 synthetic-only exact ranks, support, timing and API tests."""
import copy
from fractions import Fraction
import json
import math
import unittest

import numpy as np
import pandas as pd

import monthly_sector_signedrank233_kernel as k


def block_prices(stock_steps=None, sector_steps=None):
    stock_steps = [1, -1, 0] * 4 if stock_steps is None else stock_steps
    sector_steps = [0] * 12 if sector_steps is None else sector_steps
    if len(stock_steps) != 12 or len(sector_steps) != 12:
        raise ValueError('Twelve synthetic block steps required')
    def path(steps):
        endpoints = np.r_[0., np.cumsum(steps)]
        return np.exp2(np.interp(np.arange(253), np.arange(0, 253, 21), endpoints))
    return path(stock_steps), path(sector_steps)


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-10')
    x = np.arange(len(dates), dtype=float)
    adj = pd.DataFrame({
        'AAA': 100. * np.exp(.0003 * x + .2 * np.sin(x / 140.) + .05 * np.sin(x / 14.)),
        'BBB': 80. * np.exp(.0004 * x + .19 * np.cos(x / 130.) + .03 * np.cos(x / 17.)),
        'FUND': 70. * np.exp(.0003 * x + .1 * np.sin(x / 130.)),
        'XLK': 90. * np.exp(.00025 * x + .05 * np.sin(x / 95.)),
        'XLY': 95. * np.exp(.0002 * x + .1 * np.cos(x / 90.)),
    }, index=dates)
    sectors = pd.DataFrame('Technology', index=dates, columns=adj.columns)
    meta = {'dates': ['2016-08-17', '2016-08-31', '2016-09-30'],
            'symbols': [['AAA', 'BBB', 'FUND'] for _ in range(3)], 'etfs': ['FUND']}
    return {'adj': adj, 'sectors': sectors}, meta, [str(d.date()) for d in dates]


def mapper(day, symbols, sectors, etfs):
    day = pd.Timestamp(day)
    if len(sectors) and sectors.index[-1] > day:
        raise AssertionError('Future sector rows exposed')
    answer = {}
    for symbol in symbols:
        if symbol in etfs:
            answer[symbol] = None
        elif symbol == 'BABA':
            answer[symbol] = 'XLY'
        else:
            value = sectors.at[day, symbol] if day in sectors.index and symbol in sectors else None
            answer[symbol] = {'Technology': 'XLK', 'Consumer Cyclical': 'XLY',
                             'Communication Services': 'XLC'}.get(value)
    return answer


class MeasureTests(unittest.TestCase):
    def test_exact_signs_ties_ranks_and_all_counts(self):
        result = k.measure(*block_prices())
        self.assertTrue(result['available'])
        self.assertEqual(result['relative_signs'], [1, -1, 0] * 4)
        self.assertEqual(result['exact_relative_z'], ['1/3', '-1/3', '0'] * 4)
        self.assertEqual(result['twice_magnitude_ranks'], [17, 17, 5] * 4)
        self.assertEqual(result['endpoint_indices'], list(range(0, 253, 21)))
        self.assertEqual(result['block_start_indices'], list(range(0, 252, 21)))
        self.assertEqual(result['block_end_indices'], list(range(21, 253, 21)))
        for key in ('relative_positive_count', 'relative_negative_count', 'relative_tie_count'):
            self.assertEqual(result[key], 4)
        self.assertEqual(result['relative_signed_count'], 0)
        self.assertEqual(result['candidate_numerator'], 0)
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['control'], 0.)
        self.assertEqual(result['twice_rank_sum'], 156)
        self.assertEqual(result['relative_sign_observations'], 12)
        self.assertEqual(result['magnitude_rank_observations'], 12)
        self.assertEqual(result['magnitude_pair_comparisons'], 66)
        self.assertFalse(result['constant_magnitude_copy'])

    def test_many_small_wins_fewer_large_losses(self):
        result = k.measure(*block_prices([1] * 8 + [-4] * 4))
        self.assertEqual(result['twice_magnitude_ranks'], [9] * 8 + [21] * 4)
        self.assertEqual(result['control'], .25)
        self.assertEqual(result['candidate_numerator'], -12)
        self.assertEqual(result['candidate'], -12 / 208)
        self.assertLess(result['candidate'], 0.)

    def test_large_wins_also_receive_more_weight(self):
        result = k.measure(*block_prices([-1] * 8 + [4] * 4))
        self.assertEqual(result['control'], -.25)
        self.assertEqual(result['candidate'], 12 / 208)
        self.assertGreater(result['candidate'], 0.)

    def test_equal_magnitude_copy_with_mixed_signs(self):
        for steps in ([1] * 12, [-1] * 12, [0] * 12, [1] * 9 + [-1] * 3):
            result = k.measure(*block_prices(steps))
            self.assertTrue(result['constant_magnitude_copy'])
            self.assertEqual(result['twice_magnitude_ranks'], [13] * 12)
            self.assertEqual(result['candidate'].hex(), result['control'].hex())
            self.assertEqual(result['control'], sum((v > 0) - (v < 0) for v in steps) / 16)

    def test_all_zeros_participate_in_rank_and_denominator(self):
        result = k.measure(*block_prices([1] + [0] * 11))
        self.assertEqual(result['twice_magnitude_ranks'], [24] + [12] * 11)
        self.assertEqual(result['candidate'], 24 / 208)
        self.assertEqual(result['control'], 1 / 16)
        self.assertEqual(result['relative_tie_count'], 11)

    def test_relative_not_absolute_sign(self):
        result = k.measure(*block_prices([1] * 12, [2] * 12))
        self.assertEqual(result['relative_signs'], [-1] * 12)
        self.assertEqual(result['candidate'], -.75)
        self.assertEqual(result['control'], -.75)
        result = k.measure(*block_prices([1] * 12, [1] * 12))
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['control'], 0.)

    def test_exact_near_zero_despite_rounded_gross_tie(self):
        eps = 2. ** -52
        a, b, c, d = 1. + eps, 1., 1., 1. - eps
        self.assertEqual(a / b, c / d)
        self.assertEqual(a * d, c * b)
        stock = np.ones(253); sector = np.ones(253)
        stock[21] = a; sector[0] = d
        result = k.measure(stock, sector)
        self.assertEqual(result['relative_signs'][0], -1)
        z = Fraction(result['exact_relative_z'][0])
        self.assertLess(z, 0)
        gs, gb = Fraction.from_float(a) / Fraction.from_float(b), Fraction.from_float(c) / Fraction.from_float(d)
        self.assertEqual(z, (gs - gb) / (gs + gb))
        self.assertGreater(result['twice_magnitude_ranks'][0], 0)

    def test_extreme_magnitudes_rank_exactly_after_float_would_tie(self):
        tiny = float(np.nextafter(0., 1.))
        stock = np.ones(253); sector = np.ones(253)
        endpoints = [tiny, 1e307, tiny, 1e308, tiny, 1e307, tiny,
                     1e308, tiny, 1e307, tiny, 1e308, tiny]
        stock[::21] = endpoints
        result = k.measure(stock, sector)
        self.assertTrue(result['available'])
        exact = [abs(Fraction(z)) for z in result['exact_relative_z']]
        self.assertEqual({float(z) for z in exact}, {1.})
        self.assertEqual(len(set(exact)), 2)
        self.assertEqual(set(result['twice_magnitude_ranks']), {7, 19})
        self.assertEqual(result['twice_rank_sum'], 156)
        self.assertFalse(result['constant_magnitude_copy'])
        self.assertTrue(math.isfinite(result['candidate']))

    def test_positive_power_of_two_unit_rescaling(self):
        stock, sector = block_prices([1] * 8 + [-4] * 4, [1, 0] * 6)
        before = stock.copy(), sector.copy()
        self.assertEqual(k.measure(stock, sector), k.measure(stock * 8., sector / 4.))
        np.testing.assert_array_equal(stock, before[0])
        np.testing.assert_array_equal(sector, before[1])

    def test_swap_stock_sector_reverses_signed_statistics(self):
        stock, sector = block_prices([1] * 8 + [-4] * 4, [1, 0] * 6)
        a, b = k.measure(stock, sector), k.measure(sector, stock)
        self.assertEqual(a['candidate'], -b['candidate'])
        self.assertEqual(a['control'], -b['control'])
        self.assertEqual(a['twice_magnitude_ranks'], b['twice_magnitude_ranks'])
        self.assertEqual(a['relative_signs'], [-v for v in b['relative_signs']])

    def test_block_order_permutation_does_not_change_aggregation(self):
        steps = [1] * 8 + [-4] * 4
        a = k.measure(*block_prices(steps))
        b = k.measure(*block_prices(steps[::-1]))
        for key in ('candidate', 'control', 'candidate_numerator', 'twice_rank_sum'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['twice_magnitude_ranks'], b['twice_magnitude_ranks'][::-1])

    def test_only_endpoints_determine_measure_but_all_prices_support_it(self):
        stock, sector = block_prices()
        before = k.measure(stock, sector)
        interior = np.ones(253, bool); interior[::21] = False
        stock[interior] = 1e200; sector[interior] = 1e-200
        self.assertEqual(k.measure(stock, sector), before)

    def test_every_source_position_must_be_positive_finite(self):
        stock, sector = block_prices()
        for index in (0, 11, 21, 252):
            for bad in (np.nan, np.inf, -np.inf, 0., -1.):
                for side in ('stock', 'benchmark'):
                    a, b = stock.copy(), sector.copy()
                    (a if side == 'stock' else b)[index] = bad
                    result = k.measure(a, b)
                    self.assertFalse(result['available'])
                    self.assertEqual(result['reason'], 'invalid_' + side + '_price')
                    self.assertEqual(result['endpoint_indices'], [])
                    self.assertEqual(result['relative_sign_observations'], 0)
                    self.assertTrue(np.isnan(result['candidate']) and np.isnan(result['control']))

    def test_reject_boolean_string_complex_none_and_overflow(self):
        stock, sector = block_prices()
        for bad in (True, np.bool_(False), '1.0', 'bad', 1 + 0j, None):
            for side in ('stock', 'benchmark'):
                a, b = stock.astype(object), sector.astype(object)
                (a if side == 'stock' else b)[11] = bad
                result = k.measure(a, b)
                self.assertEqual(result['reason'], 'nonnumeric_' + side + '_price')
                self.assertFalse(result['available'])
        a = stock.astype(object); a[11] = 10 ** 10000
        self.assertEqual(k.measure(a, sector)['reason'], 'invalid_stock_price')

    def test_source_reason_priority_stock_then_first_bad_position(self):
        stock, sector = block_prices()
        stock = stock.astype(object); sector = sector.astype(object)
        stock[11] = np.nan; stock[21] = 'bad'; sector[0] = False
        self.assertEqual(k.measure(stock, sector)['reason'], 'invalid_stock_price')
        stock[0] = 'bad'
        self.assertEqual(k.measure(stock, sector)['reason'], 'nonnumeric_stock_price')

    def test_invalid_shapes_raise(self):
        stock, sector = block_prices()
        for a, b in ((stock[:-1], sector), (stock[:, None], sector),
                     (stock, sector[:-1]), (stock, sector[:, None])):
            with self.assertRaises(ValueError): k.measure(a, b)


class BuilderTests(unittest.TestCase):
    def test_window_dates_monthends_counts_and_raw_first_row(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertTrue(values['available'][:, :2].all())
        self.assertFalse(values['available'][:, 2].any())
        self.assertEqual(audit['relative_sign_observations'], 72)
        self.assertEqual(audit['magnitude_rank_observations'], 72)
        self.assertEqual(audit['available_cells'], 6)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertEqual(audit['control_denominator'], 16)
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual(event['price_start'], calendar[t - 294])
            self.assertEqual(event['price_end'], calendar[t - 42])
            cell = event['cells'][0]
            self.assertEqual(cell['block_start_dates'], [calendar[t - 294 + j * 21] for j in range(12)])
            self.assertEqual(cell['block_end_dates'], [calendar[t - 294 + (j + 1) * 21] for j in range(12)])
            self.assertEqual(cell['endpoint_dates'][-1], calendar[t - 42])
            self.assertEqual(cell['windows']['formation']['return_count'], 252)

    def test_verified_monthends_and_only_exact_initial_exception(self):
        panel, meta, calendar = fixture()
        for row, day in ((0, '2016-08-16'), (1, '2016-08-30')):
            bad = copy.deepcopy(meta); bad['dates'][row] = day
            with self.assertRaises(ValueError): k.build_features(panel, bad, calendar, mapper)
        bad = copy.deepcopy(meta); bad['dates'] = ['2016-07-29', '2016-08-17', '2016-09-30']
        with self.assertRaises(ValueError): k.build_features(panel, bad, calendar, mapper)
        end = calendar.index(meta['dates'][-1])
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar[:end + 1], mapper)
        # A synthetic holiday on Aug31 makes Aug30 the actual last official session.
        changed = [d for d in calendar if d != '2016-08-31']
        good = copy.deepcopy(meta); good['dates'][1] = '2016-08-30'
        self.assertEqual(k.build_features(panel, good, changed, mapper)[1]['events'], 3)

    def test_query_mapping_fixed_through_history_and_unknown_neutral(self):
        panel, meta, calendar = fixture()
        base = k.build_features(panel, meta, calendar, mapper)
        panel['sectors'].loc[:, :] = 'Unknown'
        for date in meta['dates']:
            panel['sectors'].loc[date, ['AAA', 'BBB']] = 'Technology'
        changed = k.build_features(panel, meta, calendar, mapper)
        for key in base[0]: np.testing.assert_array_equal(base[0][key], changed[0][key])
        self.assertEqual(base[1], changed[1])
        panel['sectors'].loc[meta['dates'][0], 'AAA'] = 'Unknown'
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertFalse(values['available'][0, 0])
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_benchmark')
        self.assertEqual(audit['event_history'][0]['cells'][2]['reason'], 'etf_neutral')

    def test_missing_official_row_interior_not_compressed(self):
        panel, meta, calendar = fixture()
        t = calendar.index(meta['dates'][0]); removed = calendar[t - 294 + 11]
        panel['adj'] = panel['adj'].drop(pd.Timestamp(removed))
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertFalse(values['available'][0].any())
        self.assertEqual(audit['event_history'][0]['missing_official_sessions'], [removed])
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_official_session')

    def test_insufficient_history_missing_symbols_and_benchmarks(self):
        panel, meta, calendar = fixture()
        meta = copy.deepcopy(meta); meta['dates'][0] = '2014-01-31'
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        self.assertIsNone(audit['event_history'][0]['windows'])
        panel['adj'] = panel['adj'].drop(columns='AAA')
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertEqual(audit['event_history'][1]['cells'][0]['reason'], 'missing_symbol')
        panel['adj'] = panel['adj'].drop(columns='XLK')
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertEqual(audit['event_history'][1]['cells'][1]['reason'], 'missing_benchmark_symbol')

    def test_every_prefix_future_numeric_poison_and_empty(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar, mapper)
        for n in range(4):
            poison = copy.deepcopy(panel); poison['adj'] = poison['adj'].astype(object)
            start = 0 if n == 0 else calendar.index(meta['dates'][n - 1]) - 42 + 1
            poison['adj'].iloc[start:, :] = 'unread future price'
            got, record = k.build_features(poison, meta, calendar, mapper, limit=n)
            for key in values: np.testing.assert_array_equal(got[key], values[key][:n])
            self.assertEqual(record['event_history'], audit['event_history'][:n])
            self.assertEqual(got['available'].shape, (n, 3))

    def test_source_before_window_ignored_and_input_unchanged(self):
        panel, meta, calendar = fixture(); before = copy.deepcopy(panel)
        base = k.build_features(panel, meta, calendar, mapper)
        poison = copy.deepcopy(panel); poison['adj'] = poison['adj'].astype(object)
        poison['adj'].iloc[:calendar.index(meta['dates'][0]) - 294, :] = 'unread old price'
        got = k.build_features(poison, meta, calendar, mapper)
        for key in base[0]: np.testing.assert_array_equal(got[0][key], base[0][key])
        self.assertEqual(got[1], base[1])
        pd.testing.assert_frame_equal(panel['adj'], before['adj'])
        pd.testing.assert_frame_equal(panel['sectors'], before['sectors'])

    def test_mapping_and_schema_errors(self):
        panel, meta, calendar = fixture()
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar, mapper, limit=True)
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar[::-1], mapper)
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar, lambda *args: {})
        with self.assertRaises(ValueError):
            k.build_features(panel, meta, calendar, lambda day, ss, *args: {s: 'XLK' for s in ss})
        bad = copy.deepcopy(meta); bad['symbols'][0] = ['AAA', 'AAA', 'FUND']
        with self.assertRaises(ValueError): k.build_features(panel, bad, calendar, mapper)
        bad = copy.deepcopy(meta); bad.pop('etfs')
        with self.assertRaises(ValueError): k.build_features(panel, bad, calendar, mapper)
        badpanel = copy.deepcopy(panel); badpanel['adj'] = badpanel['adj'].iloc[::-1]
        with self.assertRaises(ValueError): k.build_features(badpanel, meta, calendar, mapper)


class ExtraBuilderTests(unittest.TestCase):
    def test_audit_counts_json_and_common_raw_mask(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertEqual(set(values), {'candidate', 'control', 'available'})
        self.assertEqual(values['candidate'].dtype, np.dtype('float64'))
        self.assertEqual(values['available'].dtype, np.dtype('bool'))
        self.assertEqual(audit['source_price_observations'], 506 * 6)
        self.assertEqual(audit['magnitude_pair_comparisons'], 66 * 6)
        self.assertEqual(audit['signed_rank_estimates'], 6)
        self.assertEqual(audit['sign_balance_estimates'], 6)
        self.assertEqual(audit['checked_cells'], 9)
        self.assertEqual(audit['reason_counts'], {'available': 6, 'etf_neutral': 3})
        self.assertFalse(audit['cross_sectional_ranking_performed'])
        self.assertTrue(audit['within_cell_ranking_performed'])
        self.assertEqual(audit['candidate_denominator'], 208)
        self.assertEqual(audit['expected_twice_rank_sum'], 156)
        self.assertTrue(all(key in audit for key in k.COUNT_FIELDS))
        self.assertTrue(all(type(audit[key]) is int for key in k.COUNT_FIELDS))
        for key in ('candidate', 'control'):
            self.assertTrue(np.isnan(values[key][~values['available']]).all())
        self.assertEqual(json.loads(json.dumps(audit, allow_nan=False)), audit)

    def test_every_physical_prefix_and_empty_metadata(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar, mapper)
        for n in range(len(meta['dates']) + 1):
            prefix = dict(meta, dates=meta['dates'][:n], symbols=meta['symbols'][:n])
            if n:
                last = prefix['dates'][-1]
                end = calendar.index(last)
                physical = {key: frame.loc[:last].copy() for key, frame in panel.items()}
                cal = calendar[:end + 2]
            else:
                physical = {key: frame.iloc[:0].copy() for key, frame in panel.items()}
                cal = calendar[:1]
            actual, record = k.build_features(physical, prefix, cal, mapper)
            self.assertEqual(record['event_history'], audit['event_history'][:n])
            for key in values:
                expected = values[key][:n] if n else np.empty((0, 0), dtype=values[key].dtype)
                np.testing.assert_array_equal(actual[key], expected)
            self.assertEqual(actual['available'].shape, (n, 3) if n else (0, 0))

    def test_no_numeric_read_for_neutral_cells_and_future_sectors(self):
        panel, meta, calendar = fixture()
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].loc[:, 'FUND'] = 'ETF numeric must not be read'
        last = meta['dates'][0]
        panel['sectors'].loc[panel['sectors'].index > last, :] = 'future sector poison'
        values, audit = k.build_features(panel, meta, calendar, mapper, limit=1)
        self.assertTrue(values['available'][0, :2].all())
        self.assertEqual(audit['reason_counts'], {'available': 2, 'etf_neutral': 1})

    def test_builder_bool_and_string_not_silently_coerced(self):
        panel, meta, calendar = fixture()
        start = calendar.index(meta['dates'][0]) - 294
        for bad in (True, '1.0', 1 + 0j):
            altered = copy.deepcopy(panel)
            altered['adj'] = altered['adj'].astype(object)
            altered['adj'].loc[calendar[start + 3], 'AAA'] = bad
            values, audit = k.build_features(altered, meta, calendar, mapper)
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'nonnumeric_stock_price')
            self.assertFalse(values['available'][0, 0])

    def test_duplicate_calendar_index_columns_and_invalid_limits(self):
        panel, meta, calendar = fixture()
        for altered in (calendar[:5] + [calendar[4]] + calendar[5:],
                        calendar[:5] + ['2014-01-04'] + calendar[5:],
                        []):
            with self.assertRaises(ValueError): k.build_features(panel, meta, altered, mapper)
        bad = copy.deepcopy(panel); bad['adj'] = pd.concat([bad['adj'].iloc[:1], bad['adj']])
        with self.assertRaises(ValueError): k.build_features(bad, meta, calendar, mapper)
        bad = copy.deepcopy(panel); bad['adj'].columns = ['AAA', 'AAA', 'FUND', 'XLK', 'XLY']
        with self.assertRaises(ValueError): k.build_features(bad, meta, calendar, mapper)
        for limit in (-1, 4, 1., np.int64(1), True):
            with self.assertRaises(ValueError): k.build_features(panel, meta, calendar, mapper, limit=limit)

    def test_metadata_width_etfs_and_extra_panel_inputs_rejected(self):
        panel, meta, calendar = fixture()
        for altered in (dict(meta, etfs=['FUND', 'FUND']), dict(meta, etfs='FUND'),
                        dict(meta, dates=meta['dates'][:-1]),
                        dict(meta, symbols=[['AAA'], meta['symbols'][1], meta['symbols'][2]])):
            with self.assertRaises(ValueError): k.build_features(panel, altered, calendar, mapper)
        with self.assertRaises(ValueError):
            k.build_features(dict(panel, volume=panel['adj']), meta, calendar, mapper)
        with self.assertRaises(TypeError): k.build_features(panel, meta, calendar, None)

    def test_constant_copy_descriptor_counts_include_zero_price_path(self):
        panel, meta, calendar = fixture()
        panel['adj'].loc[:, :] = 100.
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertEqual(audit['constant_magnitude_copies'], 6)
        self.assertEqual(audit['signed_rank_estimates'], 6)
        self.assertEqual(audit['sign_balance_estimates'], 6)
        self.assertEqual(audit['magnitude_pair_comparisons'], 396)
        self.assertTrue((values['candidate'][values['available']] == 0.).all())
        for event in audit['event_history']:
            self.assertTrue(event['cells'][0]['constant_magnitude_copy'])


if __name__ == '__main__':
    unittest.main()
