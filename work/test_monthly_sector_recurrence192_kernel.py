"""Round192 synthetic-only source, exact-sign, timing and recurrence tests."""
import copy
from fractions import Fraction
import math
import unittest

import numpy as np
import pandas as pd

import monthly_sector_recurrence192_kernel as k


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
    def test_offsets_signs_and_all_counts(self):
        result = k.measure(*block_prices())
        self.assertTrue(result['available'])
        self.assertEqual(result['endpoint_indices'], list(range(0, 253, 21)))
        self.assertEqual(result['block_start_indices'], list(range(0, 252, 21)))
        self.assertEqual(result['block_end_indices'], list(range(21, 253, 21)))
        for side in ('relative', 'absolute'):
            self.assertEqual(result[side + '_signs'], [1, -1, 0] * 4)
            self.assertEqual(result[side + '_positive_count'], 4)
            self.assertEqual(result[side + '_negative_count'], 4)
            self.assertEqual(result[side + '_tie_count'], 4)
            self.assertEqual(result[side + '_signed_count'], 0)
            self.assertEqual(result[side + '_sign_observations'], 12)
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['control'], 0.)

    def test_common_support_relative_and_absolute_distinct(self):
        result = k.measure(*block_prices([1] * 12, [2] * 12))
        self.assertEqual(result['relative_signs'], [-1] * 12)
        self.assertEqual(result['absolute_signs'], [1] * 12)
        self.assertEqual(result['candidate'], -.75)
        self.assertEqual(result['control'], .75)
        self.assertEqual(result['candidate'] * .025, -.018750000000000003)

    def test_zero_ties_neutral_pseudo_blocks_and_discrete_scale(self):
        for steps, expected in (([1] * 12, .75), ([-1] * 12, -.75), ([0] * 12, 0.),
                                ([1] + [0] * 11, 1. / 16)):
            result = k.measure(*block_prices(steps))
            self.assertEqual(result['candidate'], expected)
            self.assertEqual(result['control'], expected)
        stock, sector = block_prices([1] * 12, [1] * 12)
        result = k.measure(stock, sector)
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['relative_tie_count'], 12)

    def test_exact_ratios_distinguish_rounded_ties(self):
        eps = 2. ** -52
        a, b, c, d = 1. + eps, 1., 1., 1. - eps
        self.assertEqual(a / b, c / d)
        self.assertEqual(a * d, c * b)
        self.assertEqual(k._relative_sign(a, b, c, d), -1)
        self.assertEqual(k._relative_sign(c, d, a, b), 1)
        self.assertEqual(k._relative_sign(8., 4., 2., 1.), 0)
        stock = np.ones(253); sector = np.ones(253)
        stock[21] = a; sector[0] = d
        result = k.measure(stock, sector)
        self.assertEqual(result['relative_signs'][0], -1)

    def test_extreme_finite_comparisons_do_not_overflow(self):
        tiny = float(np.nextafter(0., 1.))
        for values in ((1e308, tiny, 1e307, tiny * 2),
                       (tiny, 1e308, tiny * 2, 1e307),
                       (1e308, 1e308, tiny, tiny)):
            a, b, c, d = map(Fraction.from_float, values)
            expected = (a / b > c / d) - (a / b < c / d)
            self.assertEqual(k._relative_sign(*values), expected)
        stock = np.ones(253)
        sector = np.ones(253)
        stock[::21] = [tiny if i % 2 == 0 else 1e308 for i in range(13)]
        sector[::21] = [1e308 if i % 2 == 0 else tiny for i in range(13)]
        result = k.measure(stock, sector)
        self.assertTrue(result['available'])
        self.assertEqual(result['relative_signs'], [1, -1] * 6)
        self.assertTrue(math.isfinite(result['candidate']))

    def test_full_window_validity_even_unused_interior(self):
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
        a = stock.astype(object); a[11] = 'bad'
        self.assertEqual(k.measure(a, sector)['reason'], 'nonnumeric_stock_price')
        b = sector.astype(object); b[11] = 'bad'
        self.assertEqual(k.measure(stock, b)['reason'], 'nonnumeric_benchmark_price')
        with self.assertRaises(ValueError): k.measure(stock[:-1], sector)
        with self.assertRaises(ValueError): k.measure(stock[:, None], sector)

    def test_only_endpoints_determine_measure_but_all_prices_support_it(self):
        stock, sector = block_prices()
        before = k.measure(stock, sector)
        interior = np.ones(253, bool); interior[::21] = False
        stock[interior] = 1e200
        sector[interior] = 1e-200
        self.assertEqual(k.measure(stock, sector), before)

    def test_scale_invariance_and_no_source_mutation(self):
        stock, sector = block_prices()
        original = stock.copy(), sector.copy()
        self.assertEqual(k.measure(stock, sector), k.measure(stock * 8., sector / 4.))
        np.testing.assert_array_equal(stock, original[0])
        np.testing.assert_array_equal(sector, original[1])


class BuilderTests(unittest.TestCase):
    def test_window_dates_monthends_counts_and_raw_first_row(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar, mapper)
        self.assertTrue(values['available'][:, :2].all())
        self.assertFalse(values['available'][:, 2].any())
        self.assertEqual(audit['relative_sign_observations'], 72)
        self.assertEqual(audit['absolute_sign_observations'], 72)
        self.assertEqual(audit['available_cells'], 6)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertEqual(audit['denominator'], 16)
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


class IndependentCrossTests(unittest.TestCase):
    def test_both_fixtures_full_audit_prefix_and_shared_failure_cross(self):
        import monthly_sector_recurrence192_independent as oracle
        import test_monthly_sector_recurrence192_independent as independent_fixture
        for factory, frozen_map in ((fixture, mapper), (independent_fixture.fixture, independent_fixture.mapper)):
            panel, meta, calendar = factory()
            values, audit = k.build_features(panel, meta, calendar, frozen_map)
            self.assertTrue(oracle.verify_features(panel, meta, calendar, frozen_map, values, audit)['passed'])
            for n in range(len(meta['dates']) + 1):
                actual, record = k.build_features(panel, meta, calendar, frozen_map, limit=n)
                expected, reference = oracle.independent_features(panel, meta, calendar, frozen_map, limit=n)
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[key])
                self.assertEqual(record, reference)
            first = calendar.index(meta['dates'][0]) - 294
            for mutation in ('missing_session', 'invalid_stock', 'invalid_benchmark'):
                altered = copy.deepcopy(panel)
                if mutation == 'missing_session':
                    altered['adj'] = altered['adj'].drop(pd.Timestamp(calendar[first + 11]))
                else:
                    symbol = meta['symbols'][0][0] if mutation == 'invalid_stock' else 'XLK'
                    altered['adj'].loc[calendar[first + 11], symbol] = np.nan
                actual, record = k.build_features(altered, meta, calendar, frozen_map)
                self.assertTrue(oracle.verify_features(altered, meta, calendar, frozen_map, actual, record)['passed'])


if __name__ == '__main__':
    unittest.main()
