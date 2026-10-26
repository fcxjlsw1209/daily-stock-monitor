"""Synthetic-only tests for independent exact recurrence and frozen accounts."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_recurrence192_independent as audit


def mapper(day, names, sectors, etfs):
    helper = audit.frozen_helper()
    helper.mapping_constants()
    return {s: helper.benchmark_for(pd.Timestamp(day), s, sectors, etfs) for s in names}


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    bench = np.exp(.00015 * x + .01 * np.sin(x * .05))
    prices = pd.DataFrame({'A': bench * np.exp(.0003 * x),
        'C': bench * np.exp(.08 * np.sin(x / 130.)), 'BABA': bench * np.exp(-.0002 * x),
        'FUND': bench, 'XLK': bench, 'XLC': bench, 'XLY': bench}, index=index)
    names = ['A', 'C', 'BABA', 'FUND'][:width]
    sectors = pd.DataFrame('Technology', index=index, columns=names)
    if 'C' in sectors:
        sectors.C = 'Communication Services'
    if 'BABA' in sectors:
        sectors.BABA = 'Energy'  # Explicit foreign mapping has priority.
    dates = ['2016-08-17', '2016-08-31', '2016-09-30'][:count]
    return {'adj': prices, 'sectors': sectors}, dict(dates=dates, symbols=[names[:] for _ in dates],
        etfs=['FUND']), index.strftime('%Y-%m-%d').tolist()


def prices_from_levels(levels):
    return np.interp(np.arange(253), np.arange(0, 253, 21), np.asarray(levels, float))


class ScalarTests(unittest.TestCase):
    def test_twelve_blocks_counts_and_four_neutral_pseudoblocks(self):
        levels = [1, 2, 2, 1, 2, 4, 2, 1, 1, 2, 1, 2, 4]
        r = audit.measure_oracle(prices_from_levels(levels), np.ones(253))
        expected = [1, 0, -1, 1, 1, -1, -1, 0, 1, -1, 1, 1]
        self.assertEqual(r['relative_signs'], expected)
        self.assertEqual(r['absolute_signs'], expected)
        self.assertEqual((r['relative_positive_count'], r['relative_negative_count'], r['relative_tie_count']), (6, 4, 2))
        self.assertEqual(r['candidate'], 2 / 16)
        self.assertEqual(r['endpoint_indices'], list(range(0, 253, 21)))
        self.assertEqual((r['block_start_indices'][0], r['block_end_indices'][-1]), (0, 252))
        self.assertEqual(r['relative_sign_observations'], 12)

    def test_matched_relative_and_absolute_directions(self):
        own = prices_from_levels([2. ** k for k in range(13)])
        sector = prices_from_levels([4. ** k for k in range(13)])
        r = audit.measure_oracle(own, sector)
        self.assertEqual((r['candidate'], r['control']), (-.75, .75))
        r = audit.measure_oracle(own[::-1], np.ones(253))
        self.assertEqual((r['candidate'], r['control']), (-.75, -.75))

    def test_ties_are_valid_zero_without_minimum_count(self):
        r = audit.measure_oracle(np.ones(253), np.ones(253))
        self.assertTrue(r['available'])
        self.assertEqual((r['candidate'], r['control']), (0., 0.))
        self.assertEqual((r['relative_tie_count'], r['absolute_tie_count']), (12, 12))
        self.assertEqual(r['absolute_sign_observations'], 12)
        own = prices_from_levels([2. ** k for k in range(13)])
        r = audit.measure_oracle(own, own)
        self.assertEqual((r['candidate'], r['control']), (0., .75))

    def test_exact_rational_boundary_and_extreme_prices(self):
        lo = 1e300; hi = float(np.nextafter(lo, math.inf))
        self.assertEqual(math.log(lo), math.log(hi))
        self.assertEqual(audit.relative_sign(hi, lo, lo, lo), 1)
        self.assertEqual(audit.relative_sign(lo, lo, hi, lo), -1)
        self.assertEqual(audit.relative_sign(hi, lo, hi, lo), 0)
        self.assertEqual(audit.relative_sign(5e-324, 1e308, 5e-324, 1e308), 0)
        own = np.ones(253)
        own[::21] = [5e-324 if k % 2 else 1e308 for k in range(13)]
        r = audit.measure_oracle(own, np.ones(253))
        self.assertTrue(r['available']); self.assertEqual(r['candidate'], 0.)
        for x in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                audit.relative_sign(x, 1, 1, 1)

    def test_all253_prices_required_and_failure_priority(self):
        own = np.ones(253); sector = np.ones(253)
        for at in (0, 1, 20, 21, 251, 252):
            for bad in (0., -1., np.nan, np.inf):
                s = own.copy(); s[at] = bad
                r = audit.measure_oracle(s, sector)
                self.assertEqual(r['reason'], 'invalid_stock_price')
                self.assertEqual(r['endpoint_indices'], [])
                self.assertEqual(r['relative_sign_observations'], 0)
        s = own.astype(object); b = sector.astype(object)
        s[10] = 0.; b[10] = 'bad'
        self.assertEqual(audit.measure_oracle(s, b)['reason'], 'invalid_stock_price')
        s[10] = 'bad'
        self.assertEqual(audit.measure_oracle(s, b)['reason'], 'nonnumeric_stock_price')
        s[10] = 1.
        self.assertEqual(audit.measure_oracle(s, b)['reason'], 'nonnumeric_benchmark_price')
        b[10] = -1.
        self.assertEqual(audit.measure_oracle(s, b)['reason'], 'invalid_benchmark_price')
        with self.assertRaises(ValueError):
            audit.measure_oracle(own[:-1], sector)

    def test_independent_power_of_two_scales_and_immutability(self):
        own = prices_from_levels([1, 2, 1, 4, 8, 8, 4, 2, 4, 8, 16, 8, 16])
        sector = prices_from_levels([2. ** (k // 2) for k in range(13)])
        original = own.copy(), sector.copy(); expected = audit.measure_oracle(own, sector)
        for scale in (2. ** -500, 2. ** 500):
            audit.record_check(audit.measure_oracle(own * scale, sector / scale), expected)
        np.testing.assert_array_equal(own, original[0]); np.testing.assert_array_equal(sector, original[1])


class SourceTests(unittest.TestCase):
    def test_mapping_windows_all_cells_and_exact_receipt(self):
        p, m, c = fixture(); before = copy.deepcopy(p)
        v, r = audit.independent_features(p, m, c, mapper)
        z = audit.verify_features(p, m, c, mapper, v, r)
        self.assertEqual((z['available_cells'], z['unavailable_cells']), (9, 3))
        self.assertEqual(z['relative_sign_observations'], 108)
        self.assertEqual(z['absolute_sign_observations'], 108)
        for row in r['event_history']:
            t = c.index(row['date'])
            self.assertEqual((row['price_start'], row['price_end']), (c[t - 294], c[t - 42]))
            self.assertEqual(row['benchmarks'], dict(A='XLK', C='XLC', BABA='XLY', FUND=None))
            for cell in row['cells'][:-1]:
                self.assertEqual(cell['endpoint_dates'], [c[t - 294 + k] for k in range(0, 253, 21)])
                self.assertEqual(cell['block_end_dates'][-1], c[t - 42])
            self.assertEqual(row['cells'][-1]['reason'], 'etf_neutral')
        for key in p:
            pd.testing.assert_frame_equal(p[key], before[key])

    def test_query_date_mapping_fixed_and_no_future_callback(self):
        p, m, c = fixture(count=1); day = m['dates'][0]
        p['sectors'].loc[:, 'A'] = 'Energy'; p['sectors'].loc[day, 'A'] = 'Technology'
        def checked(d, names, sectors, etfs):
            self.assertEqual(sectors.index.max(), pd.Timestamp(d))
            return mapper(d, names, sectors, etfs)
        v, r = audit.independent_features(p, m, c, checked)
        self.assertTrue(v['available'][0, 0]); self.assertEqual(r['event_history'][0]['benchmarks']['A'], 'XLK')
        with self.assertRaises(ValueError):
            audit.independent_features(p, m, c, lambda *args: {s: 'XLY' for s in args[1]})

    def test_missing_support_unknown_and_no_sector_fallback(self):
        p, m, c = fixture(count=1); t = c.index(m['dates'][0])
        for mode, reason in [('stock', 'missing_symbol'), ('sector', 'missing_benchmark_symbol'),
                ('session', 'missing_official_session'), ('unknown', 'missing_benchmark')]:
            q = copy.deepcopy(p)
            if mode == 'stock': q['adj'] = q['adj'].drop(columns='A')
            elif mode == 'sector': q['adj'] = q['adj'].drop(columns='XLK')
            elif mode == 'session': q['adj'] = q['adj'].drop(pd.Timestamp(c[t - 293]))
            else: q['sectors'].loc[m['dates'][0], 'A'] = 'Unknown'
            v, r = audit.independent_features(q, m, c, mapper)
            self.assertFalse(v['available'][0, 0]); self.assertTrue(np.isnan(v['candidate'][0, 0]))
            self.assertEqual(r['event_history'][0]['cells'][0]['reason'], reason)
        short = [x for x in c if x >= '2016-01-01']
        v, r = audit.independent_features(p, m, short, mapper)
        self.assertEqual(r['event_history'][0]['cells'][0]['reason'], 'insufficient_history')

    def test_all_prefixes_tail_poison_and_truncated_frames(self):
        p, m, c = fixture(); v, r = audit.independent_features(p, m, c, mapper)
        for n in range(4):
            q = copy.deepcopy(p); q['adj'] = q['adj'].astype(object)
            last = -1 if n == 0 else c.index(m['dates'][n - 1]) - 42
            q['adj'].iloc[last + 1:, :] = 'unread price'
            stop = pd.Timestamp.min if n == 0 else pd.Timestamp(m['dates'][n - 1])
            q['sectors'].loc[q['sectors'].index > stop, :] = 'unread sector'
            vv, rr = audit.independent_features(q, m, c, mapper, limit=n)
            for key in v:
                np.testing.assert_array_equal(vv[key], v[key][:n])
            self.assertEqual(rr['event_history'], r['event_history'][:n])
            qq = {key: frame.loc[:stop] for key, frame in q.items()}
            m2 = dict(m, dates=m['dates'][:n], symbols=m['symbols'][:n])
            # Empty sliced metadata has no retained width; use original metadata+limit0.
            v2, r2 = audit.independent_features(qq, m if n == 0 else m2, c, mapper, limit=0 if n == 0 else None)
            for key in vv:
                np.testing.assert_array_equal(v2[key], vv[key])
            self.assertEqual(r2, rr)

    def test_monthend_requires_next_official_session_and_only_initial_exception(self):
        p, m, c = fixture(count=1)
        audit.independent_features(p, m, c, mapper)
        for day in ('2016-08-18', '2016-08-30'):
            mm = dict(m, dates=[day])
            with self.assertRaisesRegex(ValueError, 'month-end'):
                audit.independent_features(p, mm, c, mapper)
        mm = dict(m, dates=['2016-08-31'])
        with self.assertRaisesRegex(ValueError, 'month-end'):
            audit.independent_features(p, mm, c[:c.index('2016-08-31') + 1], mapper)
        mm = dict(m, dates=['2016-07-29', '2016-08-17'], symbols=m['symbols'] * 2)
        with self.assertRaisesRegex(ValueError, 'month-end'):
            audit.independent_features(p, mm, c, mapper)

    def test_audit_sign_count_date_and_raw_corruption_rejected_exactly(self):
        p, m, c = fixture(count=1); v, r = audit.independent_features(p, m, c, mapper)
        for field, value in [('relative_signs', [0] * 12), ('relative_signed_count', 11),
                ('endpoint_dates', ['2000-01-01'] * 13), ('relative_sign_observations', 11)]:
            wrong = copy.deepcopy(r); wrong['event_history'][0]['cells'][0][field] = value
            with self.assertRaises(ValueError): audit.verify_features(p, m, c, mapper, v, wrong)
        wrong = copy.deepcopy(v); wrong['candidate'][0, 0] = np.nextafter(v['candidate'][0, 0], 0.)
        with self.assertRaises(ValueError): audit.verify_features(p, m, c, mapper, wrong, r)

    def test_full49_domain_no_eligibility_filter_and_empty_prefix(self):
        p, m, c = fixture(count=1); names = ['S' + str(i) for i in range(48)] + ['FUND']
        p['adj'] = pd.DataFrame({s: p['adj'].A for s in names}, index=p['adj'].index).assign(XLK=p['adj'].XLK)
        p['sectors'] = pd.DataFrame('Technology', index=p['sectors'].index, columns=names)
        m['symbols'] = [names]
        v, r = audit.independent_features(p, m, c, mapper)
        self.assertEqual(v['available'].shape, (1, 49)); self.assertEqual(r['available_cells'], 48)
        z, rr = audit.independent_features(p, m, c, mapper, limit=0)
        self.assertEqual(z['available'].shape, (0, 49)); self.assertEqual(rr['relative_sign_observations'], 0)
        z, rr = audit.independent_features(p, dict(m, dates=[], symbols=[]), c, mapper)
        self.assertEqual(z['available'].shape, (0, 0))

    def test_schema_failures_and_frozen_mapping_hash(self):
        p, m, c = fixture()
        for limit in (True, -1, 4, .5):
            with self.assertRaises(ValueError): audit.independent_features(p, m, c, mapper, limit=limit)
        with self.assertRaises(ValueError): audit.independent_features(p, m, c[::-1], mapper)
        mm = copy.deepcopy(m); mm['symbols'][0][1] = 'A'
        with self.assertRaises(ValueError): audit.independent_features(p, mm, c, mapper)
        with patch.object(audit, 'FROZEN_HELPER_SHA', 'bad'), self.assertRaises(ValueError):
            audit.independent_features(p, m, c, mapper)


class DeltaTests(unittest.TestCase):
    def test_linear_initial_missing_extremes_and_exact_validation(self):
        v = dict(candidate=np.array([[.75, -.75], [.0625, np.nan], [-.75, 0.]]),
            control=np.array([[.5, -.5], [-.0625, np.nan], [.75, 0.]]),
            available=np.array([[1, 1], [1, 0], [1, 1]], bool))
        d = audit.independent_deltas(v); r = audit.verify_deltas(v, d)
        self.assertEqual(d[audit.ARMS[1]][1, 0], .025 / 16)
        self.assertEqual(d[audit.ARMS[1]][2, 0], -.025 * .75)
        np.testing.assert_array_equal(d[audit.ARMS[1]][0], [0., 0.])
        self.assertEqual(d[audit.ARMS[1]][1, 1], 0.)
        self.assertEqual(r['max_absolute_delta'], .01875)
        for dose in (.01, .05, True):
            with self.assertRaises(ValueError): audit.independent_deltas(v, dose=dose)
        for index in ((0, 0), (1, 1), (2, 0)):
            wrong = copy.deepcopy(d); wrong[audit.ARMS[1]][index] += 1e-12
            with self.assertRaises(ValueError): audit.verify_deltas(v, wrong)
        for raw in (.8, .1, np.inf):
            wrong = copy.deepcopy(v); wrong['candidate'][1, 0] = raw
            with self.assertRaises(ValueError): audit.independent_deltas(wrong)


class OutcomeTests(unittest.TestCase):
    def test_three_arms_nine_paths_45_slices_and_accounting_pin(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper, 'audit', audit): args = helper.account_fixture()
        r = audit.audit_outcomes(*args)
        self.assertEqual((r['checked_paths'], r['checked_slices']), (9, 45))
        self.assertEqual(set(r['paths']), set(audit.ARMS))
        wrong = copy.deepcopy(args); wrong[4][audit.ARMS[1]]['same_close_zero']['metrics']['cagr'] += .01
        with self.assertRaises((ValueError, AssertionError)): audit.audit_outcomes(*wrong)
        with patch.object(audit, 'FROZEN_HELPER_SHA', 'bad'), self.assertRaises(ValueError): audit.accounting()


if __name__ == '__main__':
    unittest.main(verbosity=2)
