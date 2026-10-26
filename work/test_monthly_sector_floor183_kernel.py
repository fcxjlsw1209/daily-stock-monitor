"""Synthetic sector-conditional source tests; no historical strategy inputs."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_floor183_kernel as k


def path_from_logs(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def paired_prices(up=70, down=70, flat=70, up_excess=.003, down_excess=.001, flat_excess=.02):
    states = np.r_[np.ones(up), -np.ones(down), np.zeros(flat)]
    sector_r = .01*states
    relative = np.where(states > 0., up_excess, np.where(states < 0., down_excess, flat_excess))
    return path_from_logs(sector_r+relative), path_from_logs(sector_r)


def fixture():
    dates = pd.bdate_range('2012-01-02', periods=360)
    sector = np.resize(np.array([100., 101., 100.]), len(dates))
    r = np.diff(np.log(sector)); state = np.sign(np.diff(sector))
    excess = np.where(state > 0., .002, np.where(state < 0., .001, .02))
    stock = 70.*path_from_logs(r+excess)
    frame = pd.DataFrame({'A': stock, 'C': stock*2., 'BABA': stock/3.,
                          'XLK': sector, 'XLC': sector*2., 'XLY': sector/2., 'FUND': stock}, index=dates)
    sectors = pd.DataFrame({'A': 'Technology', 'C': 'Communication Services', 'BABA': 'Technology'}, index=dates)
    names = ['A', 'C', 'BABA', 'FUND']
    meta = dict(dates=[str(dates[j].date()) for j in (252, 273, 294)],
                symbols=[names[:] for _ in range(3)], etfs=['FUND'])
    return {'adj': frame, 'sectors': sectors}, meta, dates.strftime('%Y-%m-%d').tolist()


def mapper(day, symbols, sectors, etfs):
    labels = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Real Estate': 'XLRE'}
    answer = {}
    for symbol in symbols:
        if symbol in etfs: answer[symbol] = None
        elif symbol == 'BABA': answer[symbol] = 'XLY'
        elif symbol in sectors and pd.Timestamp(day) in sectors.index:
            answer[symbol] = labels.get(sectors.at[pd.Timestamp(day), symbol])
        else: answer[symbol] = None
    return answer


class MeasureTests(unittest.TestCase):
    def test_two_means_and_flat_days_excluded(self):
        stock, sector = paired_prices()
        out = k.measure(stock, sector)
        self.assertTrue(out['available'])
        self.assertEqual((out['up_count'], out['down_count'], out['flat_count']), (70, 70, 70))
        self.assertAlmostEqual(out['mean_up'], .003, 14)
        self.assertAlmostEqual(out['mean_down'], .001, 14)
        self.assertAlmostEqual(out['candidate'], .001, 14)
        self.assertAlmostEqual(out['control'], .002, 14)
        self.assertEqual((out['conditional_mean_estimates'], out['pooled_mean_estimates']), (2, 1))
        self.assertEqual(len(out['excess_log_returns']), 210)
        self.assertEqual(len(out['sector_states']), 210)

    def test_control_is_frequency_weighted_not_balanced(self):
        out = k.measure(*paired_prices(up=120, down=40, flat=50, up_excess=.004, down_excess=-.002))
        self.assertAlmostEqual(out['candidate'], -.002, 14)
        self.assertAlmostEqual(out['control'], (120*.004-40*.002)/160., 14)
        self.assertNotAlmostEqual(out['control'], .001, 8)

    def test_exact40_group_minimum_shared_support(self):
        valid = k.measure(*paired_prices(up=40, down=40, flat=130))
        self.assertTrue(valid['available'])
        for up, down in ((39, 40), (40, 39), (0, 210), (210, 0), (0, 0)):
            out = k.measure(*paired_prices(up=up, down=down, flat=210-up-down))
            self.assertFalse(out['available']); self.assertEqual(out['reason'], 'insufficient_groups')
            self.assertTrue(np.isnan(out['candidate'])); self.assertTrue(np.isnan(out['control']))
            self.assertIsNone(out['mean_up']); self.assertIsNone(out['mean_down'])
            self.assertEqual((out['conditional_mean_estimates'], out['pooled_mean_estimates']), (0, 0))
            self.assertEqual(len(out['sector_states']), 210)

    def test_exact_price_signs_even_when_log_difference_rounds_to_zero(self):
        first = 1e100; last = np.nextafter(first, np.inf)
        self.assertEqual(math.log(first), math.log(last))
        p = np.resize([first, last], 211)
        out = k.measure(p, p)
        self.assertTrue(out['available']); self.assertEqual((out['up_count'], out['down_count']), (105, 105))
        self.assertEqual(out['candidate'], 0.); self.assertEqual(out['control'], 0.)

    def test_beta_relation_exposes_near_one_preference(self):
        sector_r = np.tile([.01, -.01], 105)
        alpha = .0003
        for beta in (-1., 0., .5, 1., 1.5, 2.):
            out = k.measure(path_from_logs(beta*sector_r+alpha), path_from_logs(sector_r))
            self.assertTrue(out['available'])
            self.assertAlmostEqual(out['candidate'], alpha-abs(beta-1.)*.01, 14)
            self.assertAlmostEqual(out['control'], alpha, 14)

    def test_identical_paths_zero_is_valid_no_variance_floor(self):
        _, sector = paired_prices()
        out = k.measure(sector, sector)
        self.assertTrue(out['available'])
        self.assertEqual(out['candidate'], 0.); self.assertEqual(out['control'], 0.)

    def test_scale_invariance_and_extreme_ratios(self):
        stock, sector = paired_prices(); reference = k.measure(stock, sector)
        for a, b in ((1e-200, 1e200), (1e200, 1e-200), (2.**100, 2.**-100)):
            out = k.measure(stock*a, sector*b)
            self.assertEqual(out['sector_states'], reference['sector_states'])
            self.assertAlmostEqual(out['candidate'], reference['candidate'], 12)
            self.assertAlmostEqual(out['control'], reference['control'], 12)
        sector = np.resize([1e-300, 1e300], 211)
        out = k.measure(sector[::-1], sector)
        self.assertTrue(out['available']); self.assertTrue(math.isfinite(out['candidate']))

    def test_all211_prices_required_and_source_priority(self):
        stock, sector = paired_prices()
        for pos in (0, 1, 70, 140, 209, 210):
            for bad in (0., -1., np.nan, np.inf, 'bad'):
                a, b = stock.astype(object), sector.astype(object); a[pos] = bad
                out = k.measure(a, b)
                self.assertFalse(out['available']); self.assertEqual(out['excess_log_returns'], [])
                a, b = stock.astype(object), sector.astype(object); b[pos] = bad
                out = k.measure(a, b)
                self.assertFalse(out['available']); self.assertEqual(out['sector_states'], [])
        stock = stock.astype(object); stock[0] = 'bad'; sector[0] = 0.
        self.assertEqual(k.measure(stock, sector)['reason'], 'nonnumeric_stock_price')

    def test_bad_shape_and_nonfinite_arithmetic_raise(self):
        for a, b in ((np.ones(210), np.ones(211)), (np.ones(211), np.ones((211, 1)))):
            with self.assertRaises(ValueError): k.measure(a, b)
        with patch.object(k.math, 'log', return_value=np.inf):
            with self.assertRaises(FloatingPointError): k.measure(*paired_prices())
        with patch.object(k.math, 'fsum', return_value=np.inf):
            with self.assertRaises(FloatingPointError): k.measure(*paired_prices())


class BuildTests(unittest.TestCase):
    def test_independent_full_audit_cross_on_both_fixtures(self):
        import monthly_sector_floor183_independent as independent
        from test_monthly_sector_floor183_independent import fixture as other_fixture, mapper as other_mapper
        for factory, callback in ((fixture, mapper), (other_fixture, other_mapper)):
            for mode in ('normal', 'missing', 'invalid', 'insufficient_groups'):
                p, m, c = factory()
                if mode == 'missing': p['adj'] = p['adj'].drop(pd.Timestamp(c[5]))
                elif mode == 'invalid': p['adj'].iloc[40, 0] = np.nan
                elif mode == 'insufficient_groups':
                    for symbol in ('XLK', 'XLC', 'XLY', 'XLRE', 'XLV'):
                        if symbol in p['adj']: p['adj'][symbol] = 1.
                values, audit = k.build_features(p, m, c, callback)
                self.assertTrue(independent.verify_features(p, m, c, callback, values, audit)['passed'])

    def test_window_boundaries_mapping_and_estimate_counts(self):
        p, m, c = fixture(); values, audit = k.build_features(p, m, c, mapper)
        self.assertEqual(values['available'].shape, (3, 4))
        np.testing.assert_array_equal(values['available'], np.tile([True, True, True, False], (3, 1)))
        self.assertEqual(audit['conditional_mean_estimates'], 18)
        self.assertEqual(audit['pooled_mean_estimates'], 9)
        row = audit['event_history'][0]
        self.assertEqual(row['price_start'], c[0]); self.assertEqual(row['price_end'], c[210])
        self.assertEqual(row['windows']['first_return_date'], c[1])
        self.assertEqual(row['windows']['last_return_date'], c[210])
        self.assertEqual(row['benchmarks'], {'A': 'XLK', 'C': 'XLC', 'BABA': 'XLY', 'FUND': None})
        json.dumps(audit, allow_nan=False)

    def test_mapping_uses_current_query_not_formation_or_future(self):
        p, m, c = fixture(); p['sectors'].loc[pd.Timestamp(c[252]), 'A'] = 'Communication Services'
        p['sectors'].loc[pd.Timestamp(c[253]):, 'A'] = 'Real Estate'
        seen = []
        def observed(day, symbols, sectors, etfs):
            self.assertLessEqual(sectors.index.max(), pd.Timestamp(day))
            seen.append(day)
            return mapper(day, symbols, sectors, etfs)
        values, audit = k.build_features(p, m, c, observed, 1)
        self.assertEqual(seen, [m['dates'][0]])
        self.assertEqual(audit['event_history'][0]['benchmarks']['A'], 'XLC')
        p['sectors'] = p['sectors'].drop(pd.Timestamp(c[252]))
        values, audit = k.build_features(p, m, c, mapper, 1)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_benchmark')
        self.assertTrue(values['available'][0, 2])  # Explicit inherited foreign mapping remains.

    def test_missing_xlc_or_xlre_has_no_proxy_substitution(self):
        for symbol, label, benchmark in (('C', 'Communication Services', 'XLC'), ('A', 'Real Estate', 'XLRE')):
            p, m, c = fixture(); p['sectors'][symbol] = label
            p['adj'] = p['adj'].drop(columns=[benchmark], errors='ignore')
            values, audit = k.build_features(p, m, c, mapper)
            j = m['symbols'][0].index(symbol)
            self.assertEqual(audit['event_history'][0]['cells'][j]['reason'], 'missing_benchmark_symbol')
            self.assertFalse(values['available'][:, j].any())

    def test_missing_session_stock_unknown_and_history_reasons(self):
        p, m, c = fixture(); p['adj'] = p['adj'].drop(pd.Timestamp(c[5]))
        _, audit = k.build_features(p, m, c, mapper)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_official_session')
        self.assertEqual(audit['event_history'][0]['cells'][-1]['reason'], 'etf_neutral')
        p, m, c = fixture(); p['adj'] = p['adj'].drop(columns=['A'])
        m['dates'][0] = c[251]; p['sectors']['C'] = 'Unknown'
        _, audit = k.build_features(p, m, c, mapper)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        self.assertEqual(audit['event_history'][1]['cells'][0]['reason'], 'missing_symbol')
        self.assertEqual(audit['event_history'][1]['cells'][1]['reason'], 'missing_benchmark')

    def test_recent_prices_not_read_or_required(self):
        p, m, c = fixture(); expected = k.build_features(p, m, c, mapper, 1)
        class Poison:
            def __float__(self): raise AssertionError('Recent or future price accessed')
        p['adj'] = p['adj'].astype(object); p['adj'].iloc[211:] = Poison()
        self.assertEqual(k.build_features(p, m, c, mapper, 1)[1], expected[1])
        p['adj'] = p['adj'].iloc[:211]
        self.assertEqual(k.build_features(p, m, c, mapper, 1)[1], expected[1])

    def test_prefixes_and_inputs_preserved(self):
        p, m, c = fixture(); before = {name: x.copy(deep=True) for name, x in p.items()}; mm = copy.deepcopy(m)
        full, audit = k.build_features(p, m, c, mapper)
        for n in range(4):
            value, detail = k.build_features(p, m, c, mapper, n)
            for key in full: np.testing.assert_array_equal(value[key], full[key][:n])
            self.assertEqual(detail['event_history'], audit['event_history'][:n])
        for name in p: pd.testing.assert_frame_equal(p[name], before[name])
        self.assertEqual(m, mm)

    def test_bad_mapping_and_input_contract(self):
        p, m, c = fixture()
        for callback in (None, lambda *args: {}, lambda *args: {s: 'XLK' for s in m['symbols'][0]}):
            with self.assertRaises((ValueError, TypeError)): k.build_features(p, m, c, callback)
        for limit in (-1, 4, True, 1.2):
            with self.assertRaises(ValueError): k.build_features(p, m, c, mapper, limit)
        for cc in (c[::-1], c+[c[-1]], []):
            with self.assertRaises(ValueError): k.build_features(p, m, cc, mapper)
        with self.assertRaises(ValueError): k.build_features({'adj': p['adj']}, m, c, mapper)


if __name__ == '__main__':
    unittest.main(verbosity=2)
