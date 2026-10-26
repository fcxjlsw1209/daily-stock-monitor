"""Synthetic contracts for current63 projection-risk normalization only."""
from copy import deepcopy
from fractions import Fraction
import math
import unittest

import numpy as np
import pandas as pd
import monthly_idio_risk222_kernel as k


def prices(returns, start=64.):
    result = [float(start)]
    for r in returns:
        result.append(result[-1]*(1.+r))
    return np.asarray(result)


def fixture():
    idx = pd.bdate_range('2014-01-01', '2021-01-29')
    t = np.arange(len(idx), dtype=float)
    market = 100*np.exp(.0004*t+.025*np.sin(t/8))
    frame = pd.DataFrame({'A': 100*np.exp(.0007*t+.035*np.sin(t/9)),
        'B': 150*np.exp(.0002*t+.02*np.sin(t/6)),
        'E': 50*np.exp(.0003*t+.04*np.sin(t/13)),
        'SPY': market, 'FLAT': np.ones(len(idx))*100.}, index=idx)
    dates = ['2016-08-17', '2016-08-31', '2017-01-31', '2018-06-29', '2019-12-31']
    meta = {'dates': dates, 'symbols': [['A', 'B', 'E', 'SPY', 'FLAT'] for _ in dates],
            'etfs': ['E', 'SPY']}
    return {'adj': frame}, meta, list(idx.strftime('%Y-%m-%d'))


def single_query():
    panel, meta, cal = fixture()
    meta = {**meta, 'dates': meta['dates'][-1:], 'symbols': meta['symbols'][-1:]}
    return panel, meta, cal


def arithmetic_inputs():
    t = np.arange(63, dtype=float)
    stock = prices(.004+.011*np.sin(t/3)+.006*np.cos(t/5))
    market = prices(.002+.009*np.sin(t/4))
    formation = np.linspace(64., 128., 253)
    return formation, stock, market


class ArithmeticTests(unittest.TestCase):
    def test_hand_computed_projection_against_rational_returns(self):
        sr = [1., -.5, 0.]*21; mr = [0., 1., -.5]*21
        f = np.full(253, 64.); f[-1] = 128.
        result = k.measure(f, prices(sr), prices(mr))
        sf, mf = [Fraction(x) for x in sr], [Fraction(x) for x in mr]
        sy, mx = sum(sf)/63, sum(mf)/63
        y, x = [r-sy for r in sf], [r-mx for r in mf]
        yy, xx = sum(v*v for v in y), sum(v*v for v in x)
        xy = sum(a*b for a, b in zip(x, y)); beta = xy/xx
        rss = sum((b-beta*a)**2 for a, b in zip(x, y))
        self.assertTrue(result['available']); self.assertEqual(result['momentum'], 1.)
        self.assertAlmostEqual(result['beta'], float(beta), places=15)
        self.assertAlmostEqual(result['stock_centered_sum_squares'], float(yy), places=13)
        self.assertAlmostEqual(result['residual_sum_squares'], float(rss), places=13)
        self.assertAlmostEqual(result['idiosyncratic_risk'], math.sqrt(float(rss)/62), places=15)
        self.assertAlmostEqual(result['total_risk'], math.sqrt(float(yy)/62), places=15)
        self.assertEqual(result['raw_candidate'], 1./result['idiosyncratic_risk'])

    def test_identical_market_self_projection_is_exact_zero(self):
        f, stock, _ = arithmetic_inputs()
        result = k.measure(f, stock, stock.copy())
        self.assertTrue(result['source_available']); self.assertFalse(result['available'])
        self.assertEqual(result['beta'], 1.); self.assertEqual(result['residual_sum_squares'], 0.)
        self.assertEqual(result['idiosyncratic_risk'], 0.)
        self.assertEqual(result['reason'], 'zero_idiosyncratic_risk')
        self.assertEqual(result['descriptive_beta_fit'], 1)
        self.assertTrue(np.isnan(result['raw_control']))

    def test_flat_market_preserves_total_risk_but_no_fit(self):
        f, stock, _ = arithmetic_inputs()
        result = k.measure(f, stock, np.full(64, 64.))
        self.assertTrue(result['source_available']); self.assertFalse(result['available'])
        self.assertGreater(result['total_risk'], 0.)
        self.assertEqual(result['market_centered_sum_squares'], 0.)
        self.assertEqual(result['descriptive_beta_fit'], 0)
        self.assertEqual(result['reason'], 'zero_market_variance')
        for key in ('beta', 'idiosyncratic_risk', 'raw_candidate', 'raw_control'):
            self.assertTrue(np.isnan(result[key]))

    def test_flat_stock_fits_beta_zero_and_common_missing(self):
        f, _, market = arithmetic_inputs()
        result = k.measure(f, np.full(64, 64.), market)
        self.assertEqual(result['descriptive_beta_fit'], 1)
        for key in ('beta', 'total_risk', 'idiosyncratic_risk', 'residual_sum_squares'):
            self.assertEqual(result[key], 0.)
        self.assertEqual(result['reason'], 'zero_total_risk')
        self.assertFalse(result['available'])

    def test_zero_and_negative_momentum_are_valid(self):
        f, stock, market = arithmetic_inputs()
        for ending in (64., 32.):
            f[-1] = ending; result = k.measure(f, stock, market)
            self.assertTrue(result['available'])
            self.assertEqual(result['momentum'], ending/64.-1.)
            self.assertEqual(result['raw_candidate'], result['momentum']/result['idiosyncratic_risk'])
        self.assertLess(result['raw_candidate'], result['raw_control'])

    def test_rss_not_second_centered_even_with_roundoff_mean(self):
        f, stock, market = arithmetic_inputs(); result = k.measure(f, stock, market)
        sr = [float(stock[i+1])/float(stock[i])-1. for i in range(63)]
        mr = [float(market[i+1])/float(market[i])-1. for i in range(63)]
        y = [r-math.fsum(sr)/63 for r in sr]
        x = [r-math.fsum(mr)/63 for r in mr]
        e = [b-result['beta']*a for a, b in zip(x, y)]
        self.assertNotEqual(math.fsum(e)/63, 0.)
        self.assertEqual(result['residual_sum_squares'], math.fsum(v*v for v in e))
        self.assertEqual(result['idiosyncratic_risk'], math.sqrt(math.fsum(v*v for v in e)/62))

    def test_current_risk_changes_without_changing_formation_numerator(self):
        f, stock, market = arithmetic_inputs(); before = k.measure(f, stock, market)
        stock[-1] *= 1.03; after = k.measure(f, stock, market)
        self.assertEqual(before['momentum'], after['momentum'])
        self.assertNotEqual(before['total_risk'], after['total_risk'])
        self.assertNotEqual(before['idiosyncratic_risk'], after['idiosyncratic_risk'])

    def test_no_epsilon_floor_near_market_projection(self):
        f, stock, _ = arithmetic_inputs(); perturbed = stock.copy()
        perturbed[31] = np.nextafter(perturbed[31], np.inf)
        result = k.measure(f, perturbed, stock)
        self.assertTrue(result['available']); self.assertGreater(result['idiosyncratic_risk'], 0.)
        self.assertLess(result['idiosyncratic_risk'], 1e-12)

    def test_binary_scale_invariance_and_no_mutation(self):
        args = arithmetic_inputs(); originals = [x.copy() for x in args]
        a = k.measure(*args); b = k.measure(args[0]*8., args[1]*4., args[2]*2.)
        for key in k.SCALAR_FIELDS:
            self.assertEqual(a[key], b[key])
        for original, actual in zip(originals, args):
            np.testing.assert_array_equal(original, actual)

    def test_exact_input_shapes(self):
        args = arithmetic_inputs()
        for position, lengths in ((0, (0, 252, 254)), (1, (0, 63, 65)), (2, (0, 63, 65))):
            for n in lengths:
                bad = list(args); bad[position] = np.ones(n)
                with self.assertRaises(ValueError):k.measure(*bad)

    def test_invalid_sources_missing_not_zero(self):
        for position, label in enumerate(('formation', 'stock_risk', 'market_risk')):
            for value in (0., -1., np.nan, np.inf, True, '64', None):
                args = list(arithmetic_inputs()); args[position] = list(args[position]); args[position][17] = value
                result = k.measure(*args)
                self.assertFalse(result['source_available']); self.assertFalse(result['available'])
                self.assertIn(label, result['reason']); self.assertEqual(result['descriptive_beta_fit'], 0)
                self.assertTrue(np.isnan(result['momentum']))

    def test_finite_source_overflow_is_hard_error(self):
        args = list(arithmetic_inputs()); args[0][0] = 1e-300; args[0][-1] = 1e300
        with self.assertRaises(FloatingPointError):k.measure(*args)
        args = list(arithmetic_inputs()); args[1][0] = 1e-150; args[1][1] = 1e100
        with self.assertRaises(FloatingPointError):k.measure(*args)

    def test_underflow_is_not_theoretical_zero_risk(self):
        with self.assertRaises(FloatingPointError):k._sum_squares([1e-200, -1e-200])
        with self.assertRaises(FloatingPointError):k._risk(float(np.nextafter(0., 1.)))
        self.assertEqual(k._sum_squares([0., -0.]), 0.)
        self.assertEqual(k._risk(0.), 0.)

    def test_original_percentiles_ties_and_singleton(self):
        q = k.percentile_rank(np.array([5., 5., 1., np.nan]), np.array([1, 1, 1, 0], bool))
        np.testing.assert_array_equal(q[:3], [2.5/3-.5, 2.5/3-.5, 1/3-.5])
        self.assertTrue(np.isnan(q[3])); self.assertEqual(k.percentile_rank(np.array([2.]), np.array([True]))[0], .5)


class WindowTests(unittest.TestCase):
    def test_axes_current63_clock_source_and_fit_counts(self):
        panel, meta, cal = fixture(); values, audit = k.build_features(panel, meta, cal)
        self.assertEqual(set(values), set(k.FEATURE_KEYS)); self.assertEqual(values['available'].shape, (5, 5))
        self.assertEqual(audit['source_available_cells'], 25); self.assertEqual(audit['available_cells'], 15)
        self.assertEqual(audit['descriptive_beta_fits'], 25)
        self.assertEqual(audit['descriptive_beta_fits_by_event'], [5]*5)
        self.assertEqual(audit['valid_price_observations'], 381*25)
        self.assertEqual(audit['simple_returns'], 126*25)
        self.assertEqual(audit['normalized_momentum_scores'], 30)
        self.assertTrue(values['available'][:, 2].all())
        self.assertFalse(values['available'][:, 3:].any())
        for row in audit['event_history']:
            t = cal.index(row['date'])
            self.assertEqual(row['formation_price_start'], cal[t-294])
            self.assertEqual(row['formation_price_end'], cal[t-42])
            self.assertEqual(row['risk_price_start'], cal[t-63])
            self.assertEqual(row['risk_price_end'], cal[t])
            self.assertEqual(row['first_risk_return_date'], cal[t-62])
            self.assertEqual(row['cells'][3]['beta'], 1.)
            self.assertIsNone(row['cells'][3]['candidate'])

    def test_current_query_price_is_consumed_but_future_is_not(self):
        panel, meta, cal = single_query(); before, audit = k.build_features(panel, meta, cal)
        changed = {'adj': panel['adj'].astype(object)}
        changed['adj'].loc[changed['adj'].index > meta['dates'][0], :] = 'future poison'
        after, other = k.build_features(changed, meta, cal)
        for key in before:np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(audit, other)
        changed['adj'].loc[meta['dates'][0], 'A'] *= 1.03
        later, _ = k.build_features(changed, meta, cal)
        self.assertEqual(before['momentum'][0, 0], later['momentum'][0, 0])
        self.assertNotEqual(before['total_risk'][0, 0], later['total_risk'][0, 0])

    def test_missing_stock_in_either_window_blocks_only_it(self):
        for offset in (200, 10):
            panel, meta, cal = single_query(); t = cal.index(meta['dates'][0])
            panel['adj'].loc[cal[t-offset], 'A'] = np.nan
            values, audit = k.build_features(panel, meta, cal)
            self.assertFalse(values['available'][0, 0]); self.assertTrue(values['available'][0, 1])
            self.assertFalse(audit['event_history'][0]['cells'][0]['source_available'])

    def test_market_current_window_missing_blocks_all(self):
        panel, meta, cal = single_query(); t = cal.index(meta['dates'][0])
        panel['adj'].loc[cal[t-10], 'SPY'] = np.nan
        values, audit = k.build_features(panel, meta, cal)
        self.assertFalse(values['available'].any()); self.assertEqual(audit['source_available_cells'], 0)

    def test_market_old_formation_not_required_for_other_stocks(self):
        panel, meta, cal = single_query(); t = cal.index(meta['dates'][0])
        panel['adj'].loc[cal[t-200], 'SPY'] = np.nan
        values, audit = k.build_features(panel, meta, cal)
        self.assertTrue(values['available'][0, :3].all()); self.assertFalse(values['available'][0, 3])
        self.assertEqual(audit['source_available_cells'], 4)

    def test_missing_official_sessions_never_shortens_either_window(self):
        for offset, reason in ((200, 'missing_formation_session'), (10, 'missing_risk_session')):
            panel, meta, cal = single_query(); t = cal.index(meta['dates'][0])
            panel['adj'] = panel['adj'].drop(pd.Timestamp(cal[t-offset]))
            values, audit = k.build_features(panel, meta, cal)
            self.assertFalse(values['available'].any()); self.assertEqual(audit['reason_counts'], {reason: 5})

    def test_market_column_required_but_unconsumed_columns_ignored(self):
        panel, meta, cal = fixture(); panel['adj']['UNUSED'] = 'poison'
        k.build_features(panel, meta, cal)
        panel['adj'] = panel['adj'].drop(columns='SPY')
        values, audit = k.build_features(panel, meta, cal)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reason_counts'], {'missing_market_symbol': 20, 'missing_symbol': 5})

    def test_zero_market_fits_count_separate_from_source(self):
        panel, meta, cal = fixture(); panel['adj']['SPY'] = 100.
        values, audit = k.build_features(panel, meta, cal)
        self.assertEqual(audit['source_available_cells'], 25); self.assertEqual(audit['available_cells'], 0)
        self.assertEqual(audit['descriptive_beta_fits'], 0)
        self.assertEqual(audit['descriptive_beta_fits_by_event'], [0]*5)
        self.assertTrue(np.isnan(values['candidate']).all())
        self.assertTrue(np.isfinite(values['momentum']).all())

    def test_all_physical_price_and_metadata_prefixes(self):
        panel, meta, cal = fixture(); values, audit = k.build_features(panel, meta, cal)
        for n in range(len(meta['dates'])+1):
            sm = {**meta, 'dates': meta['dates'][:n], 'symbols': meta['symbols'][:n]}
            frame = panel['adj'].loc[:meta['dates'][n-1]] if n else panel['adj'].iloc[:0]
            actual, aa = k.build_features({'adj': frame}, sm, cal)
            self.assertEqual(aa['event_history'], audit['event_history'][:n])
            for key in values:
                if n:np.testing.assert_array_equal(actual[key], values[key][:n])
                else:self.assertEqual(actual[key].shape, (0, 0))

    def test_metadata_limits_initial_partial_and_no_mutation(self):
        panel, meta, cal = fixture(); before = panel['adj'].copy(); original = deepcopy(meta)
        values, audit = k.build_features(panel, meta, cal, limit=2)
        self.assertEqual(values['available'].shape, (2, 5))
        pd.testing.assert_frame_equal(before, panel['adj']); self.assertEqual(meta, original)
        for n in (-1, True, 6):
            with self.assertRaises(ValueError):k.build_features(panel, meta, cal, limit=n)
        meta['dates'][0] = '2016-08-18'
        with self.assertRaises(ValueError):k.build_features(panel, meta, cal)

    def test_invalid_calendar_and_axes(self):
        panel, meta, cal = fixture()
        for bad in (cal[::-1], cal+[cal[-1]], ['2014-01-04']):
            with self.assertRaises(ValueError):k.build_features(panel, meta, bad)
        duplicate = deepcopy(meta); duplicate['symbols'][0][1] = 'A'
        with self.assertRaises(ValueError):k.build_features(panel, duplicate, cal)
        with self.assertRaises(ValueError):k.build_features({**panel, 'close': panel['adj']}, meta, cal)


if __name__ == '__main__':
    unittest.main(verbosity=2)
