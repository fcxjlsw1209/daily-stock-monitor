"""Synthetic-only tests. No historical panel, fitting outcomes, or replay."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_breakpoint177_kernel as k


def current():
    return 100. * np.cumprod(np.r_[1., 1. + .001 + .01*np.sin(np.arange(63))])


def path(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def fixture():
    dates = pd.bdate_range('2010-01-04', periods=365)
    n = np.arange(len(dates))
    frame = pd.DataFrame({'A': 100*np.exp(.001*n+.03*np.sin(n*.19)),
                          'B': 80*np.exp(.002*n+.04*np.cos(n*.17))}, index=dates)
    meta = {'dates': [str(dates[i].date()) for i in (294, 320, 341)],
            'symbols': [['A', 'B']] * 3}
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


class ScalarTests(unittest.TestCase):
    def test_shape(self):
        for a, b in [(np.ones(252), current()), (np.ones(253), current()[:-1])]:
            with self.assertRaises(ValueError):
                k.measure(a, b)

    def test_constant_zero_sse_exact_baseline(self):
        r = k.measure(np.ones(253), current())
        self.assertTrue(r['available'])
        self.assertEqual(r['sse0'], 0.)
        self.assertTrue(r['numerically_constant'])
        self.assertIsNone(r['candidate_fit']['statistic'])
        self.assertEqual(r['candidate_fit']['selection_rule'], 'canonical_inactive_k63')
        self.assertEqual(r['candidate'], r['baseline'])
        self.assertEqual(r['control'], r['baseline'])
        self.assertEqual(r['candidate_fit']['k'], 63)
        self.assertEqual(r['split_mean_estimates'], 127)

    def test_near_constant_geometric_noise_is_not_a_break(self):
        p = np.exp(.002*np.arange(253))
        r = k.measure(p, current())
        self.assertTrue(r['numerically_constant'])
        self.assertEqual(r['candidate_fit']['k'], 63)
        self.assertIsNone(r['candidate_fit']['statistic'])
        self.assertEqual(r['candidate_breaks'], 0)
        self.assertEqual(r['candidate'], r['baseline'])

    def test_perfect_midpoint_break(self):
        p = np.exp2(np.r_[np.arange(127), np.full(126, 126)])
        r = k.measure(p, current())
        self.assertEqual(r['candidate_fit']['k'], 126)
        self.assertTrue(r['candidate_fit']['perfect_fit'])
        self.assertIsNone(r['candidate_fit']['statistic'])
        self.assertTrue(r['candidate_fit']['accepted'])
        self.assertEqual(r['candidate'], 0.)
        self.assertEqual(r['control'], 0.)

    def test_unknown_location_differs_from_fixed_control(self):
        p = np.exp2(np.r_[np.arange(85), np.full(168, 84)])
        r = k.measure(p, current())
        self.assertEqual(r['candidate_fit']['k'], 84)
        self.assertEqual(r['control_fit']['k'], 126)
        self.assertEqual(r['candidate_fit']['sse'], 0.)
        self.assertGreater(r['control_fit']['sse'], 0.)
        self.assertEqual(r['constant_mean_estimates'], 1)
        self.assertEqual(r['split_mean_estimates'], 127)

    def test_equal_sse_selects_earliest(self):
        r = k._best([{'k': 126, 'sse': 1.}, {'k': 64, 'sse': 1.}, {'k': 63, 'sse': 2.}])
        self.assertEqual(r['k'], 64)

    def test_relative_sse_tie_uses_earliest_selected_own_sse(self):
        rows = [dict(k=63, sse=1.+5e-13, pre_mean=0., post_mean=.001),
                dict(k=189, sse=1., pre_mean=.001, post_mean=.002)]
        chosen = k._best(rows, 2.)
        self.assertEqual(chosen['k'], 63)
        fit = k._fit(chosen, 2., .001, .1, False)
        self.assertEqual(fit['statistic'], 252*(math.log(2.)-math.log(rows[0]['sse'])))
        rows[0]['sse'] = 1.+5e-12
        self.assertEqual(k._best(rows, 2.)['k'], 189)

    def test_gate_strict(self):
        split = dict(k=126, pre_mean=.001, post_mean=.002, sse=1.)
        actual = 252*(math.log(1.1)-math.log(1.))
        with patch.object(k, 'PENALTY', actual):
            self.assertFalse(k._fit(split, 1.1, .0015, .2, False)['accepted'])
        with patch.object(k, 'PENALTY', np.nextafter(actual, -math.inf)):
            self.assertTrue(k._fit(split, 1.1, .0015, .2, False)['accepted'])

    def test_baseline_matches_direct_formula(self):
        p = path(.001+.003*np.sin(np.arange(252)))
        c = current()
        expected = (p[-1]/p[0]-1)/np.std(c[1:]/c[:-1]-1, ddof=1)
        r = k.measure(p, c)
        self.assertEqual(r['baseline'], expected)
        self.assertEqual(r['candidate'], expected)
        self.assertEqual(r['control'], expected)

    def test_interior_missing_preserves_baseline(self):
        for bad, reason in [(np.nan, 'invalid_formation_price'), ('bad', 'nonnumeric_formation_price')]:
            p = np.asarray(path(np.full(252, .001)), dtype=object)
            p[150] = bad
            r = k.measure(p, current())
            self.assertTrue(r['baseline_available'])
            self.assertFalse(r['available'])
            self.assertEqual(r['reason'], reason)
            self.assertEqual(r['split_mean_estimates'], 0)

    def test_endpoints_required(self):
        for index in (0, -1):
            p = np.ones(253)
            p[index] = np.nan
            r = k.measure(p, current())
            self.assertFalse(r['baseline_available'])
            self.assertEqual(r['reason'], 'invalid_endpoint_price')

    def test_bad_current_is_first_failure(self):
        p = np.full(253, 'bad', dtype=object)
        c = np.asarray(current(), dtype=object)
        c[0] = 'bad'
        self.assertEqual(k.measure(p, c)['reason'], 'nonnumeric_current_price')
        self.assertEqual(k.measure(p, np.zeros(64))['reason'], 'invalid_current_price')

    def test_constant_current_vol(self):
        r = k.measure(np.ones(253), np.ones(64))
        self.assertFalse(r['baseline_available'])
        self.assertEqual(r['reason'], 'zero_current_volatility')

    def test_tiny_current_vol_preserves_old_domain(self):
        c = 1.+1e-14*np.sin(np.arange(64))
        r = k.measure(path(np.full(252, .001)), c)
        self.assertTrue(r['baseline_available'])
        self.assertFalse(r['available'])
        self.assertEqual(r['reason'], 'volatility_floor')

    def test_nonfinite_current_simple_returns(self):
        c = np.ones(64)
        c[1:3] = [1e-300, 1e300]
        self.assertEqual(k.measure(np.ones(253), c)['reason'], 'nonfinite_current_return')

    def test_forecast_overflow_common_mask_but_counts_retained(self):
        p = np.exp2(np.r_[np.zeros(190), 5*np.arange(1, 64)])
        r = k.measure(p, current())
        self.assertTrue(r['baseline_available'])
        self.assertEqual(r['reason'], 'nonfinite_forecast')
        self.assertTrue(r['candidate_fit']['accepted'])
        self.assertIsNone(r['candidate_fit']['forecast_growth'])
        self.assertFalse(r['available'])
        self.assertTrue(math.isnan(r['candidate']))
        self.assertTrue(math.isnan(r['control']))
        self.assertEqual(r['split_mean_estimates'], 127)

    def test_scale_invariance(self):
        p = path(np.r_[np.full(100, .001), np.full(152, .004)])
        r = k.measure(p, current())
        for scale in (1e-100, 1e100):
            scaled = k.measure(p*scale, current()*scale)
            for name in ('baseline', 'candidate', 'control'):
                self.assertAlmostEqual(scaled[name]/r[name], 1., places=10)
            self.assertEqual(scaled['candidate_fit']['k'], r['candidate_fit']['k'])

    def test_source_not_mutated(self):
        p, c = np.ones(253), current()
        a, b = p.copy(), c.copy()
        k.measure(p, c)
        np.testing.assert_array_equal(p, a)
        np.testing.assert_array_equal(c, b)


class BuilderTests(unittest.TestCase):
    def test_offsets_counts_and_json(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar)
        self.assertEqual(set(values), {'baseline', 'candidate', 'control', 'available'})
        self.assertEqual(values['baseline'].shape, (3, 2))
        self.assertEqual(audit['constant_mean_estimates'], 6)
        self.assertEqual(audit['split_mean_estimates'], 6*127)
        e = audit['event_history'][0]
        self.assertEqual(e['formation_start'], calendar[0])
        self.assertEqual(e['formation_end'], calendar[252])
        self.assertEqual(e['volatility_start'], calendar[231])
        self.assertEqual(e['volatility_end'], calendar[294])
        self.assertEqual(e['cells'][0]['splits'][0]['break_after_date'], calendar[63])
        self.assertEqual(e['cells'][0]['splits'][0]['post_first_return_date'], calendar[64])
        json.dumps(audit, allow_nan=False)

    def test_prefix_and_future_poison(self):
        class Poison:
            def __float__(self):
                raise AssertionError('Future numeric conversion')
        panel, meta, calendar = fixture()
        full, _ = k.build_features(panel, meta, calendar)
        for limit in (0, 1, 2, 3):
            prefix, _ = k.build_features(panel, meta, calendar, limit=limit)
            for key in full:
                np.testing.assert_array_equal(prefix[key], full[key][:limit])
        poisoned = panel['adj'].astype(object)
        poisoned.loc[poisoned.index[295]:] = Poison()
        one, _ = k.build_features({'adj': poisoned}, meta, calendar, limit=1)
        for key in full:
            np.testing.assert_array_equal(one[key], full[key][:1])

    def test_early_event_and_unknown(self):
        panel, meta, calendar = fixture()
        meta = {'dates': [calendar[250], calendar[320]], 'symbols': [['A', 'Z']]*2}
        values, audit = k.build_features(panel, meta, calendar)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        self.assertEqual(audit['event_history'][1]['cells'][1]['reason'], 'missing_symbol')
        self.assertFalse(values['available'][0].any())

    def test_missing_interior_session_retains_baseline(self):
        panel, meta, calendar = fixture()
        expected, _ = k.build_features(panel, meta, calendar, limit=1)
        panel = {'adj': panel['adj'].drop(panel['adj'].index[100])}
        values, audit = k.build_features(panel, meta, calendar, limit=1)
        np.testing.assert_array_equal(values['baseline'], expected['baseline'])
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_formation_session')

    def test_missing_endpoint_and_current(self):
        for index, reason in [(0, 'missing_endpoint_session'), (290, 'missing_volatility_session')]:
            panel, meta, calendar = fixture()
            panel = {'adj': panel['adj'].drop(panel['adj'].index[index])}
            values, audit = k.build_features(panel, meta, calendar, limit=1)
            self.assertFalse(np.isfinite(values['baseline']).any())
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], reason)

    def test_calendar_validation(self):
        panel, meta, calendar = fixture()
        for bad in (calendar[::-1], calendar+[calendar[-1]], calendar[:2]+['2010-01-09']):
            with self.assertRaises(ValueError):
                k.build_features(panel, meta, bad)

    def test_identity_and_limits(self):
        panel, meta, calendar = fixture()
        bad = copy.deepcopy(meta)
        bad['symbols'][0] = ['A', 'A']
        with self.assertRaises(ValueError):
            k.build_features(panel, bad, calendar)
        for limit in (-1, 4, True, 1.1):
            with self.assertRaises(ValueError):
                k.build_features(panel, meta, calendar, limit=limit)

    def test_no_panel_mutation(self):
        panel, meta, calendar = fixture()
        original = panel['adj'].copy(deep=True)
        original_meta = copy.deepcopy(meta)
        k.build_features(panel, meta, calendar)
        pd.testing.assert_frame_equal(panel['adj'], original)
        self.assertEqual(meta, original_meta)


class IndependentCrossTests(unittest.TestCase):
    def test_symmetric_nondegenerate_minimum_sse_ties(self):
        import monthly_breakpoint177_independent as independent
        # Two equally plausible break sides; the exact min/tie rule must not
        # silently diverge between independent summation implementations.
        sources = [np.exp2(np.cumsum(np.r_[0., np.ones(63), np.zeros(126), np.ones(63)])),
                   path(np.r_[np.full(63, .01), np.full(126, -.01), np.full(63, .01)])]
        for p in sources:
            got, expected = k.measure(p, current()), independent.measure_oracle(p, current())
            self.assertFalse(got['numerically_constant'])
            self.assertEqual(got['candidate_fit']['k'], 63)
            with self.subTest(got_k=got['candidate_fit']['k'], expected_k=expected['candidate_fit']['k']):
                independent.record_check(got, expected)

    def test_scalar_extremes_match_independent(self):
        import monthly_breakpoint177_independent as independent
        paths = [np.ones(253), np.exp(.002*np.arange(253)),
                 np.exp2(np.r_[np.arange(127), np.full(126, 126)]),
                 np.exp2(np.r_[np.zeros(190), 5*np.arange(1, 64)])]
        for i, p in enumerate(paths):
            got, expected = k.measure(p, current()), independent.measure_oracle(p, current())
            with self.subTest(case=i, got=got['candidate_fit'], expected=expected['candidate_fit']):
                independent.record_check(got, expected)

    def test_both_independent_and_production_fixtures(self):
        import monthly_breakpoint177_independent as independent
        from test_monthly_breakpoint177_independent import fixture as other_fixture
        for panel, meta, calendar in (fixture(), other_fixture()):
            values, audit = k.build_features(panel, meta, calendar)
            self.assertTrue(independent.verify_features(panel, meta, calendar, values, audit)['passed'])

    def test_missing_interior_and_unknown_match_oracle(self):
        import monthly_breakpoint177_independent as independent
        panel, meta, calendar = fixture()
        panel = {'adj': panel['adj'].drop(index=panel['adj'].index[100], columns=['B'])}
        values, audit = k.build_features(panel, meta, calendar)
        self.assertTrue(independent.verify_features(panel, meta, calendar, values, audit)['passed'])


if __name__ == '__main__':
    unittest.main()
