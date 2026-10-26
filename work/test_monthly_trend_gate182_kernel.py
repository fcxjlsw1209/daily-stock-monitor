"""Only synthetic source paths and score components; no fitted inputs or returns."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_trend_gate182_kernel as k


def fixture():
    dates = pd.bdate_range('2011-01-03', periods=350)
    x = np.arange(len(dates))
    frame = pd.DataFrame({'A': 100.*np.exp(.002*x+.005*np.sin(.31*x)),
                          'ETF': 80.*np.exp(-.001*x+.03*np.cos(.27*x))}, index=dates)
    meta = dict(dates=[str(dates[i].date()) for i in (252, 280, 310)],
                symbols=[['A', 'ETF'] for _ in range(3)], etfs=['ETF'])
    return {'adj': frame}, meta, dates.strftime('%Y-%m-%d').tolist()


def route_fixture():
    upstream = np.tile([.2, .3, .1, np.nan], (3, 1))
    increment = np.tile([.01, -.006, .004, 0.], (3, 1))
    pre = upstream.copy(); nz = increment != 0.; pre[nz] += increment[nz]
    consistent = np.tile([True, False, False, False], (3, 1))
    available = np.tile([True, True, True, False], (3, 1))
    return pre, upstream, increment, consistent, available


class SourceTests(unittest.TestCase):
    def test_independent_source_cross_on_both_fixtures(self):
        import monthly_trend_gate182_independent as independent
        from test_monthly_trend_gate182_independent import fixture as other_fixture
        for factory in (fixture, other_fixture):
            for mode in ('normal', 'constant', 'missing', 'invalid'):
                p, m, c = factory()
                if mode == 'constant': p['adj'].iloc[:, 0] = 1.
                elif mode == 'missing': p['adj'] = p['adj'].drop(pd.Timestamp(c[3]))
                elif mode == 'invalid': p['adj'].iloc[40, 0] = np.nan
                values, audit = k.build_features(p, m, c)
                self.assertTrue(independent.verify_features(p, m, c, values, audit)['passed'])

    def test_exact_three_blocks_and_zero_sign(self):
        p = np.exp(np.linspace(0., 3., 211)); out = k.measure(p)
        self.assertTrue(out['consistent']); self.assertEqual(out['block_return_estimates'], 3)
        np.testing.assert_allclose(out['block_log_returns'], [1., 1., 1.], atol=1e-15)
        for index in (70, 140, 210):
            q = p.copy(); q[index] = q[index-70]
            self.assertFalse(k.measure(q)['consistent'])
        self.assertFalse(k.measure(p[::-1])['consistent'])
        self.assertTrue(k.measure(np.ones(211))['available'])
        self.assertFalse(k.measure(np.ones(211))['consistent'])

    def test_complete_price_support_and_numeric_failures(self):
        for index in (0, 1, 69, 70, 139, 140, 209, 210):
            for bad in (0., -1., np.inf, np.nan, 'bad'):
                p = np.ones(211, dtype=object); p[index] = bad
                out = k.measure(p)
                self.assertFalse(out['available']); self.assertFalse(out['consistent'])
                self.assertEqual(out['block_return_estimates'], 0)
        for p in (np.ones(210), np.ones(212), np.ones((211, 1))):
            with self.assertRaises(ValueError): k.measure(p)

    def test_scale_and_extreme_prices(self):
        p = np.exp(np.linspace(-1., 1., 211)); wanted = k.measure(p)
        for scale in (1e-200, 1e200, 2.**100):
            got = k.measure(p*scale)
            self.assertEqual(got['consistent'], wanted['consistent'])
            np.testing.assert_allclose(got['block_log_returns'], wanted['block_log_returns'], atol=2e-13)
        p = np.ones(211); p[[0, 70, 140, 210]] = [1e-300, 1e-100, 1e100, 1e300]
        self.assertTrue(k.measure(p)['consistent'])
        with patch.object(k.math, 'log', return_value=np.inf):
            with self.assertRaises(FloatingPointError): k.measure(np.ones(211))

    def test_calendar_window_exact_bounds_and_counts(self):
        p, m, c = fixture(); values, audit = k.build_features(p, m, c)
        self.assertEqual(values['available'].shape, (3, 2))
        self.assertTrue(values['available'].all())
        self.assertEqual(audit['block_return_estimates'], 18)
        row = audit['event_history'][0]
        self.assertEqual(row['price_start'], c[0]); self.assertEqual(row['price_end'], c[210])
        self.assertEqual(row['windows']['formation']['return_count'], 210)
        for j, block in enumerate(row['windows']['blocks']):
            self.assertEqual(block['start_date'], c[j*70])
            self.assertEqual(block['end_date'], c[(j+1)*70])
        json.dumps(audit, allow_nan=False)

    def test_missing_official_interior_session_not_filled(self):
        p, m, c = fixture(); p['adj'] = p['adj'].drop(pd.Timestamp(c[3]))
        v, a = k.build_features(p, m, c)
        self.assertFalse(v['available'][0].any())
        self.assertEqual(a['event_history'][0]['missing_official_sessions'], [c[3]])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'], 'missing_official_session')

    def test_recent_and_future_prices_never_accessed(self):
        p, m, c = fixture(); expected = k.build_features(p, m, c, 1)
        class Poison:
            def __float__(self): raise AssertionError('Recent or future data inspected')
        p['adj'] = p['adj'].astype(object); p['adj'].iloc[211:] = Poison()
        actual = k.build_features(p, m, c, 1)
        self.assertEqual(actual[1], expected[1])
        p['adj'] = p['adj'].iloc[:211]
        self.assertEqual(k.build_features(p, m, c, 1)[1], expected[1])

    def test_all_prefixes_and_input_immutability(self):
        p, m, c = fixture(); before = p['adj'].copy(deep=True); before_meta = copy.deepcopy(m)
        full, audit = k.build_features(p, m, c)
        for n in range(4):
            values, part = k.build_features(p, m, c, n)
            for key in full: np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(part['event_history'], audit['event_history'][:n])
        pd.testing.assert_frame_equal(p['adj'], before); self.assertEqual(m, before_meta)

    def test_missing_history_symbol_and_etf_equal_treatment(self):
        p, m, c = fixture(); p['adj']['ETF'] = p['adj']['A']
        values, _ = k.build_features(p, m, c)
        np.testing.assert_array_equal(values['consistent'][:, 0], values['consistent'][:, 1])
        m['dates'][0] = c[251]; m['symbols'][1][1] = 'UNKNOWN'
        values, audit = k.build_features(p, m, c)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')
        self.assertEqual(audit['event_history'][1]['cells'][1]['reason'], 'missing_symbol')
        self.assertFalse(values['consistent'][0].any())

    def test_invalid_schema_and_prefix(self):
        p, m, c = fixture()
        for limit in (-1, 4, True, 1.1):
            with self.assertRaises(ValueError): k.build_features(p, m, c, limit)
        with self.assertRaises(ValueError): k.build_features(p, dict(dates=m['dates'], symbols=m['symbols']), c)
        for bad_c in (c[::-1], c+[c[-1]], []):
            with self.assertRaises(ValueError): k.build_features(p, m, bad_c)


class RouteTests(unittest.TestCase):
    def test_independent_routing_cross_on_both_fixtures(self):
        import monthly_trend_gate182_independent as independent
        from test_monthly_trend_gate182_independent import routing_fixture as other_fixture
        for factory in (route_fixture, other_fixture):
            args = factory()
            for n in range(len(args[0])+1):
                small = tuple(a[:n] for a in args)
                scores, audit = k.route_scores(*small)
                self.assertTrue(independent.verify_routing(*small, scores, audit)['passed'])

    def test_partial_budget_matched_with_signed_increment(self):
        p, b, d, flags, valid = route_fixture(); scores, audit = k.route_scores(p, b, d, flags, valid)
        self.assertEqual(audit['uniform_removed_fraction'], [0., .5, .5])
        self.assertEqual(scores[k.ARMS[0]][1, 0], b[1, 0])
        np.testing.assert_array_equal(scores[k.ARMS[0]][1, 1:], p[1, 1:])
        np.testing.assert_array_equal(scores[k.ARMS[1]][1, :3], b[1, :3]+.5*d[1, :3])
        self.assertEqual(audit['candidate_removed_abs_increment'], [0., .01, .01])
        self.assertEqual(audit['control_removed_abs_increment'], [0., .01, .01])
        json.dumps(audit, allow_nan=False)

    def test_initial_row_keeps_exact_original_bytes(self):
        args = route_fixture(); scores, audit = k.route_scores(*args)
        for out in scores.values(): self.assertEqual(out[0].tobytes(), args[0][0].tobytes())
        self.assertEqual(audit['event_history'][0]['consistent_abs_increment'], .01)
        self.assertEqual(audit['event_history'][0]['uniform_removed_fraction'], 0.)

    def test_missing_and_false_states_preserve_original(self):
        p, b, d, flags, valid = route_fixture(); valid[:] = False; flags[:] = False
        scores, audit = k.route_scores(p, b, d, flags, valid)
        for out in scores.values(): self.assertEqual(out.tobytes(), p.tobytes())
        self.assertEqual(audit['uniform_removed_fraction'], [0., 0., 0.])
        valid[:, :3] = True
        scores, _ = k.route_scores(p, b, d, flags, valid)
        for out in scores.values(): self.assertEqual(out.tobytes(), p.tobytes())

    def test_full_removal_copies_upstream_exactly(self):
        p, b, d, flags, valid = route_fixture(); flags[:] = valid
        scores, audit = k.route_scores(p, b, d, flags, valid)
        for out in scores.values(): self.assertEqual(out[1:].tobytes(), b[1:].tobytes())
        self.assertEqual(audit['uniform_removed_fraction'], [0., 1., 1.])

    def test_budget_uses_only_same_available_set_and_abs_values(self):
        p, b, d, flags, valid = route_fixture(); valid[:, 2] = False
        scores, audit = k.route_scores(p, b, d, flags, valid)
        self.assertEqual(audit['uniform_removed_fraction'][1], .01/.016)
        for out in scores.values(): np.testing.assert_array_equal(out[:, 2], p[:, 2])
        self.assertAlmostEqual(audit['control_removed_abs_increment'][1], .01)

    def test_zero_increment_and_subtraction_roundoff_no_new_signal(self):
        p, b, d, flags, valid = route_fixture(); d[:] = 0.; p[:] = b
        scores, audit = k.route_scores(p, b, d, flags, valid)
        for out in scores.values(): self.assertEqual(out.tobytes(), p.tobytes())
        self.assertEqual(audit['uniform_removed_fraction'], [0., 0., 0.])
        b = np.full((2, 1), 1e10); d = np.full((2, 1), .001); p = b+d
        self.assertNotEqual(float((p-b)[0, 0]), .001)
        k.route_scores(p, b, d, np.ones((2, 1), bool), np.ones((2, 1), bool))

    def test_reconstruction_mismatch_and_nan_nonzero_increment_rejected(self):
        p, b, d, flags, valid = route_fixture(); p[1, 0] = np.nextafter(p[1, 0], np.inf)
        with self.assertRaises(ValueError): k.route_scores(p, b, d, flags, valid)
        p, b, d, flags, valid = route_fixture(); d[:, 3] = .001
        with self.assertRaises(ValueError): k.route_scores(p, b, d, flags, valid)
        p, b, d, flags, valid = route_fixture(); p[1, 3] = 0.
        with self.assertRaises(ValueError): k.route_scores(p, b, d, flags, valid)

    def test_invalid_values_shapes_and_state_masks_rejected(self):
        for target in (0, 1, 2):
            for bad in (np.inf, -np.inf):
                args = list(route_fixture()); args[target][1, 0] = bad
                with self.assertRaises(ValueError): k.route_scores(*args)
        for bad in (np.nan, .013):
            args = list(route_fixture()); args[2][1, 0] = bad
            with self.assertRaises(ValueError): k.route_scores(*args)
        args = list(route_fixture()); args[3][1, 3] = True
        with self.assertRaises(ValueError): k.route_scores(*args)
        args = list(route_fixture()); args[4] = args[4].astype(float)
        with self.assertRaises(ValueError): k.route_scores(*args)
        args = list(route_fixture()); args[2] = args[2][:, :1]
        with self.assertRaises(ValueError): k.route_scores(*args)

    def test_route_prefixes_and_no_input_mutation_or_alias(self):
        args = route_fixture(); original = [a.copy() for a in args]; full, audit = k.route_scores(*args)
        for n in range(4):
            part, check = k.route_scores(*(a[:n] for a in args))
            for arm in k.ARMS: np.testing.assert_array_equal(part[arm], full[arm][:n])
            self.assertEqual(check['event_history'], audit['event_history'][:n])
        for a, before in zip(args, original): self.assertEqual(a.tobytes(), before.tobytes())
        for out in full.values():
            for a in args: self.assertFalse(np.shares_memory(out, a))


if __name__ == '__main__':
    unittest.main(verbosity=2)
