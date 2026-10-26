"""Synthetic-only checks for 174's fixed historical source measurement."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import market_absorption174_kernel as kernel


def fixture():
    anchors = [1, 50, 100, 150, 200, 250]
    growth = [1.10, 1.01, 1.20, 1.02, 1.30, 1.03]
    adjusted = np.full(1513, 100.)
    x = np.arange(1513)
    market = 100.*np.exp(.0002*x+.008*np.sin(x*.1))
    close, volume = np.ones(1259), np.ones(1259)
    for i, (s, g) in enumerate(zip(anchors, growth)):
        a = s+252
        adjusted[a:] *= .99
        adjusted[a+1:] *= g
        if i % 2 == 0:
            volume[s+19] = 2.
    return adjusted, market, close, volume


def panel_fixture(length=1640):
    dates = pd.bdate_range('2000-01-03', periods=length)
    # Only synthetic weekday sessions. Host must pin the actual official calendar.
    calendar = dates.strftime('%Y-%m-%d').tolist()
    panel = {key: pd.DataFrame(100. if key == 'adj' else 1.,
                              index=dates, columns=['A', 'B', 'SPY'])
             for key in ('adj', 'close', 'volume')}
    p, m, c, v = fixture()
    t = 1554
    for symbol in ('A', 'B'):
        panel['adj'].loc[dates[:1513], symbol] = p
        panel['close'].loc[dates[t-1321:t-63+1], symbol] = c
        panel['volume'].loc[dates[t-1321:t-63+1], symbol] = v
    panel['adj'].loc[dates[:1513], 'SPY'] = m
    return panel, [(calendar[t], ['A', 'B'])], calendar


class MeasureTests(unittest.TestCase):
    def test_constants_and_offsets(self):
        self.assertEqual(kernel.ADJUSTED_COUNT, 1513)
        self.assertEqual(kernel.ACTIVITY_COUNT, 1259)
        self.assertEqual(kernel.ADJUSTED_START_OFFSET, 1554)
        self.assertEqual(kernel.ACTIVITY_START_OFFSET, 1321)
        self.assertEqual(kernel.ACTIVITY_END_OFFSET, 63)
        self.assertEqual(kernel.BETA_WINDOW, 252)

    def test_sampling_raw_control_and_residual_candidate(self):
        p, m, c, v = fixture()
        out = kernel.measure(p, m, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['selected_anchors'], [1, 50, 100, 150, 200, 250])
        self.assertEqual(out['high_anchors'], [1, 100, 200])
        self.assertEqual(out['low_anchors'], [50, 150, 250])
        self.assertEqual(out['labels'], out['raw_labels'])
        labels, residuals, betas = [], [], []
        for s in out['selected_anchors']:
            a = s+252
            x = np.diff(np.log(m[a-253:a]))
            y = np.diff(np.log(p[a-253:a]))
            x, y = x-x.mean(), y-y.mean()
            beta = np.dot(x, y)/np.dot(x, x)
            label = math.log(p[a+21]/p[a])
            betas.append(beta)
            labels.append(label)
            residuals.append(label-beta*math.log(m[a+21]/m[a]))
        np.testing.assert_allclose(out['betas'], betas, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out['residual_labels'], residuals, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(out['control'], sum(labels[::2])/23-sum(labels[1::2])/23)
        self.assertAlmostEqual(out['candidate'], sum(residuals[::2])/23-sum(residuals[1::2])/23)
        self.assertEqual((out['beta_attempts_count'], out['beta_estimates_count']), (6, 6))

    def test_raw_arm_matches_frozen173_contrast(self):
        import shock_absorption173_kernel as prior
        p, m, c, v = fixture()
        old, new = prior.measure(p[252:], c, v), kernel.measure(p, m, c, v)
        self.assertEqual(new['control'], old['candidate'])
        for field in ('selected_anchors', 'high_anchors', 'low_anchors', 'labels', 'classifications'):
            self.assertEqual(new[field], old[field])

    def test_beta_first_and_last_price_boundaries(self):
        p, m, c, v = fixture()
        p[1491:] *= .99
        p[1512] *= 1.5
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['selected_anchors'][-1], 1239)
        self.assertEqual(out['extended_anchor_indices'][0], 253)
        self.assertEqual((out['beta_first_indices'][0], out['beta_last_indices'][0]), (0, 252))
        self.assertEqual((out['beta_first_indices'][-1], out['beta_last_indices'][-1]), (1238, 1490))
        self.assertAlmostEqual(out['labels'][-1], math.log(1.5))

    def test_anchor_return_and_forward_label_are_excluded_from_beta(self):
        p, m, c, v = fixture()
        # Give the earliest beta nonzero prehistory; changing the anchor price
        # and matured outcome still must not change that beta.
        p[:253] *= m[:253]/m[252]
        ref = kernel.measure(p, m, c, v)
        p[253] *= .98
        p[274] *= 1.07
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['selected_anchors'][0], 1)
        self.assertEqual(out['betas'][0], ref['betas'][0])
        self.assertNotEqual(out['labels'][0], ref['labels'][0])

    def test_last_preanchor_price_is_included_in_beta(self):
        p, m, c, v = fixture()
        ref = kernel.measure(p, m, c, v)
        p[252] *= 1.05
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['selected_anchors'][0], 1)
        self.assertNotEqual(out['betas'][0], ref['betas'][0])

    def test_beta_ols_with_intercept_and_negative_slope(self):
        x = np.tile([-.01, .02, .001], 84).tolist()
        for slope in (2., -3., 0.):
            y = [slope*r+.007 for r in x]
            beta, reason, ss = kernel._beta_from_returns(y, x)
            self.assertEqual(reason, 'available')
            self.assertAlmostEqual(beta, slope)
            self.assertAlmostEqual(ss, sum((r-np.mean(x))**2 for r in x))

    def test_beta_zero_variance_exactly_unavailable(self):
        x = [.01]*252
        beta, reason, ss = kernel._beta_from_returns([.02]*252, x)
        self.assertIsNone(beta)
        self.assertEqual(reason, 'zero_or_nonfinite_market_variance')
        self.assertEqual(ss, 0.)

    def test_tiny_positive_variance_not_thresholded_or_beta_clipped(self):
        x = [(-1. if i % 2 else 1.)*1e-12 for i in range(252)]
        beta, reason, ss = kernel._beta_from_returns([r*1e10 for r in x], x)
        self.assertEqual(reason, 'available')
        self.assertGreater(ss, 0.)
        self.assertAlmostEqual(beta/1e10, 1.)

    def test_beta_requires_exactly_252_returns_and_finite_inputs(self):
        with self.assertRaises(ValueError):
            kernel._beta_from_returns([0.]*251, [0.]*252)
        y = [0.]*252
        y[0] = float('nan')
        self.assertEqual(kernel._beta_from_returns(y, [0.]*252)[1], 'nonfinite_beta_input')

    def test_constant_market_invalidates_both_arms_and_records_all_anchors(self):
        p, _, c, v = fixture()
        out = kernel.measure(p, np.ones(1513), c, v)
        self.assertEqual(out['reason'], 'invalid_beta')
        self.assertFalse(out['available'])
        self.assertEqual(out['invalid_beta_anchors'], out['selected_anchors'])
        self.assertEqual(out['betas'], [None]*6)
        self.assertEqual(out['residual_labels'], [None]*6)
        self.assertEqual(out['beta_reasons'], ['zero_or_nonfinite_market_variance']*6)
        self.assertEqual((out['beta_attempts_count'], out['beta_estimates_count']), (6, 0))
        self.assertTrue(math.isnan(out['candidate']) and math.isnan(out['control']))

    def test_common_cooldown_before_activity_classification(self):
        _, m, c, v = fixture()
        p = np.full(1513, 100.)
        for s in (1, 2, 21, 22, 23, 42, 43):
            p[s+252:] *= .99
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['selected_anchors'], [1, 22, 43])
        v[:] = 0.
        zero = kernel.measure(p, m, c, v)
        self.assertEqual(zero['selected_anchors'], out['selected_anchors'])
        self.assertEqual(zero['labels'], out['labels'])
        self.assertEqual(zero['invalid_activity_anchors'], [1, 22, 43])

    def test_unmatured_negative_day_excluded(self):
        p, m, c, v = fixture()
        p[1240+252:] *= .99
        self.assertEqual(kernel.measure(p, m, c, v)['selected_anchors'], [1, 50, 100, 150, 200, 250])

    def test_strict_exact_activity_threshold(self):
        p, m, c, v = fixture()
        v[20] = 1.
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][0], 'L')
        v[20] = np.nextafter(1., 2.)
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][0], 'H')
        eps = 2.**-52
        c, v = np.full(1259, 1.+eps), np.full(1259, 1.-eps)
        for s in (1, 100, 200):
            c[s+19], v[s+19] = 1., 1.
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['high_anchors'], [1, 100, 200])

    def test_prior_activity_boundaries_and_current_zero(self):
        p, m, c, v = fixture()
        v[21] = 1e300
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][0], 'H')
        v[0] = 21.
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][0], 'L')
        v[0], v[19] = 1., 21.
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][0], 'L')
        v[69] = 0.
        self.assertEqual(kernel.measure(p, m, c, v)['classifications'][1], 'L')

    def test_group_minimum_and_failure_precedence(self):
        p, m, c, v = fixture()
        v[20] = 1.
        out = kernel.measure(p, m, c, v)
        self.assertEqual(out['reason'], 'insufficient_groups')
        self.assertEqual((out['n_high'], out['n_low']), (2, 4))
        m[:] = 1.
        self.assertEqual(kernel.measure(p, m, c, v)['reason'], 'invalid_beta')
        v[:20] = 0.
        self.assertEqual(kernel.measure(p, m, c, v)['reason'], 'zero_prior_activity')
        p[0] = 0.
        self.assertEqual(kernel.measure(p, m, c, v)['reason'], 'invalid_adjusted_price')

    def test_zero_raw_labels_are_valid(self):
        _, m, c, v = fixture()
        p = np.full(1513, 100.)
        for s in (1, 50, 100, 150, 200, 250):
            p[s+252:] *= .99
        out = kernel.measure(p, m, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['labels'], [0.]*6)
        self.assertEqual(out['control'], 0.)

    def test_negative_raw_labels_and_maximum_59_anchors(self):
        _, m, c, v = fixture()
        p = np.exp(-np.arange(1513)*.001)
        v[:] = 1.
        for i, s in enumerate(range(1, 1240, 21)):
            v[s+19] = 2. if i % 2 == 0 else .5
        out = kernel.measure(p, m, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['n_all'], 59)
        self.assertTrue(all(r < 0. for r in out['labels']))

    def test_constant_and_rising_stock_prices_have_no_anchors(self):
        _, m, c, v = fixture()
        for p in (np.ones(1513), np.linspace(1., 2., 1513)):
            out = kernel.measure(p, m, c, v)
            self.assertEqual(out['reason'], 'insufficient_groups')
            self.assertEqual(out['selected_anchors'], [])
            self.assertEqual(out['beta_attempts_count'], 0)

    def test_overflow_underflow_activity_and_unit_invariance(self):
        p, m, c, v = fixture()
        ref = kernel.measure(p, m, c, v)
        for scale in (1e300, 1e-300):
            out = kernel.measure(p, m, c*scale, v*scale)
            self.assertEqual(out['classifications'], ref['classifications'])
            self.assertEqual(out['candidate'], ref['candidate'])
        out = kernel.measure(p*8., m*16., c*16., v/16.)
        self.assertEqual(out['labels'], ref['labels'])
        self.assertEqual(out['betas'], ref['betas'])

    def test_extreme_log_ratios_and_near_equal_prices(self):
        small, big = float(np.nextafter(0., 1.)), float(np.finfo(float).max)
        self.assertTrue(math.isfinite(kernel._log_ratio(big, small)))
        self.assertAlmostEqual(kernel._log_ratio(big, small), -kernel._log_ratio(small, big))
        self.assertGreater(kernel._log_ratio(np.nextafter(1., 2.), 1.), 0.)

    def test_full_source_validation_including_beta_only_history(self):
        for number, reason, bad in ((0, 'invalid_adjusted_price', 0.),
                                    (0, 'invalid_adjusted_price', np.nan),
                                    (1, 'invalid_market_price', np.inf),
                                    (1, 'invalid_market_price', 0.),
                                    (2, 'invalid_stored_close', 0.),
                                    (3, 'invalid_stored_volume', -1.)):
            for index in (0, -1):
                arrays = list(fixture())
                arrays[number][index] = bad
                with self.subTest(number=number, index=index):
                    out = kernel.measure(*arrays)
                    self.assertEqual(out['reason'], reason)
                    self.assertEqual(out['selected_anchors'], [])

    def test_shapes_and_nonnumeric_inputs(self):
        arrays = list(fixture())
        for number in range(4):
            bad = list(arrays)
            bad[number] = bad[number][:-1]
            with self.assertRaises(ValueError):
                kernel.measure(*bad)
        arrays[1] = arrays[1].astype(object)
        arrays[1][0] = 'unreadable'
        self.assertEqual(kernel.measure(*arrays)['reason'], 'nonnumeric_market_price')

    def test_sources_not_modified(self):
        arrays = fixture()
        original = [a.copy() for a in arrays]
        kernel.measure(*arrays)
        for a, b in zip(arrays, original):
            np.testing.assert_array_equal(a, b)


class BuilderTests(unittest.TestCase):
    def test_window_dates_and_beta_audit(self):
        panel, events, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, events, calendar)
        row = audit['event_history'][0]
        cell = row['cells'][0]
        self.assertEqual(values['candidate'].shape, (1, 2))
        self.assertEqual((audit['available_cells'], audit['unavailable_cells']), (2, 0))
        self.assertEqual((row['adjusted_start'], row['adjusted_end']), (calendar[0], calendar[1512]))
        self.assertEqual((row['activity_start'], row['activity_end']), (calendar[233], calendar[1491]))
        self.assertEqual(cell['selected_anchor_dates'][0], calendar[253])
        self.assertEqual(cell['label_end_dates'][0], calendar[274])
        self.assertEqual(cell['beta_first_dates'][0], calendar[0])
        self.assertEqual(cell['beta_last_dates'][0], calendar[252])
        self.assertEqual(audit['market_symbol'], 'SPY')
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertEqual((audit['beta_attempts_count'], audit['beta_estimates_count']), (12, 12))

    def test_future_poison_ignored_after_individual_cutoffs(self):
        panel, events, calendar = panel_fixture()
        ref, ref_audit = kernel.build_features(panel, events, calendar)
        for key in panel:
            panel[key] = panel[key].astype(object)
            panel[key].iloc[1513 if key == 'adj' else 1492:, :] = 'do not inspect future'
        values, audit = kernel.build_features(panel, events, calendar)
        for key in values:
            np.testing.assert_array_equal(values[key], ref[key])
        self.assertEqual(audit, ref_audit)

    def test_truncated_inputs_and_event_prefix_identical(self):
        panel, events, calendar = panel_fixture()
        events.append((calendar[1575], ['B', 'A']))
        full, audit = kernel.build_features(panel, events, calendar)
        short, short_audit = kernel.build_features(panel, events, calendar, limit=1)
        clipped = {key: frame.iloc[:1513].copy() for key, frame in panel.items()}
        truncated, trunc_audit = kernel.build_features(clipped, events[:1], calendar)
        for key in short:
            np.testing.assert_array_equal(short[key], full[key][:1])
            np.testing.assert_array_equal(truncated[key], short[key])
        self.assertEqual(short_audit, trunc_audit)
        self.assertEqual(audit['event_history'][:1], short_audit['event_history'])
        empty, a = kernel.build_features(panel, events, calendar, limit=0)
        self.assertEqual(empty['available'].shape, (0, 2))
        self.assertEqual(a['beta_attempts_count'], 0)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, events, calendar, limit=True)

    def test_missing_sessions_never_compresses_window(self):
        panel, events, calendar = panel_fixture()
        for position in (0, 232, 233, 1491, 1512):
            bad = {key: frame.drop(pd.Timestamp(calendar[position])) for key, frame in panel.items()}
            values, audit = kernel.build_features(bad, events, calendar)
            self.assertFalse(values['available'].any())
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_official_session')

    def test_missing_market_symbol_and_invalid_market_propagate(self):
        panel, events, calendar = panel_fixture()
        bad = {key: frame.drop(columns=['SPY']) for key, frame in panel.items()}
        values, audit = kernel.build_features(bad, events, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_market_symbol')
        panel['adj'].iloc[0, 2] = np.nan
        values, audit = kernel.build_features(panel, events, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'invalid_market_price')

    def test_missing_symbol_and_bad_member_value_isolated(self):
        panel, events, calendar = panel_fixture()
        events[0] = (events[0][0], ['A', 'ABSENT'])
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[True, False]])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'missing_symbol')
        events[0] = (events[0][0], ['A', 'B'])
        panel['adj'].iloc[0, 1] = 0.
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[True, False]])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'invalid_adjusted_price')

    def test_insufficient_history_extended_beta_warmup(self):
        panel, _, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, [(calendar[1553], ['A'])], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')

    def test_unavailable_beta_dates_complete(self):
        panel, events, calendar = panel_fixture()
        panel['adj'].loc[:, 'SPY'] = 1.
        values, audit = kernel.build_features(panel, events, calendar)
        cell = audit['event_history'][0]['cells'][0]
        self.assertEqual(cell['invalid_beta_anchor_dates'], cell['selected_anchor_dates'])
        self.assertEqual(cell['candidate'], None)
        self.assertEqual(audit['beta_estimates_count'], 0)
        self.assertEqual(audit['beta_attempts_count'], 12)

    def test_axes_calendar_and_identity_validation(self):
        panel, events, calendar = panel_fixture()
        bad = dict(panel, volume=panel['volume'].iloc[1:])
        with self.assertRaises(ValueError):
            kernel.build_features(bad, events, calendar)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, events, [calendar[0], calendar[0]])
        with self.assertRaises(ValueError):
            kernel.build_features(panel, [('2000-01-01', ['A'])], calendar)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, [(events[0][0], ['A', 'A'])], calendar)

    def test_builder_preserves_all_inputs(self):
        panel, events, calendar = panel_fixture()
        original = copy.deepcopy((panel, events, calendar))
        kernel.build_features(panel, events, calendar)
        for key in panel:
            pd.testing.assert_frame_equal(panel[key], original[0][key])
        self.assertEqual(events, original[1])
        self.assertEqual(calendar, original[2])

    def test_independent_rebuild_with_both_synthetic_fixtures(self):
        import market_absorption174_independent as independent
        from test_market_absorption174_independent import fixture as other_fixture
        for builder in (panel_fixture, other_fixture):
            panel, events, calendar = builder()
            values, audit = kernel.build_features(panel, events, calendar)
            receipt = independent.verify_features(panel, events, calendar, values, audit)
            self.assertTrue(receipt['passed'])
            self.assertTrue(receipt['pre_anchor_beta_checked'])
            self.assertTrue(receipt['shared_beta_support_checked'])

    def test_independent_unavailability_audits(self):
        import market_absorption174_independent as independent
        for scenario in ('zero_market', 'zero_activity', 'missing_market', 'missing_session', 'missing_member'):
            panel, events, calendar = panel_fixture()
            if scenario == 'zero_market':
                panel['adj'].loc[:, 'SPY'] = 1.
            elif scenario == 'zero_activity':
                panel['volume'].iloc[233:253, 1] = 0.
            elif scenario == 'missing_market':
                panel = {key: frame.drop(columns=['SPY']) for key, frame in panel.items()}
            elif scenario == 'missing_session':
                panel = {key: frame.iloc[1:].copy() for key, frame in panel.items()}
            elif scenario == 'missing_member':
                events[0] = (events[0][0], ['A', 'ABSENT'])
            values, audit = kernel.build_features(panel, events, calendar)
            with self.subTest(scenario=scenario):
                self.assertTrue(independent.verify_features(panel, events, calendar, values, audit)['passed'])


if __name__ == '__main__':
    unittest.main()
