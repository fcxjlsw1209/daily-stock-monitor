"""Synthetic-only tests for the separately registered 173 source measurement."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import shock_absorption173_kernel as kernel


def fixture():
    anchors = [1, 50, 100, 150, 200, 250]
    growth = [1.10, 1.01, 1.20, 1.02, 1.30, 1.03]
    adjusted = np.full(1261, 100.)
    close = np.ones(1259)
    volume = np.ones(1259)
    for i, (s, g) in enumerate(zip(anchors, growth)):
        adjusted[s:] *= .99
        adjusted[s+1:] *= g
        if i % 2 == 0:
            volume[s+19] = 2.
    return adjusted, close, volume


def panel_fixture(length=1380):
    dates = pd.bdate_range('2000-01-03', periods=length)
    # Synthetic weekday calendar; real execution requires the pinned calendar.
    calendar = dates.strftime('%Y-%m-%d').tolist()
    panel = {key: pd.DataFrame(100. if key == 'adj' else 1.,
                              index=dates, columns=['A', 'B'])
             for key in ('adj', 'close', 'volume')}
    p, c, v = fixture()
    t = 1321
    for symbol in ('A', 'B'):
        panel['adj'].loc[dates[t-1302:t-42+1], symbol] = p
        panel['close'].loc[dates[t-1321:t-63+1], symbol] = c
        panel['volume'].loc[dates[t-1321:t-63+1], symbol] = v
    return panel, [(calendar[t], ['A', 'B'])], calendar


class MeasureTests(unittest.TestCase):
    def test_constants_and_offsets(self):
        self.assertEqual(kernel.ADJUSTED_COUNT, 1261)
        self.assertEqual(kernel.ACTIVITY_COUNT, 1259)
        self.assertEqual(kernel.LAST_ANCHOR, 1239)
        self.assertEqual(kernel.ADJUSTED_START_OFFSET, 1302)
        self.assertEqual(kernel.ACTIVITY_START_OFFSET, 1321)
        self.assertEqual(kernel.ACTIVITY_END_OFFSET, 63)

    def test_exact_sampling_and_shrunk_signed_means(self):
        p, c, v = fixture()
        out = kernel.measure(p, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['selected_anchors'], [1, 50, 100, 150, 200, 250])
        self.assertEqual(out['high_anchors'], [1, 100, 200])
        self.assertEqual(out['low_anchors'], [50, 150, 250])
        self.assertEqual(out['classifications'], ['H', 'L', 'H', 'L', 'H', 'L'])
        expected = [math.log(p[s+21]/p[s]) for s in out['selected_anchors']]
        np.testing.assert_allclose(out['labels'], expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(out['candidate'], sum(expected[::2])/23-sum(expected[1::2])/23)
        self.assertAlmostEqual(out['control'], sum(expected)/26)

    def test_cooldown_is_common_and_exactly_21(self):
        p = np.full(1261, 100.)
        for s in (1, 2, 21, 22, 23, 42, 43):
            p[s:] *= .99
        out = kernel.measure(p, np.ones(1259), np.ones(1259))
        self.assertEqual(out['selected_anchors'], [1, 22, 43])

    def test_activity_cannot_change_anchors(self):
        p, c, v = fixture()
        reference = kernel.measure(p, c, v)
        v[:] = 0.
        out = kernel.measure(p, c, v)
        self.assertEqual(out['selected_anchors'], reference['selected_anchors'])
        self.assertEqual(out['labels'], reference['labels'])
        self.assertEqual(out['reason'], 'zero_prior_activity')
        self.assertEqual(out['invalid_activity_anchors'], reference['selected_anchors'])
        self.assertEqual(out['classifications'], [None]*6)

    def test_strict_threshold_and_one_ulp(self):
        p, c, v = fixture()
        v[20] = 1.
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'L')
        v[20] = np.nextafter(1., 2.)
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'H')
        v[20] = np.nextafter(1., 0.)
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'L')

    def test_exact_products_not_pre_rounded_products(self):
        p, _, _ = fixture()
        eps = 2. ** -52
        c, v = np.full(1259, 1. + eps), np.full(1259, 1. - eps)
        # Every baseline float product rounds to 1, but its exact dyadic
        # product is 1-eps**2. Exact anchor product 1 is strictly higher.
        self.assertEqual(float(c[0] * v[0]), 1.)
        for s in (1, 100, 200):
            c[s+19], v[s+19] = 1., 1.
        out = kernel.measure(p, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['high_anchors'], [1, 100, 200])

    def test_zero_matured_returns_are_valid_not_missing(self):
        _, c, v = fixture()
        p = np.full(1261, 100.)
        for s in (1, 50, 100, 150, 200, 250):
            p[s:] *= .99
        out = kernel.measure(p, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['labels'], [0.] * 6)
        self.assertEqual(out['candidate'], 0.)
        self.assertEqual(out['control'], 0.)

    def test_prior20_boundaries_exclude_current_and_next(self):
        p, c, v = fixture()
        v[:21] = 1.
        v[20] = 2.
        v[21] = 1e300
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'H')
        v[0] = 21.
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'L')
        v[0] = 1.
        v[19] = 21.
        self.assertEqual(kernel.measure(p, c, v)['classifications'][0], 'L')

    def test_minimum_groups_three_not_total_six(self):
        p, c, v = fixture()
        v[20] = 1.
        out = kernel.measure(p, c, v)
        self.assertEqual((out['n_high'], out['n_low'], out['n_all']), (2, 4, 6))
        self.assertEqual(out['reason'], 'insufficient_groups')
        self.assertFalse(out['available'])
        self.assertTrue(math.isnan(out['candidate']) and math.isnan(out['control']))

    def test_zero_prior_invalidates_without_skipping_anchor(self):
        p, c, v = fixture()
        v[:20] = 0.
        out = kernel.measure(p, c, v)
        self.assertEqual(out['reason'], 'zero_prior_activity')
        self.assertEqual(out['invalid_activity_anchors'], [1])
        self.assertEqual(out['n_all'], 6)
        self.assertEqual(len(out['labels']), 6)

    def test_zero_current_activity_is_valid_low(self):
        p, c, v = fixture()
        v[69] = 0.
        out = kernel.measure(p, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['classifications'][1], 'L')

    def test_constant_prices_have_no_anchors(self):
        out = kernel.measure(np.ones(1261), np.ones(1259), np.ones(1259))
        self.assertEqual(out['reason'], 'insufficient_groups')
        self.assertEqual(out['selected_anchors'], [])

    def test_positive_only_days_do_not_become_anchors(self):
        out = kernel.measure(np.linspace(1., 2., 1261), np.ones(1259), np.ones(1259))
        self.assertEqual(out['n_all'], 0)

    def test_last_mature_anchor_and_final_label(self):
        p, c, v = fixture()
        p[1239:] *= .99
        p[1260] *= 1.5
        out = kernel.measure(p, c, v)
        self.assertEqual(out['selected_anchors'][-1], 1239)
        self.assertAlmostEqual(out['labels'][-1], math.log(1.5))

    def test_unmatured_negative_day_not_an_anchor(self):
        p, c, v = fixture()
        p[1240:] *= .99
        out = kernel.measure(p, c, v)
        self.assertEqual(out['selected_anchors'], [1, 50, 100, 150, 200, 250])

    def test_negative_labels_preserve_sign_and_max59_anchors(self):
        p = np.exp(-np.arange(1261)*.001)
        c, v = np.ones(1259), np.ones(1259)
        for index, s in enumerate(range(1, 1240, 21)):
            v[s+19] = 2. if index % 2 == 0 else .5
        out = kernel.measure(p, c, v)
        self.assertTrue(out['available'])
        self.assertEqual(out['n_all'], 59)
        self.assertTrue(all(y < 0 for y in out['labels']))
        self.assertAlmostEqual(out['control'], -.021*59/79)

    def test_overflowing_activity_products(self):
        p, _, v = fixture()
        out = kernel.measure(p, np.full(1259, 1e300), v*1e300)
        ref = kernel.measure(*fixture())
        self.assertEqual(out['classifications'], ref['classifications'])
        self.assertEqual(out['candidate'], ref['candidate'])

    def test_underflowing_activity_products(self):
        p, _, v = fixture()
        out = kernel.measure(p, np.full(1259, 1e-300), v*1e-300)
        ref = kernel.measure(*fixture())
        self.assertEqual(out['classifications'], ref['classifications'])
        self.assertEqual(out['candidate'], ref['candidate'])

    def test_price_and_activity_unit_changes(self):
        p, c, v = fixture()
        ref = kernel.measure(p, c, v)
        out = kernel.measure(p*8., c*16., v/16.)
        self.assertEqual(out['classifications'], ref['classifications'])
        self.assertEqual(out['labels'], ref['labels'])

    def test_extreme_log_ratios_are_finite(self):
        small, big = float(np.nextafter(0., 1.)), float(np.finfo(float).max)
        up = kernel._log_ratio(big, small)
        down = kernel._log_ratio(small, big)
        self.assertTrue(math.isfinite(up) and math.isfinite(down))
        self.assertAlmostEqual(up, -down)
        self.assertEqual(kernel._log_ratio(big, big), 0.)

    def test_full_required_source_validation(self):
        for number, reason, bad in ((0, 'invalid_adjusted_price', np.nan),
                                    (0, 'invalid_adjusted_price', 0.),
                                    (1, 'invalid_stored_close', 0.),
                                    (1, 'invalid_stored_close', np.inf),
                                    (2, 'invalid_stored_volume', -1.),
                                    (2, 'invalid_stored_volume', np.nan)):
            arrays = list(fixture())
            arrays[number][-1] = bad
            with self.subTest(number=number, bad=bad):
                out = kernel.measure(*arrays)
                self.assertEqual(out['reason'], reason)
                self.assertFalse(out['available'])

    def test_shape_and_nonnumeric(self):
        with self.assertRaises(ValueError):
            kernel.measure(np.ones(1260), np.ones(1259), np.ones(1259))
        p, c, v = fixture()
        p = p.astype(object)
        p[10] = 'poison'
        self.assertEqual(kernel.measure(p, c, v)['reason'], 'nonnumeric_adjusted_price')

    def test_inputs_not_modified(self):
        source = fixture()
        before = [v.copy() for v in source]
        kernel.measure(*source)
        for x, y in zip(source, before):
            np.testing.assert_array_equal(x, y)


class BuilderTests(unittest.TestCase):
    def test_exact_source_dates_and_audit(self):
        panel, events, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, events, calendar)
        row, cell = audit['event_history'][0], audit['event_history'][0]['cells'][0]
        self.assertEqual(values['candidate'].shape, (1, 2))
        self.assertEqual((audit['available_cells'], audit['unavailable_cells']), (2, 0))
        self.assertEqual(row['adjusted_start'], calendar[19])
        self.assertEqual(row['adjusted_end'], calendar[1279])
        self.assertEqual(row['activity_start'], calendar[0])
        self.assertEqual(row['activity_end'], calendar[1258])
        self.assertEqual(cell['selected_anchor_dates'][0], calendar[20])
        self.assertEqual(cell['label_end_dates'][0], calendar[41])
        self.assertEqual(cell['symbol'], 'A')

    def test_future_numeric_poison_is_never_converted(self):
        panel, events, calendar = panel_fixture()
        ref, _ = kernel.build_features(panel, events, calendar)
        for key in panel:
            panel[key] = panel[key].astype(object)
            start = 1280 if key == 'adj' else 1259
            panel[key].iloc[start:, :] = 'do not inspect future'
        out, _ = kernel.build_features(panel, events, calendar)
        for key in ref:
            np.testing.assert_array_equal(out[key], ref[key])

    def test_truncated_panel_parity(self):
        panel, events, calendar = panel_fixture()
        ref, _ = kernel.build_features(panel, events, calendar)
        short = {key: frame.iloc[:1280].copy() for key, frame in panel.items()}
        out, _ = kernel.build_features(short, events, calendar)
        for key in ref:
            np.testing.assert_array_equal(out[key], ref[key])

    def test_limit_prefix_and_empty_prefix(self):
        panel, events, calendar = panel_fixture()
        events.append((calendar[1342], ['B', 'A']))
        out, audit = kernel.build_features(panel, events, calendar)
        short, short_audit = kernel.build_features(panel, events, calendar, limit=1)
        for key in short:
            np.testing.assert_array_equal(short[key], out[key][:1])
        self.assertEqual(short_audit['event_history'], audit['event_history'][:1])
        empty, e_audit = kernel.build_features(panel, events, calendar, limit=0)
        self.assertEqual(empty['candidate'].shape, (0, 2))
        self.assertEqual(e_audit['events'], 0)
        with self.assertRaises(ValueError):
            kernel.build_features(panel, events, calendar, limit=True)

    def test_missing_official_session_does_not_compress_window(self):
        panel, events, calendar = panel_fixture()
        for remove in (calendar[0], calendar[19], calendar[1279]):
            with self.subTest(remove=remove):
                bad = {key: frame.drop(pd.Timestamp(remove)) for key, frame in panel.items()}
                values, audit = kernel.build_features(bad, events, calendar)
                self.assertFalse(values['available'].any())
                self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_official_session')

    def test_missing_symbol_and_bad_value_are_isolated(self):
        panel, events, calendar = panel_fixture()
        events[0] = (events[0][0], ['A', 'MISSING'])
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[True, False]])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'missing_symbol')
        events[0] = (events[0][0], ['A', 'B'])
        panel['volume'].loc[calendar[0], 'B'] = np.nan
        values, audit = kernel.build_features(panel, events, calendar)
        np.testing.assert_array_equal(values['available'], [[True, False]])
        self.assertEqual(audit['event_history'][0]['cells'][1]['reason'], 'invalid_stored_volume')

    def test_insufficient_history_includes_activity_warmup(self):
        panel, _, calendar = panel_fixture()
        values, audit = kernel.build_features(panel, [(calendar[1320], ['A'])], calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'insufficient_history')

    def test_builder_does_not_override_initial_event(self):
        panel, events, calendar = panel_fixture()
        values, _ = kernel.build_features(panel, events, calendar)
        self.assertNotEqual(values['candidate'][0, 0], 0.)

    def test_panel_axes_and_calendar_validated(self):
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

    def test_builder_preserves_panel_events_and_calendar(self):
        panel, events, calendar = panel_fixture()
        before = copy.deepcopy((panel, events, calendar))
        kernel.build_features(panel, events, calendar)
        for key in panel:
            pd.testing.assert_frame_equal(panel[key], before[0][key])
        self.assertEqual(events, before[1])
        self.assertEqual(calendar, before[2])

    def test_independent_interface_with_both_synthetic_fixtures(self):
        import shock_absorption173_independent as independent
        from test_shock_absorption173_independent import fixture as other_fixture
        for builder in (panel_fixture, other_fixture):
            panel, events, calendar = builder()
            values, audit = kernel.build_features(panel, events, calendar)
            receipt = independent.verify_features(panel, events, calendar, values, audit)
            self.assertTrue(receipt['passed'])
            self.assertTrue(receipt['common_event_cooldown_checked'])

    def test_independent_unavailable_audit_matches(self):
        import shock_absorption173_independent as independent
        panel, events, calendar = panel_fixture()
        panel['volume'].iloc[:20, 1] = 0.
        values, audit = kernel.build_features(panel, events, calendar)
        receipt = independent.verify_features(panel, events, calendar, values, audit)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['unavailable_cells'], 1)


if __name__ == '__main__':
    unittest.main()
