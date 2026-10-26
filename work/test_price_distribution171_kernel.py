"""Synthetic-only formula, scaling and official-calendar window tests."""
import copy
import importlib.util
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_price_distribution171', P/'price_distribution171_kernel.py')
k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k)


def arrays():
    i = np.arange(210, dtype=float)
    return np.r_[80.+.17*i+4.*np.sin(i/11.), 120.], 90.+.13*i, 1e5+200.*i


def panel_fixture():
    # Explicitly omit two weekdays to model official holidays. Do not infer
    # official positions from the DataFrame's row number or generic weekdays.
    sessions = pd.bdate_range('2019-01-02', periods=330).delete([33, 123])
    i = np.arange(len(sessions), dtype=float)
    panel = dict(
        adj=pd.DataFrame({'A':90.+.1*i+np.sin(i/9.), 'B':110.-.05*i+np.sin(i/7.)}, index=sessions),
        close=pd.DataFrame({'A':100.+.13*i, 'B':120.+.02*i}, index=sessions),
        volume=pd.DataFrame({'A':1e5+300.*i, 'B':2e5+50.*i}, index=sessions))
    calendar = sessions.strftime('%Y-%m-%d').tolist()
    events = [(calendar[i], ['A','B']) for i in (251, 252, 280, 300)]
    return panel, calendar, events


def assert_shared_unavailable(case, result, reason):
    case.assertFalse(result['available'])
    case.assertTrue(math.isnan(result['candidate']))
    case.assertTrue(math.isnan(result['control']))
    case.assertEqual(result['reason'], reason)


class FormulaTests(unittest.TestCase):
    def test_independent_scalar_formula(self):
        adjusted, close, volume = arrays()
        g = [(float(adjusted[-1])-float(p))/(float(adjusted[-1])+float(p)) for p in adjusted[:-1]]
        weights = [float(c)*float(v) for c, v in zip(close, volume)]
        control = math.fsum(g)/210.
        candidate = math.fsum(w*x for w, x in zip(weights, g))/math.fsum(weights)-control
        result = k.measure(adjusted, close, volume)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['control'], control, places=15)
        self.assertAlmostEqual(result['candidate'], candidate, places=15)

    def test_positive_adjusted_price_and_activity_scaling_invariance(self):
        adjusted, close, volume = arrays(); original = k.measure(adjusted, close, volume)
        for price_scale, close_scale, volume_scale in ((.001, 1., 1.), (1000., 1., 1.),
                (1., .002, 1.), (1., 1000., 1.), (1., 1., .003), (1., 1., 1000.),
                (100., 100., 1e4)):
            with self.subTest(scales=(price_scale, close_scale, volume_scale)):
                result = k.measure(adjusted*price_scale, close*close_scale, volume*volume_scale)
                self.assertTrue(result['available'])
                self.assertAlmostEqual(result['candidate'], original['candidate'], places=14)
                self.assertAlmostEqual(result['control'], original['control'], places=14)

    def test_constant_activity_exactly_zero_without_erasing_control(self):
        adjusted, close, volume = arrays()
        for c, v in ((np.ones(210)*64., np.ones(210)*16.),
                     (np.tile([1., 2., 4., 8., 16.], 42), np.tile([16., 8., 4., 2., 1.], 42))):
            result = k.measure(adjusted, c, v)
            self.assertTrue(result['available'])
            self.assertEqual(result['candidate'], 0.)
            self.assertNotEqual(result['control'], 0.)

    def test_zero_volume_days_allowed_but_all_zero_unavailable(self):
        adjusted, close, volume = arrays(); volume[:] = 0.; volume[17] = 1.
        result = k.measure(adjusted, close, volume)
        self.assertTrue(result['available']); self.assertEqual(result['positive_activity_days'], 1)
        g = (adjusted[-1]-adjusted[:-1])/(adjusted[-1]+adjusted[:-1])
        self.assertAlmostEqual(result['candidate'], g[17]-g.mean(), places=15)
        volume[:] = 0.
        assert_shared_unavailable(self, k.measure(adjusted, close, volume), 'zero_total_activity')

    def test_constant_historical_price_location_exactly_zero_candidate(self):
        _, close, volume = arrays()
        for reference in (75., 100., 137.):
            result = k.measure(np.r_[np.full(210, 100.), reference], close, volume)
            self.assertTrue(result['available'])
            self.assertEqual(result['candidate'], 0.)
            self.assertAlmostEqual(result['control'], (reference-100.)/(reference+100.), places=15)

    def test_equal_rounded_logs_do_not_erase_distinct_activity(self):
        history = np.tile([80., 120.], 105)
        adjusted = np.r_[history, 100.]
        close = np.full(210, 2.**500)
        volume = np.tile([np.nextafter(1., 2.), 1.], 105)
        logs = np.log(close)+np.log(volume)
        self.assertTrue(np.all(logs == logs[0]))
        self.assertFalse(np.all(close*volume == close[0]*volume[0]))
        result = k.measure(adjusted, close, volume)
        with localcontext() as context:
            context.prec = 80
            ref = Decimal.from_float(100.)
            g = [(ref-Decimal.from_float(float(p)))/(ref+Decimal.from_float(float(p))) for p in history]
            w = [Decimal.from_float(float(c))*Decimal.from_float(float(v)) for c, v in zip(close, volume)]
            expected = sum(x*y for x,y in zip(w,g))/sum(w)-sum(g)/Decimal(210)
        self.assertTrue(result['available'])
        self.assertGreater(result['candidate'], 0.)
        self.assertAlmostEqual(result['candidate']/float(expected), 1., places=14)

    def test_common_support_rejects_invalid_inputs_even_zero_activity_rows(self):
        for slot, reason, bad_values in ((0, 'invalid_adjusted_price', [np.nan,np.inf,0.,-1.]),
            (1, 'invalid_stored_close', [np.nan,np.inf,0.,-1.]),
            (2, 'invalid_stored_volume', [np.nan,np.inf,-1.])):
            for value in bad_values:
                source = list(arrays()); source[slot][17] = value
                if slot != 2:
                    source[2][17] = 0.
                with self.subTest(slot=slot, value=value):
                    assert_shared_unavailable(self, k.measure(*source), reason)
        source = list(arrays()); source[0][-1] = np.nan
        assert_shared_unavailable(self, k.measure(*source), 'invalid_adjusted_price')

    def test_nonnumeric_source_has_explicit_joint_failure(self):
        source = list(arrays()); source[1] = source[1].astype(object); source[1][3] = 'bad'
        assert_shared_unavailable(self, k.measure(*source), 'nonnumeric_stored_close')

    def test_exact_lengths_not_shortened_windows(self):
        for slot in range(3):
            for change in (-1, 1):
                source = list(arrays())
                source[slot] = source[slot][:-1] if change < 0 else np.r_[source[slot], 1.]
                with self.subTest(slot=slot, change=change), self.assertRaises(ValueError):
                    k.measure(*source)

    def test_large_finite_products_and_sum_do_not_overflow_feature(self):
        adjusted, close, volume = arrays()
        original = k.measure(adjusted, close, volume)
        result = k.measure(adjusted*1e300, close*1e250, volume*1e250)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['candidate'], original['candidate'], places=13)
        self.assertAlmostEqual(result['control'], original['control'], places=14)
        # Separate max normalization would underflow every product here.
        c = np.tile([1e300, 1e-300], 105); v = c[::-1].copy()
        result = k.measure(adjusted, c, v)
        self.assertTrue(result['available']); self.assertEqual(result['candidate'], 0.)

    def test_joint_permutation_invariance_and_inputs_unchanged(self):
        source = arrays(); before = [x.copy() for x in source]; result = k.measure(*source)
        permutation = np.random.default_rng(171).permutation(210)
        got = k.measure(np.r_[source[0][:-1][permutation], source[0][-1]],
                        source[1][permutation], source[2][permutation])
        self.assertAlmostEqual(got['candidate'], result['candidate'], places=15)
        self.assertAlmostEqual(got['control'], result['control'], places=15)
        for actual, prior in zip(source, before):
            np.testing.assert_array_equal(actual, prior)


class CalendarTests(unittest.TestCase):
    def test_first_available_event_and_exact_official_210_window(self):
        panel, calendar, events = panel_fixture()
        values, audit = k.build_features(panel, events, calendar)
        self.assertEqual(values['candidate'].shape, (4,2))
        self.assertFalse(values['available'][0].any())
        self.assertTrue(values['available'][1:].all())
        self.assertEqual(audit['event_history'][0]['reasons'], ['insufficient_history']*2)
        detail = audit['event_history'][1]
        self.assertEqual((detail['window_start'], detail['history_end'], detail['reference_date']),
                         (calendar[0],calendar[209],calendar[210]))
        for row, position in enumerate((252,280,300), start=1):
            for column, symbol in enumerate(['A','B']):
                dates = calendar[position-252:position-42+1]
                expected = k.measure(panel['adj'].loc[dates,symbol].to_numpy(),
                    panel['close'].loc[dates[:-1],symbol].to_numpy(),
                    panel['volume'].loc[dates[:-1],symbol].to_numpy())
                self.assertEqual(values['candidate'][row,column], expected['candidate'])
                self.assertEqual(values['control'][row,column], expected['control'])
        self.assertEqual(audit['available_cells'], 6)
        self.assertEqual(audit['unavailable_cells'], 2)
        json.dumps(audit, allow_nan=False)

    def test_recent_future_and_reference_activity_are_not_read(self):
        panel, calendar, events = panel_fixture(); event = [events[1]]
        expected, audit = k.build_features(panel, event, calendar)
        altered = {key:value.astype(object) for key,value in panel.items()}
        altered['adj'].loc[calendar[211]:] = 'future poison'
        for key in ('close','volume'):
            altered[key].loc[calendar[210]:] = 'reference activity and future poison'
        actual, _ = k.build_features(altered, event, calendar)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_price_and_activity_at_every_consumed_endpoint_matter(self):
        panel, calendar, events = panel_fixture(); event = [events[1]]
        for key, position in (('adj',0), ('adj',209), ('adj',210),
                              ('close',0), ('close',209), ('volume',0), ('volume',209)):
            altered = copy.deepcopy(panel); altered[key].loc[calendar[position],'A'] = np.nan
            values, audit = k.build_features(altered, event, calendar)
            self.assertFalse(values['available'][0,0]); self.assertTrue(values['available'][0,1])
            self.assertTrue(np.isnan(values['candidate'][0,0])); self.assertTrue(np.isnan(values['control'][0,0]))

    def test_missing_official_session_no_compression_and_extra_noncalendar_ignored(self):
        panel, calendar, events = panel_fixture(); event = [events[1]]
        removed = {key:value.drop(pd.Timestamp(calendar[17])) for key,value in panel.items()}
        values, audit = k.build_features(removed, event, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['event_history'][0]['missing_official_sessions'], [calendar[17]])
        self.assertEqual(audit['event_history'][0]['reasons'], ['missing_official_session']*2)
        extra = pd.Timestamp(calendar[32])+pd.offsets.BDay(1)
        self.assertNotIn(str(extra.date()), calendar)
        enriched = {key:value.reindex(value.index.union([extra])) for key,value in panel.items()}
        for value in enriched.values():
            value.loc[extra] = 1e99
        expected, _ = k.build_features(panel, event, calendar)
        actual, _ = k.build_features(enriched, event, calendar)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_missing_symbol_keeps_shape_other_symbols_and_reasons(self):
        panel, calendar, events = panel_fixture()
        values, audit = k.build_features(panel, [(events[1][0],['A','ABSENT'])], calendar)
        self.assertTrue(values['available'][0,0]); self.assertFalse(values['available'][0,1])
        self.assertEqual(audit['event_history'][0]['reasons'], ['available','missing_symbol'])
        self.assertTrue(np.isnan(values['candidate'][0,1])); self.assertTrue(np.isnan(values['control'][0,1]))

    def test_prefix_determinism_and_no_mutation(self):
        panel, calendar, events = panel_fixture(); before = copy.deepcopy(panel)
        full, audit = k.build_features(panel, events, calendar)
        for limit in range(5):
            got, info = k.build_features(panel, events, calendar, limit=limit)
            for key in full:
                np.testing.assert_array_equal(got[key], full[key][:limit])
            self.assertEqual(info['event_history'], audit['event_history'][:limit])
        for key in panel:
            pd.testing.assert_frame_equal(panel[key], before[key])
        empty, info = k.build_features(panel, [], calendar)
        self.assertEqual(empty['candidate'].shape, (0,0)); self.assertEqual(info['events'],0)

    def test_schema_rejects_misalignment_order_duplicates_and_bad_events(self):
        panel, calendar, events = panel_fixture()
        for key in ('adj','close','volume'):
            altered = copy.deepcopy(panel); altered[key] = altered[key].iloc[::-1]
            with self.subTest(key=key), self.assertRaises(ValueError):
                k.build_features(altered, events, calendar)
        altered = copy.deepcopy(panel); altered['volume'] = altered['volume'][['B','A']]
        with self.assertRaises(ValueError):
            k.build_features(altered, events, calendar)
        for bad_calendar in (calendar[::-1], calendar[:3]+[calendar[2]]+calendar[3:], ['2019-01-05']+calendar):
            with self.assertRaises(ValueError):
                k.build_features(panel, events, bad_calendar)
        for bad_events in (events[::-1], [events[1],events[1]], [('2019-01-05',['A','B'])],
                           [(events[1][0],['A','A'])], [(events[1][0],['A']),(events[2][0],['A','B'])]):
            with self.assertRaises(ValueError):
                k.build_features(panel, bad_events, calendar)
        for bad_limit in (-1,5,True,1.5):
            with self.assertRaises(ValueError):
                k.build_features(panel, events, calendar, limit=bad_limit)


if __name__ == '__main__':
    unittest.main()
