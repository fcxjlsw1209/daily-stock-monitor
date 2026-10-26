"""Synthetic-only checks for state-conditioned skip42 daily drift."""
from copy import deepcopy
import hashlib
import math
import unittest

import numpy as np
import pandas as pd

import monthly_market_drift231_model as m


def fixture():
    dates = pd.bdate_range('2013-01-01', '2019-02-01')
    t = np.arange(len(dates), dtype=float)
    adj = pd.DataFrame({'A': 100*np.exp(.0003*t+.04*np.sin(t/19)),
        'B': 80*np.exp(.0001*t+.05*np.cos(t/23)),
        'C': 60*np.exp(.0002*t+.03*np.sin(t/7)),
        'SPY': 100*np.exp(.0001*t+.12*np.sin(t/53))}, index=dates)
    signals = ['2016-08-17', '2016-08-31', '2017-01-31', '2018-06-29']
    meta = dict(dates=signals, ends=['2016-08-31', '2016-09-30', '2017-02-28', '2018-07-31'],
                symbols=[['A', 'B', 'C'] for _ in signals], etfs=[])
    arrays = dict(eligible=np.ones((len(signals), 3), bool),
                  state_calm10=np.asarray([False, True, False, True]))
    return {'adj': adj}, meta, list(dates.strftime('%Y-%m-%d')), arrays


def physical_prefix(args, n):
    panel, meta, calendar, arrays = deepcopy(args)
    cutoff = pd.Timestamp(meta['dates'][n-1]) if n else None
    panel['adj'] = panel['adj'].loc[:cutoff].copy() if n else panel['adj'].iloc[:0].copy()
    for key in ('dates', 'ends', 'symbols'):
        meta[key] = meta[key][:n]
    arrays = {key: value[:n].copy() for key, value in arrays.items()}
    return panel, meta, calendar, arrays


def one_query(*, calm=False, flat_market=True):
    panel, meta, calendar, arrays = fixture()
    for key in ('dates', 'ends', 'symbols'):
        meta[key] = meta[key][-1:]
    arrays = dict(eligible=np.ones((1, 3), bool), state_calm10=np.array([calm]))
    if flat_market:
        panel['adj']['SPY'] = 64.
    return panel, meta, calendar, arrays


def assert_same(test, a, b):
    test.assertEqual(set(a), set(b))
    for key in a:
        test.assertEqual(a[key].dtype, b[key].dtype)
        np.testing.assert_array_equal(a[key], b[key])


class ScalarTests(unittest.TestCase):
    def test_strict_market_state_and_exact_equality(self):
        p = [64.]*200
        self.assertEqual(m.market_state(p), (False, 64.))
        p[-1] = 65.
        self.assertEqual(m.market_state(p), (True, math.fsum(p)/200))
        p[-1] = 63.
        self.assertFalse(m.market_state(p)[0])

    def test_exact42_matching_and_fixed63_shrink(self):
        p = np.exp(.01*np.sin(np.arange(211)/9)+.001*np.arange(211))
        mask = np.zeros(210, bool); mask[:42] = True
        result = m.drift(p, mask)
        r = [math.log(float(p[i+1]))-math.log(float(p[i])) for i in range(210)]
        mu, conditional = math.fsum(r)/210, math.fsum(r[:42])/42
        self.assertEqual(result['control_raw'], mu)
        self.assertEqual(result['candidate_raw'], mu+(42/(42+63.))*(conditional-mu))
        mask[41] = False
        with self.assertRaises(ValueError):m.drift(p, mask)

    def test_all_matching_and_equal_means_copy_exactly(self):
        p = np.asarray([2.**(i%3) for i in range(253)])
        result = m.drift(p, np.ones(252, bool))
        self.assertEqual(result['candidate_raw'], result['control_raw'])
        p[:] = 4.; mask = np.zeros(252, bool); mask[:42] = True
        result = m.drift(p, mask)
        self.assertEqual(result['candidate_raw'], 0.)
        self.assertEqual(result['candidate_raw'], result['control_raw'])

    def test_extreme_positive_prices_use_stable_log_difference(self):
        p = np.asarray([1e-300 if i%2 == 0 else 1e300 for i in range(211)])
        result = m.drift(p, np.ones(210, bool))
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        self.assertEqual(result['control_raw'], 0.)

    def test_scalar_shape_type_and_invalid_source(self):
        for p in ([1.]*199, [1.]*201):
            with self.assertRaises(ValueError):m.market_state(p)
        for bad in (0., -1., np.nan, np.inf, True, '1', None):
            p = [1.]*200; p[12] = bad
            with self.assertRaises(ValueError):m.market_state(p)
        with self.assertRaises(ValueError):m.drift(np.ones(211), np.ones(210, int))
        with self.assertRaises(ValueError):m.drift(np.ones(210), np.ones(210, bool))
        with self.assertRaises(ValueError):m.drift(np.ones(64), np.ones(63, bool))

    def test_finite_market_sum_overflow_fails(self):
        with self.assertRaises(FloatingPointError):m.market_state([1e308]*200)
        with self.assertRaises(FloatingPointError):m._mean([np.nextafter(0., 1.), 0.])

    def test_ties_bounds_and_constant_effect(self):
        raw = np.asarray([1., 1., 2., 3.])
        expected = np.asarray([(.5/3)-.5, (.5/3)-.5, (2./3)-.5, .5])
        np.testing.assert_array_equal(m.effect_ranks(raw), expected)
        np.testing.assert_array_equal(m.effect_ranks(np.ones(3)), np.zeros(3))
        with self.assertRaises(ValueError):m.effect_ranks([1.])
        with self.assertRaises(FloatingPointError):m.effect_ranks([1., np.inf])


class BuildTests(unittest.TestCase):
    def test_fixture_output_contract_and_counters(self):
        built = m.build(*fixture()); arrays, audit = built['arrays'], built['audit']
        self.assertEqual(set(arrays), set(m.ARRAY_KEYS))
        self.assertEqual(arrays['ready'].dtype, bool)
        self.assertFalse(arrays['ready'][0]); self.assertTrue(arrays['ready'][1:].all())
        self.assertEqual(set(audit['counts']), set(m.COUNT_FIELDS))
        for key in m.COUNT_FIELDS:
            self.assertEqual(audit['counts'][key], sum(e['counts'][key] for e in audit['event_history']))
        self.assertEqual(audit['counts']['query_months'], 4)
        self.assertEqual(audit['counts']['supported_months'], 3)
        self.assertEqual(audit['counts']['drift_estimates'], 18)
        self.assertEqual(audit['counts']['market_price_observations'],
                         200*audit['counts']['market_state_evaluations'])
        self.assertEqual(audit['new_model_fits'], 0)

    def test_two_window_lengths_and_lagged_state_dates(self):
        args = fixture(); sessions = pd.DatetimeIndex(args[2]); built = m.build(*args)
        for event in built['audit']['event_history'][1:]:
            t = sessions.get_loc(event['date']); L = 210 if event['calm'] else 252
            self.assertEqual(event['formation_returns'], L)
            self.assertEqual(event['formation_start'], str(sessions[t-42-L].date()))
            self.assertEqual(event['formation_end'], str(sessions[t-42].date()))
            self.assertEqual(event['historical_state_first'], event['formation_start'])
            self.assertEqual(event['historical_state_last'], str(sessions[t-43].date()))
            spy = args[0]['adj']['SPY']
            states = [float(spy.iloc[j]) > math.fsum(float(x) for x in spy.iloc[j-199:j+1])/200
                      for j in range(t-42-L, t-42)]
            query = float(spy.iloc[t]) > math.fsum(float(x) for x in spy.iloc[t-199:t+1])/200
            self.assertEqual(event['historical_up_days'], sum(states))
            self.assertEqual(event['matching_state_days'], sum(x == query for x in states))

    def test_current_market_state_can_change_but_stock_recent42_cannot(self):
        args = one_query(flat_market=False); before = m.build(*args)
        date = pd.Timestamp(args[1]['dates'][0]); idx = args[0]['adj'].index; t = idx.get_loc(date)
        changed = deepcopy(args)
        changed[0]['adj'].loc[idx[t-41]:date, ['A', 'B', 'C']] = np.nan
        assert_same(self, before['arrays'], m.build(*changed)['arrays'])
        current_market = deepcopy(args)
        oldstate = before['audit']['event_history'][0]['query_state_up']
        current_market[0]['adj'].loc[date, 'SPY'] = 1. if oldstate else 10000.
        after = m.build(*current_market)
        self.assertNotEqual(after['audit']['event_history'][0]['query_state_up'], oldstate)
        self.assertEqual(after['audit']['event_history'][0]['historical_up_days'],
                         before['audit']['event_history'][0]['historical_up_days'])
        self.assertFalse(np.array_equal(after['arrays']['candidate_raw'], before['arrays']['candidate_raw']))

    def test_initial_and_small_domain_do_not_read_prices(self):
        args = fixture(); args[0]['adj'].loc[:, :] = np.nan
        initial = m.build(*args, limit=1)
        self.assertEqual(initial['audit']['event_history'][0]['status'], 'initial_partial')
        self.assertEqual(initial['audit']['counts']['checked_stock_cells'], 0)
        args = one_query(); args[0]['adj'].loc[:, :] = np.nan
        args[3]['eligible'][0] = [True, False, False]
        built = m.build(*args)
        self.assertEqual(built['audit']['event_history'][0]['status'], 'fewer_than_two_eligible')
        self.assertEqual(built['audit']['counts']['market_state_evaluations'], 0)

    def test_constant_supported_zero_effect_not_fallback(self):
        args = one_query(); args[0]['adj'].loc[:, ['A', 'B', 'C']] = 32.
        built = m.build(*args)
        self.assertTrue(built['arrays']['ready'][0])
        for arm in ('candidate', 'control'):
            np.testing.assert_array_equal(built['arrays'][arm+'_effect'], np.zeros((1, 3)))
            np.testing.assert_array_equal(built['arrays'][arm+'_raw'], np.zeros((1, 3)))
        self.assertEqual(built['audit']['event_history'][0]['status'], 'supported')

    def test_one_bad_stock_falls_back_whole_month_still_checks_all(self):
        args = one_query(); date = pd.Timestamp(args[1]['dates'][0]); t = args[0]['adj'].index.get_loc(date)
        args[0]['adj'].iloc[t-100, 1] = np.nan
        built = m.build(*args); event = built['audit']['event_history'][0]
        self.assertEqual(event['status'], 'invalid_stock_source')
        self.assertEqual(event['counts']['checked_stock_cells'], 3)
        self.assertEqual(event['counts']['source_valid_stock_cells'], 2)
        self.assertEqual(event['counts']['drift_estimates'], 0)
        self.assertTrue(np.isnan(built['arrays']['control_raw']).all())
        self.assertFalse(built['arrays']['ready'][0])
        np.testing.assert_array_equal(built['arrays']['candidate_effect'], np.zeros((1, 3)))

    def test_ineligible_invalid_prices_are_not_consumed(self):
        args = one_query(); args[3]['eligible'][0, 1] = False
        args[0]['adj']['B'] = 'poison'
        built = m.build(*args)
        self.assertTrue(built['arrays']['ready'][0])
        self.assertTrue(np.isnan(built['arrays']['candidate_raw'][0, 1]))
        self.assertEqual(built['arrays']['candidate_effect'][0, 1], 0.)
        self.assertEqual([c['symbol'] for c in built['audit']['event_history'][0]['cells']], ['A', 'C'])

    def test_insufficient_state_support_and_source_priority(self):
        args = one_query(); date = pd.Timestamp(args[1]['dates'][0]); args[0]['adj'].loc[date, 'SPY'] = 128.
        built = m.build(*args); event = built['audit']['event_history'][0]
        self.assertEqual(event['matching_state_days'], 0)
        self.assertEqual(event['status'], 'insufficient_matching_state')
        self.assertEqual(event['counts']['source_valid_stock_cells'], 3)
        self.assertEqual(event['counts']['stock_log_returns'], 0)
        args[0]['adj']['A'] = np.nan
        self.assertEqual(m.build(*args)['audit']['event_history'][0]['status'], 'invalid_stock_source')

    def test_market_missing_and_invalid_reasons(self):
        for bad, reason in ((None, 'missing_market_symbol'), (np.nan, 'invalid_market_price'),
                            ('oops', 'nonnumeric_market_price')):
            args = one_query()
            if bad is None:args[0]['adj'] = args[0]['adj'].drop(columns='SPY')
            else:args[0]['adj']['SPY'] = bad
            built = m.build(*args)
            self.assertEqual(built['audit']['event_history'][0]['status'], reason)
            self.assertEqual(built['audit']['counts']['checked_stock_cells'], 0)
            self.assertEqual(built['audit']['counts']['market_state_evaluations'], 0)

    def test_source_calendar_gap_is_not_silently_shortened(self):
        args = one_query(); frame = args[0]['adj']; t = frame.index.get_loc(args[1]['dates'][0])
        args[0]['adj'] = frame.drop(index=frame.index[t-150])
        built = m.build(*args)
        self.assertEqual(built['audit']['event_history'][0]['status'], 'missing_market_session')
        self.assertFalse(built['arrays']['ready'][0])

    def test_insufficient_calendar_has_no_partial_clock_or_estimation(self):
        args = one_query(); args[1]['dates'] = ['2013-01-31']
        built = m.build(*args); event = built['audit']['event_history'][0]
        self.assertEqual(event['status'], 'insufficient_calendar')
        for key in ('formation_start', 'formation_end', 'historical_state_first', 'historical_state_last'):
            self.assertIsNone(event[key])

    def test_all_logical_and_physical_prefixes(self):
        args = fixture(); full = m.build(*args)
        for n in range(5):
            logical = m.build(*args, limit=n); physical = m.build(*physical_prefix(args, n))
            assert_same(self, logical['arrays'], physical['arrays'])
            self.assertEqual(logical['audit'], physical['audit'])
            for key in m.ARRAY_KEYS:
                np.testing.assert_array_equal(logical['arrays'][key], full['arrays'][key][:n])
            self.assertEqual(logical['audit']['event_history'], full['audit']['event_history'][:n])

    def test_future_price_and_identity_poison_unused_prefix(self):
        args = fixture(); original = m.build(*args, limit=2)
        cutoff = args[1]['dates'][1]
        args[0]['adj'].loc[args[0]['adj'].index > cutoff, :] = np.nan
        args[1]['symbols'][2:] = [None, ['bad']]
        args[1]['dates'][2:] = ['bad', 'bad']
        after = m.build(*args, limit=2)
        assert_same(self, original['arrays'], after['arrays'])
        self.assertEqual(original['audit'], after['audit'])

    def test_empty_physical_width_and_calm_column(self):
        args = physical_prefix(fixture(), 0)
        args[3]['eligible'] = np.zeros((0, 0), bool)
        built = m.build(*args)
        self.assertEqual(built['arrays']['candidate_raw'].shape, (0, 0))
        self.assertEqual(built['audit']['counts'], dict.fromkeys(m.COUNT_FIELDS, 0))
        args = one_query(); a = m.build(*args)
        args[3]['state_calm10'] = args[3]['state_calm10'][:, None]
        b = m.build(*args); self.assertEqual(a['audit'], b['audit'])
        assert_same(self, a['arrays'], b['arrays'])

    def test_metadata_shapes_types_and_monthend_fail_hard(self):
        for kind in range(8):
            args = one_query()
            if kind == 0:args[3]['eligible'] = np.ones((1, 3), int)
            elif kind == 1:args[3]['state_calm10'] = np.ones((1, 3), bool)
            elif kind == 2:args[1]['symbols'][0] = ['A', 'A', 'C']
            elif kind == 3:args[1]['dates'][0] = '2018-06-28'
            elif kind == 4:args[1]['ends'] = []
            elif kind == 5:args[2].append(args[2][-1])
            elif kind == 6:args[0]['volume'] = args[0]['adj']
            else:args[0]['adj'] = args[0]['adj'].iloc[::-1]
            with self.assertRaises(ValueError):m.build(*args)
        with self.assertRaises(ValueError):m.build(*one_query(), limit=True)
        with self.assertRaises(ValueError):m.build(*one_query(), limit=2)

    def test_inputs_remain_unchanged(self):
        args = fixture(); before = deepcopy(args)
        m.build(*args)
        pd.testing.assert_frame_equal(args[0]['adj'], before[0]['adj'])
        self.assertEqual(args[1], before[1]); self.assertEqual(args[2], before[2])
        for key in args[3]:np.testing.assert_array_equal(args[3][key], before[3][key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
