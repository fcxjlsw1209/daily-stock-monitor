"""Synthetic-only235 source support, strict gate, causality and API checks."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

import monthly_relative_gate235_kernel as k


def fixture():
    days = pd.bdate_range('2015-01-01', '2016-11-10')
    frame = pd.DataFrame(np.nan, index=days, columns=['AAA', 'BBB', 'FUND', 'SPY'])
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'],
        symbols=[['AAA', 'BBB', 'FUND', 'MISSING'] for _ in range(4)], etfs=['FUND'])
    arrays = dict(eligible=np.array([[True, True, True, False]] * 4),
                  state_weak_narrow=np.array([True, True, False, True]))
    calendar = [str(d.date()) for d in days]
    panel = {'adj': frame}
    for i, spy in ((1, 125.), (3, 75.)):
        start, end = endpoints(meta, calendar, i)
        frame.loc[start, :] = 100.
        frame.loc[end, ['AAA', 'BBB', 'FUND', 'SPY']] = [125., 150., 112.5, spy]
    return panel, meta, calendar, arrays


def endpoints(meta, calendar, i):
    t = calendar.index(meta['dates'][i])
    return pd.Timestamp(calendar[t - 252]), pd.Timestamp(calendar[t - 42])


def run(data, limit=None):
    return k.build_features(*data, limit=limit)


def physical(data, n):
    panel, meta, calendar, arrays = data
    mm = copy.deepcopy(meta)
    mm['dates'], mm['symbols'] = mm['dates'][:n], mm['symbols'][:n]
    aa = {key: value[:n].copy() for key, value in arrays.items()}
    # Price source physically stops at the latest actually needed endpoint.
    active = [i for i in range(n) if bool(arrays['state_weak_narrow'][i])
              and meta['dates'][i] != k.INITIAL_PARTIAL]
    pp = {'adj': panel['adj'].loc[:endpoints(meta, calendar, active[-1])[1]].copy()
          if active else panel['adj'].iloc[:0].copy()}
    cc = calendar[:calendar.index(meta['dates'][n - 1]) + 2] if n else []
    return pp, mm, cc, aa


class GateTests(unittest.TestCase):
    def assert_values(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_strict_equal_excluded_weak_only_initial_unchanged(self):
        values, audit = run(fixture())
        expected = np.array([[1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]], bool)
        np.testing.assert_array_equal(values['candidate_eligible'], expected)
        np.testing.assert_array_equal(values['required'], [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0]])
        np.testing.assert_array_equal(values['momentum'][1, :3], [.25, .5, .125])
        self.assertEqual(values['spy_momentum'][1], .25)
        self.assertEqual(values['spy_momentum'][3], -.25)
        self.assertEqual(values['threshold'][3], 0.)
        self.assertEqual([x['branch'] for x in audit['event_history']], ['initial_partial', 'weak_relative_gate', 'state_inactive', 'weak_relative_gate'])

    def test_exact_adjacent_binary64_no_epsilon(self):
        data = fixture();panel, meta, calendar, _ = data
        _, end = endpoints(meta, calendar, 1)
        for price, accepted in ((125., False), (np.nextafter(125., np.inf), True), (np.nextafter(125., 0.), False)):
            panel['adj'].at[end, 'AAA'] = price
            values, _ = run(data)
            self.assertEqual(values['candidate_eligible'][1, 0], accepted)

    def test_spy_nonpositive_redundant_with_required_positive_E(self):
        data = fixture();panel, meta, calendar, arrays = data
        _, end = endpoints(meta, calendar, 1)
        for price in (100., 90., 1.):
            panel['adj'].at[end, 'SPY'] = price
            values, _ = run(data)
            np.testing.assert_array_equal(values['candidate_eligible'][1], arrays['eligible'][1])

    def test_etf_subject_to_same_gate_and_no_type_heuristic(self):
        data = fixture();values, _ = run(data)
        self.assertFalse(values['candidate_eligible'][1, 2])
        data[1]['etfs'] = ['AAA', 'MISSING', 'anything']
        changed, _ = run(data)
        self.assert_values(values, changed)

    def test_original_E_never_promotes_absent_name(self):
        values, _ = run(fixture())
        self.assertFalse(np.any(values['candidate_eligible'] & ~values['original_eligible']))
        self.assertTrue(np.isnan(values['momentum'][:, 3]).all())
        self.assertFalse(values['required'][:, 3].any())

    def test_named_partial_not_generic_first_row(self):
        data = fixture();panel, meta, calendar, arrays = data
        meta['dates'] = meta['dates'][1:2];meta['symbols'] = meta['symbols'][1:2]
        arrays = {name: value[1:2].copy() for name, value in arrays.items()}
        values, audit = run((panel, meta, calendar, arrays))
        self.assertEqual(audit['active_events'], 1)
        np.testing.assert_array_equal(values['candidate_eligible'][0], [False, True, False, False])

    def test_initial_bypass_with_no_prices_or_warmup(self):
        data = physical(fixture(), 1)
        data[2][:] = ['2016-08-17', '2016-08-18']
        values, audit = run(data)
        self.assertTrue(np.isnan(values['momentum']).all())
        self.assertTrue(np.isnan(values['spy_momentum']).all())
        self.assertEqual(audit['source_price_observations'], 0)
        self.assertEqual(audit['spy_price_observations'], 0)

    def test_bypass_evidence_is_nan_and_audit_json_null(self):
        values, audit = run(fixture())
        for i in (0, 2):
            self.assertTrue(np.isnan(values['momentum'][i]).all())
            for key in ('spy_momentum', 'threshold'):
                self.assertTrue(np.isnan(values[key][i]))
                self.assertIsNone(audit['event_history'][i][key])
            for key in ('source_start', 'source_end'):
                self.assertIsNone(audit['event_history'][i][key])
            self.assertEqual(audit['event_history'][i]['source_positions'], [])
        json.dumps(audit, allow_nan=False)

    def test_exact_counts_and_official_endpoint_indices(self):
        data = fixture();_, audit = run(data)
        expected = dict(events=4, symbols_per_event=4, checked_cells=16,
            original_eligible_cells=12, weak_events=3, active_events=2, required_cells=6,
            source_price_observations=12, spy_price_observations=4, rejected_cells=2,
            candidate_eligible_cells=10, new_model_fits=0, predictive_model_fits=0)
        self.assertEqual({key:audit[key] for key in k.COUNT_FIELDS}, expected)
        for i in (1, 3):
            t = data[2].index(data[1]['dates'][i]);event = audit['event_history'][i]
            self.assertEqual(event['source_positions'], [t - 252, t - 42])
            self.assertEqual(event['source_start'], data[2][t - 252])
            self.assertEqual(event['source_end'], data[2][t - 42])
            self.assertEqual(event['required_indices'], [0, 1, 2])
        self.assertEqual(audit['event_history'][1]['rejected_indices'], [0, 2])

    def test_positive_power_two_source_scaling_preserves_binary_results(self):
        data = fixture();values, audit = run(data)
        for symbol, scale in zip(data[0]['adj'].columns, (2., 16., .25, 8.)):
            data[0]['adj'][symbol] *= scale
        other, other_audit = run(data)
        self.assert_values(values, other);self.assertEqual(audit, other_audit)

    def test_active_empty_E_still_requires_SPY(self):
        data = fixture();data[3]['eligible'][1] = False
        values, audit = run(data)
        self.assertEqual(values['spy_momentum'][1], .25)
        self.assertFalse(values['required'][1].any())
        self.assertEqual(audit['spy_price_observations'], 4)
        _, end = endpoints(data[1], data[2], 1)
        data[0]['adj'].at[end, 'SPY'] = np.nan
        with self.assertRaises(ValueError):run(data)

    def test_spy_candidate_is_independently_required_and_fails_equality(self):
        data = fixture();data[1]['symbols'][1][0] = 'SPY'
        values, audit = run(data, 2)
        self.assertEqual(values['momentum'][1, 0], values['spy_momentum'][1])
        self.assertFalse(values['candidate_eligible'][1, 0])
        self.assertEqual(audit['source_price_observations'], 6)
        self.assertEqual(audit['spy_price_observations'], 2)


class SourceTests(unittest.TestCase):
    assert_values = GateTests.assert_values
    def test_only_two_endpoints_not_full211_history(self):
        data = fixture();expected, audit = run(data)
        start, end = endpoints(data[1], data[2], 1)
        frame = data[0]['adj'].astype(object)
        interior = frame.index[(frame.index > start) & (frame.index < end)]
        # Preserve every required endpoint from any active query.
        protected = {d for i in (1, 3) for d in endpoints(data[1], data[2], i)}
        for day in interior:
            if day not in protected:frame.loc[day] = 'not-required'
        data[0]['adj'] = frame
        actual, aa = run(data)
        self.assert_values(expected, actual);self.assertEqual(audit, aa)

    def test_noneligible_and_inactive_bad_values_are_ignored(self):
        data = fixture();expected, audit = run(data)
        data[0]['adj']['MISSING'] = 'bad'
        start, end = endpoints(data[1], data[2], 2)
        data[0]['adj'] = data[0]['adj'].astype(object)
        data[0]['adj'].loc[[start, end], :] = 'inactive'
        actual, aa = run(data)
        self.assert_values(expected, actual);self.assertEqual(audit, aa)

    def test_invalid_required_numeric_sources_fail_both_endpoints(self):
        for symbol in ('AAA', 'SPY'):
            for endpoint in (0, 1):
                for bad in (0., -1., np.nan, np.inf, -np.inf, True, np.bool_(False), '125', None, 1+0j, 10**400):
                    data = fixture();data[0]['adj'] = data[0]['adj'].astype(object)
                    day = endpoints(data[1], data[2], 1)[endpoint]
                    data[0]['adj'].at[day, symbol] = bad
                    with self.subTest(symbol=symbol, endpoint=endpoint, bad=str(bad)[:30]):
                        with self.assertRaises(ValueError):run(data)

    def test_derived_overflow_aborts_not_unavailable(self):
        for symbol in ('AAA', 'SPY'):
            data = fixture();start, end = endpoints(data[1], data[2], 1)
            data[0]['adj'].at[start, symbol] = np.nextafter(0., 1.)
            data[0]['adj'].at[end, symbol] = np.finfo(float).max
            with self.assertRaises(FloatingPointError):run(data)

    def test_nonpositive_required_momentum_is_source_cache_mismatch(self):
        for last in (100., 50., np.nextafter(100., 0.)):
            data = fixture();_, end = endpoints(data[1], data[2], 1)
            data[0]['adj'].at[end, 'AAA'] = last
            with self.assertRaisesRegex(ValueError, 'positive-momentum mismatch'):run(data)

    def test_missing_required_endpoints_or_symbols_fail(self):
        for symbol in ('SPY', 'AAA'):
            data = fixture();data[0]['adj'] = data[0]['adj'].drop(columns=symbol)
            with self.assertRaises(ValueError):run(data)
        for endpoint in (0, 1):
            data = fixture();day = endpoints(data[1], data[2], 1)[endpoint]
            data[0]['adj'] = data[0]['adj'].drop(index=day)
            with self.assertRaises(ValueError):run(data)

    def test_insufficient_active_history_fails_even_empty_E(self):
        data = fixture();data[2][:] = data[2][-220:]
        data[3]['eligible'][:] = False
        with self.assertRaisesRegex(ValueError, 'insufficient official history'):run(data)


class PrefixTests(unittest.TestCase):
    assert_values = GateTests.assert_values
    def test_every_logical_and_physical_prefix(self):
        data = fixture();values, audit = run(data)
        for n in range(5):
            logical, la = run(data, n);physical_values, pa = run(physical(data, n))
            self.assert_values({key:value[:n] for key,value in values.items()}, logical)
            self.assert_values(logical, physical_values);self.assertEqual(la, pa)
            self.assertEqual(la['event_history'], audit['event_history'][:n])
            self.assertEqual(la['events'], n)
            self.assertEqual(la['symbols_per_event'], 4)

    def test_future_numeric_state_and_metadata_poison_not_consumed(self):
        data = fixture();expected, audit = run(data, 2)
        _, end = endpoints(data[1], data[2], 1)
        data[0]['adj'] = data[0]['adj'].astype(object)
        data[0]['adj'].loc[data[0]['adj'].index > end] = 'future'
        for key in data[3]:
            data[3][key] = data[3][key].astype(object);data[3][key][2:] = 'future'
        data[1]['dates'][2:] = ['not a date', 'future']
        data[1]['symbols'][2:] = [None, ['bad']]
        actual, aa = run(data, 2)
        self.assert_values(expected, actual);self.assertEqual(audit, aa)
        with self.assertRaises(ValueError):run(data)

    def test_nonmutation_and_detached_output(self):
        data = fixture();frame = data[0]['adj'].copy(deep=True)
        arrays = {key:value.copy() for key,value in data[3].items()};meta = copy.deepcopy(data[1])
        values, audit = run(data)
        pd.testing.assert_frame_equal(data[0]['adj'], frame)
        self.assertEqual(data[1], meta)
        for key in arrays:np.testing.assert_array_equal(arrays[key], data[3][key])
        values['original_eligible'][0] = False;values['candidate_eligible'][1] = True
        audit['event_history'][0]['symbols'][0] = 'MUTATED'
        self.assertEqual(data[1], meta)
        for key in arrays:np.testing.assert_array_equal(arrays[key], data[3][key])

    def test_shapes_boolean_types_and_limits_rejected(self):
        for key, replacement in (('eligible', np.ones((4,4),int)), ('eligible', np.ones(4,bool)), ('eligible', np.empty((4,0),bool)), ('state_weak_narrow', np.ones((4,1),bool)), ('state_weak_narrow', np.ones(3,bool)), ('state_weak_narrow', np.ones(4,int))):
            data = fixture();data[3][key] = replacement
            with self.subTest(key=key, shape=replacement.shape):
                with self.assertRaises(ValueError):run(data)
        for limit in (-1, 5, True, 1., '1'):
            with self.assertRaises(ValueError):run(fixture(), limit)

    def test_calendar_dates_duplicates_order_and_monthend_witness(self):
        for change in ('duplicate', 'reverse', 'weekend', 'timestamp', 'noncanonical', 'empty'):
            data = fixture()
            if change=='duplicate':data[2][1] = data[2][0]
            elif change=='reverse':data[2].reverse()
            elif change=='weekend':data[2][1] = '2015-01-03'
            elif change=='timestamp':data[2][1] = pd.Timestamp(data[2][1])
            elif change=='noncanonical':data[2][0] = '2015-1-1'
            else:data[2].clear()
            with self.subTest(change=change):
                with self.assertRaises(ValueError):run(data)
        data = physical(fixture(), 2);data[2].pop()
        with self.assertRaisesRegex(ValueError, 'month-end'):run(data)

    def test_query_metadata_axes_dates_and_ids_fail(self):
        for change in ('date_count','symbol_count','width','duplicate','blank','event_order','not_monthend','noncanonical','partial_later'):
            data = fixture();meta = data[1]
            if change=='date_count':meta['dates'].pop()
            elif change=='symbol_count':meta['symbols'].pop()
            elif change=='width':meta['symbols'][1].pop()
            elif change=='duplicate':meta['symbols'][1][1] = 'AAA'
            elif change=='blank':meta['symbols'][1][1] = ' '
            elif change=='event_order':meta['dates'][2] = meta['dates'][1]
            elif change=='not_monthend':meta['dates'][1] = '2016-08-30'
            elif change=='noncanonical':meta['dates'][0] = '2016-8-17'
            else:meta['dates'][1] = '2016-08-17'
            with self.subTest(change=change):
                with self.assertRaises(ValueError):run(data)

    def test_price_axes_and_panel_schema_fail(self):
        for change in ('extra','notframe','notdatetime','duplicate_date','reverse','timezone','time','duplicate_column','blank_column'):
            data = fixture();frame = data[0]['adj']
            if change=='extra':data[0]['sectors'] = frame
            elif change=='notframe':data[0]['adj'] = frame.to_numpy()
            elif change=='notdatetime':frame.index = frame.index.astype(str)
            elif change=='duplicate_date':frame.index = pd.DatetimeIndex([frame.index[0]]+frame.index[:-1].tolist())
            elif change=='reverse':data[0]['adj'] = frame.iloc[::-1]
            elif change=='timezone':frame.index = frame.index.tz_localize('UTC')
            elif change=='time':frame.index = frame.index + pd.Timedelta(hours=1)
            elif change=='duplicate_column':frame.columns = ['AAA','AAA','FUND','SPY']
            else:frame.columns = ['AAA','BBB','FUND',' ']
            with self.subTest(change=change):
                with self.assertRaises((ValueError,TypeError)):run(data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
