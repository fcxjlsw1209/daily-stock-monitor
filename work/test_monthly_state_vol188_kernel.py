"""Synthetic-only source, timing, state and midpoint-CDF checks."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_state_vol188_kernel as k


def fixture(count=3, width=2):
    index = pd.bdate_range('2008-01-01', periods=1160)
    x = np.arange(len(index)-1, dtype=float)
    spy_scale = np.where((x.astype(int)//180)%2 == 0, .002, .014)
    spy_r = spy_scale * np.sin(x*1.37) + .0002
    own_r = .0004 + (.008 + .003*np.sin(x/110.))*np.cos(x*.67)
    panel = {'adj': pd.DataFrame({
        'AAA': np.r_[100., 100.*np.cumprod(1.+own_r)],
        'ETF': np.r_[80., 80.*np.cumprod(1.+.001+ .009*np.sin(x*.89))],
        'SPY': np.r_[200., 200.*np.cumprod(1.+spy_r)]}, index=index)}
    symbols = ['AAA', 'ETF'][:width]
    dates = [str(index[p].date()) for p in [1000, 1050, 1100][:count]]
    meta = {'dates': dates, 'symbols': [symbols.copy() for _ in dates], 'etfs': ['ETF']}
    return panel, meta, [str(d.date()) for d in index]


def one_event(panel, position, symbols=('AAA', 'ETF')):
    return {'dates': [str(panel['adj'].index[position].date())],
            'symbols': [list(symbols)], 'etfs': ['ETF']}


class ReferenceTests(unittest.TestCase):
    def test_exact_midpoint_ties_and_matched_support(self):
        risks = [1.]*126 + [2.]*126 + [3.]*126 + [4.]*378
        states = [True]*378 + [False]*378
        result = k.measure_reference(2., True, risks, states)
        self.assertTrue(result['available'])
        self.assertEqual(result['matched_count'], 378)
        self.assertEqual(result['reference_below'], 126)
        self.assertEqual(result['reference_equal'], 126)
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['control'], -.25)

    def test_common_support_minima_inclusive(self):
        risks = [1.]*252 + [None]*504
        states = [True]*63 + [False]*189 + [None]*504
        self.assertTrue(k.measure_reference(1., True, risks, states)['available'])
        states[62] = False
        result = k.measure_reference(1., True, risks, states)
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], 'insufficient_matched_history')
        self.assertTrue(math.isnan(result['control']))
        risks[251] = None
        result = k.measure_reference(1., True, risks, states)
        self.assertEqual(result['reason'], 'insufficient_joint_history')

    def test_zero_volatility_is_valid_exact_tie(self):
        result = k.measure_reference(0., True, [0.]*756, [True]*756)
        self.assertTrue(result['available'])
        self.assertEqual(result['candidate'], 0.)
        self.assertEqual(result['control'], 0.)
        self.assertEqual(result['reference_equal'], 756)

    def test_bounds_and_unknown_state_not_false(self):
        for current, expected in [(0., -.5), (2., .5)]:
            result = k.measure_reference(current, False, [1.]*756, [False]*756)
            self.assertEqual(result['candidate'], expected)
            self.assertEqual(result['control'], expected)
        result = k.measure_reference(1., None, [1.]*756, [False]*756)
        self.assertEqual(result['reference_count'], 756)
        self.assertEqual(result['matched_count'], 0)
        self.assertEqual(result['reason'], 'current_market_state_unavailable')
        self.assertIsNone(result['matched_equal'])

    def test_missing_current_preserves_reference_counts(self):
        result = k.measure_reference(None, True, [1.]*756, [True]*756)
        self.assertEqual(result['reference_count'], 756)
        self.assertEqual(result['matched_count'], 756)
        self.assertIsNone(result['reference_below'])
        self.assertEqual(result['reason'], 'current_volatility_unavailable')

    def test_strict_calm_boundary(self):
        sigma = .10 / math.sqrt(252.)
        self.assertFalse(k._calm(sigma)[0])
        self.assertTrue(k._calm(sigma*.999999)[0])
        self.assertFalse(k._calm(sigma*1.000001)[0])
        self.assertEqual(k._calm(0.), (True, 0.))
        self.assertEqual(k._calm(None), (None, None))

    def test_invalid_reference_contract(self):
        with self.assertRaises(ValueError): k.measure_reference(1., True, [1.]*755, [True]*755)
        with self.assertRaises(ValueError): k.measure_reference(1., 1, [1.]*756, [True]*756)
        with self.assertRaises(ValueError): k.measure_reference(1., True, [-1.]*756, [True]*756)


class SourceTests(unittest.TestCase):
    def test_exact_windows_current_excluded_and_pandas_counts(self):
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar)
        r = panel['adj'].pct_change(fill_method=None)
        own = r.rolling(126, min_periods=126).std(ddof=1)
        spy = r.SPY.rolling(63, min_periods=63).std(ddof=1)
        states = spy*math.sqrt(252.) < .10
        for i, event in enumerate(audit['event_history']):
            t = event['session_position']
            self.assertEqual(event['reference_start'], calendar[t-756])
            self.assertEqual(event['reference_end'], calendar[t-1])
            self.assertEqual(event['own_price_start'], calendar[t-126])
            self.assertEqual(event['market_price_start'], calendar[t-63])
            for j, symbol in enumerate(meta['symbols'][i]):
                hist = own[symbol].iloc[t-756:t]
                current = own[symbol].iloc[t]
                valid = hist.notna() & spy.iloc[t-756:t].notna()
                matched = valid & (states.iloc[t-756:t] == states.iloc[t])
                a = hist.loc[matched]
                b = hist.loc[valid]
                self.assertEqual(values['candidate'][i,j], ((a<current).sum()+.5*(a==current).sum())/len(a)-.5)
                self.assertEqual(values['control'][i,j], ((b<current).sum()+.5*(b==current).sum())/len(b)-.5)
        self.assertEqual(audit['reference_observations'], 3*2*756)
        self.assertEqual(audit['predictive_model_fits'], 0)
        self.assertFalse(audit['ranking_performed'])

    def test_constant_panel_and_etf_same_treatment(self):
        panel, meta, calendar = fixture()
        panel['adj'].loc[:,:] = 100.
        values, audit = k.build_features(panel, meta, calendar)
        self.assertTrue(values['available'].all())
        np.testing.assert_array_equal(values['candidate'], np.zeros((3,2)))
        np.testing.assert_array_equal(values['control'], np.zeros((3,2)))
        self.assertTrue(all(e['current_calm'] is True for e in audit['event_history']))

    def test_insufficient_full_calendar_history_not_shortened(self):
        panel, _, calendar = fixture()
        meta = one_event(panel, 755)
        values, audit = k.build_features(panel, meta, calendar)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reference_observations'], 0)
        self.assertEqual(audit['reason_counts'], {'insufficient_history':2})
        self.assertIsNotNone(audit['event_history'][0]['cells'][0]['current_sigma'])
        meta = one_event(panel, 756)
        values, audit = k.build_features(panel, meta, calendar)
        self.assertEqual(audit['event_history'][0]['reference_slot_count'], 756)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reference_count'], 630)

    def test_missing_official_row_matches_explicit_nan_without_compression(self):
        panel, meta, calendar = fixture(count=1)
        index = panel['adj'].index
        missing = {'adj': panel['adj'].drop(index[650])}
        explicit = copy.deepcopy(panel)
        explicit['adj'].loc[index[650],:] = np.nan
        va, aa = k.build_features(missing, meta, calendar)
        vb, ab = k.build_features(explicit, meta, calendar)
        for key in va: np.testing.assert_array_equal(va[key], vb[key])
        self.assertEqual(aa['event_history'][0]['missing_official_sessions'], [calendar[650]])
        self.assertEqual(aa['event_history'][0]['cells'], ab['event_history'][0]['cells'])
        self.assertEqual(aa['event_history'][0]['cells'][0]['reference_count'], 629)

    def test_invalid_prices_are_missing_adjacent_returns(self):
        panel, meta, calendar = fixture(count=1)
        for invalid in [0., -1., np.inf, -np.inf, np.nan]:
            a = copy.deepcopy(panel)
            a['adj'].iloc[1000,0] = invalid
            values, audit = k.build_features(a, meta, calendar)
            self.assertFalse(values['available'][0,0])
            self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'current_volatility_unavailable')
            self.assertTrue(values['available'][0,1])

    def test_missing_spy_is_not_noncalm_and_missing_member_neutral(self):
        panel, meta, calendar = fixture(count=1)
        values, audit = k.build_features({'adj':panel['adj'].drop(columns='SPY')}, meta, calendar)
        self.assertFalse(values['available'].any())
        self.assertIsNone(audit['event_history'][0]['current_calm'])
        self.assertEqual(audit['reason_counts'], {'current_market_state_unavailable':2})
        values, audit = k.build_features({'adj':panel['adj'].drop(columns='AAA')}, meta, calendar)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'], 'missing_symbol')
        self.assertTrue(values['available'][0,1])

    def test_all_prefixes_future_poison_and_empty_no_numeric_reads(self):
        panel, meta, calendar = fixture()
        full, full_a = k.build_features(panel, meta, calendar)
        for n in range(4):
            poison = copy.deepcopy(panel)
            poison['adj'] = poison['adj'].astype(object)
            after = 0 if not n else [1000,1050,1100][n-1]+1
            poison['adj'].iloc[after:,:] = 'future unreadable'
            values, audit = k.build_features(poison, meta, calendar, limit=n)
            for key in values: np.testing.assert_array_equal(values[key], full[key][:n])
            self.assertEqual(audit['event_history'], full_a['event_history'][:n])
            self.assertEqual(values['available'].shape, (n,2))

    def test_unused_columns_and_precalendar_rows_not_consumed(self):
        panel, meta, calendar = fixture(count=1)
        expected = k.build_features(panel, meta, calendar)
        modified = copy.deepcopy(panel)
        modified['adj']['UNUSED'] = 'not numeric'
        before = pd.DataFrame('not numeric', index=[pd.Timestamp('2007-12-31')], columns=modified['adj'].columns)
        modified['adj'] = pd.concat([before, modified['adj']])
        actual = k.build_features(modified, meta, calendar)
        for key in expected[0]: np.testing.assert_array_equal(actual[0][key], expected[0][key])
        self.assertEqual(actual[1], expected[1])

    def test_scale_invariance_and_inputs_unchanged(self):
        panel, meta, calendar = fixture()
        snapshot = copy.deepcopy(panel)
        values, audit = k.build_features(panel, meta, calendar)
        scaled = {'adj':panel['adj'] * 8.}
        actual, other = k.build_features(scaled, meta, calendar)
        for key in values: np.testing.assert_array_equal(actual[key], values[key])
        pd.testing.assert_frame_equal(panel['adj'], snapshot['adj'])
        self.assertEqual(audit, other)

    def test_bad_schema_and_past_nonnumeric(self):
        panel, meta, calendar = fixture()
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar, limit=True)
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar, limit=4)
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar[::-1])
        bad = copy.deepcopy(meta); bad.pop('etfs')
        with self.assertRaises(ValueError): k.build_features(panel, bad, calendar)
        bad = copy.deepcopy(meta); bad['symbols'][1] = ['AAA']
        with self.assertRaises(ValueError): k.build_features(panel, bad, calendar)
        panel['adj'] = panel['adj'].astype(object); panel['adj'].iloc[0,0] = 'invalid history'
        with self.assertRaises(ValueError): k.build_features(panel, meta, calendar)

    def test_valid_input_arithmetic_overflow_fails(self):
        panel, meta, calendar = fixture(count=1)
        panel['adj'].iloc[500,0] = 1e-300
        panel['adj'].iloc[501,0] = 1e300
        with self.assertRaises(FloatingPointError): k.build_features(panel, meta, calendar)
        panel, meta, calendar = fixture(count=1)
        panel['adj'].iloc[500,0] = 1e200
        with self.assertRaises(FloatingPointError): k.build_features(panel, meta, calendar)

    def test_independent_full_audit_and_prefix_cross(self):
        import monthly_state_vol188_independent as oracle
        panel, meta, calendar = fixture()
        values, audit = k.build_features(panel, meta, calendar)
        self.assertTrue(oracle.verify_features(panel, meta, calendar, values, audit)['passed'])
        for n in range(4):
            actual, record = k.build_features(panel, meta, calendar, limit=n)
            expected, reference = oracle.independent_features(panel, meta, calendar, limit=n)
            for key in actual: np.testing.assert_array_equal(actual[key], expected[key])
            oracle.record_check(record, reference)
        for missing in ('SPY', 'AAA'):
            case = {'adj':panel['adj'].drop(columns=missing)}
            actual, record = k.build_features(case, meta, calendar)
            self.assertTrue(oracle.verify_features(case, meta, calendar, actual, record)['passed'])
        case = {'adj':panel['adj'].drop(index=panel['adj'].index[650])}
        actual, record = k.build_features(case, meta, calendar)
        self.assertTrue(oracle.verify_features(case, meta, calendar, actual, record)['passed'])
        from test_monthly_state_vol188_independent import fixture as independent_fixture
        other_panel, other_meta, other_calendar = independent_fixture()
        actual, record = k.build_features(other_panel, other_meta, other_calendar)
        self.assertTrue(oracle.verify_features(other_panel, other_meta, other_calendar, actual, record)['passed'])


if __name__ == '__main__':
    unittest.main()
