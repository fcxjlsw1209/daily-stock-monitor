"""Synthetic-only exact-state, matured-label and mapping tests for round189."""
import copy
import math
from fractions import Fraction
import unittest

import numpy as np
import pandas as pd

import monthly_sector_transition189_kernel as k


def prices_for_states(states, query=1):
    if len(states) != 60:
        raise ValueError('Exactly60 historical states')
    levels = np.zeros(71)
    for j, state in enumerate(states):
        levels[j+10] = levels[j] + state
    levels[70] = levels[60] + query
    coordinates = np.arange(1471)
    levels = np.interp(coordinates, np.arange(0,1471,21), levels)
    return np.exp2(levels), np.ones(1471)


def fixture():
    dates = pd.bdate_range('2008-01-01', periods=1730)
    x = np.arange(len(dates), dtype=float)
    adj = pd.DataFrame({
        'AAA': 100.*np.exp(.0003*x+.2*np.sin(x/140.)+.05*np.sin(x/14.)),
        'BBB': 80.*np.exp(.0004*x+.19*np.cos(x/130.)+.03*np.cos(x/17.)),
        'FUND': 70.*np.exp(.0003*x+.1*np.sin(x/130.)),
        'XLK': 90.*np.exp(.00025*x+.05*np.sin(x/95.)),
        'XLY': 95.*np.exp(.0002*x+.1*np.cos(x/90.)),
    }, index=dates)
    sectors = pd.DataFrame('Technology', index=dates, columns=adj.columns)
    meta = {'dates':[str(dates[t].date()) for t in (1550,1600,1650)],
            'symbols':[['AAA','BBB','FUND'] for _ in range(3)], 'etfs':['FUND']}
    return {'adj':adj,'sectors':sectors}, meta, [str(d.date()) for d in dates]


def mapper(day, symbols, sectors, etfs):
    day = pd.Timestamp(day)
    if len(sectors) and sectors.index[-1] > day:
        raise AssertionError('Future sector rows exposed')
    answer = {}
    for symbol in symbols:
        if symbol in etfs:
            answer[symbol] = None
        elif symbol == 'BABA':
            answer[symbol] = 'XLY'
        else:
            value = sectors.at[day,symbol] if day in sectors.index and symbol in sectors else None
            answer[symbol] = {'Technology':'XLK','Consumer Cyclical':'XLY'}.get(value)
    return answer


class MeasureTests(unittest.TestCase):
    def test_exact_rational_sign_beats_rounded_ratios(self):
        eps = 2.**-52
        a,b,c,d = 1.+eps,1.,1.,1.-eps
        self.assertEqual(a/b,c/d)
        self.assertEqual(a*d,c*b)
        self.assertEqual(k._relative_sign(a,b,c,d),-1)
        self.assertEqual(k._relative_sign(c,d,a,b),1)
        self.assertEqual(k._relative_sign(8.,4.,2.,1.),0)

    def test_extreme_rational_comparison_without_float_product(self):
        values = (1e308,1e-308,1e307,1e-307)
        self.assertEqual(k._relative_sign(*values),1)
        expected = Fraction.from_float(values[0])*Fraction.from_float(values[3])
        other = Fraction.from_float(values[2])*Fraction.from_float(values[1])
        self.assertEqual(k._relative_sign(*values),1 if expected>other else -1)
        for a,b in [(1e308,1e-308),(1e-308,1e308),(1.,np.nextafter(1.,2.))]:
            self.assertTrue(math.isfinite(k._log_ratio(a,b)))
            self.assertAlmostEqual(k._log_ratio(a,b),math.log(a)-math.log(b),delta=3e-13)

    def test_exact_offsets_and_state_labels(self):
        states = [1,-1,0]*20
        stock, sector = prices_for_states(states, query=0)
        out = k.measure(stock,sector)
        self.assertEqual(out['states'],states)
        self.assertEqual(out['query_state'],0)
        self.assertEqual(out['state_start_indices'],list(range(0,1260,21)))
        self.assertEqual(out['predictor_indices'],list(range(210,1470,21)))
        self.assertEqual(out['label_end_indices'],list(range(231,1471,21)))
        self.assertEqual(out['matching_indices'],list(range(2,60,3)))
        self.assertEqual(out['matching_count'],20)
        expected = [math.log(stock[b]/stock[a]) for a,b in zip(out['predictor_indices'],out['label_end_indices'])]
        np.testing.assert_allclose(out['labels'],expected,rtol=0,atol=1e-15)
        self.assertEqual(out['training_label_observations'],60)
        self.assertEqual(out['matched_label_observations'],20)

    def test_shrinkage_mean_and_matched_control(self):
        stock, sector = prices_for_states([1]*15+[-1]*45)
        out = k.measure(stock,sector)
        mu = math.fsum(out['labels'])/60
        matching = [out['labels'][j] for j in range(15)]
        expected = (math.fsum(matching)+20*mu)/35
        self.assertEqual(out['matching_count'],15)
        self.assertEqual(out['control'],mu)
        self.assertEqual(out['candidate'],expected)
        self.assertEqual(out['conditional_mean_estimates'],1)
        self.assertEqual(out['pooled_mean_estimates'],1)
        self.assertNotEqual(out['candidate'],out['control'])

    def test_minimum_matches_common_mask_and_diagnostics(self):
        for count in (0,9,10,59,60):
            out = k.measure(*prices_for_states([1]*count+[-1]*(60-count)))
            self.assertEqual(out['matching_count'],count)
            self.assertEqual(out['available'],count>=10)
            self.assertEqual(len(out['labels']),60)
            if count < 10:
                self.assertTrue(math.isnan(out['candidate']) and math.isnan(out['control']))
                self.assertIsNone(out['shrunken_payoff'])
                self.assertEqual(out['training_label_observations'],0)
                self.assertEqual(out['matched_label_observations'],0)
            else:
                self.assertEqual(out['training_label_observations'],60)
                self.assertEqual(out['matched_label_observations'],count)

    def test_all_matching_exact_identity_and_zero_state_valid(self):
        for state in (-1,0,1):
            out = k.measure(*prices_for_states([state]*60, query=state))
            self.assertTrue(out['available'] and out['all_states_match'])
            self.assertEqual(np.float64(out['candidate']).tobytes(),np.float64(out['control']).tobytes())
            self.assertEqual(out['matching_count'],60)
        stock = np.exp(np.linspace(0.,.4,1471)+.05*np.sin(np.arange(1471)/24.))
        out = k.measure(stock,stock*8.)
        self.assertEqual(out['states'],[0]*60)
        self.assertEqual(out['query_state'],0)
        self.assertEqual(out['candidate'],0.)

    def test_scale_invariance_and_source_immutability(self):
        stock, sector = prices_for_states([1,-1,0]*20, query=0)
        before = stock.copy()
        a = k.measure(stock,sector)
        b = k.measure(stock*8.,sector*.25)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(stock,before)

    def test_missing_any_interior_price_invalidates_both(self):
        stock,sector = prices_for_states([1]*60)
        for value in (np.nan,np.inf,-np.inf,0.,-1.):
            bad = stock.copy();bad[11]=value
            out = k.measure(bad,sector)
            self.assertEqual(out['reason'],'invalid_stock_price')
            self.assertEqual(out['labels'],[])
            self.assertFalse(out['available'])
        bad = sector.astype(object);bad[23]='bad'
        self.assertEqual(k.measure(stock,bad)['reason'],'nonnumeric_benchmark_price')
        with self.assertRaises(ValueError):k.measure(stock[:-1],sector)

    def test_large_valid_label_prices_remain_finite(self):
        stock = np.exp(np.linspace(-700.,700.,1471))
        sector = np.exp(np.linspace(700.,-700.,1471))
        out = k.measure(stock,sector)
        self.assertTrue(out['available'])
        self.assertTrue(all(math.isfinite(v) for v in out['labels']))
        self.assertGreater(out['candidate'],0.)


class BuilderTests(unittest.TestCase):
    def test_dates_maturity_counts_and_original_mapping(self):
        panel,meta,calendar = fixture()
        values,audit = k.build_features(panel,meta,calendar,mapper)
        self.assertTrue(values['available'][:,:2].all())
        self.assertFalse(values['available'][:,2].any())
        self.assertEqual(audit['conditional_mean_estimates'],6)
        self.assertEqual(audit['pooled_mean_estimates'],6)
        self.assertEqual(audit['training_label_observations'],360)
        for event in audit['event_history']:
            t = event['session_position']
            self.assertEqual(event['price_start'],calendar[t-1512])
            self.assertEqual(event['price_end'],calendar[t-42])
            cell=event['cells'][0]
            self.assertEqual(cell['state_start_dates'][0],calendar[t-1512])
            self.assertEqual(cell['predictor_dates'][0],calendar[t-1302])
            self.assertEqual(cell['label_end_dates'][-1],calendar[t-42])
            self.assertEqual(cell['query_state'],k._relative_sign(*[
                panel['adj'].at[pd.Timestamp(calendar[d]),s] for d,s in
                [(t-42,'AAA'),(t-252,'AAA'),(t-42,'XLK'),(t-252,'XLK')]]))

    def test_query_mapping_fixed_through_history_and_etf_unknown_neutral(self):
        panel,meta,calendar = fixture()
        base=k.build_features(panel,meta,calendar,mapper)
        panel['sectors'].loc[:,:]='Unknown'
        for date in meta['dates']:
            panel['sectors'].loc[date,['AAA','BBB']]='Technology'
        changed=k.build_features(panel,meta,calendar,mapper)
        for key in base[0]:np.testing.assert_array_equal(base[0][key],changed[0][key])
        self.assertEqual(base[1],changed[1])
        panel['sectors'].loc[meta['dates'][0],'AAA']='Unknown'
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertFalse(values['available'][0,0])
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'],'missing_benchmark')
        self.assertEqual(audit['event_history'][0]['cells'][2]['reason'],'etf_neutral')

    def test_missing_official_session_not_compressed(self):
        panel,meta,calendar=fixture()
        day=panel['adj'].index[500]
        panel['adj']=panel['adj'].drop(day)
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertFalse(values['available'].any())
        self.assertEqual(audit['reason_counts'],{'missing_official_session':6,'etf_neutral':3})
        self.assertEqual(audit['event_history'][0]['missing_official_sessions'],[str(day.date())])

    def test_insufficient_history_and_missing_symbols(self):
        panel,meta,calendar=fixture()
        meta=copy.deepcopy(meta);meta['dates'][0]=calendar[1511]
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertEqual(audit['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        panel['adj']=panel['adj'].drop(columns='AAA')
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertEqual(audit['event_history'][1]['cells'][0]['reason'],'missing_symbol')
        panel['adj']=panel['adj'].drop(columns='XLK')
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertEqual(audit['event_history'][1]['cells'][1]['reason'],'missing_benchmark_symbol')

    def test_every_prefix_future_price_poison_and_empty(self):
        panel,meta,calendar=fixture()
        values,audit=k.build_features(panel,meta,calendar,mapper)
        for n in range(4):
            poison=copy.deepcopy(panel);poison['adj']=poison['adj'].astype(object)
            start=0 if n==0 else [1550,1600,1650][n-1]-42+1
            poison['adj'].iloc[start:,:]='unused future price'
            got,record=k.build_features(poison,meta,calendar,mapper,limit=n)
            for key in values:np.testing.assert_array_equal(got[key],values[key][:n])
            self.assertEqual(record['event_history'],audit['event_history'][:n])
            self.assertEqual(got['available'].shape,(n,3))

    def test_support_before_start_ignored_input_unchanged(self):
        panel,meta,calendar=fixture();before=copy.deepcopy(panel)
        base=k.build_features(panel,meta,calendar,mapper)
        poison=copy.deepcopy(panel);poison['adj']=poison['adj'].astype(object)
        poison['adj'].iloc[:38,:]='unused early price'
        got=k.build_features(poison,meta,calendar,mapper)
        for key in base[0]:np.testing.assert_array_equal(got[0][key],base[0][key])
        self.assertEqual(got[1],base[1])
        pd.testing.assert_frame_equal(panel['adj'],before['adj'])
        pd.testing.assert_frame_equal(panel['sectors'],before['sectors'])

    def test_mapping_and_schema_errors(self):
        panel,meta,calendar=fixture()
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar,mapper,limit=True)
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar[::-1],mapper)
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar,lambda *args:{})
        with self.assertRaises(ValueError):k.build_features(panel,meta,calendar,lambda day,ss,*args:{s:'XLK' for s in ss})
        bad=copy.deepcopy(meta);bad['symbols'][0]=['AAA','AAA','FUND']
        with self.assertRaises(ValueError):k.build_features(panel,bad,calendar,mapper)
        bad=copy.deepcopy(meta);bad.pop('etfs')
        with self.assertRaises(ValueError):k.build_features(panel,bad,calendar,mapper)

    def test_independent_full_audit_and_prefix_cross(self):
        import monthly_sector_transition189_independent as oracle
        panel,meta,calendar=fixture()
        values,audit=k.build_features(panel,meta,calendar,mapper)
        self.assertTrue(oracle.verify_features(panel,meta,calendar,mapper,values,audit)['passed'])
        for n in range(4):
            actual,record=k.build_features(panel,meta,calendar,mapper,limit=n)
            expected,reference=oracle.independent_features(panel,meta,calendar,mapper,limit=n)
            for key in ('candidate','control'):
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0,equal_nan=True)
            np.testing.assert_array_equal(actual['available'],expected['available'])
            oracle.record_check(record,reference)
        for n in (9,10,60):
            stock,sector=prices_for_states([1]*n+[-1]*(60-n))
            actual=k.measure(stock,sector);expected=oracle.measure_oracle(stock,sector)
            actual=copy.deepcopy(actual)
            if not actual['available']:actual['candidate']=actual['control']=None
            oracle.record_check(actual,expected)
        from test_monthly_sector_transition189_independent import fixture as other_fixture, mapper as other_mapper
        other_panel,other_meta,other_calendar=other_fixture()
        actual,record=k.build_features(other_panel,other_meta,other_calendar,other_mapper)
        self.assertTrue(oracle.verify_features(other_panel,other_meta,other_calendar,other_mapper,actual,record)['passed'])


if __name__=='__main__':unittest.main()
