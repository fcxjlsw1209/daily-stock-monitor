"""Independent synthetic-only relative-state, mature-label and account tests."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_sector_transition189_independent as audit


def mapper(day,names,sectors,etfs):
    return {s:audit.benchmark_for(pd.Timestamp(day),s,sectors,etfs) for s in names}


def fixture(count=3,width=4):
    index=pd.bdate_range('2008-01-01',periods=1600);x=np.arange(len(index),dtype=float)
    bench=np.exp(.00015*x+.01*np.sin(x*.05))
    frame=pd.DataFrame({'A':bench*np.exp(.0003*x),'C':bench*np.exp(.08*np.sin(x/130.)),
        'BABA':bench*np.exp(-.0002*x),'FUND':bench,'XLK':bench,'XLC':bench,'XLY':bench},index=index)
    names=['A','C','BABA','FUND'][:width]
    sectors=pd.DataFrame('Technology',index=index,columns=names)
    if 'C' in sectors:sectors.C='Communication Services'
    if 'BABA' in sectors:sectors.BABA='Energy' # Original explicit foreign map takes priority.
    calendar=index.strftime('%Y-%m-%d').tolist();dates=[calendar[t] for t in (1512,1533,1554)[:count]]
    return {'adj':frame,'sectors':sectors},dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['FUND']),calendar


def rare_state_path(n):
    levels=np.ones(71);levels[70-n:70]=2.;levels[70]=4.
    return np.interp(np.arange(1471),np.arange(0,1471,21),levels),np.ones(1471)


class ScalarTests(unittest.TestCase):
    def test_mature_60_labels_and_shrunk_conditional_mean(self):
        s,b=rare_state_path(10);r=audit.measure_oracle(s,b)
        self.assertTrue(r['available']);self.assertEqual(r['matching_count'],10)
        self.assertEqual(r['matching_indices'],list(range(50,60)))
        self.assertEqual(r['state_start_indices'],list(range(0,1240,21)))
        self.assertEqual(r['predictor_indices'][-1],1449);self.assertEqual(r['label_end_indices'][-1],1470)
        self.assertEqual((r['query_start_index'],r['query_end_index']),(1260,1470))
        self.assertAlmostEqual(r['unconditional_mean'],2*math.log(2)/60)
        self.assertAlmostEqual(r['candidate'],(math.log(2)+20*r['unconditional_mean'])/30)
        self.assertNotEqual(r['candidate'],r['control'])

    def test_minimum_10_boundary_retains_diagnostics_no_counts_when_failed(self):
        for n in (0,9,10):
            r=audit.measure_oracle(*rare_state_path(n))
            self.assertEqual(r['matching_count'],n)
            self.assertEqual(r['available'],n>=10)
            self.assertEqual(len(r['labels']),60)
            self.assertEqual(r['conditional_mean_estimates'],int(n>=10))
            self.assertEqual(r['training_label_observations'],60 if n>=10 else 0)
            if n<10:self.assertIsNone(r['candidate']);self.assertIsNone(r['shrunken_payoff'])

    def test_all60_identity_and_relative_ties(self):
        s=np.exp(.001*np.arange(1471));b=np.ones(1471)
        r=audit.measure_oracle(s,b)
        self.assertEqual(r['matching_count'],60);self.assertTrue(r['all_states_match'])
        self.assertEqual(r['candidate'],r['control']);self.assertEqual(r['candidate'],r['shrunken_payoff'])
        r=audit.measure_oracle(s,s)
        self.assertEqual(r['states'],[0]*60);self.assertEqual(r['query_state'],0)
        self.assertEqual(r['candidate'],0.);self.assertEqual(r['control'],0.)

    def test_exact_binary_float_rational_signs_not_log_or_float_products(self):
        lo=1e300;hi=float(np.nextafter(lo,math.inf))
        self.assertEqual(math.log(lo),math.log(hi))
        self.assertEqual(audit.relative_state(hi,lo,lo,lo),1)
        self.assertEqual(audit.relative_state(lo,lo,hi,lo),-1)
        self.assertEqual(audit.relative_state(hi,lo,hi,lo),0)
        self.assertTrue(math.isinf(hi*lo))
        self.assertEqual(audit.relative_state(5e-324,1e308,5e-324,1e308),0)

    def test_extreme_positive_prices_stable_log_labels(self):
        s=np.array([5e-324 if i%2==0 else 1e308 for i in range(1471)])
        r=audit.measure_oracle(s,np.ones(1471))
        self.assertTrue(r['available']);self.assertEqual(r['candidate'],r['control'])
        self.assertTrue(all(math.isfinite(x) for x in r['labels']))
        self.assertGreater(max(r['labels']),1000.)
        with patch.object(audit.math,'fsum',return_value=math.inf),self.assertRaises(FloatingPointError):audit.measure_oracle(*rare_state_path(10))

    def test_full_daily_support_and_source_failure_priority(self):
        s,b=rare_state_path(10)
        for at in (0,1,209,210,1260,1469,1470):
            z=s.copy();z[at]=np.nan
            r=audit.measure_oracle(z,b);self.assertEqual(r['reason'],'invalid_stock_price');self.assertEqual(r['labels'],[])
        s=s.astype(object);b=b.astype(object);s[0]=0.;b[0]='bad'
        self.assertEqual(audit.measure_oracle(s,b)['reason'],'invalid_stock_price')
        s[0]='bad';self.assertEqual(audit.measure_oracle(s,b)['reason'],'nonnumeric_stock_price')
        s[0]=1.;self.assertEqual(audit.measure_oracle(s,b)['reason'],'nonnumeric_benchmark_price')
        with self.assertRaises(ValueError):audit.measure_oracle(np.ones(1470),np.ones(1471))

    def test_power_of_two_scale_and_input_immutability(self):
        s,b=rare_state_path(10);old=(s.copy(),b.copy());r=audit.measure_oracle(s,b)
        for scale in (2.**-600,2.**600):audit.record_check(audit.measure_oracle(s*scale,b*scale),r)
        np.testing.assert_array_equal(s,old[0]);np.testing.assert_array_equal(b,old[1])


class SourceTests(unittest.TestCase):
    def test_full_source_audit_mapping_dates_counts_and_flags(self):
        p,m,c=fixture();before=copy.deepcopy(p)
        v,r=audit.independent_features(p,m,c,mapper);z=audit.verify_features(p,m,c,mapper,v,r)
        self.assertEqual(z['conditional_mean_estimates'],z['available_cells'])
        self.assertEqual(z['training_label_observations'],60*z['available_cells'])
        self.assertEqual(z['mapping_source_sha256'],audit.MAPPING_SHA)
        for t,row in zip((1512,1533,1554),r['event_history']):
            self.assertEqual((row['price_start'],row['price_end']),(c[t-1512],c[t-42]))
            self.assertEqual(row['benchmarks'],dict(A='XLK',C='XLC',BABA='XLY',FUND=None))
            self.assertEqual(row['cells'][-1]['reason'],'etf_neutral')
            for cell in row['cells'][:-1]:
                self.assertEqual(cell['label_end_dates'][-1],c[t-42]);self.assertLess(cell['predictor_dates'][-1],cell['label_end_dates'][-1])
                self.assertEqual(cell['windows']['query_state']['price_start'],c[t-252])
        for key in p:pd.testing.assert_frame_equal(p[key],before[key])

    def test_query_mapping_fixed_whole_window_and_callback_prefix(self):
        p,m,c=fixture(count=1);p['sectors'].loc[:,'A']='Energy';p['sectors'].loc[c[1512],'A']='Technology'
        def checked(day,names,sectors,etfs):
            self.assertEqual(sectors.index.max(),pd.Timestamp(day));return mapper(day,names,sectors,etfs)
        v,r=audit.independent_features(p,m,c,checked)
        self.assertEqual(r['event_history'][0]['benchmarks']['A'],'XLK');self.assertTrue(v['available'][0,0])
        def bad(*args):return {s:'XLE' for s in args[1]}
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,bad)

    def test_history_missing_unknown_and_no_sector_substitute(self):
        p,m,c=fixture(count=1)
        for mode,reason in [('short','insufficient_history'),('stock','missing_symbol'),('benchmark','missing_benchmark_symbol'),('session','missing_official_session'),('unknown','missing_benchmark')]:
            q=copy.deepcopy(p);mm=copy.deepcopy(m)
            if mode=='short':mm['dates']=[c[1511]]
            elif mode=='stock':q['adj']=q['adj'].drop(columns='A')
            elif mode=='benchmark':q['adj']=q['adj'].drop(columns='XLK')
            elif mode=='session':q['adj']=q['adj'].drop(pd.Timestamp(c[7]))
            else:q['sectors'].loc[c[1512],'A']='Unrecognized'
            v,r=audit.independent_features(q,mm,c,mapper)
            self.assertEqual(r['event_history'][0]['cells'][0]['reason'],reason)
            self.assertFalse(v['available'][0,0]);self.assertTrue(np.isnan(v['candidate'][0,0]))

    def test_all_prefixes_and_skip42_future_poison(self):
        p,m,c=fixture();v,r=audit.independent_features(p,m,c,mapper)
        for n in range(4):
            q=copy.deepcopy(p);q['adj']=q['adj'].astype(object)
            end=-1 if n==0 else [1512,1533,1554][n-1]-42
            q['adj'].iloc[end+1:,:]='after maturity poison'
            vv,rr=audit.independent_features(q,m,c,mapper,limit=n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(rr['event_history'],r['event_history'][:n])

    def test_audit_sign_date_count_and_identity_mutations_rejected(self):
        p,m,c=fixture(count=1);v,r=audit.independent_features(p,m,c,mapper)
        for field,value in [('query_state',-1),('matching_count',59),('label_end_indices',[230]*60)]:
            bad=copy.deepcopy(r);bad['event_history'][0]['cells'][0][field]=value
            with self.assertRaises(ValueError):audit.verify_features(p,m,c,mapper,v,bad)
        bad=copy.deepcopy(v);bad['candidate'][0,0]=np.nextafter(bad['candidate'][0,0],math.inf)
        with self.assertRaisesRegex(ValueError,'All60 identity'):audit.verify_features(p,m,c,mapper,bad,r)

    def test_all49_columns_before_eligibility_no_rank(self):
        p,m,c=fixture(count=1);names=['S'+str(i) for i in range(48)]+['FUND']
        p['adj']=pd.DataFrame({s:p['adj'].A for s in names},index=p['adj'].index).assign(XLK=p['adj'].XLK)
        p['sectors']=pd.DataFrame('Technology',index=p['sectors'].index,columns=names)
        m['symbols']=[names]
        v,r=audit.independent_features(p,m,c,mapper)
        self.assertEqual(v['available'].shape,(1,49));self.assertEqual(r['available_cells'],48)
        self.assertEqual(r['training_label_observations'],48*60);self.assertFalse(r['ranking_performed'])

    def test_schema_invalid_calendar_and_duplicate_names(self):
        p,m,c=fixture()
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,mapper,limit=True)
        with self.assertRaises(ValueError):audit.independent_features(p,m,c[::-1],mapper)
        m['symbols'][0]=['A','A','BABA','FUND']
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,mapper)


class DeltaTests(unittest.TestCase):
    def test_fixed_tanh_same_domain_no_rank_initial_and_bound(self):
        v=dict(candidate=np.array([[1.,1.],[.05,np.nan],[-100.,0.]]),control=np.array([[2.,2.],[-.05,np.nan],[100.,0.]]),available=np.array([[1,1],[1,0],[1,1]],bool))
        d=audit.independent_deltas(v);r=audit.verify_deltas(v,d)
        self.assertEqual(d[audit.ARMS[1]][1,0],.02*math.tanh(1.));self.assertEqual(d[audit.ARMS[1]][2,0],-.02)
        np.testing.assert_array_equal(d[audit.ARMS[1]][0],[0.,0.]);self.assertEqual(d[audit.ARMS[1]][1,1],0.)
        self.assertEqual(r['scale'],.05);self.assertEqual(r['max_absolute_delta'],.02)
        with self.assertRaises(ValueError):audit.independent_deltas(v,dose=.01)
        for row in (0,1):
            huge=copy.deepcopy(v);huge['candidate'][row,0]=1e308
            with self.assertRaises(FloatingPointError):audit.independent_deltas(huge)
        bad=copy.deepcopy(d);bad[audit.ARMS[1]][1,0]=.05*.02
        with self.assertRaises(ValueError):audit.verify_deltas(v,bad)
        bad=copy.deepcopy(d);bad[audit.ARMS[1]][0,0]=1e-18
        with self.assertRaises(ValueError):audit.verify_deltas(v,bad)


class OutcomeTests(unittest.TestCase):
    def test_three_new_arms_nine_paths_45_slices_and_tamper(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        r=audit.audit_outcomes(*args)
        self.assertEqual((r['checked_paths'],r['checked_slices']),(9,45));self.assertEqual(set(r['paths']),set(audit.ARMS))
        bad=copy.deepcopy(args);bad[4][audit.ARMS[1]]['same_close_zero']['metrics']['cagr']+=.01
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


if __name__=='__main__':unittest.main(verbosity=2)
