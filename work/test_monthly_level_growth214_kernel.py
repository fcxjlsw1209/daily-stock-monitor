"""Synthetic level-growth clocks, shape identities, shared support and prefixes."""
import copy
import json
import math
import unittest

import numpy as np
import pandas as pd
import monthly_level_growth214_kernel as k


def fixture():
    dates=pd.bdate_range('2014-01-01','2017-01-04'); x=np.arange(len(dates),dtype=float)
    px=pd.DataFrame({'A':100.+.06*x+3*np.sin(.13*x), 'B':200.+.02*x+5*np.cos(.23*x),
        'FUND':80.+.02*x+4*np.sin(.37*x)},index=dates)
    return {'adj':px},dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
        symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND']),dates.strftime('%Y-%m-%d').tolist()


def extend(formation):
    out=list(formation)
    for j in range(42):out.append(out[-1]*(1.01 if j%2 else .985))
    return np.array(out)


def prices():
    return extend([100.*math.exp(.3*(i/252.)**2+.015*math.sin(i*.2)) for i in range(253)])


class MeasureTests(unittest.TestCase):
    def test_loglinear_rate_matches_original_endpoint_algebraically(self):
        for b in (-.002,0.,.002):
            p=extend([100.*math.exp(b*j) for j in range(253)]); a=k.measure(p)
            self.assertTrue(a['available'])
            self.assertAlmostEqual(a['log_lift'],252.*b,delta=2e-15)
            self.assertAlmostEqual(a['numerator_candidate'],math.expm1(252.*b),delta=4e-15)
            self.assertAlmostEqual(a['numerator_candidate'],a['numerator_control'],delta=4e-15)
            self.assertEqual(a['numerator_control'],float(p[252])/float(p[0])-1.)
        # Equality is approximate; the implementation has no path-identification/copy branch.
        self.assertEqual(k.LIFT_SCALE,252./127.)

    def test_scale_invariance_and_input_immutability(self):
        p=prices();original=p.copy();a=k.measure(p)
        for scale in (2.**-400,2.**400):
            b=k.measure(p*scale)
            self.assertEqual(a['numerator_control'],b['numerator_control'])
            self.assertEqual(a['volatility'],b['volatility'])
            self.assertAlmostEqual(a['log_lift'],b['log_lift'],delta=2e-13)
            self.assertAlmostEqual(a['raw_candidate'],b['raw_candidate'],delta=3e-11)
        np.testing.assert_array_equal(p,original)

    def test_flat_formation_is_valid_zero_if_current_risk_positive(self):
        a=k.measure(extend([100.]*253))
        self.assertTrue(a['available']);self.assertEqual(a['log_lift'],0.)
        self.assertEqual(a['numerator_candidate'],0.);self.assertEqual(a['numerator_control'],0.)
        self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)
        z=k.measure(np.ones(295)*100.)
        self.assertTrue(z['source_available']);self.assertFalse(z['available']);self.assertEqual(z['reason'],'zero_volatility')
        self.assertEqual(z['volatility'],0.);self.assertEqual(z['numerator_candidate'],0.)
        self.assertTrue(np.isnan(z['raw_candidate']));self.assertTrue(np.isnan(z['raw_control']))

    def test_step_timing_is_not_endpoint_return(self):
        nums=[]
        for jump in (1,126,252):
            p=extend([100. if j<jump else 150. for j in range(253)]);a=k.measure(p)
            self.assertEqual(a['numerator_control'],.5);nums.append(a['numerator_candidate'])
        self.assertAlmostEqual(nums[0],nums[2],delta=3e-15)
        self.assertGreater(nums[1],nums[0]);self.assertGreater(nums[1],.5)
        self.assertAlmostEqual(nums[1],math.expm1((252./127.)*math.log(1.5)),delta=4e-15)

    def test_middle_price_checked_but_does_not_enter_any_formula(self):
        p=prices();a=k.measure(p);p[126]=1e200;b=k.measure(p)
        for name in ('log_lift','numerator_candidate','numerator_control','volatility','raw_candidate','raw_control'):
            self.assertEqual(a[name],b[name])
        self.assertEqual(len(a['early_logs']),126);self.assertEqual(len(a['late_logs']),126)
        self.assertNotEqual(a['omitted_center_price'],b['omitted_center_price'])
        p[126]=0.;self.assertFalse(k.measure(p)['available'])

    def test_recent42_only_changes_risk_not_either_numerator(self):
        p=prices();a=k.measure(p)
        p[253:]=[p[252]*(1.+.06*((j%3)-1)) for j in range(42)];b=k.measure(p)
        for name in ('log_lift','numerator_candidate','numerator_control'):
            self.assertEqual(a[name],b[name])
        self.assertNotEqual(a['volatility'],b['volatility'])
        self.assertEqual(b['raw_candidate'],b['numerator_candidate']/b['volatility'])
        self.assertEqual(b['raw_control'],b['numerator_control']/b['volatility'])

    def test_exact_arithmetic_means_and_current_sample_risk(self):
        p=prices();a=k.measure(p)
        early=[math.log(float(v)) for v in p[:126]];late=[math.log(float(v)) for v in p[127:253]]
        em=math.fsum(early)/126.;lm=math.fsum(late)/126.;lift=(252./127.)*(lm-em)
        returns=[float(p[j])/float(p[j-1])-1. for j in range(232,295)]
        mean=math.fsum(returns)/63.;vol=math.sqrt(math.fsum((r-mean)**2 for r in returns)/62.)
        self.assertEqual(a['early_logs'],early);self.assertEqual(a['late_logs'],late)
        self.assertEqual(a['early_log_mean'],em);self.assertEqual(a['late_log_mean'],lm)
        self.assertEqual(a['mean_difference'],lm-em);self.assertEqual(a['log_lift'],lift)
        self.assertEqual(a['numerator_candidate'],math.expm1(lift));self.assertEqual(a['simple_returns'],returns)
        self.assertEqual(a['return_mean'],mean);self.assertEqual(a['volatility'],vol)

    def test_time_reversed_formation_inverts_geometric_growth(self):
        p=prices();a=k.measure(p);q=extend(list(p[:253][::-1]));b=k.measure(q)
        self.assertEqual(a['log_lift'],-b['log_lift'])
        self.assertAlmostEqual((1+a['numerator_candidate'])*(1+b['numerator_candidate']),1.,delta=3e-16)
        self.assertAlmostEqual((1+a['numerator_control'])*(1+b['numerator_control']),1.,delta=3e-16)

    def test_non_numeric_missing_and_full295_common_support(self):
        for bad,reason in ((None,'nonnumeric_adjusted_price'),(True,'nonnumeric_adjusted_price'),
            ('1','nonnumeric_adjusted_price'),(1j,'nonnumeric_adjusted_price'),(0.,'invalid_adjusted_price'),
            (-1.,'invalid_adjusted_price'),(np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price')):
            for index in (0,126,253,294):
                p=list(prices());p[index]=bad;a=k.measure(p)
                self.assertEqual(a['reason'],reason);self.assertFalse(a['source_available'])
                self.assertEqual(a['early_logs'],[]);self.assertTrue(np.isnan(a['raw_candidate']))
        for p in (np.ones(294),np.ones((295,1)),1.):
            with self.assertRaises(ValueError):k.measure(p)

    def test_valid_source_nonfinite_derivatives_fail_hard(self):
        for mode in ('candidate','endpoint','risk_return','risk_variance','normalized_candidate'):
            p=prices()
            if mode=='candidate':
                p[:126]=1e-200;p[127:253]=1e200;p[0]=p[252]=1.
            elif mode=='endpoint':p[0]=1e-308;p[252]=1e308
            elif mode=='risk_return':p[231]=1e-308;p[232]=1e308
            elif mode=='risk_variance':p[231]=1.;p[232]=1e200
            else:
                p[:]=1.;p[127:231]=1e181
                for j in range(232,295):p[j]=p[j-1]*(1.+(1e-14 if j%2 else -2e-14))
            with self.subTest(mode=mode),self.assertRaises(FloatingPointError):k.measure(p)

    def test_tiny_positive_risk_is_not_floored(self):
        p=prices();p[231:]=100.
        for j in range(232,295):p[j]=p[j-1]*(1.+(1e-14 if j%2 else -2e-14))
        a=k.measure(p);self.assertTrue(a['available']);self.assertGreater(a['volatility'],0.)
        self.assertLess(a['volatility'],1e-12);self.assertEqual(a['raw_candidate'],a['numerator_candidate']/a['volatility'])


class BuilderTests(unittest.TestCase):
    def test_schema_counts_clocks_and_json_audit(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(set(v),set(k.VALUE_KEYS));self.assertEqual(a['available_by_event'],[3]*3)
        expected=dict(valid_price_observations=295*9,formation_log_observations=252*9,
            simple_return_observations=63*9,segment_mean_evaluations=18,lift_evaluations=9,
            numerator_evaluations=18,risk_evaluations=9,ranked_observations=18)
        self.assertEqual(set(expected),set(k.COUNT_FIELDS))
        for name,value in expected.items():self.assertEqual(a[name],value)
        for name in ('descriptive_model_fits','new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            self.assertEqual(a[name],0)
        self.assertTrue(np.isfinite(v['candidate'][0]).all());json.dumps(a,allow_nan=False)
        for row in a['event_history']:
            t=row['session_position']
            fields=('price_start','price_end','formation_price_start','formation_price_end','early_mean_start',
                'early_mean_end','omitted_center_date','late_mean_start','late_mean_end','risk_price_start','risk_price_end')
            self.assertEqual([row[x] for x in fields],[c[t-294],c[t],c[t-294],c[t-42],c[t-294],
                c[t-169],c[t-168],c[t-167],c[t-42],c[t-63],c[t]])

    def test_original_ties_singleton_empty_no_eligibility_filter(self):
        np.testing.assert_array_equal(k.percentile_rank(np.array([1.,2.,2.,4.,np.nan]),np.array([True]*4+[False])),[-.25,.125,.125,.5,np.nan])
        np.testing.assert_array_equal(k.percentile_rank(np.ones(3),np.ones(3,bool)),[2/3-.5]*3)
        p,m,c=fixture();p['adj'].loc[:,['B','FUND']]=np.nan;v,a=k.build_features(p,m,c)
        self.assertEqual(a['available_by_event'],[1]*3);np.testing.assert_array_equal(v['candidate'][:,0],[.5]*3)
        p['adj'].loc[:,:]=100.;v,a=k.build_features(p,m,c)
        self.assertEqual(a['source_available_cells'],9);self.assertEqual(a['available_cells'],0)
        self.assertEqual(a['numerator_evaluations'],18);self.assertEqual(a['segment_mean_evaluations'],18)
        self.assertTrue(all(np.isnan(x).all() for key,x in v.items() if key!='available'));json.dumps(a,allow_nan=False)

    def test_missing_symbol_official_session_and_checked_center(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(index=c[t-100],columns='B')
        _,a=k.build_features(p,m,c,1)
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells']],['missing_official_session','missing_symbol','missing_official_session'])
        _,a=k.build_features(p,m,c[t-100:],1);self.assertEqual(a['event_history'][0]['price_end'],None)
        self.assertEqual(a['reason_counts'],{'insufficient_history':3})
        p,m,c=fixture();p['adj'].loc[c[t-168],'A']=0.;v,a=k.build_features(p,m,c,1)
        self.assertFalse(v['available'][0,0]);self.assertEqual(a['source_available_cells'],2)

    def test_all_physical_prefixes_future_poison_and_no_mutation(self):
        p,m,c=fixture();original=p['adj'].copy(deep=True);mm=copy.deepcopy(m);v,a=k.build_features(p,m,c)
        for n in range(4):
            pp={'adj':p['adj'].loc[:m['dates'][n-1]].copy() if n else p['adj'].iloc[:0].copy()}
            vv,aa=k.build_features(pp,m,c,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
        pp={'adj':p['adj'].astype(object)};pp['adj'].loc[pp['adj'].index>m['dates'][0]]='FUTURE';pp['adj']['UNUSED']='POISON'
        vv,aa=k.build_features(pp,m,c,1)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:1])
        self.assertEqual(aa['event_history'],a['event_history'][:1]);pd.testing.assert_frame_equal(original,p['adj']);self.assertEqual(m,mm)

    def test_cross_section_permutation_and_etf_same_domain(self):
        p,m,c=fixture();v,_=k.build_features(p,m,c);perm=[2,0,1]
        other=dict(m,symbols=[[x[j] for j in perm] for x in m['symbols']],etfs=[])
        vv,_=k.build_features(p,other,c)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:,perm])

    def test_input_protocol(self):
        p,m,c=fixture()
        for n in (-1,4,True,1.):
            with self.assertRaises(ValueError):k.build_features(p,m,c,n)
        for key,value in [('dates',['2016-08-17','2016-08-30','2016-09-30']),('symbols',[['A'],['A','B'],['A']]),('etfs',['FUND','FUND'])]:
            bad=dict(m);bad[key]=value
            with self.assertRaises(ValueError):k.build_features(p,bad,c)
        with self.assertRaises(ValueError):k.build_features(p,m,c[:c.index(m['dates'][-1])+1])
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'],'volume':p['adj']},m,c)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'].iloc[::-1]},m,c)
        for raw,mask in ((np.zeros(2),np.ones(2,int)),(np.zeros(2),np.ones(1,bool)),(np.array([np.inf]),np.ones(1,bool))):
            with self.assertRaises(ValueError):k.percentile_rank(raw,mask)


if __name__=='__main__':unittest.main(verbosity=2)
