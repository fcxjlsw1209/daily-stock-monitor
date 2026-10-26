"""Invented paths only; no actual feature, order, model, or return replay."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

import monthly_level_growth214_independent as independent


def fixture(count=3, width=4):
    dates=pd.bdate_range('2014-01-01',periods=1000)
    t=np.arange(len(dates),dtype=float)
    frame=pd.DataFrame({'A':12*np.exp(.0008*t+.04*np.sin(t/11.)),
        'B':22*np.exp(.0003*t+.08*np.cos(t/37.)),
        'FUND':17*np.exp(-.0001*t+.045*np.sin(t/29.)),
        'FLAT':np.full(len(t),11.)},index=dates)
    names=['A','B','FUND','FLAT'][:width]
    query=['2016-08-17','2016-08-31','2016-09-30'][:count]
    return {'adj':frame},dict(dates=query,symbols=[names[:] for _ in query],etfs=['FUND']),dates.strftime('%Y-%m-%d').tolist()


def prices(kind='wave'):
    t=np.arange(253,dtype=float)
    if kind=='step':z=np.where(t<126,0.,.4)
    elif kind=='up':z=.004*t
    elif kind=='down':z=-.004*t
    elif kind=='flat':z=np.zeros(253)
    else:z=.001*t+.13*np.sin(t/15.)
    p=np.empty(295);p[:253]=10*np.exp(z)
    for j in range(253,295):p[j]=p[j-1]*(1.02 if j%2 else .99)
    return p


def arrays(n=3,width=4):
    calm=np.array([i%3==2 for i in range(n)],bool)
    cw=np.where(calm,.5,.52)
    core=np.tile((np.arange(width)+1)/width-.5,(n,1))*cw[:,None]
    return dict(core=core,core_weight=cw[:,None],state_calm10=calm,
                eligible=np.zeros((n,width),bool),not_consumed=object())


def targets(values,original,full=None):
    n,width=values['available'].shape
    result={}
    for arm,key in independent.RAW.items():
        x=np.full((n if full is None else full,width),np.nan)
        x[:n]=values[key]
        x[:n][original['state_calm10'][:n]]=np.nan
        if n:x[0]=np.nan
        result[arm]=x
    return result


class Tests(unittest.TestCase):
    def test_declared_two_half_means_and_centroid_factor(self):
        p=prices();a=independent.measure_oracle(p)
        early=[math.log(float(p[k])) for k in range(126)]
        late=[math.log(float(p[k])) for k in range(127,253)]
        self.assertEqual(a['early_logs'],early);self.assertEqual(a['late_logs'],late)
        self.assertEqual(a['early_log_mean'],math.fsum(early)/126.)
        self.assertEqual(a['late_log_mean'],math.fsum(late)/126.)
        early_centroid=sum(range(126))/126
        late_centroid=sum(range(127,253))/126
        self.assertEqual(late_centroid-early_centroid,127.)
        self.assertEqual(a['log_lift'],252./127.*(a['late_log_mean']-a['early_log_mean']))
        self.assertEqual(a['numerator_candidate'],math.expm1(a['log_lift']))

    def test_excluded_center_is_not_logged_but_must_be_valid(self):
        p=prices();a=independent.measure_oracle(p)
        p[126]=1e300;b=independent.measure_oracle(p)
        for k in a:
            if k!='omitted_center_price':self.assertEqual(a[k],b[k],k)
        p[126]=0.
        self.assertEqual(independent.measure_oracle(p)['reason'],'invalid_adjusted_price')

    def test_loglinear_identity_algebra_without_tie_snapping(self):
        for slope in (.004,-.004,0.):
            p=prices();p[:253]=np.exp(slope*np.arange(253))
            a=independent.measure_oracle(p)
            self.assertAlmostEqual(a['log_lift'],252*slope,places=14)
            self.assertAlmostEqual(a['numerator_candidate'],a['numerator_control'],places=14)
            self.assertEqual(a['numerator_candidate'],math.expm1(a['log_lift']))
        # A synthetic loglinear path can have a last-bit difference: never copy control.
        p=prices('up');a=independent.measure_oracle(p)
        self.assertEqual(a['numerator_candidate'],math.expm1(a['log_lift']))

    def test_step_and_interior_level_change_differ_from_endpoint(self):
        p=prices('step');a=independent.measure_oracle(p)
        self.assertGreater(a['numerator_candidate'],a['numerator_control'])
        p=prices();a=independent.measure_oracle(p);p[60]*=2.;b=independent.measure_oracle(p)
        self.assertNotEqual(a['numerator_candidate'],b['numerator_candidate'])
        self.assertEqual(a['numerator_control'],b['numerator_control'])
        self.assertEqual(a['volatility'],b['volatility'])

    def test_negative_and_finite_zero_are_not_clipped_or_missing(self):
        a=independent.measure_oracle(prices('down'))
        self.assertLess(a['raw_candidate'],0.);self.assertLess(a['raw_control'],0.)
        a=independent.measure_oracle(prices('flat'))
        self.assertTrue(a['available']);self.assertGreater(a['volatility'],0.)
        self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)
        self.assertEqual(a['reason'],'available')

    def test_current63_risk_and_skip42_source_offsets(self):
        p=prices();a=independent.measure_oracle(p)
        returns=[float(p[j])/float(p[j-1])-1. for j in range(232,295)]
        self.assertEqual(a['simple_returns'],returns)
        self.assertEqual(a['return_mean'],math.fsum(returns)/63.)
        self.assertAlmostEqual(a['volatility'],float(np.std(returns,ddof=1)),places=15)
        self.assertEqual(a['numerator_control'],p[252]/p[0]-1.)
        p[-1]*=1.3;b=independent.measure_oracle(p)
        self.assertEqual(a['numerator_candidate'],b['numerator_candidate'])
        self.assertEqual(a['numerator_control'],b['numerator_control'])
        self.assertNotEqual(a['volatility'],b['volatility'])

    def test_zero_risk_joint_missing_retains_source_evidence(self):
        p=prices();p[231:]=p[231]
        a=independent.measure_oracle(p)
        self.assertTrue(a['source_available']);self.assertFalse(a['available'])
        self.assertEqual(a['reason'],'zero_volatility')
        self.assertEqual(len(a['early_logs']),126);self.assertEqual(len(a['late_logs']),126)
        self.assertIsNone(a['raw_candidate']);self.assertIsNone(a['raw_control'])
        p=prices();p[231:]=2.**np.arange(64)
        self.assertEqual(independent.measure_oracle(p)['volatility'],0.)

    def test_complete295_numeric_nonboolean_positive_source(self):
        for idx in (0,1,125,126,127,160,230,231,252,253,294):
            for bad in (0.,-1.,np.nan,np.inf):
                p=prices();p[idx]=bad;a=independent.measure_oracle(p)
                self.assertEqual(a['reason'],'invalid_adjusted_price')
                self.assertFalse(a['source_available'])
        for bad in (True,None,'2',complex(2,0)):
            p=prices().astype(object);p[120]=bad
            self.assertEqual(independent.measure_oracle(p)['reason'],'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(294))

    def test_valid_source_overflow_hardfails_instead_of_missing(self):
        for idx,value in ((0,5e-324),(293,5e-324)):
            p=prices();p[idx]=value
            with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=prices();p[:126]=1e-300;p[127:253]=1e300
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=prices();p[293]=1.;p[294]=1e200
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)

    def test_price_scale_invariance_is_not_a_float_tie_rule(self):
        p=prices();a=independent.measure_oracle(p)
        for scale in (2.**100,2.**-100):
            b=independent.measure_oracle(p*scale)
            self.assertEqual(a['numerator_control'],b['numerator_control'])
            self.assertEqual(a['volatility'],b['volatility'])
            self.assertAlmostEqual(a['numerator_candidate'],b['numerator_candidate'],places=12)
            self.assertEqual(b['numerator_candidate'],math.expm1(b['log_lift']))

    def test_original_pct_ties_singletons_and_missing(self):
        for row in ([1.,1.,1.],[2.],[np.nan,np.nan],[-1.,0.,1.,1.]):
            x=np.asarray([row]);mask=np.isfinite(x)
            actual=independent.original_pct_ranks(x,mask)
            np.testing.assert_array_equal(actual[0],pd.Series(row).rank(pct=True).to_numpy()-.5)
        self.assertEqual(independent.original_pct_ranks(np.array([[1.]]),np.array([[True]]))[0,0],.5)
        x=np.array([[1.,np.nextafter(1.,2.)]])
        np.testing.assert_array_equal(independent.original_pct_ranks(x,np.ones((1,2),bool)),[[0.,.5]])

    def test_complete_audit_counts_no_model_fits_and_exact_endpoints(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['source_available_by_event'],[4]*3)
        self.assertEqual(a['available_by_event'],[3]*3)
        for key,mult in (('valid_price_observations',295),('formation_log_observations',252),
                         ('simple_return_observations',63),('segment_mean_evaluations',2),
                         ('lift_evaluations',1),('numerator_evaluations',2),('risk_evaluations',1)):
            self.assertEqual(a[key],mult*12)
        self.assertEqual(a['ranked_observations'],18)
        r=independent.verify_features(p,m,c,v,a)
        self.assertTrue(all(r[k] is True for k in independent.FEATURE_FLAGS))
        for k in ('descriptive_model_fits','new_model_fits','predictive_model_fits','new_predictive_model_fits'):
            self.assertEqual(r[k],0)
        self.assertEqual(r['numerical_tolerance'],0.)
        self.assertTrue(v['available'][:,2].all())
        event=a['event_history'][0];t=event['session_position']
        for key,offset in (('price_start',-294),('early_mean_end',-169),('omitted_center_date',-168),
                           ('late_mean_start',-167),('numerator_end',-42),('risk_price_start',-63)):
            self.assertEqual(event[key],c[t+offset])

    def test_common_missing_all_eight_float_matrices(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for key in set(v)-{'available'}:
            self.assertTrue(np.isnan(v[key][:,3]).all())
        bad=copy.deepcopy(v);bad['raw_control'][0,3]=0.
        with self.assertRaises(ValueError):independent.verify_features(p,m,c,bad,a)
        bad=copy.deepcopy(v);bad['available']=bad['available'].astype(int)
        with self.assertRaises(ValueError):independent.verify_features(p,m,c,bad,a)

    def test_all_prefixes_and_future_numeric_poison(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for n in range(4):
            f=p['adj'].astype(object).copy()
            cutoff=pd.Timestamp.min if n==0 else pd.Timestamp(m['dates'][n-1])
            f.loc[f.index>cutoff,:]='FUTURE'
            z,b=independent.independent_features({'adj':f},m,c,n)
            for key in v:np.testing.assert_array_equal(z[key],v[key][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])

    def test_source_calendar_and_query_structures(self):
        p,m,c=fixture()
        v,a=independent.independent_features({'adj':p['adj'].drop(pd.Timestamp('2016-07-05'))},m,c)
        self.assertFalse(v['available'].any())
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_official_session')
        v,a=independent.independent_features({'adj':p['adj'].drop(columns='A')},m,c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_symbol')
        young=dict(dates=['2014-01-31'],symbols=[['A']],etfs=[])
        _,a=independent.independent_features(p,young,c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c,True)

    def test_corrupt_arrays_ranks_audit_and_counts_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for key in set(v)-{'available'}:
            bad=copy.deepcopy(v);bad[key][1,0]=np.nextafter(bad[key][1,0],np.inf)
            with self.assertRaises(ValueError):independent.verify_features(p,m,c,bad,a)
        for key in independent.COUNT_FIELDS+('descriptive_model_fits','new_predictive_model_fits'):
            bad=copy.deepcopy(a);bad[key]+=1
            with self.assertRaises(ValueError):independent.verify_features(p,m,c,v,bad)
        bad=copy.deepcopy(a);bad['event_history'][0]['cells'][0]['early_logs'][0]+=1e-6
        with self.assertRaises(ValueError):independent.verify_features(p,m,c,v,bad)

    def test_halfcore_first_calm_missing_no_second_rank(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);original=arrays()
        tt=targets(v,original);dd=independent.independent_deltas(v,original,tt)
        for arm,key in independent.RAW.items():
            np.testing.assert_array_equal(dd[arm][[0,2]],0.)
            self.assertEqual(dd[arm][1,3],0.)
            expected=.5*(.52*v[key][1,:3]-original['core'][1,:3])
            np.testing.assert_array_equal(dd[arm][1,:3],expected)
        r=independent.verify_deltas(v,original,tt,dd)
        self.assertTrue(all(r[k] for k in independent.DELTA_FLAGS))
        self.assertEqual(r['max_absolute_delta'],.26)
        original['core'][1,:3]=.52*v['control'][1,:3]
        dd=independent.independent_deltas(v,original,tt)
        np.testing.assert_array_equal(dd['endpoint_growth_control'],0.)

    def test_delta_invalid_state_weight_target_and_future_prefix(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);original=arrays()
        tt=targets(v,original);dd=independent.independent_deltas(v,original,tt)
        for key,value in (('core_weight',np.full((3,1),.51)),('state_calm10',np.array([0,0,1]))):
            bad=copy.deepcopy(original);bad[key]=value
            with self.assertRaises(ValueError):independent.verify_deltas(v,bad,tt,dd)
        bad=copy.deepcopy(tt);bad[independent.ARMS[1]][0]=0.
        with self.assertRaises(ValueError):independent.verify_deltas(v,original,bad,dd)
        with self.assertRaises(ValueError):independent.verify_deltas(v,original,tt,dd,.25)
        for n in range(4):
            short={k:x[:n].copy() for k,x in v.items()};orig=copy.deepcopy(original)
            for k in ('core','core_weight','state_calm10'):
                orig[k]=orig[k].astype(object);orig[k][n:]='FUTURE'
            t=targets(short,original,full=3)
            d=independent.independent_deltas(short,orig,t)
            independent.verify_deltas(short,orig,t,d)

    def test_inputs_unchanged_and_pinned_accounting_dependencies(self):
        p,m,c=fixture();before=p['adj'].copy(deep=True);mm=copy.deepcopy(m)
        independent.independent_features(p,m,c)
        pd.testing.assert_frame_equal(p['adj'],before);self.assertEqual(m,mm)
        files=independent.dependency_files()
        self.assertIn(independent.FROZEN_HELPER,files)
        self.assertFalse(any('214_kernel' in str(x) for x in files))

    def test_both_producer_and_independent_fixtures_complete_audit_all_prefixes(self):
        import monthly_level_growth214_kernel as producer
        import test_monthly_level_growth214_kernel as producer_test
        for p,m,c in (fixture(),producer_test.fixture()):
            for n in range(len(m['dates'])+1):
                cutoff=pd.Timestamp.min if n==0 else pd.Timestamp(m['dates'][n-1])
                short={'adj':p['adj'].loc[:cutoff]}
                v,a=producer.build_features(short,m,c,limit=n)
                independent.verify_features(short,m,c,v,a,limit=n)
                original=arrays(len(m['dates']),len(m['symbols'][0]))
                tt=targets(v,original);dd=independent.independent_deltas(v,original,tt)
                independent.verify_deltas(v,original,tt,dd)


if __name__=='__main__':unittest.main(verbosity=2)
