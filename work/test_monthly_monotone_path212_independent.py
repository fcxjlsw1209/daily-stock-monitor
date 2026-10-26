"""Invented paths only; no actual feature, order, model, or return replay."""
import copy
import itertools
import math
import unittest

import numpy as np
import pandas as pd

import monthly_monotone_path212_independent as independent


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


def minmax_projection(z):
    # Closed-form max-min characterization, exhaustive only for tiny toys.
    return [max(min(math.fsum(z[a:b+1])/(b-a+1) for b in range(i,len(z)))
                for a in range(i+1)) for i in range(len(z))]


class Tests(unittest.TestCase):
    def test_projection_against_exhaustive_minmax_and_kkt(self):
        for length in range(1,6):
            for z in itertools.product((-1.,0.,2.),repeat=length):
                blocks,fit=independent.isotonic_oracle(list(z))
                np.testing.assert_allclose(fit,minmax_projection(z),rtol=0,atol=2e-15)
                independent.certify_projection(list(z),fit,blocks)
        blocks,fit=independent.isotonic_oracle([0.,0.,2.,1.,3.])
        self.assertEqual(blocks,[dict(start_index=0,end_index=1,count=2,mean=0.),
            dict(start_index=2,end_index=3,count=2,mean=1.5),
            dict(start_index=4,end_index=4,count=1,mean=3.)])

    def test_kkt_certificate_rejects_wrong_monotone_projection(self):
        z=[3.,0.,2.]
        # Nondecreasing and whole-block mean is insufficient without dual signs.
        blocks=[dict(start_index=0,end_index=2,count=3,mean=5./3.)]
        with self.assertRaises(ValueError):independent.certify_projection(z,[5./3.]*3,blocks)
        blocks,fit=independent.isotonic_oracle(z)
        fit[-1]+=1e-6
        with self.assertRaises(ValueError):independent.certify_projection(z,fit,blocks)

    def test_exact_path_branches_and_step_difference(self):
        for z,iso,line in (([0.,1.,2.],1.,1.),([2.,1.,0.],0.,0.)):
            mean=math.fsum(z)/len(z);sst=math.fsum((x-mean)**2 for x in z)
            result=independent.path_fits(z,mean,sst)
            self.assertEqual(result['q_iso'],iso);self.assertEqual(result['q_line'],line)
        a=independent.measure_oracle(prices('step'))
        self.assertEqual(a['q_iso'],1.)
        self.assertLess(a['q_line'],a['q_iso'])
        # Repeated log levels can leave a tiny mean-rounding residual; no snap.
        self.assertLess(a['sse_iso'],1e-28)
        self.assertGreater(a['raw_candidate'],a['raw_control'])

    def test_nonpositive_momentum_exactly_preserved(self):
        p=prices('wave');p[0]=p[252]*2
        a=independent.measure_oracle(p)
        self.assertTrue(a['available']);self.assertEqual(a['numerator'],-.5)
        self.assertEqual(a['raw_candidate'],a['normalized_momentum'])
        self.assertEqual(a['raw_control'],a['normalized_momentum'])
        self.assertFalse(a['attenuation_applied'])
        p[0]=p[252]
        a=independent.measure_oracle(p)
        self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)

    def test_zero_variance_and_zero_volatility_joint_support(self):
        a=independent.measure_oracle(prices('flat'))
        self.assertTrue(a['source_available']);self.assertFalse(a['available'])
        self.assertEqual(a['reason'],'zero_formation_variance')
        self.assertGreater(a['volatility'],0.);self.assertEqual(a['sst'],0.)
        p=prices();p[231:]=p[231]
        a=independent.measure_oracle(p)
        self.assertEqual(a['reason'],'zero_volatility')
        self.assertEqual(a['descriptive_pava_fits'],0)
        self.assertEqual(a['blocks'],[])
        self.assertEqual(independent.measure_oracle(np.ones(295))['reason'],'zero_volatility')

    def test_clamping_only_roundoff_not_general_quality_filter(self):
        self.assertEqual(independent.quality(1.+5e-13,1.)[0],0.)
        with self.assertRaises(ValueError):independent.quality(1.+2e-12,1.)
        self.assertEqual(independent.quality(0.,1.),(1.,1.,'zero_sse'))
        self.assertEqual(independent.quality(1.,1.,'single_block'),(0.,0.,'single_block'))

    def test_sample_std_and_original_current_return_clocks(self):
        p=prices();a=independent.measure_oracle(p)
        r=[p[j]/p[j-1]-1 for j in range(232,295)]
        self.assertEqual(a['simple_returns'],r)
        self.assertAlmostEqual(a['volatility'],float(np.std(r,ddof=1)),places=15)
        self.assertEqual(a['numerator'],p[252]/p[0]-1)
        p[-1]*=1.3;b=independent.measure_oracle(p)
        self.assertEqual(a['formation_logs'],b['formation_logs'])
        self.assertEqual(a['q_iso'],b['q_iso'])
        self.assertNotEqual(a['volatility'],b['volatility'])

    def test_complete295_numeric_nonboolean_positive_source(self):
        for idx in (0,1,160,230,231,252,253,294):
            for bad in (0.,-1.,np.nan,np.inf):
                p=prices();p[idx]=bad
                a=independent.measure_oracle(p)
                self.assertEqual(a['reason'],'invalid_adjusted_price')
                self.assertFalse(a['source_available'])
        for bad in (True,None,'2',complex(2,0)):
            p=prices().astype(object);p[120]=bad
            self.assertEqual(independent.measure_oracle(p)['reason'],'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(294))

    def test_valid_input_arithmetic_nonfinite_hardfails(self):
        for idx,value in ((0,5e-324),(293,5e-324)):
            p=prices();p[idx]=value
            with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=prices();p[293]=1.;p[294]=1e200
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        a=independent.measure_oracle(prices()*2.**500)
        self.assertTrue(a['available'])

    def test_original_pct_ties_singletons_and_missing(self):
        for row in ([1.,1.,1.],[2.],[np.nan,np.nan],[-1.,0.,1.,1.]):
            x=np.asarray([row]);mask=np.isfinite(x)
            actual=independent.original_pct_ranks(x,mask)
            np.testing.assert_array_equal(actual[0],pd.Series(row).rank(pct=True).to_numpy()-.5)
        self.assertEqual(independent.original_pct_ranks(np.array([[1.]]),np.array([[True]]))[0,0],.5)

    def test_complete_audit_and_descriptive_not_predictive_counts(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['source_available_by_event'],[4]*3)
        self.assertEqual(a['available_by_event'],[3]*3)
        for key,mult in (('valid_price_observations',295),('formation_log_observations',253),
                         ('simple_return_observations',63)):
            self.assertEqual(a[key],mult*12)
        self.assertEqual(a['descriptive_pava_fits'],9)
        self.assertEqual(a['descriptive_affine_fits'],9)
        self.assertEqual(a['ranked_observations'],18)
        r=independent.verify_features(p,m,c,v,a)
        self.assertTrue(all(r[k] is True for k in independent.FEATURE_FLAGS))
        self.assertEqual(r['new_predictive_model_fits'],0)
        self.assertEqual(r['numerical_tolerance'],0.)
        self.assertEqual(r['kkt_relative_tolerance'],1e-10)
        self.assertTrue(v['available'][:,2].all()) # ETF included, never eligibility screened.

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

    def test_corrupt_arrays_ranks_audit_certificate_and_counts_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for key in set(v)-{'available'}:
            bad=copy.deepcopy(v);bad[key][1,0]=np.nextafter(bad[key][1,0],np.inf)
            with self.assertRaises(ValueError):independent.verify_features(p,m,c,bad,a)
        for key in ('descriptive_pava_fits','new_predictive_model_fits'):
            bad=copy.deepcopy(a);bad[key]+=1
            with self.assertRaises(ValueError):independent.verify_features(p,m,c,v,bad)
        bad=copy.deepcopy(a);bad['event_history'][0]['cells'][0]['fitted_iso'][0]+=1e-6
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

    def test_delta_invalid_state_weight_target_and_future_prefix(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);original=arrays()
        tt=targets(v,original);dd=independent.independent_deltas(v,original,tt)
        for key,value in (('core_weight',np.full((3,1),.51)),('state_calm10',np.array([0,0,1]))):
            bad=copy.deepcopy(original);bad[key]=value
            with self.assertRaises(ValueError):independent.verify_deltas(v,bad,tt,dd)
        bad=copy.deepcopy(tt);bad[independent.ARMS[1]][0]=0.
        with self.assertRaises(ValueError):independent.verify_deltas(v,original,bad,dd)
        for n in range(4):
            short={k:x[:n].copy() for k,x in v.items()}
            orig=copy.deepcopy(original)
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
        self.assertFalse(any('212_kernel' in str(x) for x in files))

    def test_both_producer_and_independent_fixtures_complete_audit_all_prefixes(self):
        import monthly_monotone_path212_kernel as producer
        import test_monthly_monotone_path212_kernel as producer_test
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
