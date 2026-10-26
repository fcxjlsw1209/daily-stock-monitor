"""Synthetic source clocks, robust dispersion, shared ranks and prefix tests."""
import copy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_median_risk201_kernel as k


def fixture():
    dates = pd.bdate_range('2014-01-01', '2017-01-04'); x=np.arange(len(dates),dtype=float)
    px=pd.DataFrame({'A':100.+.03*x+3*np.sin(.13*x), 'B':200.+.02*x+5*np.cos(.23*x),
                     'FUND':80.+.02*x+4*np.sin(.37*x)},index=dates)
    return {'adj':px},dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
        symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND']),dates.strftime('%Y-%m-%d').tolist()


def prices(returns=None):
    p=np.ones(295)*100.
    if returns is None: returns=[-.02,.01,.03]*21
    for i,r in enumerate(returns,232):p[i]=p[i-1]*(1+r)
    return p


class MeasureTests(unittest.TestCase):
    def test_exact_arithmetic_order_and_window(self):
        p=prices();p[0]=80.
        a=k.measure(p);r=[float(p[i])/float(p[i-1])-1 for i in range(232,295)]
        med=sorted(r)[31];dev=[abs(v-med) for v in r]
        self.assertEqual(a['simple_returns'],r);self.assertEqual(a['absolute_deviations'],dev)
        self.assertEqual(a['numerator'],p[252]/p[0]-1)
        self.assertEqual(a['median_return'],med);self.assertEqual(a['median_deviation'],sorted(dev)[31])
        self.assertEqual(a['mean_deviation'],math.fsum(dev)/63)
        self.assertEqual(a['raw_candidate'],a['numerator']/sorted(dev)[31])
        self.assertEqual(a['raw_control'],a['numerator']/(math.fsum(dev)/63))
        p[230]=1e100;self.assertEqual(k.measure(p)['simple_returns'],r)
    def test_zero_median_and_positive_mean_joint_missing(self):
        p=np.ones(295)*100.;p[-1]=200.;a=k.measure(p)
        self.assertTrue(a['source_available']);self.assertFalse(a['available'])
        self.assertEqual(a['median_deviation'],0.);self.assertGreater(a['mean_deviation'],0.)
        self.assertEqual(a['reason'],'zero_dispersion');self.assertEqual(a['dispersion_estimates'],2)
        self.assertTrue(np.isnan(a['raw_candidate']) and np.isnan(a['raw_control']))
    def test_constant_zero_dispersion_and_finite_zero_numerator(self):
        a=k.measure(np.ones(295));self.assertEqual(a['dispersion_estimates'],2);self.assertFalse(a['available'])
        p=prices();p[0]=p[252];a=k.measure(p)
        self.assertTrue(a['available']);self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)
    def test_full295_source_and_numeric_nonboolean(self):
        for val,reason in [(None,'nonnumeric_adjusted_price'),(True,'nonnumeric_adjusted_price'),
                ('100','nonnumeric_adjusted_price'),(1j,'nonnumeric_adjusted_price'),
                (0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price'),(np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price')]:
            p=list(prices());p[30]=val;a=k.measure(p)
            self.assertEqual(a['reason'],reason);self.assertEqual(a['dispersion_estimates'],0)
            self.assertFalse(a['source_available']);self.assertEqual(a['simple_returns'],[])
        for p in (np.ones(294),np.ones((295,1)),1.):
            with self.assertRaises(ValueError):k.measure(p)
    def test_overflow_is_hard_failure(self):
        for where in ('numerator','return','sum','raw'):
            p=prices()
            if where=='numerator':p[0]=1e-308;p[252]=1e308
            if where=='return':p[231]=1e-308;p[232]=1e308
            if where=='sum':
                p[:]=1.
                for i in range(232,295):p[i]=1e308 if i%2==0 else 1.
            if where=='raw':p[:]=prices([-.0001,.0002,.0003]*21);p[0]=1e-304;p[252]=100.
            with self.subTest(where=where),self.assertRaises(FloatingPointError):k.measure(p)
    def test_tiny_positive_deviation_not_floored(self):
        p=prices([1e-14,-2e-14,3e-14]*21);a=k.measure(p)
        self.assertTrue(a['available']);self.assertGreater(a['median_deviation'],0)
        self.assertLess(a['median_deviation'],1e-12)
    def test_scale_invariance_and_source_immutability(self):
        p=prices();before=p.copy();a=k.measure(p)
        for scale in (2.**-500,2.**500):
            b=k.measure(p*scale)
            for key in ('simple_returns','absolute_deviations','raw_candidate','raw_control','median_return','numerator'):
                self.assertEqual(a[key],b[key])
        np.testing.assert_array_equal(p,before)


class BuilderTests(unittest.TestCase):
    def test_schema_counts_and_clocks(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(set(v),{'candidate','control','raw_candidate','raw_control','available'})
        self.assertEqual(a['source_available_by_event'],[3]*3);self.assertEqual(a['available_by_event'],[3]*3)
        for key,expected in dict(valid_price_observations=295*9,simple_return_observations=63*9,
                dispersion_estimates=18,ratio_observations=18,ranked_score_observations=18,new_model_fits=0).items():self.assertEqual(a[key],expected)
        self.assertTrue(np.isfinite(v['candidate'][0]).all())
        for row in a['event_history']:
            t=row['session_position']
            self.assertEqual([row[x] for x in ('price_start','price_end','numerator_start','numerator_end','risk_price_start','risk_price_end')],
                [c[t-294],c[t],c[t-294],c[t-42],c[t-63],c[t]])
    def test_original_rank_ties_empty_and_singleton(self):
        np.testing.assert_array_equal(k.percentile_rank(np.array([1.,2.,2.,4.,np.nan]),np.array([True]*4+[False])),[-.25,.125,.125,.5,np.nan])
        np.testing.assert_array_equal(k.percentile_rank(np.ones(3),np.ones(3,bool)),[2/3-.5]*3)
        np.testing.assert_array_equal(k.percentile_rank(np.array([1.,np.nan]),np.array([True,False])),[.5,np.nan])
        p,m,c=fixture();p['adj'].loc[:,['B','FUND']]=np.nan;v,a=k.build_features(p,m,c)
        self.assertEqual(a['available_by_event'],[1]*3)
        for key in ('candidate','control'):np.testing.assert_array_equal(v[key][:,0],np.full(3,.5))
        p['adj'].loc[:,:]=100.;v,a=k.build_features(p,m,c)
        self.assertEqual(a['source_available_cells'],9);self.assertEqual(a['available_cells'],0)
        self.assertEqual(a['dispersion_estimates'],18);self.assertEqual(a['ratio_observations'],0)
        self.assertTrue(all(np.isnan(x).all() for key,x in v.items() if key!='available'))
    def test_missing_priority_and_full_calendar_support(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(index=c[t-100],columns='B')
        _,a=k.build_features(p,m,c,limit=1)
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells']],['missing_official_session','missing_symbol','missing_official_session'])
        _,a=k.build_features(p,m,c[t-100:],limit=1)
        self.assertEqual(a['event_history'][0]['price_end'],None)
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells']],['insufficient_history']*3)
    def test_current_risk_not_skip42_and_future_isolated(self):
        p,m,c=fixture();expected,a=k.build_features(p,m,c,limit=1)
        pp={'adj':p['adj'].astype(object)};pp['adj'].loc[pp['adj'].index>m['dates'][0]]='future poison'
        pp['adj']['UNUSED']='unused poison';actual,b=k.build_features(pp,m,c,limit=1)
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(a,b)
        pp['adj'].loc[m['dates'][0],'A']='current poison';actual,b=k.build_features(pp,m,c,limit=1)
        self.assertFalse(actual['available'][0,0])
    def test_all_prefixes_permutation_and_no_mutation(self):
        p,m,c=fixture();px=p['adj'].copy(deep=True);mm=copy.deepcopy(m);v,a=k.build_features(p,m,c)
        for n in range(4):
            pp={'adj':p['adj'].loc[:m['dates'][n-1]].copy() if n else p['adj'].iloc[:0].copy()}
            vv,aa=k.build_features(pp,m,c,limit=n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
        perm=[2,0,1];m2=dict(m,symbols=[[s[j] for j in perm] for s in m['symbols']]);vv,_=k.build_features(p,m2,c)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:,perm])
        pd.testing.assert_frame_equal(px,p['adj']);self.assertEqual(m,mm)
    def test_metadata_invalid_prefix_and_monthend(self):
        p,m,c=fixture()
        for n in (-1,4,True,1.):
            with self.assertRaises(ValueError):k.build_features(p,m,c,n)
        for key,value in [('dates',['2016-08-17','2016-08-30','2016-09-30']),('symbols',[['A'],['A','B'],['A']]),('etfs',['FUND','FUND'])]:
            bad=dict(m);bad[key]=value
            with self.assertRaises(ValueError):k.build_features(p,bad,c)
        with self.assertRaises(ValueError):k.build_features(p,m,c[:c.index(m['dates'][-1])+1])
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'],'close':p['adj']},m,c)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'].iloc[::-1]},m,c)


if __name__=='__main__':unittest.main(verbosity=2)
