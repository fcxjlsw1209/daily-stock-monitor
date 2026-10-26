"""Synthetic current-risk adjacency, shared ranks, source clocks and prefix tests."""
import copy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_adjacent_risk205_kernel as k


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
        a=k.measure(p);r=[float(p[i])/float(p[i-1])-1. for i in range(232,295)]
        pairs=[abs(r[i])*abs(r[i-1]) for i in range(1,63)];squares=[v*v for v in r]
        self.assertEqual(a['simple_returns'],r);self.assertEqual(a['absolute_adjacent_products'],pairs)
        self.assertEqual(a['squared_returns'],squares);self.assertEqual(a['numerator'],float(p[252])/float(p[0])-1.)
        self.assertEqual(a['adjacent_sum'],math.fsum(pairs));self.assertEqual(a['squared_sum'],math.fsum(squares))
        vpair=(math.pi/2.)*math.fsum(pairs)/62.;vrms=math.fsum(squares)/63.
        self.assertEqual(a['candidate_variance'],vpair);self.assertEqual(a['control_variance'],vrms)
        self.assertEqual(a['candidate_denominator'],math.sqrt(vpair));self.assertEqual(a['control_denominator'],math.sqrt(vrms))
        self.assertEqual(a['raw_candidate'],a['numerator']/math.sqrt(vpair))
        self.assertEqual(a['raw_control'],a['numerator']/math.sqrt(vrms))
        self.assertEqual([a[x] for x in ('start_price','formation_end_price','risk_start_price','end_price')],list(p[[0,252,231,294]]))
        p[230]=1e100;self.assertEqual(k.measure(p)['simple_returns'],r)
    def test_isolated_jump_zero_pair_positive_rms_joint_missing(self):
        p=np.ones(295)*100.;p[-1]=200.;a=k.measure(p)
        self.assertTrue(a['source_available']);self.assertFalse(a['available'])
        self.assertEqual(a['candidate_denominator'],0.);self.assertGreater(a['control_denominator'],0.)
        self.assertEqual(a['reason'],'zero_risk');self.assertEqual(a['risk_estimates'],2)
        self.assertEqual(len(a['simple_returns']),63);self.assertEqual(len(a['absolute_adjacent_products']),62)
        self.assertTrue(np.isnan(a['raw_candidate']) and np.isnan(a['raw_control']))
    def test_flat_risk_zero_and_finite_zero_numerator(self):
        a=k.measure(np.ones(295));self.assertEqual(a['risk_estimates'],2);self.assertFalse(a['available'])
        self.assertEqual(a['candidate_denominator'],0.);self.assertEqual(a['control_denominator'],0.)
        p=prices();p[0]=p[252];a=k.measure(p)
        self.assertTrue(a['available']);self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)
    def test_constant_nonzero_returns_are_not_demeaned(self):
        a=k.measure(prices([1.]*63))
        self.assertTrue(a['available']);self.assertEqual(a['simple_returns'],[1.]*63)
        self.assertEqual(a['adjacent_sum'],62.);self.assertEqual(a['squared_sum'],63.)
        self.assertEqual(a['candidate_denominator'],math.sqrt((math.pi/2.)*62./62.))
        self.assertEqual(a['control_denominator'],1.)
    def test_same_return_multiset_reordering_changes_pair_not_rms(self):
        one=[0.]*63;one[30:34]=[1.,.5,0.,.5]
        two=[0.]*63;two[30:34]=[1.,0.,.5,.5]
        p,q=prices(one),prices(two);p[0]=q[0]=80.;a,b=k.measure(p),k.measure(q)
        self.assertEqual(sorted(a['simple_returns']),sorted(b['simple_returns']))
        self.assertTrue(a['available'] and b['available']);self.assertEqual(a['numerator'],b['numerator'])
        self.assertEqual(a['control_denominator'],b['control_denominator']);self.assertEqual(a['raw_control'],b['raw_control'])
        self.assertNotEqual(a['candidate_denominator'],b['candidate_denominator']);self.assertNotEqual(a['raw_candidate'],b['raw_candidate'])
    def test_nonadjacent_jumps_joint_missing_adjacent_jumps_available(self):
        separated=[0.]*63;separated[30]=separated[32]=1.
        adjacent=[0.]*63;adjacent[30]=adjacent[31]=1.
        a,b=k.measure(prices(separated)),k.measure(prices(adjacent))
        self.assertFalse(a['available']);self.assertTrue(b['available'])
        self.assertEqual(a['adjacent_sum'],0.);self.assertEqual(b['adjacent_sum'],1.)
        self.assertEqual(a['control_denominator'],b['control_denominator'])
    def test_endpoint_pair_count_without_wraparound(self):
        r=[0.]*63;r[:2]=[.5,1.];r[-2:]=[1.,.5];a=k.measure(prices(r))
        self.assertEqual(len(a['absolute_adjacent_products']),62)
        self.assertEqual(a['absolute_adjacent_products'][0],.5);self.assertEqual(a['absolute_adjacent_products'][-1],.5)
        self.assertEqual(a['adjacent_sum'],1.)
    def test_products_use_absolute_magnitudes(self):
        up=[0.]*63;up[:2]=[.5,.5]
        down=[0.]*63;down[:2]=[.5,-.5]
        a,b=k.measure(prices(up)),k.measure(prices(down))
        self.assertEqual(a['absolute_adjacent_products'],b['absolute_adjacent_products'])
        self.assertEqual(a['squared_returns'],b['squared_returns'])
        self.assertEqual(a['candidate_denominator'],b['candidate_denominator'])
        self.assertEqual(a['control_denominator'],b['control_denominator'])
    def test_full295_source_and_numeric_nonboolean(self):
        for val,reason in [(None,'nonnumeric_adjusted_price'),(True,'nonnumeric_adjusted_price'),
                ('100','nonnumeric_adjusted_price'),(1j,'nonnumeric_adjusted_price'),
                (0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price'),(np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price')]:
            p=list(prices());p[30]=val;a=k.measure(p)
            self.assertEqual(a['reason'],reason);self.assertEqual(a['risk_estimates'],0)
            self.assertFalse(a['source_available']);self.assertEqual(a['simple_returns'],[])
            self.assertEqual(a['absolute_adjacent_products'],[]);self.assertEqual(a['squared_returns'],[])
        for p in (np.ones(294),np.ones((295,1)),1.):
            with self.assertRaises(ValueError):k.measure(p)
    def test_nonfinite_arithmetic_is_hard_failure(self):
        for where in ('numerator','return','square','product','sum','variance','raw'):
            p=prices()
            if where=='numerator':p[0]=1e-308;p[252]=1e308
            if where=='return':p[231]=1e-308;p[232]=1e308
            if where=='square':p[:]=1.;p[232]=1e200
            if where=='product':p[:]=1.;p[231]=1e-200;p[232]=1.;p[233]=1e200
            if where=='sum':p[:]=1.;p[232]=p[234]=1e154
            if where=='variance':
                p[:]=1.;p[231]=1e-320
                for i in range(232,236):p[i]=p[i-1]*6.5e153
            if where=='raw':p=prices([1e-14,-2e-14,3e-14]*21);p[0]=1e-298
            with self.subTest(where=where),self.assertRaises(FloatingPointError):k.measure(p)
    def test_tiny_positive_risk_not_floored(self):
        a=k.measure(prices([1e-14,-2e-14,3e-14]*21))
        self.assertTrue(a['available'])
        for key in ('candidate_denominator','control_denominator'):
            self.assertGreater(a[key],0.);self.assertLess(a[key],1e-12)
    def test_power_of_two_scale_invariance_and_source_immutability(self):
        p=prices();before=p.copy();a=k.measure(p)
        for scale in (2.**-500,2.**500):
            b=k.measure(p*scale)
            for key in ('simple_returns','absolute_adjacent_products','squared_returns','adjacent_sum','squared_sum',
                    'candidate_variance','control_variance','candidate_denominator','control_denominator','raw_candidate','raw_control','numerator'):
                self.assertEqual(a[key],b[key])
        np.testing.assert_array_equal(p,before)


class BuilderTests(unittest.TestCase):
    def test_schema_counts_and_clocks(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(set(v),{'candidate','control','raw_candidate','raw_control','available'})
        self.assertEqual(a['source_available_by_event'],[3]*3);self.assertEqual(a['available_by_event'],[3]*3)
        for key,expected in dict(valid_price_observations=295*9,simple_return_observations=63*9,
                adjacent_product_observations=62*9,squared_return_observations=63*9,risk_estimates=18,
                ratio_observations=18,ranked_score_observations=18,new_model_fits=0).items():self.assertEqual(a[key],expected)
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
        self.assertEqual(a['risk_estimates'],18);self.assertEqual(a['ratio_observations'],0)
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

    def test_joint_missing_and_finite_zero_numerator_use_common_ranks(self):
        p,m,c=fixture();day=m['dates'][0];t=c.index(day);p['adj'].loc[:,:]=100.
        p['adj'].loc[c[t-1]:day,['A','FUND']]=np.array([[150.,150.],[225.,225.]])
        p['adj'].loc[day,'B']=200.
        v,a=k.build_features(p,m,c,limit=1)
        self.assertEqual(a['source_available_by_event'],[3]);self.assertEqual(a['available_by_event'],[2])
        self.assertEqual(a['risk_estimates'],6);self.assertEqual(a['ratio_observations'],4)
        np.testing.assert_array_equal(v['available'],[[True,False,True]])
        for key in ('raw_candidate','raw_control'):np.testing.assert_array_equal(v[key],[[0.,np.nan,0.]])
        for key in ('candidate','control'):np.testing.assert_array_equal(v[key],[[.25,np.nan,.25]])
        self.assertEqual(a['event_history'][0]['cells'][1]['reason'],'zero_risk')
    def test_rank_validation_empty_mask_and_full_tie_domain(self):
        np.testing.assert_array_equal(k.percentile_rank(np.zeros(4),np.zeros(4,bool)),[np.nan]*4)
        for raw,mask in ((np.zeros(2),np.ones(2,int)),(np.zeros(2),np.ones(1,bool)),
                         (np.array([np.inf]),np.ones(1,bool)),(np.zeros((1,2)),np.ones((1,2),bool))):
            with self.assertRaises(ValueError):k.percentile_rank(raw,mask)


if __name__=='__main__':unittest.main(verbosity=2)
