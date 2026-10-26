"""Only synthetic paired blocks, source clocks, ranking and prefix validation."""
import copy
import math
import unittest
import numpy as np
import pandas as pd
import monthly_breadth_pair202_kernel as k


def mapper(day,symbols,sectors,etfs):
    mapping={'A':'XLK','C':'XLC','BABA':'XLY'}
    return {s:None if s in etfs else mapping.get(s) for s in symbols}


def fixture():
    dates=pd.bdate_range('2014-01-01','2017-01-04');x=np.arange(len(dates),dtype=float)
    names=['A','C','BABA','FUND']+list(k.BREADTH_ETFS)+['XLC','XLRE']
    px=pd.DataFrame({s:100.+i+.015*x+3*np.sin(x*(.07+.023*i)) for i,s in enumerate(names)},index=dates)
    sectors=pd.DataFrame({'A':'Technology','C':'Communication Services','BABA':'Technology'},index=dates)
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],symbols=[['A','C','BABA','FUND']]*3,etfs=['FUND'])
    return {'adj':px,'sectors':sectors},meta,dates.strftime('%Y-%m-%d').tolist()


def toy(reverse=False,m=8):
    stock=np.ones(253)*100.;benchmark=np.ones(253)*100.;breadth=np.ones((253,m))*100.
    for b in range(1,13):
        end=21*b;previous=end-21
        stock[end]=stock[previous]*(1.05 if b%2 else .98)
        weak=(b%2==1) != reverse
        breadth[end]=breadth[previous]*(.9 if weak else 1.1)
    return stock,benchmark,breadth


class MeasureTests(unittest.TestCase):
    def test_canonical_pairing_positive_and_negative_covariance(self):
        a=k.measure(*toy());b=k.measure(*toy(True))
        self.assertEqual(a['control'],b['control']);self.assertGreater(a['candidate'],a['control'])
        self.assertLess(b['candidate'],b['control']);self.assertGreater(a['covariance'],0.)
        e=a['excess_log_returns'];w=a['breadth'];c=math.fsum(e)/12.;wb=math.fsum(w)/12.
        self.assertEqual(a['candidate'],c+math.fsum((x-wb)*(y-c) for x,y in zip(w,e))/12.)
        self.assertEqual(a['endpoint_indices'],list(range(0,253,21)))
        self.assertEqual(a['down_counts'],[8,0]*6)
    def test_strict_down_ties_eight_or_nine_and_consumption_counts(self):
        p,b,x=toy(m=9);x[:,0]=100.;a=k.measure(p,b,x)
        self.assertEqual(a['down_counts'],[8,0]*6);self.assertEqual(a['breadth'],[8/9,0.]*6)
        self.assertEqual(a['valid_price_observations'],253*11);self.assertEqual(a['down_sign_comparisons'],108)
        self.assertEqual(a['covariance_estimates'],1)
    def test_exact_constant_breadth_and_excess_copy(self):
        p,b,x=toy();x[:]=100.;a=k.measure(p,b,x)
        self.assertEqual(a['constant_branch'],'all_breadth_equal');self.assertEqual(a['candidate'].hex(),a['control'].hex())
        p=b.copy();_,_,x=toy();a=k.measure(p,b,x)
        self.assertEqual(a['constant_branch'],'all_excess_equal');self.assertEqual(a['candidate'].hex(),a['control'].hex())
        self.assertEqual(a['covariance'],0.)
    def test_full_middle_sources_and_strict_nonnumeric(self):
        for ix,label in ((0,'stock'),(1,'benchmark'),(2,'breadth')):
            for val,reason in [(np.nan,'invalid'),(0.,'invalid'),(-1.,'invalid'),(np.inf,'invalid'),(True,'nonnumeric'),('2','nonnumeric'),(None,'nonnumeric')]:
                arrays=[x.astype(object) for x in toy()]
                if ix==2:arrays[ix][19,3]=val
                else:arrays[ix][19]=val
                a=k.measure(*arrays);self.assertEqual(a['reason'],reason+'_'+label+'_price')
                self.assertFalse(a['available']);self.assertEqual(a['excess_log_returns'],[])
                self.assertTrue(all(a[f]==0 for f in k.COUNT_FIELDS))
    def test_shape_invalid_before_numeric_and_extreme_finite_logs(self):
        p,b,x=toy()
        for pp,bb,xx in ((p[:-1],b,x),(p,b[:-1],x),(p,b,x[:,:7]),(p,b,x.reshape(-1))):
            with self.assertRaises(ValueError):k.measure(pp,bb,xx)
        p[::21]=[1e-300,1e300]*6+[1e-300]
        a=k.measure(p,b,x);self.assertTrue(a['available']);self.assertTrue(math.isfinite(a['candidate']))
    def test_source_immutability(self):
        arrays=toy();before=[x.copy() for x in arrays];k.measure(*arrays)
        for a,b in zip(arrays,before):np.testing.assert_array_equal(a,b)


class BuilderTests(unittest.TestCase):
    def test_mapping_baskets_dates_and_counts(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c,mapper)
        self.assertEqual(set(v),{'candidate','control','available','candidate_rank','control_rank'})
        self.assertEqual(a['available_by_event'],[3]*3);self.assertEqual(a['eight_sector_cells_by_event'],[2]*3)
        self.assertEqual(a['nine_sector_cells_by_event'],[1]*3);self.assertEqual(a['valid_price_observations'],253*31*3)
        self.assertEqual(a['down_sign_comparisons'],12*25*3)
        self.assertTrue(np.isfinite(v['candidate'][0,:3]).all());self.assertTrue((v['candidate_rank'][:,3]==0).all())
        for event in a['event_history']:
            t=event['session_position'];self.assertEqual(event['endpoint_dates'],[c[t-294+i] for i in range(0,253,21)])
            self.assertNotIn('XLK',event['cells'][0]['breadth_etfs']);self.assertEqual(event['cells'][1]['breadth_count'],9)
            self.assertNotIn('XLY',event['cells'][2]['breadth_etfs'])
    def test_own_excluded_from_breadth_still_required_and_unused_not_read(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);p['adj'].loc[c[t-100],'XLK']=np.nan
        v,a=k.build_features(p,m,c,mapper,limit=1)
        self.assertFalse(v['available'][0,0]);self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'invalid_benchmark_price')
        p,m,c=fixture();wanted,_=k.build_features(p,m,c,mapper,limit=1)
        p['adj']['XLRE']='unused poison';got,_=k.build_features(p,m,c,mapper,limit=1)
        for key in wanted:np.testing.assert_array_equal(wanted[key],got[key])
    def test_missing_priority_no_calendar_compression(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(index=c[t-100],columns='A')
        _,a=k.build_features(p,m,c,mapper,limit=1)
        self.assertEqual([z['reason'] for z in a['event_history'][0]['cells']],['missing_symbol','missing_official_session','missing_official_session','etf_neutral'])
        _,a=k.build_features(p,m,c[t-100:],mapper,limit=1)
        self.assertEqual(a['event_history'][0]['endpoint_dates'],None)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
    def test_unknown_missing_benchmark_and_missing_basket(self):
        p,m,c=fixture();unknown=lambda d,ss,s,e:{x:None for x in ss}
        v,a=k.build_features(p,m,c,unknown);self.assertFalse(v['available'].any())
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_benchmark')
        p['adj']=p['adj'].drop(columns='XLC');_,a=k.build_features(p,m,c,mapper,limit=1)
        self.assertEqual(a['event_history'][0]['cells'][1]['reason'],'missing_benchmark_symbol')
        p['adj']=p['adj'].drop(columns='XLB');_,a=k.build_features(p,m,c,mapper,limit=1)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_breadth_symbol')
    def test_symmetric_rank_all_ties_singleton_and_missing(self):
        np.testing.assert_array_equal(k.centered_rank(np.array([1.,2.,2.,4.,np.nan]),np.array([True]*4+[False])),[-.5,0.,0.,.5,0.])
        np.testing.assert_array_equal(k.centered_rank(np.ones(3),np.ones(3,bool)),np.zeros(3))
        np.testing.assert_array_equal(k.centered_rank(np.array([1.,np.nan]),np.array([True,False])),np.zeros(2))
        p,m,c=fixture();p['adj'].loc[:,['C','BABA']]=np.nan;v,a=k.build_features(p,m,c,mapper)
        self.assertEqual(a['available_by_event'],[1]*3);self.assertTrue(v['available'][:,0].all())
        self.assertTrue((v['candidate_rank']==0.).all())
    def test_all_prefixes_future_source_and_mapping_isolation(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c,mapper)
        for n in range(4):
            pp={key:frame.loc[:m['dates'][n-1]].copy() if n else frame.iloc[:0].copy() for key,frame in p.items()}
            vv,aa=k.build_features(pp,m,c,mapper,limit=n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
        t=c.index(m['dates'][0]);pp={key:frame.astype(object) for key,frame in p.items()}
        pp['adj'].loc[pd.Timestamp(c[t-41]):]='future poison'
        pp['sectors'].loc[pp['sectors'].index>m['dates'][0]]='future poison'
        def checked(day,ss,sectors,etfs):
            self.assertLessEqual(sectors.index[-1],pd.Timestamp(day));return mapper(day,ss,sectors,etfs)
        vv,aa=k.build_features(pp,m,c,checked,limit=1)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:1])
        self.assertEqual(aa['event_history'],a['event_history'][:1])
    def test_permutation_and_no_mutation(self):
        p,m,c=fixture();before=copy.deepcopy((p,m,c));v,_=k.build_features(p,m,c,mapper)
        perm=[2,0,3,1];mm=dict(m,symbols=[[ss[j] for j in perm] for ss in m['symbols']]);vv,_=k.build_features(p,mm,c,mapper)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:,perm])
        for key in p:pd.testing.assert_frame_equal(p[key],before[0][key])
        self.assertEqual(m,before[1]);self.assertEqual(c,before[2])
    def test_mapping_shape_metadata_and_identity_guards(self):
        p,m,c=fixture()
        for n in (-1,4,True,1.):
            with self.assertRaises(ValueError):k.build_features(p,m,c,mapper,n)
        for fn in (lambda *x:{},lambda d,ss,s,e:{x:'XLK' for x in ss}):
            with self.assertRaises(ValueError):k.build_features(p,m,c,fn)
        mm=dict(m,symbols=[['XLK','C','BABA','FUND']]*3)
        fn=lambda d,ss,s,e:{x:None if x in e else 'XLK' for x in ss}
        with self.assertRaises(ValueError):k.build_features(p,mm,c,fn)
        mm=copy.deepcopy(m);mm['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):k.build_features(p,mm,c,mapper)
        with self.assertRaises(ValueError):k.build_features(p,m,c[:c.index(m['dates'][-1])+1],mapper)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj']},m,c,mapper)


if __name__=='__main__':unittest.main(verbosity=2)
