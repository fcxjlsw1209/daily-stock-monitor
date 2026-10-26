import unittest
import numpy as np
import pandas as pd
import monthly_cmf_state200_kernel as k

class Tests(unittest.TestCase):
    def test_increasing_extreme(self):
        r=k.market_gate(np.arange(252.));self.assertAlmostEqual(r['gate_candidate'],251/252);self.assertEqual(r['state'],1.)
    def test_decreasing_extreme(self):
        r=k.market_gate(np.arange(252.)[::-1]);self.assertAlmostEqual(r['gate_candidate'],-251/252);self.assertEqual(r['state'],-1.)
    def test_constant(self):
        r=k.market_gate(np.ones(252));self.assertEqual(r['gate_candidate'],0.);self.assertEqual(r['state'],-1.)
    def test_asymmetric_median_tie(self):
        for x in (np.r_[np.zeros(50),np.ones(202)],np.r_[np.full(50,2.),np.ones(202)]):
            r=k.market_gate(x);self.assertEqual(r['gate_candidate'],0.);self.assertEqual(r['state'],-1.)
    def test_midpoint_not_observed(self):
        r=k.market_gate(np.r_[np.zeros(126),np.ones(126)])
        self.assertEqual(r['median'],.5);self.assertEqual(r['gate_candidate'],.5)
    def test_invalid(self):
        for x in (np.zeros(251),np.zeros((252,1)),np.r_[np.zeros(251),np.nan],np.r_[np.zeros(251),np.inf],np.r_[np.zeros(251),-1.]):
            with self.assertRaises(ValueError):k.market_gate(x)
    def test_ranks_ties_missing(self):
        np.testing.assert_array_equal(k.cmf_ranks([1,1,2,np.nan]),[0.,0.,.5,0.])
    def test_rank_singleton(self):
        np.testing.assert_array_equal(k.cmf_ranks([np.nan,2]),[0.,.5])
    def test_rank_invalid(self):
        with self.assertRaises(ValueError):k.cmf_ranks([1,np.inf])
    def fixture(self):
        sessions=pd.bdate_range('2014-01-01','2016-10-31')
        px=100*np.cumprod(1+.001+.01*np.sin(np.arange(len(sessions))*.8))
        adj=pd.DataFrame({'SPY':px,'A':px*2,'B':px*3},index=sessions)
        close=adj.copy();high=close*1.01;low=close*.99
        panel=dict(adj=adj,close=close,high=high,low=low,volume=adj*0+100)
        meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],symbols=[['A','B']]*3,etfs=[])
        return panel,meta,sessions.strftime('%Y-%m-%d').tolist()
    def test_build_shape_and_counts(self):
        p,m,c=self.fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(v['cmf_rank'].shape,(3,2));self.assertEqual(v['sigma_window'].shape,(3,252))
        self.assertEqual(a['spy_price_observations'],945);self.assertEqual(a['new_model_fits'],0)
    def test_prefix_physical(self):
        p,m,c=self.fixture();v,a=k.build_features(p,m,c)
        for n in range(4):
            small={key:f.iloc[:0].copy() if n==0 else f.loc[:m['dates'][n-1]].copy() for key,f in p.items()}
            z,b=k.build_features(small,m,c,n)
            for key in v:np.testing.assert_array_equal(z[key],v[key][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])
    def test_future_poison(self):
        p,m,c=self.fixture();v,_=k.build_features(p,m,c,1)
        for key in p:
            p[key]=p[key].astype(object);p[key].loc[p[key].index>m['dates'][0],:]='FUTURE'
        z,_=k.build_features(p,m,c,1)
        for key in v:np.testing.assert_array_equal(z[key],v[key])
    def test_missing_official_session(self):
        p,m,c=self.fixture()
        for key in p:p[key]=p[key].drop(pd.Timestamp('2016-08-01'))
        with self.assertRaises(ValueError):k.build_features(p,m,c)
    def test_invalid_spy(self):
        for value in (0.,np.nan,-1.):
            p,m,c=self.fixture();p['adj'].loc['2016-08-01','SPY']=value
            with self.assertRaises(ValueError):k.build_features(p,m,c)
    def test_missing_cmf_neutral_and_zero_range(self):
        p,m,c=self.fixture();p['volume']['A']=0.;p['high']['B']=p['low']['B']
        v,a=k.build_features(p,m,c)
        np.testing.assert_array_equal(v['cmf_rank'],[[0.,.5]]*3)
        self.assertEqual(a['neutral_cmf_cells'],3)
    def test_invalid_limit(self):
        p,m,c=self.fixture()
        for n in (-1,4,True,1.5):
            with self.assertRaises(ValueError):k.build_features(p,m,c,n)

if __name__=='__main__':unittest.main()
