import unittest
import numpy as np
import pandas as pd
from ordered_study import (P,read,pack,basis,pairs,objective,limits,panels,
                           load_prices,make_pack,ranks_and_picks,metrics,fit)

class OrderedResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=read(P/'INPUTS.json');cls.d=pack('top32_theme17_percentile',cls.meta)

    def test_original_order_preserved(self):
        for p,ss,ids in zip(self.meta['periods'],self.d['symbols'],self.d['order']):
            from ordered_study import actual
            for a,j in zip(p['holdings'],ids):
                if j>=0:self.assertEqual(ss[j],actual(a))

    def test_order_changes_loss_but_not_membership(self):
        X,spec=basis(self.d,self.meta['cores'][0],1.,False)
        mask=np.zeros(len(X),bool);mask[0]=True
        D,pw,g,mo=pairs(X,self.d,mask)
        flipped=dict(self.d,order=self.d['order'].copy());flipped['order'][0]=flipped['order'][0][::-1]
        D2,pw2,g2,mo2=pairs(X,flipped,mask)
        np.testing.assert_allclose(sorted(map(tuple,D[g==0])),sorted(map(tuple,D2[g2==0])))
        np.testing.assert_allclose(D[g==1],-D2[g2==1][::-1])
        self.assertAlmostEqual(pw[g==0].sum(),.5)
        self.assertAlmostEqual(pw[g==1].sum(),.5)

    def test_gradient(self):
        rng=np.random.default_rng(6);D=rng.normal(size=(50,5));pw=np.full(50,1/50);w=rng.random(5);prior=rng.random(5)
        val,grad=objective(w,D,pw,.01,prior)
        eps=1e-6;numeric=[]
        for k in range(5):
            direction=np.eye(5)[k]*eps
            numeric.append((objective(w+direction,D,pw,.01,prior)[0]-objective(w-direction,D,pw,.01,prior)[0])/(2*eps))
        np.testing.assert_allclose(grad,numeric,rtol=1e-6,atol=1e-8)

    def test_frozen_weights_constraints(self):
        for row in read(P/'TRIALS.json'):
            A,b,bounds,prior=limits(row['spec']);w=np.array(row['weights'])
            self.assertAlmostEqual(w.sum(),1.,places=8)
            self.assertTrue((A@w<=b+1e-8).all())
            for x,(lo,hi) in zip(w,bounds):self.assertTrue(lo-1e-8<=x<=hi+1e-8)
            self.assertGreaterEqual(w[0],.5-1e-8)

    def test_future_author_labels_do_not_affect_fit(self):
        mask=np.array(self.meta['development']);X,spec=basis(self.d,self.meta['cores'][0],1.,False)
        w,obj=fit(X,self.d,mask,spec,.01)
        altered=dict(self.d,order=self.d['order'].copy(),labels=self.d['labels'].copy())
        altered['order'][~mask]=-1;altered['labels'][~mask]=False
        w2,obj2=fit(X,altered,mask,spec,.01)
        np.testing.assert_allclose(w,w2,atol=1e-12,rtol=0);self.assertEqual(obj,obj2)

    def test_sparse_eligible_cash_and_order_metrics(self):
        scores=np.array([[.3,.2,.1],[.1,.4,.2]])
        d=dict(eligible=np.array([[True,False,False],[True,True,True]]),labels=np.ones((2,3),bool),order=np.array([[0,-1,-1],[1,2,0]]))
        ranks,picks=ranks_and_picks(scores,d['eligible']);self.assertEqual(picks,[[],[1,2,0]])
        m=metrics(scores,d,np.array([False,True]));self.assertEqual(m['order_exact'],1);self.assertEqual(m['pairs_correct'],3)

    def test_features_use_no_future_prices(self):
        px,raw,members,periods,sources,pools=load_prices();ss=['SPY','QQQ','AAPL','NVDA']
        px=px[ss];raw={k:v[ss] for k,v in raw.items()}
        full,cores,states=panels(px,raw);cut='2023-06-30'
        prefix,_,st=panels(px.loc[:cut],{k:v.loc[:cut] for k,v in raw.items()})
        for k in full:pd.testing.assert_series_equal(full[k].loc[cut],prefix[k].loc[cut],check_exact=False,rtol=1e-12,atol=1e-12)
        pd.testing.assert_series_equal(states.loc[cut],st.loc[cut])

    def test_recent_price_jump_does_not_change_raw_core(self):
        px,raw,members,periods,sources,pools=load_prices();ss=['SPY','QQQ','AAPL','NVDA'];px=px[ss];raw={k:v[ss] for k,v in raw.items()}
        a,cores,_=panels(px,raw);d='2024-08-30';j=px.index.get_loc(d);changed=px.copy();changed.iloc[j-40:j+1,2:]*=2
        b,_,_=panels(changed,raw)
        for k in ['mom252_skip42','mom294_skip42','calendar12_skip2_raw','calendar14_skip2_raw']:
            pd.testing.assert_series_equal(a[k].loc[d],b[k].loc[d])

    def test_local_feasibility_certificates(self):
        report=read(P/'FAMILY_FEASIBILITY.json')
        for i,row in enumerate(report['months']):
            if row['best_spec'] is None:continue
            s=report['specifications'][row['best_spec']];d=pack('top32_theme17_'+s['normalization'],self.meta)
            X,spec=basis(d,s['core'],s['volatility_sign'],True);mask=np.zeros(len(X),bool);mask[i]=True
            D,pw,g,mo=pairs(X,d,mask);w=np.array(row['oracle_weights'])
            actual_margin=np.einsum('ij,j->i',D,w,optimize=False).min()
            self.assertAlmostEqual(actual_margin,row['best_margin'],places=7)
            A,b,bounds,prior=limits(spec);self.assertTrue((np.einsum('ij,j->i',A,w,optimize=False)<=b+1e-7).all())

    def test_market_state_weights_are_small_and_constrained(self):
        frozen=read(P/'FROZEN.json')['winner'];base=np.array(frozen['weights']);A,b,bounds,prior=limits(frozen['spec'])
        for row in read(P/'STATE_FROZEN.json')['trials']:
            for group,values in row['weights'].items():
                w=np.array(values);self.assertLessEqual(np.abs(w-base).sum(),2*row['budget']+1e-7)
                self.assertAlmostEqual(w.sum(),1.,places=7)
                self.assertTrue((np.einsum('ij,j->i',A,w,optimize=False)<=b+1e-7).all())
                for x,(lo,hi) in zip(w,bounds):self.assertTrue(lo-1e-7<=x<=hi+1e-7)

    def test_minimum_residual_impossibility_is_not_pool_failure(self):
        from diagnose import minimum_correction
        d=dict(order=np.array([[0,1,2]]),eligible=np.array([[True,True,True,True]]),labels=np.array([[True,True,True,False]]))
        self.assertLess(minimum_correction(np.array([.3,.2,.1,0]),d,0),1e-9)
        # Three strict ordering gaps are needed: a>b>c>outsider.
        self.assertAlmostEqual(minimum_correction(np.array([0.,.1,.2,.3]),d,0),.1500015,places=7)

if __name__=='__main__':unittest.main()
