import unittest
import numpy as np
import pandas as pd
import open_inverse as m

class InverseTests(unittest.TestCase):
 def data(self):
  rng=np.random.default_rng(21);X=rng.normal(size=(8,7,3));eligible=np.ones((8,7),bool)
  order=np.tile([0,2,4],(8,1));labels=np.zeros_like(eligible);labels[:,[0,2,4]]=True
  return X,dict(order=order,labels=labels,eligible=eligible,symbols=[list('ABCDEFG') for _ in range(8)])
 def test_future_holdings_do_not_change_fit(self):
  X,d=self.data();mask=np.arange(8)<4;groups=['core','risk','industry']
  w=m.fit(X,d,mask,groups,.01,.25,1)
  changed={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in d.items()}
  changed['order'][4:]=[6,5,1];changed['labels'][4:]=False;changed['labels'][4:,[6,5,1]]=True
  np.testing.assert_allclose(w,m.fit(X,changed,mask,groups,.01,.25,1),atol=1e-12)
 def test_set_loss_invariant_to_author_order(self):
  X,d=self.data();mask=np.ones(8,bool);groups=['core','risk','industry']
  w=m.fit(X,d,mask,groups,.01,0,1);other=dict(d,order=d['order'][:,::-1])
  np.testing.assert_allclose(w,m.fit(X,other,mask,groups,.01,0,1),atol=1e-9)
 def test_core_nonnegative_and_no_arbitrary_sum_constraint(self):
  X,d=self.data();w=m.fit(X,d,np.ones(8,bool),['core','risk','industry'],.01,.25,1)
  self.assertGreaterEqual(w[0],0);self.assertGreater(abs(w.sum()-1),.01)
 def test_choices_have_no_author_input(self):
  X,d=self.data();local=X[:,:,0];chosen=m.choices(local,d,5)
  d['labels'][:]=False;d['order'][:]=-1
  self.assertEqual(chosen,m.choices(local,d,5))
 def test_ineligible_not_force_included_and_two_or_cash(self):
  X,d=self.data();d['eligible'][0]=False;d['eligible'][0,[0,1]]=True;d['eligible'][1]=False;d['eligible'][1,2]=True
  ch=m.choices(X[:,:,0],d,5);self.assertEqual(len(ch[0]),2);self.assertEqual(ch[1],[])
 def test_core_state_basis_monotonic(self):
  meta,packs,state=m.load();d=next(iter(packs.values()));X,names,groups=m.design(d,state,'regime')
  self.assertTrue(((state>=0)&(state<=1)).all());self.assertEqual(len(names),88)
  self.assertEqual(groups[names.index('core_252_raw_v0__state0')],'core')
 def test_no_future_market_values(self):
  rng=np.random.default_rng(3);idx=pd.bdate_range('2018-01-01',periods=500)
  px=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0003,.012,(500,3)),axis=0))*100,index=idx,columns=['SPY','ABC','XLK'])
  raw=dict(close=px,high=px*1.01,low=px*.99,volume=px*0+1000000,sector60=px.pct_change(60,fill_method=None))
  f,rel,_=m.feature_bank(px,raw);p,pr,_=m.feature_bank(px.iloc[:420],{k:v.iloc[:420] for k,v in raw.items()})
  for k in f:pd.testing.assert_frame_equal(f[k].iloc[:420],p[k])
  for k in rel:pd.testing.assert_frame_equal(rel[k].iloc[:420],pr[k])
 def test_bare_core_excludes_recent42_sessions(self):
  rng=np.random.default_rng(8);idx=pd.bdate_range('2018-01-01',periods=500)
  px=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0003,.01,(500,2)),axis=0))*100,index=idx,columns=['SPY','ABC'])
  def raw(z):return dict(close=z,high=z*1.01,low=z*.99,volume=z*0+1e6,sector60=z.pct_change(60,fill_method=None))
  f,_,_=m.feature_bank(px,raw(px));z=px.copy();z.iloc[-42:,1]*=np.linspace(1.2,3,42)
  g,_,_=m.feature_bank(z,raw(z))
  for k in ['core_252_raw_v0','core_294_log_v0','core_mean_sharpe_252','core_trend_quality_294']:
   np.testing.assert_allclose(f[k].iloc[-1],g[k].iloc[-1],atol=1e-12)
 def test_metrics_distinguish_set_and_order(self):
  result=m.matching([['B','A','C']],[dict(holdings=['A','B','C'])],np.array([True]))
  self.assertEqual(result['set_exact'],1);self.assertEqual(result['order_exact'],0)

if __name__=='__main__':unittest.main()
