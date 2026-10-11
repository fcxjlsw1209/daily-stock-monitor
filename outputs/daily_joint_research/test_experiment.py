import unittest,copy
import numpy as np
import pandas as pd
import experiment as e


class JointTests(unittest.TestCase):
 def test_ridge_and_logistic_learn_known_relation(self):
  rng=np.random.RandomState(30);x=rng.normal(size=(1000,3));prob=1/(1+np.exp(-(x[:,0]-x[:,1])))
  y=np.column_stack([.01*x[:,0],.02+.005*x[:,0],.03-.005*x[:,1],rng.binomial(1,prob)])
  model=e.fit_models(x,y);pred=e.predict(model,x)
  self.assertTrue(model['converged']);self.assertTrue(((pred[:,3]>0)&(pred[:,3]<1)).all())
  self.assertLess(np.mean((pred[:,0]-y[:,0])**2),1e-7)
  self.assertLess(np.mean((pred[:,3]-y[:,3])**2),np.mean((y[:,3]-y[:,3].mean())**2))
 def test_path_labels_require_entire_future_window(self):
  idx=pd.bdate_range('2020-01-01',periods=12);a=pd.DataFrame(100.,index=idx,columns=['A'])
  d=dict(adj=a,close=a,opening=a,high=a+10,low=a-5);f=dict(atr=a*.02)
  out=e.path_labels(d,f,5)
  self.assertAlmostEqual(out['up'].iloc[0,0],.1);self.assertAlmostEqual(out['down'].iloc[0,0],.05)
  self.assertEqual(out['bad'].iloc[0,0],1);self.assertTrue(out['mu'].iloc[-6:].isna().all().all())
  d['low']=d['low'].copy();d['low'].iloc[3]=np.nan
  self.assertTrue(e.path_labels(d,f,5)['mu'].iloc[0].isna().all())
 def test_context_features_do_not_read_future(self):
  idx=pd.bdate_range('2017-01-01',periods=500);rng=np.random.RandomState(5)
  a=pd.DataFrame(100*np.cumprod(1+rng.normal(.0004,.01,(500,3)),axis=0),index=idx,columns=['A','B','SPY'])
  d=dict(adj=a,close=a,opening=a*.999,volume=a*1000,sector60=a.pct_change(60,fill_method=None),issuers={'A':'A','B':'B','SPY':'FUND_SPY'})
  x=a.rank(axis=1,pct=True);f=dict(X=[x]*8,groups=[x]*4)
  before=e.context_features(d,f);d2=copy.deepcopy(d)
  for key in ['adj','close','opening','volume','sector60']:d2[key].iloc[450:]*=3
  after=e.context_features(d2,f)
  for x,y in zip(before,after):pd.testing.assert_frame_equal(x.iloc[:450],y.iloc[:450])
 def test_model_excludes_unmatured_labels(self):
  idx=pd.bdate_range('2016-01-01','2019-01-31');rng=np.random.RandomState(3)
  x=pd.DataFrame(rng.normal(size=(len(idx),4)),index=idx,columns=['A','B','C','SPY'])
  f=dict(valid=x.notna(),train=x.notna());labels={'mu':x*.01,'up':x.abs()*.02,'down':x.abs()*.01,'bad':(x>0).astype(float)}
  ids={s:s if s!='SPY' else 'FUND_SPY' for s in x};p,audit=e.walk_fit(f,[x],labels,ids,5,split=True)
  cutoff=idx.get_loc(pd.Timestamp('2019-01-01'))-6
  altered={k:v.copy() for k,v in labels.items()}
  for k in altered:altered[k].iloc[cutoff:]=0 if k=='bad' else 100
  q,_=e.walk_fit(f,[x],altered,ids,5,split=True)
  for k in p:pd.testing.assert_frame_equal(p[k],q[k])
  self.assertTrue(all(v['latest_label_session']<v['first_prediction'] for v in audit))


if __name__=='__main__':unittest.main()
