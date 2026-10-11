import unittest
import numpy as np,pandas as pd
from factors import matured_average,ridge_predictions
class CausalityTests(unittest.TestCase):
 def test_unfinished_labels_not_visible(self):
  idx=pd.bdate_range('2020-01-01',periods=50);y=pd.DataFrame(1.,index=idx,columns=['A','B']);e=y>0
  first=matured_average(y,e,window=30);y.iloc[30:]=1000;second=matured_average(y,e,window=30)
  pd.testing.assert_frame_equal(first.iloc[:36],second.iloc[:36]);self.assertGreater(second.iloc[36,0],first.iloc[36,0])
 def test_training_cutoff_strictly_before_prediction(self):
  idx=pd.bdate_range('2015-01-01',periods=600);rng=np.random.default_rng(3);a=pd.DataFrame(rng.normal(size=(600,4)),index=idx)
  f=dict(score=a,X=[a,a*.3+1],y=a*.01,event=a.notna())
  _,audit=ridge_predictions(f,first='2016-08-01')
  self.assertTrue(audit)
  for r in audit:self.assertLess(r['latest_label_open'],r['first_prediction'])
if __name__=='__main__':unittest.main()

class FullFeatureCausalityTests(unittest.TestCase):
 def test_future_prices_do_not_change_past_groups(self):
  from factors import build
  rng=np.random.default_rng(91);idx=pd.bdate_range('2014-01-01',periods=1000);a=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0007,.02,(1000,4)),axis=0)),index=idx,columns=list('ABCD'))
  d={'adj':a,'close':a,'opening':a*.999,'high':a*1.015,'low':a*.985,'volume':a*0+1e7,'eligible':a.notna(),'sectors':a*0,'sector60':a*0+.03,'market_close':a.mean(axis=1)}
  f=build(d);changed={k:v.copy() for k,v in d.items()};cut=800
  for k in ['adj','close','opening','high','low']:changed[k].iloc[cut:]*=3
  g=build(changed)
  for first,second in zip(f['groups'],g['groups']):pd.testing.assert_frame_equal(first.iloc[:cut],second.iloc[:cut])
 def test_ridge_future_labels_do_not_change_earlier_predictions(self):
  idx=pd.bdate_range('2015-01-01',periods=1000);rng=np.random.default_rng(7);a=pd.DataFrame(rng.normal(size=(1000,4)),index=idx)
  f=dict(score=a,X=[a,a*.3+1],y=a*.01,event=a.notna());first,_=ridge_predictions(f,first='2016-08-01')
  f['y']=f['y'].copy();f['y'].iloc[800:]=99;second,_=ridge_predictions(f,first='2016-08-01')
  pd.testing.assert_frame_equal(first.iloc[:800],second.iloc[:800])
