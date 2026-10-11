import copy,unittest
import numpy as np,pandas as pd
import horizons as h
class HorizonTests(unittest.TestCase):
 def test_skill_cannot_use_unmatured_outcomes(self):
  rng=np.random.RandomState(19);idx=pd.bdate_range('2015-01-01',periods=450);p=pd.DataFrame(rng.normal(size=(450,8)),index=idx);y=p*.2+pd.DataFrame(rng.normal(size=p.shape),index=idx);event=p.notna()
  for delay in [11,21]:
   before=h.skill(p,y,event,delay);changed=y.copy();changed.iloc[300:]*=-3;after=h.skill(p,changed,event,delay)
   pd.testing.assert_series_equal(before.iloc[:300+delay],after.iloc[:300+delay])
 def test_aligned_factors_ignore_future_prices(self):
  rng=np.random.RandomState(11);idx=pd.bdate_range('2015-01-01',periods=550);a=pd.DataFrame(100*np.exp(rng.normal(.001,.02,(550,8)).cumsum(0)),index=idx,columns=list('ABCDEFGH'))
  d=dict(adj=a,close=a.copy(),opening=a*.998,high=a*1.02,low=a*.98);rank=a.rank(axis=1,pct=True);f=dict(groups=[rank]*4,X=[rank]*8,valid=a.notna(),event=a<a.shift())
  changed=copy.deepcopy(d)
  for v in changed.values():v.iloc[400:]*=5
  for days in [10,20]:
   ff=h.features(d,f,days,True);future=h.features(changed,f,days,True)
   for group in [1,2]:pd.testing.assert_frame_equal(ff['groups'][group].iloc[:400],future['groups'][group].iloc[:400])
 def test_long_horizon_training_excludes_unmatured_labels(self):
  idx=pd.bdate_range('2016-01-01',periods=1000);rng=np.random.RandomState(42);x=pd.DataFrame(rng.rand(1000,8),index=idx)
  f=dict(valid=x.notna(),train=x.notna(),event=x.notna(),X=[x,x*x,x*.3,x*.8],groups=[x]*4,y=x*.1-.05)
  for delay in [11,21]:
   original,_,_=h.z.r.model(f,delay);changed=dict(f);changed['y']=f['y'].copy();t=np.flatnonzero(idx.to_period('M')==pd.Period('2019-01'))[0];changed['y'].iloc[t-delay:]=1000
   new,_,audit=h.z.r.model(changed,delay);month=idx.to_period('M')==pd.Period('2019-01')
   pd.testing.assert_frame_equal(original.loc[month],new.loc[month]);self.assertTrue(all(v['latest_label_session']<v['first_prediction'] for v in audit))
if __name__=='__main__':unittest.main()
