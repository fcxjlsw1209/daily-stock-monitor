import unittest,copy
import numpy as np
import pandas as pd
import rebuild as b
from clock_execution import run_account

class Tests(unittest.TestCase):
 def test_cap_forward_only_and_split_basis(self):
  idx=pd.bdate_range('2020-01-01',periods=5);cap=pd.Series([np.nan,100,100,100,100],idx);obs=pd.Series([pd.NaT]+[idx[0]]*4,idx)
  c=pd.Series([50,51,52,53,54],idx)
  out=b.scale_cap(cap,obs,c)
  self.assertEqual(out.iloc[2],104)
  pd.testing.assert_series_equal(out,b.scale_cap(cap,obs,c/2))
  later=c.copy();later.iloc[3:]=900;pd.testing.assert_series_equal(out.iloc[:3],b.scale_cap(cap,obs,later).iloc[:3])
 def test_history_delay(self):
  idx=pd.bdate_range('2019-01-01',periods=20);v=pd.DataFrame(.1,index=idx,columns=['A','B']);event=v>0;bucket=v*0
  first=b.conditional_history(v,event,bucket);later=v.copy();later.iloc[7:]=100
  second=b.conditional_history(later,event,bucket)
  self.assertTrue(first.iloc[:6].isna().all().all());self.assertAlmostEqual(first.iloc[6,0],.1)
  pd.testing.assert_frame_equal(first.iloc[:13],second.iloc[:13])
 def test_models_never_use_unmatured_or_future_labels(self):
  rng=np.random.RandomState(11);idx=pd.bdate_range('2010-01-01',periods=400);xs=[rng.rand(400,6,8) for i in range(4)];y=rng.normal(0,.01,(400,6));train=np.ones((400,6),bool)
  p,t,a=b.ridge_fit_predict(xs,y,train,idx,first='2011-01-01')
  cut=idx.searchsorted('2011-05-01');y2=y.copy();y2[cut-5:]=100;xs2=[x.copy() for x in xs]
  for x in xs2:x[cut:]=0
  pp,tt,aa=b.ridge_fit_predict(xs2,y2,train,idx,first='2011-01-01')
  np.testing.assert_allclose(p[:cut],pp[:cut],equal_nan=True);np.testing.assert_allclose(t[:cut],tt[:cut],equal_nan=True)
  self.assertTrue(all(v['latest_label_session']<v['first_prediction'] for v in a))
 def fixture(self):
  dates=pd.bdate_range('2020-01-01',periods=20);c=pd.DataFrame(100.,index=dates,columns=['A','B','C','D']);v=c*0+.8
  f=dict(rule=v, momentum=v,event=c>0,annual=c*0+.2,atr=c*0+5,ma5=c)
  models={'entry_experts':v*.025,'hold_experts':v*.015};state=dict(equity=100000,positions={})
  return dates,c,f,models,state
 def call(self,cfg={},clock='next_open',day_i=8):
  dates,c,f,models,state=self.fixture();pl,ff=b.planner({**b.BASE,**cfg},f,models,clock)
  return pl('baseline',state,dates[day_i],dates,c,c,f['atr'],ff['mask'],ff['score'],{s:'Tech' for s in c},list(c),{s:s for s in c})
 def test_max_three_and_equal_budgets(self):
  v=self.call();self.assertEqual(len(v['buys']),3);self.assertAlmostEqual(sum(t['budget'] for t in v['buys']),100000)
 def test_no_event_no_entry(self):
  dates,c,f,m,state=self.fixture();f['event'][:]=False;pl,ff=b.planner(b.BASE,f,m,'next_open')
  out=pl('baseline',state,dates[8],dates,c,c,f['atr'],ff['mask'],ff['score'],{s:'Tech' for s in c},list(c),{s:s for s in c})
  self.assertEqual(out['buys'],[])
 def test_holder_does_not_need_new_pullback(self):
  dates,c,f,m,state=self.fixture();f['event'][:]=False;state['positions']={s:dict(entry_date=str(dates[6].date()),entry_price=100,atr=5) for s in ['A','B']}
  pl,ff=b.planner(b.BASE,f,m,'next_open');out=pl('baseline',state,dates[8],dates,c,c,f['atr'],ff['mask'],ff['score'],{s:'Tech' for s in c},list(c),{s:s for s in c})
  self.assertEqual(out['sells'],[])
 def test_replacement_pairs_are_distinct(self):
  dates,c,f,m,state=self.fixture();state['positions']={s:dict(entry_date=str(dates[4].date()),entry_price=100,atr=5) for s in ['A','B','C']}
  pl,ff=b.planner(b.BASE,f,m,'next_open');out=pl('baseline',state,dates[8],dates,c,c,f['atr'],ff['mask'],ff['score'],{s:'Tech' for s in c},list(c),{s:s for s in c})
  self.assertEqual(len(out['replacement_pairs']),1);self.assertEqual(out['replacement_pairs'][0]['buy'],'D')
 def test_signal_dates_are_explicit(self):
  self.assertLess(self.call()['signal_date'],self.call()['session'])
  v=self.call(clock='same_close_diagnostic');self.assertEqual(v['signal_date'],v['session'])

if __name__=='__main__':unittest.main()
