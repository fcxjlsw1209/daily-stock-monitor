import unittest,copy
import numpy as np
import pandas as pd
import frontier as z
from test_timing import SignalTests,PlannerTests

class FrontierTests(unittest.TestCase):
 def test_new_factors_ignore_future_prices(self):
  d,f,s=SignalTests().fixture();x=z.extras(d,f);altered=copy.deepcopy(d)
  for k in altered:altered[k].iloc[300:]*=3
  y=z.extras(altered,f)
  for key in x:pd.testing.assert_frame_equal(x[key].iloc[:300],y[key].iloc[:300])
 def test_raw_half_changes_only_one_eighth_total_score(self):
  d,f,s=SignalTests().fixture();x=z.extras(d,f);x['raw']=s*0+.9
  actual,_=z.signals(f,s,x,{**z.BASE,'score_mode':'momentum_rawhalf'})
  pd.testing.assert_frame_equal(actual,s+.125*(x['raw']-f['groups'][0]))
 def test_geometric_penalizes_unbalanced_groups(self):
  d,f,s=SignalTests().fixture();f['groups']=[s*0+.1,s*0+.9,s*0+.9,s*0+.9]
  base=sum(f['groups'])/4;actual,_=z.signals(f,base,z.extras(d,f),{**z.BASE,'score_mode':'geometric_groups'})
  self.assertTrue((actual<base).all().all())
 def call(self,cfg,args,day_i=10):
  dates,c,sc,candidate,atr,mask,vol,gaps,state=args
  return z.planner({**z.BASE,**cfg},gaps,vol)('baseline',state,dates[day_i],dates,c,c,atr,mask,sc,{s:s for s in c},list(c),{s:s for s in c})
 def test_profit_target_preserves_stop_and_age(self):
  args=PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100.,atr=5.)
  c.loc[dates[9],'A']=116
  self.assertIn(dict(symbol='A',reason='close_take_profit_atr'),self.call({},args)['sells'])
  self.assertNotIn('A',[v['symbol'] for v in self.call({'tp':4},args)['sells']])
  c.loc[dates[9],'A']=89
  self.assertIn(dict(symbol='A',reason='close_stop_atr'),self.call({'tp':4},args)['sells'])
 def test_replacement_minimum_age(self):
  args=PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[8].date()),entry_price=100.,atr=5.)
  sc.loc[dates[9],'A']=.6;sc.loc[dates[9],['B','C','D']]=.95
  self.assertIn('A',[v['symbol'] for v in self.call({},args)['sells']])
  self.assertNotIn('A',[v['symbol'] for v in self.call({'min_age':3},args)['sells']])
 def test_protect_winner_does_not_disable_hard_cap(self):
  args=PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[0].date()),entry_price=100.,atr=5.)
  c.loc[dates[9],'A']=106
  self.assertIn(dict(symbol='A',reason='holding_limit'),self.call({'protect_winner':True},args)['sells'])
 def test_extended_horizon_requires_all_prior_close_conditions(self):
  args=PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[0].date()),entry_price=100.,atr=5.)
  c.loc[dates[9],'A']=106;vol.loc[dates[9],'A']=.9
  self.assertNotIn('A',[v['symbol'] for v in self.call({'extend':True},args)['sells']])
  c.iloc[10:]=1e6
  self.assertNotIn('A',[v['symbol'] for v in self.call({'extend':True},args)['sells']])
  vol.loc[dates[9],'A']=.7
  self.assertIn(dict(symbol='A',reason='holding_limit'),self.call({'extend':True},args)['sells'])

if __name__=='__main__':unittest.main()
