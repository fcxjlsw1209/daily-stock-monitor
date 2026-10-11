import unittest,copy
import numpy as np
import pandas as pd
import frontier as z
import shape
import test_frontier
import test_timing

class FollowupTests(unittest.TestCase):
 def test_two_name_cap_and_half_budgets(self):
  args=test_timing.PlannerTests().fixture();decision=test_frontier.FrontierTests().call({'positions':2},args)
  self.assertEqual(len(decision['buys']),2)
  self.assertTrue(all(v['budget']==50000 for v in decision['buys']))
 def test_two_name_policy_does_not_allow_single_entry(self):
  args=test_timing.PlannerTests().fixture();args[5].loc[:]=False;args[5].loc[:,'A']=True
  decision=test_frontier.FrontierTests().call({'positions':2},args)
  self.assertEqual(decision['buys'],[])
 def test_continuity_retains_only_surviving_incumbent(self):
  args=test_timing.PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  mask.loc[:]=False;state['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100.,atr=5.)
  call=test_frontier.FrontierTests().call
  self.assertIn(dict(symbol='A',reason='fewer_than_two_holdings'),call({},args)['sells'])
  self.assertEqual(call({'keep_lone':True},args)['sells'],[])
  c.loc[dates[9],'A']=89;mask.loc[:,'B']=True
  out=call({'keep_lone':True},args)
  self.assertIn(dict(symbol='A',reason='close_stop_atr'),out['sells']);self.assertEqual(out['buys'],[])
 def test_single_entry_keeps_unallocated_cash_and_no_topup(self):
  args=test_timing.PlannerTests().fixture();args[5].loc[:]=False;args[5].loc[:,'A']=True
  call=test_frontier.FrontierTests().call
  for floor in [2,3]:
   out=call({'min_positions':1,'budget_floor':floor},args)
   self.assertEqual(len(out['buys']),1);self.assertEqual(out['buys'][0]['budget'],100000/floor)
  dates=args[0];args[-1]['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100.,atr=5.)
  self.assertEqual(call({'min_positions':1,'budget_floor':3},args)['buys'],[])
 def test_single_entry_does_not_bypass_holding_limit(self):
  args=test_timing.PlannerTests().fixture();dates=args[0];args[5].loc[:]=False
  args[-1]['positions']['A']=dict(entry_date=str(dates[0].date()),entry_price=100.,atr=5.)
  out=test_frontier.FrontierTests().call({'min_positions':1,'budget_floor':2},args)
  self.assertIn(dict(symbol='A',reason='holding_limit'),out['sells'])
 def test_trailing_uses_completed_close_and_still_honors_hard_stop(self):
  args=test_timing.PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100.,atr=5.)
  c.loc[dates[7],'A']=115;c.loc[dates[9],'A']=104
  cfg=dict(tp=None,cap=20,trail_atr=2.);call=test_frontier.FrontierTests().call
  self.assertIn(dict(symbol='A',reason='close_trailing_atr'),call(cfg,args)['sells'])
  c.loc[dates[7],'A']=100;c.iloc[10:]=1e6
  self.assertNotIn('A',[v['symbol'] for v in call(cfg,args)['sells']])
  c.loc[dates[9],'A']=89
  self.assertIn(dict(symbol='A',reason='close_stop_atr'),call(cfg,args)['sells'])
 def test_ma_exit_and_rank_still_work_without_profit_target(self):
  args=test_timing.PlannerTests().fixture();dates,c,sc,_,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100.,atr=5.)
  c.loc[dates[9],'A']=99
  call=test_frontier.FrontierTests().call
  self.assertIn(dict(symbol='A',reason='close_below_ma'),call(dict(tp=None,ma_exit=5),args)['sells'])
  c.loc[dates[9],'A']=120;sc.loc[dates[9],'A']=.2
  self.assertIn(dict(symbol='A',reason='rank_deterioration'),call(dict(tp=None,trail_atr=2),args)['sells'])
 def test_exact252_summary_uses_initial_cash_without_extra_day(self):
  dates=pd.bdate_range('2025-01-01',periods=252)
  rows=[dict(date=str(day.date()),equity=float(value),exposure=.5) for day,value in zip(dates,np.linspace(101000,205000,252))]
  out=dict(daily=rows,metrics=dict(equity=205000),state=dict(fees=0),orders=[])
  v=z.summarize(out)
  self.assertAlmostEqual(v['last252']['return_'],1.05);self.assertIsNone(v['ex_recent'])
  self.assertIsNone(v['development']);self.assertEqual(v['full']['max_drawdown'],0.)
 def test_shape_and_mature_path_overlays_are_causal(self):
  rng=np.random.RandomState(155);idx=pd.bdate_range('2016-01-01',periods=600);cols=list('ABCDEFGH')
  c=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001,.02,(600,8)),axis=0)),index=idx,columns=cols)
  d=dict(adj=c.copy(),close=c.copy(),opening=c*.997,high=c*1.01,low=c*.985,volume=c*0+5e6,
   sector60=c*0,market_close=c.iloc[:,0],eligible=c.notna(),sectors=pd.DataFrame('Tech',index=idx,columns=cols))
  f,_=z.r.features(d);base=c*0+.8;first=shape.shape_scores(d,f,base)
  altered=copy.deepcopy(d)
  for k in ['adj','close','opening','high','low']:altered[k].iloc[450:]*=3
  ff,_=z.r.features(altered);second=shape.shape_scores(altered,ff,base)
  for key in first:pd.testing.assert_frame_equal(first[key].iloc[:450],second[key].iloc[:450])

if __name__=='__main__':unittest.main()
