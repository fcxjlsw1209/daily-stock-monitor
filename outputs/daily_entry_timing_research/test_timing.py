import copy
import unittest
import numpy as np
import pandas as pd
import timing as t

class SignalTests(unittest.TestCase):
 def fixture(self):
  dates=pd.bdate_range('2017-01-02',periods=310);cols=['A','B']
  c=pd.DataFrame(np.broadcast_to(np.linspace(80,110,len(dates))[:,None],(len(dates),2)).copy(),index=dates,columns=cols)
  d=dict(close=c,adj=c.copy(),opening=c-1,high=c+.4,low=c-1.,volume=c*0+1000)
  f=dict(valid=c.notna(),event=c.lt(0),atr=c*0+2,groups=[c*0+.5]*4)
  s=c*0+.8
  return d,f,s
 def test_armed_requires_prior_setup_and_expires(self):
  d,f,s=self.fixture();f['event'].iloc[290]=True
  _,m,origin=t.armed_signals(d,f,s)
  self.assertFalse(m.iloc[290].any());self.assertTrue(m.iloc[291:294].all().all());self.assertFalse(m.iloc[294].any())
  self.assertTrue((origin[291:294]==290).all())
 def test_latest_setup_replaces_old_and_current_invalid_blocks(self):
  d,f,s=self.fixture();f['event'].iloc[290:292]=True;s.iloc[291]=.9;f['valid'].iloc[292,1]=False
  sc,m,origin=t.armed_signals(d,f,s)
  self.assertEqual(sc.iloc[292,0],.9);self.assertEqual(origin[292,0],291);self.assertFalse(m.iloc[292,1])
 def test_no_chase_gate(self):
  d,f,s=self.fixture();f['event'].iloc[290]=True
  d['close'].iloc[291]=d['close'].iloc[290]+3
  self.assertFalse(t.armed_signals(d,f,s)[1].iloc[291].any())
 def test_future_prices_and_scores_cannot_change_past_signals(self):
  d,f,s=self.fixture();f['event'].iloc[280:300:3]=True
  q=copy.deepcopy(d);sf=s.copy();sf.iloc[300:]=1
  for k in q:q[k].iloc[300:]*=3
  for kind in ['armed_up3','armed_break3','hybrid_recovery','intraday_gate','intraday_shape','shock_pause2']:
   p=t.entry_inputs(d,f,s,kind);v=t.entry_inputs(q,f,sf,kind)
   for one,two in zip(p,v):pd.testing.assert_frame_equal(one.iloc[:300],two.iloc[:300])
 def test_shock_uses_lagged_baseline_and_has_finite_pause(self):
  d,f,s=self.fixture();f['event'].iloc[280:]=True
  d['opening'].iloc[290]=d['close'].iloc[289]*.8;d['volume'].iloc[290]=2000
  _,_,mask=t.entry_inputs(d,f,s,'shock_pause2')
  self.assertFalse(mask.iloc[290:293].any().any());self.assertTrue(mask.iloc[293].all())

class PlannerTests(unittest.TestCase):
 def fixture(self):
  dates=pd.bdate_range('2020-01-01',periods=20);cols=list('ABCD');c=pd.DataFrame(100.,index=dates,columns=cols)
  score=c*0+.8;candidate=score.copy();atr=c*0+5;mask=c.notna();vol=score*0+.5;gaps=pd.Series(.1,index=dates)
  state=t.a.old.prior.ENGINE.initial_state()
  return dates,c,score,candidate,atr,mask,vol,gaps,state
 def call(self,kind,args,cooldowns=None,day_i=10):
  dates,c,score,candidate,atr,mask,vol,gaps,state=args
  p=t.planner(kind,gaps,candidate,vol,cooldowns or {})
  return p('baseline',state,dates[day_i],dates,c,c,atr,mask,score,{s:s for s in c},list(c),{s:s for s in c})
 def test_current_execution_day_close_not_used(self):
  args=self.fixture();p=self.call('stable_control',args);args[1].iloc[10:]=1e9
  self.assertEqual(p,self.call('stable_control',args))
 def test_stop_cooldown_blocks_three_next_sessions(self):
  args=self.fixture();p=self.call('stop_pause3',args,{'A':7},10)
  self.assertNotIn('A',[v['symbol'] for v in p['buys']])
  q=self.call('stop_pause3',args,{'A':7},11)
  self.assertIn('A',[v['symbol'] for v in q['buys']])
 def test_highvol_horizon_uses_entry_signal_not_current_volatility(self):
  args=self.fixture();dates,c,score,candidate,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[5].date()),entry_price=100.,atr=5.)
  vol.loc[dates[4],'A']=.9;vol.loc[dates[9],'A']=.1
  p=self.call('fast_highvol',args)
  self.assertIn(dict(symbol='A',reason='holding_limit'),p['sells'])
  vol.loc[dates[4],'A']=.1;vol.loc[dates[9],'A']=.9
  q=self.call('fast_highvol',args)
  self.assertNotIn('A',[v['symbol'] for v in q['sells']])
 def test_armed_candidate_does_not_override_held_current_exit_score(self):
  args=self.fixture();dates,c,score,candidate,atr,mask,vol,gaps,state=args
  state['positions']['A']=dict(entry_date=str(dates[8].date()),entry_price=100.,atr=5.)
  score.loc[dates[9],'A']=.2;candidate.loc[dates[9],'A']=.95
  p=self.call('armed_up3',args)
  self.assertIn(dict(symbol='A',reason='rank_deterioration'),p['sells'])

if __name__=='__main__':unittest.main()
