import unittest,copy
import pandas as pd
import numpy as np
import rebuild as b
from clock_execution import run_account

class ClockTests(unittest.TestCase):
 def fixture(self):
  dates=pd.bdate_range('2020-01-01',periods=20);c=pd.DataFrame(100.,index=dates,columns=['A','B','C']);v=c*0+.8
  d={k:c.copy() for k in ['adj','close','opening','high','low','volume','sector60','raw_per_signal','dividends','splits']}
  d.update(eligible=c>0,issuers={s:s for s in c},sectors=pd.DataFrame('Tech',index=dates,columns=c.columns),market_close=pd.Series(100.,dates),market_dividends=pd.Series(0.,dates),signal_basis='split_adjusted',execution_basis='raw_with_explicit_actions')
  d['dividends'][:]=0;d['splits'][:]=1;d['raw_per_signal'][:]=1;d['high'][:]=101;d['low'][:]=99
  f=dict(rule=v,momentum=v,event=c>0,annual=v,atr=c*0+5,ma5=c)
  return dates,d,f
 def test_nextopen_adapter_reproduces_audited_executor(self):
  dates,d,f=self.fixture();pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},'next_open');start=str(dates[3].date());end=str(dates[-1].date())
  oldpl,oldex=b.r.h.plan,b.r.h.execute
  try:
   b.r.h.plan=pl;b.r.h.execute=lambda name,state,decision,bars,dates,sector=None:b.r.ORIGINAL_EXEC(name,state,decision,bars,dates,sector=None)
   ref=b.r.h.run_account('baseline',d,start,end,{'baseline':ff})
  finally:b.r.h.plan,b.r.h.execute=oldpl,oldex
  result=run_account(d,ff,pl,start,end,10,'next_open');self.assertEqual(result['status'],'complete');self.assertEqual(ref['orders'],result['orders']);self.assertEqual(ref['daily'],result['daily'])
 def test_sameclose_entry_at_split_has_correct_atr_and_price(self):
  dates,d,f=self.fixture();d['splits'].loc[dates[8]]=2;d['raw_per_signal'].loc[:dates[7]]=2
  pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},'same_close_diagnostic')
  out=run_account(d,ff,pl,str(dates[8].date()),str(dates[9].date()),0,'same_close_diagnostic')
  self.assertEqual(out['status'],'complete');self.assertEqual(out['state']['equity'],100000)
  for p in out['state']['positions'].values():self.assertEqual(p['atr'],5);self.assertEqual(p['entry_price'],100)
 def test_sameclose_held_split_no_phantom_profit_or_stop(self):
  dates,d,f=self.fixture();d['splits'].loc[dates[8]]=2;d['raw_per_signal'].loc[:dates[7]]=2
  pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},'same_close_diagnostic');out=run_account(d,ff,pl,str(dates[5].date()),str(dates[9].date()),0,'same_close_diagnostic')
  self.assertEqual(out['status'],'complete');self.assertEqual(len(out['orders']),3);self.assertAlmostEqual(out['state']['equity'],100000)
 def test_diagnostic_fill_uses_current_close(self):
  dates,d,f=self.fixture();d['opening'][:]=90
  pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},'same_close_diagnostic');out=run_account(d,ff,pl,str(dates[5].date()),str(dates[9].date()),0,'same_close_diagnostic')
  self.assertTrue(all(x['price']==100 for x in out['orders']))
 def test_missing_held_bar_blocks_and_does_not_erase_trade(self):
  dates,d,f=self.fixture();d['close'].at[dates[7],'A']=np.nan
  pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},'next_open');out=run_account(d,ff,pl,str(dates[5].date()),str(dates[9].date()),0,'next_open')
  self.assertEqual(out['status'],'blocked');self.assertEqual(out['blocked_session'],str(dates[7].date()));self.assertEqual(len(out['orders']),3)
if __name__=='__main__':unittest.main()

class SpinTests(unittest.TestCase):
 fixture=ClockTests.fixture
 def test_spinoff_keeps_parent_quantity_and_credits_child(self):
  dates,d,f=self.fixture();d['splits'].at[dates[8],'A']=1.054;d['raw_per_signal'].loc[:dates[7],'A']=1.054
  d['spin_distributions']=[dict(event_id='test_spin',security_id='A',child_security_id='CHILD',execution_session=str(dates[8].date()),known_at='2019-12-31T00:00:00-05:00',ratio=.124079,provider_adjustment=1.054,terms_verified=True,source_sha256='fixture',child_open=dict(session=str(dates[8].date()),kind='regular_way_open',price=15.39,source_sha256='fixture'))]
  for clock in ['next_open','same_close_diagnostic']:
   pl,ff=b.planner({**b.BASE,'mode':'rules'},f,{},clock);out=run_account(d,ff,pl,str(dates[5].date()),str(dates[9].date()),0,clock)
   self.assertEqual(out['status'],'complete');p=out['state']['positions']['A'];q=100000/3/105.4
   self.assertAlmostEqual(p['qty'],q);self.assertAlmostEqual(out['state']['spin_proceeds'],q*.124079*15.39)
   self.assertEqual(len([a for a in out['actions'] if a.get('kind')=='spin_distribution_liquidation']),1)
