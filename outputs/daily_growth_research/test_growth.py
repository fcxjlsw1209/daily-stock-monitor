import copy,unittest
import numpy as np
import pandas as pd
import growth as g
from test_timing import PlannerTests

class GrowthTests(unittest.TestCase):
 def args(self):
  a=PlannerTests().fixture();dates,c,sc,_,atr,mask,mom,gaps,state=a
  ctx=dict(ma20=c*0+99,spy_bear5=pd.Series(False,index=dates),dual_bear5=pd.Series(False,index=dates),ret=c*0+.01)
  return a,ctx
 def call(self,args,ctx,cfg={},day_i=10):
  dates,c,sc,_,atr,mask,mom,gaps,state=args
  return g.planner({**g.BASE,**cfg},gaps,mom,ctx)('baseline',state,dates[day_i],dates,c,c,atr,mask,sc,{s:'Tech' for s in c},list(c),{s:s for s in c})
 def test_context_never_uses_future_prices(self):
  rng=np.random.RandomState(101);idx=pd.bdate_range('2015-01-01',periods=500)
  a=pd.DataFrame(100*np.exp(rng.normal(0,.02,(500,4)).cumsum(0)),index=idx,columns=['SPY','QQQ','A','B'])
  d=dict(adj=a,close=a.copy());f=dict(valid=a.notna());first=g.context(d,f);later=copy.deepcopy(d)
  for v in later.values():v.iloc[400:]*=5
  second=g.context(later,f)
  for k in first:
   if isinstance(first[k],pd.DataFrame):pd.testing.assert_frame_equal(first[k].iloc[:400],second[k].iloc[:400])
   else:pd.testing.assert_series_equal(first[k].iloc[:400],second[k].iloc[:400])
 def test_extension_requires_profit_trend_and_momentum(self):
  p=dict(entry_price=100,atr=5)
  cfg={**g.BASE,'strong_cap':45}
  self.assertEqual(g.cap_for(cfg,p,106,103,.7),45)
  for price,ma,mom in [(104,103,.7),(106,108,.7),(106,103,.5)]:self.assertEqual(g.cap_for(cfg,p,price,ma,mom),30)
 def test_extended_holder_still_has_protective_stop(self):
  p=dict(entry_price=100,atr=5);cfg={**g.BASE,'strong_cap':60}
  self.assertEqual(g.exit_reason(cfg,p,20,87,120,85,.9,.8),'close_stop_atr')
 def test_bear_exit_uses_previous_close(self):
  args,ctx=self.args();dates=args[0];args[-1]['positions']['A']=dict(entry_date=str(dates[6].date()),entry_price=100,atr=5)
  ctx['spy_bear5'].iloc[10]=True
  self.assertNotIn('market_bear_cash',[v['reason'] for v in self.call(args,ctx,{'liquidate_gate':'spy_bear5'})['sells']])
  ctx['spy_bear5'].iloc[9]=True;out=self.call(args,ctx,{'liquidate_gate':'spy_bear5'})
  self.assertEqual(out['buys'],[]);self.assertEqual(out['sells'],[dict(symbol='A',reason='market_bear_cash')])
 def test_protect_trend_does_not_disable_holding_cap(self):
  args,ctx=self.args();dates=args[0];args[-1]['positions']['A']=dict(entry_date=str(dates[0].date()),entry_price=100,atr=5)
  args[1].iloc[9]=110
  self.assertIn(dict(symbol='A',reason='holding_limit'),self.call(args,ctx,{'protect_trend':True,'cap':10})['sells'])
 def test_late_trail_not_active_before_gain(self):
  cfg={**g.BASE,'trail_atr':3.,'trail_activate':3.};p=dict(entry_price=100,atr=5)
  self.assertIsNone(g.exit_reason(cfg,p,4,98,114,100,.7,.8))
  self.assertEqual(g.exit_reason(cfg,p,4,99,115,100,.7,.8),'close_trailing_atr')
 def test_sector_cap_and_single_issuer(self):
  args,ctx=self.args();out=self.call(args,ctx,{'sector_cap':2})
  self.assertEqual(len(out['buys']),2);self.assertTrue(all(v['budget']==50000 for v in out['buys']))
 def test_vol_budget_cannot_increase_cash_allocation(self):
  args,ctx=self.args();dates=args[0];idx=pd.bdate_range(end=dates[-1],periods=100)
  ctx['ret']=pd.DataFrame(np.repeat((np.arange(100)%2*.1-.05)[:,None],4,axis=1),index=idx,columns=list('ABCD'))
  out=self.call(args,ctx,{'budget_vol':.3})
  self.assertTrue(all(0<v['budget']<100000/3 for v in out['buys']))
  ctx['ret'].loc[dates[10]:]=500
  self.assertEqual(out,self.call(args,ctx,{'budget_vol':.3}))
 def test_breakeven_needs_observed_gain(self):
  cfg={**g.BASE,'breakeven_atr':3.};p=dict(entry_price=100,atr=5)
  self.assertIsNone(g.exit_reason(cfg,p,5,99,114,100,.7,.8))
  self.assertEqual(g.exit_reason(cfg,p,5,99,115,100,.7,.8),'close_breakeven_after_gain')

if __name__=='__main__':unittest.main()
