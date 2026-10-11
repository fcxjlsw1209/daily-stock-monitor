import unittest
from types import SimpleNamespace
import pandas as pd
from research_round2 import hold_plan,gap_buys
class RulesTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2026-01-05',periods=12)
  self.close=pd.DataFrame({'X':[100.]*12},index=self.dates)
  self.data=SimpleNamespace(close=self.close,sector={'X':'Technology'})
  self.f={'ma5':self.close.copy(),'atr':self.close.copy()*0+2}
  self.mask=self.close.copy()>0;self.score=self.close.copy()*0+.7
  self.state={'equity':100000.,'positions':{'X':{'entry_date':str(self.dates[0].date()),'entry_price':100.,'atr':2.,'reclaimed':False}}}
 def q(self,i=2):return hold_plan(self.state,self.dates[i],self.dates,self.data,self.f,self.mask,self.score)
 def test_no_exit_before_reclaim(self):
  self.close.iloc[1,0]=99.;self.assertEqual(self.q()['sells'],[])
 def test_reclaimed_then_lost_exits_next_session_no_reentry(self):
  self.state['positions']['X']['reclaimed']=True;self.close.iloc[1,0]=99.
  q=self.q();self.assertEqual(q['sells'],[{'symbol':'X','reason':'lost_sma5_after_reclaim'}]);self.assertEqual(q['buys'],[])
 def test_equal_ma_is_not_loss(self):
  self.state['positions']['X']['reclaimed']=True;self.assertEqual(self.q()['sells'],[])
 def test_stop_without_reclaim(self):
  self.close.iloc[1,0]=95.;self.assertEqual(self.q()['sells'][0]['reason'],'close_below_2atr')
 def test_max_holding(self):self.assertEqual(self.q(10)['sells'][0]['reason'],'maximum_sessions')
 def test_future_bars_cannot_change_current_plan(self):
  q=self.q();self.close.iloc[2:]=10000;self.assertEqual(q,self.q())
 def test_gap_skip_and_adverse_price(self):
  b=[{'symbol':'X','reference_close':100,'atr':2}]
  self.assertEqual(gap_buys(b,{'X':{'open':101}})[0],[])
  bars={'X':{'open':99}};yes,no=gap_buys(b,bars,True)
  self.assertEqual(len(yes),1);self.assertEqual(bars['X']['open'],101);self.assertEqual(no,[])
if __name__=='__main__':unittest.main()
