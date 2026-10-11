import unittest
import pandas as pd
from engine import initial_state,plan,execute
class ForwardTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2025-01-01',periods=30);self.c=pd.DataFrame({'A':100.},index=self.dates);self.atr=self.c*.02;self.mask=self.c>0;self.score=self.c/100;self.cfg={'exit':'fixed20','allocation':.2,'sector_cap':2,'max_positions':5}
 def test_budget_prior_equity_and_actual_next_open(self):
  s=initial_state();d=plan(s,self.dates[1],self.dates,self.c,self.c,self.atr,self.mask,self.score,{'A':'x'},self.cfg)
  self.assertEqual(d['buys'][0]['budget'],20000)
  n,day,t,o,a=execute(s,d,{'A':{'open':200,'close':200}},self.dates,10,{'A':'x'})
  self.assertAlmostEqual(n['positions']['A']['qty'],20000/1.0005/200)
  self.assertEqual(s['positions'],{})
 def test_twenty_sessions_no_premature_exit_or_reentry(self):
  s=initial_state();s['positions']={'A':{'entry_date':str(self.dates[1].date()),'entry_price':100,'atr':2}}
  early=plan(s,self.dates[20],self.dates,self.c,self.c,self.atr,self.mask,self.score,{'A':'x'},self.cfg)
  self.assertEqual(early['sells'],[])
  due=plan(s,self.dates[21],self.dates,self.c,self.c,self.atr,self.mask,self.score,{'A':'x'},self.cfg)
  self.assertEqual(due['sells'],[{'symbol':'A','reason':'maximum_sessions'}]);self.assertEqual(due['buys'],[])
 def test_future_close_does_not_change_plan(self):
  a=plan(initial_state(),self.dates[1],self.dates,self.c,self.c,self.atr,self.mask,self.score,{'A':'x'},self.cfg)
  changed=self.c.copy();changed.loc[self.dates[1]:]=1000
  b=plan(initial_state(),self.dates[1],self.dates,changed,self.c,self.atr,self.mask,self.score,{'A':'x'},self.cfg)
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
