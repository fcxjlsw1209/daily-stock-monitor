import unittest,copy
import pandas as pd
from run import adjust
class Variants(unittest.TestCase):
 def setUp(self):
  self.d=pd.bdate_range('2025-01-01',periods=3);self.p=self.d[0]
  self.c=pd.DataFrame({'A':[99.,100.,100.],'B':[101.,100.,100.]},index=self.d);self.ma=self.c*0+100;self.v=self.c*0+.5
  self.state={'positions':{'A':{}}};self.plan={'sells':[],'buys':[{'symbol':'B','budget':20000.}]}
 def test_vol_budget_and_no_mutation(self):
  result=adjust(self.plan,self.state,self.p,self.c,self.ma,self.v,'vol')
  self.assertEqual(result['buys'][0]['budget'],10000);self.assertEqual(self.plan['buys'][0]['budget'],20000)
 def test_no_leverage(self):
  self.v[:]=.1
  self.assertEqual(adjust(self.plan,self.state,self.p,self.c,self.ma,self.v,'vol')['buys'][0]['budget'],20000)
 def test_fast_and_future_invariance(self):
  a=adjust(self.plan,self.state,self.p,self.c,self.ma,self.v,'vol_fast');self.assertEqual(a['sells'],[{'symbol':'A','reason':'close_below_sma10'}])
  self.c.loc[self.d[1]:]=1e6;self.v.loc[self.d[1]:]=100
  self.assertEqual(a,adjust(self.plan,self.state,self.p,self.c,self.ma,self.v,'vol_fast'))
if __name__=='__main__':unittest.main()
