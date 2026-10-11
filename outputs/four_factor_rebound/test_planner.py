import unittest
import pandas as pd
from run import planner,VARIANTS
class PlanTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2020-01-01',periods=5);self.c=pd.DataFrame(10.,index=self.dates,columns=['A','B','C']);self.score=self.c*.08;self.atr=self.c*.1;self.state={'positions':{},'equity':100000};self.sectors={'A':'Tech','B':'Tech','C':'Energy'};self.ids={s:s for s in self.c}
 def call(self,mask):return planner(VARIANTS[2])('baseline',self.state,self.dates[1],self.dates,self.c,self.c,self.atr,mask,self.score,self.sectors,list(self.c),self.ids)
 def test_two_or_three_entry_budgets(self):
  mask=self.c>0;r=self.call(mask);self.assertEqual(len(r['buys']),3);self.assertAlmostEqual(sum(x['budget'] for x in r['buys']),100000)
  mask['C']=False;r=self.call(mask);self.assertEqual([x['budget'] for x in r['buys']],[50000,50000])
 def test_one_candidate_means_cash_and_no_forced_pick(self):
  mask=self.c<0;mask['A']=True;r=self.call(mask);self.assertEqual(r['buys'],[])
if __name__=='__main__':unittest.main()
