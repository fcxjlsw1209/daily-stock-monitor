import unittest
import pandas as pd
from union_execution import plan,engine,ACCOUNTS

class UnionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2020-01-01',periods=30);self.cols=['old','new','other']
        self.c=pd.DataFrame(100.,index=self.dates,columns=self.cols)
        self.mask=self.c>0;self.score=self.c/100;self.sectors={s:'tech' for s in self.cols};self.issuers={'old':'same','new':'same','other':'different'}
    def call(self,name,state,reps,day=1):
        return plan(name,state,self.dates[day],self.dates,self.c,self.c+10,self.c*.02,self.mask,self.score,self.sectors,reps,self.issuers)
    def test_all_six_use_next_open_and_original_sizing(self):
        for name in ACCOUNTS:
            e=engine(name);state=e.initial_state();d=self.call(name,state,['new','other'])
            self.assertEqual(d['signal_date'],str(self.dates[0].date()))
            self.assertEqual(d['session'],str(self.dates[1].date()))
            self.assertEqual(d['buys'][0]['budget'],10000 if name in ['baseline','pressure','resilience'] else 20000)
            bars={s:dict(open=110.,close=110.,dividends=0.,split=1.) for s in ['new','other']}
            after,_,_,orders,_=e.execute(state,d,bars,self.dates,cost_bps=10,sector=self.sectors)
            self.assertEqual(orders[0]['price'],110.)
            self.assertEqual(state['positions'],{})
            self.assertGreater(after['fees'],0)
    def test_representative_change_does_not_duplicate_or_force_sell(self):
        for name in ACCOUNTS:
            e=engine(name);s=e.initial_state();d=self.call(name,s,['old'])
            s,*_=e.execute(s,d,{'old':dict(open=100.,close=100.,dividends=0.,split=1.)},self.dates,cost_bps=10,sector=self.sectors)
            d=self.call(name,s,['new','other'],day=2)
            self.assertEqual(d['sells'],[])
            self.assertNotIn('new',[x['symbol'] for x in d['buys']])
    def test_duplicate_classes_rejected(self):
        with self.assertRaises(ValueError):self.call('baseline',engine('baseline').initial_state(),['old','new'])
    def test_missing_sector_rejected(self):
        del self.sectors['new']
        with self.assertRaises(ValueError):self.call('baseline',engine('baseline').initial_state(),['new'])
if __name__=='__main__':unittest.main()
