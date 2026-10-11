import unittest
import numpy as np
import pandas as pd
from run import simulate

class ExecutionTests(unittest.TestCase):
 def fixture(self,short=False):
  ix=pd.bdate_range('2018-12-28',periods=28)
  raw={k:pd.DataFrame(100.,index=ix,columns=['A']) for k in ['Open','Close']}
  raw['Dividends']=pd.DataFrame(0.,index=ix,columns=['A'])
  meta=pd.DataFrame({'CIK':[1],'GICS Sector':['Test']},index=['A'])
  mask=pd.DataFrame(False,index=ix,columns=['A']);mask.iloc[0]=True
  none=mask&False;score=mask.astype(float)
  return raw,meta,(none,score,mask,score) if short else (mask,score,none,score)
 def test_next_open_hold_cost_accounting(self):
  r,m,f=self.fixture();r['Open'].iloc[1]=110
  d,t,o,e=simulate(r,m,f,10,start='2018-12-31')
  self.assertEqual(o.iloc[0].price,110)
  self.assertEqual(t.iloc[0].sessions,20)
  self.assertEqual(o.iloc[0].date,'2018-12-31')
  self.assertAlmostEqual(d.iloc[-1].equity,100000+t.net_pnl.sum())
 def test_short_dividend_borrow_and_cost(self):
  r,m,f=self.fixture(True);r['Dividends'].iloc[2]=1
  d,t,o,e=simulate(r,m,f,10,.05,True,start='2018-12-31')
  self.assertLess(t.iloc[0].dividends,0)
  self.assertGreater(t.iloc[0].borrow,0)
  self.assertLess(t.iloc[0].net_pnl,-200)
  self.assertAlmostEqual(d.iloc[-1].equity,100000+t.net_pnl.sum())
 def test_future_close_does_not_change_entry(self):
  r,m,f=self.fixture();d,t,o,e=simulate(r,m,f,start='2018-12-31')
  r['Close'].iloc[1]=np.nan
  dd,tt,oo,ee=simulate(r,m,f,start='2018-12-31')
  pd.testing.assert_series_equal(o.iloc[0],oo.iloc[0])
  self.assertTrue(any(x['event']=='stale_close_mark' for x in ee))
 def test_delayed_exit_is_reported(self):
  r,m,f=self.fixture();r['Open'].iloc[21]=np.nan
  d,t,o,e=simulate(r,m,f,start='2018-12-31')
  self.assertEqual(t.iloc[0].sessions,21)
  self.assertTrue(any(x['event']=='exit_deferred_missing_open' for x in e))
if __name__=='__main__':unittest.main()
