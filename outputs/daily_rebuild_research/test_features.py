import unittest,copy
import numpy as np,pandas as pd
import rebuild as b

class FeatureTests(unittest.TestCase):
 def fixture(self):
  rng=np.random.RandomState(33);idx=pd.bdate_range('2010-01-01',periods=600);c=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0008,.015,(600,4)),axis=0)),index=idx,columns=['SPY','A','B','C'])
  return dict(adj=c,close=c,opening=c*.999,high=c*1.02,low=c*.98,volume=c*0+1e7,eligible=c>0,sectors=pd.DataFrame('Tech',index=idx,columns=c.columns),sector60=c*0,market_close=c.SPY)
 def test_close_target_is_actual_close_to_close(self):
  d=self.fixture();f=b.features(d,'same_close_diagnostic')
  self.assertAlmostEqual(f['y'].iloc[300,1],d['adj'].iloc[305,1]/d['adj'].iloc[300,1]-1)
 def test_future_prices_cannot_change_either_clock_features(self):
  d=self.fixture();later=copy.deepcopy(d)
  for key in ['adj','close','opening','high','low','volume']:
   later[key]=later[key].copy();later[key].iloc[500:]*=1.25
  for clock in ['next_open','same_close_diagnostic']:
   first=b.features(d,clock);second=b.features(later,clock)
   for key in ['rule','event','histmean','momentum','atr']:
    pd.testing.assert_frame_equal(first[key].iloc[:500],second[key].iloc[:500])
if __name__=='__main__':unittest.main()
