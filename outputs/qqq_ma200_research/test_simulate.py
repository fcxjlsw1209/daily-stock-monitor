import unittest
import numpy as np
import pandas as pd
from simulate import load,factors,run,rebalance

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=load()
 def test_prefix_causality(self):
  a,b,s=factors(self.data);cut=pd.Timestamp('2005-06-30')
  x,y,z=factors({k:v.loc[:cut] for k,v in self.data.items()})
  pd.testing.assert_frame_equal(a.loc[:cut],x);pd.testing.assert_frame_equal(b.loc[:cut],y);pd.testing.assert_series_equal(s.loc[:cut],z)
 def test_actual_returns_not_double_charged(self):
  a,b,s=factors(self.data);d=self.data['QLD'];r=d['Adj Close']/d['Adj Close'].shift(1)
  np.testing.assert_allclose((a.QLD*b.QLD).reindex(r.index).iloc[1:],r.iloc[1:],rtol=1e-12)
 def test_synthetic_single_daily_reset(self):
  a,b,s=factors(self.data);q=self.data['QQQ'];d=pd.Timestamp('2000-01-04');prev=q.index[q.index.get_loc(d)-1]
  irx=self.data['IRX'].loc[:prev].Close.iloc[-1]/100;r=irx/(1-irx*91/360)*365/360
  expected=1+2*(q.loc[d,'Adj Close']/q.loc[prev,'Adj Close']-1)-((r+.005)+.0095)*(d-prev).days/365
  self.assertAlmostEqual(a.loc[d,'QLD']*b.loc[d,'QLD'],expected)
 def test_old_position_earns_overnight_then_new_intraday(self):
  ix=pd.date_range('2020-01-02',periods=3);a=pd.DataFrame(1.,index=ix,columns=['QQQ','QLD','TQQQ','cash']);b=a.copy()
  a.loc[ix[1],'QLD']=.8;b.loc[ix[1],'QLD']=.5
  b.loc[ix[2],'QLD']=1.1;a.loc[ix[2],'QLD']=2
  sig=pd.Series([True,False,True],index=ix)
  df,t=run(a,b,sig,'MA200_QLD_cash',start=str(ix[0].date()),fee=0)
  np.testing.assert_allclose(df.equity,[100000,80000,88000])
 def test_turnover_fee(self):
  new,cost=rebalance(np.array([100000.,0,0,0]),np.array([0.,1,0,0]),.0005)
  self.assertAlmostEqual(new.sum(),100000*(1-.0005)/(1+.0005));self.assertAlmostEqual(new.sum()+cost,100000)
 def test_signal_is_prior_close(self):
  a,b,s=factors(self.data);c=self.data['QQQ']['Adj Close'];expected=(c>c.rolling(200).mean()).shift(1)
  ix=s.dropna().index;np.testing.assert_array_equal(s.loc[ix],expected.loc[ix])
if __name__=='__main__':unittest.main()
