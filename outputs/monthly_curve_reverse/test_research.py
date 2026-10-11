import unittest,json
import numpy as np,pandas as pd
from extract import P
from inverse import portfolio_paths,interpolation_matrix,score_paths
from hypotheses import simulate,metrics

class ResearchTests(unittest.TestCase):
 def test_fixed_shares_not_daily_equal_weight(self):
  c,p=portfolio_paths(np.array([[100,100,100],[200,100,100],[200,200,100]],float))
  i=np.flatnonzero(np.all(c==[0,1,2],axis=1))[0]
  np.testing.assert_allclose(p[i],[1,4/3,5/3]);self.assertNotAlmostEqual(p[i,2],(4/3)*(1+1/3))
 def test_two_positions_and_cash(self):
  c,p=portfolio_paths(np.array([[100,100,100],[200,100,100]],float));i=np.flatnonzero(np.all(c==[0,1,3],axis=1))[0]
  self.assertAlmostEqual(p[i,-1],1.5);np.testing.assert_allclose(p[-1],1)
 def test_calendar_interpolation(self):
  m=interpolation_matrix(np.array([0.,3.,4.]),np.array([1.5,3.5]));np.testing.assert_allclose(m.sum(axis=0),1);np.testing.assert_allclose(np.array([100,130,90])@m,[115,110])
 def test_exact_curve_recovers_known_triple(self):
  rng=np.random.default_rng(731);prices=np.exp(np.r_[np.zeros((1,6)),rng.normal(0,.03,(21,6))].cumsum(axis=0))*100;c,p=portfolio_paths(prices);truth=np.flatnonzero(np.all(c==[0,2,4],axis=1))[0];nav=p[truth];x=np.arange(len(nav),dtype=float)*2
  cal={'wealth':{'y0':464,'log_scale':39.5},'relative':{'y0':713,'log_scale':26.1},'dd':{'y0':801,'scale':370}}
  ob={}
  for panel in ['wealth','relative']:
   a=cal[panel];y=a['y0']-a['log_scale']*np.log(nav);ob[panel]=pd.DataFrame(dict(x=x,log_wealth=np.log(nav),y=y,ymin=y,ymax=y))
  y=801-370*(nav/np.maximum.accumulate(nav)-1);ob['dd']=pd.DataFrame(dict(x=x,y=y,ymin=y,ymax=y))
  result=score_paths(p,x,ob,cal);score=result[0];self.assertEqual(int(np.argmin(score)),truth);self.assertLess(result[3][truth],1e-20) # wealth exact; DD includes raster-footprint center penalty
 def test_identical_assets_not_identifiable(self):
  prices=np.column_stack([np.array([100,105,103,110])]*5);c,p=portfolio_paths(prices);np.testing.assert_allclose(p[:-1],np.tile(p[0],(len(p)-1,1)))
 def data(self):
  dates=pd.bdate_range('2015-01-01','2016-10-31');cols=['A','B','C','D'];px=pd.DataFrame({s:100*np.exp(np.arange(len(dates))*(i+1)/10000) for i,s in enumerate(cols)},index=dates);member=pd.DataFrame(True,index=dates,columns=cols);score=pd.DataFrame(np.tile([4,3,2,1],(len(dates),1)),index=dates,columns=cols);return px,member,score,member.copy()
 def test_fewer_than_two_cash(self):
  p,m,s,g=self.data();g.loc[:,['B','C','D']]=False;nav,orders=simulate(p,m,s,g,False);np.testing.assert_allclose(nav,1);self.assertEqual(orders[0]['holdings'],[])
 def test_future_prices_cannot_change_prior_orders(self):
  p,m,s,g=self.data();a,oa=simulate(p,m,s,g,False);p2=p.copy();p2.loc['2016-09-15':]*=1.3;b,ob=simulate(p2,m,s,g,False);pd.testing.assert_series_equal(a.loc[:'2016-09-14'],b.loc[:'2016-09-14']);self.assertEqual(oa[0],ob[0])
 def test_missing_held_price_fails(self):
  p,m,s,g=self.data();p.at[pd.Timestamp('2016-09-09'),'A']=np.nan
  with self.assertRaisesRegex(ValueError,'Cannot value held'):simulate(p,m,s,g,False)
 def test_metrics(self):
  z=pd.Series([1,1.5,1.2,2],index=pd.bdate_range('2020-01-01',periods=4));r=metrics(z);self.assertAlmostEqual(r['wealth'],2);self.assertAlmostEqual(r['max_dd'],-.2)
 def test_axis_validation(self):
  c=json.loads((P/'CALIBRATION.json').read_text());self.assertLess(c['wealth']['validation_rmse'],.75);self.assertLess(abs(c['qqq_wealth']-6.78),.001);self.assertLess(c['qqq_dd_nearest_visible_pixel_median'],1)
if __name__=='__main__':unittest.main(verbosity=2)
