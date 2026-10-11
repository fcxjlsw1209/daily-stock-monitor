import unittest
import numpy as np
import pandas as pd
import return_priority as m

class MonthlyPortfolioTests(unittest.TestCase):
 def test_monthly_fixed_units_not_daily_equal_weights(self):
  blocks=[np.array([[1.,1.],[2.,1.],[2.,2.]])]
  r=m.fast_returns([[0,1]],blocks,[np.array([True,True])])
  np.testing.assert_allclose(r,[.5,1/3]);self.assertAlmostEqual(np.prod(1+r),2.)
 def test_three_then_two_then_cash(self):
  sc=np.tile([3.,2.,1.],(3,1));good=np.array([[1,1,1],[0,1,1],[0,0,1]],bool)
  self.assertEqual(m.choose(sc,good,[list('ABC')]*3),[[0,1,2],[1,2],[]])
 def test_top5_retains_previous_only_at_monthend(self):
  sc=np.array([[6,5,4,3,2,1],[6,3,2,5,4,1]],float);good=np.ones_like(sc,bool)
  keep=m.choose(sc,good,[list('ABCDEF')]*2,5);fresh=m.choose(sc,good,[list('ABCDEF')]*2,3)
  self.assertEqual(keep,[[0,1,2],[0,1,2]]);self.assertEqual(fresh,[[0,1,2],[0,3,4]])
 def test_removal_from_pool_forces_monthly_replacement(self):
  sc=np.array([[3.,2.,1.],[3.,2.,1.]]);good=np.ones_like(sc,bool)
  self.assertEqual(m.choose(sc,good,[list('ABC'),list('ADE')],5),[[0,1,2],[0,1,2]])
 def test_missing_selected_price_raises(self):
  blocks=[np.array([[1.,1.],[1.1,np.nan]])];valid=[np.array([True,False])]
  with self.assertRaises(AssertionError):m.fast_returns([[0,1]],blocks,valid)
  np.testing.assert_allclose(m.fast_returns([[0]],blocks,valid),[.1])
 def test_training_decisions_ignore_later_features(self):
  names=['core_252_raw_v0','vol126','cmf21','market_trend','market_highvol','downside63','cmf63','acceleration_skip42','own_sector_skip42_252'];rng=np.random.default_rng(42);X=rng.normal(size=(4,5,len(names)))
  meta=dict(features=names,symbols=[list('ABCDE')]*4);data=dict(X=X,eligible=np.ones((4,5),bool));spec=m.specs()[0]
  ch,_=m.decisions(spec,meta,data,2);X[2:]*=10000;data['eligible'][2:]=False
  self.assertEqual(ch,m.decisions(spec,meta,data,2)[0])
 def test_training_paths_ignore_future_prices(self):
  idx=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']);px=pd.DataFrame({'A':[10.,11.,12.]},index=idx);meta=dict(dates=['2020-01-01'],ends=['2020-01-02'],symbols=[['A']])
  b,_,_,v=m.paths(px,meta,1);px.iloc[-1]=99999;b2,_,_,_=m.paths(px,meta,1);np.testing.assert_array_equal(b[0],b2[0])
 def test_drawdown_includes_initial_value(self):
  s=m.stats_rr(np.array([-.2,.1]),'2021-01-01','2022-01-01');self.assertAlmostEqual(s['max_drawdown'],-.2);self.assertAlmostEqual(s['total_return'],-.12)
 def test_independent_share_replay_matches_engine(self):
  idx=pd.bdate_range('2021-01-04',periods=5);px=pd.DataFrame({'A':[10,12,13,12,14],'B':[10,9,8,10,11]},index=idx,dtype=float)
  orders=[dict(date=str(idx[0].date()),end=str(idx[2].date()),signal_date=str(idx[0].date()),symbols=['A','B'],scores=[]),dict(date=str(idx[2].date()),end=str(idx[-1].date()),signal_date=str(idx[2].date()),symbols=['B'],scores=[])]
  _,eq,_,_=m.e.engine.replay(px,px,orders,cost=0.,execution='close',start=idx[0],end=idx[-1]);np.testing.assert_allclose(eq,m.e.previous.independent(px,orders),atol=1e-8)

if __name__=='__main__':unittest.main()
