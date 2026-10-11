import unittest,json
from pathlib import Path
import numpy as np,pandas as pd
from research import P,O,Simulator,fee_adjusted_nav_fraction,load,raw_features,train_models,MODEL_FEATURES,POOLS

class ReconstructionTests(unittest.TestCase):
 def data(self):
  idx=pd.bdate_range('2014-01-01','2026-09-25');px=pd.DataFrame(100.,index=idx,columns=['A','B','C','D']);m=pd.DataFrame(True,index=idx,columns=px.columns);sc=pd.DataFrame(np.tile([4,3,2,1],(len(idx),1)),index=idx,columns=px.columns);fs={'opening':px.copy()};return px,fs,m,sc
 def test_fee_initial_buy(self):self.assertAlmostEqual(fee_adjusted_nav_fraction({}, {'A':1},10),1/1.0005,places=12)
 def test_fee_roundtrip_replacement(self):self.assertAlmostEqual(fee_adjusted_nav_fraction({'A':1},{'B':1},10),.9995/1.0005,places=12)
 def test_fee_no_trade(self):self.assertAlmostEqual(fee_adjusted_nav_fraction({'A':.5,'B':.5},{'A':.5,'B':.5},30),1,places=12)
 def test_pool_count_and_leverage_separation(self):
  for key,v in POOLS.items():self.assertEqual(len(set(v)),17)
  self.assertNotIn('TQQQ',POOLS['core17']);self.assertNotIn('TQQQ',POOLS['theme17']);self.assertIn('TQQQ',POOLS['leveraged17_DIAGNOSTIC'])
 def test_fixed_shares(self):
  px,fs,m,sc=self.data();px.loc['2016-09-01':,'A']=200;px.loc['2016-09-02':,'B']=200;sim=Simulator(px,fs,m);nav,o=sim.run(sc,m);self.assertAlmostEqual(nav.at[pd.Timestamp('2016-09-01')],4/3);self.assertAlmostEqual(nav.at[pd.Timestamp('2016-09-02')],5/3)
 def test_two_positions_equal(self):
  px,fs,m,sc=self.data();m.loc[:,['C','D']]=False;px.loc['2016-09-01':,'A']=200;nav,_=Simulator(px,fs,m).run(sc,m);self.assertAlmostEqual(nav.at[pd.Timestamp('2016-09-01')],1.5)
 def test_one_candidate_cash(self):
  px,fs,m,sc=self.data();m.loc[:,['B','C','D']]=False;nav,orders=Simulator(px,fs,m).run(sc,m);np.testing.assert_allclose(nav,1);self.assertEqual(orders[0]['holdings'],[])
 def test_missing_held_price_stops(self):
  px,fs,m,sc=self.data();px.at[pd.Timestamp('2016-09-06'),'A']=np.nan
  with self.assertRaisesRegex(ValueError,'Unpriceable held path'):Simulator(px,fs,m).run(sc,m)
 def test_signal_lag(self):
  px,fs,m,sc=self.data();sc.loc[pd.Timestamp('2016-08-31'),'D']=5;sim=Simulator(px,fs,m);_,a=sim.run(sc,m);_,b=sim.run(sc,m,signal_lag=1);self.assertIn('D',a[0]['holdings']);self.assertNotIn('D',b[0]['holdings']);self.assertEqual(b[0]['signal_date'],'2016-08-30')
 def test_label_maturity_all_fits(self):
  audit=json.loads((P/'TRAINING_AUDIT.json').read_text());self.assertEqual(len(audit),12)
  for records in audit.values():
   for row in records:
    self.assertLess(row['last_train_signal'],row['signal_date']);self.assertLessEqual(row['latest_label_end'],row['signal_date'])
 def test_legacy_monthly_replay_unchanged(self):
  old=json.loads((O/'monthly_curve_reverse/RULE_COMPARISON.json').read_text())['results']['r12_2_replace'];new=json.loads((P/'RESULTS.json').read_text())['core17__m252_42']
  for k in ['wealth','cagr_252','max_dd']:self.assertAlmostEqual(old[k],new[k],places=10)
 def test_inverse_monthly_continuity(self):
  for name in ['core17_q0','theme17_q0','theme17_q0.5']:
   z=json.loads((P/f'INVERSE_{name}.json').read_text());self.assertTrue(z['not_trading_strategy']);nav=pd.read_csv(P/f'INVERSE_{name}_NAV.csv',index_col=0,parse_dates=True).iloc[:,0];self.assertFalse(nav.index.duplicated().any());self.assertAlmostEqual(nav.iloc[0],1);self.assertAlmostEqual(nav.iloc[-1],z['wealth']);self.assertEqual(len(z['orders']),121)
if __name__=='__main__':unittest.main(verbosity=2)
