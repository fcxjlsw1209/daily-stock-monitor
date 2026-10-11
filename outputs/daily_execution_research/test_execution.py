import unittest,copy
import numpy as np
import pandas as pd
import run_execution as e

class PlanTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2020-01-01',periods=15);self.day=self.dates[5];self.cols=['A','B','C','D','E']
  self.close=pd.DataFrame(100.,index=self.dates,columns=self.cols);self.atr=self.close*.1
  self.score=pd.DataFrame(np.tile([.9,.6,.62,.8,.7],(len(self.dates),1)),index=self.dates,columns=self.cols)
  self.mask=self.score.ge(.75);self.state=e.prior.ENGINE.initial_state()
  self.state['positions']={s:dict(entry_date=str(self.dates[2].date()),entry_price=100.,atr=10.,qty=300,cost=30000.,entry_fee=15.,dividends=0.) for s in ['A','B','C']}
  self.sectors=dict(A='x',B='y',C='z',D='y',E='z');self.ids={s:s for s in self.cols}
 def call(self,kind,day=None):
  return e.planner(kind)('baseline',copy.deepcopy(self.state),day or self.day,self.dates,self.close,self.close,self.atr,self.mask,self.score,self.sectors,self.cols,self.ids)
 def test_held_high_score_cannot_replace_two_others(self):
  self.mask['D']=False
  old=self.call('validated_base');new=self.call('matched_replacement')
  self.assertEqual({s['symbol'] for s in old['sells']},{'A','B','C'})
  self.assertFalse(new['sells']);self.assertFalse(new['buys'])
 def test_one_new_issuer_causes_only_one_replacement(self):
  x=self.call('matched_replacement')
  self.assertEqual([s['symbol'] for s in x['sells']],['B'])
  self.assertEqual([s['symbol'] for s in x['buys']],['D'])
  self.assertEqual(x['replacement_pairs'][0]['sell'],'B')
 def test_mandatory_vacancy_reserves_candidate(self):
  self.close.loc[self.dates[4],'C']=70.
  x=self.call('matched_replacement');self.assertEqual([s['symbol'] for s in x['sells']],['C']);self.assertEqual([s['symbol'] for s in x['buys']],['D']);self.assertFalse(x['replacement_pairs'])
 def test_duplicate_issuer_not_admissible(self):
  self.ids['D']='A';self.assertFalse(self.call('matched_replacement')['sells'])
 def test_net_weights_only_on_membership_changes(self):
  self.mask['D']=False;x=self.call('matched_net');self.assertNotIn('target_weights',x)
  self.mask['D']=True;x=self.call('matched_net');self.assertEqual(set(x['target_weights']),{'A','C','D'});self.assertAlmostEqual(sum(x['target_weights'].values()),1.)
 def test_future_bars_do_not_change_past_decision(self):
  for kind in ['matched_replacement','net_targets','trailing_profit','no_progress']:
   x=self.call(kind);original=self.close.copy();self.close.loc[self.day:]=10000.
   self.assertEqual(x,self.call(kind));self.close=original
 def test_trailing_profit_uses_only_since_entry_peak(self):
  self.score[:]=.8;self.mask[:]=True;self.close.loc[:self.dates[1],'A']=1000.
  self.close.loc[self.dates[3],'A']=140.;self.close.loc[self.dates[4],'A']=130.
  self.assertNotIn('A',[s['symbol'] for s in self.call('trailing_profit')['sells']])
  self.close.loc[self.dates[4],'A']=115.
  self.assertEqual(next(s['reason'] for s in self.call('trailing_profit')['sells'] if s['symbol']=='A'),'close_trailing_profit')
 def test_missing_one_candidate_requires_cash(self):
  self.state['positions']={};self.mask[:]=False;self.mask['A']=True
  x=self.call('matched_replacement');self.assertFalse(x['buys'])
 def test_sector_cap_filters_candidate(self):
  self.state['positions']={s:self.state['positions'][s] for s in ['A','B']};self.sectors.update(A='x',B='x',D='x');self.mask['C']=False;self.score[['A','B']]=.9
  x=self.call('sector_cap2');self.assertFalse(x['buys'])

class FeatureTests(unittest.TestCase):
 def test_no_future_feature_leakage(self):
  idx=pd.bdate_range('2018-01-01',periods=400);rng=np.random.RandomState(49)
  a=pd.DataFrame(100*np.cumprod(1+rng.normal(.0002,.01,(400,3)),axis=0),index=idx,columns=['A','B','SPY']);d=dict(adj=a,close=a,volume=a*20000)
  x=e.causal_features(d,a.notna());changed=copy.deepcopy(d)
  for v in changed.values():v.iloc[300:]*=2
  y=e.causal_features(changed,a.notna())
  for k in x:pd.testing.assert_frame_equal(x[k].iloc[:300],y[k].iloc[:300])
 def test_net_orders_do_not_use_close_for_sizing(self):
  dates=pd.bdate_range('2020-01-01',periods=3);s=e.prior.ENGINE.initial_state();bars={x:dict(open=100.,close=100.,split=1.,dividends=0.) for x in ['A','B']}
  decision=dict(session=str(dates[1].date()),sells=[],buys=[dict(symbol=x,budget=50000.,atr=3.) for x in bars],target_weights={'A':.5,'B':.5})
  x=e.prior.net_execute('baseline',copy.deepcopy(s),decision,bars,dates);changed=copy.deepcopy(bars);changed['A']['close']=9999.
  y=e.prior.net_execute('baseline',copy.deepcopy(s),decision,changed,dates)
  self.assertEqual(x[3],y[3]);self.assertGreaterEqual(x[0]['cash'],-1e-7);self.assertAlmostEqual(x[0]['equity'],100000.-sum(v['fee'] for v in x[3]))

if __name__=='__main__':unittest.main()
