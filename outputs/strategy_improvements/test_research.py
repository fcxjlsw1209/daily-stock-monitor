import unittest,copy
import numpy as np,pandas as pd
import research as r
class NetTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2020-01-01',periods=10);self.state=r.ENGINE.initial_state();self.bars={s:dict(open=p,close=p,dividends=0.,split=1.) for s,p in [('A',10.),('B',20.),('C',50.)]}
 def dec(self,i=1,weight=None):
  weight={'A':.5,'B':.5} if weight is None else weight
  return dict(session=str(self.dates[i].date()),sells=[],buys=[dict(symbol=s,budget=1.,atr=1.) for s in weight],target_weights=weight)
 def test_repeat_identical_target_no_turnover(self):
  a=r.net_execute('baseline',self.state,self.dec(),self.bars,self.dates);b=r.net_execute('baseline',a[0],self.dec(2),self.bars,self.dates)
  self.assertFalse(b[3]);self.assertAlmostEqual(a[0]['equity'],b[0]['equity']);self.assertAlmostEqual(a[0]['cash'],0.)
 def test_partial_sale_conserves_cash_and_basis(self):
  a=r.net_execute('baseline',self.state,self.dec(),self.bars,self.dates)[0]
  b=r.net_execute('baseline',a,self.dec(2,{'A':.25,'B':.25}),self.bars,self.dates)
  self.assertTrue(all(o['side']=='sell' for o in b[3]));self.assertEqual(len(b[2]),2)
  self.assertAlmostEqual(b[0]['equity'],a['equity']-sum(o['fee'] for o in b[3]))
  for s in ['A','B']:
   trade=next(t for t in b[2] if t['symbol']==s);self.assertAlmostEqual(b[0]['positions'][s]['cost']+trade['entry_outlay'],a['positions'][s]['cost'])
  self.assertAlmostEqual(b[0]['cash']/b[0]['equity'],.5)
 def test_split_dividend_applied_once(self):
  a=r.net_execute('baseline',self.state,self.dec(),self.bars,self.dates)[0];q=a['positions']['A']['qty'];bars=copy.deepcopy(self.bars);bars['A']=dict(open=5.,close=5.,dividends=.1,split=2.)
  b=r.net_execute('baseline',a,dict(session=str(self.dates[2].date()),sells=[],buys=[]),bars,self.dates)
  self.assertAlmostEqual(b[0]['positions']['A']['qty'],q*2);self.assertAlmostEqual(b[0]['equity'],a['equity']+q*.2)
 def test_zero_cost_matches_equal_liquidation(self):
  a=r.net_execute('baseline',self.state,self.dec(),self.bars,self.dates,cost_bps=0)[0];bars=copy.deepcopy(self.bars);bars['A']['open']=bars['A']['close']=15.
  b=r.net_execute('baseline',a,self.dec(2,{'B':.5,'C':.5}),bars,self.dates,cost_bps=0)[0]
  decision=self.dec(2,{'B':.5,'C':.5});decision['sells']=[dict(symbol=s,reason='test') for s in a['positions']]
  c=r.r.execute_equal_close('baseline',a,decision,bars,self.dates,cost_bps=0)[0]
  self.assertAlmostEqual(b['equity'],c['equity'])
  for s in b['positions']:self.assertAlmostEqual(b['positions'][s]['qty'],c['positions'][s]['qty'])
 def test_no_future_price_sizing(self):
  bars=copy.deepcopy(self.bars);bars['A']['close']=9999.
  a=r.net_execute('baseline',self.state,self.dec(),self.bars,self.dates)[0];b=r.net_execute('baseline',self.state,self.dec(),bars,self.dates)[0]
  self.assertEqual(a['positions'],b['positions'])
 def test_negative_or_leveraged_weights_rejected(self):
  for w in [{'A':1.1},{'A':-.1}]:
   with self.assertRaises(ValueError):r.net_execute('baseline',self.state,self.dec(weight=w),self.bars,self.dates)
 def test_staged_deadline_and_previous_signal(self):
  dates=pd.bdate_range('2020-01-01','2020-02-10');cols=['A','B','C'];score=pd.DataFrame([.9,.8,.7],index=cols).T.reindex(range(len(dates))).ffill();score.index=dates;mask=score.gt(0);ret=score*0;event=mask&False;sectors=dict(zip(cols,['x','y','z']));ids={s:s for s in cols};end=pd.Timestamp('2020-01-31');p=r.monthly_plan('staged5',{end},ret,event)
  def step(day):return p('baseline',self.state,pd.Timestamp(day),dates,score*100,score,score,mask,score,sectors,cols,ids)
  a=step('2020-01-31');self.assertAlmostEqual(sum(a['target_weights'].values()),.5)
  event.loc['2020-02-03']=True;b=step('2020-02-03');self.assertNotIn('target_weights',b)
  c=step('2020-02-04');self.assertAlmostEqual(sum(c['target_weights'].values()),1.)
  event[:]=False;p=r.monthly_plan('staged5',{end},ret,event);step('2020-01-31')
  self.assertNotIn('target_weights',step('2020-02-06'));self.assertIn('target_weights',step('2020-02-07'))
if __name__=='__main__':unittest.main()

class FeatureTests(unittest.TestCase):
 def test_future_prices_do_not_change_past_factors(self):
  dates=pd.bdate_range('2017-01-01',periods=520);cols=['A','B','SPY'];rng=np.random.RandomState(42);a=pd.DataFrame(100*np.cumprod(1+rng.normal(.0005,.01,(520,3)),axis=0),index=dates,columns=cols)
  sectors=pd.DataFrame([['Tech','Tech','Market']]*520,index=dates,columns=cols);d=dict(adj=a,sectors=sectors,sector60=a.pct_change(60,fill_method=None),issuers={'A':'A','B':'B','SPY':'FUND_SPY'});mf=dict(valid=a.notna(),base=a.rank(axis=1,pct=True));u=a.notna();before=r.added_features(d,u,mf)
  d2=copy.deepcopy(d);d2['adj'].iloc[490:]*=2;d2['sector60'].iloc[490:]=100;after=r.added_features(d2,u,mf)
  for k in ['industry_score','residual_score','path_score','breadth']:
   pd.testing.assert_frame_equal(before[k].iloc[:490],after[k].iloc[:490])
