import unittest,copy
import numpy as np,pandas as pd
import allocation as a
class AllocationTests(unittest.TestCase):
 def fixture(self):
  dates=pd.bdate_range('2020-01-01',periods=80);c=pd.DataFrame(25.,index=dates,columns=['A','B']);ctx=dict(ret=c.pct_change(),raw_per_signal=c*0+4)
  state=dict(equity=100000,cash=0,positions={s:dict(qty=500,atr=2.,entry_price=25,entry_date=str(dates[0].date())) for s in c})
  day=dates[70];decision=dict(session=str(day.date()),signal_date=str(dates[69].date()),sells=[],buys=[],candidate_count=2,equity_basis=100000)
  return dates,c,ctx,state,day,decision
 def call(self,cfg,fixture):
  dates,c,ctx,state,day,decision=fixture
  return a.decorate(copy.deepcopy(decision),state,day,dates,c,c*0+2,c*0+.8,c*0+.8,ctx,cfg)
 def test_no_net_order_without_membership_change(self):
  out=self.call(dict(schedule='change'),self.fixture());self.assertNotIn('target_weights',out)
 def test_band_compares_values_in_ledger_units(self):
  fixture=self.fixture();out=self.call(dict(schedule='band'),fixture);self.assertNotIn('target_weights',out)
  fixture[3]['positions']['A']['qty']=650;fixture[3]['positions']['B']['qty']=350
  out=self.call(dict(schedule='band'),fixture);self.assertEqual(out['target_weights'],{'A':.5,'B':.5})
 def test_weight_tilt_is_capped_long_only(self):
  targets=['A','B','C'];mom=pd.Series([1,.001,.001],index=targets);ret=pd.DataFrame(np.random.RandomState(3).normal(size=(63,3))*[.001,.3,.3],columns=targets)
  for mode in ['equal','momentum','inversevol']:
   w=a.weights_for(targets,mode,mom,ret);self.assertAlmostEqual(sum(w.values()),1);self.assertTrue(all(0<=v<=.5+1e-10 for v in w.values()))
 def test_allocation_does_not_read_execution_day_close(self):
  x=self.fixture();x[-1]['buys']=[dict(symbol='A')];cfg=dict(schedule='change');first=self.call(cfg,x);x[1].loc[x[4]:]*=10
  self.assertEqual(first,self.call(cfg,x))
 def test_net_execution_cash_basis_and_split_once(self):
  engine=a.z.t.a.old.prior;dates=pd.bdate_range('2020-01-01',periods=3);state=engine.ENGINE.initial_state();state['cash']=50000;state['positions']={'A':dict(entry_date='2020-01-01',entry_price=100.,qty=500.,cost=50000.,entry_fee=0.,dividends=0.,atr=5.)}
  decision=dict(session='2020-01-02',target_weights={'A':.5,'B':.5},sells=[],buys=[dict(symbol='A',atr=2.5),dict(symbol='B',atr=2.)]);bars={'A':dict(open=50.,close=50.,split=2.,dividends=1.),'B':dict(open=100.,close=100.,split=1.,dividends=0.)}
  out,row,trades,orders,events,cancelled=engine.net_execute('baseline',state,decision,bars,dates,cost_bps=10)
  self.assertGreaterEqual(out['cash'],-1e-6);self.assertAlmostEqual(out['dividends'],1000.)
  self.assertAlmostEqual(out['equity']+out['fees'],101000.,places=6)
  self.assertAlmostEqual(sum(p['qty']*bars[s]['close'] for s,p in out['positions'].items())+out['cash'],out['equity'])
if __name__=='__main__':unittest.main()
