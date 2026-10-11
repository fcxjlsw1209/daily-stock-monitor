import unittest,copy
import pandas as pd,numpy as np
from research import combo_daily,monthly_plan,execute_equal_close,model,h
class ResearchTest(unittest.TestCase):
 def test_one_capital(self):
  a=[dict(date='x',equity=200000)];b=[dict(date='x',equity=150000)]
  self.assertEqual(combo_daily(a,b,.5)[0]['equity'],175000)
 def test_month_end_prior_signal_only(self):
  dates=pd.bdate_range('2020-01-01','2020-02-03');sc=pd.DataFrame(.5,index=dates,columns=['A','B','C']);mask=sc.gt(0);close=sc*200;atr=sc*2;sectors={'A':'a','B':'b','C':'c'};issuers={s:s for s in sectors};state=h.engine('baseline').initial_state();plan=monthly_plan({pd.Timestamp('2020-01-31')})
  d=plan('baseline',state,pd.Timestamp('2020-01-30'),dates,close,close,atr,mask,sc,sectors,list(sectors),issuers);self.assertFalse(d['buys'])
  args=('baseline',state,pd.Timestamp('2020-01-31'),dates,close,close,atr,mask,sc,sectors,list(sectors),issuers);before=plan(*args);sc.loc['2020-01-31']=[100,0,0];after=plan(*args);self.assertEqual(before,after)
 def test_equal_weights_after_liquidation_cost(self):
  dates=pd.bdate_range('2020-01-01',periods=3);state=h.engine('baseline').initial_state();decision=dict(session=str(dates[1].date()),sells=[],buys=[dict(symbol=s,budget=1,atr=1) for s in ['A','B','C']]);bars={s:dict(open=p,close=p,split=1,dividends=0) for s,p in zip(['A','B','C'],[10,20,40])}
  out=execute_equal_close('baseline',state,decision,bars,dates);st=out[0];vals=[p['qty']*bars[s]['close'] for s,p in st['positions'].items()];self.assertAlmostEqual(max(vals),min(vals));self.assertAlmostEqual(st['cash'],0);self.assertLess(st['equity'],100000)
  dec=copy.deepcopy(decision);dec['session']=str(dates[2].date());dec['sells']=[dict(symbol=s,reason='month_end') for s in st['positions']];after=execute_equal_close('baseline',st,dec,bars,dates)[0];self.assertLess(after['equity'],st['equity']);self.assertEqual(len(after['positions']),3)
 def test_model_mature_labels(self):
  idx=pd.bdate_range('2017-01-01','2019-03-29');rng=np.random.RandomState(1);x=pd.DataFrame(rng.rand(len(idx),4),index=idx);f=dict(valid=x.notna(),train=x.notna(),X=[x,x*.8],y=x*.03)
  p,_,audit=model(f,22);self.assertTrue(all(r['latest_label_session']<r['first_prediction'] for r in audit))
  f['y']=f['y'].copy();f['y'].loc['2019-01-01':]=1000;q,_,_=model(f,22);pd.testing.assert_frame_equal(p.loc['2019-01'],q.loc['2019-01'])
if __name__=='__main__':unittest.main()
