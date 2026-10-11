import unittest,copy
import numpy as np
import pandas as pd
import adaptation as a

class CausalityTests(unittest.TestCase):
 def test_weighted_ridge_equal_weights_match_closed_form(self):
  rng=np.random.RandomState(8);x=rng.normal(size=(700,4));y=.02*x[:,0]-.01*x[:,1]+rng.normal(0,.01,700)
  mu,sd,ym,coef=a.weighted_ridge(x,y,np.ones(len(y)))
  z=(x-x.mean(0))/x.std(0);expected=np.linalg.solve(np.einsum('ni,nj->ij',z,z)+10*np.eye(4),np.einsum('ni,n->i',z,y-y.mean()))
  np.testing.assert_allclose(coef,expected,rtol=1e-12,atol=1e-12)
 def test_weighted_model_excludes_unmatured_outcomes(self):
  dates=pd.bdate_range('2015-01-01','2019-01-31');rng=np.random.RandomState(88);x=pd.DataFrame(rng.normal(size=(len(dates),4)),index=dates,columns=list('ABCD'))
  f=dict(valid=x.notna(),train=x.notna(),X=[x,x*.5],y=x*.01)
  p,audit=a.walk_model(f,halflife=252);altered=copy.deepcopy(f);cut=dates.get_loc(pd.Timestamp('2019-01-01'))-6;altered['y'].iloc[cut:]=1000
  q,_=a.walk_model(altered,halflife=252);pd.testing.assert_frame_equal(p,q)
  self.assertTrue(all(v['latest_label']<v['first_prediction'] for v in audit))
 def test_recent_history_ignores_unmatured_values(self):
  dates=pd.bdate_range('2015-01-01',periods=500);rng=np.random.RandomState(9);x=pd.DataFrame(rng.normal(0,.02,(500,3)),index=dates,columns=list('ABC'));event=x.notna();p=a.recent_history(x,event)
  changed=x.copy();changed.iloc[395:]=999;q=a.recent_history(changed,event)
  pd.testing.assert_frame_equal(p.iloc[:401],q.iloc[:401]);self.assertTrue(q.iloc[401:].gt(p.iloc[401:]).any().any())
 def test_monthly_gap_uses_only_prior_month_information(self):
  dates=pd.bdate_range('2018-01-01','2020-03-31');refs={g:pd.Series(g*.001,index=dates) for g in [.1,.15,.2]}
  p,audit=a.monthly_gap(dates,refs);changed=copy.deepcopy(refs);changed[.1].loc['2020-02-03':]=100
  q,_=a.monthly_gap(dates,changed)
  pd.testing.assert_series_equal(p.loc[:'2020-02-28'],q.loc[:'2020-02-28']);self.assertEqual(q.loc['2020-03-02'],.1)
  self.assertTrue(all(v['selected_on']<v['first_signal'] for v in audit))
 def test_dispersion_uses_only_completed_signal_rows(self):
  dates=pd.bdate_range('2018-01-01',periods=400);rng=np.random.RandomState(12);s=pd.DataFrame(rng.uniform(size=(400,8)),index=dates)
  p,_=a.gap_schedule(dates,s,'dispersion_gap');changed=s.copy();changed.iloc[300:]*=10;q,_=a.gap_schedule(dates,changed,'dispersion_gap')
  pd.testing.assert_series_equal(p.iloc[:300],q.iloc[:300]);self.assertTrue(p.between(.075,.225).all())

class EntryLimitTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2020-01-01',periods=5);self.state=a.old.prior.ENGINE.initial_state()
  self.decision=dict(session=str(self.dates[1].date()),sells=[],buys=[dict(symbol=s,budget=50000.,reference_close=100.,atr=10.) for s in ['A','B']])
  self.bars={s:dict(open=104.,close=104.,split=1.,dividends=0.) for s in ['A','B']}
 def test_opening_above_fixed_limit_cancelled_without_replacement(self):
  self.bars['A']['open']=106.;out=a.execute_limit('baseline',self.state,self.decision,self.bars,self.dates)
  self.assertEqual(set(out[0]['positions']),{'B'});self.assertEqual(out[5][0]['reason'],'opening_above_predeclared_limit')
 def test_split_normalizes_predeclared_limit(self):
  self.bars['A']=dict(open=53.,close=53.,split=2.,dividends=0.)
  out=a.execute_limit('baseline',self.state,self.decision,self.bars,self.dates);self.assertEqual(out[5][0]['limit'],52.5)
 def test_fill_decision_does_not_read_intraday_close(self):
  p=a.execute_limit('baseline',copy.deepcopy(self.state),self.decision,self.bars,self.dates);changed=copy.deepcopy(self.bars);changed['A']['close']=1.
  q=a.execute_limit('baseline',copy.deepcopy(self.state),self.decision,changed,self.dates);self.assertEqual(p[3],q[3]);self.assertEqual(p[5],q[5])

if __name__=='__main__':unittest.main()

class StableTieTests(unittest.TestCase):
 def test_near_equal_scores_keep_previous_parameter(self):
  from stable_ties import schedule
  dates=pd.bdate_range('2018-01-01','2020-02-28');refs={g:pd.Series(.001,index=dates) for g in [.1,.15,.2]}
  refs[.1]+=.000000000001
  selected,audit=schedule(dates,refs)
  self.assertTrue(selected.eq(.15).all())
 def test_later_returns_do_not_change_current_choice(self):
  from stable_ties import schedule
  dates=pd.bdate_range('2018-01-01','2020-02-28');refs={g:pd.Series(g*.001,index=dates) for g in [.1,.15,.2]}
  p,_=schedule(dates,refs);refs[.1].loc['2020-01-02':]=100;q,_=schedule(dates,refs)
  pd.testing.assert_series_equal(p.loc[:'2020-01-31'],q.loc[:'2020-01-31'])
