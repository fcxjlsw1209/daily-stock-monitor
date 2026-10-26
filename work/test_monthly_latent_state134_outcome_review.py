"""Synthetic accounting and tamper tests; no real prices or outcome files read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_latent_state134_outcome_review as m

def order(day,end,names):return dict(date=str(day.date()),signal_date=str(day.date()),end=str(end.date()),symbols=list(names),scores=[.2]*len(names))

def small_prices():
 days=pd.bdate_range('2016-01-04',periods=4)
 close=pd.DataFrame({'A':[10.,20.,30.,40.],'B':[20.,20.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 opening=pd.DataFrame({'A':[5.,10.,20.,30.],'B':[10.,10.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 return days,close,opening

def synthetic_outputs():
 days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(245.)
 close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq))for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6)]},index=days)
 opening=close*np.array([.999,1.001,1.0002]);meta=dict(dates=[str(x.date())for x in days[:-1:2]],ends=[str(x.date())for x in days[2::2]],symbols=[['A','B','C']for _ in range(122)],etfs=[])
 orders={name:[]for name in m.NAMES}
 for i,(day,end)in enumerate(zip(meta['dates'],meta['ends'])):
  for name in m.NAMES:
   changed=(name==m.NAMES[2]and i%3==0)or(name==m.NAMES[1]and i%4==0)
   orders[name].append(order(pd.Timestamp(day),pd.Timestamp(end),['A','C']if changed else['A','B']))
 data={};results={};navs={};metrics={};bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 for name in m.NAMES:
  results[name]={}
  for execution,cost,key in m.PATHS:
   met,nav,tr,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
   results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span)for k,span in bounds.items()})
   data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v))for d,v in zip(days,nav)];data[prefix+'_TRADES.json']=tr;data[prefix+'_HOLDINGS.json']=hold
   if key=='same_close':navs[name]=nav;metrics[name]=met
 rows=[dict(name=name,**m.h.measures(days,navs[name],meta))for name in m.NAMES];selection=m.select(rows)
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['STATE_AUDIT.json']=dict(years=[dict(year=2016,cutoff='2015-12-31',lower_bound_exclusive='2010-12-31',fit_attempted=True,status='available',model=dict(fit_attempted=True,status='available'))],signals=[dict(signal_index=i,signal_date=day,year=2016,available=True,reason=None,first_invalid_query_date=None,filtered_probability=.4,memoryless_probability=.6)for i,day in enumerate(meta['dates'])],new_model_fits=1)
 data['FEATURE_AUDIT.json']=dict(original_arrays_unchanged=True,returns_or_nav_computed=False,new_model_fits=1)
 data['VALIDATION.json']=dict(primary=selection['primary'],original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,new_model_fits=1,pristine_oos=False)
 return close,opening,orders,meta,data

class AccountingTests(unittest.TestCase):
 def test_fixed_shares_not_daily_rebalanced(self):
  dates,px,op=small_prices();met,nav,tr,hold=m.replay(px,op,[order(dates[0],dates[-1],['A','B'])],dates)
  np.testing.assert_allclose(nav,[100000,150000,200000,250000]);self.assertEqual(len(tr),2);self.assertEqual(met['fees_paid'],0.);self.assertEqual(hold[0]['fee'],0.)
 def test_next_open_uses_next_session_price(self):
  dates,px,op=small_prices();_,nav,tr,_=m.replay(px,op,[order(dates[0],dates[-1],['A','B'])],dates,'open',0.)
  self.assertEqual(nav[0],100000.);self.assertEqual(nav[1],200000.);self.assertEqual(tr[0]['date'],str(dates[1].date()));self.assertEqual({t['adjusted_price']for t in tr},{10.})
 def test_fee_equation_on_purchase_and_liquidation(self):
  dates,px,op=small_prices();orders=[order(dates[0],dates[2],['B','C']),order(dates[2],dates[-1],[])]
  met,nav,tr,hold=m.replay(px,op,orders,dates,'close',.0005);post=100000/1.0005;final=post*(1-.0005)
  self.assertAlmostEqual(nav[0],post,places=7);self.assertAlmostEqual(nav[-1],final,places=7);self.assertAlmostEqual(met['fees_paid'],100000-final,places=7);self.assertEqual(len(tr),4)
 def test_three_equal_initial_weights_remain_fixed_units(self):
  dates,px,op=small_prices();_,nav,tr,hold=m.replay(px,op,[order(dates[0],dates[-1],['A','B','C'])],dates)
  np.testing.assert_allclose(nav,[100000.,100000.*4/3,100000.*5/3,200000.],rtol=1e-14)
  self.assertEqual(len(tr),3)
 def test_reorder_same_holdings_has_no_spurious_trade(self):
  dates,px,op=small_prices();px[:]=10.;op[:]=10.
  rows=[order(dates[0],dates[2],['A','B','C']),order(dates[2],dates[-1],['C','B','A'])]
  met,nav,tr,hold=m.replay(px,op,rows,dates)
  self.assertEqual(len(tr),3);np.testing.assert_array_equal(nav,np.full(4,100000.))
 def test_cash_remains_cash(self):
  dates,px,op=small_prices();met,nav,tr,_=m.replay(px,op,[order(dates[0],dates[-1],[])],dates)
  np.testing.assert_array_equal(nav,np.full(4,100000.));self.assertEqual(tr,[]);self.assertEqual(met['total_return'],0.)
 def test_invalid_consumed_price_rejected(self):
  dates,px,op=small_prices();px.at[dates[2],'A']=float('nan')
  with self.assertRaises(AssertionError):m.replay(px,op,[order(dates[0],dates[-1],['A','B'])],dates)
 def test_unregistered_execution_cost_rejected(self):
  dates,px,op=small_prices()
  with self.assertRaisesRegex(ValueError,'scenario'):m.replay(px,op,[],dates,'close',.005)
 def test_display_aliases_only_for_comparison(self):
  rows=[dict(symbol='research_equity_FB'),dict(symbols=['BRK.B','A'])];before=deepcopy(rows)
  self.assertEqual(m.normalize(rows),[dict(symbol='META'),dict(symbols=['BRK-B','A'])]);self.assertEqual(rows,before)
 def test_sample_sharpe_and_elapsed_years(self):
  dates=pd.DatetimeIndex(['2016-01-04','2016-01-05','2016-01-06']);nav=np.array([100000,110000,99000])
  out=m.h.stat(dates,nav);self.assertAlmostEqual(out['sharpe'],0.,places=12);self.assertAlmostEqual(out['max_drawdown'],-.1);self.assertEqual(out['years'],2/365.2425)

class OutcomeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
 def test_all_nine_paths(self):
  out=self.check();self.assertTrue(out['passed']);self.assertEqual(out['execution_paths_verified'],9);self.assertEqual(len(out['replay_details']),9)
 def test_nav_tamper_rejected(self):
  self.data['incumbent_local67_same_close_DAILY.json'][4]['equity']+=10
  with self.assertRaises(AssertionError):self.check()
 def test_trade_fee_tamper_rejected(self):
  self.data['incumbent_local67_next_open_cost_TRADES.json'][0]['cost']+=1
  with self.assertRaises(AssertionError):self.check()
 def test_wrong_execution_date_rejected(self):
  self.data['incumbent_local67_next_open_zero_TRADES.json'][0]['date']=self.meta['dates'][0]
  with self.assertRaises(AssertionError):self.check()
 def test_one_security_is_not_invested(self):
  self.orders['incumbent_local67'][0]['symbols']=['A'];self.orders['incumbent_local67'][0]['scores']=[.1]
  with self.assertRaisesRegex(ValueError,'two/three'):m.validate_orders(self.px,self.op,self.orders,self.meta)
 def test_original_identifier_must_match_pool(self):
  self.orders['incumbent_local67'][0]['symbols']=['META','B']
  with self.assertRaisesRegex(ValueError,'security identifiers'):m.validate_orders(self.px,self.op,self.orders,self.meta)
 def candidate(self):
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);return rows
 def test_tie_retains_control(self):self.assertEqual(m.select(self.candidate())['primary'],m.NAMES[0])
 def test_memoryless_control_is_never_promoted(self):
  rows=self.candidate();rows[1]['training']['cagr']=100.;self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_only_candidate_strict_win(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.1;self.assertEqual(m.select(rows)['primary'],m.NAMES[2])
 def test_strict_cagr_tolerance(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.5e-12;self.assertFalse(m.select(rows)['admitted'])
 def test_each_risk_guard_rejects(self):
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2][section][key]-=tol+.001;self.assertFalse(m.select(rows)['admitted'])
 def test_guard_equal_boundaries_are_admitted(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.01
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:rows[2][section][key]=rows[0][section][key]-tol
  self.assertTrue(m.select(rows)['admitted'])
 def test_only_first88_metrics_affect_selection(self):
  rows=self.candidate();rows[2]['later34']={'cagr':999};rows[0]['later34']={'cagr':-999}
  self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_nonfinite_or_missing_retains_baseline(self):
  for value in [float('nan'),float('inf'),None,True,'0.9']:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2]['training']['sharpe']=value;self.assertFalse(m.select(rows)['admitted'])
  rows=self.candidate();del rows[2]['recent']['cagr'];self.assertFalse(m.select(rows)['admitted'])
 def test_attribution_uses_only_changed_sets(self):
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings']['filtered_latent_state']=list(reversed(rows[0]['holdings']['incumbent_local67']))
  metrics={name:self.data['RESULTS.json'][name]['same_close']['metrics']for name in m.NAMES};out=m.attribution(rows,metrics);self.assertNotIn(rows[0]['date'],[x['date']for x in out['comparisons']['filtered_vs_incumbent']['changes']])
 def test_frozen_selection_tamper(self):
  self.data['FROZEN.json']['selection']['primary']='invalid_frozen_choice'
  with self.assertRaises(AssertionError):self.check()

class BindingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.p=self.root/m.STUDY;self.p.mkdir(parents=True)
  self.addCleanup(self.tmp.cleanup)
  def file(relative,value):
   path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
  def put(name,value):
   path=file(m.STUDY+'/'+name,value);return hashlib.sha256(path.read_bytes()).hexdigest()
  self.put=put;self.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  for name in m.CACHE_FILES:put(name,{'synthetic':name})
  original=file('outputs/monthly_local_linear70/local_linear_ORDERS.json',[])
  put('PRETRAIN_ORDERS.json',{name:[]for name in m.NAMES})
  inputs={key:file('input/'+key+'.json',{})for key in('proposal','meta','policy','protected_manifest')}
  inputs.update(prices=file(m.PRICE,{}),original_orders=original,transition=file('outputs/monthly_target70_research/TRANSITION.npz',{}))
  hashes={str(f):self.digest(f)for f in m.source_files()};hashes.update({str(p):self.digest(p)for p in inputs.values()})
  self.plan=dict(round=134,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(passed=True,stage='prepare',**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
  put('INDEPENDENT_PREPARE.json',self.prep);put('TRAIN_TRIALS.json',{});put('WALKFORWARD.json',{})
  self.freeze()
 def freeze(self):self.put('FROZEN.json',dict(hashes={f:self.digest(self.p/f)for f in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
 def check(self):return m.bindings(m.Reader(self.root),self.plan_sha)
 def test_complete_chain(self):self.assertEqual(self.check(),self.p)
 def test_missing_transition_input_rejected(self):
  del self.plan['inputs']['transition'];self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Exact named inputs'):self.check()
 def test_state_audit_is_mandatory_cache_member(self):
  del self.cache['STATE_AUDIT.json'];self.put('CACHE.json',self.cache)
  with self.assertRaisesRegex(ValueError,'Exact CACHE'):self.check()
 def test_wrong_plan_sha(self):
  with self.assertRaisesRegex(ValueError,'registered PLAN'):m.bindings(m.Reader(self.root),'0'*64)
 def test_changed_cache_member(self):
  self.put('SCORES.npz',{'changed':True})
  with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.check()
 def test_failed_prepare_before_replay(self):
  self.prep['passed']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'prepare passed'):self.check()
 def test_frozen_hash_set_must_be_exact(self):
  x=json.loads((self.p/'FROZEN.json').read_bytes());del x['hashes']['TRAIN_TRIALS.json'];self.put('FROZEN.json',x)
  with self.assertRaisesRegex(ValueError,'FROZEN input set'):self.check()
 def test_reader_detects_mid_review_change(self):
  reader=m.Reader(self.root);reader.sha(self.p/'TRAIN_TRIALS.json');self.put('TRAIN_TRIALS.json',{'changed':True})
  with self.assertRaisesRegex(ValueError,'changed during review'):reader.sha(self.p/'TRAIN_TRIALS.json')
 def test_prepare_requires_all_seven_cache_artifacts(self):
  del self.prep['artifact_hashes'][str(self.p/'PRICE_PARITY.json')];self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'artifact coverage'):self.check()
 def test_bool_model_count_rejected(self):
  self.prep['monthly_label_models_fit']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'prepare scope'):self.check()
 def test_outcome_code_must_be_plan_bound(self):
  del self.plan['hashes'][str(m.source_files()[0])];self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Outcome reviewer source'):self.check()
 def test_no_unbound_prepare_extra(self):
  self.prep['artifact_hashes'][str(self.p/'TRAIN_TRIALS.json')]=self.digest(self.p/'TRAIN_TRIALS.json');self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'unbound'):self.check()
 def test_duplicate_json_keys_rejected(self):
  (self.p/'x.json').write_text('{"a":1,"a":2}')
  with self.assertRaisesRegex(ValueError,'Duplicate'):m.Reader(self.root).json(self.p/'x.json')
 def test_baseline_order_change_rejected(self):
  self.put('PRETRAIN_ORDERS.json',{name:[{'tamper':True}]if name==m.NAMES[0]else[]for name in m.NAMES})
  self.cache['PRETRAIN_ORDERS.json']=self.digest(self.p/'PRETRAIN_ORDERS.json');self.put('CACHE.json',self.cache)
  self.prep['orders_sha256']=self.digest(self.p/'PRETRAIN_ORDERS.json');self.prep['cache_sha256']=self.digest(self.p/'CACHE.json')
  self.prep['artifact_hashes']={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.put('INDEPENDENT_PREPARE.json',self.prep);self.freeze()
  with self.assertRaises(AssertionError):self.check()

class MatchedControlTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
 def test_control_cannot_be_promoted(self):
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);rows[1]['training']['cagr']=100.
  self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_three_attribution_comparisons(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{'filtered_vs_incumbent','memoryless_vs_incumbent','filtered_vs_memoryless'})
 def test_attribution_omission_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['filtered_vs_incumbent']['changes'].pop()
  with self.assertRaises(ValueError):self.check()
 def test_attribution_retains_only_empty_guidance_placeholder(self):
  for group in self.data['ATTRIBUTION.json']['comparisons'].values():
   self.assertTrue(all(row['guidance_events']==[]for row in group['changes']))
  self.data['ATTRIBUTION.json']['comparisons']['filtered_vs_incumbent']['changes'][0]['guidance_events']=[{'unexpected':'source'}]
  with self.assertRaises(ValueError):self.check()
 def test_attribution_sign_tamper_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['filtered_vs_incumbent']['changes'][0]['log_contribution']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_monthly_only_financial_comparison_keys(self):
  self.assertEqual(set(self.data['MONTH_COMPARISON.json'][0]),{'date','end','split','holdings','returns'})
 def test_model_count_type_and_sign(self):
  for bad in [False,-1,1.0]:
   self.data['VALIDATION.json']['new_model_fits']=bad
   with self.subTest(bad=bad),self.assertRaises(ValueError):self.check()
 def test_state_model_count_must_match_feature_and_validation(self):
  for key in ['STATE_AUDIT.json','FEATURE_AUDIT.json','VALIDATION.json']:
   old=self.data[key]['new_model_fits'];self.data[key]['new_model_fits']=old+1
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Model fit count'):self.check()
   self.data[key]['new_model_fits']=old
 def test_bool_fit_attempted_rejects_integer(self):
  self.data['STATE_AUDIT.json']['years'][0]['fit_attempted']=1
  with self.assertRaisesRegex(ValueError,'fit count type'):self.check()
 def test_degenerate_fitted_model_still_counts(self):
  state=self.data['STATE_AUDIT.json'];state['years'][0]['status']='unavailable';state['years'][0]['model']['status']='unavailable'
  for row in state['signals']:row.update(available=False,reason='low_occupancy',filtered_probability=None,memoryless_probability=None)
  self.assertEqual(m.model_audit_counts(state,self.data['FEATURE_AUDIT.json'],self.data['VALIDATION.json'],self.meta),1)
 def test_missing_training_without_fit_counts_zero(self):
  state=self.data['STATE_AUDIT.json'];state['years'][0].update(fit_attempted=False,status='unavailable',model=None)
  for row in state['signals']:row.update(available=False,reason='incomplete_training_prices',filtered_probability=None,memoryless_probability=None)
  for obj in [state,self.data['FEATURE_AUDIT.json'],self.data['VALIDATION.json']]:obj['new_model_fits']=0
  self.assertEqual(m.model_audit_counts(state,self.data['FEATURE_AUDIT.json'],self.data['VALIDATION.json'],self.meta),0)
 def test_unavailable_signal_cannot_contain_probability(self):
  self.data['STATE_AUDIT.json']['signals'][0]['available']=False
  with self.assertRaisesRegex(ValueError,'missing probabilities'):self.check()
 def test_nested_fit_count_mismatch_rejected(self):
  self.data['STATE_AUDIT.json']['years'][0]['model']['fit_attempted']=False
  with self.assertRaisesRegex(ValueError,'Nested annual fit'):self.check()
 def test_fold_selection_tamper_rejected(self):
  self.data['WALKFORWARD.json']['folds'][0]['selection']['primary']=m.NAMES[1]
  with self.assertRaises(ValueError):self.check()
 def test_training_replay_prefix_ignores_later_prices(self):
  dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
  first=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[2]][:88],prefix)[1]
  changed=self.px.copy();changed.loc[changed.index>end]*=1000
  other=m.replay(changed,self.op,self.orders[m.NAMES[2]],dates)[1]
  np.testing.assert_array_equal(first,other[:len(first)])

if __name__=='__main__':unittest.main(verbosity=2)
