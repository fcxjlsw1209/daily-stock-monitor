"""Synthetic accounting and tamper tests; no real prices or outcome files read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_profitability141_outcome_review as m

def order(day,end,names):return dict(date=str(day.date()),signal_date=str(day.date()),end=str(end.date()),symbols=list(names),scores=[.2]*len(names))

def small_prices():
 days=pd.bdate_range('2016-01-04',periods=4)
 close=pd.DataFrame({'A':[10.,20.,30.,40.],'B':[20.,20.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 opening=pd.DataFrame({'A':[5.,10.,20.,30.],'B':[10.,10.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 return days,close,opening

def synthetic_outputs():
 days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(245.)
 close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq))for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6)]},index=days)
 opening=close*np.array([.999,1.001,1.0002]);meta=dict(dates=[str(x.date())for x in days[:-1:2]],ends=[str(x.date())for x in days[2::2]],symbols=[['A','B','C']+['S'+str(i)for i in range(3,32)]+['ETF'+str(i)for i in range(17)]for _ in range(122)],etfs=['ETF'+str(i)for i in range(17)])
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
 cells=[]
 for i,(day,pool)in enumerate(zip(meta['dates'],meta['symbols'])):
  for j,symbol in enumerate(pool):
   available=j<2
   cells.append(dict(signal_index=i,pool_index=j,signal_date=day,symbol=symbol,
    available=available,reason='available'if available else'etf_not_applicable'if j>=32 else'no_known_event',
    current_event_id='current'if available else None,prior_event_id='prior'if available else None,
    current_available_from='2016-01-01'if available else None,prior_available_from='2015-01-02'if available else None,
    current_period_end='2015-12-31'if available else None,prior_period_end='2014-12-31'if available else None,
    current_days=365 if available else None,prior_days=365 if available else None,currency_key='issuer_dollars'if available else None,
    gross_profit_growth_sign=.5 if j==0 else -.5 if j==1 else None,
    gross_profitability_improvement_sign=.5 if j==0 else -.5 if j==1 else None))
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,cells)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['FEATURE_AUDIT.json']=dict(original_arrays_unchanged=True,returns_or_nav_computed=False,new_model_fits=0,original_auxiliary_states_and_core_coefficients_preserved=True,original_months=122,original_width=49,common_available_cells=244,latest_event_before_validity_filter=True,sign_method='exact_decimal_cross_products_after_declared_scale_normalization',cells=cells)
 data['VALIDATION.json']=dict(primary=selection['primary'],original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,new_model_fits=0,pristine_oos=False)
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
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);rows[1]=dict(deepcopy(rows[0]),name=m.NAMES[1]);return rows
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
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings']['annual_gross_profitability_improvement_sign']=list(reversed(rows[0]['holdings']['incumbent_local67']))
  metrics={name:self.data['RESULTS.json'][name]['same_close']['metrics']for name in m.NAMES};out=m.attribution(rows,metrics);self.assertNotIn(rows[0]['date'],[x['date']for x in out['comparisons']['profitability_vs_incumbent']['changes']])
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
  inputs={key:file('input/'+key+'.json',{})for key in('proposal','meta','policy','sessions','protected_manifest')}
  inputs.update(prices=file(m.PRICE,{}),original_orders=original)
  hashes={str(f):self.digest(f)for f in m.source_files()};hashes.update({str(p):self.digest(p)for p in inputs.values()})
  inputs['admitted_dataset']=file('input/admitted_dataset.json',dict(schema_version=1,events={}))
  fundamental=file('input/source.html','original issuer synthetic statement')
  source_hashes={str(fundamental):self.digest(fundamental)}
  self.admission=dict(round=141,stage='source_admission',passed=True,semantic_review_passed=True,availability_review_passed=True,identity_review_passed=True,period_review_passed=True,factor_or_return_calculations=0,source_hashes=source_hashes,**{field:self.digest(inputs[key])for field,key in [('dataset_sha256','admitted_dataset'),('meta_sha256','meta'),('sessions_sha256','sessions'),('proposal_sha256','proposal')]})
  inputs['source_admission']=file('input/source_admission.json',self.admission)
  hashes.update({str(p):self.digest(p)for p in inputs.values()});hashes.update(source_hashes)
  from unittest.mock import patch
  patcher=patch.object(m,'PROPOSAL_SHA',self.digest(inputs['proposal']));patcher.start();self.addCleanup(patcher.stop)
  self.plan=dict(fundamental_source_hashes=source_hashes,round=141,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(source_admission_bound=True,common_availability_verified=True,passed=True,stage='prepare',**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
  put('INDEPENDENT_PREPARE.json',self.prep);put('TRAIN_TRIALS.json',{});put('WALKFORWARD.json',{})
  self.freeze()
 def freeze(self):
  self.put('FROZEN.json',dict(hashes={f:self.digest(self.p/f)for f in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
  self.marker=dict(plan_sha256=self.plan_sha,frozen_sha256=self.digest(self.p/'FROZEN.json'),started_utc='2000-01-01T00:00:00+00:00')
  self.put('DIAGNOSE_STARTED.json',self.marker)
 def check(self):return m.bindings(m.Reader(self.root),self.plan_sha)
 def test_complete_chain(self):self.assertEqual(self.check(),self.p)
 def test_diagnostic_marker_is_hashed(self):
  reader=m.Reader(self.root);m.bindings(reader,self.plan_sha)
  self.assertEqual(reader.seen[str(self.p/'DIAGNOSE_STARTED.json')],self.digest(self.p/'DIAGNOSE_STARTED.json'))
 def test_missing_diagnostic_marker_rejected(self):
  (self.p/'DIAGNOSE_STARTED.json').unlink()
  with self.assertRaisesRegex(ValueError,'start marker required'):self.check()
 def test_diagnostic_marker_both_hashes_bound(self):
  for key in('plan_sha256','frozen_sha256'):
   changed=dict(self.marker);changed[key]='0'*64;self.put('DIAGNOSE_STARTED.json',changed)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'marker hash binding'):self.check()
 def test_diagnostic_marker_schema_rejected(self):
  changed=dict(self.marker,extra=True);self.put('DIAGNOSE_STARTED.json',changed)
  with self.assertRaisesRegex(ValueError,'marker fields'):self.check()
 def test_diagnostic_marker_requires_utc_before_audit(self):
  for value in('2000-01-01T00:00:00','2000-01-01T00:00:00+01:00','2999-01-01T00:00:00Z','invalid+00:00',None):
   self.put('DIAGNOSE_STARTED.json',dict(self.marker,started_utc=value))
   with self.subTest(value=value),self.assertRaisesRegex(ValueError,'UTC'):self.check()
 def test_diagnostic_marker_exact_audit_start_rejected(self):
  cutoff=m.datetime(2026,1,1,tzinfo=m.timezone.utc)
  self.put('DIAGNOSE_STARTED.json',dict(self.marker,started_utc=cutoff.isoformat()))
  with self.assertRaisesRegex(ValueError,'precede audit'):m.bindings(m.Reader(self.root),self.plan_sha,cutoff)
 def test_diagnostic_marker_z_utc_accepted(self):
  self.put('DIAGNOSE_STARTED.json',dict(self.marker,started_utc='2000-01-01T00:00:00Z'))
  self.assertEqual(self.check(),self.p)
 def test_missing_sessions_input_rejected(self):
  del self.plan['inputs']['sessions'];self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Exact named inputs'):self.check()
 def test_feature_audit_is_mandatory_cache_member(self):
  del self.cache['FEATURE_AUDIT.json'];self.put('CACHE.json',self.cache)
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
 def test_prepare_requires_all_six_cache_artifacts(self):
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

class AdmissionBindingTests(unittest.TestCase):
 setUp=BindingTests.setUp
 freeze=BindingTests.freeze
 def test_source_receipt_each_review_required(self):
  for key in ('passed','semantic_review_passed','availability_review_passed','identity_review_passed','period_review_passed'):
   receipt=deepcopy(self.admission);receipt[key]=False
   Path(self.plan['inputs']['source_admission']['path']).write_text(json.dumps(receipt))
   with self.subTest(key=key),self.assertRaises(ValueError):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
 def test_source_receipt_input_hashes_required(self):
  for key in ('dataset_sha256','meta_sha256','sessions_sha256','proposal_sha256'):
   receipt=deepcopy(self.admission);receipt[key]='0'*64
   Path(self.plan['inputs']['source_admission']['path']).write_text(json.dumps(receipt))
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Admission input binding'):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
 def test_source_receipt_before_computation(self):
  for value in (1,False):
   receipt=deepcopy(self.admission);receipt['factor_or_return_calculations']=value
   Path(self.plan['inputs']['source_admission']['path']).write_text(json.dumps(receipt))
   with self.subTest(value=value),self.assertRaisesRegex(ValueError,'precede factors'):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
 def test_source_mapping_exact_not_subset(self):
  plan=deepcopy(self.plan);plan['fundamental_source_hashes'][str(m.HELPER)]=m.HELPER_SHA
  with self.assertRaisesRegex(ValueError,'Exact admitted source'):m.verify_admission(m.Reader(self.root),plan,plan['hashes'])

class MatchedControlTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
 def test_control_cannot_be_promoted(self):
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);rows[1]['training']['cagr']=100.
  self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_three_attribution_comparisons(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{'profitability_vs_incumbent','growth_vs_incumbent','profitability_vs_growth'})
 def test_attribution_omission_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['profitability_vs_incumbent']['changes'].pop()
  with self.assertRaises(ValueError):self.check()
 def test_attribution_retains_only_empty_guidance_placeholder(self):
  for group in self.data['ATTRIBUTION.json']['comparisons'].values():
   self.assertTrue(all(row['guidance_events']==[]for row in group['changes']))
  self.data['ATTRIBUTION.json']['comparisons']['profitability_vs_incumbent']['changes'][0]['guidance_events']=[{'unexpected':'source'}]
  with self.assertRaises(ValueError):self.check()
 def test_attribution_sign_tamper_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['profitability_vs_incumbent']['changes'][0]['log_contribution']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_monthly_only_financial_comparison_keys(self):
  self.assertEqual(set(self.data['MONTH_COMPARISON.json'][0]),{'date','end','split','holdings','returns','profitability'})
 def test_model_count_type_and_sign(self):
  for bad in [False,-1,1.0]:
   self.data['VALIDATION.json']['new_model_fits']=bad
   with self.subTest(bad=bad),self.assertRaises(ValueError):self.check()
 def test_state_model_count_must_match_feature_and_validation(self):
  for key in ['FEATURE_AUDIT.json','VALIDATION.json']:
   old=self.data[key]['new_model_fits'];self.data[key]['new_model_fits']=old+1
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'No model fits'):self.check()
   self.data[key]['new_model_fits']=old
 def test_fold_selection_tamper_rejected(self):
  self.data['WALKFORWARD.json']['folds'][0]['selection']['primary']=m.NAMES[1]
  with self.assertRaises(ValueError):self.check()
 def test_training_replay_prefix_ignores_later_prices(self):
  dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
  first=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[2]][:88],prefix)[1]
  changed=self.px.copy();changed.loc[changed.index>end]*=1000
  other=m.replay(changed,self.op,self.orders[m.NAMES[2]],dates)[1]
  np.testing.assert_array_equal(first,other[:len(first)])


 def test_new_feature_core_lineage_claim_required(self):
  self.data['FEATURE_AUDIT.json']['original_auxiliary_states_and_core_coefficients_preserved']=False
  with self.assertRaisesRegex(ValueError,'preparation scope'):self.check()

 def test_all_missing_month_forces_exact_both_orders(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells'])
  for row in cells[:49]:
   row.update(available=False,reason='no_known_event',gross_profit_growth_sign=None,gross_profitability_improvement_sign=None)
  with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,cells)
  for name in m.NAMES[1:]:self.orders[name][0]=deepcopy(self.orders[m.NAMES[0]][0])
  m.monthly(self.orders,self.px,self.meta,cells)
 def test_ranked_zero_per_arm_exact_bypass(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells'])
  for row in cells[:2]:row['gross_profit_growth_sign']=0.
  with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,cells)
  self.orders[m.NAMES[1]][0]=deepcopy(self.orders[m.NAMES[0]][0]);m.monthly(self.orders,self.px,self.meta,cells)
 def test_cell_order_and_population(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0],cells[1]=cells[1],cells[0]
  with self.assertRaisesRegex(ValueError,'identity/order'):m.validate_cells(cells,self.meta)
 def test_common_available_count_tampering(self):
  self.data['FEATURE_AUDIT.json']['common_available_cells']=0
  with self.assertRaisesRegex(ValueError,'available sign count'):self.check()
 def test_feature_matrices_independently_match_cells(self):
  feature=self.data['FEATURE_AUDIT.json'];keys=('gross_profit_growth_sign','gross_profitability_improvement_sign');arrays={k:np.full((122,49),np.nan)for k in keys};arrays['common_available']=np.zeros((122,49),bool)
  for c in feature['cells']:
   if c['available']:
    i,j=c['signal_index'],c['pool_index'];arrays['common_available'][i,j]=True
    for k in keys:arrays[k][i,j]=c[k]
  self.assertEqual(m.verify_feature_matrices(arrays,feature,self.meta),3)
  arrays['gross_profitability_improvement_sign'][0,0]+=.1
  with self.assertRaises(AssertionError):m.verify_feature_matrices(arrays,feature,self.meta)
 def test_out_of_range_sign_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['gross_profit_growth_sign']=.25
  with self.assertRaisesRegex(ValueError,'Common finite'):m.validate_cells(cells,self.meta)
 def test_unmatched_missing_rank_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['gross_profit_growth_sign']=None
  with self.assertRaisesRegex(ValueError,'Common finite'):m.validate_cells(cells,self.meta)
 def test_etf_factor_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['symbol']='ETF0';meta=deepcopy(self.meta);meta['symbols'][0][0]='ETF0'
  with self.assertRaisesRegex(ValueError,'ETF receives'):m.validate_cells(cells,meta)
 def test_monthly_cell_context_tamper_detected(self):
  self.data['MONTH_COMPARISON.json'][0]['profitability']=deepcopy(self.data['MONTH_COMPARISON.json'][0]['profitability']);self.data['MONTH_COMPARISON.json'][0]['profitability'][0]['gross_profit_growth_sign']=.1
  with self.assertRaises(ValueError):self.check()
 def test_annual_contiguity_required(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['current_days']=366
  with self.assertRaisesRegex(ValueError,'Contiguous'):m.validate_cells(cells,self.meta)
 def test_forward_availability_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['current_available_from']='2017-01-01'
  with self.assertRaisesRegex(ValueError,'available by signal'):m.validate_cells(cells,self.meta)
 def test_nonannual_duration_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['prior_days']=329
  with self.assertRaisesRegex(ValueError,'Annual duration'):m.validate_cells(cells,self.meta)
 def test_single_available_stock_allowed_without_reranking(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[1].update(available=False,reason='no_known_event',gross_profit_growth_sign=None,gross_profitability_improvement_sign=None)
  self.assertEqual(len(m.validate_cells(cells,self.meta)),122)
 def test_stale_current_report_rejected(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['current_period_end']='2013-12-31';cells[0]['prior_period_end']='2012-12-31'
  with self.assertRaisesRegex(ValueError,'staleness'):m.validate_cells(cells,self.meta)

class DualReferenceSelectionTests(unittest.TestCase):
 def rows(self):
  return [dict(name=name,training=dict(cagr=.3,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.4))for name in m.NAMES]
 def test_baseline_tie_rejected_even_if_constant_worse(self):
  rows=self.rows();rows[1]['training']['cagr']=.2
  self.assertEqual(m.select(rows),dict(primary=m.NAMES[0],admitted=False,guards=False,guards_against={m.NAMES[0]:False,m.NAMES[1]:True}))
 def test_beta_exact_tie_rejected_after_baseline_strict_win(self):
  rows=self.rows();rows[0]['training']['cagr']=.2
  self.assertFalse(m.select(rows)['admitted'])
 def test_beta_strict_cagr_tolerance(self):
  rows=self.rows();rows[0]['training']['cagr']=.2;rows[2]['training']['cagr']+=.5e-12
  self.assertFalse(m.select(rows)['admitted'])
  rows[2]['training']['cagr']+=2e-12
  self.assertTrue(m.select(rows)['admitted'])
 def test_each_control_risk_guard_independently_required(self):
  for ref in(0,1):
   for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
    rows=self.rows();rows[2]['training']['cagr']=.4;rows[ref][section][key]+=tol+.001
    result=m.select(rows);self.assertFalse(result['admitted']);self.assertFalse(result['guards_against'][m.NAMES[ref]]);self.assertTrue(result['guards_against'][m.NAMES[1-ref]])
 def test_each_reference_nonfinite_unavailable_blocks(self):
  for ref in(0,1,2):
   for value in(None,True,float('nan'),float('inf'),'1.0'):
    rows=self.rows();rows[2]['training']['cagr']=.4;rows[ref]['recent']['cagr']=value
    self.assertFalse(m.select(rows)['admitted'])
 def test_control_never_selected_even_if_sole_winner(self):
  rows=self.rows();rows[1]['training']['cagr']=.9;rows[2]['training']['cagr']=.4
  self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_extra_later_results_cannot_change_admission(self):
  rows=self.rows();rows[2]['training']['cagr']=.4;expected=m.select(rows)
  for row in rows:row['later34']={'cagr':float('nan')};row['full']={'cagr':100.}
  self.assertEqual(m.select(rows),expected)

class SourceInterpretationTests(unittest.TestCase):
 def fixture(self):
  return dict(source_scope=dict(kind='current_retrieval_annual_issuer_release_feed',symbols=['MSFT','NVDA'],first_public_version_verified=False,companyfacts_admitted=False),interpretation_limits=['Historical original pages retrieved later, not original-vintage snapshots.'],identity_continuity_assumption=dict(future_information_used=False,basis='same-event anchor and fixed expiry'))
 def test_frozen_limits_and_scope_copied_to_outcome(self):
  p=self.fixture();r=m.interpretation_report(p)
  self.assertEqual(r['limitations'][-1],p['interpretation_limits'][0]);self.assertEqual(r['source_scope'],p['source_scope'])
  self.assertFalse(r['first_public_version_verified']);self.assertTrue(r['source_admission_is_conditional'])
 def test_copied_scope_does_not_mutate_proposal(self):
  p=self.fixture();before=deepcopy(p);r=m.interpretation_report(p)
  r['source_scope']['symbols'].append('OTHER');r['identity_continuity_assumption']['basis']='changed'
  self.assertEqual(p,before)
 def test_cannot_claim_first_public_version_or_companyfacts(self):
  for key in ('first_public_version_verified','companyfacts_admitted'):
   p=self.fixture();p['source_scope'][key]=True
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'certification'):m.interpretation_report(p)
 def test_missing_limits_or_future_identity_rejected(self):
  p=self.fixture();p['interpretation_limits']=[]
  with self.assertRaisesRegex(ValueError,'limits'):m.interpretation_report(p)
  p=self.fixture();p['identity_continuity_assumption']['future_information_used']=True
  with self.assertRaisesRegex(ValueError,'Causal'):m.interpretation_report(p)

if __name__=='__main__':unittest.main(verbosity=2)
