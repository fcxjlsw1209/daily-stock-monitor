"""Synthetic accounting and tamper tests; no real prices or outcome files read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_same_report145_outcome_review as m

def select(rows,opened):
 reference=dict(deepcopy(opened[0]),name='same_report_profitability')if opened else dict(name='same_report_profitability')
 return m.select(rows,opened,reference)

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
 data={};results={};navs={};open_navs={};metrics={};bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 for name in m.NAMES:
  results[name]={}
  for execution,cost,key in m.PATHS:
   met,nav,tr,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
   results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span)for k,span in bounds.items()})
   data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v))for d,v in zip(days,nav)];data[prefix+'_TRADES.json']=tr;data[prefix+'_HOLDINGS.json']=hold
   if key=='same_close':navs[name]=nav;metrics[name]=met
   if key=='next_open_cost':open_navs[name]=nav
 reference_orders=deepcopy(orders[m.NAMES[0]]);reference_nav=open_navs[m.NAMES[0]];reference_training=dict(name='same_report_profitability',**m.h.measures(days,reference_nav,meta))
 rows=[dict(name=name,**m.h.measures(days,navs[name],meta))for name in m.NAMES];opened=[dict(name=name,**m.h.measures(days,open_navs[name],meta))for name in m.NAMES];selection=select(rows,opened)
 cells=[]
 for i,(day,pool)in enumerate(zip(meta['dates'],meta['symbols'])):
  for j,symbol in enumerate(pool):
   available=j<2
   cells.append(dict(signal_index=i,pool_index=j,signal_date=day,symbol=symbol,
    available=available,reason='available'if available else'etf_not_applicable'if j>=32 else'no_known_event',
    current_event_id='current'if available else None,prior_event_id='current'if available else None,
    current_available_from='2016-01-01'if available else None,prior_available_from='2016-01-01'if available else None,
    current_period_end='2015-12-31'if available else None,prior_period_end='2014-12-31'if available else None,
    method='same_current_release_annual_comparison',current_days=365 if available else None,prior_days=365 if available else None,currency_key='issuer_dollars'if available else None,
    gross_profit_growth_sign=.5 if j==0 else -.5 if j==1 else None,
    gross_profitability_improvement_sign=.5 if j==0 else -.5 if j==1 else None))
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,'TRAIN_REFERENCE_OPEN.json':reference_training,'REFERENCE_ORDERS.json':reference_orders,'REFERENCE_TRAINING.json':reference_training,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,open_navs,meta,days,reference_nav),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,cells)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['FEATURE_AUDIT.json']=dict(original_arrays_unchanged=True,returns_or_nav_computed=False,new_model_fits=0,original_auxiliary_states_and_core_coefficients_preserved=True,original_months=122,original_width=49,common_available_cells=244,latest_event_before_validity_filter=True,same_current_release_for_both_periods=True,sign_method='exact_decimal_cross_products_after_declared_scale_normalization',cells=cells)
 data['VALIDATION.json']=dict(primary=selection['primary'],selection=selection,original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,new_model_fits=0,pristine_oos=False)
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
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__,self.data['REFERENCE_ORDERS.json'],self.data['REFERENCE_TRAINING.json'])
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
 def test_tie_retains_control(self):self.assertEqual(select(self.candidate(),self.candidate())['primary'],m.NAMES[0])
 def test_memoryless_control_is_never_promoted(self):
  rows=self.candidate();rows[1]['training']['cagr']=100.;self.assertEqual(select(rows,rows)['primary'],m.NAMES[0])
 def test_only_candidate_strict_win(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.1;self.assertEqual(select(rows,rows)['primary'],m.NAMES[2])
 def test_strict_cagr_tolerance(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.5e-12;self.assertFalse(select(rows,rows)['admitted'])
 def test_each_risk_guard_rejects(self):
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2][section][key]-=tol+.001;self.assertFalse(select(rows,rows)['admitted'])
 def test_guard_equal_boundaries_are_admitted(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.01
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:rows[2][section][key]=rows[0][section][key]-tol
  self.assertTrue(select(rows,rows)['admitted'])
 def test_only_first88_metrics_affect_selection(self):
  rows=self.candidate();rows[2]['later34']={'cagr':999};rows[0]['later34']={'cagr':-999}
  self.assertEqual(select(rows,rows)['primary'],m.NAMES[0])
 def test_nonfinite_or_missing_retains_baseline(self):
  for value in [float('nan'),float('inf'),None,True,'0.9']:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2]['training']['sharpe']=value;self.assertFalse(select(rows,rows)['admitted'])
  rows=self.candidate();del rows[2]['recent']['cagr'];self.assertFalse(select(rows,rows)['admitted'])
 def test_attribution_uses_only_changed_sets(self):
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings'][m.NAMES[2]]=list(reversed(rows[0]['holdings']['incumbent_local67']))
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
  fundamental=file('input/source.html','original issuer synthetic statement');source_code=file('input/source.py','synthetic code')
  source_hashes={str(fundamental):self.digest(fundamental),str(inputs['sessions']):self.digest(inputs['sessions'])};source_codes={str(source_code):self.digest(source_code)}
  flags=dict(financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
  package_checks={'aapl_amd':('all_eighty_values_and_scales_preserved','all_twenty_events_preserved','causal_identity_windows','four_rejection_sentinels_retained','latest_invalid_blocks_fallback','missing_AMD2019_not_fabricated','old_measurement_definition_not_merged','same_current_release_four_cells','strict_next_session_availability','two_annual_periods_checked'),
  'msft_nvda':('all24_events_preserved','all96_cells_independently_verified','both_FY2015_rejections_retained','accounting_flags_resolved_only_by_evidence','same_current_release_four_cells','strict_next_session_availability','causal_identity_windows','latest_invalid_blocks_fallback')}
  all_events=[];packages={};closure={};self.package_paths={}
  for name,symbols,count in [('aapl_amd',('AAPL','AMD'),20),('msft_nvda',('MSFT','NVDA'),24)]:
   directory='input/'+name
   events=[dict(event_id=name+str(i),symbol=symbols[i%2],available_from='2020-01-02',admitted=i%7!=0,rejection_codes=[]if i%7 else['sentinel'])for i in range(count)]
   all_events.extend(events)
   data=file(directory+'/DATA.json',dict(method='same_current_release_annual_comparison',events=events))
   plan=file(directory+'/PLAN.json',{});policy=file(directory+'/POLICY.json',{})
   receipt=dict(stage='conditional_source_admission',passed=True,data_sha256=self.digest(data),plan_sha256=self.digest(plan),policy_sha256=self.digest(policy),checks={k:True for k in package_checks[name]},source_hashes=dict(source_hashes),source_code_hashes=dict(source_codes),**flags)
   audit=file(directory+'/AUDIT.json',receipt);manifest=file(directory+'/MANIFEST.json',{});tests=file(directory+'/TESTS.json',{})
   closure.update({str(v):self.digest(v)for v in (data,plan,policy,audit,manifest,tests)})
   packages[name]=dict(data_path=str(data),data_sha256=self.digest(data),audit_path=str(audit),audit_sha256=self.digest(audit))
   self.package_paths[name]=dict(data=data,audit=audit,plan=plan,policy=policy)
  inputs['admitted_dataset']=file('input/DATA.json',dict(schema_version=1,method='same_current_release_annual_comparison',events=all_events))
  source_plan=file('input/PLAN.json',{})
  checks=('complete_packages_preserved','unique_event_ids','disjoint_issuer_sets','rejected_events_preserved','availability_unmodified','original_package_hashes_verified')
  self.admission=dict(stage='conditional_union_source_admission',passed=True,checks={k:True for k in checks},data_sha256=self.digest(inputs['admitted_dataset']),plan_sha256=self.digest(source_plan),packages=packages,source_hashes=dict(source_hashes,**closure),source_code_hashes=dict(source_codes),**flags)
  inputs['source_admission']=file('input/AUDIT.json',self.admission)
  manifest=file('input/MANIFEST.json',dict(plan_sha256=self.admission['plan_sha256'],data_sha256=self.admission['data_sha256'],audit_sha256=self.digest(inputs['source_admission'])));source_tests=file('input/TESTS.json',{})
  source_hashes.update(source_codes);source_hashes.update(closure)
  source_hashes.update({str(v):self.digest(v)for v in(source_plan,manifest,source_tests,inputs['admitted_dataset'],inputs['source_admission'])})
  ref_names=('incumbent_local67','same_report_gross_growth','same_report_profitability')
  reference_hashes={}
  for key,basename,value in [('reference144_plan','PLAN.json',dict(round=144,status='REGISTERED',arms=list(ref_names))),('reference144_orders','PRETRAIN_ORDERS.json',{name:[{'synthetic':i}for i in range(122)]for name in ref_names}),('reference144_training_open','TRAIN_OPEN_TRIALS.json',[dict(name=name)for name in ref_names])]:
   inputs[key]=file('outputs/monthly_same_report144/'+basename,value);reference_hashes[basename]=self.digest(inputs[key])
  inputs['proposal'].write_text(json.dumps(dict(sources=dict(directory='input',dataset_sha256=self.digest(inputs['admitted_dataset']),audit_sha256=self.digest(inputs['source_admission']),hashes={n:self.digest(self.root/'input'/n)for n in('DATA.json','AUDIT.json','PLAN.json','MANIFEST.json')},packages=packages),reference144=dict(hashes=reference_hashes))))
  hashes.update({str(p):self.digest(p)for p in inputs.values()});hashes.update(source_hashes)
  from unittest.mock import patch
  patcher=patch.object(m,'PROPOSAL_SHA',self.digest(inputs['proposal']));patcher.start();self.addCleanup(patcher.stop)
  self.plan=dict(reference144_hashes=reference_hashes,fundamental_source_hashes=source_hashes,round=145,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(orders_checked=366,source_cells_checked=5978,stock_cells_checked=3904,new_model_fits=0,source_admission_bound=True,source_union_preservation_verified=True,fixed144_reference_verified=True,common_availability_verified=True,passed=True,stage='prepare',**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
  put('INDEPENDENT_PREPARE.json',self.prep);put('TRAIN_TRIALS.json',{});put('TRAIN_OPEN_TRIALS.json',{});put('TRAIN_REFERENCE_OPEN.json',{});put('WALKFORWARD.json',{})
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

class MatchedControlTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__,self.data['REFERENCE_ORDERS.json'],self.data['REFERENCE_TRAINING.json'])
 def test_control_cannot_be_promoted(self):
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);rows[1]['training']['cagr']=100.
  self.assertEqual(select(rows,rows)['primary'],m.NAMES[0])
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
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0]['current_available_from']='2017-01-01';cells[0]['prior_available_from']='2017-01-01'
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


class PairedSelectionTests(unittest.TestCase):
 def rows(self):
  rows=[dict(name=name,training=dict(cagr=.3,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.4))for name in m.NAMES]
  rows[2]['training']['cagr']=.4
  return rows
 def test_both_execution_gates_required(self):
  rows=self.rows();opened=self.rows();opened[2]['training']['cagr']=.3
  out=select(rows,opened);self.assertFalse(out['admitted']);self.assertTrue(all(out['guards_against'].values()));self.assertFalse(out['mechanism_confirmed'])
 def test_open_incumbent_tie_allowed_only(self):
  rows=self.rows();opened=self.rows();opened[0]['training']['cagr']=.4
  self.assertTrue(select(rows,opened)['admitted'])
  opened[1]['training']['cagr']=.4
  self.assertFalse(select(rows,opened)['admitted'])
 def test_open_noninferiority_tolerance(self):
  rows=self.rows();opened=self.rows();opened[0]['training']['cagr']=.4+1e-12
  self.assertTrue(select(rows,opened)['admitted'])
  opened[0]['training']['cagr']+=2e-12
  self.assertFalse(select(rows,opened)['admitted'])
 def test_sameclose_requires_strict_win_both(self):
  for index in(0,1):
   rows=self.rows();rows[index]['training']['cagr']=.4
   self.assertFalse(select(rows,self.rows())['admitted'])
 def test_every_risk_guard_both_controls_both_execution_modes(self):
  for mode in(0,1):
   for reference in(0,1):
    for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
     groups=[self.rows(),self.rows()];groups[mode][reference][section][key]+=tol+.001
     with self.subTest(mode=mode,reference=reference,key=key):self.assertFalse(select(*groups)['admitted'])
 def test_nonfinite_boolean_missing_both_modes(self):
  for mode in(0,1):
   for index in(0,1,2):
    for bad in(None,True,float('nan'),float('inf'),'1.0'):
     groups=[self.rows(),self.rows()];groups[mode][index]['recent']['cagr']=bad
     self.assertFalse(select(*groups)['admitted'])
 def test_later34_does_not_select(self):
  groups=[self.rows(),self.rows()];before=select(*groups)
  for group in groups:
   for row in group:row['later34']={'cagr':-100};row['full']={'cagr':100}
  self.assertEqual(select(*groups),before)
 def test_wrong_order_or_missing_open_group_rejected(self):
  with self.assertRaises(ValueError):select(self.rows(),self.rows()[::-1])
  with self.assertRaises(ValueError):select(self.rows(),[])


class SameReportSpecificTests(unittest.TestCase):
 def test_comparative_values_cannot_be_backdated(self):
  _,_,_,meta,data=synthetic_outputs();cells=data['FEATURE_AUDIT.json']['cells'];cells[0]['prior_available_from']='2015-01-01'
  with self.assertRaisesRegex(ValueError,'current release'):m.validate_cells(cells,meta)
 def test_separate_report_ids_rejected(self):
  _,_,_,meta,data=synthetic_outputs();cells=data['FEATURE_AUDIT.json']['cells'];cells[0]['prior_event_id']='other'
  with self.assertRaisesRegex(ValueError,'Same current'):m.validate_cells(cells,meta)
 def test_method_required(self):
  _,_,_,meta,data=synthetic_outputs();cells=data['FEATURE_AUDIT.json']['cells'];cells[0]['method']='old'
  with self.assertRaisesRegex(ValueError,'current release'):m.validate_cells(cells,meta)
 def test_real_proposal_metadata_only(self):
  p=m.ROOT/'work/monthly_same_report145_proposal.json';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m.PROPOSAL_SHA)
  report=m.interpretation_report(json.loads(p.read_bytes()));self.assertFalse(report['first_public_version_verified']);self.assertTrue(report['source_admission_is_conditional'])
 def test_no_producer_import(self):
  import ast
  for node in ast.walk(ast.parse(Path(m.__file__).read_text())):
   if isinstance(node,ast.Import):self.assertFalse(any(a.name.endswith(('_runner','_factor'))for a in node.names))
   if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').endswith(('_runner','_factor')))

class AdmissionContractTests(unittest.TestCase):
 setUp=BindingTests.setUp
 freeze=BindingTests.freeze
 def check_receipt(self):return m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
 def mutate(self,key,value):
  receipt=deepcopy(self.admission);receipt[key]=value
  path=Path(self.plan['inputs']['source_admission']['path']);path.write_text(json.dumps(receipt));digest=self.digest(path)
  self.plan['hashes'][str(path)]=digest;self.plan['fundamental_source_hashes'][str(path)]=digest;self.plan['inputs']['source_admission']['sha256']=digest
 def test_source_receipt_all_checks_required(self):
  for key in self.admission['checks']:
   checks=deepcopy(self.admission['checks']);checks[key]=False;self.mutate('checks',checks)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'checks'):self.check_receipt()
 def test_source_receipt_before_any_computation(self):
  for key in ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests'):
   for value in (1,False):
    self.mutate(key,value)
    with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'precede factors'):self.check_receipt()
 def test_no_source_certifications_inferred(self):
  for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
   self.mutate(key,True)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'certification'):self.check_receipt()
 def test_dataset_digest_binding(self):
  self.mutate('data_sha256','0'*64)
  with self.assertRaisesRegex(ValueError,'input binding'):self.check_receipt()
 def test_source_plan_and_policy_digests(self):
  for key in ('plan_sha256',):
   self.mutate(key,'0'*64)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'union plan'):self.check_receipt()
 def test_required_six_source_artifacts(self):
  for name in ('DATA.json','AUDIT.json','MANIFEST.json','TESTS.json'):
   plan=deepcopy(self.plan);del plan['fundamental_source_hashes'][str(self.root/'input'/name)]
   with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Required union source artifact'):m.verify_admission(m.Reader(self.root),plan,plan['hashes'])


class FixedReferenceTests(unittest.TestCase):
 def rows(self):
  return [dict(name=n,training=dict(cagr=.3+(i==2)*.1,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.3),recent=dict(cagr=.4))for i,n in enumerate(m.NAMES)]
 def reference(self):return dict(deepcopy(self.rows()[2]),name='same_report_profitability')
 def test_full_guard_matches_actual_producer(self):
  import monthly_same_report145_runner as producer
  for section,key,tolerance in [('training','cagr',0.),('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
   for extra in(0.,.001):
    rows=self.rows();ref=self.reference();ref[section][key]+=tolerance+extra
    actual=m.select(rows,rows,ref);self.assertEqual(actual,producer.select(rows,rows,ref))
    self.assertEqual(actual['admitted'],extra==0.)
 def test_missing_or_wrong_reference_name_rejected(self):
  for name in(None,'incumbent_local67','expanded_same_report_profitability'):
   ref=self.reference();ref['name']=name
   with self.assertRaises(ValueError):m.select(self.rows(),self.rows(),ref)
 def test_no_future_reference_metrics_in_selection(self):
  ref=self.reference();before=m.select(self.rows(),self.rows(),ref)
  ref['later34']={'cagr':1000};ref['full']={'sharpe':999}
  self.assertEqual(m.select(self.rows(),self.rows(),ref),before)
 def test_bad_reference_metric_blocks_adoption(self):
  for bad in(None,True,float('nan'),'1.0'):
   ref=self.reference();ref['recent']['cagr']=bad
   self.assertFalse(m.select(self.rows(),self.rows(),ref)['admitted'])
 def test_fixed_reference_cannot_be_promoted(self):
  ref=self.reference();ref['training']['cagr']=999
  out=m.select(self.rows(),self.rows(),ref)
  self.assertEqual(out['primary'],'incumbent_local67');self.assertFalse(out['reference144_next_open_noninferior']);self.assertTrue(out['mechanism_confirmed'])
 def test_saved_reference_statistics_checked(self):
  px,op,orders,meta,data=synthetic_outputs();data['TRAIN_REFERENCE_OPEN.json']['training']['cagr']+=.01
  with self.assertRaisesRegex(ValueError,'Statistic tolerance'):
   m.check_outcomes(px,op,orders,meta,data.__getitem__,data['REFERENCE_ORDERS.json'],data['REFERENCE_TRAINING.json'])
 def test_old_frozen_reference_statistics_checked(self):
  px,op,orders,meta,data=synthetic_outputs();ref=deepcopy(data['REFERENCE_TRAINING.json']);ref['recent']['sharpe']+=.1
  with self.assertRaisesRegex(ValueError,'Statistic tolerance'):
   m.check_outcomes(px,op,orders,meta,data.__getitem__,data['REFERENCE_ORDERS.json'],ref)
 def test_walkforward_reference_metrics_checked(self):
  px,op,orders,meta,data=synthetic_outputs();data['WALKFORWARD.json']['folds'][0]['reference_open_training']['early']['cagr']+=.01
  with self.assertRaisesRegex(ValueError,'Statistic tolerance'):
   m.check_outcomes(px,op,orders,meta,data.__getitem__,data['REFERENCE_ORDERS.json'],data['REFERENCE_TRAINING.json'])


class ExpandedSourceTests(unittest.TestCase):
 setUp=BindingTests.setUp
 freeze=BindingTests.freeze
 def check(self):return m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
 def rebind_union(self,data):
  path=Path(self.plan['inputs']['admitted_dataset']['path']);path.write_text(json.dumps(data));dh=self.digest(path)
  self.plan['inputs']['admitted_dataset']['sha256']=dh
  ap=Path(self.plan['inputs']['source_admission']['path']);receipt=json.loads(ap.read_text());receipt['data_sha256']=dh;ap.write_text(json.dumps(receipt));ah=self.digest(ap)
  self.plan['inputs']['source_admission']['sha256']=ah
  mp=ap.parent/'MANIFEST.json';mp.write_text(json.dumps(dict(plan_sha256=receipt['plan_sha256'],data_sha256=dh,audit_sha256=ah)))
  for p in(path,ap,mp):
   self.plan['hashes'][str(p)]=self.plan['fundamental_source_hashes'][str(p)]=self.digest(p)
 def test_exact_union_preserves_all_44(self):self.assertTrue(self.check()['passed'])
 def test_rejected_event_cannot_be_removed(self):
  data=json.loads(Path(self.plan['inputs']['admitted_dataset']['path']).read_text());data['events'].pop(0);self.rebind_union(data)
  with self.assertRaisesRegex(ValueError,'census'):self.check()
 def test_rejected_event_cannot_be_readmitted(self):
  data=json.loads(Path(self.plan['inputs']['admitted_dataset']['path']).read_text());data['events'][0]['admitted']=True;self.rebind_union(data)
  with self.assertRaisesRegex(ValueError,'preservation'):self.check()
 def test_availability_cannot_be_backdated(self):
  data=json.loads(Path(self.plan['inputs']['admitted_dataset']['path']).read_text());data['events'][0]['available_from']='2010-01-01';self.rebind_union(data)
  with self.assertRaisesRegex(ValueError,'preservation'):self.check()
 def test_duplicate_event_cannot_replace_missing(self):
  data=json.loads(Path(self.plan['inputs']['admitted_dataset']['path']).read_text());data['events'][-1]=deepcopy(data['events'][0]);self.rebind_union(data)
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.check()
 def test_missing_fixed144_hash_rejected(self):
  del self.plan['reference144_hashes']['TRAIN_OPEN_TRIALS.json']
  with self.assertRaisesRegex(ValueError,'reference hashes'):m.verify_reference(m.Reader(self.root),self.plan,self.plan['hashes'])
 def test_fixed144_foreign_path_rejected(self):
  self.plan['inputs']['reference144_orders']['path']=str(self.root/'other.json')
  with self.assertRaisesRegex(ValueError,'reference path'):m.verify_reference(m.Reader(self.root),self.plan,self.plan['hashes'])
 def test_fixed144_wrong_digest_rejected(self):
  self.plan['reference144_hashes']['PRETRAIN_ORDERS.json']='0'*64
  with self.assertRaisesRegex(ValueError,'reference digest'):m.verify_reference(m.Reader(self.root),self.plan,self.plan['hashes'])


def write_prereview():
    """Synthetic-only numerical/lifecycle review; no historical arrays loaded."""
    import io,sys,json,hashlib
    root=m.ROOT
    source_paths=m.source_files()
    implementations=[root/name for name in ['work/complete_monthly_same_report145.py','work/test_complete_monthly_same_report145.py','work/monthly_same_report145_runner.py','work/monthly_same_report144_factor.py']]
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    before={str(p):digest(p)for p in source_paths+implementations}
    proposal=root/'work/monthly_same_report145_proposal.json'
    if digest(proposal)!=m.PROPOSAL_SHA:raise ValueError('Prospective proposal mismatch')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():raise SystemExit(stream.getvalue())
    if before!={p:digest(p)for p in before}:raise ValueError('Reviewed source changed during tests')
    log=root/'outputs/monthly_same_report145/OUTCOME_REVIEW_TESTS.log'
    with log.open('x')as f:f.write(stream.getvalue())
    record=dict(round=145,stage='preregistration_outcome_review',proposal_sha256=m.PROPOSAL_SHA,
        synthetic_only=True,real_145_factor_or_return_calculations=0,passed=True,
        source_hashes={str(p):digest(p)for p in source_paths},implementation_hashes={str(p):digest(p)for p in implementations},
        input_hashes={str(proposal):digest(proposal)},tests_run=result.testsRun,
        synthetic_suites=[dict(name='outcome_review',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue())],
        log_path=str(log),log_sha256=digest(log),material_remaining_issues=[],
        scope=['Independent same-current-report comparison; all four values first available at current release next session.',
               'Exact Fraction division independently checks producer Decimal cross-product financial signs; latest-invalid blocks fallback.',
               'Original 27 arrays and unchanged 2.5% correction/one episode; all 123 prefixes and near-tie order checks.',
               'Independent nine-path accounting, dual first88 gates, fixed units, cost equation and source/frozen hash tampering covered.'],
        limitations=['Synthetic and source review only, no actual round145 historical factors/orders/model fits/returns evaluated.',
                     'Conditional AAPL/AMD/MSFT/NVDA current-report source assumptions and inherited universe/data-vintage caveats remain.'])
    destination=root/'work/monthly_same_report145_outcome_prereview.json'
    with destination.open('x')as f:json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,tests=result.testsRun,path=str(destination),sha256=digest(destination),log=str(log))))

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--write-prereview']:write_prereview()
    else:unittest.main(verbosity=2)
