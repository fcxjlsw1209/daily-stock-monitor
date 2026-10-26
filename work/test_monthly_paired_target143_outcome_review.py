"""Synthetic accounting and tamper tests; no real prices or outcome files read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_paired_target143_outcome_review as m

def order(day,end,names):return dict(date=str(day.date()),signal_date=str(day.date()),end=str(end.date()),symbols=list(names),scores=[.2]*len(names))

def small_prices():
 days=pd.bdate_range('2016-01-04',periods=4)
 close=pd.DataFrame({'A':[10.,20.,30.,40.],'B':[20.,20.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 opening=pd.DataFrame({'A':[5.,10.,20.,30.],'B':[10.,10.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 return days,close,opening

def synthetic_outputs():
 days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(245.)
 close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq))for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6),('D',15,.0009,5)]},index=days)
 opening=close*np.array([.999,1.001,1.0002,.9998]);meta=dict(dates=[str(x.date())for x in days[:-1:2]],ends=[str(x.date())for x in days[2::2]],symbols=[['A','B','C','D']+['S'+str(i)for i in range(4,32)]+['ETF'+str(i)for i in range(17)]for _ in range(122)],etfs=['ETF'+str(i)for i in range(17)])
 orders={name:[]for name in m.NAMES}
 for i,(day,end)in enumerate(zip(meta['dates'],meta['ends'])):
  for name in m.NAMES:
   changed=(name==m.NAMES[2]and i%3==0)or(name==m.NAMES[1]and i%4==0)
   orders[name].append(order(pd.Timestamp(day),pd.Timestamp(end),['A','B','D']if changed else['A','B','C']))
 data={};results={};navs={};open_navs={};metrics={};bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 for name in m.NAMES:
  results[name]={}
  for execution,cost,key in m.PATHS:
   met,nav,tr,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
   results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span)for k,span in bounds.items()})
   data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v))for d,v in zip(days,nav)];data[prefix+'_TRADES.json']=tr;data[prefix+'_HOLDINGS.json']=hold
   if key=='same_close':navs[name]=nav;metrics[name]=met
   if key=='next_open_cost':open_navs[name]=nav
 rows=[dict(name=name,**m.h.measures(days,navs[name],meta))for name in m.NAMES]
 opened=[dict(name=name,**m.h.measures(days,open_navs[name],meta))for name in m.NAMES]
 selection=m.select(rows,opened)
 months=[dict(signal_index=i,date=day,original_population=49,paired_ready=True,fallback_to_local_incumbent=False)for i,day in enumerate(meta['dates'])]
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,open_navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,months)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['RANK_AUDIT.json']=dict(months=months)
 data['MODEL_AUDIT.json']=model_fixture(meta)
 data['VALIDATION.json']=dict(primary=selection['primary'],selection=selection,original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,pristine_oos=False)
 return close,opening,orders,meta,data

def model_fixture(meta):
 years=sorted({pd.Timestamp(day).year for day in meta['dates']})
 annual=[dict(year=year,cutoff=str(year-1)+'-12-31',status='cold_start',training_rows=0,training_months=0,prediction_rows=0,latest_available_session=None,latest_calendar_end=None)for year in years]
 return dict(targets=['close','open'],shared_rows_weights_scaler=True,new_pooled_model_fits=0,annual=annual,local=[])


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
 def test_tie_retains_control(self):self.assertEqual(m.select(self.candidate(),self.candidate())['primary'],m.NAMES[0])
 def test_memoryless_control_is_never_promoted(self):
  rows=self.candidate();rows[1]['training']['cagr']=100.;self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])
 def test_only_candidate_strict_win(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.1;self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[2])
 def test_strict_cagr_tolerance(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.5e-12;self.assertFalse(m.select(rows,rows)['admitted'])
 def test_each_risk_guard_rejects(self):
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2][section][key]-=tol+.001;self.assertFalse(m.select(rows,rows)['admitted'])
 def test_guard_equal_boundaries_are_admitted(self):
  rows=self.candidate();rows[2]['training']['cagr']+=.01
  for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:rows[2][section][key]=rows[0][section][key]-tol
  self.assertTrue(m.select(rows,rows)['admitted'])
 def test_only_first88_metrics_affect_selection(self):
  rows=self.candidate();rows[2]['later34']={'cagr':999};rows[0]['later34']={'cagr':-999}
  self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])
 def test_nonfinite_or_missing_retains_baseline(self):
  for value in [float('nan'),float('inf'),None,True,'0.9']:
   rows=self.candidate();rows[2]['training']['cagr']+=.1;rows[2]['training']['sharpe']=value;self.assertFalse(m.select(rows,rows)['admitted'])
  rows=self.candidate();del rows[2]['recent']['cagr'];self.assertFalse(m.select(rows,rows)['admitted'])
 def test_attribution_uses_only_changed_sets(self):
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings']['paired_open_target']=list(reversed(rows[0]['holdings']['incumbent_local67']))
  metrics={name:self.data['RESULTS.json'][name]['same_close']['metrics']for name in m.NAMES};out=m.attribution(rows,metrics);self.assertNotIn(rows[0]['date'],[x['date']for x in out['comparisons']['open_vs_incumbent']['changes']])
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
  put('MODEL_AUDIT.json',dict(new_pooled_model_fits=0))
  original=file('outputs/monthly_local_linear70/local_linear_ORDERS.json',[])
  put('PRETRAIN_ORDERS.json',{name:[]for name in m.NAMES})
  inputs={key:file('input/'+key+'.json',{})for key in('proposal','meta','policy','train_inputs','training_meta','protected_manifest')}
  inputs.update(prices=file(m.PRICE,{}),original_orders=original)
  hashes={str(f):self.digest(f)for f in m.source_files()};hashes.update({str(p):self.digest(p)for p in inputs.values()})
  self.plan=dict(round=143,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.proposal_patch=patch.object(m,'PROPOSAL_SHA',self.digest(inputs['proposal']));self.proposal_patch.start();self.addCleanup(self.proposal_patch.stop)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(passed=True,stage='prepare',orders_checked=366,months_checked=122,candidate_cells_checked=5978,training_rows_checked=8820,forecast_arrays_verified=6,prefixes_checked=123,new_model_fits=0,**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
  put('INDEPENDENT_PREPARE.json',self.prep);put('TRAIN_TRIALS.json',{});put('TRAIN_OPEN_TRIALS.json',{});put('WALKFORWARD.json',{})
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
 def test_missing_training_input_rejected(self):
  del self.plan['inputs']['train_inputs'];self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Exact named inputs'):self.check()
 def test_rank_audit_is_mandatory_cache_member(self):
  del self.cache['RANK_AUDIT.json'];self.put('CACHE.json',self.cache)
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
 def test_prepare_requires_all_cache_artifacts(self):
  del self.prep['artifact_hashes'][str(self.p/'PRICE_PARITY.json')];self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'artifact coverage'):self.check()
 def test_bool_model_count_rejected(self):
  self.prep['new_model_fits']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'prepare model count'):self.check()
 def test_outcome_code_must_be_plan_bound(self):
  del self.plan['hashes'][str(m.source_files()[0])];self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Outcome reviewer source'):self.check()
 def test_no_unbound_prepare_extra(self):
  self.prep['artifact_hashes'][str(self.p/'TRAIN_TRIALS.json')]=self.digest(self.p/'TRAIN_TRIALS.json');self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'unbound'):self.check()
 def test_prepare_counts_exact_and_integer(self):
  for key in ('orders_checked','months_checked','candidate_cells_checked','training_rows_checked','forecast_arrays_verified','prefixes_checked'):
   old=self.prep[key];self.prep[key]=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'prepare counts'):self.check()
   self.prep[key]=old
 def test_each_new_prepare_flag_required(self):
  for key in m.PREPARE_CHECKS:
   self.prep[key]=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'prepare passed'):self.check()
   self.prep[key]=True
 def test_wrong_prospective_proposal_rejected(self):
  with patch.object(m,'PROPOSAL_SHA','0'*64),self.assertRaisesRegex(ValueError,'prospective proposal'):self.check()
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

class PairedAuditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
 def test_three_attribution_comparisons(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{'open_vs_incumbent','close_vs_incumbent','open_vs_close'})
 def test_attribution_omission_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['open_vs_incumbent']['changes'].pop()
  with self.assertRaises(ValueError):self.check()
 def test_attribution_log_contribution_tamper(self):
  self.data['ATTRIBUTION.json']['comparisons']['open_vs_incumbent']['changes'][0]['log_contribution']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_attribution_guidance_is_empty(self):
  row=self.data['ATTRIBUTION.json']['comparisons']['open_vs_incumbent']['changes'][0];row['guidance_events']=[{'unexpected':'source'}]
  with self.assertRaises(ValueError):self.check()
 def test_attribution_exact_paired_context(self):
  row=self.data['ATTRIBUTION.json']['comparisons']['open_vs_incumbent']['changes'][0]
  row['paired_target']=deepcopy(row['paired_target']);row['paired_target']['original_population']=2
  with self.assertRaises(ValueError):self.check()
 def test_monthly_exact_paired_context(self):
  row=self.data['MONTH_COMPARISON.json'][0]
  row['paired_target']=deepcopy(row['paired_target']);row['paired_target']['original_population']=2
  with self.assertRaises(ValueError):self.check()
 def test_open_training_metric_tamper(self):
  self.data['TRAIN_OPEN_TRIALS.json'][0]['training']['cagr']+=.01
  with self.assertRaises(ValueError):self.check()
 def test_open_fold_metric_tamper(self):
  self.data['WALKFORWARD.json']['folds'][0]['open_training'][0]['training']['cagr']+=.01
  with self.assertRaises(ValueError):self.check()
 def test_fold_selection_tamper(self):
  self.data['WALKFORWARD.json']['folds'][0]['selection']['primary']=m.NAMES[1]
  with self.assertRaises(ValueError):self.check()
 def test_model_count_type(self):
  for bad in(False,-1,1.0,2):
   item=deepcopy(self.data['MODEL_AUDIT.json']);item['new_pooled_model_fits']=bad
   with self.subTest(bad=bad),self.assertRaises(ValueError):m.model_checks(item,self.meta)
 def fit_model(self):
  model=deepcopy(self.data['MODEL_AUDIT.json'])
  model['annual'][0].update(status='fit',prediction_rows=1,training_rows=36,training_months=18,latest_available_session='2015-12-31',latest_calendar_end='2015-12-30')
  model.update(new_pooled_model_fits=2,local=[dict(event_row=0,date=self.meta['dates'][0],symbol='A',year=2016,cutoff='2015-12-31',own_rows=0,own_months=0,latest_available_session=None,status='insufficient_own_history',corrections=dict(close=0.,open=0.))])
  return model
 def test_dynamic_two_pooled_fits_not_zero(self):self.assertEqual(m.model_checks(self.fit_model(),self.meta),2)
 def test_latest_training_maturity_after_cutoff_rejected(self):
  model=self.fit_model();model['annual'][0]['latest_available_session']='2016-01-04'
  with self.assertRaisesRegex(ValueError,'maturity'):m.model_checks(model,self.meta)
 def test_same_day_calendar_end_not_mature(self):
  model=self.fit_model();model['annual'][0]['latest_calendar_end']='2015-12-31'
  with self.assertRaisesRegex(ValueError,'follow calendar'):m.model_checks(model,self.meta)
 def test_wrong_local_identity_or_future_local_training(self):
  for key,value in [('symbol','B'),('event_row',True),('cutoff','2016-12-31'),('latest_available_session','2016-01-01')]:
   model=self.fit_model();model['local'][0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.model_checks(model,self.meta)
 def test_missing_local_query_audit(self):
  model=self.fit_model();model['local']=[]
  with self.assertRaisesRegex(ValueError,'query audit count'):m.model_checks(model,self.meta)
 def test_duplicate_local_query_rejected(self):
  model=self.fit_model();model['annual'][0]['prediction_rows']=2;model['local']*=2
  with self.assertRaisesRegex(ValueError,'Unique local'):m.model_checks(model,self.meta)
 def test_nonfinite_local_correction(self):
  for value in [float('nan'),float('inf'),True]:
   model=self.fit_model();model['local'][0]['corrections']['open']=value
   with self.subTest(value=value),self.assertRaises(ValueError):m.model_checks(model,self.meta)
 def test_training_prefix_both_execution_modes_ignore_later_prices(self):
  dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
  for execution,cost in [('close',0.),('open',.0005)]:
   first=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[2]][:88],prefix,execution,cost)[1]
   px=self.px.copy();op=self.op.copy();px.loc[px.index>end]*=1000;op.loc[op.index>end]*=2000
   other=m.replay(px,op,self.orders[m.NAMES[2]],dates,execution,cost)[1]
   np.testing.assert_array_equal(first,other[:len(first)])
 def test_month_schema_identity_population_and_complement(self):
  original=self.data['RANK_AUDIT.json']['months']
  for key,value in [('signal_index',True),('date','wrong'),('original_population',True),('original_population',50),('paired_ready',1),('fallback_to_local_incumbent',True),('unknown',1)]:
   months=deepcopy(original);months[0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.validate_months(months,self.meta)
  with self.assertRaises(ValueError):m.validate_months(original[:-1],self.meta)
 def test_missing_pair_forces_both_exact_incumbent_orders(self):
  months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0].update(paired_ready=False,fallback_to_local_incumbent=True)
  for name in m.NAMES[1:]:self.orders[name][0]=deepcopy(self.orders[m.NAMES[0]][0])
  m.monthly(self.orders,self.px,self.meta,months)
  self.orders[m.NAMES[1]][0]['scores'][0]+=.01
  with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,months)
 def arrays(self):
  arrays={k:np.full((122,49),.1,dtype=float)for k in m.EXTRA[:4]}
  arrays.update({k:np.ones((122,49),bool)for k in ['paired_available','paired_local_available','eligible','forecast_available']})
  return arrays
 def test_six_paired_forecast_arrays(self):self.assertEqual(m.verify_feature_matrices(self.arrays(),self.data['RANK_AUDIT.json'],self.meta),6)
 def test_all_array_shape_dtype_required(self):
  for key in m.EXTRA:
   arrays=self.arrays();arrays[key]=arrays[key][:-1]
   with self.subTest(key=key),self.assertRaises(ValueError):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
  arrays=self.arrays();arrays['paired_available']=arrays['paired_available'].astype(float)
  with self.assertRaises(ValueError):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
 def test_paired_finite_and_availability_match(self):
  for key in m.EXTRA[:4]:
   for value in [float('nan'),float('inf')]:
    arrays=self.arrays();arrays[key][0,0]=value
    with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
 def test_local_availability_subset(self):
  arrays=self.arrays();arrays['paired_available'][0,0]=False
  with self.assertRaisesRegex(ValueError,'subset'):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
 def test_no_own_history_is_exact_pooled(self):
  arrays=self.arrays();arrays['paired_local_available'][0,0]=False
  m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
  arrays['paired_raw_open'][0,0]+=.01
  with self.assertRaises(AssertionError):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
 def test_missing_query_pair_forces_whole_month_fallback(self):
  arrays=self.arrays();arrays['paired_available'][0,0]=False;arrays['paired_local_available'][0,0]=False
  for key in m.EXTRA[:4]:arrays[key][0,0]=np.nan
  with self.assertRaisesRegex(ValueError,'readiness'):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)
  feature=deepcopy(self.data['RANK_AUDIT.json']);feature['months'][0].update(paired_ready=False,fallback_to_local_incumbent=True)
  self.assertEqual(m.verify_feature_matrices(arrays,feature,self.meta),6)
 def test_population_not_reduced_to_available(self):
  arrays=self.arrays();arrays['eligible'][0,0]=False
  with self.assertRaisesRegex(ValueError,'population'):m.verify_feature_matrices(arrays,self.data['RANK_AUDIT.json'],self.meta)

class PairedSelectionTests(unittest.TestCase):
 def rows(self):
  rows=[dict(name=name,training=dict(cagr=.3,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.4))for name in m.NAMES]
  rows[2]['training']['cagr']=.4
  return rows
 def test_both_execution_gates_required(self):
  rows=self.rows();opened=self.rows();opened[2]['training']['cagr']=.3
  out=m.select(rows,opened);self.assertFalse(out['admitted']);self.assertTrue(all(out['guards_against'].values()));self.assertFalse(out['mechanism_confirmed'])
 def test_open_incumbent_tie_allowed_only(self):
  rows=self.rows();opened=self.rows();opened[0]['training']['cagr']=.4
  self.assertTrue(m.select(rows,opened)['admitted'])
  opened[1]['training']['cagr']=.4
  self.assertFalse(m.select(rows,opened)['admitted'])
 def test_open_noninferiority_tolerance(self):
  rows=self.rows();opened=self.rows();opened[0]['training']['cagr']=.4+1e-12
  self.assertTrue(m.select(rows,opened)['admitted'])
  opened[0]['training']['cagr']+=2e-12
  self.assertFalse(m.select(rows,opened)['admitted'])
 def test_sameclose_requires_strict_win_both(self):
  for index in(0,1):
   rows=self.rows();rows[index]['training']['cagr']=.4
   self.assertFalse(m.select(rows,self.rows())['admitted'])
 def test_every_risk_guard_both_controls_both_execution_modes(self):
  for mode in(0,1):
   for reference in(0,1):
    for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
     groups=[self.rows(),self.rows()];groups[mode][reference][section][key]+=tol+.001
     with self.subTest(mode=mode,reference=reference,key=key):self.assertFalse(m.select(*groups)['admitted'])
 def test_nonfinite_boolean_missing_both_modes(self):
  for mode in(0,1):
   for index in(0,1,2):
    for bad in(None,True,float('nan'),float('inf'),'1.0'):
     groups=[self.rows(),self.rows()];groups[mode][index]['recent']['cagr']=bad
     self.assertFalse(m.select(*groups)['admitted'])
 def test_later34_does_not_select(self):
  groups=[self.rows(),self.rows()];before=m.select(*groups)
  for group in groups:
   for row in group:row['later34']={'cagr':-100};row['full']={'cagr':100}
  self.assertEqual(m.select(*groups),before)
 def test_wrong_order_or_missing_open_group_rejected(self):
  with self.assertRaises(ValueError):m.select(self.rows(),self.rows()[::-1])
  with self.assertRaises(ValueError):m.select(self.rows(),[])

 def test_frozen_proposal_limitations_use_actual_schema(self):
  path=m.ROOT/'work/monthly_paired_target143_proposal.json'
  self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),m.PROPOSAL_SHA)
  proposal=json.loads(path.read_bytes());limits=m.proposal_limitations(proposal)
  self.assertIn(proposal['contrast_interpretation'],limits)
  self.assertIn(proposal['budget_implication'],limits)
  self.assertIn(proposal['evaluation']['later'],limits)
  self.assertIn(proposal['novelty_limits']['registry_only_previous_definition'],limits)
  self.assertTrue(proposal['novelty_limits']['bounded_novelty'])
 def test_missing_limitations_fail_closed(self):
  with self.assertRaises(ValueError):m.proposal_limitations({})


def write_prereview():
 """Only pure synthetic tests plus read-only source/proposal metadata."""
 import importlib.util,io,sys
 root=m.ROOT
 sources=[root/'work/monthly_paired_target143_runner.py',root/'work/monthly_paired_target143_model.py',
          root/'work/after142_paired_target_labels.py',root/'work/complete_monthly_paired_target143.py',
          root/'work/test_complete_monthly_paired_target143.py']
 digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 before={str(path):digest(path)for path in [*sources,*m.source_files()]}
 proposal=root/'work/monthly_paired_target143_proposal.json'
 if digest(proposal)!=m.PROPOSAL_SHA:raise ValueError('Final prospective proposal changed')
 spec=importlib.util.spec_from_file_location('paired143_closer_synthetic',root/'work/test_complete_monthly_paired_target143.py')
 closer_tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(closer_tests)
 reports=[]
 for name,module in [('outcome_review',sys.modules[__name__]),('completion',closer_tests)]:
  stream=io.StringIO()
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
  if not result.wasSuccessful():raise SystemExit(stream.getvalue())
  reports.append(dict(name=name,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue()))
 if before!={path:digest(path)for path in before}:raise ValueError('Source changed during synthetic prereview')
 report=dict(round=143,stage='preregistration_outcome_review',proposal_sha256=m.PROPOSAL_SHA,
  synthetic_only=True,real_143_label_fit_or_return_calculations=0,passed=True,
  source_hashes={str(path):digest(path)for path in m.source_files()},
  implementation_hashes={str(path):digest(path)for path in sources},
  input_hashes={str(proposal):digest(proposal)},synthetic_suites=reports,
  review_scope=[
   'Prospective independent source/contract review plus generated-fixture tests only.',
   'Frozen independent scalar accounting helper; no producer/model/label imports in outcome or closer.',
   'Nine execution paths, 366 orders, fixed units, cost equation, NAV/trades/holdings/partial years/slices and both first88 causal prefixes.',
   'Same-close strict win against both references; next-open5bp strict win versus matched close and noninferiority versus incumbent; all risk guards and all3folds independently checked.',
   'Six cached forecast arrays, shared availability, exact no-own-history pooled fallback and whole-month incumbent fallback;122 rank contexts and three attribution contrasts.',
   'Annual model-audit cutoff/identity/count verification; pooled fits dynamically two per fitted year, local corrections separately audited; outcome itself fits zero models.',
   'Exact prepare11 checks,366/122/5978/8820/6/123 counts, registered source/input/cache/frozen/diagnostic marker hash closure.',
   'Formal closure requires all actual stages, independent accounting and all21444 protected members before one atomic142/926 to143/929 update; exclusive completion publication and exact-hash idempotent resume.'
  ],material_remaining_issues=[],
  limitations=m.proposal_limitations(json.loads(proposal.read_bytes()))+[
   'This certificate verifies code and synthetic lifecycle behavior only; no actual143 labels, model fits, ranks, holdings or returns have been evaluated.',
   'Actual efficacy and completion remain contingent on all registered offline phases and independent receipts passing.'
  ])
 destination=root/'work/monthly_paired_target143_outcome_prereview.json'
 with destination.open('x')as handle:json.dump(report,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
 print(json.dumps(dict(path=str(destination),sha256=digest(destination),passed=True,tests={r['name']:r['tests_run']for r in reports})))

if __name__=='__main__':
 import sys
 if sys.argv[1:]==['--write-prereview']:write_prereview()
 else:unittest.main(verbosity=2)
