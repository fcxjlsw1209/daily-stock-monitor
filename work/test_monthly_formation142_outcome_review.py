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
import monthly_formation142_outcome_review as m

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
 data={};results={};navs={};metrics={};bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
 for name in m.NAMES:
  results[name]={}
  for execution,cost,key in m.PATHS:
   met,nav,tr,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
   results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span)for k,span in bounds.items()})
   data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v))for d,v in zip(days,nav)];data[prefix+'_TRADES.json']=tr;data[prefix+'_HOLDINGS.json']=hold
   if key=='same_close':navs[name]=nav;metrics[name]=met
 rows=[dict(name=name,**m.h.measures(days,navs[name],meta))for name in m.NAMES];selection=m.select(rows)
 months=[]
 for i,(day,pool)in enumerate(zip(meta['dates'],meta['symbols'])):
  calm=i%2==0;count=10 if calm else 12
  blocks=[dict(start_price_index=100+21*k,end_price_index=121+21*k,deleted_returns=21)for k in range(count)]
  selectors={}
  for name,arm in zip(m.NAMES[1:],('mean_scenario','worst_scenario')):
   keys=orders[name][i]['symbols'];indices=[pool.index(key)for key in keys]
   selectors[name]=dict(arm=arm,selected_indices=indices,selected_keys=keys,candidate_indices=[0,1,2,3],candidate_keys=pool[:4],scenario_count=count,objective=.2,fallback=False,reason='selected_by_'+arm)
  months.append(dict(signal_index=i,date=day,calm=calm,horizon=252 if calm else 294,ready=True,reason='ready',blocks=blocks,original_finite_mask=[True]*49,selectors=selectors))
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,months)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['FEATURE_AUDIT.json']=dict(original_arrays_unchanged=True,returns_or_nav_computed=False,new_model_fits=0,original_months=122,original_width=49,raw_core_mask_and_rank_verified=True,monthly_label_models_fit=0,months=months)
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
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings']['formation_common_delete_worst']=list(reversed(rows[0]['holdings']['incumbent_local67']))
  metrics={name:self.data['RESULTS.json'][name]['same_close']['metrics']for name in m.NAMES};out=m.attribution(rows,metrics);self.assertNotIn(rows[0]['date'],[x['date']for x in out['comparisons']['worst_vs_incumbent']['changes']])
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
  self.plan=dict(round=142,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.proposal_patch=patch.object(m,'PROPOSAL_SHA',self.digest(inputs['proposal']));self.proposal_patch.start();self.addCleanup(self.proposal_patch.stop)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(passed=True,stage='prepare',orders_checked=366,months_checked=122,candidate_cells_checked=5978,new_model_fits=0,**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
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
 def test_prepare_counts_exact_and_integer(self):
  for key in ('orders_checked','months_checked','candidate_cells_checked','new_model_fits'):
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

class MatchedControlTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
 def test_control_cannot_be_promoted(self):
  rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[2]=dict(deepcopy(rows[0]),name=m.NAMES[2]);rows[1]['training']['cagr']=100.
  self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_three_attribution_comparisons(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{'worst_vs_incumbent','mean_vs_incumbent','worst_vs_mean'})
 def test_attribution_omission_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['worst_vs_incumbent']['changes'].pop()
  with self.assertRaises(ValueError):self.check()
 def test_attribution_retains_only_empty_guidance_placeholder(self):
  for group in self.data['ATTRIBUTION.json']['comparisons'].values():
   self.assertTrue(all(row['guidance_events']==[]for row in group['changes']))
  self.data['ATTRIBUTION.json']['comparisons']['worst_vs_incumbent']['changes'][0]['guidance_events']=[{'unexpected':'source'}]
  with self.assertRaises(ValueError):self.check()
 def test_attribution_sign_tamper_rejected(self):
  self.data['ATTRIBUTION.json']['comparisons']['worst_vs_incumbent']['changes'][0]['log_contribution']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_monthly_only_financial_comparison_keys(self):
  self.assertEqual(set(self.data['MONTH_COMPARISON.json'][0]),{'date','end','split','holdings','returns','formation_scenario'})
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
  self.data['FEATURE_AUDIT.json']['raw_core_mask_and_rank_verified']=False
  with self.assertRaisesRegex(ValueError,'preparation scope'):self.check()
 def test_formation_month_identity_and_population(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);months[0],months[1]=months[1],months[0]
  with self.assertRaisesRegex(ValueError,'identity/order'):m.validate_months(months,self.meta)
  with self.assertRaisesRegex(ValueError,'122-month'):m.validate_months(months[:-1],self.meta)
 def test_audit_count_requires_integer(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);months[0]['selectors'][m.NAMES[1]]['scenario_count']=True
  with self.assertRaisesRegex(ValueError,'scenario count'):m.validate_months(months,self.meta)
 def test_block_partition_tampering(self):
  for field,value in [('start_price_index',101),('end_price_index',122),('deleted_returns',20)]:
   months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);months[0]['blocks'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.validate_months(months,self.meta)
 def test_wrong_calm_horizon_or_mask(self):
  for field,value in [('calm',1),('horizon',294),('original_finite_mask',[True]*48)]:
   months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);months[0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.validate_months(months,self.meta)
 def test_active_selector_keys_match_real_orders(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);item=months[0]['selectors'][m.NAMES[1]]
  item['selected_indices']=[0,1,2];item['selected_keys']=['A','B','C']
  with self.assertRaisesRegex(ValueError,'actual ordered holdings'):m.monthly(self.orders,self.px,self.meta,months)
 def fallback(self,months,i=0,reason='missing_scenarios'):
  row=months[i]
  if reason=='missing_scenarios':row.update(ready=False,reason='incomplete_originally_measurable_formation',blocks=[])
  for name in m.NAMES[1:]:
   item=row['selectors'][name];keys=self.orders[m.NAMES[0]][i]['symbols']
   item.update(selected_indices=[self.meta['symbols'][i].index(k)for k in keys],selected_keys=keys,scenario_count=len(row['blocks']),objective=None,fallback=True,reason=reason)
 def test_all_missing_month_forces_exact_both_orders(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);self.fallback(months)
  for name in m.NAMES[1:]:self.orders[name][0]=deepcopy(self.orders[m.NAMES[0]][0])
  m.monthly(self.orders,self.px,self.meta,months)
  self.orders[m.NAMES[1]][0]['scores'][0]+=.1
  with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,months)
 def test_candidate_history_fallback_requires_invalid_candidate(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);self.fallback(months,reason='candidate_history_unavailable')
  for name in m.NAMES[1:]:self.orders[name][0]=deepcopy(self.orders[m.NAMES[0]][0])
  with self.assertRaisesRegex(ValueError,'invalid candidate'):m.monthly(self.orders,self.px,self.meta,months)
  months[0]['original_finite_mask'][3]=False
  m.monthly(self.orders,self.px,self.meta,months)
 def test_active_invalid_candidate_rejected(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);months[0]['original_finite_mask'][3]=False
  with self.assertRaisesRegex(ValueError,'Active three-security'):m.validate_months(months,self.meta)
 def test_both_arms_share_candidate_set(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);item=months[0]['selectors'][m.NAMES[1]];item['candidate_indices'].append(4);item['candidate_keys'].append('S4')
  with self.assertRaisesRegex(ValueError,'same ordered candidate'):m.validate_months(months,self.meta)
 def test_unchanged_weighted_core_exact_bypass(self):
  months=deepcopy(self.data['FEATURE_AUDIT.json']['months']);self.fallback(months,reason='unchanged_weighted_core')
  for name in m.NAMES[1:]:self.orders[name][0]=deepcopy(self.orders[m.NAMES[0]][0])
  m.monthly(self.orders,self.px,self.meta,months)
  self.orders[m.NAMES[2]][0]['scores'][0]+=.1
  with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,months)
 def arrays(self,feature=None):
  feature=feature or self.data['FEATURE_AUDIT.json'];months=feature['months']
  arrays=dict(formation_rank=np.full((122,12,49),np.nan),formation_count=np.array([len(r['blocks'])for r in months],dtype=np.int64),formation_ready=np.array([r['ready']for r in months],bool),formation_valid=np.array([r['original_finite_mask']for r in months],bool))
  for i,row in enumerate(months):
   if row['ready']:
    mask=arrays['formation_valid'][i];rank=np.zeros(49);rank[mask]=np.arange(1,mask.sum()+1)/mask.sum()-.5
    arrays['formation_rank'][i,:len(row['blocks'])]=rank
  return arrays
 def test_four_arrays_verified(self):self.assertEqual(m.verify_feature_matrices(self.arrays(),self.data['FEATURE_AUDIT.json'],self.meta),4)
 def test_matrix_wrong_shape_or_dtype(self):
  for key in ('formation_rank','formation_count','formation_ready','formation_valid'):
   arrays=self.arrays();arrays[key]=arrays[key][:-1]
   with self.subTest(key=key),self.assertRaises(ValueError):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
  arrays=self.arrays();arrays['formation_count']=arrays['formation_count'].astype(float)
  with self.assertRaises(ValueError):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
 def test_matrix_count_readiness_or_mask_mismatch(self):
  for key,value in [('formation_count',12),('formation_ready',False),('formation_valid',False)]:
   arrays=self.arrays();arrays[key][0]=value
   with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
 def test_unused_scenario_rows_are_nan(self):
  arrays=self.arrays();arrays['formation_rank'][0,10]=0.
  with self.assertRaisesRegex(ValueError,'Unused scenario'):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
 def test_active_scenario_nonfinite_or_outside_rank_range(self):
  for value in (float('nan'),float('inf'),.6):
   arrays=self.arrays();arrays['formation_rank'][0,0,0]=value
   with self.subTest(value=value),self.assertRaisesRegex(ValueError,'bounded active'):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
 def test_finite_mask_neutrality_not_rank_zero_inference(self):
  feature=deepcopy(self.data['FEATURE_AUDIT.json']);feature['months'][0]['original_finite_mask'][-1]=False;arrays=self.arrays(feature)
  self.assertEqual(m.verify_feature_matrices(arrays,feature,self.meta),4)
  arrays['formation_rank'][0,0,-1]=.01
  with self.assertRaisesRegex(ValueError,'unavailable core remains neutral'):m.verify_feature_matrices(arrays,feature,self.meta)
 def test_rank_mass_tampering(self):
  arrays=self.arrays();arrays['formation_rank'][0,0,0]+=.01
  with self.assertRaisesRegex(ValueError,'rank mass'):m.verify_feature_matrices(arrays,self.data['FEATURE_AUDIT.json'],self.meta)
 def test_monthly_scenario_context_tamper_detected(self):
  self.data['MONTH_COMPARISON.json'][0]['formation_scenario']=deepcopy(self.data['MONTH_COMPARISON.json'][0]['formation_scenario']);self.data['MONTH_COMPARISON.json'][0]['formation_scenario']['horizon']=294
  with self.assertRaises(ValueError):self.check()
 def test_attribution_scenario_context_tamper_detected(self):
  row=self.data['ATTRIBUTION.json']['comparisons']['worst_vs_incumbent']['changes'][0];row['formation_scenario']=deepcopy(row['formation_scenario']);row['formation_scenario']['horizon']=294
  with self.assertRaises(ValueError):self.check()

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

def write_prereview():
 import io,sys
 stream=io.StringIO()
 suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(stream.getvalue())
 implementation={
  'work/monthly_formation142_runner.py':'74bfd1f41a5424b4e0c4d465842ff2e49d81bbb0a46a2b37a4a73b2cb76f1724',
  'work/monthly_formation142_features.py':'63d03372bec4370155563c32d222e017b2bd655ac7956234167d0d5c5ca34da2',
  'work/monthly_formation142_selector.py':'e65b56e4b7ee60c15c7e37a3746fdd2356159beedf1db42d6a275a4838fa3f1a'}
 digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 for relative,expected in implementation.items():
  if digest(m.ROOT/relative)!=expected:raise ValueError('Reviewed producer source changed: '+relative)
 if digest(m.ROOT/'work/monthly_formation142_proposal.json')!=m.PROPOSAL_SHA:raise ValueError('Final proposal changed')
 report=dict(round=142,stage='preregistration_outcome_review',proposal_sha256=m.PROPOSAL_SHA,
  synthetic_only=True,real_142_feature_or_return_calculations=0,passed=True,
  source_hashes={str(path):digest(path)for path in m.source_files()},
  implementation_hashes={str(m.ROOT/path):expected for path,expected in implementation.items()},
  tests=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue()),
  independently_reviewed_contracts=[
   'Nine accounting paths reuse the exact frozen independent helper; producer replay, feature, selector and runner modules are not imported.',
   '366 original monthly orders; fixed units, next-session execution, fees, NAV, trades, holdings, annual partial flags, metrics and slices checked.',
   'First88 dual-reference gate and three folds checked independently; later34 cannot influence selection.',
   'All122 monthly scenario records and three attribution comparisons carry exact matching source-month metadata; guidance_events remains an empty placeholder.',
   'Four formation arrays checked against monthly masks/count/readiness plus active rank bounds, neutral invalid slots, average-rank mass and NaN padding.',
   'Latest producer passes original finite-core candidate validity to the selector; missing-candidate fallback is verified and preserves full original orders and scores.',
   'Raw scenario and choice reconstruction is required through the separately hash-bound independent prepare receipt, including all9 checks and exact366/122/5978/0 counts.',
   'Final proposal, original order/price inputs, prepare code/artifacts, frozen selection and diagnostic start marker are hash-bound before replay.'
  ],
  material_remaining_issues=[],
  limits=[
   'Prospective code and synthetic review only: no actual142 features, choices, ranks or returns have been computed by this review.',
   'Actual nine-path verification requires a completed registered evaluation and a valid independent prepare receipt.',
   'Both candidate arms replace the original final episode action; changes versus incumbent are not pure attribution to worst-scenario aggregation.',
   'Existing reused-history, fixed-membership, adjusted-price and execution limitations remain; full historical novelty is not certified.'
  ])
 destination=m.ROOT/'work/monthly_formation142_outcome_prereview.json'
 with destination.open('x')as handle:json.dump(report,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
 print(json.dumps(dict(path=str(destination),sha256=digest(destination),tests_run=result.testsRun,passed=True)))

if __name__=='__main__':
 import sys
 if sys.argv[1:]==['--write-prereview']:write_prereview()
 else:unittest.main(verbosity=2)
