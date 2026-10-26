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
import monthly_peer_earnings139_outcome_review as m

def order(day,end,names):return dict(date=str(day.date()),signal_date=str(day.date()),end=str(end.date()),symbols=list(names),scores=[.2]*len(names))

def small_prices():
 days=pd.bdate_range('2016-01-04',periods=4)
 close=pd.DataFrame({'A':[10.,20.,30.,40.],'B':[20.,20.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 opening=pd.DataFrame({'A':[5.,10.,20.,30.],'B':[10.,10.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 return days,close,opening

def synthetic_cells(meta):
 cells=[];arrays={key:np.full((122,49),np.nan)for key in m.RAW_KEYS+m.HAND_KEYS}
 arrays.update(source_available=np.zeros((122,49),dtype=bool),common_available=np.zeros((122,49),dtype=bool),peer_count=np.zeros((122,49),dtype=np.int64))
 for i,(day,pool)in enumerate(zip(meta['dates'],meta['symbols'])):
  for j,symbol in enumerate(pool):
   valid=j<3;peers=[k for k in range(3)if k!=j]if valid else[]
   raw=(.03,-.01,.02)if valid else(None,None,None)
   row=dict(i=i,j=j,signal_date=day,symbol=symbol,sector_benchmark='XLK'if j<32 else None,issuer_key=symbol if j<32 else None,event_id='event_'+symbol if valid else None,anchor_session=str((pd.Timestamp(day)-pd.offsets.BDay(10)).date())if valid else None,age_sessions=10 if valid else None,source_available=valid,available=valid,reason='available'if valid else'etf'if j>=32 else'missing_event',peer_count=len(peers),peer_symbols=[pool[k]for k in peers],peer_issuer_keys=[pool[k]for k in peers],peer_event_ids=['event_'+pool[k]for k in peers])
   row.update(zip(m.RAW_KEYS,raw));cells.append(row)
   arrays['source_available'][i,j]=valid;arrays['common_available'][i,j]=valid;arrays['peer_count'][i,j]=len(peers)
   for key,hand,value in zip(m.RAW_KEYS,m.HAND_KEYS,raw):
    if value is not None:arrays[key][i,j]=value;arrays[hand][i,j]=.5*np.tanh(value/.05)
 return cells,arrays

def synthetic_outputs():
 days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(245.)
 close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq))for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6)]},index=days)
 opening=close*np.array([.999,1.001,1.0002]);meta=dict(dates=[str(x.date())for x in days[:-1:2]],ends=[str(x.date())for x in days[2::2]],symbols=[['A','B','C']+['S'+str(i)for i in range(3,32)]+['ETF'+str(i)for i in range(17)]for _ in range(122)],etfs=['ETF'+str(i)for i in range(17)])
 orders={name:[]for name in m.NAMES}
 for i,(day,end)in enumerate(zip(meta['dates'],meta['ends'])):
  for name in m.NAMES:
   changed=(name==m.NAMES[3]and i%3==0)or(name==m.NAMES[2]and i%5==0)or(name==m.NAMES[1]and i%4==0)
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
 cells,arrays=synthetic_cells(meta)
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,cells)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['FEATURE_AUDIT.json']=dict(method='fixed_latest_event10_peer_predictive_aggregation_v1',candidate_rows=5978,signal_dates=122,candidates_per_date=49,all_three_masks_equal=True,source_peer_trading_eligibility_used=False,input_provenance={'synthetic':True},new_model_fits=0,cells=cells)
 data['VALIDATION.json']=dict(primary=selection['primary'],original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,new_model_fits=0,pristine_oos=False)
 return close,opening,orders,meta,data,arrays
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
  for name,value in [('PROPOSAL_SHA',self.digest(inputs['proposal'])),('PROTECTED_SHA',self.digest(inputs['protected_manifest']))]:
   replacement=patch.object(m,name,value);replacement.start();self.addCleanup(replacement.stop)
  self.plan=dict(runtime=dict(python=m.sys.version,numpy=np.__version__,pandas=pd.__version__),tolerances=dict(m.TOLERANCES),event_input_hashes={str(inputs['meta']):self.digest(inputs['meta'])},round=139,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[3],output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.HELPER):m.HELPER_SHA},inputs={key:dict(path=str(path),sha256=self.digest(path))for key,path in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
  self.plan_sha=put('PLAN.json',self.plan)
  self.cache={name:self.digest(self.p/name)for name in m.CACHE_FILES};cache_sha=put('CACHE.json',self.cache)
  artifacts={str(self.p/name):self.digest(self.p/name)for name in(*m.CACHE_FILES,'PLAN.json','CACHE.json')}
  self.prep=dict(orders_checked=488,source_cells_checked=5978,latest_known_event_independently_checked=True,original_event10_values_preserved=True,all_signal_prefixes_verified=True,original_sector_map_independently_reproduced=True,calendar_overlay_independently_verified=True,raw_prices_independently_verified=True,raw_event_prices_reconstructed=False,passed=True,stage='prepare',**{k:True for k in m.PREPARE_CHECKS},producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,plan_sha256=self.plan_sha,cache_sha256=cache_sha,orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.HELPER):m.HELPER_SHA},artifact_hashes=artifacts)
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


 def test_new_scope_cannot_be_an_old_prepare_receipt(self):
  del self.prep['orders_checked'];self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'Round139 prepare population'):self.check()
 def test_latest_known_event_check_required(self):
  self.prep['latest_known_event_independently_checked']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
  with self.assertRaisesRegex(ValueError,'Round139 independent source scope'):self.check()
 def test_runtime_changed(self):
  self.plan['runtime']['numpy']='wrong';self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'runtime'):self.check()
 def test_tolerance_changed(self):
  self.plan['tolerances']['scores_atol']=.1;self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'tolerances'):self.check()
 def test_event_input_unbound(self):
  extra=self.root/'extra.json';extra.write_text('{}');self.plan['event_input_hashes'][str(extra)]=self.digest(extra);self.plan_sha=self.put('PLAN.json',self.plan)
  with self.assertRaisesRegex(ValueError,'Source closure'):self.check()
 def test_proposal_digest_locked(self):
  with patch.object(m,'PROPOSAL_SHA','f'*64):
   with self.assertRaisesRegex(ValueError,'Immutable proposal'):self.check()

class SelectionTests(unittest.TestCase):
 def rows(self):return [dict(name=name,training=dict(cagr=.3,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.4))for name in m.NAMES]
 def test_candidate_only_beats_all_three(self):
  rows=self.rows();rows[3]['training']['cagr']=.4;before=deepcopy(rows)
  self.assertEqual(m.select(rows),dict(primary=m.NAMES[3],admitted=True,guards=True,guards_against={name:True for name in m.NAMES[:3]}));self.assertEqual(rows,before)
 def test_third_reference_veto(self):
  rows=self.rows();rows[3]['training']['cagr']=.4;rows[2]['training']['cagr']=.4
  result=m.select(rows);self.assertFalse(result['guards_against'][m.NAMES[2]]);self.assertEqual(result['primary'],m.NAMES[0])
 def test_controls_never_promoted(self):
  for ref in(1,2):
   rows=self.rows();rows[ref]['training']['cagr']=100
   self.assertEqual(m.select(rows)['primary'],m.NAMES[0])
 def test_cagr_strict_boundary_each_reference(self):
  for ref in range(3):
   rows=self.rows();rows[ref]['training']['cagr']=.4;rows[3]['training']['cagr']=.4+1e-12
   self.assertFalse(m.select(rows)['admitted']);rows[3]['training']['cagr']+=2e-12
   self.assertTrue(m.select(rows)['admitted'])
 def test_each_risk_guard_and_boundary(self):
  for ref in range(3):
   for group,key,tolerance in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
    rows=self.rows();rows[3]['training']['cagr']=.4;rows[ref][group][key]+=.1
    rows[3][group][key]=rows[ref][group][key]-tolerance
    self.assertTrue(m.select(rows)['admitted']);rows[3][group][key]-=1e-9
    self.assertFalse(m.select(rows)['admitted'])
 def test_missing_nonfinite_nonnumeric_any_arm(self):
  for arm in range(4):
   for bad in(None,False,'1',float('nan'),float('inf'),float('-inf')):
    rows=self.rows();rows[3]['training']['cagr']=.4;rows[arm]['recent']['cagr']=bad
    self.assertFalse(m.select(rows)['admitted'])
   rows=self.rows();rows[3]['training']['cagr']=.4;del rows[arm]['recent']['cagr'];self.assertFalse(m.select(rows)['admitted'])
 def test_only_four_ordered_arms(self):
  for bad in(self.rows()[:3],self.rows()+[self.rows()[0]],list(reversed(self.rows())),None):
   with self.assertRaises(ValueError):m.select(bad)
 def test_later34_never_rescues_candidate(self):
  rows=self.rows();expected=m.select(rows);rows[3]['later34']={'cagr':999};rows[3]['full']={'cagr':999}
  self.assertEqual(m.select(rows),expected)

class TrainingConventionTests(unittest.TestCase):
 def fixture(self,rate=0.,noise=0.):
  dates=pd.bdate_range('2016-01-04',periods=245)
  meta=dict(dates=[str(x.date())for x in dates[:-1:2]],ends=[str(x.date())for x in dates[2::2]])
  returns=rate+noise*np.where(np.arange(244)%2,1.,-1.)
  nav=100000.*np.r_[1.,np.cumprod(1.+returns)]
  navs={name:np.full(245,100000.)for name in m.NAMES}
  navs[m.NAMES[3]]=nav
  return dates,meta,navs
 def assert_fold_choice(self,dates,meta,navs,wanted):
  folds=m.walkforward(dates,navs,meta)['folds']
  self.assertEqual([(x['n'],x['end'])for x in folds],[(36,60),(60,76),(76,88)])
  self.assertTrue(all(x['selection']['primary']==wanted for x in folds))
 def test_cash_training_and_folds_use_zero_sharpe(self):
  dates,meta,navs=self.fixture();nav=navs[m.NAMES[3]]
  self.assertIsNone(m.h.measures(dates,nav,meta)['training']['sharpe'])
  expected=dict(cagr=0.,sharpe=0.,max_drawdown=0.,total_return=0.)
  self.assertEqual(m.training_measures(dates,nav,meta),{name:expected for name in ('training','early','recent')})
  self.assert_fold_choice(dates,meta,navs,m.NAMES[0])
 def test_constant_growth_training_and_folds_use_zero_sharpe(self):
  dates,meta,navs=self.fixture(rate=.0001);nav=navs[m.NAMES[3]]
  measured=m.training_measures(dates,nav,meta)
  self.assertTrue(all(x['sharpe']==0. for x in measured.values()))
  end=dates.get_loc(pd.Timestamp(meta['ends'][87]));years=(dates[end]-dates[0]).days/365.2425
  self.assertAlmostEqual(measured['training']['cagr'],(nav[end]/nav[0])**(1./years)-1.,places=12)
  self.assertGreater(measured['training']['cagr'],0.)
  self.assert_fold_choice(dates,meta,navs,m.NAMES[3])
 def test_tiny_variance_training_and_folds_use_registered_threshold(self):
  dates,meta,navs=self.fixture(rate=.0001,noise=5e-15);nav=navs[m.NAMES[3]]
  end=dates.get_loc(pd.Timestamp(meta['ends'][87]));rr=nav[1:end+1]/nav[:end]-1.
  self.assertGreater(np.std(rr,ddof=1),0.);self.assertLess(np.std(rr,ddof=1),1e-14)
  self.assertTrue(all(x['sharpe']==0. for x in m.training_measures(dates,nav,meta).values()))
  self.assert_fold_choice(dates,meta,navs,m.NAMES[3])
 def test_training_measures_do_not_consume_later_nav(self):
  dates,meta,navs=self.fixture(rate=.0001,noise=.00002);nav=navs[m.NAMES[3]]
  expected=m.training_measures(dates,nav,meta)
  end=dates.get_loc(pd.Timestamp(meta['ends'][87]));changed=nav.copy();changed[end+1:]=np.nan
  self.assertEqual(m.training_measures(dates,changed,meta),expected)

class OutcomeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.fixture=synthetic_outputs()
 def setUp(self):self.px,self.op,self.orders,self.meta,self.data,self.arrays=deepcopy(self.fixture)
 def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__,self.arrays)
 def test_all_twelve_paths_and_488_orders(self):
  result=self.check();self.assertTrue(result['passed']);self.assertEqual(result['execution_paths_verified'],12);self.assertEqual(result['total_monthly_orders_verified'],488);self.assertEqual(len(result['replay_details']),12)
 def test_each_of_twelve_navs_is_verified(self):
  for arm in m.NAMES:
   for _,_,key in m.PATHS:
    with self.subTest(arm=arm,path=key):
     rows=self.data[arm+'_'+key+'_DAILY.json'];rows[4]['equity']+=10
     with self.assertRaises(AssertionError):self.check()
     rows[4]['equity']-=10
 def test_each_of_twelve_trade_and_holdings_paths_is_verified(self):
  for arm in m.NAMES:
   for _,_,key in m.PATHS:
    for suffix,field in [('TRADES','date'),('HOLDINGS','date')]:
     with self.subTest(arm=arm,path=key,artifact=suffix):
      rows=self.data[arm+'_'+key+'_'+suffix+'.json'];saved=deepcopy(rows[0]);rows[0][field]='1900-01-01'
      with self.assertRaises((AssertionError,ValueError)):self.check()
      rows[0]=saved
 def test_fees_verified(self):
  self.data[m.NAMES[3]+'_next_open_cost_TRADES.json'][0]['cost']+=1
  with self.assertRaises(AssertionError):self.check()
 def test_training_frozen_selection_and_fold_tamper(self):
  for file in('FROZEN.json','WALKFORWARD.json'):
   saved=deepcopy(self.data[file]);target=self.data[file]if file=='FROZEN.json'else self.data[file]['folds'][0]
   target['selection']['primary']=m.NAMES[1]
   with self.assertRaises((AssertionError,ValueError)):self.check()
   self.data[file]=saved
 def test_training_prefix_ignores_later_prices(self):
  dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
  first=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[3]][:88],prefix)[1]
  changed=self.px.copy();changed.loc[changed.index>end]*=1000
  second=m.replay(changed,self.op,self.orders[m.NAMES[3]],dates)[1];np.testing.assert_array_equal(first,second[:len(first)])
 def test_exact_four_arm_schema(self):
  del self.orders[m.NAMES[2]]
  with self.assertRaisesRegex(ValueError,'Four exact'):self.check()
 def test_one_position_rejected(self):
  self.orders[m.NAMES[3]][0]['symbols']=['A'];self.orders[m.NAMES[3]][0]['scores']=[.2]
  with self.assertRaisesRegex(ValueError,'two/three'):self.check()
 def test_three_candidate_comparisons(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{m.NAMES[3]+'_vs_'+name for name in m.NAMES[:3]})
 def test_attribution_cell_and_log_tamper(self):
  key=m.NAMES[3]+'_vs_'+m.NAMES[0];change=self.data['ATTRIBUTION.json']['comparisons'][key]['changes'][0];change['log_contribution']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_monthly_context_and_no_138_schema(self):
  row=self.data['MONTH_COMPARISON.json'][0];self.assertEqual(set(row),{'date','end','split','holdings','returns','peer_earnings'})
  row['peer_earnings']=deepcopy(row['peer_earnings']);row['peer_earnings'][0]['raw_peer_event10']+=.1
  with self.assertRaises(ValueError):self.check()
 def test_zero_and_missing_require_exact_baseline_orders(self):
  for name,key in zip(m.NAMES[1:],m.RAW_KEYS):
   for fill in(None,0.):
    cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);orders=deepcopy(self.orders)
    for row in cells[:49]:
     if row['available']:row[key]=fill if fill is not None else 0.
    with self.assertRaises(AssertionError):m.monthly(orders,self.px,self.meta,cells)
    orders[name][0]=deepcopy(orders[m.NAMES[0]][0]);m.monthly(orders,self.px,self.meta,cells)
 def test_all_missing_common_mask(self):
  cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);orders=deepcopy(self.orders)
  for row in cells[:49]:
   row.update(available=False,source_available=False,reason='etf'if row['symbol']in self.meta['etfs']else'missing_event',peer_count=0,peer_symbols=[],peer_issuer_keys=[],peer_event_ids=[],**{key:None for key in m.RAW_KEYS})
  with self.assertRaises(AssertionError):m.monthly(orders,self.px,self.meta,cells)
  for name in m.NAMES[1:]:orders[name][0]=deepcopy(orders[m.NAMES[0]][0])
  m.monthly(orders,self.px,self.meta,cells)
 def test_all_nine_feature_matrices_verified(self):
  cells=self.data['FEATURE_AUDIT.json']['cells']
  for key in m.EXTRA_KEYS:
   arrays=deepcopy(self.arrays);arrays[key][0,0]=False if arrays[key].dtype==bool else 999
   with self.subTest(key=key),self.assertRaises(ValueError):m.validate_feature_arrays(arrays,cells,self.meta)
 def test_peer_membership_and_common_mask(self):
  for field,value in [('peer_symbols',['B','S3']),('peer_issuer_keys',['B','B']),('peer_count',1),('age_sessions',9),('available',False)]:
   cells=deepcopy(self.data['FEATURE_AUDIT.json']['cells']);cells[0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.validate_cells(cells,self.meta)
 def test_model_fits_must_be_integer_zero(self):
  for file in('FEATURE_AUDIT.json','VALIDATION.json'):
   for value in(False,1,0.):
    self.data[file]['new_model_fits']=value
    with self.assertRaisesRegex(ValueError,'No model fits'):self.check()
   self.data[file]['new_model_fits']=0
 def test_registration_lock_before_any_real_read(self):
  with patch.object(m,'PROPOSAL_SHA','UNREGISTERED_DRAFT'),patch.object(m.Reader,'sha',side_effect=AssertionError('unexpected file read')):
   with self.assertRaisesRegex(ValueError,'UNREGISTERED_DRAFT'):m.audit('/path/never/read','0'*64)

if __name__=='__main__':unittest.main(verbosity=2)
