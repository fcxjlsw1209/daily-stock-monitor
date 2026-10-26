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
import monthly_endpoint155_outcome_review as m

def order(day,end,names):return dict(date=str(day.date()),signal_date=str(day.date()),end=str(end.date()),symbols=list(names),scores=[.2]*len(names))

def small_prices():
 days=pd.bdate_range('2016-01-04',periods=4)
 close=pd.DataFrame({'A':[10.,20.,30.,40.],'B':[20.,20.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 opening=pd.DataFrame({'A':[5.,10.,20.,30.],'B':[10.,10.,20.,20.],'C':[10.,10.,10.,10.]},index=days)
 return days,close,opening

def synthetic_outputs():
 days=pd.bdate_range('2016-01-05',periods=245);x=np.arange(245.)
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
 months=[dict(signal_index=i,date=day,shared_query_population=49,available=True,initial_partial_fallback=False,unchanged_return_correction=False,unchanged_loss_correction=False)for i,day in enumerate(meta['dates'])]
 data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,open_navs,meta),'MONTH_COMPARISON.json':m.monthly(orders,close,meta,months)})
 data['ATTRIBUTION.json']=m.attribution(data['MONTH_COMPARISON.json'],metrics)
 data['RANK_AUDIT.json']=dict(months=months)
 data['MODEL_AUDIT.json']=model_fixture(meta)
 data['VALIDATION.json']=dict(primary=selection['primary'],selection=selection,original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,pristine_oos=False)
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
  rows=deepcopy(self.data['MONTH_COMPARISON.json']);rows[0]['holdings']['matched_loss_indicator_ridge']=list(reversed(rows[0]['holdings']['incumbent_local67']))
  metrics={name:self.data['RESULTS.json'][name]['same_close']['metrics']for name in m.NAMES};out=m.attribution(rows,metrics);self.assertNotIn(rows[0]['date'],[x['date']for x in out['comparisons']['loss_vs_incumbent']['changes']])
 def test_frozen_selection_tamper(self):
  self.data['FROZEN.json']['selection']['primary']='invalid_frozen_choice'
  with self.assertRaises(AssertionError):self.check()


def model_fixture(meta):
    years=sorted({pd.Timestamp(d).year for d in meta['dates']})
    annual=[dict(year=y,cutoff=str(y-1)+'-12-31',lookback_exclusive_start=str(y-6)+'-12-31',status='cold_start',training_rows=0,training_months=0,prediction_rows=0,latest_label_end=None,training_indices=[],training_indices_sha256=m.prep.digest(np.array([],dtype=int)),model_fits=0) for y in years]
    return dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,local_model_fits=0,new_pooled_model_fits=0,training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False,annual=annual)

class AdditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_outputs()
    def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
    def test_three_attributions(self):self.assertEqual(set(self.data['ATTRIBUTION.json']['comparisons']),{'loss_vs_incumbent','return_vs_incumbent','loss_vs_return'})
    def test_no_intramonth_trade_insertion(self):
        self.data[m.NAMES[0]+'_same_close_TRADES.json'][0]['date']=str(self.px.index[1].date())
        with self.assertRaises(AssertionError):m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_future_price_no_prefix_impact(self):
        dates=m.validate_orders(self.px,self.op,self.orders,self.meta);end=pd.Timestamp(self.meta['ends'][87]);prefix=dates[dates<=end]
        for execution,cost in [('close',0.),('open',.0005)]:
            expected=m.replay(self.px.loc[:end],self.op.loc[:end],self.orders[m.NAMES[2]][:88],prefix,execution,cost)[1]
            px=self.px.copy();op=self.op.copy();px.loc[px.index>end]*=2;op.loc[op.index>end]*=3
            actual=m.replay(px,op,self.orders[m.NAMES[2]],dates,execution,cost)[1]
            np.testing.assert_array_equal(expected,actual[:len(expected)])
    def test_no_terminal_forced_sale(self):
        days,px,op=small_prices();_,_,trades,_=m.replay(px,op,[order(days[0],days[-1],['A','B','C'])],days)
        self.assertEqual(len(trades),3);self.assertTrue(all(t['side']=='buy' for t in trades))
    def test_partial_flags_tamper(self):
        self.data[m.NAMES[0]+'_same_close_DAILY.json'][-1]['equity']+=100
        with self.assertRaises(AssertionError):m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_training_audit_omission(self):
        self.data['MODEL_AUDIT.json']['annual']=[]
        with self.assertRaises(ValueError):m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_model_fit_count_bool_rejected(self):
        model=self.data['MODEL_AUDIT.json'];model['new_pooled_model_fits']=False
        with self.assertRaises(ValueError):m.model_checks(model,self.meta)
    def test_model_endpoint_maturity_rejected(self):
        model=self.data['MODEL_AUDIT.json'];model['annual'][0]['latest_label_end']='2099-01-31'
        with self.assertRaises(ValueError):m.model_checks(model,self.meta)
    def test_shared_scaler_assertion_required(self):
        model=self.data['MODEL_AUDIT.json'];row=model['annual'][0];row.update(status='fit',training_months=18,prediction_rows=3,model_fits=2,shared_rows_weights_scalers=False);model['new_pooled_model_fits']=2
        with self.assertRaises(ValueError):m.model_checks(model,self.meta)
    def test_mean_return_control_never_selected_alone(self):
        rows=deepcopy(self.data['TRAIN_TRIALS.json']);rows[1]['training']['cagr']=100
        self.assertEqual(m.select(rows,rows)['primary'],m.NAMES[0])
    def test_open_incumbent_tie_is_allowed_only_with_mean_win(self):
        rows=[]
        for name,cagr in zip(m.NAMES,[.5,.4,.6]):rows.append(dict(name=name,training=dict(cagr=cagr,sharpe=1.,max_drawdown=-.2),early=dict(cagr=.2),recent=dict(cagr=.2)))
        opened=deepcopy(rows);opened[2]['training']['cagr']=.5
        self.assertTrue(m.select(rows,opened)['admitted'])
        opened[1]['training']['cagr']=.5;self.assertFalse(m.select(rows,opened)['admitted'])
    def test_missing_signal_context_rejected(self):
        months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0]['shared_query_population']=True
        with self.assertRaises(ValueError):m.validate_months(months,self.meta)
    def test_initial_partial_requires_no_queries(self):
        meta=deepcopy(self.meta);meta['dates'][0]='2016-08-17';months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0].update(date='2016-08-17',initial_partial_fallback=True)
        with self.assertRaises(ValueError):m.validate_months(months,meta)
    def test_zero_hand_exact_order_required(self):
        months=deepcopy(self.data['RANK_AUDIT.json']['months']);months[0]['unchanged_loss_correction']=True
        with self.assertRaises(AssertionError):m.monthly(self.orders,self.px,self.meta,months)
    def arrays(self):
        p=np.tile(np.arange(49),(122,1)).astype(float);mask=np.ones((122,49),bool)
        return dict(m.prep.ranked_arrays(dict(mean_return_prediction=p,loss_prediction=p,common_available=mask),(122,49)),eligible=mask.copy())
    def test_five_arrays(self):
        a=self.arrays();self.assertEqual(m.verify_feature_matrices(a,m.prep.rank_audit(a,self.meta),self.meta),5)
    def test_rank_drift_rejected(self):
        a=self.arrays();audit=m.prep.rank_audit(a,self.meta);a['mean_return_rank'][0,0]+=.1
        with self.assertRaises(AssertionError):m.verify_feature_matrices(a,audit,self.meta)
    def test_nonfinite_prediction_mask_rejected(self):
        a=self.arrays();a['loss_prediction'][0,0]=np.nan
        with self.assertRaises(ValueError):m.verify_feature_matrices(a,m.prep.rank_audit(a,self.meta),self.meta)
    def test_ineligible_query_rejected(self):
        a=self.arrays();a['eligible'][0,0]=False
        with self.assertRaises(ValueError):m.verify_feature_matrices(a,m.prep.rank_audit(a,self.meta),self.meta)
    def test_qqq_or_spy_not_required_for_endpoint_accounting(self):
        self.assertNotIn('SPY',self.px.columns);self.assertTrue(m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)['passed'])

class BindingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.p=self.root/m.STUDY;self.p.mkdir(parents=True)
        self.write=lambda name,obj:(self.p/name).write_text(json.dumps(obj))
        self.sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.plan_sha='a'*64;self.source=Path(__file__).resolve();self.sources={str(self.source):self.sha(self.source)}
        self.plan=dict(independent_outcome_source_hashes=self.sources,independent_prepare_source_hashes=self.sources,inputs={'original_orders':{'path':str(self.p/'original.json')}})
        for file in m.CACHE_FILES:self.write(file,{})
        self.write('PLAN.json',self.plan);self.plan_sha=self.sha(self.p/'PLAN.json')
        self.write('MODEL_AUDIT.json',{'new_pooled_model_fits':0});self.write('PRETRAIN_ORDERS.json',{name:[{}] for name in m.NAMES});self.write('original.json',[{}])
        self.cache={file:self.sha(self.p/file) for file in m.CACHE_FILES};self.write('CACHE.json',self.cache)
        self.artifacts={str(self.p/file):self.sha(self.p/file) for file in (*m.CACHE_FILES,'PLAN.json','CACHE.json')}
        self.report=dict(stage='prepare',passed=True,producer_modules_imported=False,returns_or_nav_computed=False,new_model_fits=0,orders_checked=366,months_checked=122,candidate_cells_checked=5978,training_rows_checked=8820,forecast_arrays_verified=5,prefixes_checked=123,
                         plan_sha256=self.plan_sha,cache_sha256=self.sha(self.p/'CACHE.json'),orders_sha256=self.sha(self.p/'PRETRAIN_ORDERS.json'),source_hashes=self.sources,artifact_hashes=self.artifacts,**{k:True for k in m.PREPARE_CHECKS})
        self.write('INDEPENDENT_PREPARE.json',self.report)
        for file in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json'):self.write(file,{})
        self.freeze()
        patches=[patch.object(m,'source_files',return_value=[self.source]),patch.object(m.prep,'validate_plan',return_value=(self.p,self.plan,self.sources,self.sources,{})),patch.object(m.prep,'verify_arrays'),patch.object(m.Reader,'npz',return_value={})]
        for p in patches:p.start();self.addCleanup(p.stop)
    def freeze(self):
        self.write('FROZEN.json',dict(hashes={file:self.sha(self.p/file) for file in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=self.sha(self.p/'FROZEN.json'),started_utc='2000-01-01T00:00:00+00:00'))
    def check(self):return m.bindings(m.Reader(self.root),self.plan_sha)
    def test_full_binding_after_registered_source_gate(self):self.assertEqual(self.check(),self.p)
    def test_every_prepare_flag_required(self):
        for k in m.PREPARE_CHECKS:
            test=dict(self.report);test[k]=False;self.write('INDEPENDENT_PREPARE.json',test)
            with self.subTest(k=k),self.assertRaises(ValueError):self.check()
    def test_cache_member_drift(self):
        self.write('SCORES.npz',{'changed':1})
        with self.assertRaises(ValueError):self.check()
    def test_prepare_missing_artifact(self):
        report=deepcopy(self.report);del report['artifact_hashes'][str(self.p/'LABELS.npz')];self.write('INDEPENDENT_PREPARE.json',report)
        with self.assertRaises(ValueError):self.check()
    def test_prepare_count_type(self):
        report=dict(self.report,forecast_arrays_verified=True);self.write('INDEPENDENT_PREPARE.json',report)
        with self.assertRaises(ValueError):self.check()
    def test_marker_future_rejected(self):
        self.write('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=self.sha(self.p/'FROZEN.json'),started_utc='2999-01-01T00:00:00+00:00'))
        with self.assertRaises(ValueError):self.check()
    def test_no_unbound_extra_in_prepare(self):
        report=deepcopy(self.report);report['artifact_hashes'][str(self.p/'original.json')]=self.sha(self.p/'original.json');self.write('INDEPENDENT_PREPARE.json',report)
        with self.assertRaises(ValueError):self.check()
    def test_frozen_must_bind_prepare(self):
        frozen=json.loads((self.p/'FROZEN.json').read_text());del frozen['hashes']['INDEPENDENT_PREPARE.json'];self.write('FROZEN.json',frozen)
        with self.assertRaises(ValueError):self.check()

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser();cli.add_argument('--receipt');args=cli.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if args.receipt:
        with Path(args.receipt).open('x') as f:json.dump(m.synthetic_receipt(result),f,indent=2,sort_keys=True);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
