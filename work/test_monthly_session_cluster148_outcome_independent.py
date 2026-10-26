"""Generated-price accounting and tamper tests; never open real market outputs."""
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import hashlib
import io
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import monthly_session_cluster148_outcome_independent as m


def order(day,end,names):
    return dict(date=str(pd.Timestamp(day).date()),signal_date=str(pd.Timestamp(day).date()),end=str(pd.Timestamp(end).date()),symbols=list(names),scores=[.2]*len(names))


class SelectionTests(unittest.TestCase):
    def groups(self):
        rows=[dict(name=n,training=dict(cagr=.5,sharpe=2.,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=.6)) for n in m.NAMES]
        return rows,deepcopy(rows)
    def winner(self):
        rows,op=self.groups();rows[2]['training']['cagr']=.6;op[2]['training']['cagr']=.6
        return rows,op
    def test_tie_retains_incumbent(self):self.assertEqual(m.select(*self.groups())['primary'],m.NAMES[0])
    def test_primary_only_can_be_promoted(self):
        rows,op=self.groups();rows[1]['training']['cagr']=9;op[1]['training']['cagr']=9
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_strict_win_all_controls_both_paths(self):self.assertTrue(m.select(*self.winner())['admitted'])
    def test_open_incumbent_noninferiority_allowed(self):
        rows,op=self.winner();op[0]['training']['cagr']=.6
        self.assertTrue(m.select(rows,op)['admitted'])
    def test_open_control_ties_rejected_individually(self):
        for i in (1,):
            rows,op=self.winner();op[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_closed_all_ties_rejected_individually(self):
        for i in range(2):
            rows,op=self.winner();rows[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_each_guard_on_each_path_and_reference(self):
        for which in (0,1):
            for i in range(2):
                for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
                    pair=self.winner();pair[which][i][section][key]=pair[which][2][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)['admitted'])
    def test_boundaries_inclusive(self):
        rows,op=self.winner()
        for group in (rows,op):
            for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:group[2][section][key]=group[0][section][key]-tol
        self.assertTrue(m.select(rows,op)['admitted'])
    def test_tiny_strict_advantage_rejected(self):
        rows,op=self.winner();rows[2]['training']['cagr']=.5+5e-13
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_nonfinite_boolean_or_missing_fails(self):
        for value in (None,True,'2',float('nan'),float('inf')):
            rows,op=self.winner();op[2]['training']['sharpe']=value
            self.assertFalse(m.select(rows,op)['admitted'])
        rows,op=self.winner();del rows[2]['early']['cagr'];self.assertFalse(m.select(rows,op)['admitted'])
    def test_later34_irrelevant(self):
        rows,op=self.groups();rows[2]['later34']={'cagr':1000}
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_exact_arm_order_required(self):
        rows,op=self.winner();rows[1],rows[2]=rows[2],rows[1]
        with self.assertRaises(ValueError):m.select(rows,op)


class AccountingTests(unittest.TestCase):
    def small(self):
        d=pd.bdate_range('2016-01-04',periods=4)
        p=pd.DataFrame(dict(A=[10.,20.,30.,40.],B=[20.,20.,20.,20.],C=[10.,10.,10.,10.]),index=d)
        o=p.copy();o.iloc[1]=[10.,10.,10.]
        return d,p,o
    def test_fixed_shares(self):
        d,p,o=self.small();_,nav,trades,_=m.replay(p,o,[order(d[0],d[-1],['A','B'])],d)
        np.testing.assert_allclose(nav,[100000,150000,200000,250000]);self.assertEqual(len(trades),2)
    def test_next_open(self):
        d,p,o=self.small();_,nav,trades,_=m.replay(p,o,[order(d[0],d[-1],['A','B'])],d,'open',0)
        self.assertEqual(nav[0],100000);self.assertEqual(nav[1],200000);self.assertEqual(trades[0]['date'],str(d[1].date()))
    def test_fees_entry_exit(self):
        d,p,o=self.small();met,nav,_,_=m.replay(p,o,[order(d[0],d[2],['B','C']),order(d[2],d[-1],[])],d,'close',.0005)
        final=100000/1.0005*(1-.0005)
        self.assertAlmostEqual(nav[-1],final);self.assertAlmostEqual(met['fees_paid'],100000-final)
    def test_cash_and_bad_cost(self):
        d,p,o=self.small();_,nav,tr,_=m.replay(p,o,[order(d[0],d[-1],[])],d)
        np.testing.assert_array_equal(nav,np.full(4,100000.));self.assertEqual(tr,[])
        with self.assertRaises(ValueError):m.replay(p,o,[],d,'close',.01)



def synthetic_outputs():
    c=m.contract
    days=pd.bdate_range('2015-01-02',periods=497);x=np.arange(len(days))
    names=['S'+str(i) for i in range(32)]+['ETF'+str(i) for i in range(17)]
    close=pd.DataFrame({s:(10+j)*np.exp((.0003+.00001*j)*x+.007*np.sin(x/(4+j%7))) for j,s in enumerate(names)},index=days)
    opening=close.copy()
    for j,s in enumerate(names):opening[s]*=np.where(x%(3+j%9)==0,1.01,.999)
    meta=dict(dates=[str(d.date()) for d in days[252:-1:2]],ends=[str(d.date()) for d in days[254::2]],
        symbols=[names[:] for _ in range(122)],etfs=names[32:])
    orders={name:[order(day,end,['S0','S2'] if k and i%(k+2)==0 else ['S0','S1']) for i,(day,end) in enumerate(zip(meta['dates'],meta['ends']))] for k,name in enumerate(m.NAMES)}
    data={};results={};navs={};opens={};run=days[252:]
    bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),
        original2021_2026=(meta['dates'][53],meta['ends'][-1]),
        training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
    for name in m.NAMES:
        results[name]={}
        for execution,cost,key in m.PATHS:
            met,nav,trades,hold=m.replay(close,opening,orders[name],run,execution,cost);prefix=name+'_'+key
            results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(run,nav,*span) for k,span in bounds.items()})
            data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v)) for d,v in zip(run,nav)]
            data[prefix+'_TRADES.json']=trades;data[prefix+'_HOLDINGS.json']=hold
            if key=='same_close':navs[name]=nav
            if key=='next_open_cost':opens[name]=nav
    rows=[dict(name=name,**m.h.measures(run,navs[name],meta)) for name in m.NAMES]
    opened=[dict(name=name,**m.h.measures(run,opens[name],meta)) for name in m.NAMES]
    selection=m.select(rows,opened);r=c.features_oracle(meta,days,close,opening)
    feature=dict(method=c.METHOD,window_events=210,adjacent_pairs=209,close_rows=211,
        opening_rows=210,skip_sessions=42,width=49,signal_count=122,sign_zero_tolerance=1e-12,
        rank_definition='0.5-average_tied_zero_based_rank/(n_valid-1); n_valid<=1 gives0',
        rank_direction='lower_frequency_or_covariance_is_higher',raw_missing='NaN',rank_missing=0,
        common_mask=True,constant_event_sequence_valid=True,covariance_uses_two209_pair_marginals=True,
        covariance_arithmetic='integer_joint_count_times209_minus_left_times_right_over209_squared',
        caller_verified_complete_session_grid_required=True,caller_verified_common_adjusted_price_basis_required=True,
        etfs_allowed=True,no_trading_eligibility_changes=True,dose_not_applied=True,
        new_model_fits=0,portfolio_returns_or_nav_computed=False,reasons=dict(Counter(v['reason'] for v in r['cells'])),
        cells=r['cells'],months=r['months'],feature_contract=c.feature_contract(),
        price_cache_sha256=c.PRICE_SHA,sessions_sha256=c.SESSIONS_SHA,calendar_plan_sha256=c.CALENDAR_SHA)
    data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,
        'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(run,navs,opens,meta),
        'FEATURE_AUDIT.json':feature,'VALIDATION.json':dict(selection=selection,new_model_fits=0,
        no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,pristine_oos=False)})
    return close,opening,orders,meta,data,r['arrays']


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_outputs()
    def setUp(self):self.px,self.op,self.orders,self.meta,self.data,self.arrays=deepcopy(self.fixture)
    def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_nine_paths_and366_orders(self):
        r=self.check();self.assertTrue(r['passed']);self.assertEqual(r['execution_paths_verified'],9)
        self.assertEqual(r['total_monthly_orders_verified'],366);self.assertEqual(len(r['independent_monthly_comparison']),122)
    def test_nav_tamper(self):
        self.data[m.NAMES[2]+'_next_open_cost_DAILY.json'][4]['equity']+=10
        with self.assertRaises(AssertionError):self.check()
    def test_trade_cost_tamper(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost']+=1
        with self.assertRaises(AssertionError):self.check()
    def test_holdings_tamper(self):
        self.data[m.NAMES[1]+'_same_close_HOLDINGS.json'][0]['symbols']=[]
        with self.assertRaises((AssertionError,ValueError)):self.check()
    def test_wrong_trade_date(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date']=self.meta['dates'][0]
        with self.assertRaises(AssertionError):self.check()
    def test_single_name_rejected(self):
        self.orders[m.NAMES[0]][0].update(symbols=['S0'],scores=[.2])
        with self.assertRaises(ValueError):m.validate_orders(self.px,self.op,self.orders,self.meta)
    def test_frozen_choice_tamper(self):
        self.data['FROZEN.json']['selection']['primary']='forged'
        with self.assertRaises(AssertionError):self.check()
    def test_training_measure_tamper(self):
        self.data['TRAIN_OPEN_TRIALS.json'][2]['training']['sharpe']+=.01
        with self.assertRaises(ValueError):self.check()
    def test_fold_choice_tamper(self):
        self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']=not self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']
        with self.assertRaises(ValueError):self.check()
    def test_all_five_matrices(self):
        self.assertEqual(m.verify_feature_matrices(self.arrays,self.data['FEATURE_AUDIT.json'],self.meta),5)
        self.arrays['cluster_rank'][0,0]+=.001
        with self.assertRaises(AssertionError):m.verify_feature_matrices(self.arrays,self.data['FEATURE_AUDIT.json'],self.meta)
    def test_counts_cannot_be_forged(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['covariance_numerator']+=1
        with self.assertRaises(ValueError):m.feature_checks(self.data['FEATURE_AUDIT.json'],self.data['VALIDATION.json'],self.meta)
    def test_mask_cannot_be_forged(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['available']=False
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_etf_allowed(self):
        groups=m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
        self.assertTrue(groups[0][32]['available'])
    def test_new_fit_rejected(self):
        self.data['FEATURE_AUDIT.json']['new_model_fits']=1
        with self.assertRaises(ValueError):m.feature_checks(self.data['FEATURE_AUDIT.json'],self.data['VALIDATION.json'],self.meta)
    def test_no_new_producer_imported(self):
        self.assertNotIn('monthly_session_cluster148_factor',sys.modules)
        self.assertNotIn('monthly_session_cluster148_runner',sys.modules)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.p=self.root/m.STUDY;self.p.mkdir(parents=True);c=m.contract
        self.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        def file(name,value):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
        self.file=file;self.put=lambda name,value:file(m.STUDY+'/'+name,value)
        for name in m.CACHE_FILES:self.put(name,{'synthetic':name})
        self.put('PRETRAIN_ORDERS.json',{name:[] for name in m.NAMES})
        inputs={k:file('input/'+k+'.json',{}) for k in ('proposal','meta','policy','protected_manifest')}
        inputs['prices']=file(c.PRICE,{})
        inputs['sessions']=file(c.SESSIONS,{'sessions':[]})
        inputs['original_orders']=file('outputs/monthly_local_linear70/local_linear_ORDERS.json',[])
        concept=file(c.CONCEPT,{'synthetic':True});calendar=file(c.CALENDAR,{})
        predecessor=file('input/predecessor.json',{'passed':True})
        self.inputs=inputs;self.concept=concept
        for key,path in (('CONCEPT_SHA',concept),('PRICE_SHA',inputs['prices']),('SESSIONS_SHA',inputs['sessions']),('CALENDAR_SHA',calendar)):
            p=patch.object(c,key,self.digest(path));p.start();self.addCleanup(p.stop)
        ref=lambda p:dict(path=str(p),sha256=self.digest(p))
        proposal=dict(round=148,arms=list(m.NAMES),primary_challenger=m.NAMES[2],dose=.01,
            concept=dict(path=c.CONCEPT,sha256=c.CONCEPT_SHA),selection_contract=c.proposal_contract(),
            feature_contract=c.feature_contract(),actual_new_factors=0,actual_new_orders=0,actual_new_returns=0,
            fixed=dict(signal_months=122,training_months=88,width=49,original_arrays=27,
                       new_downloads=0,new_model_fits=0,observer_changes=0,period=['2016-08-17','2026-09-24']),
            sources=dict(prices=dict(path=c.PRICE,sha256=c.PRICE_SHA),
                sessions=dict(path=c.SESSIONS,sha256=c.SESSIONS_SHA),calendar=dict(path=c.CALENDAR,sha256=c.CALENDAR_SHA),
                predecessor=ref(predecessor)))
        inputs['proposal'].write_text(json.dumps(proposal))
        hashes={str(p):self.digest(p) for p in [*m.source_files(),*inputs.values(),concept,calendar,predecessor]}
        self.plan=dict(round=148,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[2],
            dose=.01,runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),output_directory=str(self.p),
            hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},
            independent_prepare_source_hashes={str(m.old.HELPER):m.old.HELPER_SHA},inputs={k:ref(v) for k,v in inputs.items()},
            actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
        self.plan_sha=self.digest(self.put('PLAN.json',self.plan))
        self.put('CACHE.json',{name:self.digest(self.p/name) for name in m.CACHE_FILES})
        artifacts={str(self.p/name):self.digest(self.p/name) for name in (*m.CACHE_FILES,'PLAN.json','CACHE.json')}
        flags={key:True for key in (*m.PREPARE_CHECKS,'raw_prices_independently_verified',
            'calendar_overlay_independently_verified','all_signal_prefixes_verified',
            'session_event_arithmetic_verified','common_valid49_ranking_before_eligibility_verified')}
        self.prep=dict(passed=True,stage='prepare',**flags,orders_checked=366,source_cells_checked=5978,
            stock_cells_checked=3904,new_model_fits=0,monthly_label_models_fit=0,producer_modules_imported=False,
            returns_or_nav_computed=False,plan_sha256=self.plan_sha,cache_sha256=self.digest(self.p/'CACHE.json'),
            orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),
            source_hashes={str(m.old.HELPER):m.old.HELPER_SHA},artifact_hashes=artifacts)
        self.put('INDEPENDENT_PREPARE.json',self.prep)
        for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json'):self.put(name,{})
        self.put('FROZEN.json',dict(hashes={n:self.digest(self.p/n) for n in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.put('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2000-01-01T00:00:00+00:00'))
    def check(self):return m.bindings(m.Reader(self.root),self.plan_sha)
    def replace_proposal(self,mutate):
        path=self.inputs['proposal'];value=json.loads(path.read_text());mutate(value);path.write_text(json.dumps(value))
        self.plan['inputs']['proposal']['sha256']=self.digest(path);self.plan['hashes'][str(path)]=self.digest(path)
    def proposal(self):return m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_complete_chain(self):self.assertEqual(self.check(),self.p)
    def test_bad_plan_sha(self):
        with self.assertRaises(ValueError):m.bindings(m.Reader(self.root),'0'*64)
    def test_changed_cache(self):
        self.put('DATA.npz',{'mutated':True})
        with self.assertRaises(ValueError):self.check()
    def test_failed_prepare(self):
        self.prep['passed']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_prepare_rank_flag_required(self):
        self.prep.pop('common_valid49_ranking_before_eligibility_verified');self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_prepare_order_count_exact(self):
        self.prep['orders_checked']=488;self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_missing_diagnostic(self):
        (self.p/'DIAGNOSE_STARTED.json').unlink()
        with self.assertRaises(ValueError):self.check()
    def test_future_diagnostic(self):
        self.put('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2999-01-01T00:00:00Z'))
        with self.assertRaises(ValueError):self.check()
    def test_frozen_set_exact(self):
        frozen=json.loads((self.p/'FROZEN.json').read_text());frozen['hashes'].pop('TRAIN_OPEN_TRIALS.json');self.put('FROZEN.json',frozen)
        with self.assertRaises(ValueError):self.check()
    def test_concept_bound(self):
        self.plan['hashes'].pop(str(self.concept))
        with self.assertRaises(ValueError):self.proposal()
    def test_all_selection_fields(self):
        original=self.inputs['proposal'].read_text()
        for key in m.proposal_contract():
            self.inputs['proposal'].write_text(original);self.replace_proposal(lambda p:p['selection_contract'].update({key:None}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.proposal()
    def test_all_feature_fields(self):
        original=self.inputs['proposal'].read_text()
        for key in m.contract.feature_contract():
            self.inputs['proposal'].write_text(original);self.replace_proposal(lambda p:p['feature_contract'].update({key:None}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.proposal()
    def test_bool_is_not_int(self):
        self.replace_proposal(lambda p:p['selection_contract'].update(new_model_fits=False))
        with self.assertRaises(ValueError):self.proposal()
    def test_changed_dose(self):
        self.replace_proposal(lambda p:p.update(dose=.02))
        with self.assertRaises(ValueError):self.proposal()
    def test_no_prior_returns(self):
        self.replace_proposal(lambda p:p.update(actual_new_returns=1))
        with self.assertRaises(ValueError):self.proposal()
    def test_original_price_digest_required(self):
        self.replace_proposal(lambda p:p['sources']['prices'].update(sha256='0'*64))
        with self.assertRaises(ValueError):self.proposal()
    def test_source_file_api(self):
        self.assertEqual(m.source_files(self.root)[0],self.root/'work/monthly_session_cluster148_outcome_independent.py')
if __name__=='__main__':
    write_review='--write-review' in sys.argv
    if write_review:sys.argv.remove('--write-review')
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log=stream.getvalue();print(log)
    if write_review:
        directory=m.ROOT/m.STUDY;directory.mkdir(parents=True,exist_ok=True)
        logpath=directory/'OUTCOME_REVIEW_TESTS.log'
        with logpath.open('x') as f:f.write(log)
        receipt=m.synthetic_receipt(result)
        receipt['test_log']=dict(path=str(logpath),sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
        dest=m.ROOT/'work/monthly_session_cluster148_outcome_prereview.json'
        with dest.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
