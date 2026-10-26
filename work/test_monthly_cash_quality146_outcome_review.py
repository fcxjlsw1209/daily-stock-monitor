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
import monthly_cash_quality146_outcome_review as m


def order(day,end,names):
    return dict(date=str(pd.Timestamp(day).date()),signal_date=str(pd.Timestamp(day).date()),end=str(pd.Timestamp(end).date()),symbols=list(names),scores=[.2]*len(names))


def synthetic_outputs():
    days=pd.bdate_range('2016-01-04',periods=245);x=np.arange(len(days))
    close=pd.DataFrame({s:base*np.exp(rate*x+.007*np.sin(x/freq)) for s,base,rate,freq in [('A',10,.001,4),('B',20,.0007,7),('C',30,.0013,6)]},index=days)
    opening=close*np.array([.999,1.001,1.0002])
    meta=dict(dates=[str(d.date()) for d in days[:-1:2]],ends=[str(d.date()) for d in days[2::2]],symbols=[['A','B','C']+['S'+str(i) for i in range(3,32)]+['ETF'+str(i) for i in range(17)] for _ in range(122)],etfs=['ETF'+str(i) for i in range(17)])
    orders={name:[order(day,end,['A','C'] if k and i%(k+2)==0 else ['A','B']) for i,(day,end) in enumerate(zip(meta['dates'],meta['ends']))] for k,name in enumerate(m.NAMES)}
    data={};results={};navs={};opens={}
    bounds=dict(early2016_2020=(meta['dates'][0],meta['ends'][52]),original2021_2026=(meta['dates'][53],meta['ends'][-1]),training2016_2023=(meta['dates'][0],meta['ends'][87]),later34=(meta['dates'][88],meta['ends'][-1]))
    for name in m.NAMES:
        results[name]={}
        for execution,cost,key in m.PATHS:
            met,nav,trades,hold=m.replay(close,opening,orders[name],days,execution,cost);prefix=name+'_'+key
            results[name][key]=dict(metrics=met,slices={k:m.h.slice_stat(days,nav,*span) for k,span in bounds.items()})
            data[prefix+'_DAILY.json']=[dict(date=str(d.date()),equity=float(v)) for d,v in zip(days,nav)]
            data[prefix+'_TRADES.json']=trades;data[prefix+'_HOLDINGS.json']=hold
            if key=='same_close':navs[name]=nav
            if key=='next_open_cost':opens[name]=nav
    rows=[dict(name=name,**m.h.measures(days,navs[name],meta)) for name in m.NAMES]
    opened=[dict(name=name,**m.h.measures(days,opens[name],meta)) for name in m.NAMES]
    selection=m.select(rows,opened)
    cells=[]
    for i,(day,pool) in enumerate(zip(meta['dates'],meta['symbols'])):
        for j,symbol in enumerate(pool):
            good=j<2
            cells.append(dict(signal_index=i,pool_index=j,signal_date=day,symbol=symbol,available=good,reason='available' if good else 'etf_not_applicable' if j>=32 else 'no_known_event',method=m.METHOD,
                current_event_id='synthetic' if good else None,current_available_from='2016-01-01' if good else None,current_period_end='2015-12-31' if good else None,prior_period_end='2014-12-31' if good else None,current_days=365 if good else None,prior_days=365 if good else None,currency_key='same-report-dollar:synthetic' if good else None,
                earnings_bounded=.2 if good else None,cash_flow_bounded=.4 if good else None,low_accrual_bounded=.25 if good else None))
    feature=dict(original_arrays_unchanged=True,original_months=122,original_width=49,new_model_fits=0,returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True,common_available_cells=244,reasons=dict(Counter(c['reason'] for c in cells)),arithmetic_method='exact_fraction_scaled_amounts_average_assets_ratios_then_float64_bounded_mapping',annual_duration_normalization=False,comparison_flow_values_used=False,latest_event_before_validity_filter=True,same_current_release_for_all_cells=True,cells=cells)
    data.update({'RESULTS.json':results,'TRAIN_TRIALS.json':rows,'TRAIN_OPEN_TRIALS.json':opened,'FROZEN.json':dict(selection=selection),'WALKFORWARD.json':m.walkforward(days,navs,opens,meta),'FEATURE_AUDIT.json':feature,'VALIDATION.json':dict(selection=selection,new_model_fits=0,no_observer_writes=True,no_intramonth_trades=True,independent_outcome_review_pending=True,pristine_oos=False)})
    return close,opening,orders,meta,data


class SelectionTests(unittest.TestCase):
    def groups(self):
        rows=[dict(name=n,training=dict(cagr=.5,sharpe=2.,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=.6)) for n in m.NAMES]
        return rows,deepcopy(rows)
    def winner(self):
        rows,op=self.groups();rows[3]['training']['cagr']=.6;op[3]['training']['cagr']=.6
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
        for i in (1,2):
            rows,op=self.winner();op[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_closed_all_ties_rejected_individually(self):
        for i in range(3):
            rows,op=self.winner();rows[i]['training']['cagr']=.6
            self.assertFalse(m.select(rows,op)['admitted'])
    def test_each_guard_on_each_path_and_reference(self):
        for which in (0,1):
            for i in range(3):
                for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:
                    pair=self.winner();pair[which][i][section][key]=pair[which][3][section][key]+tol+.001
                    self.assertFalse(m.select(*pair)['admitted'])
    def test_boundaries_inclusive(self):
        rows,op=self.winner()
        for group in (rows,op):
            for section,key,tol in [('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)]:group[3][section][key]=group[0][section][key]-tol
        self.assertTrue(m.select(rows,op)['admitted'])
    def test_tiny_strict_advantage_rejected(self):
        rows,op=self.winner();rows[3]['training']['cagr']=.5+5e-13
        self.assertFalse(m.select(rows,op)['admitted'])
    def test_nonfinite_boolean_or_missing_fails(self):
        for value in (None,True,'2',float('nan'),float('inf')):
            rows,op=self.winner();op[3]['training']['sharpe']=value
            self.assertFalse(m.select(rows,op)['admitted'])
        rows,op=self.winner();del rows[3]['early']['cagr'];self.assertFalse(m.select(rows,op)['admitted'])
    def test_later34_irrelevant(self):
        rows,op=self.groups();rows[3]['later34']={'cagr':1000}
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


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_outputs()
    def setUp(self):self.px,self.op,self.orders,self.meta,self.data=deepcopy(self.fixture)
    def check(self):return m.check_outcomes(self.px,self.op,self.orders,self.meta,self.data.__getitem__)
    def test_twelve_paths(self):
        out=self.check();self.assertTrue(out['passed']);self.assertEqual(out['execution_paths_verified'],12);self.assertEqual(out['total_monthly_orders_verified'],488);self.assertEqual(len(out['independent_monthly_comparison']),122)
    def test_nav_tamper(self):
        self.data[m.NAMES[3]+'_next_open_cost_DAILY.json'][4]['equity']+=10
        with self.assertRaises(AssertionError):self.check()
    def test_trade_cost_tamper(self):
        self.data[m.NAMES[0]+'_next_open_cost_TRADES.json'][0]['cost']+=1
        with self.assertRaises(AssertionError):self.check()
    def test_holdings_units_tamper(self):
        rows=self.data[m.NAMES[1]+'_same_close_HOLDINGS.json'];rows[0]['symbols']=[]
        with self.assertRaises((AssertionError,ValueError)):self.check()
    def test_wrong_execution_date(self):
        self.data[m.NAMES[0]+'_next_open_zero_TRADES.json'][0]['date']=self.meta['dates'][0]
        with self.assertRaises(AssertionError):self.check()
    def test_single_name_order_rejected(self):
        self.orders[m.NAMES[0]][0].update(symbols=['A'],scores=[.2])
        with self.assertRaises(ValueError):m.validate_orders(self.px,self.op,self.orders,self.meta)
    def test_feature_mask_disagreement(self):
        self.data['FEATURE_AUDIT.json']['cells'][3]['earnings_bounded']=.2
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_future_availability_rejected(self):
        self.data['FEATURE_AUDIT.json']['cells'][0]['current_available_from']='2017-01-01'
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_etf_cannot_receive_financials(self):
        cell=deepcopy(self.data['FEATURE_AUDIT.json']['cells'][0]);cell.update(pool_index=32,symbol='ETF0')
        self.data['FEATURE_AUDIT.json']['cells'][32]=cell
        with self.assertRaises(ValueError):m.validate_cells(self.data['FEATURE_AUDIT.json']['cells'],self.meta)
    def test_exact_matrix_match_and_tamper(self):
        feature=self.data['FEATURE_AUDIT.json'];arrays={k:np.full((122,49),np.nan) for k in m.HANDS};arrays['common_available']=np.zeros((122,49),bool)
        for c in feature['cells']:
            i,j=c['signal_index'],c['pool_index'];arrays['common_available'][i,j]=c['available']
            if c['available']:
                for k in m.HANDS:arrays[k][i,j]=c[k]
        self.assertEqual(m.verify_feature_matrices(arrays,feature,self.meta),4)
        arrays[m.HANDS[2]][0,0]+=.0001
        with self.assertRaises(AssertionError):m.verify_feature_matrices(arrays,feature,self.meta)
    def test_frozen_choice_tamper(self):
        self.data['FROZEN.json']['selection']['primary']='forged'
        with self.assertRaises(AssertionError):self.check()
    def test_training_measure_tamper(self):
        self.data['TRAIN_OPEN_TRIALS.json'][3]['training']['sharpe']+=.01
        with self.assertRaises(ValueError):self.check()
    def test_walkforward_measure_tamper(self):
        self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']=not self.data['WALKFORWARD.json']['folds'][0]['selection']['admitted']
        with self.assertRaises(ValueError):self.check()
    def test_new_fit_or_comparison_flow_use_rejected(self):
        for key,val in (('new_model_fits',1),('comparison_flow_values_used',True)):
            f=deepcopy(self.data['FEATURE_AUDIT.json']);f[key]=val
            with self.assertRaises(ValueError):m.feature_checks(f,self.data['VALIDATION.json'],self.meta)
    def test_no_new_producer_imported(self):
        self.assertNotIn('monthly_cash_quality146_runner',sys.modules);self.assertNotIn('monthly_cash_quality146_factor',sys.modules)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.p=self.root/m.STUDY;self.p.mkdir(parents=True)
        self.digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        def file(name,value):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
        self.file=file
        def put(name,value):return file(m.STUDY+'/'+name,value)
        self.put=put
        for name in m.CACHE_FILES:put(name,{'synthetic':name})
        put('PRETRAIN_ORDERS.json',{name:[] for name in m.NAMES})
        inputs={k:file('input/'+k+'.json',{}) for k in ('proposal','meta','policy','sessions','protected_manifest')}
        inputs['prices']=file(m.PRICE,{})
        inputs['original_orders']=file('outputs/monthly_local_linear70/local_linear_ORDERS.json',[])
        source_policy=file('input/source_policy.json',{'source_only':True})
        concept=file(m.CONCEPT,{'synthetic_concept':True})
        self.concept=concept
        inputs['proposal'].write_text(json.dumps(dict(round=146,arms=list(m.NAMES),primary_challenger=m.NAMES[3],dose=.01,selection_contract=m.proposal_contract(),concept=dict(path=m.CONCEPT,sha256=self.digest(concept)))))
        events=[dict(event_id='S'+str(i),fact_admitted=False,model_admitted=False,strategy_admitted=False,prior_field_admitted=False,inherited_rejection_codes=['hold'],cash_quality_admission=dict(admitted=False,reasons=['hold'],definition='consolidated_gaap_cfo_net_income_average_assets_v1')) for i in range(44)]
        inputs['admitted_dataset']=file('input/DATASET.json',dict(method=m.METHOD,conditional_research_only=True,policy_sha256=self.digest(source_policy),events=events))
        self.admission=dict(stage='conditional_cash_quality_source_admission',passed=True,data_sha256=self.digest(inputs['admitted_dataset']),method=m.METHOD,policy_sha256=self.digest(source_policy),source_hashes={str(inputs['sessions']):self.digest(inputs['sessions']),str(source_policy):self.digest(source_policy)},first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,financial_ratios=0,factor_calculations=0,portfolio_calculations=0,return_calculations=0,network_requests=0,observer_changes=0,original_raw_events_preserved=True,original_rejections_preserved=True,events=44,admitted_events=0,held_events=44,admitted_event_ids=[],holds=[dict(event_id=e['event_id'],**e['cash_quality_admission']) for e in events])
        inputs['source_admission']=file('input/ADMISSION.json',self.admission)
        proposal=json.loads(inputs['proposal'].read_text())
        proposal.update(status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',actual_new_factors=0,actual_new_orders=0,actual_new_returns=0,
            sources=dict(policy=dict(path=str(source_policy),sha256=self.digest(source_policy)),dataset=dict(path=str(inputs['admitted_dataset']),sha256=self.digest(inputs['admitted_dataset'])),admission=dict(path=str(inputs['source_admission']),sha256=self.digest(inputs['source_admission'])),full_inventory=44,admitted=0,held=44))
        inputs['proposal'].write_text(json.dumps(proposal))
        hashes={str(p):self.digest(p) for p in [*m.source_files(),*inputs.values(),source_policy,concept]}
        self.inputs=inputs
        self.plan=dict(source_policy=dict(path=str(source_policy),sha256=self.digest(source_policy)),round=146,status='REGISTERED',arms=list(m.NAMES),primary_challenger=m.NAMES[3],dose=.01,runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),output_directory=str(self.p),hashes=hashes,raw_source_hashes={str(inputs['prices']):self.digest(inputs['prices'])},independent_prepare_source_hashes={str(m.old.HELPER):m.old.HELPER_SHA},inputs={k:dict(path=str(p),sha256=self.digest(p)) for k,p in inputs.items()},actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,actual_model_fits_before_registration=0)
        patcher=patch.object(m,'CONCEPT_SHA',self.digest(concept));patcher.start();self.addCleanup(patcher.stop)
        self.plan_sha=self.digest(put('PLAN.json',self.plan))
        self.cache={name:self.digest(self.p/name) for name in m.CACHE_FILES};put('CACHE.json',self.cache)
        artifacts={str(self.p/name):self.digest(self.p/name) for name in (*m.CACHE_FILES,'PLAN.json','CACHE.json')}
        flags={key:True for key in (*m.PREPARE_CHECKS,'source_admission_bound','source_union_preservation_verified','raw_prices_independently_verified','calendar_overlay_independently_verified','all_signal_prefixes_verified','exact_integer_closed_form_arithmetic')}
        self.prep=dict(passed=True,stage='prepare',**flags,orders_checked=488,source_cells_checked=5978,stock_cells_checked=3904,new_model_fits=0,monthly_label_models_fit=0,producer_modules_imported=False,returns_or_nav_computed=False,plan_sha256=self.plan_sha,cache_sha256=self.digest(self.p/'CACHE.json'),orders_sha256=self.digest(self.p/'PRETRAIN_ORDERS.json'),source_hashes={str(m.old.HELPER):m.old.HELPER_SHA},artifact_hashes=artifacts)
        put('INDEPENDENT_PREPARE.json',self.prep)
        for name in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','WALKFORWARD.json'):put(name,{})
        self.freeze()
    def freeze(self):
        self.put('FROZEN.json',dict(hashes={name:self.digest(self.p/name) for name in m.FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.put('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2000-01-01T00:00:00+00:00'))
    def check(self):return m.bindings(m.Reader(self.root),self.plan_sha)
    def test_complete_chain(self):self.assertEqual(self.check(),self.p)
    def test_wrong_plan_sha(self):
        with self.assertRaises(ValueError):m.bindings(m.Reader(self.root),'0'*64)
    def test_changed_cache(self):
        self.put('SCORES.npz',{'changed':True})
        with self.assertRaises(ValueError):self.check()
    def test_failed_prepare(self):
        self.prep['passed']=False;self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_missing_prepare_artifact(self):
        del self.prep['artifact_hashes'][str(self.p/'PRICE_PARITY.json')];self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_wrong_order_count(self):
        self.prep['orders_checked']=366;self.put('INDEPENDENT_PREPARE.json',self.prep)
        with self.assertRaises(ValueError):self.check()
    def test_missing_diagnostic_marker(self):
        (self.p/'DIAGNOSE_STARTED.json').unlink()
        with self.assertRaises(ValueError):self.check()
    def test_future_diagnostic_marker(self):
        self.put('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2999-01-01T00:00:00Z'))
        with self.assertRaises(ValueError):self.check()
    def test_diagnostic_requires_utc(self):
        self.put('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,started_utc='2000-01-01T00:00:00'))
        with self.assertRaises(ValueError):self.check()
    def test_frozen_set_exact(self):
        frozen=json.loads((self.p/'FROZEN.json').read_text());del frozen['hashes']['TRAIN_OPEN_TRIALS.json'];self.put('FROZEN.json',frozen)
        with self.assertRaises(ValueError):self.check()
    def test_mid_review_change(self):
        reader=m.Reader(self.root);reader.sha(self.p/'TRAIN_TRIALS.json');self.put('TRAIN_TRIALS.json',{'mutated':True})
        with self.assertRaises(ValueError):reader.sha(self.p/'TRAIN_TRIALS.json')
    def test_missing_concept_pin_rejected(self):
        pins=dict(self.plan['hashes']);del pins[str(self.concept)]
        with self.assertRaises(ValueError):m.verify_proposal(m.Reader(self.root),self.plan,pins)
    def replace_proposal(self,mutate):
        path=self.inputs['proposal'];value=json.loads(path.read_text());mutate(value)
        path.write_text(json.dumps(value));self.plan['inputs']['proposal']['sha256']=self.digest(path);self.plan['hashes'][str(path)]=self.digest(path)
    def test_proposal_selection_contract_each_field_frozen(self):
        original=self.inputs['proposal'].read_text()
        for key,value in m.proposal_contract().items():
            self.inputs['proposal'].write_text(original)
            self.replace_proposal(lambda p:p['selection_contract'].update({key:None}))
            with self.subTest(field=key),self.assertRaises((ValueError,AssertionError)):
                m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_proposal_no_renamed_primary_or_dose(self):
        original=self.inputs['proposal'].read_text()
        for key,value in [('round',145),('dose',.02),('primary_challenger','cash_flow_only'),('arms',list(reversed(m.NAMES)))]:
            self.inputs['proposal'].write_text(original);self.replace_proposal(lambda p:p.update({key:value}))
            with self.subTest(field=key),self.assertRaises((ValueError,AssertionError)):
                m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_proposal_sources_must_match_registered(self):
        self.replace_proposal(lambda p:p['sources']['dataset'].update(sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'Proposal source'):m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_proposal_prior_real_returns_rejected(self):
        self.replace_proposal(lambda p:p.update(actual_new_returns=1))
        with self.assertRaises(ValueError):m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_source_policy_path_and_sha_exact(self):
        self.plan['source_policy']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source policy'):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_source_files_accept_root(self):
        self.assertEqual(m.source_files(self.root)[0],self.root/'work/monthly_cash_quality146_outcome_review.py')
    def test_proposal_wrong_concept_binding(self):
        self.replace_proposal(lambda p:p['concept'].update(sha256='0'*64))
        with self.assertRaises(ValueError):m.verify_proposal(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_certification_upgrade_rejected(self):
        self.admission['first_public_version_verified']=True;self.inputs['source_admission'].write_text(json.dumps(self.admission))
        with self.assertRaises(ValueError):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])
    def test_holds_cannot_be_cleared(self):
        data=json.loads(self.inputs['admitted_dataset'].read_text());data['events'][0]['cash_quality_admission'].update(admitted=True,reasons=[]);self.inputs['admitted_dataset'].write_text(json.dumps(data))
        self.admission['data_sha256']=self.digest(self.inputs['admitted_dataset']);self.inputs['source_admission'].write_text(json.dumps(self.admission));self.plan['inputs']['admitted_dataset']['sha256']=self.admission['data_sha256']
        with self.assertRaisesRegex(ValueError,'old holds'):m.verify_admission(m.Reader(self.root),self.plan,self.plan['hashes'])

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
        dest=m.ROOT/'work/monthly_cash_quality146_outcome_prereview.json'
        with dest.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
