"""272 lifecycle and prefix tests using synthetic49-wide arrays only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import monthly_roa_increment272_runner as h
import monthly_roa_increment272_factor as f
import monthly_roa_increment272_independent as ind
import monthly_selector_scope268_independent as si
from test_monthly_selector_scope268_host import pure_function
from test_monthly_roa_increment272_factor import data,Poison,metrics


def fixture(n=5,cw_shape='column'):
    dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31','2016-11-30'][:n]
    symbols=['AAPL','AMD','MSFT']+['S%02d'%i for i in range(3,48)]+['ETF']
    m=dict(dates=dates,ends=dates[1:]+(['2016-12-30'] if n else []),symbols=[symbols[:] for _ in dates],etfs=['ETF'],early_months=53,train_months=88)
    P=np.tile(.5-np.arange(49)*.001,(n,1)).astype(np.float64)
    a=dict(baseline=np.zeros((n,49)),core=np.zeros((n,49)),core_weight=np.full((n,1),.52),eligible=np.ones((n,49),bool),effects=np.zeros((n,49)),state_calm10=np.zeros(n,bool),state_weak_narrow=np.zeros(n,bool),synthetic_pre=P.copy())
    if cw_shape=='vector':a['core_weight']=a['core_weight'][:,0].copy()
    if cw_shape=='matrix':a['core_weight']=np.full((n,49),.52)
    for i in range(19):a['unused_'+str(i)]=np.zeros((n,49))
    d=data(dates or ['2016-08-17']);hand=np.full((n,49),np.nan)
    if n:hand[:,3]=.1;hand[:,4]=-0.
    financial=dict(hand=hand,mask=np.isfinite(hand),catalog=d['full_catalog'],records=d['accepted_records'],sessions=d['sessions'])
    selector=pure_function('outputs/monthly_daily_episodes/episodes.py','select_boundary')
    adapter=h.old.module('roa272_synthetic_original_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
    calls=[]
    def pre(spec,arrays,count):
        assert all(v.shape[0]==count for v in arrays.values()),'pre saw future tail'
        calls.append(count);return arrays['synthetic_pre'].copy()
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],signal_date=meta['dates'][i],end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[i,j]) for j in ids]) for i,ids in enumerate(choices)]
    study=SimpleNamespace(configs=lambda:[None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')],modified_pre_score=pre,adapter=adapter,ep=SimpleNamespace(select_boundary=selector),rp=SimpleNamespace(make_orders=orders),pre_calls=calls)
    ref=dict(meta=deepcopy(m))
    for name,hnd in (('incumbent',None),('control',hand)):
        c,F,_=h.episode(study,P,hnd,a,m);ref[name]=dict(orders=orders(c,F,m),scores=F)
    ref['incumbent']['pre_scores']=P.copy()
    auth=h._SourceAuthority(ref,a,financial,dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(auth)
    return study,a,m,ref,financial,auth


def run(args,n=None,**kw):
    s,a,m,r,fin,auth=args
    return h.prepare_one(s,a,m,r,fin,len(m['dates']) if n is None else n,authority=auth,**kw)


def prepared_fixture():
    args=fixture();full,main,cache=run(args);cached=[];physical=[]
    for n in range(6):
        got,e,_=run(args,n,cached_pre=cache);h.exact_prefix(full,got,n);cached.append(e)
    for n in (0,1,3,4):
        got,e,_=run(args,n);h.exact_prefix(full,got,n);physical.append(e)
    e=deepcopy(main);e['reuses_main_verified_full']=True;physical.append(e)
    proof=dict(schema='roa_increment272_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True,financial_source_unchanged=True)
    proof['execution_accounting']=h.execution_accounting(proof)
    source=dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
    return full,proof,source,args


class PrefixTests(unittest.TestCase):
    def test_full_two_reference_gates_and_candidate(self):
        args=fixture();out,e,_=run(args);h.entry_check(e,5,49)
        for arm,key in zip(h.ARMS,('incumbent','control')):
            self.assertEqual(out[0][arm],args[3][key]['orders']);h.exact_array(out[1][arm],args[3][key]['scores'],'reference')
        self.assertGreater(e['overlay_counts']['filled_cells'],0)
        self.assertEqual(e['producer_counts']['original_selector_calls'],3)
        self.assertTrue(e['source_audit']['both_reference_gates_before_overlay'])
        self.assertEqual(args[0].pre_calls,[5])

    def test_physical_slice_precedes_every_original_pre(self):
        for shape in ('vector','column','matrix'):
            args=fixture(cw_shape=shape)
            for n in (0,1,3):run(args,n)
            self.assertEqual(args[0].pre_calls,[0,1,3])
            for value in args[1].values():self.assertEqual(value.shape[0],5)

    def test_future_numeric_poison_and_initial_only(self):
        args=fixture()
        for key in ('synthetic_pre','effects','baseline','core_weight'):args[1][key][2:]=np.inf
        args[4]['hand'][2:]=np.inf
        args[3]['control']['scores'][2:]=np.inf
        for n in (0,1,2):run(args,n)
        with self.assertRaises(ValueError):run(args,3)

    def test_n0_does_not_read_financial_body(self):
        args=fixture();args[4]['catalog']=Poison();args[4]['records']=Poison();args[4]['sessions']=Poison()
        out,e,_=run(args,0);self.assertEqual(e['overlay_counts']['ratio_evaluations'],0)
        for arm in h.ARMS:self.assertEqual(out[0][arm],[])

    def test_reference_corruption_stops_before_overlay(self):
        for role,field in (('incumbent','scores'),('control','scores'),('control','orders')):
            args=fixture()
            if field=='scores':args[3][role][field][0,0]+=.01
            else:args[3][role][field][0]['symbols']=['bad']
            with patch.object(f,'overlay_prefix') as overlay,self.assertRaises(ValueError):run(args)
            overlay.assert_not_called()

    def test_wrong_original_pre_stops_before_overlay(self):
        args=fixture();args[0].modified_pre_score=lambda spec,a,n:a['synthetic_pre']+.01
        with patch.object(f,'overlay_prefix') as overlay,self.assertRaises(ValueError):run(args)
        overlay.assert_not_called()

    def test_control_independent_failure_stops_before_overlay(self):
        with patch.object(ind,'verify_pre_score',side_effect=ValueError('independent control')),patch.object(f,'overlay_prefix') as overlay,self.assertRaisesRegex(ValueError,'independent control'):run(fixture())
        overlay.assert_not_called()

    def test_overlay_corruption_independently_rejected(self):
        original=f.overlay_prefix
        def wrong(*a):
            hand,mask,audit=original(*a);hand[0,0]=.2;return hand,mask,audit
        with patch.object(f,'overlay_prefix',side_effect=wrong),self.assertRaises(ValueError):run(fixture())

    def test_adapter_point5_bound_including_ineligible_no_clipping(self):
        args=fixture();P=args[3]['incumbent']['pre_scores'];hand=args[4]['hand'].copy();hand[0,0]=.5000000000000001;args[1]['eligible'][0,0]=False
        with patch.object(args[0].ep,'select_boundary') as select,self.assertRaisesRegex(ValueError,'.5 hand'):h.episode(args[0],P,hand,args[1],args[2])
        select.assert_not_called()
        self.assertEqual(hand[0,0],.5000000000000001)

    def test_149_correction_exact_zero_missing_and_eligible(self):
        args=fixture();P=args[3]['incumbent']['pre_scores'].copy();P[0,4]=-0.
        hand=args[4]['hand'].copy();args[1]['eligible'][0,3]=False
        _,_,actual=h.episode(args[0],P,hand,args[1],args[2]);ind.verify_pre_score(P,hand,args[1]['eligible'],actual)
        h.exact_array(actual[0],P[0],'ineligible and signed zero')
        self.assertEqual(actual[1,3],P[1,3]+.01*hand[1,3])

    def test_source_arrays_preserved(self):
        args=fixture();before=h.old.array_snapshot(args[1]);hand=args[4]['hand'].copy();mask=args[4]['mask'].copy()
        run(args);self.assertEqual(before,h.old.array_snapshot(args[1]));h.exact_array(hand,args[4]['hand'],'154hand');h.exact_array(mask,args[4]['mask'],'154mask')

    def test_all_prefixes_and_full_alias_actual_ledger(self):
        full,e,source,args=prepared_fixture();h.preparation_evidence(e,full[3],source,total=5,width=49,prefixes=(0,1,3,4,5))
        work=e['execution_accounting'];self.assertEqual(work['actual_prepare_entries'],11)
        self.assertEqual(work['producer']['original_pre_score_calls'],5);self.assertEqual(work['producer']['original_selector_calls'],33)
        self.assertEqual(work['independent_pre_score']['pre_score_verifications'],22)
        self.assertEqual(work['independent_selector']['selector_calls'],33)
        self.assertEqual(args[0].pre_calls,[5,0,1,3,4])

    def test_balanced_prefix_selector_count_forgery_rejected(self):
        full,e,source,_=prepared_fixture();e['cached_prefixes'][1]['independent'][h.ARMS[0]]['selector']['work']['partial_events']=0;e['execution_accounting']=h.execution_accounting(e)
        with self.assertRaisesRegex(ValueError,'per-prefix selector'):h.preparation_evidence(e,full[3],source,total=5,width=49,prefixes=(0,1,3,4,5))

    def test_alias_cannot_claim_second_execution(self):
        full,e,source,_=prepared_fixture();e['physical_prefixes'][-1]['producer_counts']['original_selector_calls']=2
        with self.assertRaises(ValueError):h.preparation_evidence(e,full[3],source,total=5,width=49,prefixes=(0,1,3,4,5))

    def test_cached_original_pre_requires_issued_binding(self):
        args=fixture();_,_,cache=run(args)
        with self.assertRaises(ValueError):run(args,1,cached_pre=object())
        cache.values[2:]=np.inf;run(args,2,cached_pre=cache)
        with self.assertRaisesRegex(ValueError,'Cached original P'):run(args,3,cached_pre=cache)

    def test_wrong_source_identity_rejected(self):
        args=list(fixture());args[-1]=object()
        with self.assertRaises(ValueError):run(args)


class LifecycleTests(unittest.TestCase):
    def test_plan_before_any_loader(self):
        with patch.object(h,'gate',side_effect=ValueError('PLAN first')),patch.object(h.old,'local') as load,self.assertRaisesRegex(ValueError,'PLAN first'):h.load_sources('x')
        load.assert_not_called()

    def test_only_sameclose_can_select(self):
        rows=[metrics(a,.6 if i==2 else .5) for i,a in enumerate(h.ARMS)]
        values={a:{'same_close_zero':{k:v for k,v in row.items() if k!='name'}} for a,row in zip(h.ARMS,rows)}
        self.assertEqual(h.select_first88(values)['primary'],h.ARMS[2])
        values[h.ARMS[0]]['next_open_zero']=Poison()
        with self.assertRaises(ValueError):h.select_first88(values)

    def test_sameclose_fallback_never_promotes_control(self):
        values={a:{'same_close_zero':{k:v for k,v in metrics(a,.7 if i==1 else .5).items() if k!='name'}} for i,a in enumerate(h.ARMS)}
        self.assertEqual(h.select_first88(values)['primary'],h.ARMS[0])

    def test_three_arm_accounting_bridge_is_local_no_globals(self):
        before=h.prior.ARMS
        value={a:{'sentinel':a} for a in h.ARMS}
        def check(px,op,orders,exports,results,meta):
            self.assertEqual(set(orders),set(before));return dict(passed=True,paths={a:{'sentinel':v['sentinel']} for a,v in orders.items()})
        with patch.object(h.prior,'audit_outcomes',side_effect=check):result=h.audit_outcomes(None,None,value,value,value,{})
        self.assertEqual(set(result['paths']),set(h.ARMS));self.assertEqual(h.prior.ARMS,before)

    def test_outcome_requires_nine_paths_fortyfive_slices(self):
        with self.assertRaises(ValueError):h.outcome_check(dict(passed=True,checked_paths=8))

    def test_zero_fee_nested_actual_shape(self):
        exports={};results={}
        for a in h.ARMS:
            exports[a]={p:dict(TRADES=[dict(cost=0.)],HOLDINGS=[dict(fee=0.)]) for p in ('same_close_zero','next_open_zero')}
            results[a]={p:dict(metrics=dict(cost_per_side=0.,fees_paid=0.),slices=dict(full122=dict(cost_per_side=0.,fees_paid=0.))) for p in exports[a]}
        h.zero_cost_check(exports,results);results[h.ARMS[0]]['same_close_zero']['metrics']['cost_per_side']=.0005
        with self.assertRaises(ValueError):h.zero_cost_check(exports,results)

    def test_exact_bits_reject_signedzero_dtype_nanpayload(self):
        for x in (np.array([0.]),np.array([-0.],dtype=np.float32)):
            with self.assertRaises(ValueError):h.exact_array(np.array([-0.]),x,'bits')
        x=np.array([np.nan]);y=x.copy();y.view(np.uint64)[0]+=1
        with self.assertRaises(ValueError):h.exact_array(x,y,'nanpayload')


def contract_fixture():
    pins={'work/'+name:'a'*64 for name in h.code_names};pins.update({name:'a'*64 for name in h.SUPPLEMENTAL})
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,only_promotable=h.ARMS[2],diagnostic_only=[h.ARMS[1]],predecessor=dict(directory=h.PRIOR.name,completion_sha256=h.PRIOR_COMPLETION_SHA,result_review_sha256='b'*64),input_sha256=deepcopy(pins))
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,code_files=['work/'+name for name in h.code_names],supplemental_input_files=list(h.SUPPLEMENTAL),input_sha256=deepcopy(pins))
    return c,d


class ContractTests(unittest.TestCase):
    def check(self,c,d):
        with patch.object(Path,'is_file',return_value=True),patch.object(h,'sha',return_value='a'*64),patch.object(h,'read',side_effect=lambda path:c if path==h.CONTEXT else d):return h.contract_check()

    def test_exact_six_plus_two_and_single_promotable(self):
        c,d=contract_fixture();self.check(c,d)
        for mutate in (lambda c,d:d['code_files'].reverse(),lambda c,d:d['code_files'].append('work/extra.py'),lambda c,d:d['supplemental_input_files'].pop(),lambda c,d:c.update(only_promotable=h.ARMS[1]),lambda c,d:c.update(actual_data_started=True)):
            c,d=contract_fixture();mutate(c,d)
            with self.assertRaises(ValueError):self.check(c,d)

    def test_missing_wrong_reader_hash_in_either_paper(self):
        for which in (0,1):
            for value in (None,'b'*64,False):
                papers=contract_fixture();papers[which]['input_sha256'][h.SUPPLEMENTAL[0]]=value
                with self.assertRaises(ValueError):self.check(*papers)

    def test_actual_terminal_owner_and_true_exit(self):
        records={name:dict(session_id=123,exit_code=0,chunk_id='abc123',artifact_path=str(h.OUT/rel),artifact_sha256='a'*64) for name,rel in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]}
        terminal=dict(plan_sha256='p',phases=records)
        with patch.object(h,'read',return_value=terminal),patch.object(h,'sha',return_value='a'*64):
            h.phase_terminals_check('p');records['prepare']['owner']='/root';h.phase_terminals_check('p')
            for bad in (True,1):
                records['prepare']['exit_code']=bad
                with self.assertRaises(ValueError):h.phase_terminals_check('p')
            records['prepare']['exit_code']=0;records['prepare']['owner']='other'
            with self.assertRaises(ValueError):h.phase_terminals_check('p')

    def test_overlay_cannot_mutate_baseline_copy_then_pass_oracle(self):
        original=f.overlay_prefix
        def mutate(meta,hand,*args):
            hand[0,3]=.2;return original(meta,hand,*args)
        with patch.object(f,'overlay_prefix',side_effect=mutate),self.assertRaisesRegex(ValueError,'mutated frozen154'):run(fixture())


if __name__=='__main__':unittest.main()
