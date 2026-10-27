"""Synthetic integration for 268 host; no real cache, model or return loading."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast, json, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_selector_scope268_host as h
import monthly_selector_scope268_kernel as k
import monthly_selector_scope268_independent as ind


def pure_function(relative,name):
    path=h.ROOT/relative;tree=ast.parse(path.read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    return env[name]


def fixture(n=5,cw_shape='column'):
    w=6;P=np.tile(np.array([.5,.49,.48,.479,.478,.477]),(n,1))
    dates=['2016-08-17']+[str(x.date()) for x in pd.date_range('2016-08-31',periods=max(0,n-1),freq='ME')]
    dates=dates[:n];ends=dates[1:]+(['2027-01-31'] if n else [])
    m=dict(dates=dates,ends=ends,symbols=[list('ABCDEF') for _ in range(n)],etfs=['F'],early_months=53,train_months=88)
    a=dict(baseline=np.zeros((n,w)),core=np.zeros((n,w)),core_weight=np.full((n,1),.52),eligible=np.ones((n,w),bool),effects=np.tile([-.5,-.5,0.,.5,.5,.5],(n,1)),state_calm10=np.zeros(n,bool),state_weak_narrow=np.zeros(n,bool),synthetic_pre=P.copy())
    if cw_shape=='vector':a['core_weight']=a['core_weight'][:,0].copy()
    if cw_shape=='matrix':a['core_weight']=np.full((n,w),.52)
    for i in range(19):a['synthetic_unused_'+str(i)]=np.zeros((n,w))
    original=pure_function('outputs/monthly_daily_episodes/episodes.py','select_boundary')
    cw=pure_function('outputs/monthly_path70_research/incumbent_adapter.py','_core_weight')
    def make_orders(choices,scores,meta):return [dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[i,j]) for j in ids]) for i,ids in enumerate(choices)]
    study=SimpleNamespace(configs=lambda:[None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')],modified_pre_score=lambda spec,data,n:data['synthetic_pre'][:n].copy(),ep=SimpleNamespace(select_boundary=original),adapter=SimpleNamespace(_core_weight=cw),rp=SimpleNamespace(make_orders=make_orders))
    ss={x:a[x] for x in ('baseline','core','state_calm10','state_weak_narrow')};ss['core_weight']=cw(a['core_weight'],P.shape,n)
    choices,F=original(P,a['eligible'],m['symbols'],a['effects'],ss,dict(budget=.05,ceiling=5,gap=.02))
    ref=dict(meta=deepcopy(m),orders=make_orders(choices,F,m),scores=F,pre_scores=P.copy(),eligible=a['eligible'].copy())
    auth=h._SourceAuthority(ref,a,dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(auth)
    return study,a,m,ref,auth


def run(args,n=None,**kwargs):
    study,a,m,ref,auth=args
    return h.prepare_one(study,a,m,ref,len(m['dates']) if n is None else n,authority=auth,**kwargs)


def contract_fixture():
    pins={'work/'+name:'a'*64 for name in h.code_names};pins.update({name:'a'*64 for name in h.SUPPLEMENTAL})
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,promotable_arms=list(h.ARMS[1:]),diagnostic_only=[],predecessor=dict(directory=h.PRIOR.name,completion_sha256=h.PRIOR_COMPLETION_SHA,result_review_sha256='b'*64),input_sha256=deepcopy(pins))
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,code_files=['work/'+name for name in h.code_names],supplemental_input_files=list(h.SUPPLEMENTAL),input_sha256=deepcopy(pins))
    return c,d


class PreparationTests(unittest.TestCase):
    def test_real_original_gate_and_three_pure_selectors(self):
        args=fixture();got,e,_=run(args)
        self.assertEqual(set(got[0]),set(h.ARMS));h.entry_check(e,5,6)
        self.assertEqual(got[0][h.ARMS[0]],args[3]['orders'])
        self.assertEqual(got[0][h.ARMS[1]][1]['symbols'],['D','E','C'])
        self.assertEqual(got[0][h.ARMS[2]][1]['symbols'],['D','E','F'])
        self.assertEqual(e['producer_counts']['original_selector_calls'],1)
        self.assertEqual(e['producer_counts']['experimental_selector_calls'],2)
        self.assertEqual(h.independent_work(e)['selector_calls'],3)

    def test_firstpartial_is_exact_incumbent(self):
        got,_,_=run(fixture())
        for a in h.ARMS[1:]:
            self.assertEqual(got[0][a][0],got[0][h.ARMS[0]][0])
            h.exact_array(got[1][a][:1],got[1][h.ARMS[0]][:1],'partial')

    def test_original_metadata_static_extras_preserved(self):
        args=fixture();before=deepcopy(args[2]);run(args)
        self.assertEqual(args[2],before);self.assertEqual(args[2]['early_months'],53)

    def test_core_weight_all_three_original_shapes(self):
        got=[run(fixture(cw_shape=shape))[0] for shape in ('column','vector','matrix')]
        for other in got[1:]:h.exact_prefix(got[0],other,5)

    def test_unissued_or_cross_source_capability_rejected(self):
        args=list(fixture());args[-1]=object()
        with self.assertRaisesRegex(ValueError,'Host-issued'):run(args)
        args=list(fixture());args[3]=deepcopy(args[3])
        with self.assertRaisesRegex(ValueError,'Host-issued'):run(args)

    def test_original_pre_score_failure_before_new_selector(self):
        args=fixture();args[0].modified_pre_score=lambda spec,data,n:data['synthetic_pre'][:n]+.001
        with patch.object(k,'select') as candidate,self.assertRaisesRegex(ValueError,'original P'):run(args)
        candidate.assert_not_called()

    def test_reference_order_or_score_failure_before_candidate(self):
        for kind in ('orders','scores'):
            args=fixture()
            if kind=='orders':args[3][kind][1]['symbols']=['bad']
            else:args[3][kind][1,0]+=.01
            with patch.object(k,'select') as candidate,self.assertRaises(ValueError):run(args)
            candidate.assert_not_called()

    def test_original_verifier_failure_prevents_candidate(self):
        with patch.object(ind,'verify',side_effect=ValueError('proof fail')),patch.object(k,'select') as candidate,self.assertRaisesRegex(ValueError,'proof fail'):run(fixture())
        candidate.assert_not_called()

    def test_fake_receipt_does_not_pass_gate(self):
        with patch.object(ind,'verify',return_value={'passed':True}),patch.object(k,'select') as candidate,self.assertRaises(ValueError):run(fixture())
        candidate.assert_not_called()

    def test_verifier_source_mutation_prevents_candidate(self):
        original=ind.verify
        def mutate(*args):
            result=original(*args);args[0][0,0]+=.1;return result
        with patch.object(ind,'verify',side_effect=mutate),patch.object(k,'select') as candidate,self.assertRaisesRegex(ValueError,'mutated'):run(fixture())
        candidate.assert_not_called()

    def test_candidate_mutation_prevents_second_candidate(self):
        original=k.select;calls=[]
        def mutate(*args):
            calls.append(args[-1]);result=original(*args);args[3][0,0]+=.1;return result
        with patch.object(k,'select',side_effect=mutate),self.assertRaisesRegex(ValueError,'mutated'):run(fixture())
        self.assertEqual(calls,[h.ARMS[1]])

    def test_candidate_wrong_order_is_independently_rejected(self):
        original=k.select
        def mutate(*args):
            choices,scores,audit=original(*args);choices[1]=list(reversed(choices[1]));return choices,scores,audit
        with patch.object(k,'select',side_effect=mutate),self.assertRaisesRegex(ValueError,'ordered choices'):run(fixture())

    def test_top5_excludes_unadjusted_rank6(self):
        args=fixture();args[1]['effects'][1:]=-.5
        # Fresh fixture authority and reference after intentional synthetic input change.
        args=rebind(args)
        got,_,_=run(args)
        self.assertNotIn('F',got[0][h.ARMS[1]][1]['symbols'])
        self.assertEqual(got[0][h.ARMS[1]][1]['symbols'],['A','B','C'])

    def test_n0_and_prefix_ignore_poisoned_future_numeric(self):
        args=fixture();args[1]['effects'][2:]=np.nan;args[1]['synthetic_pre'][2:]=np.nan
        args[3]['scores'][2:]=np.nan
        for n in (0,1,2):run(args,n)
        with self.assertRaises(ValueError):run(args,3)

    def test_cached_pre_forgery_and_future_scope(self):
        args=fixture();full,e,cache=run(args)
        with self.assertRaises(ValueError):run(args,1,cached_pre=object())
        cache.values[2:]=np.nan
        run(args,2,cached_pre=cache)
        with self.assertRaisesRegex(ValueError,'Cached original P'):run(args,3,cached_pre=cache)

    def test_all_prefixes_exact_and_full_alias_ledger(self):
        args=fixture();full,main,cache=run(args);cached=[];physical=[]
        for n in range(6):
            got,e,_=run(args,n,cached_pre=cache);h.exact_prefix(full,got,n);cached.append(e)
        for n in (0,1,3):
            got,e,_=run(args,n);h.exact_prefix(full,got,n);physical.append(e)
        alias=deepcopy(main);alias['reuses_main_verified_full']=True;physical.append(alias)
        evidence=dict(schema='selector_scope268_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True)
        evidence['execution_accounting']=h.execution_accounting(evidence)
        source=dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
        h.preparation_evidence(evidence,full[3],source,total=5,width=6,prefixes=(0,1,3,5))
        self.assertEqual(evidence['execution_accounting']['actual_prepare_entries'],10)
        self.assertEqual(evidence['execution_accounting']['producer']['original_pre_score_calls'],4)
        self.assertEqual(evidence['execution_accounting']['independent']['selector_calls'],30)
        bad=deepcopy(evidence);bad['physical_prefixes'][-1]['producer_counts']['original_selector_calls']=2
        with self.assertRaises(ValueError):h.preparation_evidence(bad,full[3],source,total=5,width=6,prefixes=(0,1,3,5))

    def test_exact_array_signed_zero_dtype_nanpayload(self):
        for bad in (np.array([0.]),np.array([-0.],dtype=np.float32)):
            with self.assertRaises(ValueError):h.exact_array(np.array([-0.]),bad,'bits')
        a=np.array([0.]);a.view(np.uint64)[0]=0x7ff8000000000123;b=a.copy();b.view(np.uint64)[0]+=1
        with self.assertRaises(ValueError):h.exact_array(a,b,'nan payload')

    def test_source_arrays_unchanged_and_source_counter_exact(self):
        args=fixture();before=h.old.array_snapshot(args[1]);_,e,_=run(args)
        self.assertEqual(before,h.old.array_snapshot(args[1]))
        self.assertEqual(e['source_audit']['source_integrity_work']['cache_values_compared'],4*sum(a.size for a in args[1].values()))

    def test_plan_gate_precedes_any_actual_loader(self):
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.old,'local') as load,self.assertRaisesRegex(ValueError,'not registered'):h.load_sources('x')
        load.assert_not_called()


def rebind(args):
    study,a,m,_,_=args;n=len(m['dates']);P=a['synthetic_pre'].copy();inputs=h.selector_inputs(study,P,a,m,n)
    source=dict(baseline=inputs[4],core=inputs[5],core_weight=inputs[6],state_calm10=inputs[7],state_weak_narrow=inputs[8])
    choices,F=study.ep.select_boundary(P,inputs[1],inputs[2],inputs[3],source,dict(budget=.05,ceiling=5,gap=.02))
    ref=dict(meta=deepcopy(m),orders=study.rp.make_orders(choices,F,m),scores=F,pre_scores=P,eligible=a['eligible'].copy())
    auth=h._SourceAuthority(ref,a,dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(auth)
    return study,a,m,ref,auth


class ClosureTests(unittest.TestCase):
    def contract(self,c,d):
        with patch.object(Path,'is_file',return_value=True),patch.object(h,'read',side_effect=lambda p:c if p==h.CONTEXT else d),patch.object(h,'sha',return_value='a'*64):return h.contract_check()

    def test_formal8_and_reader2_exact(self):
        c,d=contract_fixture();self.contract(c,d)
        self.assertEqual(len(h.code_names),8);self.assertEqual(len(h.PREP_FILES),8);self.assertEqual(len(h.RESULT_FILES),38)

    def test_missing_or_wrong_source_pin_rejected_both_papers(self):
        for paper in (0,1):
            for name in ['work/'+h.code_names[0],h.SUPPLEMENTAL[0],h.SUPPLEMENTAL[1]]:
                for value in (None,'b'*64):
                    c,d=contract_fixture()
                    if value is None:(c,d)[paper]['input_sha256'].pop(name)
                    else:(c,d)[paper]['input_sha256'][name]=value
                    with self.subTest(paper=paper,name=name),self.assertRaises(ValueError):self.contract(c,d)

    def test_inventory_draft_role_and_context_mutants_rejected(self):
        for key,value in [('code_files',[]),('code_files',list(reversed(['work/'+x for x in h.code_names]))),('supplemental_input_files',list(h.SUPPLEMENTAL)+['extra']),('context_sha256','b'*64),('frozen',False),('actual_data_started',True)]:
            c,d=contract_fixture();d[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.contract(c,d)
        c,d=contract_fixture();c['promotable_arms']=[h.ARMS[2]]
        with self.assertRaises(ValueError):self.contract(c,d)

    def test_counter_263_1300_is_required(self):
        state=dict(completed=[dict(directory='x',trials=4)]*262+[dict(directory=h.PRIOR.name,trials=252)])
        h.initial_state(state);state['completed'].pop()
        with self.assertRaises(ValueError):h.initial_state(state)

    def test_zero_cost_real_nested_shape(self):
        exports={};results={}
        for a in h.ARMS:
            exports[a]={};results[a]={}
            for p in ('same_close_zero','next_open_zero'):
                exports[a][p]=dict(TRADES=[dict(cost=0.)],HOLDINGS=[dict(fee=0.)])
                metric=dict(cost_per_side=0.,fees_paid=0.)
                results[a][p]=dict(metrics=metric,slices=dict(full122=dict(metric)))
        h.zero_cost_check(exports,results)
        results[h.ARMS[0]]['next_open_zero']['metrics']['fees_paid']=1e-20
        with self.assertRaises(ValueError):h.zero_cost_check(exports,results)

    def test_phase_exit_and_exact_artifact_binding(self):
        p='c'*64;phases={}
        for name,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:phases[name]=dict(session_id=123,exit_code=0,chunk_id='chunk',artifact_path=str(h.OUT/relative),artifact_sha256='a'*64)
        record=dict(plan_sha256=p,phases=phases)
        with patch.object(h,'read',return_value=record),patch.object(h,'sha',return_value='a'*64):h.phase_terminals_check(p)
        phases['prepare']['exit_code']=1
        with patch.object(h,'read',return_value=record),patch.object(h,'sha',return_value='a'*64),self.assertRaises(ValueError):h.phase_terminals_check(p)

    def test_saved_review_mandatory_before_close(self):
        with patch.object(h,'phase_terminals_check',side_effect=ValueError('missing owned phases')),patch.object(h,'read') as read,self.assertRaisesRegex(ValueError,'owned phases'):h.result_review_check('x')
        read.assert_not_called()

    def test_alias_pin_conflict_and_central_refusal(self):
        relative='work/monthly_selector_scope268_host.py'
        digest=h.sha(h.ROOT/relative)
        pins=h.canonical_pins({relative:digest,str(h.ROOT/relative):digest})
        self.assertEqual(len(pins),1)
        with self.assertRaises(ValueError):h.canonical_pins({relative:'a'*64,str(h.ROOT/relative):'b'*64})
        with self.assertRaises(ValueError):h.canonical_pins({str(h.CENTRAL):'a'*64})


if __name__=='__main__':unittest.main()
