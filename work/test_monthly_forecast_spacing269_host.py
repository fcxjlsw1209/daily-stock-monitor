"""269 lifecycle integration on synthetic arrays only; no source/replay loading."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_forecast_spacing269_host as h
import monthly_forecast_spacing269_kernel as k
import monthly_forecast_spacing269_independent as ind
import monthly_selector_scope268_independent as selector_ind
import test_monthly_selector_scope268_host as oldfixtures


def extract(relative,names,env):
    path=h.ROOT/relative;tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


def fixture(n=5,cw_shape='column'):
    study,a,m,_,_=oldfixtures.fixture(n,cw_shape)
    for i in range(10):a.pop('synthetic_unused_'+str(i))
    shape=(n,6)
    a.update(forecast_available=np.ones(shape,bool),model_available=np.ones(shape,bool),local_available=np.ones(shape,bool),
             raw_source_log=np.tile([0.,1.,2.,3.,4.,100.],(n,1)),
             raw_local_constant=np.tile([1.,3.,4.,5.,7.,10.],(n,1)),
             raw_local_linear=np.tile([0.,10.,11.,12.,13.,30.],(n,1)),
             rank_log_excess=np.zeros(shape),rank_local_constant=np.zeros(shape),rank_local_linear=np.zeros(shape),paired_month_ready=np.zeros(n,bool))
    a['state_weak_narrow'][:]=True
    env=extract('outputs/monthly_quality_research/explore.py',['ranks'],{'np':np,'pd':pd})
    adapter=extract('outputs/monthly_path70_research/incumbent_adapter.py',['_matrix'],{'np':np})
    renv=extract('outputs/monthly_local_linear70/study.py',['_shape','original_rank_inputs','source_ranked_forecast','ranked_forecasts'],
        {'np':np,'pd':pd,'ARMS':('local_constant','local_linear'),'adapter':SimpleNamespace(_matrix=adapter['_matrix']),
         'ep':SimpleNamespace(old=SimpleNamespace(prior=SimpleNamespace(ranks=env['ranks'])))})
    study.source_ranked_forecast=renv['source_ranked_forecast'];study.ranked_forecasts=renv['ranked_forecasts']
    return rebind((study,a,m,None,None))


def rebind(args,*,refresh_ranks=True):
    study,a,m,_,_=args;n=len(m['dates'])
    if refresh_ranks:
        a['rank_log_excess']=study.source_ranked_forecast(a['raw_source_log'],a)
        ranks,ready=study.ranked_forecasts({x:a['raw_'+x] for x in ('local_constant','local_linear')},a['model_available'],a,n)
        for key in ranks:a['rank_'+key]=ranks[key]
        a['paired_month_ready']=ready
    P=a['synthetic_pre'].copy();inputs=h.selector_inputs(study,P,a,m,n)
    source=dict(baseline=inputs[4],core=inputs[5],core_weight=inputs[6],state_calm10=inputs[7],state_weak_narrow=inputs[8])
    choices,F=study.ep.select_boundary(P,inputs[1],inputs[2],inputs[3],source,dict(budget=.05,ceiling=5,gap=.02))
    ref=dict(meta=deepcopy(m),orders=study.rp.make_orders(choices,F,m),scores=F,pre_scores=P,eligible=a['eligible'].copy())
    auth=h._SourceAuthority(ref,a,dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(auth)
    return study,a,m,ref,auth


def run(args,n=None,**kwargs):
    study,a,m,ref,auth=args
    return h.prepare_one(study,a,m,ref,len(m['dates']) if n is None else n,authority=auth,**kwargs)


def prepared_fixture():
    args=fixture();full,main,cache=run(args);cached=[];physical=[]
    for n in range(6):
        got,e,_=run(args,n,cached_pre=cache);h.exact_prefix(full,got,n);cached.append(e)
    for n in (0,1,3,4):
        got,e,_=run(args,n);h.exact_prefix(full,got,n);physical.append(e)
    alias=deepcopy(main);alias['reuses_main_verified_full']=True;physical.append(alias)
    evidence=dict(schema='forecast_spacing269_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True)
    evidence['execution_accounting']=h.execution_accounting(evidence)
    source=dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
    return full,evidence,source,args[2]


def contract_fixture():
    pins={'work/'+name:'a'*64 for name in h.code_names};pins.update({name:'a'*64 for name in h.SUPPLEMENTAL})
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,promotable_arms=list(h.ARMS[1:]),diagnostic_only=[],predecessor=dict(directory=h.PRIOR.name,completion_sha256=h.PRIOR_COMPLETION_SHA,result_review_sha256='b'*64),input_sha256=deepcopy(pins))
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,code_files=['work/'+name for name in h.code_names],supplemental_input_files=list(h.SUPPLEMENTAL),input_sha256=deepcopy(pins))
    return c,d


class PreparationTests(unittest.TestCase):
    def test_exact_original_and_two_transform_original_selector_arms(self):
        args=fixture();got,e,_=run(args)
        self.assertEqual(got[0][h.ARMS[0]],args[3]['orders']);h.entry_check(e,5,6)
        self.assertEqual(e['producer_counts']['original_selector_calls'],3)
        self.assertEqual(e['producer_counts']['experimental_transform_calls'],2)
        self.assertEqual(h.independent_work(e)['selector']['selector_calls'],3)
        for mode in h.ARMS[1:]:
            self.assertTrue(np.any(got[2][mode][1:]!=got[2][h.ARMS[0]][1:]))
            self.assertEqual(e['independent'][mode]['selector']['mode'],'incumbent_local67')

    def test_initial_partial_identity_preserves_original_episode(self):
        got,_,_=run(fixture())
        for arm in h.ARMS[1:]:
            self.assertEqual(got[0][arm][0],got[0][h.ARMS[0]][0])
            h.exact_array(got[1][arm][:1],got[1][h.ARMS[0]][:1],'initial F')
            h.exact_array(got[2][arm][:1],got[2][h.ARMS[0]][:1],'initial P')

    def test_nonweak_and_not_jointready_rows_copy(self):
        args=fixture();args[1]['state_weak_narrow'][1]=False
        args[1]['raw_local_constant'][2,0]=np.nan
        args=rebind(args);got,_,_=run(args)
        for arm in h.ARMS[1:]:
            h.exact_array(got[2][arm][1:3],got[2][h.ARMS[0]][1:3],'fallback P')
            self.assertEqual(got[0][arm][1:3],got[0][h.ARMS[0]][1:3])
        self.assertFalse(args[1]['paired_month_ready'][2])

    def test_local_available_does_not_shrink_original_u(self):
        args=fixture();expected=run(args)[0];args[1]['local_available'][:]=False;args=rebind(args)
        h.exact_prefix(expected,run(args)[0],5)

    def test_use_is_original_eligibility_intersection_forecast_availability(self):
        args=fixture();args[1]['forecast_available'][:,5]=False;args[1]['eligible'][:,4]=False;args=rebind(args)
        got,_,_=run(args)
        for arm in h.ARMS[1:]:
            h.exact_array(got[2][arm][:,4:],got[2][h.ARMS[0]][:,4:],'Outside U unchanged')
            self.assertTrue(all(event['use_indices']==[0,1,2,3] for event in got[3]['arms'][arm]['transform']['events']))

    def test_constant_raw_identity_both_arms(self):
        args=fixture()
        for name in ('raw_source_log','raw_local_constant','raw_local_linear'):args[1][name][:]=7.
        args=rebind(args);got,_,_=run(args)
        for arm in h.ARMS[1:]:
            h.exact_array(got[2][arm],got[2][h.ARMS[0]],'Identity P')
            h.exact_array(got[1][arm],got[1][h.ARMS[0]],'Identity F')
            self.assertEqual(got[0][arm],got[0][h.ARMS[0]])

    def test_old_metadata_and_three_original_weight_shapes(self):
        got=[run(fixture(cw_shape=x))[0] for x in ('column','vector','matrix')]
        for other in got[1:]:h.exact_prefix(got[0],other,5)
        args=fixture();before=deepcopy(args[2]);run(args);self.assertEqual(args[2],before)

    def test_bad_p_or_original_reference_rejected_before_transform(self):
        for kind in ('P','orders','scores'):
            args=fixture()
            if kind=='P':args[0].modified_pre_score=lambda spec,a,n:a['synthetic_pre'][:n]+.001
            elif kind=='orders':args[3]['orders'][1]['symbols']=['bad']
            else:args[3]['scores'][1,0]+=.1
            with patch.object(k,'transform') as t,self.assertRaises(ValueError):run(args)
            t.assert_not_called()

    def test_pooled_local_rank_and_jointready_corruption_block_transform(self):
        for name in ('rank_log_excess','rank_local_linear','rank_local_constant','paired_month_ready'):
            args=fixture()
            if name=='paired_month_ready':args[1][name][1]=False
            else:args[1][name][1,0]+=.01
            args=rebind(args,refresh_ranks=False)
            with patch.object(k,'transform') as t,self.assertRaisesRegex(ValueError,'source parity'):run(args)
            t.assert_not_called()

    def test_rank_helper_mutation_is_rejected(self):
        args=fixture();original=args[0].ranked_forecasts
        def mutate(raw,available,a,n):
            result=original(raw,available,a,n);a['effects'][0,0]=.125;return result
        args[0].ranked_forecasts=mutate
        with patch.object(k,'transform') as t,self.assertRaisesRegex(ValueError,'mutated'):run(args)
        t.assert_not_called()

    def test_original_independent_gate_precedes_candidates(self):
        with patch.object(selector_ind,'verify',side_effect=ValueError('original gate')),patch.object(k,'transform') as t,self.assertRaisesRegex(ValueError,'original gate'):run(fixture())
        t.assert_not_called()

    def test_transform_mutation_stops_second_arm(self):
        original=k.transform;calls=[]
        def mutate(*args):
            calls.append(args[-1]);result=original(*args);args[0][0,0]+=.1;return result
        with patch.object(k,'transform',side_effect=mutate),self.assertRaisesRegex(ValueError,'mutated'):run(fixture())
        self.assertEqual(calls,[h.ARMS[1]])

    def test_wrong_transform_and_fake_independent_receipt_fail(self):
        original=k.transform
        def change(*args):
            p,a=original(*args);p[1,0]+=.0001;return p,a
        with patch.object(k,'transform',side_effect=change),self.assertRaises(ValueError):run(fixture())
        with patch.object(ind,'verify_transform',return_value={'passed':True}),self.assertRaises(ValueError):run(fixture())

    def test_n0_partial_prefix_no_future_numeric_consumption(self):
        args=fixture()
        for name in ('raw_source_log','raw_local_constant','raw_local_linear','effects','synthetic_pre'):args[1][name][2:]=np.inf
        args[3]['scores'][2:]=np.nan
        for n in (0,1,2):run(args,n)
        with self.assertRaises((ValueError,FloatingPointError)):run(args,3)

    def test_cached_p_must_be_issued_and_prefix_exact(self):
        args=fixture();_,_,cache=run(args)
        with self.assertRaises(ValueError):run(args,1,cached_pre=object())
        cache.values[2:]=np.nan;run(args,2,cached_pre=cache)
        with self.assertRaisesRegex(ValueError,'Cached original P'):run(args,3,cached_pre=cache)

    def test_all_prefixes_and_actual_work_ledger(self):
        full,evidence,source,_=prepared_fixture()
        h.preparation_evidence(evidence,full[3],source,total=5,width=6,prefixes=(0,1,3,4,5))
        ledger=evidence['execution_accounting'];self.assertEqual(ledger['actual_prepare_entries'],11)
        self.assertEqual(ledger['producer']['original_pre_score_calls'],5)
        self.assertEqual(ledger['producer']['original_selector_calls'],33)
        self.assertEqual(ledger['producer']['source_joint_rank_calls'],11)
        self.assertEqual(ledger['independent_selector']['selector_calls'],33)
        self.assertEqual(ledger['independent_transform']['transform_calls'],22)

    def test_balanced_cached_selector_counter_forgery_rejected(self):
        full,e,source,_=prepared_fixture()
        e['cached_prefixes'][1]['independent'][h.ARMS[2]]['selector']['work']['partial_events']=0
        e['execution_accounting']=h.execution_accounting(e)
        with self.assertRaisesRegex(ValueError,'prefix work'):h.preparation_evidence(e,full[3],source,total=5,width=6,prefixes=(0,1,3,4,5))

    def test_source27_bytes_and_four_integrity_checks(self):
        args=fixture();self.assertEqual(len(args[1]),27);before=h.old.array_snapshot(args[1]);_,entry,_=run(args)
        self.assertEqual(before,h.old.array_snapshot(args[1]))
        self.assertEqual(entry['source_audit']['source_integrity_work']['cache_values_compared'],4*sum(x.size for x in args[1].values()))

    def test_unissued_or_wrong_source_identity_rejected(self):
        args=list(fixture());args[-1]=object()
        with self.assertRaisesRegex(ValueError,'Host-issued'):run(args)
        args=list(fixture());args[3]=deepcopy(args[3])
        with self.assertRaisesRegex(ValueError,'Host-issued'):run(args)

    def test_plan_before_any_actual_loader(self):
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.old,'local') as load,self.assertRaisesRegex(ValueError,'not registered'):h.load_sources('x')
        load.assert_not_called()

    def test_exact_array_negative_zero_dtype_and_nan_payload(self):
        for bad in (np.array([0.]),np.array([-0.],dtype=np.float32)):
            with self.assertRaises(ValueError):h.exact_array(np.array([-0.]),bad,'bits')
        a=np.array([0.]);a.view(np.uint64)[0]=0x7ff8000000000123;b=a.copy();b.view(np.uint64)[0]+=1
        with self.assertRaises(ValueError):h.exact_array(a,b,'nan payload')


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

    def test_counter_264_1303_is_required(self):
        state=dict(completed=[dict(directory='x',trials=4)]*263+[dict(directory=h.PRIOR.name,trials=251)])
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
        relative='work/monthly_forecast_spacing269_host.py'
        digest=h.sha(h.ROOT/relative)
        pins=h.canonical_pins({relative:digest,str(h.ROOT/relative):digest})
        self.assertEqual(len(pins),1)
        with self.assertRaises(ValueError):h.canonical_pins({relative:'a'*64,str(h.ROOT/relative):'b'*64})
        with self.assertRaises(ValueError):h.canonical_pins({str(h.CENTRAL):'a'*64})


if __name__=='__main__':unittest.main()
