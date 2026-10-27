"""267 host tests: synthetic caller fixtures only, never actual research data."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import unittest
from unittest.mock import patch
import numpy as np
import monthly_boundary_pair267_host as h
import monthly_boundary_pair267_kernel as k
import monthly_boundary_pair267_source as s
import monthly_boundary_pair267_independent as ind
import test_monthly_boundary_pair267_source as fixture_source


def selector():
    # Parse only the frozen function source; do not import its data loading module.
    path=h.ROOT/'outputs/monthly_daily_episodes/episodes.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_boundary')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env);return env['select_boundary']


def fixture():
    meta,join,train,arrays,P,panel=fixture_source.fixture();n,w=P.shape
    arrays.update(core=np.zeros((n,w)),core_weight=np.full((n,1),.52),state_calm10=np.zeros(n,bool),state_weak_narrow=np.zeros(n,bool),synthetic_pre=P.copy())
    def make_orders(ch,sc,m):return [dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],symbols=[m['symbols'][i][j] for j in ids],scores=[float(sc[i,j]) for j in ids]) for i,ids in enumerate(ch)]
    study=SimpleNamespace(configs=lambda:[None]*4+[dict(id=4,base='incumbent_scope67',mode='local_linear')],modified_pre_score=lambda spec,a,n:a['synthetic_pre'][:n].copy(),ep=SimpleNamespace(select_boundary=selector()),rp=SimpleNamespace(make_orders=make_orders))
    ch,F,_=k.select_once(study.ep.select_boundary,P,arrays['eligible'],meta['monthly']['symbols'],arrays['effects'],h.original_selector_source(arrays,n))
    reference=dict(meta=meta['monthly'],pre_scores=P.copy(),eligible=arrays['eligible'].copy(),scores=F,orders=make_orders(ch,F,meta['monthly']))
    # Explicit fixture-only issuer. Actual load_sources requires gate+all files.
    authority=h._SourceAuthority(reference,dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(authority)
    return study,join,train,arrays,panel,reference,authority


def run(args,n=None,**kwargs):
    study,join,train,arrays,panel,reference,auth=args
    return h.prepare_one(study,join,train,arrays,panel,reference,join['events'] if n is None else n,authority=auth,**kwargs)


def contract_fixture():
    pins={'work/'+name:'a'*64 for name in h.code_names};pins.update({name:'a'*64 for name in h.SUPPLEMENTAL})
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,only_promotable=h.ARMS[2],diagnostic_only=h.ARMS[1],predecessor=dict(directory=h.PRIOR.name,completion_sha256=h.PRIOR_COMPLETION_SHA,result_review_sha256='b'*64),input_sha256=deepcopy(pins))
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,code_files=['work/'+n for n in h.code_names],supplemental_input_files=list(h.SUPPLEMENTAL),input_sha256=deepcopy(pins))
    return c,d


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.full=run(cls.args)

    def test_actual_three_roles_model_source_and_selector_gates(self):
        models,got,e,source=self.full;n=self.args[1]['events'];w=self.args[1]['width']
        self.assertEqual(set(got[0]),set(h.ARMS));h.entry_check(e,n,w)
        self.assertEqual(e['producer_counts']['original_selector_calls'],3)
        self.assertEqual(e['producer_counts']['producer_ridge_solves'],2)
        self.assertEqual(h.independent_work(e)['independent_svd_solves'],2)
        self.assertEqual(e['original_selection'],e['selection_proofs'][h.ARMS[0]])
        self.assertFalse(source['audit']['labels_computed'])
        self.assertEqual(models['annual']['2019']['cohort']['common_months'],28)

    def test_static_original_metadata_preserved_and_canonical_prefix_exact(self):
        args=fixture();args[5]['meta'].update(early_months=53,train_months=88)
        # Issuer snapshots the actual full metadata, while the source adapter's
        # canonical join intentionally contains only four identity/calendar axes.
        auth=h._SourceAuthority(args[5],dict(synthetic_only=True),h._SOURCE_TOKEN);h._ISSUED.append(auth)
        args=(*args[:-1],auth)
        for n in (0,1,15):run(args,n)
        self.assertEqual(args[5]['meta']['early_months'],53)
        self.assertEqual(args[5]['meta']['train_months'],88)
        args[5]['meta']['symbols'][0][0]='WRONG'
        with self.assertRaises(ValueError):run(args,1)

    def test_original_P_failure_before_new_price_or_model(self):
        args=fixture();args[3]['synthetic_pre'][2,0]+=.01
        with patch.object(s,'prepare_inputs') as source,patch.object(k,'fit_models') as fit,self.assertRaises(ValueError):run(args)
        source.assert_not_called();fit.assert_not_called()

    def test_original_order_or_score_mismatch_before_labels(self):
        for field in ('orders','scores'):
            args=fixture()
            if field=='orders':args[5][field][2]['symbols']=['WRONG']
            else:args[5][field][2,0]+=.01
            with self.subTest(field=field),patch.object(s,'prepare_inputs') as source,self.assertRaises(ValueError):run(args)
            source.assert_not_called()

    def test_unissued_or_reused_reference_rejected(self):
        args=fixture();args=list(args);args[-1]=object()
        with patch.object(k,'fit_models') as fit,self.assertRaisesRegex(ValueError,'Host-issued'):run(args)
        fit.assert_not_called()
        args=fixture();args=list(args);args[5]=deepcopy(args[5])
        with self.assertRaisesRegex(ValueError,'Host-issued'):run(args)

    def test_model_verifier_failure_prevents_deployment(self):
        with patch.object(ind,'verify_models',side_effect=ValueError('proof failure')),patch.object(k,'adjust_effects') as adjust,self.assertRaisesRegex(ValueError,'proof failure'):run(fixture())
        adjust.assert_not_called()

    def test_fake_pass_is_not_full_gate(self):
        with patch.object(ind,'verify_models',return_value={'passed':True}),patch.object(k,'adjust_effects') as adjust,self.assertRaises(ValueError):run(fixture())
        adjust.assert_not_called()

    def test_verifier_model_mutation_rejected(self):
        verify=ind.verify_models
        def corrupt(months,year,b):
            p=verify(months,year,b);b['models'][h.ARMS[1]]['beta'][0]+=.1;return p
        with patch.object(ind,'verify_models',side_effect=corrupt),patch.object(k,'adjust_effects') as adjust,self.assertRaisesRegex(ValueError,'mutated'):run(fixture())
        adjust.assert_not_called()

    def test_source_mutation_during_model_verification_rejected(self):
        verify=ind.verify_models
        def corrupt(months,year,b):
            p=verify(months,year,b)
            for row in months:
                if 'features' in row:row['features'][0,0]+=.01;break
            return p
        with patch.object(ind,'verify_models',side_effect=corrupt),patch.object(k,'adjust_effects') as adjust,self.assertRaisesRegex(ValueError,'binding changed'):run(fixture())
        adjust.assert_not_called()

    def test_n0_partial_no_training_panel_or_model_bodies(self):
        for n in (0,1):
            args=list(fixture());args[2]=fixture_source.Poison();args[4]=fixture_source.Poison()
            with self.subTest(n=n),patch.object(k,'fit_models') as fit:
                models,got,e,_=run(args,n)
            fit.assert_not_called();self.assertEqual(models['annual'],{});self.assertEqual(e['producer_counts']['producer_ridge_solves'],0)
            h.entry_check(e,n,6)

    def test_cached_all_prefixes_no_new_fits_and_exact(self):
        models,wanted,_,source=self.full
        with patch.object(k,'fit_models') as fit,patch.object(ind,'verify_models') as verify:
            for n in range(self.args[1]['events']+1):
                _,got,e,_=run(self.args,n,cached_source=source,cached_models=models)
                h.exact_prefix(wanted,got,n);h.entry_check(e,n,6,cached=True)
        fit.assert_not_called();verify.assert_not_called()

    def test_physical_prefix_models_exact_and_full_identity(self):
        package,wanted,_,_=self.full
        for n in (0,1,15,25):
            small,got,e,_=run(self.args,n);h.exact_prefix(wanted,got,n);h.annual_prefix(package,small)
            self.assertTrue(e['model_recomputed'])

    def test_cached_future_model_not_consumed_but_current_mutation_fails(self):
        package,wanted,_,source=self.full;copy=dict(package,annual=dict(package['annual']))
        copy['annual']['2019']=fixture_source.Poison()
        _,got,_,_=run(self.args,1,cached_source=source,cached_models=copy);h.exact_prefix(wanted,got,1)
        copy=deepcopy(package);copy['annual']['2019']['models'][h.ARMS[1]]['beta'][0]+=.1
        with self.assertRaisesRegex(ValueError,'Cached consumed'):run(self.args,cached_source=source,cached_models=copy)

    def test_cached_model_proof_mutation_rejected(self):
        package,_,_,source=self.full;copy=deepcopy(package)
        copy['proofs']['2019']['svd_beta'][h.ARMS[1]]['values'][0]='0x1.0p+0'
        with self.assertRaisesRegex(ValueError,'model proof changed'):run(self.args,cached_source=source,cached_models=copy)

    def test_projection_integrity_work_is_counted(self):
        _,_,e,_=self.full;work=e['source_integrity_work']
        self.assertEqual(work['checks'],len(e['model_proofs'])+1)
        self.assertEqual(work['counts']['projected_event_records_checked'],work['checks']*self.args[1]['events'])
        bad=deepcopy(e);bad['source_integrity_work']['counts']['projected_event_records_checked']-=1
        with self.assertRaisesRegex(ValueError,'integrity projection'):h.entry_check(bad,self.args[1]['events'],6)

    def test_active_prediction_requires_svd_query_gate(self):
        verify=ind.verify_adjustment
        def weak(*args,**kwargs):
            proof=verify(*args,**kwargs);proof['independent_svd_prediction_checked']=False;return proof
        with patch.object(ind,'verify_adjustment',side_effect=weak),self.assertRaisesRegex(ValueError,'SVD prediction'):run(fixture())

    def test_prefix_shape_dtype_and_signedzero_are_exact(self):
        _,wanted,_,_=self.full
        for kind in ('signedzero','dtype','nanpayload'):
            bad=deepcopy(wanted)
            if kind=='dtype':bad[1][h.ARMS[0]]=bad[1][h.ARMS[0]].astype('float32')
            elif kind=='signedzero':
                wanted2=deepcopy(wanted);wanted2[1][h.ARMS[0]][0,0]=0.;bad=deepcopy(wanted2);bad[1][h.ARMS[0]][0,0]=-0.
            else:
                wanted2=deepcopy(wanted);wanted2[1][h.ARMS[0]][0,0]=np.array([0x7ff8000000000001],dtype=np.uint64).view(float)[0];bad=deepcopy(wanted2);bad[1][h.ARMS[0]][0,0]=np.array([0x7ff8000000000002],dtype=np.uint64).view(float)[0]
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.exact_prefix(wanted if kind=='dtype' else wanted2,bad,self.args[1]['events'])

    def test_unchanged_effects_preserve_incumbent_P_F_orders(self):
        _,got,_,_=run(self.args,15)
        for arm in h.ARMS[1:]:
            self.assertEqual(got[0][arm],got[0][h.ARMS[0]])
            h.exact_array(got[1][arm],got[1][h.ARMS[0]],'cold F');h.exact_array(got[2][arm],got[2][h.ARMS[0]],'cold P')

    def test_serialized_models_floatbits_and_rejection(self):
        value=np.array([0.,-0.,1./3.]);encoded=h.encode_models(value)
        self.assertEqual(encoded,dict(__ndarray__=True,dtype='<f8',shape=[3],values_hex=[x.hex() for x in value]))
        for bad in (np.array([1],dtype=np.int64),np.array([np.nan]),np.array([np.inf])):
            with self.assertRaises(ValueError):h.encode_models(bad)

    def test_accounting_aliases_full_no_double_count(self):
        package,wanted,main,source=self.full;total=self.args[1]['events'];cached=[];physical=[]
        for n in range(total+1):cached.append(run(self.args,n,cached_source=source,cached_models=package)[2])
        for n in (0,1,15,25):physical.append(run(self.args,n)[2])
        reuse=deepcopy(main);reuse.update(model_recomputed=False,reuses_main_verified_full=True);physical.append(reuse)
        e=dict(schema='monthly_boundary_pair267_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,source_training_unchanged=True,source_panel_unchanged=True,original_reference_unchanged=True)
        e['fit_execution_accounting']=h.execution_accounting(e)
        models=dict(schema='monthly_boundary_pair267_saved_models_v1',annual_years=main['source_audit']['consumed_years'],annual=[h.encode_models(package['annual'][str(y)]) for y in main['source_audit']['consumed_years']],producer_counts=main['producer_counts'])
        sa=dict(join=self.args[1],main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
        h.preparation_evidence(e,models,wanted[4],sa,total=total,width=6,prefixes=(0,1,15,25,total))
        a=e['fit_execution_accounting'];self.assertEqual(a['cached']['producer']['producer_ridge_solves'],0);self.assertEqual(a['physical_fresh']['executions'],4)
        bad=deepcopy(e);bad['fit_execution_accounting']['total']['producer']['producer_ridge_solves']+=2
        with self.assertRaises(ValueError):h.preparation_evidence(bad,models,wanted[4],sa,total=total,width=6,prefixes=(0,1,15,25,total))


class LifecycleTests(unittest.TestCase):
    def contract(self,c,d):
        with patch.object(h.Path,'is_file',return_value=True),patch.object(h,'sha',return_value='a'*64),patch.object(h,'read',side_effect=lambda p:c if p==h.CONTEXT else d):return h.contract_check()
    def test_final_exact_twelve_pins_and_roles(self):self.contract(*contract_fixture())
    def test_contract_missing_wrong_extra_or_typed_inventory_rejected(self):
        c,d=contract_fixture()
        for field in ('code_files','supplemental_input_files'):
            for bad in (d[field][1:],d[field]+['wrong.py'],tuple(d[field]),list(reversed(d[field]))):
                dd=deepcopy(d);dd[field]=bad
                with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):self.contract(c,dd)
    def test_each_paper_all_formal_and_reader_pins_required(self):
        for paper in (0,1):
            for name in ['work/'+h.code_names[0],*h.SUPPLEMENTAL]:
                for missing in (True,False):
                    values=list(contract_fixture());mapping=values[paper]['input_sha256']
                    if missing:mapping.pop(name)
                    else:mapping[name]='b'*64
                    with self.subTest(paper=paper,name=name,missing=missing),self.assertRaises(ValueError):self.contract(*values)
    def test_draft_or_actual_flags_prevent_registration(self):
        for key,value in [('frozen',False),('registration_authorized',False),('actual_data_started',True),('actual_models_fitted',1),('actual_models_fitted',False),('actual_orders_computed',True),('actual_returns_computed',True)]:
            c,d=contract_fixture();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.contract(c,d)
    def test_register_does_not_load_source_without_final_contract(self):
        with patch.object(h,'contract_check',side_effect=ValueError('draft')),patch.object(h.old,'local') as load,self.assertRaisesRegex(ValueError,'draft'):h.register()
        load.assert_not_called()
    def test_loader_cannot_decode_before_plan_gate(self):
        with patch.object(h,'gate',side_effect=ValueError('no plan')),patch.object(h.old,'local') as load,patch.object(h.ev,'load_arrays') as decode,self.assertRaisesRegex(ValueError,'no plan'):h.load_sources('a'*64)
        load.assert_not_called();decode.assert_not_called()
    def test_initial_counter_exact_and_current_queue(self):
        state=dict(completed=[dict(trials=4,directory='x') for _ in range(261)]+[dict(trials=3,directory=h.PRIOR.name)],current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='a')])
        state['completed'][0]['trials']+=250
        h.registration_check(state,'a');state['completed'][-1]['trials']+=1
        with self.assertRaises(ValueError):h.registration_check(state,'a')
    def test_canonical_alias_same_hash_conflict_and_central(self):
        with patch.object(h.old,'verify'):
            p='work/test.py';self.assertEqual(len(h.canonical_pins({p:'a'*64,str(h.ROOT/p):'a'*64})),1)
            for value in ({p:'a'*64,str(h.ROOT/p):'b'*64},{str(h.CENTRAL):'a'*64},{'work/../escape':'a'*64}):
                with self.assertRaises(ValueError):h.canonical_pins(value)
    def test_nested_zero_cost_no_epsilon(self):
        ex={a:{p:dict(TRADES=[dict(cost=0.)],HOLDINGS=[dict(fee=0.)]) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        rr={a:{p:dict(metrics=dict(cost_per_side=0.,fees_paid=0.),slices=dict(full122=dict(cost_per_side=0.,fees_paid=0.))) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        h.zero_cost_check(ex,rr)
        for field in ('cost_per_side','fees_paid'):
            bad=deepcopy(rr);bad[h.ARMS[2]]['same_close_zero']['metrics'][field]=np.nextafter(0.,1.)
            with self.assertRaises(ValueError):h.zero_cost_check(ex,bad)
    def test_phase_terminals_exact_artifacts_and_true_exit(self):
        rows={p:dict(session_id=1,exit_code=0,chunk_id='root',artifact_path=str(h.OUT/path),artifact_sha256='a'*64) for p,path in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]}
        value=dict(plan_sha256='p',phases=rows)
        with patch.object(h,'read',return_value=value),patch.object(h,'sha',return_value='a'*64):h.phase_terminals_check('p')
        value['phases']['prepare']['exit_code']=1
        with patch.object(h,'read',return_value=value),patch.object(h,'sha',return_value='a'*64),self.assertRaises(ValueError):h.phase_terminals_check('p')


if __name__=='__main__':unittest.main(verbosity=2)
