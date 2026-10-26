"""266 lifecycle tests use only synthetic caller-supplied source/reference data."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch
import numpy as np
import monthly_monotone_tree266_host as h
import monthly_monotone_tree266_kernel as k
import monthly_monotone_tree266_independent as ind
import monthly_monotone_tree266_reference as ref
import test_monthly_monotone_tree266_reference as fixtures


def fixture():return fixtures.f.fixture()

def original(args,n=None):return k.original_decisions(args[0],args[3],args[2]['monthly'],n)

def prepare(args,reference,n=None):
    n=len(args[2]['monthly']['dates']) if n is None else n
    old=original(args,n)
    return h.prepare_one(args[0],args[1],args[2],args[3],ind,n,old[0],old,reference=reference,historical_binding=h.reference_binding())

def contract_fixture():
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,only_promotable=h.ARMS[2],diagnostic_only=h.ARMS[1],predecessor=dict(directory=h.PRIOR.name,completion_sha256='b'*64,result_review_sha256='c'*64),input_sha256={str(p.relative_to(h.ROOT)):'a'*64 for p in (h.RUNTIME_PINS,h.RUNTIME_REVIEW)})
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,input_sha256={})
    return c,d


class GatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.reference=fixtures.reference(cls.args);cls.full=prepare(cls.args,cls.reference)

    def test_actual_three_stages_and_real_work(self):
        b,got,e=self.full;self.assertEqual(set(got[0]),set(h.ARMS));self.assertEqual(len(h.code_names),10)
        for kind,flags in [('calibration',ind.CALIBRATION_FLAGS),('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS)]:h.receipt(e[kind],4,flags,kind)
        h.fit_binding(e['calibration'],e['control'],e['models'],b['audit'],4)
        self.assertFalse(e['reference261_control']['comparison']['receipt_authentication_performed'])
        self.assertEqual(e['control_strategy']['independent']['original_episode_calls'],1)
        self.assertEqual(e['reference261_calibration']['comparison']['counts']['ridge_models_compared'],2)
        self.assertEqual(e['reference261_control']['comparison']['counts']['tree_models_compared'],2)
        self.assertEqual(e['models']['shape_certificates_checked'],12)
        for role in h.ROLES:self.assertIn(role,b)

    def test_no_legacy_ridge_as_economic_control(self):
        b,got,e=self.full
        self.assertEqual(got[0][h.ARMS[1]],self.reference['orders']['weak_pooled_tree'])
        self.assertEqual(b['unconstrained_control']['raw'].tobytes(),self.reference['forecasts']['candidate_raw'].tobytes())
        self.assertEqual(e['reference261_control']['comparison']['reference_role'],'candidate')

    def test_raw_reference_failure_prevents_all_trees(self):
        bad=deepcopy(self.reference);bad['forecasts']['control_raw'][6]+=.01
        with patch.object(k,'_fit_tree') as fits,self.assertRaises(ValueError):prepare(self.args,bad)
        fits.assert_not_called()

    def test_control_model_failure_prevents_candidate(self):
        bad=deepcopy(self.reference);row=next(x for x in bad['model_audit']['candidate']['annual'] if x['status']=='fit');row['tree']['parameters']['max_depth']=1
        calls=[];fit=k._fit_tree
        def record(*args):calls.append(args[-1]);return fit(*args)
        with patch.object(k,'_fit_tree',side_effect=record),self.assertRaises(ValueError):prepare(self.args,bad)
        self.assertEqual(calls,['unconstrained_control']*2)

    def test_control_score_or_order_failure_prevents_candidate(self):
        for field in ('scores','pre_scores','orders'):
            bad=deepcopy(self.reference)
            if field=='orders':bad[field]['weak_pooled_tree'][1]['symbols']=['WRONG']
            else:bad[field]['weak_pooled_tree'][1,0]+=.01
            calls=[];fit=k._fit_tree
            def record(*args):calls.append(args[-1]);return fit(*args)
            with self.subTest(field=field),patch.object(k,'_fit_tree',side_effect=record),self.assertRaises(ValueError):prepare(self.args,bad)
            self.assertNotIn('monotone_candidate',calls)

    def test_early_original_failure_prevents_model_work(self):
        bad=deepcopy(self.reference);bad['pre_scores']['incumbent_local67'][1,0]+=.01
        with patch.object(k,'fit_models') as fits,self.assertRaises(ValueError):prepare(self.args,bad)
        fits.assert_not_called()

    def test_full_independent_failure_prevents_any_candidate_decision(self):
        with patch.object(ind,'verify_models',side_effect=ValueError('full failed')),patch.object(k,'decisions') as decide,self.assertRaisesRegex(ValueError,'full failed'):prepare(self.args,self.reference)
        decide.assert_not_called()

    def test_model_receipt_mutation_and_fake_pass_rejected(self):
        for kind in ('calibration','control','models'):
            bad=deepcopy(self.full[2][kind]);bad['manual_tree_traversals']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.receipt(bad,4,(),kind)
        with patch.object(ind,'verify_calibration_models',return_value={'passed':True}),patch.object(k,'_fit_tree') as fits,self.assertRaises(ValueError):prepare(self.args,self.reference)
        fits.assert_not_called()

    def test_independent_gate_cannot_mutate_source(self):
        args=fixtures.f.clone(self.args);verify=ind.verify_calibration_models
        def corrupt(*a,**kw):
            result=verify(*a,**kw);args[3]['baseline'][0,0]+=.01;return result
        with patch.object(ind,'verify_calibration_models',side_effect=corrupt),patch.object(k,'_fit_tree') as fits,self.assertRaises(ValueError):prepare(args,self.reference)
        fits.assert_not_called()

    def test_reference_binding_exact_full_inventory(self):
        bad=h.reference_binding();bad['files_sha256'].pop(next(iter(bad['files_sha256'])))
        old=original(self.args)
        with patch.object(k,'fit_models') as fits,self.assertRaises(ValueError):h.prepare_one(self.args[0],self.args[1],self.args[2],self.args[3],ind,4,old[0],old,reference=self.reference,historical_binding=bad)
        fits.assert_not_called()

    def test_all_small_cached_prefixes_fit_zero(self):
        bundle,wanted,_=self.full
        with patch.object(k,'fit_models',side_effect=AssertionError('cached fit')),patch.object(k,'manual_predict',side_effect=AssertionError('cached traversal')):
            for n in range(5):
                aa=h.old.cache_prefix(self.args[3],n);mm=h.old.meta_prefix(self.args[2]['monthly'],n);cb,proof=h.incumbent_callback(ind,aa,mm,n,original(self.args,n)[0])
                got=k.decisions(self.args[0],aa,mm,bundle,n,incumbent_validator=cb,expected_incumbent_orders=original(self.args,n)[0]);h.exact_prefix(wanted,got,n)
                check=ind.verify_selection(aa,mm,bundle,got[0],scores=got[1],pre_scores=got[2],audit=got[3],n=n);h.receipt(check,n,ind.SELECTION_FLAGS,'selection',width=6)

    def test_physical_prefixes_and_partial_no_numeric_training(self):
        for n in range(5):
            train,meta,arrays,scope=h.physical_inputs(self.args[1],self.args[2],self.args[3],n)
            args=(self.args[0],train,meta,arrays)
            small,got,e=prepare(args,self.reference,n);h.bundle_parity(self.full[0],small,n);h.exact_prefix(self.full[1],got,n)
            h.physical_scope_check(scope,n,small['audit']['counts'])
            if n<=1:self.assertEqual(train,{});self.assertEqual(small['audit']['counts']['actual_model_fits'],0)

    def test_saved_layout_digest_reconstructable(self):
        for stage in ('calibration','control'):
            env=self.full[2]['reference261_'+stage];U=np.zeros((4,6),dtype=bool)
            for i,ids in enumerate(env['original_u_indices']):U[i,ids]=True
            lay=dict(n=4,width=6,monthly=h.old.meta_prefix(self.args[2]['monthly'],4),original_u=U,annual_cohorts=env['annual_cohorts'])
            self.assertEqual(h.structured_digest(lay),env['source_layout_sha256']);h.reference_receipt(env,4,6,stage,h.reference_binding())

    def test_prefix_bits_signed_zero_dtype_nan_payload(self):
        b=deepcopy(self.full[0]);p=deepcopy(b)
        for role in h.ROLES:
            b[role]['raw'][0]=0.;p[role]['raw'][0]=-0.
        with self.assertRaises(ValueError):h.bundle_parity(b,p,4)
        p=deepcopy(b);p['calibration_ridge']['rank']=p['calibration_ridge']['rank'].astype(np.float32)
        with self.assertRaises(ValueError):h.bundle_parity(b,p,4)
        a=np.array([0x7ff8000000000001],dtype=np.uint64).view(np.float64);z=np.array([0x7ff8000000000002],dtype=np.uint64).view(np.float64)
        with self.assertRaises(ValueError):h.exact_array(a,z,'NaN payload')
        wanted=deepcopy(self.full[1]);got=deepcopy(wanted);wanted[1][h.ARMS[0]][0,0]=0.;got[1][h.ARMS[0]][0,0]=-0.
        with self.assertRaises(ValueError):h.exact_prefix(wanted,got,4)

    def test_label_prefix_reads_only_selected_global_ids(self):
        class Poison:
            def __getitem__(self,key):raise AssertionError('empty prefix read full labels')
            def __iter__(self):raise AssertionError('iterated future labels')
            def __array__(self,*args,**kwargs):raise AssertionError('coerced future labels')
        for n in (0,1):
            full=deepcopy(self.full[0]);part=deepcopy(full)
            for role in h.ROLES:
                for key in ('raw','available'):part[role][key]=part[role][key][:n*6]
                part[role]['rank']=part[role]['rank'][:n]
            part['training_row_ids']=np.empty(0,dtype=full['training_row_ids'].dtype);part['label_values']=np.empty(0,dtype=np.float64);full['label_values']=Poison()
            h.bundle_parity(full,part,n)
        full=deepcopy(self.full[0]);part=deepcopy(full);indices=np.array([0,2],dtype=np.int64);values=full['label_values'][indices].copy()
        class SelectedOnly(Poison):
            def __getitem__(self,key):
                if not isinstance(key,np.ndarray) or key.dtype!=np.dtype('int64') or key.tolist()!=[0,2]:raise AssertionError('read outside selected labels')
                return values.copy()
        part['training_row_ids']=full['training_row_ids'][indices].copy();part['label_values']=values.copy();full['label_values']=SelectedOnly();h.bundle_parity(full,part,4)
        part['label_values'][0]+=1
        with self.assertRaises(ValueError):h.bundle_parity(full,part,4)

    def test_full_independent_source_mutation_stops_selection(self):
        args=fixtures.f.clone(self.args);verify=ind.verify_models
        def corrupt(*a,**kw):
            result=verify(*a,**kw);args[3]['baseline'][0,0]+=.01;return result
        with patch.object(ind,'verify_models',side_effect=corrupt),patch.object(k,'decisions') as decide,self.assertRaisesRegex(ValueError,'source mutation before decisions'):prepare(args,self.reference)
        decide.assert_not_called()

    def test_model_stage_counts_cannot_hide_candidate_work(self):
        for kind in ('calibration','control','models'):
            bad=deepcopy(self.full[2][kind]);bad['counts']['monotone_tree_fits']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.receipt(bad,4,(),kind)
        for key in ('independent_canonical_solves','independent_svd_solves','shape_certificates_checked','original_episode_calls'):
            bad=deepcopy(self.full[2]['models']);bad[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.receipt(bad,4,(), 'models')


class LifecycleTests(unittest.TestCase):
    def test_fixed_files_source_and_counter_contract(self):
        self.assertEqual((len(h.code_names),len(h.PREP_FILES),len(h.RESULT_FILES),len(h.REFERENCE_SHA),len(h.WORK_FIELDS)),(10,10,38,13,13));self.assertEqual(h.ROLES,('calibration_ridge','unconstrained_control','monotone_candidate'))
        self.assertEqual(h.PRIOR.name,'monthly_gain_concentration265');self.assertEqual(h.PREFIXES,(0,1,53,88,122))
        s=dict(completed=[dict(directory='past',trials=5) for _ in range(260)]+[dict(directory=h.PRIOR.name,trials=2)])
        s['completed'][0]['trials']=1;s['completed'][1]['trials']=1
        h.initial_state(s);s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')]);h.registration_check(s,'p')
        s['completed'][-1]['trials']=3
        with self.assertRaises(ValueError):h.initial_state(s)

    def test_draft_or_missing_predecessor_cannot_register(self):
        c,d=contract_fixture()
        for branch,key,value in [('context','frozen',False),('contract','registration_authorized',False),('context','actual_data_started',True),('context','actual_models_fitted',1)]:
            cc,dd=deepcopy(c),deepcopy(d);(cc if branch=='context' else dd)[key]=value
            with self.subTest(key=key),patch.object(Path,'is_file',return_value=True),patch.object(h,'read',side_effect=lambda p:cc if p==h.CONTEXT else dd),patch.object(h,'sha',side_effect=lambda p:h.SOURCE_REVIEW_SHA if p==h.SOURCE_REVIEW else 'a'*64),self.assertRaises(ValueError):h.contract_check()
        c['predecessor']['completion_sha256']=None
        with patch.object(Path,'is_file',return_value=True),patch.object(h,'read',side_effect=lambda p:c if p==h.CONTEXT else d),patch.object(h,'sha',return_value='a'*64),self.assertRaisesRegex(ValueError,'final SHA256'):h.contract_check()
        with patch.object(h,'contract_check',side_effect=ValueError('unfrozen')),patch.object(h.ev,'load_arrays') as load,self.assertRaises(ValueError):h.register()
        load.assert_not_called()

    def test_prepare_gate_before_any_actual_load(self):
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.old,'local') as local,patch.object(h.ev,'load_arrays') as load,self.assertRaises(ValueError):h.prepare('bad')
        local.assert_not_called();load.assert_not_called()

    def test_historical_manifest_requires_all_thirteen(self):
        pp=dict(status='COMPLETE',events=122,new_returns_computed=False,plan_sha256=h.REFERENCE_SHA[h.REFERENCE/'PLAN.json'],artifacts_sha256={n:h.REFERENCE_SHA[h.REFERENCE/'prepared'/n] for n in h.PREP_FILES})
        cc=dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',protected_files=21444,plan_sha256=pp['plan_sha256'],model_audit_sha256=pp['artifacts_sha256']['MODEL_AUDIT.json'],independent_prepare_sha256=pp['artifacts_sha256']['INDEPENDENT_PREPARE.json'])
        with patch.object(h,'sha',side_effect=lambda p:h.REFERENCE_SHA[p]),patch.object(h,'read',side_effect=lambda p:cc if p==h.REFERENCE/'COMPLETION.json' else pp):
            self.assertEqual(h.reference_closure_check({str(k):v for k,v in h.REFERENCE_SHA.items()}),h.reference_binding())
            for dropped in h.REFERENCE_SHA:
                pins={str(k):v for k,v in h.REFERENCE_SHA.items()};pins.pop(str(dropped))
                with self.subTest(path=dropped.name),self.assertRaises(ValueError):h.reference_closure_check(pins)
            pp['artifacts_sha256'].pop('FORECASTS.npz')
            with self.assertRaises(ValueError):h.reference_closure_check()

    def test_phase_terminal_three_owned_exitzero_bindings(self):
        rels={'prepare':'prepared/COMPLETION.json','select':'selected/FROZEN.json','diagnose':'results/COMPLETION.json'}
        term=dict(plan_sha256='plan',phases={k:dict(session_id=i+1,exit_code=0,chunk_id='chunk',artifact_path=str(h.OUT/v),artifact_sha256='digest') for i,(k,v) in enumerate(rels.items())})
        with patch.object(h,'read',return_value=term),patch.object(h,'sha',return_value='digest'):h.phase_terminals_check('plan')
        for key,value in [('exit_code',False),('exit_code',1),('chunk_id',''),('artifact_sha256','wrong'),('artifact_path',str(h.OUT/'RESULTS.json'))]:
            bad=deepcopy(term);bad['phases']['prepare'][key]=value
            with self.subTest(key=key),patch.object(h,'read',return_value=bad),patch.object(h,'sha',return_value='digest'),self.assertRaises(ValueError):h.phase_terminals_check('plan')

    def test_zero_cost_all_six_streams_including_config(self):
        exports={a:{p:dict(TRADES=[{'cost':0.}],HOLDINGS=[{'fee':0.}]) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        results={a:{p:dict(cost_per_side=0.,metrics={'fees_paid':0.},slices={'full122':{'fees_paid':0.}}) for p in exports[a]} for a in h.ARMS};h.zero_cost_check(exports,results)
        for arm in h.ARMS:
            for path in exports[arm]:
                bad=deepcopy(results);bad[arm][path]['cost_per_side']=1e-30
                with self.assertRaises(ValueError):h.zero_cost_check(exports,bad)
                for field,key in [('TRADES','cost'),('HOLDINGS','fee')]:
                    bad=deepcopy(exports);bad[arm][path][field][0][key]=True
                    with self.assertRaises(ValueError):h.zero_cost_check(bad,results)

    def test_nine_paths_fortyfive_slices_eighteen_training_groups(self):
        q=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices=dict.fromkeys(('full122','first88','early53','recent35','later34'),{})) for p in h.ev.PATHS} for a in h.ARMS})
        h.outcome_check(q)
        for key,value in [('checked_paths',6),('checked_slices',30),('checked_training_groups',12),('terminal_liquidation',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):h.outcome_check(dict(q,**{key:value}))

    def test_dual_reference_admission_delegated_without_ridge_alias(self):
        from test_monthly_monotone_tree266_admission import training
        m,o=training();self.assertTrue(h.select_first88(m,o)['admitted'])
        o[h.ARMS[1]]=deepcopy(o[h.ARMS[2]]);self.assertFalse(h.select_first88(m,o)['admitted'])


if __name__=='__main__':unittest.main(verbosity=2)
