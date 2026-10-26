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

def verified_reference(reference):
    # Inject only synthetic artifacts into the real loading/issuer seam. This
    # does not purport to verify actual saved261 bytes or authorize a real run.
    names={'FORECASTS.npz':'forecasts','MODEL_AUDIT.json':'model_audit','LABELS.npz':'labels','SCORES.npz':'scores','PRE_SCORES.npz':'pre_scores','ORDERS.json':'orders','META.json':'meta'}
    with patch.object(h,'reference_closure_check',return_value=h.reference_binding()),patch.object(h.ev,'load_arrays',side_effect=lambda p:reference[names[p.name]]),patch.object(h,'read',side_effect=lambda p:reference[names[p.name]]):return h.load_reference(h.reference_binding())

def prepare(args,reference,n=None,reference_proof=None):
    n=len(args[2]['monthly']['dates']) if n is None else n
    if reference_proof is None:reference,reference_proof=verified_reference(reference)
    old=original(args,n)
    return h.prepare_one(args[0],args[1],args[2],args[3],ind,n,old[0],old,reference=reference,historical_binding=h.reference_binding(),reference_proof=reference_proof)

def contract_fixture():
    c=dict(frozen=True,registration_authorized=True,actual_data_started=False,actual_models_fitted=0,actual_orders_computed=False,actual_returns_computed=False,arms=list(h.ARMS),registered_trials=3,only_promotable=h.ARMS[2],diagnostic_only=h.ARMS[1],predecessor=dict(directory=h.PRIOR.name,completion_sha256='b'*64,result_review_sha256='c'*64),input_sha256={str(p.relative_to(h.ROOT)):'a'*64 for p in (h.RUNTIME_PINS,h.RUNTIME_REVIEW)})
    supplemental=['work/monthly_monotone_tree266_saved_reader.py','work/test_monthly_monotone_tree266_saved_reader.py']
    c['input_sha256'].update({p:'a'*64 for p in supplemental})
    d=dict(frozen=True,registration_authorized=True,actual_data_started=False,context_sha256='a'*64,code_files=['work/'+name for name in h.code_names],supplemental_input_files=supplemental,input_sha256={p:'a'*64 for p in supplemental})
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
        with patch.object(k,'fit_models') as fits,self.assertRaises(ValueError):h.prepare_one(self.args[0],self.args[1],self.args[2],self.args[3],ind,4,old[0],old,reference=self.reference,historical_binding=bad,reference_proof=None)
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
        for field in ('array','identity'):
            args=fixtures.f.clone(self.args);verify=ind.verify_models
            def corrupt(*a,**kw):
                result=verify(*a,**kw)
                if field=='array':args[3]['baseline'][0,0]+=.01
                else:args[2]['monthly']['etfs'].append('MUTATION')
                return result
            with self.subTest(field=field),patch.object(ind,'verify_models',side_effect=corrupt),patch.object(k,'decisions') as decide,self.assertRaisesRegex(ValueError,'source mutation before decisions'):prepare(args,self.reference)
            decide.assert_not_called()

    def test_all123_work_sums_four_physical_and_main122_once(self):
        # Counter arithmetic fixtures, not claims of executed122-event models.
        train,meta,arrays,_=h.physical_inputs(self.args[1],self.args[2],self.args[3],0)
        empty=prepare((self.args[0],train,meta,arrays),self.reference,0)[2]
        def retag(n):
            q=deepcopy(empty if n<=1 else self.full[2])
            for key in ('calibration','control','models'):
                q[key]['counts']['events']=n;q[key]['producer_equivalent_counts']['events']=n
            q['main_model_counts']['counts']['events']=n
            return q
        e=retag(122);e['physical_prefixes']=[]
        for n in h.PREFIXES:
            q=retag(n);q.update(prefix=n,model=q.pop('models'),model_counts=q.pop('main_model_counts'));e['physical_prefixes'].append(q)
        e['cached_prefixes']=[dict(prefix=n,incumbent=deepcopy(empty['incumbent']),selection=deepcopy(empty['selection'])) for n in range(123)]
        totals=h.execution_accounting(e);self.assertEqual(totals['model_invocations'],5);self.assertTrue(totals['physical122_reuses_main'])
        self.assertEqual([v['events'] for v in totals['entries']],[122,0,1,53,88])
        self.assertEqual(totals['cached_decisions']['invocations'],123);self.assertEqual(totals['cached_decisions']['model_fit_calls'],0)
        self.assertEqual(totals['totals']['producer_episode_calls'],399);self.assertEqual(totals['totals']['original_episode_calls'],527)
        self.assertEqual(totals['totals']['producer_matched_ridge_fits'],6);self.assertEqual(totals['totals']['producer_unconstrained_tree_fits'],6);self.assertEqual(totals['totals']['producer_monotone_tree_fits'],6)
        self.assertEqual(totals['totals']['independent_canonical_solves'],18);self.assertEqual(totals['totals']['independent_svd_solves'],18);self.assertEqual(totals['totals']['independent_tree_refits'],18);self.assertEqual(totals['totals']['shape_certificates_checked'],54)
        self.assertEqual(totals['totals']['total_model_fit_calls'],72)
        for v in totals['entries'][1:3]:self.assertEqual(v['total_model_fit_calls'],0)
        # A dishonest122 reuse entry cannot inflate the summed work.
        e['physical_prefixes'][-1]['model']['independent_canonical_solves']=999999
        self.assertEqual(totals,h.execution_accounting(e))

    def test_issued_reference_proof_required_and_bound_to_object(self):
        source,proof=verified_reference(deepcopy(self.reference));old=original(self.args)
        common=dict(reference=source,historical_binding=h.reference_binding())
        with self.assertRaises(TypeError):h.prepare_one(self.args[0],self.args[1],self.args[2],self.args[3],ind,4,old[0],old,**common)
        for bad in (None,{},object()):
            with patch.object(k,'fit_models') as fit,self.assertRaisesRegex(ValueError,'Host-issued'):h.prepare_one(self.args[0],self.args[1],self.args[2],self.args[3],ind,4,old[0],old,reference_proof=bad,**common)
            fit.assert_not_called()
        with self.assertRaisesRegex(ValueError,'Same verified historical object'):prepare(self.args,deepcopy(source),reference_proof=proof)
        source['scores'][h.ARMS[0]][1,0]+=.01
        with patch.object(k,'fit_models') as fit,self.assertRaisesRegex(ValueError,'historical original reference mutated'):prepare(self.args,source,reference_proof=proof)
        fit.assert_not_called()

    def test_calibration_helper_reference_mutation_stops_all_trees(self):
        source=deepcopy(self.reference);actual=ref.compare_calibration_reference
        def corrupt(reference,*a,**kw):
            result=actual(reference,*a,**kw);reference['forecasts']['control_raw'][6]+=.01;return result
        with patch.object(ref,'compare_calibration_reference',side_effect=corrupt),patch.object(k,'_fit_tree') as fit,self.assertRaisesRegex(ValueError,'historical consumed reference mutated:calibration'):prepare(self.args,source)
        fit.assert_not_called()

    def test_control_helper_reference_mutation_stops_candidate(self):
        source=deepcopy(self.reference);actual=ref.compare_control_reference;fit=k._fit_tree;roles=[]
        def corrupt(reference,*a,**kw):
            result=actual(reference,*a,**kw);reference['pre_scores']['weak_pooled_tree'][1,0]+=.01;return result
        def record(*a):roles.append(a[-1]);return fit(*a)
        with patch.object(ref,'compare_control_reference',side_effect=corrupt),patch.object(k,'_fit_tree',side_effect=record),self.assertRaisesRegex(ValueError,'historical consumed reference mutated:control'):prepare(self.args,source)
        self.assertEqual(roles,['unconstrained_control']*2)

    def test_reference_cap_prefix_never_consumes_future_numeric_bodies(self):
        class Poison:
            def __deepcopy__(self,memo):return self  # The admitted full-load copy is explicit.
            def __float__(self):raise AssertionError('future float consumed')
            def __iter__(self):raise AssertionError('future body consumed')
            def __repr__(self):raise AssertionError('future numeric hashed')
            def __len__(self):raise AssertionError('future numeric inspected')
        class PrefixOnly(np.ndarray):
            def __new__(cls,value,limit):
                out=value.view(cls);out.limit=limit;return out
            def __deepcopy__(self,memo):return self
            def __getitem__(self,key):
                if not isinstance(key,slice) or key.start not in (None,0) or key.stop is None or key.stop>self.limit:raise AssertionError('future score numeric sliced')
                return np.ndarray.__getitem__(self,key).view(np.ndarray)
            def tobytes(self,*a,**kw):raise AssertionError('full future score numeric hashed')
            def __iter__(self):raise AssertionError('full future score numeric iterated')
        for n in (0,1,2):
            source=deepcopy(self.reference);train,meta,arrays,_=h.physical_inputs(self.args[1],self.args[2],self.args[3],n)
            for key,value in source['forecasts'].items():
                if key.endswith('available'):continue
                value=value.astype(object);value[n if key.endswith('rank') else n*6:]=Poison();source['forecasts'][key]=value
            for field in ('scores','pre_scores'):
                for arm,value in source[field].items():
                    source[field][arm]=PrefixOnly(value,n)
            for arm in ref.ARMS:source['orders'][arm][n:]=[Poison() for _ in source['orders'][arm][n:]]
            for key in ('dates','ends','symbols'):source['meta'][key][n:]=[Poison() for _ in source['meta'][key][n:]]
            if n<=1:
                source['labels']={key:Poison() for key in source['labels']}
                for role in ('control','candidate'):source['model_audit'][role]['annual']=Poison()
            else:
                size=len(train['target_log_excess']);values=source['labels']['label_values'].astype(object);values[size:]=Poison();source['labels']['label_values']=values
                for role in ('control','candidate'):
                    row=source['model_audit'][role]['annual'][1]
                    for key in row:
                        if key!='year':row[key]=Poison()
            with self.subTest(prefix=n):prepare((self.args[0],train,meta,arrays),source,n)

    def test_reference_cap_noquery_year_does_not_read_later_full_fit(self):
        class Poison:
            def __deepcopy__(self,memo):return self
            def __repr__(self):raise AssertionError('noquery model consumed')
            def __iter__(self):raise AssertionError('noquery body consumed')
            def __len__(self):raise AssertionError('noquery body inspected')
        args=fixtures.f.clone(self.args);args[3]['eligible'][2]=False;fixtures.f.refresh(args);source=fixtures.reference(args)
        for role in ('control','candidate'):
            row=source['model_audit'][role]['annual'][1];self.assertEqual(row['status'],'fit')
            for key in ('weights','mean','scale','beta','intercept','gram','rhs','tree','raw_predictions','manual_predictions'):
                if key in row:row[key]=Poison()
        train,meta,arrays,_=h.physical_inputs(args[1],args[2],args[3],3)
        bundle,_,_=prepare((args[0],train,meta,arrays),source,3)
        self.assertEqual(bundle['calibration_ridge']['audit']['annual'][1]['status'],'no_prediction_rows')

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

    def test_contract_exact_ordered_code_and_supplemental_inventories(self):
        def check(c,d):
            with patch.object(Path,'is_file',return_value=True),patch.object(h,'read',side_effect=lambda p:c if p==h.CONTEXT else d),patch.object(h,'sha',side_effect=lambda p:h.SOURCE_REVIEW_SHA if p==h.SOURCE_REVIEW else 'a'*64):return h.contract_check()
        c,d=contract_fixture();self.assertEqual(check(c,d),(c,d))
        for key in ('code_files','supplemental_input_files'):
            for value in (None,{},tuple(d[key]),'not-list',d[key][:-1],d[key]+['work/unreviewed.py'],list(reversed(d[key])),d[key]+[d[key][0]], [True]+d[key][1:]):
                cc,dd=deepcopy(c),deepcopy(d);dd[key]=value
                with self.subTest(field=key,value_type=type(value).__name__),self.assertRaises(ValueError):check(cc,dd)
            dd=deepcopy(d);del dd[key]
            with self.assertRaises(ValueError):check(c,dd)

    def test_both_papers_must_pin_current_reader_and_test_bytes(self):
        c,d=contract_fixture()
        def check(cc,dd):
            with patch.object(Path,'is_file',return_value=True),patch.object(h,'read',side_effect=lambda p:cc if p==h.CONTEXT else dd),patch.object(h,'sha',side_effect=lambda p:h.SOURCE_REVIEW_SHA if p==h.SOURCE_REVIEW else 'a'*64):return h.contract_check()
        for paper in ('context','contract'):
            for relative in d['supplemental_input_files']:
                for value in (None,'b'*64,'',True,0,[],{'sha256':'a'*64}):
                    cc,dd=deepcopy(c),deepcopy(d);pins=(cc if paper=='context' else dd)['input_sha256']
                    if value is None:pins.pop(relative)
                    else:pins[relative]=value
                    with self.subTest(paper=paper,path=relative,type=type(value).__name__),self.assertRaises(ValueError):check(cc,dd)
            cc,dd=deepcopy(c),deepcopy(d);(cc if paper=='context' else dd)['input_sha256']=[]
            with self.assertRaises(ValueError):check(cc,dd)
        for missing in d['supplemental_input_files']:
            with patch.object(Path,'is_file',lambda p:p!=h.ROOT/missing),patch.object(h,'read',side_effect=lambda p:c if p==h.CONTEXT else d),patch.object(h,'sha',return_value='a'*64),self.assertRaisesRegex(ValueError,'current saved reader source binding'):h.contract_check()

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
        results={a:{p:dict(metrics={'fees_paid':0.,'cost_per_side':0.},slices={'full122':{'fees_paid':0.,'cost_per_side':0.}}) for p in exports[a]} for a in h.ARMS};h.zero_cost_check(exports,results)
        for arm in h.ARMS:
            for path in exports[arm]:
                bad=deepcopy(results);bad[arm][path]['metrics']['cost_per_side']=1e-30
                with self.assertRaises(ValueError):h.zero_cost_check(exports,bad)
                bad=deepcopy(results);bad[arm][path]['slices']['full122']['cost_per_side']=1e-30
                with self.assertRaises(ValueError):h.zero_cost_check(exports,bad)
                bad=deepcopy(results);bad[arm][path]['metrics']['cost_per_side']=True
                with self.assertRaises(ValueError):h.zero_cost_check(exports,bad)
                for field,key in [('TRADES','cost'),('HOLDINGS','fee')]:
                    bad=deepcopy(exports);bad[arm][path][field][0][key]=True
                    with self.assertRaises(ValueError):h.zero_cost_check(bad,results)

    def test_nine_paths_fortyfive_slices_eighteen_training_groups(self):
        q=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices=dict.fromkeys(('full122','first88','early53','recent35','later34'),{})) for p in h.ev.PATHS} for a in h.ARMS})
        h.outcome_check(q)
        for key,value in [('checked_paths',6),('checked_slices',30),('checked_training_groups',12),('terminal_liquidation',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):h.outcome_check(dict(q,**{key:value}))

    def test_saved_review_covers_every_artifact_and_terminals(self):
        paths=[h.PHASE_TERMINALS,h.OUT/'PLAN.json',h.CONTEXT,h.CONTRACT,h.OUT/'prepared/COMPLETION.json',h.OUT/'selected/FROZEN.json',h.OUT/'selected/TRAIN_RESULTS.json',h.OUT/'results/COMPLETION.json']
        paths += [h.OUT/'prepared'/n for n in h.PREP_FILES]+[h.OUT/'results'/n for n in h.RESULT_FILES]
        review=dict(passed=True,unresolved_blockers=[],plan_sha256='digest',context_sha256='digest',prepared_completion_sha256='digest',results_completion_sha256='digest',implementation_contract_sha256='digest',input_sha256={str(p):'digest' for p in paths})
        with patch.object(h,'phase_terminals_check') as phase,patch.object(h,'read',return_value=review),patch.object(h,'sha',return_value='digest'),patch.object(h.old,'verify'):
            h.result_review_check('digest');phase.assert_called_once_with('digest')
            for path in paths:
                bad=deepcopy(review);bad['input_sha256'].pop(str(path))
                with self.subTest(path=path.name),patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.result_review_check('digest')

    def test_source_closure_all12_and_complete_chain(self):
        pins,records=closure_fixture()
        def check(p,r):
            with patch.object(h,'read',side_effect=lambda x:r[x]),patch.object(h,'sha',side_effect=lambda x:h.SOURCE_CHAIN169[x]):return h.source_closure_check(p)
        result=check(pins,records);self.assertFalse(result['npz_decoded']);self.assertFalse(result['source_cache_rebuilt']);self.assertEqual(set(result['artifacts_sha256']),h.SOURCE_FILES169)
        for path in pins:
            bad=dict(pins);bad.pop(path)
            with self.subTest(path=path),self.assertRaises(ValueError):check(bad,records)
        changes=[(h.REBUILD/'PARITY.json','source_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/PLAN.json','rebuild_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/results/COMPLETION.json','evaluation_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/results/COMPLETION.json','exact_control_rechecked',False),(h.REBUILD.parent/'independent_review/COMPLETION.json','status','PENDING'),(h.REBUILD.parent/'independent_review/COMPLETION.json','evaluation_completion_sha256','wrong')]
        for path,key,value in changes:
            bad=deepcopy(records);bad[path][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check(pins,bad)
        bad=deepcopy(records);bad[h.REBUILD/'PARITY.json']['artifacts_sha256'].pop('BASE_DATA.npz')
        with self.assertRaises(ValueError):check(pins,bad)

    def test_dual_reference_admission_delegated_without_ridge_alias(self):
        from test_monthly_monotone_tree266_admission import training
        m,o=training();self.assertTrue(h.select_first88(m,o)['admitted'])
        o[h.ARMS[1]]=deepcopy(o[h.ARMS[2]]);self.assertFalse(h.select_first88(m,o)['admitted'])


def closure_fixture():
    pins={str(p):v for p,v in h.SOURCE_CHAIN169.items()};artifacts={name:('a'*64) for name in h.SOURCE_FILES169}
    pins.update({str(h.REBUILD/name):v for name,v in artifacts.items()});pins[str(h.ROOT/'work/synthetic_dependency.py')]='b'*64
    bp=h.REBUILD.parent/'PLAN.json';pp=h.REBUILD/'PARITY.json';ep=h.REBUILD.parent/'evaluation/PLAN.json';ec=h.REBUILD.parent/'evaluation/results/COMPLETION.json';ic=h.REBUILD.parent/'independent_review/COMPLETION.json'
    records={bp:dict(input_sha256={'work/synthetic_dependency.py':'b'*64}),pp:dict(passed=True,arm='control',returns_computed=False,source_plan_sha256=pins[str(bp)],artifacts_sha256=artifacts),ep:dict(rebuild_plan_sha256=pins[str(bp)],input_sha256={str(p.relative_to(h.ROOT)):pins[str(p)] for p in [h.REBUILD/n for n in h.SOURCE_FILES169]+[bp,pp]}),ec:dict(status='COMPLETE',evaluation_plan_sha256=pins[str(ep)],rebuild_plan_sha256=pins[str(bp)],exact_control_rechecked=True,source_hashes_verified_before_after=True),ic:dict(status='PASS',checked_paths=6,checked_slices=30,evaluation_completion_sha256=pins[str(ec)])}
    return pins,records


if __name__=='__main__':unittest.main(verbosity=2)
