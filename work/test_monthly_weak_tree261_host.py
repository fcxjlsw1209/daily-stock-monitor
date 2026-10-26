"""Synthetic lifecycle/receipt tests; no market arrays or saved outcomes are read."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import monthly_weak_tree261_host as h


def training():
    base=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(base) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['A','B','D']
    return m,o


def terminals():
    names=(('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json'))
    return dict(plan_sha256='plan',phases={p:dict(session_id=i+1,exit_code=0,chunk_id='chunk'+str(i),artifact_path=str(h.OUT/f),artifact_sha256='digest') for i,(p,f) in enumerate(names)})


def sources():
    plan=dict(original169_source_closure={'synthetic':True},raw_source_hashes={'synthetic':'a'},input_sha256={str(h.REBUILD/name):name+' hash' for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')})
    s=dict(original169_source_closure=copy.deepcopy(plan['original169_source_closure']),raw_source_hashes=copy.deepcopy(plan['raw_source_hashes']),fresh_training_array_reconstruction=False,cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_other_target_fits=0,original_local_fits=0,matched_ridge_calibration_refits_counted=True)
    for k,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:s[k]=plan['input_sha256'][str(h.REBUILD/name)]
    return s,plan


class LifecycleTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','matched_pooled_ridge','weak_pooled_tree'))
        self.assertEqual(h.ROLES,('control','candidate'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(10,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_leader_utility260')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))
        self.assertEqual(h.SOURCE_REVIEW.name,'after260_gbdt_slot_source_review.json')
    def test_tree_is_only_promotable(self):
        m,o=training();self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[2])
        m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.4
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_inc_close_gain_open_noninferiority(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.499)]:
            m,o=training();m[h.ARMS[2]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_matched_must_exact_incumbent_not_competing_reference(self):
        m,o=training();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.9
        with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();o[h.ARMS[1]][1]['symbols']=['A','B','D']
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_incumbent_risk_and_segment_guards(self):
        for p in ('same_close_zero','next_open_zero'):
            for group,key,value in [('training','sharpe',1.8),('training','max_drawdown',-.2),('early','cagr',.5),('recent','cagr',.7)]:
                m,o=training()
                for ref in h.ARMS[:2]:m[ref][p][group][key]=value
                self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_inc_support_three_across_eras(self):
        for ids in ((1,54),(1,2,3),(53,54,55)):
            m,o=training();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[2]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_same_members_different_order_is_not_support(self):
        m,o=training()
        for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['C','A','B']
        self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_main_cagr_only_roundoff(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.505-2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])
        for group,key,value in [('training','sharpe',1.65-5e-13),('early','cagr',.375-5e-13)]:
            m,o=training();m[h.ARMS[2]]['same_close_zero'][group][key]=value
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_undefined_sharpe_policy(self):
        for a in h.ARMS:
            for p in ('same_close_zero','next_open_zero'):
                m,o=training();m[a][p]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
                if a!=h.ARMS[2]:
                    with self.assertRaises(ValueError):h.select_first88(m,o)
                else:self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_bad_metrics_fail_not_unavailability(self):
        for a in h.ARMS:
            for bad in (True,'0.8',float('nan'),float('inf')):
                m,o=training();m[a]['same_close_zero']['training']['cagr']=bad
                with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['sharpe']=None
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_later_full_cost_cannot_rescue(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.2
        expected=h.select_first88(m,o)
        for a in h.ARMS:
            for i in range(88,122):o[a][i]['symbols']=['Q','R','S']
        self.assertEqual(h.select_first88(m,o),expected)
        for key in ('full122','later34','next_open_cost'):
            bad=copy.deepcopy(m);bad[h.ARMS[2]][key]={'cagr':10.}
            with self.assertRaises(ValueError):h.select_first88(bad,o)
    def test_missing_arm_short_orders_fail(self):
        m,o=training();o[h.ARMS[1]]=o[h.ARMS[1]][:87]
        with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();del m[h.ARMS[1]]
        with self.assertRaises(ValueError):h.select_first88(m,o)
    def test_counter_queue_and_predecessor(self):
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(255)]+[dict(directory=h.PRIOR.name,trials=4)])
        h.initial_state(s);s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')]);h.registration_check(s,'p')
        for key,value in [('current','old'),('registered_queue',[])]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):h.registration_check(bad,'p')
        s['completed'][-1]['trials']=5
        with self.assertRaises(ValueError):h.initial_state(s)
    def test_predecessor_completion_digest(self):
        s={'completed':[{'completion':dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='digest')}]}
        with patch.object(h,'sha',return_value='digest'):
            h.predecessor_binding(s);s['completed'][-1]['completion']['path']=str(h.OUT/'COMPLETION.json')
            with self.assertRaises(ValueError):h.predecessor_binding(s)
    def test_source_pins_not_new_reconstruction(self):
        s,p=sources();h.source_evidence(s,p)
        for k,bad in [('original_training_hash','bad'),('fresh_training_array_reconstruction',True),('original_orders_exact',False),('cached_training_basis','fresh')]:
            v=dict(s,**{k:bad})
            with self.assertRaises(ValueError):h.source_evidence(v,p)
    def test_phase_terminals_exact(self):
        t=terminals();t['phases']['select']['session_id']=None
        with patch.object(h,'read',return_value=t),patch.object(h,'sha',return_value='digest'):h.phase_terminals_check('plan')
        for key,value in [('exit_code',1),('exit_code',False),('session_id',False),('chunk_id',''),('artifact_path',str(h.OUT/'results/RESULTS.json')),('artifact_sha256','bad')]:
            bad=terminals();bad['phases']['diagnose'][key]=value
            with patch.object(h,'read',return_value=bad),patch.object(h,'sha',return_value='digest'),self.assertRaises(ValueError):h.phase_terminals_check('plan')
        for missing in ('prepare','select','diagnose'):
            bad=terminals();bad['phases'].pop(missing)
            with patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.phase_terminals_check('plan')
    def test_zero_fees_all_six_streams_exact(self):
        exports={a:{p:dict(TRADES=[{'cost':0.}],HOLDINGS=[{'fee':0.}]) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        results={a:{p:dict(metrics={'fees_paid':0.},slices={'full122':{'fees_paid':0.}}) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        h.zero_cost_check(exports,results)
        for a in h.ARMS:
            for p in ('same_close_zero','next_open_zero'):
                for kind,key in [('TRADES','cost'),('HOLDINGS','fee')]:
                    for v in (1e-30,True):
                        bad=copy.deepcopy(exports);bad[a][p][kind][0][key]=v
                        with self.assertRaises(ValueError):h.zero_cost_check(bad,results)
                for branch in ('metrics','full122'):
                    bad=copy.deepcopy(results)
                    (bad[a][p]['metrics'] if branch=='metrics' else bad[a][p]['slices']['full122'])['fees_paid']=1e-30
                    with self.assertRaises(ValueError):h.zero_cost_check(exports,bad)
    def test_all_outcome_counts(self):
        q=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices=dict.fromkeys(('full122','first88','early53','recent35','later34'),{})) for p in h.ev.PATHS} for a in h.ARMS})
        h.outcome_check(q)
        for k,v in [('checked_paths',6),('checked_slices',30),('checked_training_groups',12),('terminal_liquidation',True)]:
            with self.assertRaises(ValueError):h.outcome_check(dict(q,**{k:v}))
    def test_prepare_rejects_before_source_load(self):
        with patch.object(h,'gate',side_effect=ValueError('not registered')),patch.object(h.old,'local') as local,self.assertRaises(ValueError):h.prepare('bad')
        local.assert_not_called()
    def test_full_score_prefix_and_labels(self):
        orders={a:[dict(symbols=['A'])] for a in h.ARMS};scores={a:np.array([[.1]]) for a in h.ARMS}
        wanted=(orders,scores,scores,{})
        h.exact_prefix(wanted,copy.deepcopy(wanted),1)
        for a in h.ARMS:
            got=copy.deepcopy(wanted);got[1][a][0,0]+=.01
            with self.assertRaises(AssertionError):h.exact_prefix(wanted,got,1)
        full=dict(width=1,training_row_ids=np.array([1,2]),label_values=np.array([np.nan,.2]))
        small=dict(width=1,training_row_ids=np.array([1]),label_values=np.array([np.nan]))
        for role in h.ROLES:
            full[role]=dict(raw=np.array([.1,.2]),available=np.array([True,True]),rank=np.array([[.1],[.2]]))
            small[role]=dict(raw=np.array([.1]),available=np.array([True]),rank=np.array([[.1]]))
        h.bundle_parity(full,small,1);small['label_values'][0]=0.
        with self.assertRaises(AssertionError):h.bundle_parity(full,small,1)


def model_counts(n,control=False):
    import monthly_weak_tree261_independent as ind
    k=int(n>1);c=dict.fromkeys(ind.MODEL_COUNT_FIELDS,0)
    c.update(events=n,training_rows=2*k,valid_training_rows=2*k,ordinary_years=k,supported_years=k,matched_ridge_fits=k,prediction_rows=2*k,actual_model_fits=k)
    if not control:c.update(tree_fits=k,trees=100*k,boosting_iterations=100*k,tree_nodes=100*k,manual_tree_traversals=200*k,actual_model_fits=2*k)
    return c

def proof(kind,n):
    import monthly_weak_tree261_independent as ind
    flags={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    r=dict.fromkeys(flags,True);r.update(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,**dict.fromkeys(h.WORK_FIELDS,0))
    if kind in ('control','models'):
        c=model_counts(n,kind=='control')
        r.update(counts=c,producer_equivalent_counts=copy.deepcopy(c),continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,
          full_bundle_supplied=True,full_control_supplied=True,full_candidate_supplied=kind=='models',full_label_preservation_supplied=True,full_model_audit_supplied=True,
          candidate_models_computed=kind=='models',shared_tree_trainer=True,independent_tree_training_algorithm=False,independent_numeric_system=h.NUMERICAL_VERIFICATION['independent_numeric_system'],
          independent_canonical_solves=c['matched_ridge_fits'],independent_svd_solves=c['matched_ridge_fits'],independent_tree_refits=c['tree_fits'],manual_tree_traversals=2*c['manual_tree_traversals'])
    else:
        r.update(full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,ordered_choices_checked=n if kind=='incumbent' else 3*n,original_episode_calls=1 if kind=='incumbent' else 3)
        if kind=='incumbent':r['candidate_selection_computed']=False
        else:
            c=dict.fromkeys(ind.COUNT_FIELDS,0);c.update(events=n,symbols_per_event=49,initial_partial_events=int(n>0),formal_events=max(n-1,0),incumbent_episode_calls=1,matched_episode_calls=1,tree_episode_calls=1)
            r.update(full_bundle_supplied=True,full_audit_supplied=True,counts=c)
    return r

def model_audit(c):
    m=dict(schema='monthly_weak_tree261_model_audit_v1',counts=copy.deepcopy(c),computes_returns=False,
       original_cache_parity=dict(passed=True,events=c['events'],original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False),
       execution_runtime=[dict(year=2020,fit_threadpools=[dict(filepath='/fake/lib.so',num_threads=1)],predict_threadpools=[dict(filepath='/fake/lib.so',num_threads=1)])] if c['tree_fits'] else [])
    for k in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only'):m[k]=True
    return m

def label_audit(c):
    import monthly_weak_tree261_independent as ind
    return dict(schema='monthly_weak_tree261_labels_preserved_v1',rows=c['training_rows'],valid_rows=c['valid_training_rows'],target_keys=list(ind.TARGETS),target_sha256={k:'a'*64 for k in ind.TARGETS},common_good_sha256='b'*64,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)

def bundle(n,control=False):
    c=model_counts(n,control);f=dict(raw=np.zeros(n),available=np.ones(n,bool),rank=np.zeros((n,1)),audit={})
    return dict(n=n,width=1,audit=model_audit(c),label_preservation=label_audit(c),control=f,candidate=None if control else copy.deepcopy(f),control_gate=None,training_row_ids=np.arange(c['training_rows']),label_values=np.zeros(c['training_rows']))

def matched(n):
    return dict(passed=True,events=n,matched_ranks_exact=True,delta_exact_zero=True,pre_scores_exact=True,final_scores_exact=True,ordered_choices_exact=True,producer_episode_calls=1,independent=proof('incumbent',n))

def evidence():
    e={k:proof(k,122) for k in ('control','models','incumbent','selection')}
    e.update(early_incumbent=proof('incumbent',122),matched_strategy=matched(122),main_model_counts=model_audit(e['models']['counts']),all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    e['cached_prefixes']=[dict(prefix=n,exact=True,incumbent=proof('incumbent',n),selection=proof('selection',n)) for n in range(123)]
    e['physical_prefixes']=[]
    for n in h.PREFIXES:
        v=dict(prefix=n,exact=True,model_recomputed=n!=122,early_incumbent=proof('incumbent',n),matched_strategy=matched(n))
        for key in ('control','models','incumbent','selection'):v['model' if key=='models' else key]=proof(key,n)
        v['model_counts']=model_audit(v['model']['counts'])
        if n==122:v['reuses_main_verified_full']=True
        else:v['input_scope']=dict(events=n,training_months=int(n>1),training_rows=v['model']['counts']['training_rows'],cutoff='2019-12-31' if n>1 else None,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=bool(n>1),physically_shortened=True)
        e['physical_prefixes'].append(v)
    e['fit_execution_accounting']=h.execution_accounting(e)
    ma=dict(model=copy.deepcopy(e['main_model_counts']),labels=label_audit(e['models']['counts']))
    da=dict(schema='monthly_weak_tree261_decisions_v1',events=[{} for _ in range(122)],counts=copy.deepcopy(e['selection']['counts']),incumbent_gate=copy.deepcopy(e['incumbent']),computes_returns=False)
    for k in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):da[k]=True
    return e,ma,da

class ReceiptTests(unittest.TestCase):
    def test_complete_receipts_n0_partial_and_three_arms(self):
        for kind in ('control','models','incumbent','selection'):
            for n in (0,1,2,122):h.receipt(proof(kind,n),n,(),kind)
    def test_missing_standard_flags_cannot_be_disabled(self):
        import monthly_weak_tree261_independent as ind
        for kind,flags in [('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]:
            for flag in flags:
                r=proof(kind,2);r.pop(flag)
                with self.assertRaises(ValueError):h.receipt(r,2,(),kind)
    def test_full_supplied_scope_tolerances_and_integer_types(self):
        for kind in ('control','models','incumbent','selection'):
            r=proof(kind,2)
            for key in [k for k,v in r.items() if k.startswith('full_')]:
                bad=dict(r);bad.pop(key)
                with self.assertRaises(ValueError):h.receipt(bad,2,(),kind)
            for key in [k for k in r if 'tolerance' in k or k in ('continuous_atol','continuous_rtol')]:
                for value in (False,True,float('nan'),1e-12):
                    with self.assertRaises(ValueError):h.receipt(dict(r,**{key:value}),2,(),kind)
            for key,value in [('checked_events',True),('producer_modules_imported',True),('computes_returns',True)]:
                with self.assertRaises(ValueError):h.receipt(dict(r,**{key:value}),2,(),kind)
    def test_model_counts_schema_and_relations(self):
        for key,value in [('training_rows',True),('valid_training_rows',3),('ordinary_years',2),('supported_years',2),('no_prediction_years',1),('tree_fits',0),('original_other_target_fits',1),('original_local_fits',1),('trees',99),('boosting_iterations',99),('manual_tree_traversals',199),('actual_model_fits',1),('tree_nodes',99)]:
            r=proof('models',2);r['counts'][key]=value;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
            with self.assertRaises(ValueError):h.receipt(r,2,(),'models')
        r=proof('models',2);r['counts']['geometry_builds']=1
        with self.assertRaises(ValueError):h.receipt(r,2,(),'models')
    def test_no_prediction_is_supported_but_no_fits(self):
        c=model_counts(2);c.update(matched_ridge_fits=0,no_prediction_years=1,tree_fits=0,prediction_rows=0,trees=0,boosting_iterations=0,tree_nodes=0,manual_tree_traversals=0,actual_model_fits=0)
        h.model_counts_check(c,2);c['tree_fits']=1
        with self.assertRaises(ValueError):h.model_counts_check(c,2)
    def test_independent_shared_trainer_and_actual_ledgers(self):
        for kind in ('control','models','incumbent','selection'):
            for key in h.WORK_FIELDS:
                r=proof(kind,2);del r[key]
                with self.assertRaises(ValueError):h.receipt(r,2,(),kind)
        for key,value in [('independent_canonical_solves',0),('independent_svd_solves',0),('independent_tree_refits',0),('manual_tree_traversals',200),('shared_tree_trainer',False),('independent_tree_training_algorithm',True)]:
            r=proof('models',2);r[key]=value
            with self.assertRaises(ValueError):h.receipt(r,2,(),'models')
    def test_control_no_tree_work(self):
        for key,value in [('candidate_models_computed',True),('full_candidate_supplied',True),('manual_tree_traversals',1),('independent_tree_refits',1)]:
            r=proof('control',2);r[key]=value
            with self.assertRaises(ValueError):h.receipt(r,2,(),'control')
    def test_failed_control_and_source_mutation_stop_matched_gate(self):
        import monthly_weak_tree261_kernel as kernel
        arrays={'a':np.array([1.])};train={'a':np.array([2.])};meta={'monthly':{'dates':[]}}
        original=([],np.zeros((0,1)),np.zeros((0,1)));view=bundle(0,True)
        oracle=SimpleNamespace(CONTROL_FLAGS=(),INCUMBENT_FLAGS=(),verify_control_models=Mock(return_value=dict(proof('control',0),passed=False)),verify_incumbent_selection=Mock(return_value=proof('incumbent',0)))
        with patch.object(kernel,'matched_decisions',return_value=original) as match:
            cb,_=h.model_callbacks(oracle,train,meta,arrays,0,study=None,original=original,expected_orders=[])
            with self.assertRaises(ValueError):cb(copy.deepcopy(view))
            match.assert_not_called()
            def mutate(*args,**kwargs):train['a'][0]+=1;return proof('control',0)
            oracle.verify_control_models.side_effect=mutate
            cb,_=h.model_callbacks(oracle,train,meta,arrays,0,study=None,original=original,expected_orders=[])
            with self.assertRaises(ValueError):cb(copy.deepcopy(view))
            match.assert_not_called()
    def test_matched_strategy_gate_and_one_callback(self):
        import monthly_weak_tree261_kernel as kernel
        original=([],np.zeros((0,1)),np.zeros((0,1)));meta={'monthly':{'dates':[],'ends':[],'symbols':[],'etfs':[]}}
        oracle=SimpleNamespace(CONTROL_FLAGS=(),INCUMBENT_FLAGS=(),verify_control_models=Mock(return_value=proof('control',0)),verify_incumbent_selection=Mock(return_value=proof('incumbent',0)))
        with patch.object(kernel,'matched_decisions',return_value=original):
            cb,rs=h.model_callbacks(oracle,{},meta,{},0,study=None,original=original,expected_orders=[])
            self.assertEqual(cb(bundle(0,True)),proof('control',0));h.matched_strategy_check(rs['matched_strategy'],0)
            with self.assertRaises(ValueError):cb(bundle(0,True))
        with patch.object(kernel,'matched_decisions',return_value=([{}],np.zeros((0,1)),np.zeros((0,1)))):
            cb,_=h.model_callbacks(oracle,{},meta,{},0,study=None,original=original,expected_orders=[])
            with self.assertRaises(ValueError):cb(bundle(0,True))
    def test_bad_early_original_stops_fit(self):
        import monthly_weak_tree261_kernel as kernel
        with patch.object(kernel,'original_decisions',return_value=([{'symbols':['wrong']}],np.zeros((1,1)),np.zeros((1,1)))),patch.object(kernel,'fit_models') as fit:
            with self.assertRaises(ValueError):h.prepare_one(None,{},dict(monthly={}),{},None,1,[{'symbols':['expected']}])
            fit.assert_not_called()
    def test_frozen_incumbent_before_candidate(self):
        oracle=SimpleNamespace(INCUMBENT_FLAGS=(),verify_incumbent_selection=Mock(return_value=proof('incumbent',1)))
        cb,records=h.incumbent_callback(oracle,{}, {},1,[{'symbols':['A']}])
        with self.assertRaises(ValueError):cb([{'symbols':['B']}],None,None)
        oracle.verify_incumbent_selection.assert_not_called();cb([{'symbols':['A']}],None,None)
        self.assertEqual(set(records),{'incumbent'})
    def test_three_episode_calls_and_exact_width(self):
        r=proof('selection',1);r['ordered_choices_checked']=2
        with self.assertRaises(ValueError):h.receipt(r,1,(),'selection')
        for k in ('incumbent_episode_calls','matched_episode_calls','tree_episode_calls'):
            r=proof('selection',0);r['counts'][k]=0
            with self.assertRaises(ValueError):h.receipt(r,0,(),'selection')
        r=proof('selection',1);r['counts']['symbols_per_event']=6
        h.receipt(r,1,(),'selection',width=6)
        with self.assertRaises(ValueError):h.receipt(r,1,(),'selection',width=49)
    def test_labels_and_support_preservation(self):
        b=bundle(2);h.bundle_preservation(b,2)
        for key in [k for k,v in b['audit'].items() if v is True]:
            bad=copy.deepcopy(b);bad['audit'][key]=False
            with self.assertRaises(ValueError):h.bundle_preservation(bad,2)
        for key,value in [('recentered',True),('rows',True),('common_good_sha256','bad')]:
            bad=copy.deepcopy(b);bad['label_preservation'][key]=value
            with self.assertRaises(ValueError):h.bundle_preservation(bad,2)
        b['candidate']['available'][0]=False
        with self.assertRaises(ValueError):h.bundle_preservation(b,2)
    def test_runtime_fit_and_predict_both_required(self):
        b=bundle(2)
        for branch in ('fit_threadpools','predict_threadpools'):
            for badvalue in (True,0,2):
                bad=copy.deepcopy(b);bad['audit']['execution_runtime'][0][branch][0]['num_threads']=badvalue
                with self.assertRaises(ValueError):h.bundle_preservation(bad,2)
            bad=copy.deepcopy(b);bad['audit']['execution_runtime'][0][branch]=[]
            with self.assertRaises(ValueError):h.bundle_preservation(bad,2)
        b['audit']['execution_runtime']=[]
        with self.assertRaises(ValueError):h.bundle_preservation(b,2)
    def test_all_prefix_evidence(self):h.preparation_evidence(*evidence())
    def test_missing_prefix_earlygate_or_recomputed122(self):
        for mode in ('cached','physical','reuse','gate','counts','early','matched'):
            e,m,d=evidence()
            if mode=='cached':e['cached_prefixes'].pop(2)
            elif mode=='physical':e['physical_prefixes'].pop(1)
            elif mode=='reuse':e['physical_prefixes'][-1]['model_recomputed']=True
            elif mode=='gate':d['incumbent_gate']['passed']=False
            elif mode=='counts':e['models']['counts']['tree_fits']=0
            elif mode=='early':e['early_incumbent']['passed']=False
            else:e['matched_strategy']['delta_exact_zero']=False
            with self.assertRaises(ValueError):h.preparation_evidence(e,m,d)
    def test_actual_work_no_doublecount122(self):
        e,m,d=evidence();a=e['fit_execution_accounting'];self.assertEqual(a['model_invocations'],5)
        self.assertEqual(a['totals']['producer_matched_ridge_fits'],3)
        self.assertEqual(a['totals']['producer_tree_fits'],3)
        self.assertEqual(a['totals']['independent_tree_refits'],3)
        self.assertEqual(a['totals']['independent_canonical_solves'],6)
        self.assertEqual(a['totals']['independent_svd_solves'],6)
        self.assertEqual(a['totals']['total_model_fit_calls'],21)
        self.assertEqual(a['totals']['producer_manual_tree_traversals'],600)
        self.assertEqual(a['totals']['manual_tree_traversals'],1200)
        self.assertEqual(a['totals']['producer_episode_calls'],394)
        self.assertEqual(a['totals']['original_episode_calls'],522)
        self.assertEqual(a['entries'][1]['total_model_fit_calls'],0)
        e['fit_execution_accounting']['totals']['independent_canonical_solves']=0
        with self.assertRaises(ValueError):h.preparation_evidence(e,m,d)
    def test_review_binds_terminals_every_artifact_and_contract(self):
        paths=[h.PHASE_TERMINALS,h.OUT/'PLAN.json',h.CONTEXT,h.CONTRACT,h.OUT/'prepared/COMPLETION.json',h.OUT/'selected/FROZEN.json',h.OUT/'selected/TRAIN_RESULTS.json',h.OUT/'results/COMPLETION.json']
        paths += [h.OUT/'prepared'/n for n in h.PREP_FILES]+[h.OUT/'results'/n for n in h.RESULT_FILES]
        r=dict(passed=True,unresolved_blockers=[],plan_sha256='digest',context_sha256='digest',prepared_completion_sha256='digest',results_completion_sha256='digest',implementation_contract_sha256='digest',input_sha256={str(p):'digest' for p in paths})
        with patch.object(h,'phase_terminals_check') as phase,patch.object(h,'read',return_value=r),patch.object(h,'sha',return_value='digest'),patch.object(h.old,'verify'):
            h.result_review_check('digest');phase.assert_called_once_with('digest')
            for p in paths:
                bad=copy.deepcopy(r);del bad['input_sha256'][str(p)]
                with patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.result_review_check('digest')


class PhysicalInputTests(unittest.TestCase):
    @staticmethod
    def fixture():
        dates=['2016-08-17','2016-08-31','2017-01-31'];w=2
        monthly=dict(dates=dates,ends=['2016-08-31','2016-09-30','2017-02-28'],symbols=[['A','B'] for _ in dates],etfs=[])
        tm=dict(dates=['2015-11-30','2015-12-31','2016-11-30','2016-12-31'],ends=['2015-12-31','2016-01-31','2016-12-31','2017-01-31'],symbols=[['A','B'] for _ in range(4)])
        train=dict(offsets=np.arange(5)*2,dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),2),ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),2),train_plain=np.arange(80.).reshape(8,10),common_good=np.ones(8,bool),event_dates=np.repeat(np.array(dates,dtype='datetime64[D]'),2),event_plain=np.arange(60.).reshape(6,10))
        for k in ('simple_excess','log_excess','vol_scaled','loss_utility'):train['target_'+k]=np.arange(8.)
        arrays={'baseline':np.zeros((3,w)),'eligible':np.ones((3,w),bool)}
        return train,dict(monthly=monthly,training=tm),arrays
    def test_true_small_axes_prior_december_boundary_and_global_ids(self):
        for n,months,rows in ((2,1,2),(3,3,6)):
            t,m,a=self.fixture();small,meta,aa,scope=h.physical_inputs(t,m,a,n)
            self.assertEqual(aa['baseline'].shape,(n,2));self.assertEqual(len(meta['monthly']['dates']),n)
            self.assertEqual(len(meta['training']['dates']),months);self.assertEqual(small['train_plain'].shape,(rows,10))
            self.assertEqual(small['event_plain'].shape,(n*2,10));np.testing.assert_array_equal(small['offsets'],t['offsets'][:months+1])
            np.testing.assert_array_equal(small['target_log_excess'],np.arange(rows))
            h.physical_scope_check(scope,n,dict(training_rows=rows))
            small['target_log_excess'][0]=999;self.assertEqual(t['target_log_excess'][0],0.)
    def test_n0_partial_does_not_access_training(self):
        class NoTraining(dict):
            def __getitem__(self,k):raise AssertionError('Training touched')
        for n in (0,1):
            _,m,a=self.fixture();m['training']=NoTraining()
            t,mm,aa,scope=h.physical_inputs(NoTraining(),m,a,n)
            self.assertEqual(t,{});self.assertEqual(mm['training'],dict(dates=[],ends=[],symbols=[]))
            h.physical_scope_check(scope,n,dict(training_rows=0))
    def test_future_numeric_and_symbol_bodies_are_not_read(self):
        class Poison:
            def __float__(self):raise AssertionError('Future conversion')
            def __deepcopy__(self,memo):raise AssertionError('Future copy')
            def __iter__(self):raise AssertionError('Future iteration')
        t,m,a=self.fixture()
        for k in ('train_plain','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility'):
            t[k]=t[k].astype(object);t[k][2:]=Poison()
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][4:]=Poison()
        m['training']['symbols'][1:]=[Poison() for _ in range(3)];m['monthly']['symbols'][2]=Poison()
        small,mm,aa,scope=h.physical_inputs(t,m,a,2)
        self.assertEqual(len(mm['training']['symbols']),1);self.assertEqual(small['train_plain'].shape,(2,10))
        self.assertTrue(np.isfinite(np.asarray(small['train_plain'],float)).all())
    def test_noncontiguous_maturity_and_bad_source_shape_rejected(self):
        t,m,a=self.fixture();m['training']['ends'][2]='2015-12-31'
        with self.assertRaises(ValueError):h.physical_inputs(t,m,a,2)
        t,m,a=self.fixture();t['event_plain']=t['event_plain'][:-1]
        with self.assertRaises(ValueError):h.physical_inputs(t,m,a,2)
    def test_physical_saved_scope_cannot_claim_full_or_rebased(self):
        e,m,d=evidence()
        for key,value in [('physically_shortened',False),('global_row_ids_start',1),('training_rows',999),('training_numeric_consumed',True)]:
            bad=copy.deepcopy(e);bad['physical_prefixes'][0]['input_scope'][key]=value
            with self.assertRaises(ValueError):h.preparation_evidence(bad,m,d)


def closure_fixture():
    pins={str(p):v for p,v in h.SOURCE_CHAIN169.items()};artifacts={name:('a'*64) for name in h.SOURCE_FILES169}
    pins.update({str(h.REBUILD/name):v for name,v in artifacts.items()});pins[str(h.ROOT/'work/synthetic_dependency.py')]='b'*64
    bp=h.REBUILD.parent/'PLAN.json';pp=h.REBUILD/'PARITY.json';ep=h.REBUILD.parent/'evaluation/PLAN.json';ec=h.REBUILD.parent/'evaluation/results/COMPLETION.json';ic=h.REBUILD.parent/'independent_review/COMPLETION.json'
    records={bp:dict(input_sha256={'work/synthetic_dependency.py':'b'*64}),pp:dict(passed=True,arm='control',returns_computed=False,source_plan_sha256=pins[str(bp)],artifacts_sha256=artifacts),ep:dict(rebuild_plan_sha256=pins[str(bp)],input_sha256={str(p.relative_to(h.ROOT)):pins[str(p)] for p in [h.REBUILD/n for n in h.SOURCE_FILES169]+[bp,pp]}),ec:dict(status='COMPLETE',evaluation_plan_sha256=pins[str(ep)],rebuild_plan_sha256=pins[str(bp)],exact_control_rechecked=True,source_hashes_verified_before_after=True),ic:dict(status='PASS',checked_paths=6,checked_slices=30,evaluation_completion_sha256=pins[str(ec)])}
    return pins,records

class OriginalSourceClosureTests(unittest.TestCase):
    def check(self,pins,records):
        with patch.object(h,'read',side_effect=lambda p:records[p]),patch.object(h,'sha',side_effect=lambda p:h.SOURCE_CHAIN169[p]):return h.source_closure_check(pins)
    def test_complete_twelve_artifact_and_metadata_chain(self):
        pins,records=closure_fixture();proof=self.check(pins,records)
        self.assertEqual(set(proof['artifacts_sha256']),h.SOURCE_FILES169);self.assertFalse(proof['npz_decoded']);self.assertFalse(proof['source_cache_rebuilt'])
    def test_missing_or_substituted_cache_pins_fail(self):
        for name in h.SOURCE_FILES169:
            pins,records=closure_fixture();pins.pop(str(h.REBUILD/name))
            with self.assertRaises(ValueError):self.check(pins,records)
        pins,records=closure_fixture();records[h.REBUILD/'PARITY.json']['artifacts_sha256'].pop('BASE_DATA.npz')
        with self.assertRaises(ValueError):self.check(pins,records)
    def test_source_input_closure_and_evaluation_bindings_required(self):
        pins,records=closure_fixture();pins.pop(str(h.ROOT/'work/synthetic_dependency.py'))
        with self.assertRaises(ValueError):self.check(pins,records)
        pins,records=closure_fixture();records[h.REBUILD.parent/'evaluation/PLAN.json']['input_sha256'].pop(str((h.REBUILD/'DATA.npz').relative_to(h.ROOT)))
        with self.assertRaises(ValueError):self.check(pins,records)
    def test_chain_hash_plan_links_completion_and_independent_status(self):
        changes=[(h.REBUILD/'PARITY.json','source_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/PLAN.json','rebuild_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/results/COMPLETION.json','evaluation_plan_sha256','wrong'),(h.REBUILD.parent/'evaluation/results/COMPLETION.json','exact_control_rechecked',False),(h.REBUILD.parent/'independent_review/COMPLETION.json','status','PENDING'),(h.REBUILD.parent/'independent_review/COMPLETION.json','evaluation_completion_sha256','wrong')]
        for path,key,value in changes:
            pins,records=closure_fixture();records[path][key]=value
            with self.assertRaises(ValueError):self.check(pins,records)
        pins,records=closure_fixture();pins[str(h.REBUILD/'PARITY.json')]='wrong'
        with self.assertRaises(ValueError):self.check(pins,records)
    def test_saved_source_certificate_exact_plan_binding(self):
        source,plan=sources();source['original169_source_closure']={}
        with self.assertRaises(ValueError):h.source_evidence(source,plan)

if __name__=='__main__':unittest.main(verbosity=2)

