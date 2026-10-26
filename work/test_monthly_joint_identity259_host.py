"""Synthetic lifecycle/receipt tests; no market arrays or saved outcomes are read."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import monthly_joint_identity259_host as h


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
    plan=dict(raw_source_hashes={'synthetic':'a'},input_sha256={str(h.REBUILD/name):name+' hash' for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json')})
    s=dict(raw_source_hashes=copy.deepcopy(plan['raw_source_hashes']),fresh_training_array_reconstruction=False,cached_training_basis='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.',original_arrays_unchanged=True,original_training_unchanged=True,original_orders_exact=True)
    for k,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:s[k]=plan['input_sha256'][str(h.REBUILD/name)]
    return s,plan


class LifecycleTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','sequential_identity','joint_identity'))
        self.assertEqual(h.ROLES,('control','diagnostic','candidate'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(10,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_downside_basket258')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))
        self.assertEqual(h.SOURCE_REVIEW.name,'after258_joint_source_review.json')
    def test_joint_is_only_promotable(self):
        m,o=training();self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[2])
        m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.4
        for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=.9
        self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[0])
    def test_inc_close_gain_open_noninferiority(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.499)]:
            m,o=training();m[h.ARMS[2]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_sequential_cannot_be_worse_reference_ignored(self):
        for p in ('same_close_zero','next_open_zero'):
            m,o=training();m[h.ARMS[1]][p]['training']['cagr']=.53
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_both_references_risk_and_segment_guards(self):
        for ref in h.ARMS[:2]:
            for p in ('same_close_zero','next_open_zero'):
                for group,key,value in [('training','sharpe',1.8),('training','max_drawdown',-.2),('early','cagr',.5),('recent','cagr',.7)]:
                    m,o=training();m[ref][p][group][key]=value
                    self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_inc_support_three_across_eras(self):
        for ids in ((1,54),(1,2,3),(53,54,55)):
            m,o=training();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[2]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])
    def test_seq_support_one_distinct_set(self):
        m,o=training();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[2]])
        self.assertFalse(h.select_first88(m,o)['admitted'])
        o[h.ARMS[1]][7]['symbols']=['A','B','E']
        self.assertTrue(h.select_first88(m,o)['admitted'])
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
                if a==h.ARMS[0]:
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
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(253)]+[dict(directory=h.PRIOR.name,trials=8)])
        h.initial_state(s);s.update(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')]);h.registration_check(s,'p')
        for key,value in [('current','old'),('registered_queue',[])]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):h.registration_check(bad,'p')
        s['completed'][-1]['trials']=9
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
            full[role]=dict(raw={'source_log':np.array([.1,.2])},available=np.array([True,True]),local_available=np.array([True,True]))
            small[role]=dict(raw={'source_log':np.array([.1])},available=np.array([True]),local_available=np.array([True]))
        h.bundle_parity(full,small,1);small['label_values'][0]=0.
        with self.assertRaises(AssertionError):h.bundle_parity(full,small,1)


def proof(kind,n):
    import monthly_joint_identity259_independent as ind
    flags={'control':ind.CONTROL_FLAGS,'models':ind.MODEL_FLAGS,'incumbent':ind.INCUMBENT_FLAGS,'selection':ind.SELECTION_FLAGS}[kind]
    r=dict.fromkeys(flags,True);r.update(checked_events=n,producer_modules_imported=False,computes_returns=False)
    k=int(n>0);pc=dict(target_fits=4*k,log_refits=k,local_evaluations=2*k,local_solves=k,actual_regression_solves=6*k)
    sc=dict(pooled_log_solves=k,local_linear_solves=k,total_regression_solves=2*k)
    if kind in ('control','models'):
        r.update(full_control_supplied=True,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,independent_design_builds=0 if kind=='control' else k)
        r['independent_canonical_solve_counts']=dict(pooled_log_solves=0,sequential_dummy_solves=0,joint_block_solves=0,total_regression_solves=0)
    if kind=='control':r.update(candidate_labels_computed=False,candidate_models_computed=False,episode_calls=0,producer_equivalent_fit_counts=pc,independent_solve_counts=sc)
    elif kind=='models':
        c=dict.fromkeys(ind.MODEL_COUNT_FIELDS,0);c.update(training_rows=2*k,valid_training_rows=2*k,annual_design_builds=k,qualified_security_columns=2*k,qualified_query_rows=2*k,diagnostic_dummy_solves=k,candidate_joint_solves=k,actual_new_model_solves=2*k,actual_regression_solves=8*k)
        for key in ('target_fits','log_refits','local_evaluations','local_solves'):c['control_'+key]=pc[key]
        r.update(counts=c,producer_equivalent_counts=copy.deepcopy(c),full_bundle_supplied=True,full_labels_supplied=True,full_diagnostic_supplied=True,full_candidate_supplied=True,independent_numeric_system=h.NUMERICAL_VERIFICATION['independent_numeric_system'],independent_solve_counts=dict(control=sc,sequential=dict(dummy_solves=k,total_regression_solves=k),joint=dict(block_solves=k,total_regression_solves=k),total_regression_solves=4*k),independent_canonical_solve_counts=dict(pooled_log_solves=k,sequential_dummy_solves=k,joint_block_solves=k,total_regression_solves=3*k))
    else:
        r.update(full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,ordered_choices_checked=n if kind=='incumbent' else 3*n)
        if kind=='incumbent':r['candidate_selection_computed']=False
        else:
            c=dict.fromkeys(ind.COUNT_FIELDS,0);c.update(events=n,symbols_per_event=49,initial_partial_events=k,formal_events=n-k,incumbent_episode_calls=1,diagnostic_episode_calls=1,candidate_episode_calls=1)
            r.update(full_bundle_supplied=True,full_audit_supplied=True,counts=c)
    return r


def model_audit(c):
    m=dict(schema='monthly_joint_identity259_model_audit_v1',counts=copy.deepcopy(c),computes_returns=False)
    for k in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','diagnostic_source_log_copied','candidate_source_log_copied','diagnostic_local_constant_copied','candidate_local_constant_copied','availability_preserved','local_support_preserved','common_design_weights_penalties','security_intercepts_only','candidate_global_coefficients_refit'):m[k]=True
    return m


def label_audit(c):
    import monthly_joint_identity259_independent as ind
    return dict(schema='monthly_joint_identity259_labels_preserved_v1',rows=c['training_rows'],valid_rows=c['valid_training_rows'],target_keys=list(ind.TARGETS),target_sha256={k:'a'*64 for k in ind.TARGETS},common_good_sha256='b'*64,labels_unchanged=True,common_good_unchanged=True,original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)


def evidence():
    e={k:proof(k,122) for k in ('control','models','incumbent','selection')}
    e.update(main_model_counts=model_audit(e['models']['counts']),all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    e['cached_prefixes']=[dict(prefix=n,exact=True,incumbent=proof('incumbent',n),selection=proof('selection',n)) for n in range(123)]
    e['physical_prefixes']=[]
    for n in h.PREFIXES:
        v=dict(prefix=n,exact=True,model_recomputed=n!=122)
        for key in ('control','models','incumbent','selection'):v['model' if key=='models' else key]=proof(key,n)
        v['model_counts']=model_audit(v['model']['counts'])
        if n==122:v['reuses_main_verified_full']=True
        e['physical_prefixes'].append(v)
    e['fit_execution_accounting']=h.execution_accounting(e)
    ma=dict(model=copy.deepcopy(e['main_model_counts']),labels=label_audit(e['models']['counts']))
    da=dict(schema='monthly_joint_identity259_decisions_v1',events=[{} for _ in range(122)],counts=copy.deepcopy(e['selection']['counts']),incumbent_gate=copy.deepcopy(e['incumbent']),computes_returns=False)
    for k in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','full_original_local_quarter','partial_exact'):da[k]=True
    return e,ma,da


class ReceiptTests(unittest.TestCase):
    def test_complete_receipts_n0_and_three_arms(self):
        for kind in ('control','models','incumbent','selection'):
            for n in (0,1,122):h.receipt(proof(kind,n),n,(),kind)
    def test_missing_standard_flags_cannot_be_disabled(self):
        import monthly_joint_identity259_independent as ind
        for kind,flags in [('control',ind.CONTROL_FLAGS),('models',ind.MODEL_FLAGS),('incumbent',ind.INCUMBENT_FLAGS),('selection',ind.SELECTION_FLAGS)]:
            for flag in flags:
                r=proof(kind,1);r.pop(flag)
                with self.assertRaises(ValueError):h.receipt(r,1,(),kind)
    def test_full_supplied_scope_tolerances_and_integer_types(self):
        for kind in ('control','models','incumbent','selection'):
            r=proof(kind,1)
            for key in [k for k in r if k.startswith('full_')]:
                bad=dict(r);bad.pop(key)
                with self.assertRaises(ValueError):h.receipt(bad,1,(),kind)
            for key in [k for k in r if 'tolerance' in k or k in ('continuous_atol','continuous_rtol')]:
                for value in (False,True,float('nan'),1e-12):
                    with self.assertRaises(ValueError):h.receipt(dict(r,**{key:value}),1,(),kind)
            for key,value in [('checked_events',True),('producer_modules_imported',True),('computes_returns',True)]:
                with self.assertRaises(ValueError):h.receipt(dict(r,**{key:value}),1,(),kind)
    def test_counts_schema_and_relations(self):
        for key,value in [('training_rows',True),('valid_training_rows',3),('control_target_fits',3),('control_local_solves',3),('annual_design_builds',2),('qualified_query_rows',1),('diagnostic_pooled_fits',1),('candidate_pooled_fits',1),('candidate_joint_solves',0),('actual_new_model_solves',1),('actual_regression_solves',7),('no_qualified_query_years',1)]:
            r=proof('models',1);r['counts'][key]=value;r['producer_equivalent_counts']=copy.deepcopy(r['counts'])
            with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
        r=proof('models',1);r['counts']['geometry_builds']=1
        with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
    def test_svd_and_canonical_ledgers_both_required(self):
        for key in ('independent_solve_counts','independent_canonical_solve_counts','independent_design_builds'):
            r=proof('models',1);del r[key]
            with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
        for path in [('independent_solve_counts','joint','block_solves'),('independent_solve_counts','control','pooled_log_solves'),('independent_canonical_solve_counts','sequential_dummy_solves'),('independent_canonical_solve_counts','total_regression_solves')]:
            r=proof('models',1);d=r
            for key in path[:-1]:d=d[key]
            d[path[-1]]+=1
            with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
        r=proof('models',1);r['producer_equivalent_counts']['candidate_joint_solves']=0
        with self.assertRaises(ValueError):h.receipt(r,1,(),'models')
    def test_control_precedes_new_fits(self):
        oracle=SimpleNamespace(CONTROL_FLAGS=(),verify_control_models=Mock(return_value=proof('control',0)))
        cb,records=h.model_callbacks(oracle,{}, {},{},0);cb({});self.assertEqual(set(records),{'control'})
        with self.assertRaises(ValueError):cb({})
        for key,value in [('candidate_models_computed',True),('candidate_labels_computed',True),('episode_calls',1),('independent_design_builds',1)]:
            oracle.verify_control_models.return_value=dict(proof('control',0),**{key:value});cb,_=h.model_callbacks(oracle,{}, {},{},0)
            with self.assertRaises(ValueError):cb({})
        bad=proof('control',1);bad['independent_canonical_solve_counts'].update(pooled_log_solves=1,total_regression_solves=1)
        with self.assertRaises(ValueError):h.receipt(bad,1,(),'control')
    def test_frozen_incumbent_before_candidate(self):
        oracle=SimpleNamespace(INCUMBENT_FLAGS=(),verify_incumbent_selection=Mock(return_value=proof('incumbent',1)))
        cb,records=h.incumbent_callback(oracle,{}, {},1,[{'symbols':['A']}])
        with self.assertRaises(ValueError):cb([{'symbols':['B']}],None,None)
        oracle.verify_incumbent_selection.assert_not_called();cb([{'symbols':['A']}],None,None)
        self.assertEqual(set(records),{'incumbent'})
    def test_three_episode_calls_and_exact_width(self):
        r=proof('selection',1);r['ordered_choices_checked']=2
        with self.assertRaises(ValueError):h.receipt(r,1,(),'selection')
        for k in ('incumbent_episode_calls','diagnostic_episode_calls','candidate_episode_calls'):
            r=proof('selection',0);r['counts'][k]=0
            with self.assertRaises(ValueError):h.receipt(r,0,(),'selection')
        r=proof('selection',1);r['counts']['symbols_per_event']=6
        h.receipt(r,1,(),'selection',width=6)
        with self.assertRaises(ValueError):h.receipt(r,1,(),'selection',width=49)
    def test_all_label_and_reference_copy_declarations(self):
        c=proof('models',1)['counts'];m=model_audit(c);lp=label_audit(c)
        h.model_preservation(m,lp,1)
        for key in [k for k,v in m.items() if v is True]:
            bad=copy.deepcopy(m);bad[key]=False
            with self.assertRaises(ValueError):h.model_preservation(bad,lp,1)
        for key,value in [('recentered',True),('rows',True),('common_good_sha256','bad')]:
            with self.assertRaises(ValueError):h.model_preservation(m,dict(lp,**{key:value}),1)
        b=dict(audit=m,label_preservation=lp)
        for role in h.ROLES:b[role]=dict(raw=dict(source_log=np.array([.1]),local_constant=np.array([.2])),available=np.array([True]),local_available=np.array([False]))
        h.bundle_preservation(b,1)
        for role in ('diagnostic','candidate'):
            bad=copy.deepcopy(b);bad[role]['raw']['local_constant'][0]+=.1
            with self.assertRaises(ValueError):h.bundle_preservation(bad,1)
    def test_all_prefix_evidence(self):h.preparation_evidence(*evidence())
    def test_missing_prefix_changed_gate_or_recomputed122(self):
        for mode in ('cached','physical','reuse','gate','counts','quarter'):
            e,m,d=evidence()
            if mode=='cached':e['cached_prefixes'].pop(2)
            elif mode=='physical':e['physical_prefixes'].pop(1)
            elif mode=='reuse':e['physical_prefixes'][-1]['model_recomputed']=True
            elif mode=='gate':d['incumbent_gate']['passed']=False
            elif mode=='counts':e['models']['counts']['candidate_joint_solves']=0
            else:d['full_original_local_quarter']=False
            with self.assertRaises(ValueError):h.preparation_evidence(e,m,d)
    def test_actual_work_no_doublecount122(self):
        e,m,d=evidence();a=e['fit_execution_accounting'];self.assertEqual(a['model_invocations'],5)
        self.assertEqual(a['totals']['producer_regression_solves'],32)
        self.assertEqual(a['totals']['independent_control_gate_svd_solves'],8)
        self.assertEqual(a['totals']['independent_model_check_svd_solves'],16)
        self.assertEqual(a['totals']['independent_model_check_canonical_solves'],12)
        self.assertEqual(a['totals']['total_regression_solves'],68)
        self.assertEqual(a['totals']['producer_new_model_solves'],8)
        self.assertEqual(a['totals']['producer_centered_design_builds'],4)
        self.assertEqual(a['entries'][1]['total_regression_solves'],0)
        e['fit_execution_accounting']['totals']['independent_model_check_canonical_solves']=0
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


if __name__=='__main__':unittest.main(verbosity=2)
