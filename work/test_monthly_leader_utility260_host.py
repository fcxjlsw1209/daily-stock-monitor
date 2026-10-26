"""Synthetic lifecycle/receipt tests; no market arrays or saved outcomes are read."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import monthly_leader_utility260_host as h


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
    for key,path in [('price_archive_sha256',h.ev.PRICE),('calendar_sha256',h.CALENDAR)]:plan['input_sha256'][str(path)]='hash';s[key]='hash'
    s['incumbent_cache_sha256']={name:'hash' for name in ('COMPLETION.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json')}
    for name,digest in s['incumbent_cache_sha256'].items():plan['input_sha256'][str(h.PRIOR/'prepared'/name)]=digest
    s.update(original_scores_exact=True,original_pre_scores_exact=True,original_predictive_models_refitted=0)
    return s,plan


class LifecycleTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','unconditional_leader_utility','state_leader_utility'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(10,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_joint_identity259')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))
        self.assertEqual(h.SOURCE_REVIEW.name,'after259_router_inputs_review.json')
        self.assertIn('MARKET_INPUTS.npz',h.PREP_FILES);self.assertNotIn('LABELS.npz',h.PREP_FILES)
    def test_state_is_only_promotable(self):
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
        s=dict(completed=[dict(directory='old',trials=5) for _ in range(254)]+[dict(directory=h.PRIOR.name,trials=6)])
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
        full={'n':2,**{key:np.array([.1,.2]) for key in ('event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p')}}
        small={'n':1,**{key:value[:1].copy() for key,value in full.items() if key!='n'}}
        h.bundle_parity(full,small,1);small['state_prediction'][0]=0.
        with self.assertRaises(AssertionError):h.bundle_parity(full,small,1)


def model_counts(n,fits=0):
    import monthly_leader_utility260_kernel as k
    c=dict.fromkeys(k.MODEL_COUNT_FIELDS,0)
    c.update(years=fits,intercept_estimates=fits,ridge_solves=fits,partial_events=int(n>0),cold_query_events=max(0,n-1))
    return c


def decision_counts(n,width=49):
    import monthly_leader_utility260_kernel as k
    c=dict.fromkeys(k.COUNT_FIELDS,0)
    c.update(events=n,symbols_per_event=width,partial_events=int(n>0),cold_events=max(0,n-1),incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1)
    return c


def proof(kind,n,width=49,fits=0):
    import monthly_leader_utility260_kernel as k
    import monthly_leader_utility260_independent as ind
    names={'inputs':'INPUT_FLAGS','targets':'TARGET_FLAGS','source':'SOURCE_FLAGS','models':'MODEL_FLAGS','incumbent':'INCUMBENT_FLAGS','selection':'SELECTION_FLAGS'}
    p=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,
        old_model_solves=0,conditional_ridge_solves=0,independent_canonical_solves=0,independent_svd_solves=0,original_episode_calls=0)
    p.update(dict.fromkeys(getattr(ind,names[kind]),True))
    if kind in ('incumbent','selection'):
        p.update(original_episode_calls=1 if kind=='incumbent' else 3,ordered_choices_checked=n if kind=='incumbent' else 3*n,
            full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.)
        if kind=='incumbent':p['candidate_selection_computed']=False
        else:p.update(counts=decision_counts(n,width),full_audit_supplied=True,full_bundle_supplied=True)
    else:
        p.update(exact_discrete=True)
        if kind in ('inputs','targets'):p['numerical_tolerance']=0.
        else:p.update(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.)
        if kind=='inputs':p.update(full_input_audit_supplied=True,counts=dict(training_rows=0,training_months=0,query_rows=n*width))
        elif kind=='targets':p.update(full_target_audit_supplied=True,counts=dict.fromkeys(k.TARGET_COUNT_FIELDS,0))
        elif kind=='source':
            c=dict(training_events=0,query_events=n,partial_events=int(n>0),state_vectors=max(n-1,0),price_observations=507*max(n-1,0))
            p.update(full_source_audit_supplied=True,counts=c,source_audit=dict(schema='synthetic',counts=c))
        elif kind=='models':p.update(full_model_audit_supplied=True,full_bundle_supplied=True,counts=model_counts(n,fits),intercept_estimates=fits,
            conditional_ridge_solves=fits,independent_canonical_solves=fits,independent_svd_solves=fits)
    return p


def run_proof(n):
    p={kind:proof('incumbent' if kind=='early_incumbent' else kind,n,fits=int(n>1)) for kind in ('early_incumbent','inputs','targets','source','models','incumbent','selection')}
    p['model_counts']=copy.deepcopy(p['models']['counts']);return p


def complete_proof():
    p=run_proof(122)
    p['cached_prefixes']=[dict(prefix=n,exact=True,incumbent=proof('incumbent',n),selection=proof('selection',n)) for n in range(123)]
    p['physical_prefixes']=[dict(prefix=n,exact=True,model_recomputed=True,**run_proof(n)) for n in (0,1,53,88)]
    p['physical_prefixes'].append(dict(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,**run_proof(122)))
    p.update(all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True)
    p['fit_execution_accounting']=h.execution_accounting(p)
    model=dict(model={'counts':p['model_counts']},target={'counts':p['targets']['counts']},market=p['source']['source_audit'])
    p['saved_model_audit_sha256']=h.value_digest(model)
    decision=dict(counts=p['selection']['counts'],incumbent_gate=p['incumbent'])
    return p,model,decision


def empty_router(gate):
    import monthly_leader_utility260_kernel as k
    return dict(schema='monthly_leader_utility260_router_v1',n=0,event_dates=np.array([],dtype='datetime64[D]'),active=np.array([],bool),
      intercept_prediction=np.array([]),state_prediction=np.array([]),intercept_p=np.array([]),state_p=np.array([]),control_gate=gate,
      audit=dict(schema='monthly_leader_utility260_model_audit_v1',annual=[],counts=model_counts(0),market_columns=list(k.COLUMNS),original_models_refitted=False))


class ReceiptTests(unittest.TestCase):
    def test_all_kind_receipts(self):
        for kind in ('inputs','targets','source','models','incumbent','selection'):h.receipt(proof(kind,2,fits=1),2,(),kind,width=49)
    def test_standard_flags_required_even_empty_flag_argument(self):
        import monthly_leader_utility260_independent as ind
        for kind,field in [('inputs',ind.INPUT_FLAGS[0]),('targets',ind.TARGET_FLAGS[0]),('source',ind.SOURCE_FLAGS[0]),('models',ind.MODEL_FLAGS[0]),('incumbent',ind.INCUMBENT_FLAGS[0]),('selection',ind.SELECTION_FLAGS[0])]:
            p=proof(kind,2);p[field]=False
            with self.assertRaises(ValueError):h.receipt(p,2,(),kind)
    def test_count_and_scope_types(self):
        for key,value in [('checked_events',True),('old_model_solves',1),('original_episode_calls',2),('producer_modules_imported',True),('computes_returns',True),('ordered_choices_checked',4)]:
            p=proof('selection',2);p[key]=value
            with self.assertRaises(ValueError):h.receipt(p,2,(),'selection')
    def test_exact_input_and_target_continuous_model_contract(self):
        for kind,key in [('inputs','numerical_tolerance'),('targets','numerical_tolerance'),('models','continuous_atol'),('source','rank_tolerance'),('selection','numerical_tolerance')]:
            p=proof(kind,2);p[key]=1e-8
            with self.assertRaises(ValueError):h.receipt(p,2,(),kind)
    def test_full_evidence_required(self):
        for kind,key in [('inputs','full_input_audit_supplied'),('targets','full_target_audit_supplied'),('source','full_source_audit_supplied'),('models','full_model_audit_supplied'),('selection','full_bundle_supplied'),('incumbent','full_pre_scores_supplied')]:
            p=proof(kind,2);p.pop(key)
            with self.assertRaises(ValueError):h.receipt(p,2,(),kind)
    def test_three_arm_width_and_episode_accounting(self):
        h.receipt(proof('selection',2,width=6),2,(),'selection',width=6)
        with self.assertRaises(ValueError):h.receipt(proof('selection',2,width=6),2,(),'selection',width=49)
        p=proof('selection',2);p['counts']['state_episode_calls']=0
        with self.assertRaises(ValueError):h.receipt(p,2,(),'selection')
    def test_source_counts_and_target_partition(self):
        p=proof('source',2);p['counts']['price_observations']-=1
        with self.assertRaises(ValueError):h.receipt(p,2,(),'source')
        p=proof('targets',2);p['counts']['valid_months']=1
        with self.assertRaises(ValueError):h.receipt(p,2,(),'targets')
    def test_only_router_solves_and_paired_intercept(self):
        for key,value in [('original_model_fits',1),('intercept_estimates',0),('active_query_events',2)]:
            p=proof('models',2,fits=1);p['counts'][key]=value
            with self.assertRaises(ValueError):h.receipt(p,2,(),'models')
        p=proof('models',2,fits=1);p['independent_svd_solves']=0
        with self.assertRaises(ValueError):h.receipt(p,2,(),'models')
    def test_actual_work_and_full_reuse(self):
        p,m,d=complete_proof();h.preparation_evidence(p,m,d)
        a=p['fit_execution_accounting'];self.assertEqual(a['fresh_model_invocations'],5)
        self.assertEqual(a['producer_conditional_ridge_solves'],3)
        self.assertEqual(a['total_linear_system_solves'],9)
        self.assertEqual(a['producer_original_episode_calls'],4*5+3*123)
        self.assertEqual(a['independent_actual_work']['original_episode_calls'],5*5+4*123)
        self.assertEqual(a['producer_original_model_fits'],0)
    def test_prefix_missing_repeat_or_full_recompute_rejected(self):
        for key in ('cached_prefixes','physical_prefixes'):
            p,m,d=complete_proof();p[key].pop()
            with self.assertRaises(ValueError):h.preparation_evidence(p,m,d)
        p,m,d=complete_proof();p['physical_prefixes'][-1]['model_recomputed']=True
        with self.assertRaises(ValueError):h.preparation_evidence(p,m,d)
    def test_counts_and_audits_not_unbound_flags(self):
        p,m,d=complete_proof();m['extra']='tamper'
        with self.assertRaises(ValueError):h.preparation_evidence(p,m,d)
        p,m,d=complete_proof();p['fit_execution_accounting']['total_linear_system_solves']+=1
        with self.assertRaises(ValueError):h.preparation_evidence(p,m,d)
        p,m,d=complete_proof();d['counts']=dict(d['counts'],state_changed_delta_cells=1)
        with self.assertRaises(ValueError):h.preparation_evidence(p,m,d)
    def test_incumbent_gate_exact_p_f_and_once(self):
        ind=SimpleNamespace(INCUMBENT_FLAGS=(),verify_incumbent=Mock(return_value=proof('incumbent',1)))
        gate,receipts=h.incumbent_callback(ind,{}, {},1,[{'symbols':['A']}],np.array([[1.]]),np.array([[2.]]))
        with self.assertRaises(AssertionError):gate([{'symbols':['A']}],np.array([[1.1]]),np.array([[2.]]))
        ind.verify_incumbent.assert_not_called()
        gate([{'symbols':['A']}],np.array([[1.]]),np.array([[2.]]))
        with self.assertRaises(ValueError):gate([{'symbols':['A']}],np.array([[1.]]),np.array([[2.]]))
    def test_n0_router_preservation(self):
        b=empty_router(proof('incumbent',0));h.bundle_preservation(b,0)
        b['audit']['original_models_refitted']=True
        with self.assertRaises(ValueError):h.bundle_preservation(b,0)
    def test_json_dates_are_values_not_repr(self):
        value={'date':np.array(['2020-01-31'],dtype='datetime64[D]'),'value':np.array([.3])}
        self.assertEqual(h.json_safe(value),{'date':['2020-01-31'],'value':[.3]})


class GateOrderTests(unittest.TestCase):
    def fixture(self):
        import monthly_leader_utility260_independent as actual_ind
        trace=[];arrays={'baseline':np.zeros((0,3))};meta={'monthly':{},'training':{}}
        inp=dict(event_dates=np.array([],dtype='datetime64[D]'),event_plain=np.empty((0,10)),labels=np.array([]))
        target=dict(dates=np.array([],dtype='datetime64[D]'),records=[],counts=proof('targets',0)['counts'])
        market=proof('source',0)['source_audit'];empty=np.empty((0,3));model=empty_router(proof('incumbent',0))
        def fitted(target,tx,qx,dates,*,control_validator):
            trace.append('fit');model['control_gate']=control_validator();return model
        def decisions(study,a,m,b,x,n,*,incumbent_validator,expected_incumbent_orders):
            trace.append('decisions');g=incumbent_validator([],empty,empty)
            return {arm:[] for arm in h.ARMS},{arm:empty.copy() for arm in h.ARMS},{arm:empty.copy() for arm in h.ARMS},dict(incumbent_gate=g,counts=decision_counts(0,3))
        kernel=SimpleNamespace(original_decisions=Mock(side_effect=lambda *a,**k:(trace.append('original') or [],empty,empty)),router_inputs=Mock(side_effect=lambda *a,**k:(trace.append('inputs') or inp)),build_expert_targets=Mock(side_effect=lambda *a:(trace.append('targets') or target)),build_market_inputs=Mock(side_effect=lambda *a:(trace.append('market') or empty,empty,market)),fit_router=Mock(side_effect=fitted),decisions=Mock(side_effect=decisions))
        ind=SimpleNamespace(**{name:getattr(actual_ind,name) for name in ('INPUT_FLAGS','TARGET_FLAGS','SOURCE_FLAGS','MODEL_FLAGS','INCUMBENT_FLAGS','SELECTION_FLAGS')})
        for name,kind in [('verify_inputs','inputs'),('verify_targets','targets'),('verify_market_inputs','source'),('verify_models','models'),('verify_incumbent','incumbent'),('verify_selection','selection')]:
            setattr(ind,name,Mock(side_effect=lambda *a,_kind=kind,**kw:(trace.append('verify_'+_kind) or proof(_kind,0,width=3))))
        return trace,arrays,meta,kernel,ind
    def test_n0_complete_phase_order_without_fits(self):
        trace,a,m,k,i=self.fixture()
        b,art,r,p=h.prepare_one(None,{},m,a,None,[],0,[],np.empty((0,3)),np.empty((0,3)),kernel=k,ind=i)
        self.assertEqual(trace,['original','verify_incumbent','inputs','verify_inputs','targets','verify_targets','market','verify_source','fit','verify_models','decisions','verify_incumbent','verify_selection'])
        self.assertEqual(b['audit']['counts']['ridge_solves'],0)
    def test_bad_original_stops_before_teacher(self):
        trace,a,m,k,i=self.fixture()
        with self.assertRaises(ValueError):h.prepare_one(None,{},m,a,None,[],0,[{'symbols':['BAD']}],None,None,kernel=k,ind=i)
        k.router_inputs.assert_not_called();k.fit_router.assert_not_called()
    def test_bad_source_stops_fits_and_candidate(self):
        trace,a,m,k,i=self.fixture();i.verify_market_inputs.return_value=None;i.verify_market_inputs.side_effect=lambda *a:dict(proof('source',0),full_source_audit_supplied=False)
        with self.assertRaises(ValueError):h.prepare_one(None,{},m,a,None,[],0,[],np.empty((0,3)),np.empty((0,3)),kernel=k,ind=i)
        k.fit_router.assert_not_called();k.decisions.assert_not_called()
    def test_bad_model_stops_candidate_selection(self):
        trace,a,m,k,i=self.fixture();i.verify_models.side_effect=lambda *a:dict(proof('models',0),passed=False)
        with self.assertRaises(ValueError):h.prepare_one(None,{},m,a,None,[],0,[],np.empty((0,3)),np.empty((0,3)),kernel=k,ind=i)
        k.decisions.assert_not_called()
    def test_teacher_cannot_mutate_verified_input(self):
        trace,a,m,k,i=self.fixture();original=k.build_expert_targets.side_effect
        def changed(inp):
            result=original(inp);inp['labels']=np.array([99.]);return result
        k.build_expert_targets.side_effect=changed
        with self.assertRaisesRegex(ValueError,'Teacher cannot rewrite'):h.prepare_one(None,{},m,a,None,[],0,[],np.empty((0,3)),np.empty((0,3)),kernel=k,ind=i)
        k.fit_router.assert_not_called();i.verify_targets.assert_not_called()
    def test_market_cannot_mutate_verified_target(self):
        trace,a,m,k,i=self.fixture();original=k.build_expert_targets.side_effect;held={}
        def target(inp):held['target']=original(inp);return held['target']
        k.build_expert_targets.side_effect=target;source=k.build_market_inputs.side_effect
        def changed(*args):
            answer=source(*args);held['target']['records'].append({'tampered':True});return answer
        k.build_market_inputs.side_effect=changed
        with self.assertRaisesRegex(ValueError,'Market builder cannot rewrite'):h.prepare_one(None,{},m,a,None,[],0,[],np.empty((0,3)),np.empty((0,3)),kernel=k,ind=i)
        k.fit_router.assert_not_called();i.verify_market_inputs.assert_not_called()
    def test_snapshot_does_not_convert_unselected_label_object(self):
        class Poison:
            def __float__(self):raise AssertionError('numeric conversion')
            def __str__(self):raise AssertionError('string conversion')
        source={'labels':np.asarray([Poison()],dtype=object),'n':0}
        self.assertEqual(h.source_mapping_snapshot(source),h.source_mapping_snapshot(source))


if __name__=="__main__":unittest.main()
