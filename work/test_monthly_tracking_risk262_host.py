"""Synthetic262 lifecycle/gate tests. No market arrays, loaders or outputs read."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import monthly_tracking_risk262_host as h

FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','stock_cells','mature_cells','warmup_cells','unmapped_warmup_cells','supported_cells','zero_total_cells','zero_tracking_cells','own_price_observations','sector_price_observations','stock_return_observations','sector_return_observations','tracking_return_observations','total_risk_statistics','tracking_risk_statistics','total_nonzero_deltas','tracking_nonzero_deltas','total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
I=SimpleNamespace(COUNT_FIELDS=FIELDS,FEATURE_FLAGS=('passed','feature_checked'),INCUMBENT_FLAGS=('passed','incumbent_checked'),SELECTION_FLAGS=('passed','selection_checked'),PREPARE_FLAGS=('passed','full_checked'),ORIGINAL_SOURCE_FLAGS=('passed','original_core_checked'))


def counts(n,width=49):
    c=dict.fromkeys(FIELDS,0);c.update(events=n,symbols_per_event=width,original_episode_calls=3,stock_cells=n,mature_cells=n,supported_cells=n,own_price_observations=65*n,sector_price_observations=65*n,total_risk_statistics=n,tracking_risk_statistics=n,stock_return_observations=63*n,sector_return_observations=63*n,tracking_return_observations=63*n)
    return c


def original(n):
    return dict(**dict.fromkeys(h.WORK_FIELDS,0),passed=True,original_core_checked=True,checked_events=n,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False,C_source_atol=1e-12,C_source_rtol=0.,rank_tolerance=0.,original_checked_cells=49*n)


def proof(kind,n,width=49):
    flags={'source':I.FEATURE_FLAGS,'cached_source':I.FEATURE_FLAGS,'incumbent':I.INCUMBENT_FLAGS,'selection':I.SELECTION_FLAGS,'full':I.PREPARE_FLAGS}[kind]
    r=dict.fromkeys(flags,True);r.update(checked_events=n,producer_modules_imported=False,computes_returns=False,original_episode_calls=0 if kind in ('source','cached_source') else 1 if kind=='incumbent' else 3,
      full_source_slot_inspections=130*n if kind in ('source','full') else 0,cached_slot_validations=130*n if kind in ('cached_source','selection') else 0,statistic_evaluations=0 if kind=='incumbent' else 2*n,rank_evaluations=2*n if kind in ('selection','full') else 0,numerical_tolerance=0.,rank_tolerance=0.,exact_discrete=True)
    if kind in ('source','cached_source'):r.update(full_features_supplied=True,source_raw_targets_none=True,candidate_selection_computed=False,source_rebuilt=kind=='source',feature_counts=counts(n,width))
    elif kind=='incumbent':r.update(full_scores_supplied=True,full_pre_scores_supplied=True,candidate_selection_computed=False,ordered_choices_checked=n)
    else:r.update(full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,counts=counts(n,width),ordered_choices_checked=3*n)
    return r


def item(n,cached=False):
    return dict(prefix=n,gates=dict(incumbent=proof('incumbent',n),source=proof('cached_source' if cached else 'source',n)),
      **{'selection' if cached else 'full':proof('selection' if cached else 'full',n)},counts=counts(n),exact=True,
      **({} if cached else dict(original_source=original(n),reused_full=False,source_recomputed=True)))


def prepared_fixture():
    features=[dict(q=i,total_target=[.5],tracking_target=[.5]) for i in range(122)];main=item(122)
    physical=[item(n) for n in h.PREFIXES[:-1]]+[dict(copy.deepcopy(main),reused_full=True,source_recomputed=False)]
    e=dict(main=main,physical_prefixes=physical,cached_prefixes=[item(n,True) for n in range(123)],full_features_sha256=h.digest(features),full_source_certificate_sha256=h.digest(main['gates']['source']),all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,predictive_model_fits=0)
    e['execution_accounting']=h.execution_accounting(e)
    c=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=counts(122))
    a=dict(counts=counts(122),events=list(range(122)),source_gate=main['gates']['source'],incumbent_gate=main['gates']['incumbent'])
    return e,c,I,a,features


def training():
    r=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(r) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['A','B','D']
    return m,o


def terminals():
    return dict(plan_sha256='plan',phases={p:dict(session_id=i+1,exit_code=0,chunk_id='chunk'+str(i),artifact_path=str(h.OUT/f),artifact_sha256='digest') for i,(p,f) in enumerate((('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')))})


def source_fixture():
    raw={'synthetic/'+str(i):'a'*64 for i in range(64)}
    bound=dict(symbol='SPCX',issuer_key='issuer',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='old')
    policy=dict(passed=True,source_admitted=False,strict_pit=False,publication_date='2026-06-15',original_history_bound=bound,path_bound_override=dict(symbol='SPCX',issuer_key='issuer',not_before='2026-06-12',available_from='2026-06-16',kind='issuer_actual_first_trade'))
    clocks=[]
    for q in range(122):
        t=300+q;clocks.append(dict(query_index=q,query_position=t,initial_partial=q==0,numeric_evaluation_required=q!=0,numerator_positions=[] if q==0 else [t-252,t-42],risk_positions=[] if q==0 else list(range(t-63,t+1)),required_positions=[] if q==0 else [t-252]+list(range(t-63,t+1)),numerator_dates=[] if q==0 else ['d']*2,risk_dates=[] if q==0 else ['d']*64,required_dates=[] if q==0 else ['d']*65))
    s=dict(schema='tracking_risk262_source_contract_v1',passed=True,metadata_checks_passed=True,metadata_only=True,raw_source_hashes_verified=True,source_admitted=False,strict_pit=False,prices_loaded=False,npz_decoded=False,loader_executed=False,actual_features_computed=False,actual_models_fitted=False,actual_orders_computed=False,actual_returns_computed=False,observers_modified=False,known_source_blockers=[],actual_source_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',actual_common_rank_support='UNKNOWN_UNTIL_REGISTERED_PREPARE',raw_source_count=64,raw_source_hashes=raw,panel_first_session='2008-01-02',calendar_sha256='calendar',metadata=dict(events=122,formal_events=121,width=49,stocks_per_event=32,stock_cells=3904,etfs=list(range(17))),clock_events=clocks,clock_fields=list(clocks[0]),mapping_records=[{}]*3904,listing_identity_cells=[{}]*5,early_listing_policy=dict(path='work/fake_policy.json',sha256='policy',policy=policy),listing_bounds=[bound],vox=dict(full2015_source_version_preserved=True,extension2014_used=False,price_field='adjClose'),structural_policy=dict(no_whole_event_missing_fallback=True))
    def read(path):
        if path==h.LOCAL/'PLAN.json':return dict(raw_source_hashes=raw)
        if path==h.ROOT/'work/fake_policy.json':return policy
        return dict(source_policy=dict(source_contract_binding=dict(path=str(h.SOURCE_REVIEW.relative_to(h.ROOT)),sha256=h.SOURCE_SHA)))
    def sha(path):return h.SOURCE_SHA if path==h.SOURCE_REVIEW else 'policy' if path==h.ROOT/'work/fake_policy.json' else 'calendar'
    return s,read,sha


class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','sector_total_risk_control','sector_tracking_risk_candidate'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(8,38,7))
        self.assertEqual(h.PRIOR.name,'monthly_weak_tree261')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))

    def test_only_tracking_candidate_can_promote(self):
        m,o=training();self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[2])
        for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=9.
        result=h.select_first88(m,o);self.assertFalse(result['admitted']);self.assertEqual(result['primary'],h.ARMS[0])

    def test_inc_gain_and_open_noninferiority(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training();m[h.ARMS[2]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_each_reference_requires_three_changes_across_segments(self):
        m,o=training();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[2]]);o[h.ARMS[1]][2]['symbols']=['Q']
        self.assertFalse(h.select_first88(m,o)['admitted'])
        for j in (54,55):o[h.ARMS[1]][j]['symbols']=['Q']
        self.assertTrue(h.select_first88(m,o)['admitted'])

    def test_inc_changed_months_require_both_eras(self):
        for ids in ((54,55),(1,2,3),(53,54,55)):
            m,o=training();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[2]][i]['symbols']=['D']
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_order_only_change_not_support(self):
        m,o=training()
        for i in (1,54,55):o[h.ARMS[2]][i]['symbols']=['C','A','B']
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_both_references_both_paths_risk_segments(self):
        for ref in (h.ARMS[0],h.ARMS[1]):
            for path in ('same_close_zero','next_open_zero'):
                for group,key,v in [('training','sharpe',1.76),('training','max_drawdown',-.26),('early','cagr',.43),('recent','cagr',.63)]:
                    m,o=training();m[ref][path][group][key]=v
                    self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_cagr_only_roundoff(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[2]]['same_close_zero']['training']['sharpe']=1.65-5e-13
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_undefined_candidate_or_control_rejects(self):
        for arm in h.ARMS[1:]:
            for path in ('same_close_zero','next_open_zero'):
                m,o=training();m[arm][path]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_bad_metrics_hardfail(self):
        for value in (None,float('nan'),float('inf'),True):
            m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=value
            with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m[h.ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_later_full_cost_cannot_rescue(self):
        m,o=training();m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.1
        expected=h.select_first88(m,o)
        for i in range(88,122):o[h.ARMS[2]][i]['symbols']=['LATER_WINNER']
        self.assertEqual(h.select_first88(m,o),expected)
        m[h.ARMS[2]]['next_open_cost']={'full122':{'cagr':100.}}
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_missing_arm_short_order_rejected(self):
        m,o=training();o[h.ARMS[2]]=o[h.ARMS[2]][:87]
        with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m.pop(h.ARMS[1])
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_source_final_bound_and_metadata_contract(self):
        s,r,f=source_fixture()
        with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f):h.source_review_check(s)
        s['early_listing_policy']['policy']['path_bound_override']['available_from']='2026-06-15'
        with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f),self.assertRaises(ValueError):h.source_review_check(s)

    def test_source_clocks_and_scope_rejected(self):
        for mode in ('clock','actual','oldbound','coverage','missing'):
            s,r,f=source_fixture()
            if mode=='clock':s['clock_events'][1]['required_positions'].pop()
            elif mode=='actual':s['prices_loaded']=True
            elif mode=='oldbound':s['listing_bounds'][0]['not_before']='2026-06-12'
            elif mode=='coverage':s['actual_source_coverage']='PASS'
            else:s['known_source_blockers']=['unresolved']
            with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f),self.assertRaises(ValueError):h.source_review_check(s)

    def test_source_map_canonicalization_and_rejections(self):
        x={'synthetic/'+str(i):'a'*64 for i in range(64)}
        self.assertEqual(h.canonical_sources(x),h.canonical_sources({str(h.ROOT/k):v for k,v in x.items()}))
        for name in ('../bad','/tmp/bad','CloudStorage/bad'):
            y=dict(x);y[name]=y.pop('synthetic/0')
            with self.assertRaises(ValueError):h.canonical_sources(y)
        y=dict(x);y[str(h.ROOT/'synthetic/1')]=y.pop('synthetic/0')
        with self.assertRaises(ValueError):h.canonical_sources(y)

    def test_all_receipt_kinds(self):
        for k in ('source','cached_source','incumbent','selection','full'):
            h.receipt(proof(k,0),0,I,k,width=49);h.receipt(proof(k,3,6),3,I,k,width=6)

    def test_missing_flags_and_numeric_contract(self):
        for k in ('source','cached_source','incumbent','selection','full'):
            for key in ('passed','producer_modules_imported','computes_returns','checked_events','original_episode_calls'):
                r=proof(k,3);r.pop(key)
                with self.assertRaises((ValueError,KeyError)):h.receipt(r,3,I,k,width=49)
        r=proof('source',3);r['rank_tolerance']=1e-10
        with self.assertRaises(ValueError):h.receipt(r,3,I,'source')

    def test_receipt_flags_cannot_be_disabled(self):
        r=proof('selection',3);r['passed']=False
        with self.assertRaises(ValueError):h.receipt(r,3,SimpleNamespace(**dict(I.__dict__,SELECTION_FLAGS=())), 'selection')
        r=proof('selection',3);r['full_audit_supplied']=False
        with self.assertRaises(ValueError):h.receipt(r,3,I,'selection')

    def test_receipt_boolean_counts_rejected(self):
        for key in ('checked_events','original_episode_calls','full_source_slot_inspections','cached_slot_validations','statistic_evaluations','rank_evaluations','ordered_choices_checked'):
            r=proof('selection',0);r[key]=False
            with self.assertRaises(ValueError):h.receipt(r,0,I,'selection')
        r=proof('full',1);r['counts']['new_model_fits']=False
        with self.assertRaises(ValueError):h.receipt(r,1,I,'full')

    def test_three_arm_order_scope_and_no_fits(self):
        r=proof('selection',3);r['ordered_choices_checked']=12
        with self.assertRaises(ValueError):h.receipt(r,3,I,'selection')
        r=proof('selection',3);r['counts']['predictive_model_fits']=1
        with self.assertRaises(ValueError):h.receipt(r,3,I,'selection')

    def test_width_and_source_selection_scope(self):
        r=proof('source',3,6)
        with self.assertRaises(ValueError):h.receipt(r,3,I,'source',width=49)
        r=proof('source',3);r['candidate_selection_computed']=True
        with self.assertRaises(ValueError):h.receipt(r,3,I,'source')
        r=proof('source',3);r['feature_counts']['tracking_changed_holdings']=1
        with self.assertRaises(ValueError):h.receipt(r,3,I,'source')

    def test_callbacks_enforce_order_and_frozen_orders(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features=Mock(return_value=proof('source',1)))
        iv,sv,got=h.callbacks(ind,{}, {},1,[{'symbols':['A']}],panel={},calendar=[],review={})
        with self.assertRaises(ValueError):sv([])
        with self.assertRaises(ValueError):iv([{'symbols':['B']}],None,None)
        self.assertFalse(ind.verify_incumbent.called);self.assertFalse(ind.verify_features.called)
        iv([{'symbols':['A']}],None,None);sv([])
        self.assertEqual(set(got),{'incumbent','source'})

    def test_cached_callback_binds_verified_full_prefix(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features_cached=Mock(return_value=proof('cached_source',1)))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],features=[{'q':0,'total_target':[.5],'tracking_target':[.5]}],certificate=proof('source',1))
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv([{'q':9}])
        self.assertFalse(ind.verify_features_cached.called);sv([{'q':0,'total_target':None,'tracking_target':None}])

    def test_cached_certificate_cannot_be_only_passed(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features_cached=Mock(return_value=proof('cached_source',1)))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],features=[{'q':0,'total_target':[.5],'tracking_target':[.5]}],certificate={'passed':True})
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv([{'q':0,'total_target':None,'tracking_target':None}])
        self.assertFalse(ind.verify_features_cached.called)

    def test_preparation_all_prefix_and_reuse_pass(self):h.preparation_check(*prepared_fixture())

    def test_preparation_missing_prefix_or_recomputed122_rejected(self):
        for change in ('cached','physical','reuse'):
            e,c,i,a,f=prepared_fixture()
            if change=='cached':e['cached_prefixes'].pop(7)
            elif change=='physical':e['physical_prefixes'].pop(1)
            else:e['physical_prefixes'][-1]['source_recomputed']=True
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a,f)

    def test_preparation_digest_and_gate_tamper_rejected(self):
        for change in ('features','certificate','audit'):
            e,c,i,a,f=prepared_fixture()
            if change=='features':f[0]['q']=8
            elif change=='certificate':e['full_source_certificate_sha256']='bad'
            else:a['incumbent_gate']=dict(a['incumbent_gate'],passed=False)
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a,f)

    def test_accounting_does_not_duplicate_full122(self):
        e,*_=prepared_fixture();a=h.execution_accounting(e)
        self.assertEqual(a['physical_preparations'],5);self.assertEqual(a['cached_decisions'],123)
        self.assertEqual(a['producer']['original_episode_calls'],384)
        self.assertEqual(a['producer']['full_source_slot_inspections'],264*130)
        self.assertEqual(a['producer']['cached_slot_validations'],7503*130)
        self.assertEqual(a['producer']['statistic_evaluations'],7767*2)
        self.assertEqual(a['independent']['original_episode_calls'],512)
        self.assertEqual(a['independent']['statistic_evaluations'],7767*4)
        self.assertEqual(a['total']['statistic_evaluations'],7767*6)
        self.assertEqual(a['return_vector_steps_computed'],63*3*7767*3)
        e['execution_accounting']['producer']['cached_slot_validations']=0
        _,c,i,au,f=prepared_fixture()
        with self.assertRaises(ValueError):h.preparation_check(e,c,i,au,f)

    def test_terminal_receipt_valid_null_instant(self):
        t=terminals();t['phases']['select']['session_id']=None
        with patch.object(h,'read',return_value=t),patch.object(h,'sha',return_value='digest'):h.phase_terminals_check('plan')

    def test_terminal_missing_phase_or_failed_exit(self):
        for mode in ('phase','exit','bool','chunk'):
            t=terminals()
            if mode=='phase':t['phases'].pop('diagnose')
            elif mode=='exit':t['phases']['prepare']['exit_code']=1
            elif mode=='bool':t['phases']['prepare']['exit_code']=False
            else:t['phases']['prepare']['chunk_id']=''
            with patch.object(h,'read',return_value=t),patch.object(h,'sha',return_value='digest'),self.assertRaises(ValueError):h.phase_terminals_check('plan')

    def test_terminal_wrong_artifact_or_digest(self):
        for key,value in [('artifact_path',str(h.OUT/'results/RESULTS.json')),('artifact_sha256','wrong'),('session_id',False)]:
            t=terminals();t['phases']['diagnose'][key]=value
            with patch.object(h,'read',return_value=t),patch.object(h,'sha',return_value='digest'),self.assertRaises(ValueError):h.phase_terminals_check('plan')

    def test_terminal_wrong_plan(self):
        with patch.object(h,'read',return_value=terminals()),self.assertRaises(ValueError):h.phase_terminals_check('other')

    def test_zero_fees_strict_every_stream(self):
        exports={a:{p:dict(TRADES=[{'cost':0.}],HOLDINGS=[{'fee':0.}]) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        results={a:{p:dict(metrics={'fees_paid':0.},slices={'full122':{'fees_paid':0.}}) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
        h.zero_cost_check(exports,results)
        for a in h.ARMS:
            for p in ('same_close_zero','next_open_zero'):
                bad=copy.deepcopy(exports);bad[a][p]['TRADES'][0]['cost']=1e-30
                with self.assertRaises(ValueError):h.zero_cost_check(bad,results)
                bad=copy.deepcopy(exports);bad[a][p]['HOLDINGS'][0]['fee']=1e-30
                with self.assertRaises(ValueError):h.zero_cost_check(bad,results)
        results[h.ARMS[0]]['same_close_zero']['metrics']['fees_paid']=1e-30
        with self.assertRaises(ValueError):h.zero_cost_check(exports,results)

    def test_outcome_scope(self):
        r=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')}) for p in h.ev.PATHS} for a in h.ARMS})
        h.outcome_check(r)
        for k,v in [('checked_paths',8),('checked_slices',44),('checked_training_groups',17),('terminal_liquidation',True)]:
            bad=copy.deepcopy(r);bad[k]=v
            with self.assertRaises(ValueError):h.outcome_check(bad)

    def test_predecessor_registration_counts(self):
        state=dict(completed=[dict(directory='old',trials=5) for _ in range(256)]+[dict(directory=h.PRIOR.name,trials=2)],current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')])
        h.registration_check(state,'p')
        state['completed'][-1]['trials']+=1
        with self.assertRaises(ValueError):h.registration_check(state,'p')

    def test_saved_review_requires_terminal_and_every_artifact_pin(self):
        paths=[h.PHASE_TERMINALS,h.OUT/'PLAN.json',h.CONTEXT,h.CONTRACT,h.OUT/'prepared/COMPLETION.json',h.OUT/'selected/FROZEN.json',h.OUT/'selected/TRAIN_RESULTS.json',h.OUT/'results/COMPLETION.json']+[h.OUT/'prepared'/p for p in h.PREP_FILES]+[h.OUT/'results'/p for p in h.RESULT_FILES]
        r=dict(passed=True,unresolved_blockers=[],plan_sha256='p',context_sha256='x',prepared_completion_sha256='x',results_completion_sha256='x',implementation_contract_sha256='x',input_sha256={str(p):'p' if p==h.OUT/'PLAN.json' else 'x' for p in paths})
        def sha(p):return 'p' if p==h.OUT/'PLAN.json' else 'x'
        with patch.object(h,'phase_terminals_check') as gate,patch.object(h,'read',return_value=r),patch.object(h,'sha',side_effect=sha),patch.object(h.old,'verify'):
            h.result_review_check('p');gate.assert_called_once_with('p')
            r['input_sha256'].pop(str(h.PHASE_TERMINALS))
            with self.assertRaises(ValueError):h.result_review_check('p')

    def test_original_source_gate_scope(self):
        h.original_source_check(original(2),2,I)
        for key,value in [('episode_calls',1),('new_target_ranks_computed',True),('C_source_atol',1e-9),('checked_events',True)]:
            r=original(2);r[key]=value
            with self.assertRaises(ValueError):h.original_source_check(r,2,I)

    def test_source_rank_stage_and_work_tamper(self):
        for key,value in [('source_raw_targets_none',False),('rank_evaluations',1),('statistic_evaluations',0),('full_source_slot_inspections',0)]:
            r=proof('source',1);r[key]=value
            with self.assertRaises(ValueError):h.receipt(r,1,I,'source')

    def test_unranked_only_consumed_body(self):
        class Poison(dict):
            def __deepcopy__(self,memo):raise AssertionError('future deepcopy')
        x=[dict(total_target=[.5],tracking_target=[.5],v=1),Poison()]
        self.assertEqual(h.unranked_prefix(x,0),[])
        self.assertEqual(h.unranked_prefix(x,1),[dict(total_target=None,tracking_target=None,v=1)])
        self.assertEqual(x[0]['total_target'],[.5])

    def test_physical_inputs_really_small(self):
        import numpy as np
        import pandas as pd
        days=['2020-01-30','2020-01-31','2020-02-03','2020-02-28','2020-03-02']
        arrays={'baseline':np.arange(6.).reshape(2,3),'eligible':np.ones((2,3),bool)}
        meta=dict(dates=[days[1],days[3]],ends=[days[3],days[4]],symbols=[['A','B','C']]*2,etfs=[])
        panel={'adj':pd.DataFrame({'A':range(5),'VOX':range(5)},index=pd.to_datetime(days))}
        aa,mm,pp,cc=h.physical_inputs(arrays,meta,panel,days,1)
        self.assertEqual(aa['baseline'].shape,(1,3));self.assertEqual(len(mm['dates']),1)
        self.assertEqual(list(pp['adj'].index),list(panel['adj'].index[:2]));self.assertEqual(cc,days[:3])
        aa,mm,pp,cc=h.physical_inputs(arrays,meta,panel,days,0)
        self.assertEqual(aa['baseline'].shape,(0,3));self.assertTrue(pp['adj'].empty);self.assertEqual(cc,[])

    def test_empty_domain_vector_passes_are_not_scalar_ranks(self):
        e,*_=prepared_fixture()
        for row in [e['main']]+e['physical_prefixes']+e['cached_prefixes']:
            row['counts']['supported_cells']=0
        a=h.execution_accounting(e)
        self.assertGreater(a['producer']['rank_evaluations'],0)
        for owner in ('producer','independent','total'):
            self.assertEqual(a[owner]['scalar_rank_evaluations'],0)
        self.assertEqual(a['rank_work']['producer']['full_scalar_rank_evaluations'],0)
        e,*_=prepared_fixture();a=h.execution_accounting(e)
        for owner in ('producer','independent'):
            self.assertEqual(a['rank_work'][owner]['full_scalar_rank_evaluations'],2*264)
            self.assertEqual(a['rank_work'][owner]['cached_scalar_rank_evaluations'],2*sum(range(123)))
            self.assertEqual(a[owner]['scalar_rank_evaluations'],2*(264+sum(range(123))))

    def test_paper_review_maps_only_two_exact_draft_snapshots(self):
        names=[str(p.relative_to(h.ROOT)) for p in (h.CONTEXT,h.CONTRACT)]
        snaps=['work/monthly_tracking_risk262_context.REVIEWED_DRAFT.json','work/monthly_tracking_risk262_IMPLEMENTATION_CONTRACT.REVIEWED_DRAFT.json']
        mapping={name:dict(snapshot_path=snap,sha256='draft'+str(i)) for i,(name,snap) in enumerate(zip(names,snaps))}
        obj=dict(reviewed_draft_snapshots=mapping,pre_registration_review=dict(path=str(h.PRE_REG_REVIEW.relative_to(h.ROOT)),sha256='paper'))
        paper=dict(passed=True,blockers=[],input_sha256=dict(zip(names,('draft0','draft1'))))
        paper['input_sha256']['work/other_source.json']='other'
        pins={h.ROOT/snaps[0]:'draft0',h.ROOT/snaps[1]:'draft1',h.PRE_REG_REVIEW:'paper',h.ROOT/'work/other_source.json':'other'}
        with patch.object(h,'sha',side_effect=lambda path:pins.get(path,'wrong')):
            h.paper_review_check(obj,obj,paper)
            for mode in ('third_mapping','wrong_snapshot','wrong_draft_hash','missing_draft','changed_other'):
                bad=copy.deepcopy(obj);pr=copy.deepcopy(paper)
                if mode=='third_mapping':bad['reviewed_draft_snapshots']['other']={}
                elif mode=='wrong_snapshot':bad['reviewed_draft_snapshots'][names[0]]['snapshot_path']='work/other_source.json'
                elif mode=='wrong_draft_hash':bad['reviewed_draft_snapshots'][names[0]]['sha256']='other'
                elif mode=='missing_draft':pr['input_sha256'].pop(names[0])
                else:pr['input_sha256']['work/other_source.json']='changed'
                with self.assertRaises(ValueError):h.paper_review_check(bad,bad,pr)

    def test_original_source_rejects_new_work(self):
        for key in h.WORK_FIELDS:
            r=original(1);r[key]=1
            with self.assertRaises(ValueError):h.original_source_check(r,1,I)

    def test_original_source_failure_precedes_kernel_prepare(self):
        import sys
        kernel=SimpleNamespace(prepare=Mock())
        ind=SimpleNamespace(**I.__dict__,verify_original_source=Mock(return_value=dict(original(0),passed=False)))
        with patch.dict(sys.modules,monthly_tracking_risk262_kernel=kernel,monthly_tracking_risk262_independent=ind):
            with self.assertRaises(ValueError):h.prepare_one(None,{},dict(dates=[]),{},[],{},0,[])
        kernel.prepare.assert_not_called()



if __name__=='__main__':unittest.main(verbosity=2)
