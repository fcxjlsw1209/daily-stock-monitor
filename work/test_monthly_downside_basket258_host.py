"""Synthetic258 lifecycle/gate tests. No market arrays, loaders or outputs read."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import monthly_downside_basket258_host as h

FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','active_events','shortlist_cells','source_available_cells','source_price_observations','enumerated_trios','feasible_trios','trio_statistics','formation_return_observations','zero_downside_trios','zero_symmetric_trios','missing_source_events','available_events','downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
I=SimpleNamespace(COUNT_FIELDS=FIELDS,FEATURE_FLAGS=('passed','feature_checked'),INCUMBENT_FLAGS=('passed','incumbent_checked'),SELECTION_FLAGS=('passed','selection_checked'),PREPARE_FLAGS=('passed','full_checked'))

def counts(n,width=49):
    c=dict.fromkeys(FIELDS,0);c.update(events=n,symbols_per_event=width,original_episode_calls=1,trio_statistics=n,formation_return_observations=252*n)
    return c


def proof(kind,n,width=49):
    flags={'source':I.FEATURE_FLAGS,'cached_source':I.FEATURE_FLAGS,'incumbent':I.INCUMBENT_FLAGS,'selection':I.SELECTION_FLAGS,'full':I.PREPARE_FLAGS}[kind]
    r=dict.fromkeys(flags,True);r.update(checked_events=n,producer_modules_imported=False,computes_returns=False,original_episode_calls=2 if kind=='full' else 1,
      descriptive_path_evaluations=n if kind in ('source','full') else 0,cached_path_validations=n if kind in ('cached_source','selection','full') else 0)
    if kind in ('source','cached_source','full'):r.update(continuous_atol=2e-10,continuous_rtol=2e-10,nav_atol=2e-12,nav_rtol=2e-12,rank_tolerance=0.,exact_discrete=True)
    else:r['numerical_tolerance']=0.
    if kind in ('source','cached_source'):r.update(full_features_supplied=True,candidate_selection_computed=False,source_rebuilt=kind=='source',feature_counts=counts(n,width))
    elif kind=='incumbent':r.update(full_scores_supplied=True,full_pre_scores_supplied=True,candidate_selection_computed=False,ordered_choices_checked=n)
    else:r.update(full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,counts=counts(n,width),ordered_choices_checked=3*n,descriptive_path_returns_checked=True)
    return r


def item(n,cached=False):
    return dict(prefix=n,gates=dict(incumbent=proof('incumbent',n),source=proof('cached_source' if cached else 'source',n)),
      **{'selection' if cached else 'full':proof('selection' if cached else 'full',n)},counts=counts(n),exact=True,
      **({} if cached else dict(reused_full=False,source_recomputed=True)))


def prepared_fixture():
    features=[{'q':i} for i in range(122)];main=item(122)
    physical=[item(n) for n in h.PREFIXES[:-1]]+[dict(copy.deepcopy(main),reused_full=True,source_recomputed=False)]
    e=dict(main=main,physical_prefixes=physical,cached_prefixes=[item(n,True) for n in range(123)],full_features_sha256=h.digest(features),full_source_certificate_sha256=h.digest(main['gates']['source']),all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,predictive_model_fits=0)
    e['execution_accounting']=h.execution_accounting(e)
    c=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=counts(122))
    a=dict(counts=counts(122),events=list(range(122)),source_gate=main['gates']['source'],incumbent_gate=main['gates']['incumbent'])
    return e,c,I,a,features


def training():
    r=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(r) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for p in m[h.ARMS[1]]:m[h.ARMS[1]][p]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o


def terminals():
    return dict(plan_sha256='plan',phases={p:dict(session_id=i+1,exit_code=0,chunk_id='chunk'+str(i),artifact_path=str(h.OUT/f),artifact_sha256='digest') for i,(p,f) in enumerate((('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')))})


def source_fixture():
    raw={'synthetic/'+str(i):'a'*64 for i in range(64)}
    s=dict(passed=True,metadata_only=True,calendar_clock_verified=True,metadata_axes_verified=True,raw_source_hashes_verified=True,raw_source_map_matches_original=True,listing_identity_metadata_verified=True,source_window_recomputed_253=True,strict_pit=False,registered_strategy=False,actual_shortlist_computed=False,prices_loaded=False,npz_decoded=False,raw_source_count=64,raw_source_hashes=raw,calendar=dict(sha256='calendar',prices=253,return_intervals=252,windows=[{}]*122),metadata=dict(events=122,width=49,formal_monthends=121),draft_path=str(h.LEGACY_DRAFT.relative_to(h.ROOT)),draft_sha256=h.LEGACY_DRAFT_SHA,source_scope=dict(VOX_used=False,sector_mapping_used=False,extra_sources=[]))
    def read(path):
        return dict(raw_source_hashes=raw) if path==h.LOCAL/'PLAN.json' else dict(input_sha256={str(h.SOURCE_REVIEW.relative_to(h.ROOT)):h.SOURCE_SHA})
    def sha(path):return h.SOURCE_SHA if path==h.SOURCE_REVIEW else h.LEGACY_DRAFT_SHA if path==h.LEGACY_DRAFT else 'calendar'
    return s,read,sha


class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','basket_downside_moment','basket_symmetric_moment'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(8,38,6))
        self.assertEqual(h.PRIOR.name,'monthly_correlation_distance257_v2')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))

    def test_only_downside_can_promote(self):
        m,o=training();self.assertEqual(h.select_first88(m,o)['primary'],h.ARMS[1])
        for p in m[h.ARMS[2]]:m[h.ARMS[2]][p]['training']['cagr']=9.
        result=h.select_first88(m,o);self.assertFalse(result['admitted']);self.assertEqual(result['primary'],h.ARMS[0])

    def test_inc_gain_and_open_noninferiority(self):
        for p,v in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training();m[h.ARMS[1]][p]['training']['cagr']=v
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_symmetric_one_distinct_month_suffices(self):
        m,o=training();o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[1]]);o[h.ARMS[2]][2]['symbols']=['Q']
        self.assertTrue(h.select_first88(m,o)['admitted'])
        o[h.ARMS[2]]=copy.deepcopy(o[h.ARMS[1]])
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_inc_changed_months_require_both_eras(self):
        for ids in ((54,55),(1,2,3),(53,54,55)):
            m,o=training();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[0]])
            for i in ids:o[h.ARMS[1]][i]['symbols']=['D']
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_order_only_change_not_support(self):
        m,o=training()
        for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['C','A','B']
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_both_references_both_paths_risk_segments(self):
        for ref in (h.ARMS[0],h.ARMS[2]):
            for path in ('same_close_zero','next_open_zero'):
                for group,key,v in [('training','sharpe',1.76),('training','max_drawdown',-.26),('early','cagr',.43),('recent','cagr',.63)]:
                    m,o=training();m[ref][path][group][key]=v
                    self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_cagr_only_roundoff(self):
        m,o=training();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[1]]['same_close_zero']['training']['sharpe']=1.65-5e-13
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_undefined_candidate_or_control_rejects(self):
        for arm in h.ARMS[1:]:
            for path in ('same_close_zero','next_open_zero'):
                m,o=training();m[arm][path]['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_bad_metrics_hardfail(self):
        for value in (None,float('nan'),float('inf'),True):
            m,o=training();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=value
            with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m[h.ARMS[0]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_later_full_cost_cannot_rescue(self):
        m,o=training();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.1
        expected=h.select_first88(m,o)
        for i in range(88,122):o[h.ARMS[1]][i]['symbols']=['LATER_WINNER']
        self.assertEqual(h.select_first88(m,o),expected)
        m[h.ARMS[1]]['next_open_cost']={'full122':{'cagr':100.}}
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_missing_arm_short_order_rejected(self):
        m,o=training();o[h.ARMS[1]]=o[h.ARMS[1]][:87]
        with self.assertRaises(ValueError):h.select_first88(m,o)
        m,o=training();m.pop(h.ARMS[2])
        with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_source_legacy_draft_accepted_not_new_draft(self):
        s,r,f=source_fixture()
        with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f):h.source_review_check(s)
        s['draft_path']='work/monthly_downside_basket258_DRAFT01.json'
        with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f),self.assertRaises(ValueError):h.source_review_check(s)

    def test_source_clocks_and_scope_rejected(self):
        for field in ('prices','return_intervals'):
            s,r,f=source_fixture();s['calendar'][field]-=1
            with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f),self.assertRaises(ValueError):h.source_review_check(s)
        s,r,f=source_fixture();s['prices_loaded']=True
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
        for key in ('checked_events','original_episode_calls','descriptive_path_evaluations','cached_path_validations','ordered_choices_checked'):
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
        r=proof('source',3);r['feature_counts']['downside_changed_holdings']=1
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
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],features=[{'q':0}],certificate=proof('source',1))
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv([{'q':9}])
        self.assertFalse(ind.verify_features_cached.called);sv([{'q':0}])

    def test_cached_certificate_cannot_be_only_passed(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features_cached=Mock(return_value=proof('cached_source',1)))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],features=[{'q':0}],certificate={'passed':True})
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv([{'q':0}])
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
        self.assertEqual(a['producer_original_episode_calls'],128)
        self.assertEqual(a['producer_descriptive_path_evaluations'],264)
        self.assertEqual(a['producer_cached_path_validations'],7503)
        self.assertEqual(a['independent_original_episode_calls'],389)
        self.assertEqual(a['producer_total_path_moment_calculations'],7767)
        self.assertEqual(a['independent_total_path_moment_calculations'],15798)
        self.assertEqual(a['total_path_moment_calculations'],23565)
        self.assertEqual(a['total_formation_return_steps_computed'],23565*252)
        e['execution_accounting']['producer_cached_path_validations']=0
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
        state=dict(completed=[dict(directory='old',trials=5) for _ in range(252)]+[dict(directory=h.PRIOR.name,trials=10)],current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')])
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


if __name__=='__main__':unittest.main(verbosity=2)
