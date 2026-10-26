"""Synthetic264 lifecycle/gate tests. No market arrays, loaders or outputs read."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import monthly_gain_survival264_host as h

FIELDS=('events', 'symbols_per_event', 'eligible_cells', 'initial_partial_events', 'calm_events', 'active_events', 'original_core_cells', 'original_finite_cells', 'required_cells', 'price_observations', 'gain_return_observations', 'positive_gain_anchors', 'survival_comparisons', 'endpoint_comparisons', 'gain_statistics', 'quality_ratio_evaluations', 'no_positive_gain_cells', 'nonpositive_original_cells', 'positive_raw_multiplications', 'empty_domain_events', 'available_events', 'endpoint_identity_events', 'survival_identity_events', 'endpoint_ranked_cells', 'survival_ranked_cells', 'endpoint_nonzero_deltas', 'survival_nonzero_deltas', 'endpoint_changed_holdings', 'survival_changed_holdings', 'candidate_control_changed_holdings', 'original_episode_calls', 'new_model_fits', 'predictive_model_fits')
OE=('query_index','date','initial_partial','calm','u0_indices','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63')
I=SimpleNamespace(COUNT_FIELDS=FIELDS,ORIGINAL_EVENT_KEYS=OE,FEATURE_FLAGS=('passed','feature_checked'),INCUMBENT_FLAGS=('passed','incumbent_checked'),SELECTION_FLAGS=('passed','selection_checked'),PREPARE_FLAGS=('passed','full_checked'),ORIGINAL_SOURCE_FLAGS=('passed','original_core_checked'))


def counts(n,width=49):
    c=dict.fromkeys(FIELDS,0);c.update(events=n,symbols_per_event=width,original_episode_calls=3,eligible_cells=width*n,active_events=n,original_core_cells=width*n,original_finite_cells=n,required_cells=n,price_observations=253*n,gain_return_observations=252*n,positive_gain_anchors=n,survival_comparisons=n,endpoint_comparisons=n,gain_statistics=n,quality_ratio_evaluations=2*n,positive_raw_multiplications=2*n,available_events=n,endpoint_ranked_cells=n,survival_ranked_cells=n)
    return c


def original(n,width=49):
    events=[dict(query_index=q,date='fake'+str(q),initial_partial=False,calm=False,u0_indices=[0],original_raw=[1.]+[None]*(width-1),original_rank=[.5]+[0.]*(width-1),frozen_core=[.26]+[0.]*(width-1),rebuilt_core=[.26]+[0.]*(width-1),core_abs_error=[0.]*width,original_sigma63=[.1]*width) for q in range(n)]
    r=dict.fromkeys(h.WORK_FIELDS,0);r.update(passed=True,original_core_checked=True,checked_events=n,episode_calls=0,new_target_ranks_computed=False,candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False,C_source_atol=1e-12,C_source_rtol=0.,rank_tolerance=0.,original_checked_cells=width*n,original_finite_cells=n,original_source_cells=width*n,original_source_price_observations=sum((301+q)*width for q in range(n)),original_events=events,u0_indices=[[0] for q in range(n)],original_source_snapshot_price_observations=(300+n)*width if n else 0,original_source_snapshot_hashes=int(n>0),original_source_snapshot_sha256='a'*64)
    return r


def feature_rows(n,width=49):
    out=[]
    for e in original(n,width)['original_events']:
        f={k:copy.deepcopy(e[k]) for k in ('query_index','date','initial_partial','calm','u0_indices')};f['symbols']=['S'+str(j) for j in range(width)]
        f['cells']=[dict(column_index=j,symbol=f['symbols'][j],original_finite=j==0,**{k:e[k][j] for k in OE[5:]}) for j in range(width)]
        f.update(endpoint_target=[.5]+[None]*(width-1),survival_target=[.5]+[None]*(width-1));out.append(f)
    return out


def proof(kind,n,width=49):
    flags={'source':I.FEATURE_FLAGS,'cached_source':I.FEATURE_FLAGS,'incumbent':I.INCUMBENT_FLAGS,'selection':I.SELECTION_FLAGS,'full':I.PREPARE_FLAGS}[kind]
    r=dict.fromkeys(h.WORK_FIELDS,0);r.update(dict.fromkeys(flags,True));r.update(checked_events=n,producer_modules_imported=False,computes_returns=False,original_episode_calls=0 if kind in ('source','cached_source') else 1 if kind=='incumbent' else 3,numerical_tolerance=0.,rank_tolerance=0.,exact_discrete=True)
    if kind!='incumbent':
        full=kind in ('source','full');selecting=kind in ('selection','full')
        r.update(h.descriptive_work(counts(n,width),full=full,selecting=selecting))
        if full:r.update(original_source_proof_sha256=h.digest(original(n,width)),original_source_proof_events=n,original_source_snapshot_sha256='a'*64,original_source_snapshot_price_observations=(300+n)*width if n else 0,original_source_snapshot_hashes=int(n>0))
    if kind in ('source','cached_source'):r.update(full_features_supplied=True,source_raw_targets_none=True,candidate_selection_computed=False,source_rebuilt=kind=='source',feature_counts=counts(n,width))
    elif kind=='incumbent':r.update(full_scores_supplied=True,full_pre_scores_supplied=True,candidate_selection_computed=False,ordered_choices_checked=n)
    else:r.update(full_features_supplied=True,full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True,counts=counts(n,width),ordered_choices_checked=3*n)
    return r


def item(n,cached=False):
    cert=original(122 if cached else n);anchor=h.validated_original_anchor(cert,122 if cached else n,I,width=49)
    projection=h.original_prefix_certificate(cert,anchor,n)
    work=dict(fullpast_rows=300+n if n else 0,union_symbols=sorted('S'+str(j) for j in range(49)) if n else [],original_source_price_observations=(300+n)*49 if n else 0,original_source_cells=n*49)
    return dict(prefix=n,gates=dict(incumbent=proof('incumbent',n),source=proof('cached_source' if cached else 'source',n)),
      **{'selection' if cached else 'full':proof('selection' if cached else 'full',n)},counts=counts(n),exact=True,
      **(dict(original_source_certificate_sha256=anchor['certificate_sha256'],consumed_original_source_sha256=h.digest(projection)) if cached else dict(original_source=cert,original_source_anchor=anchor,consumed_original_source_sha256=h.digest(projection),producer_original_source_work=work,reused_full=False,source_recomputed=True)))


def prepared_fixture():
    features=feature_rows(122);main=item(122)
    physical=[item(n) for n in h.PREFIXES[:-1]]+[dict(copy.deepcopy(main),reused_full=True,source_recomputed=False)]
    e=dict(main=main,physical_prefixes=physical,cached_prefixes=[item(n,True) for n in range(123)],full_features_sha256=h.digest(features),full_source_certificate_sha256=h.digest(main['gates']['source']),full_original_source_certificate_sha256=h.digest(main['original_source']),all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,predictive_model_fits=0)
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
    import pandas as pd
    import gain_survival264_sources as helper
    sessions=pd.bdate_range('2008-01-02','2026-09-01').strftime('%Y-%m-%d').tolist()
    dates=['2016-08-17']+pd.date_range('2016-08-01','2026-08-31',freq='BME').strftime('%Y-%m-%d').tolist()
    meta=dict(dates=dates,symbols=[['S'+str(j) for j in range(49)] for _ in dates])
    raw={'synthetic/'+str(i):'a'*64 for i in range(64)}
    bounds=[dict(symbol='IPO'+str(i),issuer_key='issuer'+str(i),not_before='2026-06-08',available_from='2026-06-09') for i in range(2)]
    metadata_path=h.ROOT/'work/fake264_meta.json'
    source=dict(schema=helper.SOURCE_SCHEMA,passed=True,metadata_only=True,source_admitted=False,strict_pit=False,prices_loaded=False,npz_decoded=False,actual_features_orders_returns_computed=False,registered_strategy=False,potential_clock_is_not_state_or_U0_admission=True,actual253_slot_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',raw_source_count=64,raw_source_hashes=raw,panel_first_session='2008-01-02',events=122,width=49,formal_monthends=121,calendar_sha256='calendar',origin_session_position=0,metadata_path=str(metadata_path),metadata_sha256='meta',clock_events=[helper.clock(q,d,meta['symbols'][q],sessions) for q,d in enumerate(dates)],listing_bounds=copy.deepcopy(bounds),identity_cells=helper.identities(meta,bounds,122),source_policy=dict(additional_provider=False,additional_sector_mapping=False,additional_vox=False))
    def read(path):
        if path==h.LOCAL/'PLAN.json':return dict(raw_source_hashes=raw)
        if path==h.CALENDAR:return dict(sessions=sessions)
        if path==metadata_path:return meta
        if path==h.ROOT/'work/peer239_maturity_policy.json':return dict(securities=bounds)
        return dict(source_policy=dict(source_contract_binding=dict(path=str(h.SOURCE_REVIEW.relative_to(h.ROOT)),sha256=h.SOURCE_SHA)))
    def sha(path):return h.SOURCE_SHA if path==h.SOURCE_REVIEW else 'meta' if path==metadata_path else 'calendar'
    return source,read,sha


class HostTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate'))
        self.assertEqual((len(h.PREP_FILES),len(h.RESULT_FILES),len(h.code_names)),(8,38,7))
        self.assertEqual(h.PRIOR.name,'monthly_risk_clock263')
        self.assertEqual(h.PREFIXES,(0,1,53,88,122))

    def test_only_survival_candidate_can_promote(self):
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
        s['listing_bounds'][0]['not_before']='2026-06-12'
        with patch.object(h,'read',side_effect=r),patch.object(h,'sha',side_effect=f),self.assertRaises(ValueError):h.source_review_check(s)

    def test_source_clocks_and_scope_rejected(self):
        for mode in ('clock','actual','oldbound','coverage','missing'):
            s,r,f=source_fixture()
            if mode=='clock':s['clock_events'][1]['required_positions'].pop()
            elif mode=='actual':s['prices_loaded']=True
            elif mode=='oldbound':s['listing_bounds'][0]['not_before']='2026-06-12'
            elif mode=='coverage':s['actual253_slot_coverage']='PASS'
            else:s['source_policy']['additional_vox']=True
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
        r=proof('source',3);r['feature_counts']['survival_changed_holdings']=1
        with self.assertRaises(ValueError):h.receipt(r,3,I,'source')

    def test_callbacks_enforce_order_and_frozen_orders(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features=Mock(return_value=proof('source',1)))
        iv,sv,got=h.callbacks(ind,{}, {},1,[{'symbols':['A']}],original_certificate=original(1),original_anchor=h.validated_original_anchor(original(1),1,I),original_proof=object(),panel={},calendar=[],review={})
        with self.assertRaises(ValueError):sv([])
        with self.assertRaises(ValueError):iv([{'symbols':['B']}],None,None)
        self.assertFalse(ind.verify_incumbent.called);self.assertFalse(ind.verify_features.called)
        iv([{'symbols':['A']}],None,None);sv(h.unranked_prefix(feature_rows(1),1))
        self.assertEqual(set(got),{'incumbent','source'})

    def test_cached_callback_binds_verified_full_prefix(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features_cached=Mock(return_value=proof('cached_source',1)))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],original_certificate=original(1),original_anchor=h.validated_original_anchor(original(1),1,I),features=feature_rows(1),certificate=proof('source',1))
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv([dict(feature_rows(1)[0],query_index=9)])
        self.assertFalse(ind.verify_features_cached.called);sv(h.unranked_prefix(feature_rows(1),1))

    def test_cached_certificate_cannot_be_only_passed(self):
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features_cached=Mock(return_value=proof('cached_source',1)))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],original_certificate=original(1),original_anchor=h.validated_original_anchor(original(1),1,I),features=feature_rows(1),certificate={'passed':True})
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv(h.unranked_prefix(feature_rows(1),1))
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
            if change=='features':f[0]['query_index']=8
            elif change=='certificate':e['full_source_certificate_sha256']='bad'
            else:a['incumbent_gate']=dict(a['incumbent_gate'],passed=False)
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a,f)

    def test_accounting_does_not_duplicate_full122(self):
        e,*_=prepared_fixture();a=h.execution_accounting(e)
        self.assertEqual(a['physical_preparations'],5);self.assertEqual(a['cached_decisions'],123)
        self.assertEqual(a['producer']['original_episode_calls'],384)
        self.assertEqual(a['producer']['full_source_slot_inspections'],264*253)
        self.assertEqual(a['producer']['cached_slot_validations'],7503*253)
        self.assertEqual(a['producer']['statistic_evaluations'],7767)
        self.assertEqual(a['independent']['original_episode_calls'],512)
        self.assertEqual(a['independent']['statistic_evaluations'],7767*2)
        self.assertEqual(a['total']['statistic_evaluations'],7767*3)
        self.assertEqual(a['return_vector_steps_computed'],252*7767*3)
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
        state=dict(completed=[dict(directory='old',trials=5) for _ in range(257)]+[dict(directory='old',trials=0),dict(directory=h.PRIOR.name,trials=3)],current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='p')])
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
        x=[dict(endpoint_target=[.5],survival_target=[.5],v=1),Poison()]
        self.assertEqual(h.unranked_prefix(x,0),[])
        self.assertEqual(h.unranked_prefix(x,1),[dict(endpoint_target=None,survival_target=None,v=1)])
        self.assertEqual(x[0]['endpoint_target'],[.5])

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

    def test_empty_domain_has_no_rank_calls(self):
        e,*_=prepared_fixture()
        for row in [e['main']]+e['physical_prefixes']+e['cached_prefixes']:
            row['counts'].update(endpoint_ranked_cells=0,survival_ranked_cells=0,available_events=0)
            for key in ('selection','full'):
                if key in row:row[key].update(rank_evaluations=0,scalar_rank_evaluations=0)
        a=h.execution_accounting(e)
        for owner in ('producer','independent','total'):
            self.assertEqual(a[owner]['scalar_rank_evaluations'],0);self.assertEqual(a[owner]['rank_evaluations'],0)

    def test_paper_review_binds_preserved_drafts_and_all_interfaces(self):
        names=['work/monthly_gain_survival264_context.REVIEW_DRAFT.json','work/monthly_gain_survival264_IMPLEMENTATION_CONTRACT.REVIEW_DRAFT.json']
        docs=dict(zip(names,('draft0','draft1')));docs['work/source_note.json']='source'
        obj=dict(reviewed_draft_sha256={p:docs[p] for p in sorted(names)},pre_registration_review=dict(path=str(h.PRE_REG_REVIEW.relative_to(h.ROOT)),sha256='paper'))
        paper=dict(passed=True,draft_only=True,blocking_findings=[],reviewed_documents=docs,referenced_interfaces_verified={'core':dict(path='work/core.py',sha256='core')})
        pins={h.ROOT/p:v for p,v in docs.items()};pins[h.ROOT/'work/core.py']='core';pins[h.PRE_REG_REVIEW]='paper'
        with patch.object(h,'sha',side_effect=lambda path:pins.get(path,'wrong')):
            h.paper_review_check(obj,obj,paper)
            for mode in ('missingdraft','changedsource','changedinterface','wrongfinalbinding','blocker'):
                bad=copy.deepcopy(obj);pr=copy.deepcopy(paper)
                if mode=='missingdraft':pr['reviewed_documents'].pop(names[0])
                elif mode=='changedsource':pr['reviewed_documents']['work/source_note.json']='changed'
                elif mode=='changedinterface':pr['referenced_interfaces_verified']['core']['sha256']='changed'
                elif mode=='wrongfinalbinding':bad['reviewed_draft_sha256'][names[0]]='changed'
                else:pr['blocking_findings']=['failure']
                with self.assertRaises(ValueError):h.paper_review_check(bad,bad,pr)

    def test_original_source_rejects_new_work(self):
        for key in h.WORK_FIELDS:
            if key in ('original_source_price_observations','original_source_cells','original_source_snapshot_price_observations','original_source_snapshot_hashes'):continue
            r=original(1);r[key]=1
            with self.assertRaises(ValueError):h.original_source_check(r,1,I)

    def test_original_u0_and_numeric_evidence_bound_before_new_oracle(self):
        for mode in ('u0','raw','sigma','core','date','width'):
            f=feature_rows(1)
            if mode=='u0':f[0]['u0_indices']=[]
            elif mode=='date':f[0]['date']='different'
            elif mode=='width':f[0]['cells'].pop()
            else:f[0]['cells'][0][{'raw':'original_raw','sigma':'original_sigma63','core':'frozen_core'}[mode]]+=1e-12
            ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features=Mock(return_value=proof('source',1)))
            iv,sv,_=h.callbacks(ind,{}, {},1,[{}],original_certificate=original(1),original_anchor=h.validated_original_anchor(original(1),1,I),original_proof=object(),panel={},calendar=[],review={})
            iv([{}],None,None)
            with self.assertRaises(ValueError):sv(f)
            ind.verify_features.assert_not_called()

    def test_source_verifier_input_mutation_rejected(self):
        f=h.unranked_prefix(feature_rows(1),1)
        def mutate(*args,**kwargs):args[5][0]['cells'][0]['original_raw']=999.;return proof('source',1)
        ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',1)),verify_features=Mock(side_effect=mutate))
        iv,sv,_=h.callbacks(ind,{}, {},1,[{}],original_certificate=original(1),original_anchor=h.validated_original_anchor(original(1),1,I),original_proof=object(),panel={},calendar=[],review={})
        iv([{}],None,None)
        with self.assertRaises(ValueError):sv(f)

    def test_original_certificate_binding_and_work_tamper_rejected(self):
        for mode in ('fullhash','cachedhash','producerwork','oraclework'):
            e,c,i,a,f=prepared_fixture()
            if mode=='fullhash':e['full_original_source_certificate_sha256']='wrong'
            elif mode=='cachedhash':e['cached_prefixes'][2]['original_source_certificate_sha256']='wrong'
            elif mode=='producerwork':e['main']['producer_original_source_work']['original_source_price_observations']+=1
            else:e['main']['full']['original_source_price_observations']+=1
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a,f)

    def test_original_fullpast_owner_work_is_distinct(self):
        e,*_=prepared_fixture();a=h.execution_accounting(e)
        self.assertEqual(a['producer']['original_source_price_observations'],sum((300+n)*49 for n in (122,1,53,88)))
        self.assertEqual(a['independent']['original_source_price_observations'],sum(sum((301+q)*49 for q in range(n)) for n in (122,0,1,53,88)))
        self.assertNotEqual(a['producer']['original_source_price_observations'],a['independent']['original_source_price_observations'])

    def test_cached_certificate_never_reads_future_bodies_or_rehashes_full(self):
        import sys
        class Poison(dict):
            def __iter__(self):raise AssertionError('future certificate iteration')
            def items(self):raise AssertionError('future certificate serialization')
            def __deepcopy__(self,memo):raise AssertionError('future certificate copying')
        valid=original(4);anchor=h.validated_original_anchor(valid,4,I);features=feature_rows(4)
        for n in (0,2):
            cert=copy.deepcopy(valid);cert['original_events'][n:]=[Poison({'blocked':object()}) for _ in range(4-n)];cert['u0_indices'][n:]=[Poison({'blocked':object()}) for _ in range(4-n)]
            with self.assertRaises((TypeError,AssertionError)):h.digest(cert)
            def decide(study,arrays,meta,f,count,**kwargs):
                ig=kwargs['incumbent_validator']([{}]*n,None,None);sg=kwargs['source_validator'](h.unranked_prefix(features,n))
                return {a:[{}]*n for a in h.ARMS},{},{},dict(counts=counts(n),incumbent_gate=ig,source_gate=sg)
            ind=SimpleNamespace(**I.__dict__,verify_incumbent=Mock(return_value=proof('incumbent',n)),verify_features_cached=Mock(return_value=proof('cached_source',n)),verify_selection=Mock(return_value=proof('selection',n)))
            kernel=SimpleNamespace(decisions=decide)
            with patch.dict(sys.modules,monthly_gain_survival264_kernel=kernel,monthly_gain_survival264_independent=ind):
                _,record=h.decisions(None,{}, {},features,n,[{}]*n,proof('source',4),original_certificate=cert,original_anchor=anchor,width=49)
            self.assertEqual(record['original_source_certificate_sha256'],anchor['certificate_sha256'])
            self.assertEqual(record['consumed_original_source_sha256'],h.digest(h.original_prefix_certificate(valid,anchor,n)))

    def test_consumed_original_body_tamper_rejected_before_new_oracle(self):
        cert=original(4);anchor=h.validated_original_anchor(cert,4,I)
        for mode in ('raw','u0','missing'):
            bad=copy.deepcopy(cert)
            if mode=='raw':bad['original_events'][0]['original_raw'][0]+=1e-12
            elif mode=='u0':bad['u0_indices'][0]=[]
            else:bad['original_events']=[]
            with self.assertRaises(ValueError):h.original_prefix_certificate(bad,anchor,2)

    def test_full_anchor_must_come_from_verified_gate(self):
        cert=original(1);anchor=h.validated_original_anchor(cert,1,I)
        with self.assertRaises(ValueError):h.original_prefix_certificate(cert,dict(anchor),1)
        with self.assertRaises(ValueError):h._OriginalAnchor(dict(anchor))
        with self.assertRaises(TypeError):anchor['certificate_sha256']='unknown'
        for change in ('passed','event','u0'):
            bad=copy.deepcopy(cert)
            if change=='passed':bad['passed']=False
            elif change=='event':bad['original_events'][0]['original_raw'][0]=None
            else:bad['u0_indices'][0]=[]
            with self.assertRaises(ValueError):h.validated_original_anchor(bad,1,I)

    def test_metadata_work_does_not_claim_cached_full_validation(self):
        e,*_=prepared_fixture();w=e['execution_accounting']['original_certificate_metadata_work']
        self.assertEqual(w['full_certificate_validations'],5);self.assertEqual(w['full_certificate_hashes'],5)
        self.assertEqual(w['full_event_hashes'],264);self.assertEqual(w['prefix_projection_validations'],128)
        self.assertEqual(w['prefix_event_hashes'],7767);self.assertEqual(w['cached_event_hashes'],7503)
        self.assertEqual(w['cached_full_certificate_validations'],0);self.assertEqual(w['cached_full_certificate_hashes'],0)

    def test_saved_anchor_json_roundtrip_and_tamper(self):
        import json
        e,c,i,a,f=prepared_fixture();saved=json.loads(json.dumps(e,allow_nan=False))
        self.assertIs(type(saved['main']['original_source_anchor']),dict)
        self.assertIs(type(saved['main']['original_source_anchor']['original_event_sha256']),list)
        h.preparation_check(saved,c,i,a,f)
        saved['main']['original_source_anchor']['original_event_sha256'][0]='changed'
        with self.assertRaises(ValueError):h.preparation_check(saved,c,i,a,f)

    def test_original_source_failure_precedes_kernel_prepare(self):
        import sys
        kernel=SimpleNamespace(prepare=Mock())
        ind=SimpleNamespace(**I.__dict__,verify_original_source=Mock(return_value=dict(original(0),passed=False)))
        with patch.dict(sys.modules,monthly_gain_survival264_kernel=kernel,monthly_gain_survival264_independent=ind):
            with self.assertRaises(ValueError):h.prepare_one(None,{},dict(dates=[]),{},[],{},0,[])
        kernel.prepare.assert_not_called()



class GainSourceTests(unittest.TestCase):
    def test_formation_clock_is_all253_old_slots(self):
        source,_,_=source_fixture()
        for e in source['clock_events'][1:]:
            t=e['query_position'];positions=list(range(t-294,t-41))
            self.assertEqual(e['formation_positions'],positions)
            self.assertEqual(e['required_positions'],positions)
            self.assertEqual((e['required_count'],e['return_count']),(253,252))
            self.assertEqual(e['current_positions'],list(range(t-63,t+1)))
            self.assertNotIn('lag_positions',e)
        partial=source['clock_events'][0]
        for k in ('numerator','current','formation','required'):
            self.assertEqual(partial[k+'_positions'],[]);self.assertEqual(partial[k+'_dates'],[])
        self.assertEqual((partial['required_count'],partial['return_count']),(0,0))

    def test_source_counts_require_nonpositive_members_and_both_full_ranks(self):
        c=counts(1);c['nonpositive_original_cells']=1;c['positive_raw_multiplications']=0
        h.counts_check(c,1,I)
        for key,value in [('price_observations',252),('gain_return_observations',251),('endpoint_ranked_cells',0),('quality_ratio_evaluations',1)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):h.counts_check(bad,1,I)
        bad=dict(c,zero_risk_fallback_events=1)
        with self.assertRaises(ValueError):h.counts_check(bad,1,I)

    def test_saved_original_proof_reuse_cannot_be_relabelled(self):
        for key,value in [('original_source_proof_sha256','wrong'),('original_source_proof_events',121),('original_source_snapshot_sha256','b'*64),('original_source_snapshot_price_observations',1),('original_source_snapshot_hashes',2)]:
            e,c,i,a,f=prepared_fixture();e['main']['full'][key]=value
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a,f)
        for kind in ('source','full'):
            r=proof(kind,1);r['original_source_cells']=49
            with self.assertRaises(ValueError):h.receipt(r,1,I,kind)

    def test_snapshot_hash_work_is_distinct_from_pandas_reconstruction(self):
        e,*_=prepared_fixture();a=h.execution_accounting(e)
        expected=sum((300+n)*49 for n in (122,1,53,88))
        self.assertEqual(a['producer']['original_source_snapshot_price_observations'],0)
        self.assertEqual(a['independent']['original_source_snapshot_price_observations'],3*expected)
        self.assertEqual(a['independent']['original_source_snapshot_hashes'],12)
        for kind in ('cached_source','incumbent','selection'):
            r=proof(kind,1);r['original_source_snapshot_price_observations']=1
            with self.assertRaises(ValueError):h.receipt(r,1,I,kind)

    def test_prepare_one_issues_one_capability_for_raw_and_full(self):
        import sys
        cap=object();f=[];orders={a:[] for a in h.ARMS};scores={};pres={}
        def prepare(study,arrays,meta,panel,calendar,review,n,**kwargs):
            ig=kwargs['incumbent_validator']([],scores,pres)
            sg=kwargs['source_validator'](f)
            return f,orders,scores,pres,dict(counts=counts(0),incumbent_gate=ig,source_gate=sg)
        ind=SimpleNamespace(**I.__dict__,verify_original_source=Mock(return_value=original(0)),validated_original_proof=Mock(return_value=cap),verify_incumbent=Mock(return_value=proof('incumbent',0)),verify_features=Mock(return_value=proof('source',0)),verify_prepared=Mock(return_value=proof('full',0)))
        with patch.dict(sys.modules,monthly_gain_survival264_kernel=SimpleNamespace(prepare=prepare),monthly_gain_survival264_independent=ind):
            _,record=h.prepare_one(None,{},dict(dates=[]),{},[],{},0,[],width=49)
        ind.verify_original_source.assert_called_once()
        ind.validated_original_proof.assert_called_once()
        self.assertIs(ind.verify_features.call_args.kwargs['original_proof'],cap)
        self.assertIs(ind.verify_prepared.call_args.kwargs['original_proof'],cap)
        self.assertEqual(record['full']['original_source_price_observations'],0)


if __name__=='__main__':unittest.main(verbosity=2)
