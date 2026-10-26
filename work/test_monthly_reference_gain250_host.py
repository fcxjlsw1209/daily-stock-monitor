"""Synthetic lifecycle, selection and complete-audit regressions, no source data."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import monthly_reference_gain250_host as h


def training_fixture():
    ref=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))
    m={a:{p:copy.deepcopy(ref) for p in ('same_close_zero','next_open_zero')} for a in h.ARMS}
    for path in m[h.ARMS[1]]:m[h.ARMS[1]][path]['training']['cagr']=.52
    o={a:[dict(symbols=['A','B','C']) for i in range(122)] for a in h.ARMS}
    for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['A','B','D']
    return m,o


def gate_receipt(n,source):
    common=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False)
    if source:common.update(source_checked=True,episode_calls=0,new_target_ranks_computed=False,C_source_atol=1e-12,C_source_rtol=0.,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.)
    else:common.update(incumbent_checked=True,ordered_choices_checked=n,numerical_tolerance=0.)
    return common


def gate_ind():return SimpleNamespace(SOURCE_GATE_FLAGS=('passed','source_checked'),INCUMBENT_GATE_FLAGS=('passed','incumbent_checked'))


def gates(n,source=True):
    d=dict(incumbent=gate_receipt(n,False))
    if source:d['source']=gate_receipt(n,True)
    return d


def receipt_fixture():
    ind=gate_ind();ind.SELECTION_FLAGS=('passed','exact_checked');ind.PREPARE_FLAGS=('passed','math_checked')
    def counts(n):return dict(events=n,new_model_fits=0,predictive_model_fits=0)
    def sc(n):return dict(passed=True,exact_checked=True,ordered_choices_checked=3*n,checked_events=n,numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,counts=counts(n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    def fc(n):return dict(passed=True,math_checked=True,full_features_supplied=True,original_core_checked=True,volume_clocks_checked=True,C_source_atol=1e-12,C_source_rtol=0.,ordered_choices_checked=3*n,checked_events=n,continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.,exact_discrete=True,producer_modules_imported=False,computes_returns=False,counts=counts(n),full_scores_supplied=True,full_pre_scores_supplied=True,full_audit_supplied=True)
    e=dict(all123_order_prefixes=True,original_arrays_unchanged=True,original_eligible_unchanged=True,new_model_fits=0,selection=sc(122),full=fc(122),pre_candidate_gates=gates(122),all123_cached_control_gates=[dict(prefix=n,gates=gates(n,False)) for n in range(123)],physical_prefixes=[dict(prefix=n,selection=sc(n),full=fc(n),counts=counts(n),pre_candidate_gates=gates(n),exact=True) for n in h.PREFIXES])
    completion=dict(new_model_fits=0,new_predictive_model_fits=0,new_returns_computed=False,events=122,descriptive_counts=counts(122))
    return e,completion,ind,dict(counts=counts(122),events=list(range(122)))


def source_fixture():
    raw={str(i):'digest' for i in range(64)}
    source=dict(schema='monthly_reference_gain250_metadata_source_review_v1',passed=True,metadata_only=True,
        source_admitted=False,strict_pit=False,registered_strategy=False,actual_features_orders_returns_computed=False,
        prices_loaded=False,volumes_loaded=False,npz_decoded=False,raw_source_count=64,raw_source_hashes=raw,
        calendar_path=str(h.CALENDAR.relative_to(h.ROOT)),calendar_sha256='fixed',
        metadata_path=str((h.LOCAL/'META.json').relative_to(h.ROOT)),metadata_sha256='fixed',
        panel_first_session='2008-01-02',events=122,width=49,formal_monthends=121,
        actual_cell_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',
        source_scope=dict(VOX_used=False,sector_mapping_used=False,new_downloads=False,extra_sources=[],original49_including_ETFs=True,eligibility_filter_used=False),
        source_policy=dict(mode='DECLARED_ASSUMPTION_ARCHIVED_MIXED_STORED_VOLUME',strict_pit=False,unit_continuity='UNKNOWN',zero_volume_rule='unresolved_zero_volume_whole_event_incumbent_fallback'),
        automatic_known_unresolved_transition_fallback_ids=[],empty_transition_list_certifies_continuity=False)
    binding=dict(raw_source_hashes=raw,input_sha256={},source=dict(panel_first_session='2008-01-02'))
    for key in ('panel_manifest','definition','definition_review','math_review','source_review','zero_policy','transition_ledger'):
        source[key+'_path']='work/synthetic-'+key+'.json';source[key+'_sha256']='fixed'
        if key=='panel_manifest':source['input_sha256']={source[key+'_path']:'fixed'}
        else:binding['input_sha256'][source[key+'_path']]='fixed'
    return source,binding



def outcome_fixture():
    return dict(passed=True,checked_paths=9,checked_slices=45,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{p[0]:dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')}) for p in h.ev.PATHS} for a in h.ARMS})


class HostTests(unittest.TestCase):
    def test_zero_policy_and_contract_cannot_be_substituted(self):
        source,binding=source_fixture()
        with patch.object(h,'read',side_effect=lambda path: binding if path==h.CONTEXT else dict(raw_source_hashes=source['raw_source_hashes'])),patch.object(h,'sha',return_value='fixed'):
            h.source_review_check(source)
            for field in ('zero_volume_rule',):
                bad=copy.deepcopy(source);bad['source_policy'][field]='accept_zero_as_actual_trading'
                with self.assertRaises(ValueError):h.source_review_check(bad)
            for key in ('math_review_sha256','zero_policy_sha256'):
                bad=copy.deepcopy(source);bad[key]='changed'
                with self.assertRaises(ValueError):h.source_review_check(bad)

    def test_new_math_and_funding_files_are_explicit(self):
        for stem in ('math','math_independent','adapter'):
            self.assertIn('monthly_reference_gain250_'+stem+'.py',h.code_names)
            self.assertNotIn('monthly_volume_breadth249_'+stem+'.py',h.code_names)

    def test_scope(self):
        self.assertEqual(h.ARMS,('incumbent_local67','lagged_volume_reference_gain','lagged_equal_reference_gain_control'))
        self.assertEqual(len(h.RESULT_FILES),38);self.assertEqual(len(h.PREP_FILES),8)
        self.assertEqual(len(h.code_names),12);self.assertEqual(h.PRIOR.name,'monthly_volume_breadth249_v2')
        self.assertIn('FEATURES.json',h.PREP_FILES)

    def test_admission_requires_frozen_incumbent_guards(self):
        result=h.select_first88(*training_fixture())
        self.assertTrue(result['admitted']);self.assertEqual(result['primary'],h.ARMS[1])
        self.assertEqual(set(result['guards']),{h.ARMS[0],h.ARMS[2]})
        for g in result['guards'].values():self.assertEqual(set(g),{'close','open','support'})


    def test_matched_mean_control_is_guard_not_promotable_backup(self):
        m,o=training_fixture()
        m[h.ARMS[2]]['same_close_zero']['training']['cagr']=.53
        result=h.select_first88(m,o)
        self.assertFalse(result['admitted']);self.assertEqual(result['primary'],h.ARMS[0])

    def test_risk_and_segment_guards_apply_to_mean_reference(self):
        for path in ('same_close_zero','next_open_zero'):
            for group,key,value in [('training','sharpe',1.76),('training','max_drawdown',-.26),('early','cagr',.43),('recent','cagr',.63)]:
                m,o=training_fixture();m[h.ARMS[2]][path][group][key]=value
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_only_training_cagr_uses_inherited_roundoff_tolerance(self):
        m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-5e-13
        self.assertTrue(h.select_first88(m,o)['admitted'])
        m[h.ARMS[1]]['same_close_zero']['training']['cagr']=.505-2e-12
        self.assertFalse(h.select_first88(m,o)['admitted'])
        m,o=training_fixture();m[h.ARMS[1]]['same_close_zero']['training']['sharpe']=1.65-5e-13
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_close_requires_complete_saved_evidence_review(self):
        required=[h.OUT/'PLAN.json',h.CONTEXT,h.OUT/'prepared/COMPLETION.json',h.OUT/'selected/FROZEN.json',h.OUT/'selected/TRAIN_RESULTS.json',h.OUT/'results/COMPLETION.json']
        required += [h.OUT/'prepared'/name for name in h.PREP_FILES]
        required += [h.OUT/'results'/name for name in h.RESULT_FILES]
        review=dict(passed=True,unresolved_blockers=[],plan_sha256='fixed',context_sha256='fixed',prepared_completion_sha256='fixed',results_completion_sha256='fixed',input_sha256={str(p):'fixed' for p in required})
        with patch.object(h,'sha',return_value='fixed'),patch.object(h.old,'verify'):
            with patch.object(h,'read',return_value=review):h.result_review_check('fixed')
            for key,value in [('passed',False),('unresolved_blockers',['blocker']),('plan_sha256','bad'),('context_sha256','bad'),('prepared_completion_sha256','bad'),('results_completion_sha256','bad'),('input_sha256',{})]:
                broken=copy.deepcopy(review);broken[key]=value
                with patch.object(h,'read',return_value=broken),self.assertRaises(ValueError):h.result_review_check('fixed')
            for path in required:
                broken=copy.deepcopy(review);broken['input_sha256'].pop(str(path))
                with patch.object(h,'read',return_value=broken),self.assertRaises(ValueError):h.result_review_check('fixed')

    def test_later34_holdings_not_used(self):
        m,o=training_fixture();expected=h.select_first88(m,o)
        for a in o:
            for i in range(88,122):o[a][i]['symbols']=['X','Y','Z']
        self.assertEqual(expected,h.select_first88(m,o))




    def test_incumbent_gain_and_nextopen_guards(self):
        for path,value in [('same_close_zero',.504),('next_open_zero',.49)]:
            m,o=training_fixture();m[h.ARMS[1]][path]['training']['cagr']=value
            self.assertFalse(h.select_first88(m,o)['admitted'])




    def test_risk_and_segment_guards_on_both_paths(self):
        for path in ('same_close_zero','next_open_zero'):
            for split,key,value in [('training','sharpe',1.64),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
                m,o=training_fixture();m[h.ARMS[1]][path][split][key]=value
                self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_change_support_against_each_reference(self):
        for a in (h.ARMS[0],h.ARMS[2]):
            m,o=training_fixture();o[a]=copy.deepcopy(o[h.ARMS[1]])
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_changes_need_three_months_and_both_segments(self):
        for months in ((54,55),(53,54,55),(1,2,3)):
            m,o=training_fixture();o[h.ARMS[1]]=copy.deepcopy(o[h.ARMS[0]])
            for i in months:o[h.ARMS[1]][i]['symbols']=['A','B','D']
            self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_reordering_is_not_changed_holdings(self):
        m,o=training_fixture()
        for i in (1,54,55):o[h.ARMS[1]][i]['symbols']=['C','B','A']
        self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_extra_or_missing_arms_rejected(self):
        for which in ('metrics','orders'):
            m,o=training_fixture();(m if which=='metrics' else o).pop(h.ARMS[1])
            with self.assertRaises(ValueError):h.select_first88(m,o)

    def test_predecessor_counter_scope_and_best_independence(self):
        s=dict(completed=[dict(directory='old',trials=5) for i in range(244)]+[dict(directory=h.PRIOR.name,trials=28)],best_training_selected_candidate=dict(directory='unrelated_new_best'))
        h.initial_state(s)
        for field,value in [('trials',31),('directory','wrong_predecessor')]:
            bad=copy.deepcopy(s);bad['completed'][-1][field]=value
            with self.assertRaises(ValueError):h.initial_state(bad)
        s['completed'].pop()
        with self.assertRaises(ValueError):h.initial_state(s)

    def test_predecessor_completion_bound_to_central(self):
        s=dict(completed=[dict(completion=dict(path=str(h.PRIOR/'COMPLETION.json'),sha256='fixed'))])
        with patch.object(h,'sha',return_value='fixed'):
            h.predecessor_binding(s)
            for key,value in [('path',str(h.PRIOR/'OTHER.json')),('sha256','wrong')]:
                bad=copy.deepcopy(s);bad['completed'][-1]['completion'][key]=value
                with self.assertRaises(ValueError):h.predecessor_binding(bad)

    def test_queue_binding(self):
        s=dict(current=h.OUT.name,registered_queue=[dict(directory=h.OUT.name,plan_sha256='fixed')])
        with patch.object(h,'initial_state'):
            h.registration_check(s,'fixed')
            for field,value in [('current','wrong'),('registered_queue',[]),('registered_queue',[dict(directory=h.OUT.name,plan_sha256='wrong')])]:
                bad=copy.deepcopy(s);bad[field]=value
                with self.assertRaises(ValueError):h.registration_check(bad,'fixed')

    def test_complete_preparation(self):h.preparation_check(*receipt_fixture())

    def test_full_audit_is_independent_and_complete(self):
        for key,val in [('math_checked',False),('ordered_choices_checked',365),('checked_events',121),('producer_modules_imported',True),('computes_returns',True)]:
            e,c,i,a=receipt_fixture();e['full'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_descriptive_path_and_features_must_be_checked(self):
        for key in ('full_features_supplied','original_core_checked','volume_clocks_checked'):
            for value in (False,None):
                e,c,i,a=receipt_fixture();e['full'][key]=value
                with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_numerical_tolerances_and_discrete_gates_cannot_drift(self):
        for key,val in [('continuous_atol',1e-5),('continuous_rtol',1e-5),('rank_tolerance',2e-10),('exact_discrete',False),('C_source_atol',2e-12),('C_source_rtol',1e-12)]:
            e,c,i,a=receipt_fixture();e['full'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_selection_scope_is_exact(self):
        for key,val in [('exact_checked',False),('ordered_choices_checked',365),('checked_events',121),('numerical_tolerance',2e-10),('producer_modules_imported',True),('computes_returns',True)]:
            e,c,i,a=receipt_fixture();e['selection'][key]=val
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_all_full_arrays_and_audit_required(self):
        for kind in ('full','selection'):
            for key in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied'):
                e,c,i,a=receipt_fixture();e[kind][key]=False
                with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_all_physical_prefixes_required(self):
        for mutation in ('missing','duplicate','false','wrongscope'):
            e,c,i,a=receipt_fixture()
            if mutation=='missing':e['physical_prefixes'].pop()
            if mutation=='duplicate':e['physical_prefixes'][-1]=e['physical_prefixes'][-2]
            if mutation=='false':e['physical_prefixes'][1]['exact']=False
            if mutation=='wrongscope':e['physical_prefixes'][2]['full']['checked_events']=122
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_audit_count_bindings(self):
        for where in ('full','selection','audit','completion','prefix'):
            e,c,i,a=receipt_fixture()
            if where in ('full','selection'):e[where]['counts']['extra']=1
            if where=='audit':a['counts']['extra']=1
            if where=='completion':c['descriptive_counts']['extra']=1
            if where=='prefix':e['physical_prefixes'][2]['counts']['extra']=1
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_no_returns_or_fits_before_selection(self):
        for key,value in [('new_model_fits',1),('new_predictive_model_fits',1),('new_returns_computed',True),('events',121)]:
            e,c,i,a=receipt_fixture();c[key]=value
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_source_review_binds_metadata_and_clock(self):
        source,binding=source_fixture()
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=binding):
            h.source_review_check(source)
            for key in ('passed','metadata_only','source_admitted','strict_pit','registered_strategy','actual_features_orders_returns_computed','prices_loaded','volumes_loaded','npz_decoded'):
                bad=copy.deepcopy(source);bad[key]=not bad[key]
                with self.assertRaises(ValueError):h.source_review_check(bad)


    def test_source_review_rejects_wrong_clocks_axes_and_bindings(self):
        source,binding=source_fixture()
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=binding):
            for key,value in [('schema','wrong'),('calendar_sha256','wrong'),('metadata_sha256','wrong'),('panel_first_session','2015-01-02'),('panel_first_session','2010-01-04'),('events',121),('width',48),('formal_monthends',122),('raw_source_count',63),('raw_source_hashes',{}),('actual_cell_coverage','COMPLETE'),('original_U0_and_core_parity','PASS')]:
                bad=copy.deepcopy(source);bad[key]=value
                with self.assertRaises(ValueError):h.source_review_check(bad)
            for key in ('panel_manifest','definition','definition_review','math_review','source_review','zero_policy','transition_ledger'):
                bad=copy.deepcopy(source);bad[key+'_sha256']='wrong'
                with self.assertRaises(ValueError):h.source_review_check(bad)
            for key,value in [('VOX_used',True),('sector_mapping_used',True),('extra_sources',['new'])]:
                bad=copy.deepcopy(source);bad['source_scope'][key]=value
                with self.assertRaises(ValueError):h.source_review_check(bad)


    def test_independent_outcome_complete(self):h.outcome_check(outcome_fixture())

    def test_independent_outcome_rejects_scope_and_liquidation(self):
        for key,value in [('passed',False),('checked_paths',6),('checked_slices',30),('independent_accounts',False),('independent_of_production_model_and_engine',False),('terminal_liquidation',True)]:
            c=outcome_fixture();c[key]=value
            with self.assertRaises(ValueError):h.outcome_check(c)
        for key,value in [('passed',False),('monthly_executions',121),('slices',{})]:
            c=outcome_fixture();c['paths'][h.ARMS[1]]['same_close_zero'][key]=value
            with self.assertRaises(ValueError):h.outcome_check(c)
        c=outcome_fixture();c['paths'].pop(h.ARMS[1])
        with self.assertRaises(ValueError):h.outcome_check(c)


    def test_prepare_failure_record_is_explicit_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'PLAN.json').write_text('{}')
            with patch.object(h,'OUT',output),patch.object(h,'sha',return_value='fixed'),patch.object(h,'prepare',side_effect=ValueError('original clock mismatch')):
                with self.assertRaisesRegex(ValueError,'original clock mismatch'):h.prepare_recorded('fixed')
                path=output/'PREPARE_FAILURE.json';first=path.read_bytes();receipt=json.loads(first)
                self.assertEqual(receipt['status'],'FAILED');self.assertFalse(receipt['successful_preparation_claim'])
                self.assertFalse(receipt['prepared_completion_present']);self.assertTrue(receipt['source_or_control_failure_not_relaxed'])
                with self.assertRaises(ValueError):h.prepare_recorded('fixed')
                self.assertEqual(first,path.read_bytes())

    def test_invalid_plan_cannot_write_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'PLAN.json').write_text('{}')
            with patch.object(h,'OUT',output),patch.object(h,'sha',return_value='different'),patch.object(h,'prepare',side_effect=ValueError('bad plan')):
                with self.assertRaises(ValueError):h.prepare_recorded('fixed')
                self.assertFalse((output/'PREPARE_FAILURE.json').exists())


    def test_pre_candidate_source_and_incumbent_gates_are_mandatory(self):
        for where in ('full','cached','physical'):
            e,c,i,a=receipt_fixture()
            target=e['pre_candidate_gates'] if where=='full' else e['all123_cached_control_gates'][53]['gates'] if where=='cached' else e['physical_prefixes'][2]['pre_candidate_gates']
            target.pop('incumbent')
            with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)
        e,c,i,a=receipt_fixture();e['all123_cached_control_gates'].pop()
        with self.assertRaises(ValueError):h.preparation_check(e,c,i,a)

    def test_source_gate_cannot_have_computed_candidate_or_returns(self):
        ind=gate_ind()
        for key,value in [('source_checked',False),('checked_events',4),('episode_calls',1),('new_target_ranks_computed',True),('candidate_selection_computed',True),('computes_returns',True),('producer_modules_imported',True),('C_source_atol',1e-10),('continuous_rtol',0.01),('rank_tolerance',1e-12)]:
            record=gate_receipt(3,True);record[key]=value
            with self.assertRaises(ValueError):h.source_gate_check(record,3,ind)

    def test_incumbent_gate_cannot_compute_candidate_or_relax_exactness(self):
        ind=gate_ind()
        for key,value in [('incumbent_checked',False),('checked_events',4),('ordered_choices_checked',6),('numerical_tolerance',1e-12),('candidate_selection_computed',True),('computes_returns',True),('producer_modules_imported',True)]:
            record=gate_receipt(3,False);record[key]=value
            with self.assertRaises(ValueError):h.incumbent_gate_check(record,3,ind)

    def test_callbacks_enforce_source_then_single_incumbent(self):
        ind=gate_ind();ind.verify_source_features=lambda *a,**k:gate_receipt(3,True);ind.verify_incumbent=lambda *a,**k:gate_receipt(3,False)
        source,inc,receipts=h.gate_callbacks(ind,{}, {}, {}, [], {},3,True)
        with self.assertRaises(ValueError):inc([],[],[])
        source([])
        with self.assertRaises(ValueError):source([])
        inc([],[],[]);h.pre_gate_check(receipts,3,ind,True)
        with self.assertRaises(ValueError):inc([],[],[])
        source,inc,receipts=h.gate_callbacks(ind,{}, {}, None,None,None,3,False)
        with self.assertRaises(ValueError):source([])
        inc([],[],[]);h.pre_gate_check(receipts,3,ind,False)

    def test_failed_source_callback_cannot_enable_incumbent(self):
        ind=gate_ind();bad=gate_receipt(3,True);bad['passed']=False
        ind.verify_source_features=lambda *a,**k:bad;ind.verify_incumbent=lambda *a,**k:gate_receipt(3,False)
        source,inc,receipts=h.gate_callbacks(ind,{}, {}, {}, [], {},3,True)
        with self.assertRaises(ValueError):source([])
        self.assertEqual(receipts,{})
        with self.assertRaises(ValueError):inc([],[],[])


    def test_manifest_is_transitively_source_bound_not_direct_context_pin(self):
        source,binding=source_fixture();manifest=source['panel_manifest_path']
        self.assertNotIn(manifest,binding['input_sha256'])
        with patch.object(h,'sha',return_value='fixed'),patch.object(h,'read',return_value=binding):
            h.source_review_check(source)
            for value in ({},{str(h.ROOT/manifest):'wrong'}):
                bad=copy.deepcopy(source);bad['input_sha256']=value
                with self.assertRaises(ValueError):h.source_review_check(bad)


if __name__=='__main__':unittest.main()
