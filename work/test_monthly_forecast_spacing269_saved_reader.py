"""Synthetic saved-evidence tests only; no actual269 source/output reads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import monthly_forecast_spacing269_saved_reader as r
import monthly_forecast_spacing269_admission as admission
import monthly_forecast_spacing269_host as h
import test_monthly_forecast_spacing269_host as fixtures


def prepared_fixture():
    full,proof,source,meta=fixtures.prepared_fixture()
    return proof,full[3],source,meta,full[0]


def preparation(v):return r.preparation(*v[:4],total=5,width=6,prefixes=(0,1,3,4,5),orders=v[4])


def refresh(v):
    proof=v[0];proof['execution_accounting']=h.execution_accounting(proof)
    v[2].clear();v[2].update(main=deepcopy(proof['main']['source_audit']),physical=[dict(prefix=e['prefix'],audit=deepcopy(e['source_audit'])) for e in proof['physical_prefixes']],cached=[dict(prefix=e['prefix'],audit=deepcopy(e['source_audit'])) for e in proof['cached_prefixes']])


def alias(v):
    v[0]['physical_prefixes'][-1]=deepcopy(v[0]['main']);v[0]['physical_prefixes'][-1]['reuses_main_verified_full']=True


def selection_fixture():
    group=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3,daily_return_sd=.01,sharpe_status="defined",sharpe_reason=None),early=dict(cagr=.4),recent=dict(cagr=.6))
    metrics={a:{p:deepcopy(group) for p in r.PATHS[:2]} for a in r.ARMS}
    orders={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in r.ARMS}
    for a in r.ARMS[1:]:
        for p in r.PATHS[:2]:metrics[a][p]['training']['cagr']=.52
        for i in (1,54,55):orders[a][i]['symbols']=['A','B','D']
    return metrics,orders

class SelectionTests(unittest.TestCase):
    def test_two_candidates_and_best_or_tied_choice(self):
        m,o=selection_fixture();self.assertEqual(r.selection(m,o)['primary'],r.ARMS[1])
        m[r.ARMS[2]]['same_close_zero']['training']['cagr']=.53
        self.assertEqual(r.selection(m,o)['primary'],r.ARMS[2])
        m[r.ARMS[2]]['same_close_zero']['training']['cagr']=.52+5e-13
        self.assertEqual(r.selection(m,o)['primary'],r.ARMS[1])
        self.assertEqual(r.selection(m,o),admission.select_first88(m,o))

    def test_candidate_failures_independent_and_complete_fallback(self):
        m,o=selection_fixture();m[r.ARMS[1]]['same_close_zero']['training']['cagr']=.504
        self.assertEqual(r.selection(m,o)['primary'],r.ARMS[2])
        m[r.ARMS[2]]['same_close_zero']['training']['cagr']=.504
        got=r.selection(m,o);self.assertEqual(got['primary'],r.ARMS[0]);self.assertFalse(got['admitted'])

    def test_every_path_and_each_risk_era_gate(self):
        for arm in r.ARMS[1:]:
            for path in r.PATHS[:2]:
                for group,key,value in [('training','cagr',.49),('training','sharpe',1.64),('training','max_drawdown',-.34),('early','cagr',.37),('recent','cagr',.57)]:
                    m,o=selection_fixture();m[arm][path][group][key]=value
                    got=r.selection(m,o)
                    with self.subTest(arm=arm,path=path,group=group,key=key):
                        self.assertNotIn(arm,got['qualified']);self.assertEqual(got,admission.select_first88(m,o))

    def test_cagr_boundary_tolerance_does_not_relax_risk_or_era(self):
        m,o=selection_fixture();a=r.ARMS[1]
        m[a]['same_close_zero']['training']['cagr']=.505-5e-13
        m[a]['next_open_zero']['training']['cagr']=.5-5e-13
        self.assertIn(a,r.selection(m,o)['qualified'])
        for path in r.PATHS[:2]:
            for group,key,threshold in [('training','sharpe',1.65),('training','max_drawdown',-.33),('early','cagr',.375),('recent','cagr',.575)]:
                mm=deepcopy(m);mm[a][path][group][key]=threshold-5e-13
                self.assertNotIn(a,r.selection(mm,o)['qualified'])

    def test_support_requires_three_sets_and_both_eras(self):
        for changed in ((1,54),(1,2,3),(53,54,55)):
            m,o=selection_fixture();o[r.ARMS[1]]=deepcopy(o[r.ARMS[0]])
            for i in changed:o[r.ARMS[1]][i]['symbols']=['A','B','D']
            self.assertFalse(r.selection(m,o)['guards'][r.ARMS[1]]['support'])
        m,o=selection_fixture()
        for i in (1,54,55):o[r.ARMS[1]][i]['symbols']=['C','B','A']
        self.assertFalse(r.selection(m,o)['guards'][r.ARMS[1]]['support'])

    def test_no_later_order_reads_and_no_cost_or_later_metric_input(self):
        class First88(list):
            def __getitem__(self,key):
                if type(key) is not int or key<0 or key>=88:raise AssertionError('Future/slice access')
                return super().__getitem__(key)
        m,o=selection_fixture();wanted=r.selection(m,o)
        self.assertEqual(r.selection(m,{a:First88(v) for a,v in o.items()}),wanted)
        for key in ('next_open_cost','later34'):
            mm=deepcopy(m)
            if key=='next_open_cost':mm[r.ARMS[1]][key]={}
            else:mm[r.ARMS[1]]['same_close_zero'][key]={}
            with self.assertRaises(ValueError):r.selection(mm,o)

    def test_undefined_candidate_disqualifies_only_it_incumbent_is_error(self):
        m,o=selection_fixture();undef=dict(sharpe=None,daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
        m[r.ARMS[1]]['same_close_zero']['training'].update(undef)
        self.assertEqual(r.selection(m,o)['primary'],r.ARMS[2])
        m[r.ARMS[0]]['same_close_zero']['training'].update(undef)
        with self.assertRaises(ValueError):r.selection(m,o)

    def test_invalid_metrics_and_undefined_evidence_rejected(self):
        for value in (None,float('nan'),float('inf'),True):
            m,o=selection_fixture();m[r.ARMS[1]]['same_close_zero']['training']['cagr']=value
            with self.assertRaises(ValueError):r.selection(m,o)
        for sd in (None,1e-30,-1.,float('nan')):
            m,o=selection_fixture();m[r.ARMS[1]]['same_close_zero']['training'].update(sharpe=None,daily_return_sd=sd,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
            with self.assertRaises(ValueError):r.selection(m,o)

    def test_malformed_holdings_and_short_history(self):
        for names in (['A'],['A','A'],['A','B','C','D'],['A',''],'ABC'):
            m,o=selection_fixture();o[r.ARMS[1]][1]['symbols']=names
            with self.assertRaises(ValueError):r.selection(m,o)
        m,o=selection_fixture();o[r.ARMS[1]]=o[r.ARMS[1]][:87]
        with self.assertRaises(ValueError):r.selection(m,o)

    def test_input_identity(self):
        m,o=selection_fixture();before=deepcopy((m,o));r.selection(m,o);self.assertEqual((m,o),before)

class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.saved=prepared_fixture()
    def values(self):return deepcopy(self.saved)

    def test_actual_pure_host_saved_schema_and_all_prefixes(self):
        v=self.values();work=preparation(v)
        self.assertEqual(work['actual_prepare_entries'],11)
        self.assertEqual(work['producer']['original_pre_score_calls'],5)
        self.assertEqual(work['producer']['original_selector_calls'],33)
        self.assertEqual(work['producer']['experimental_transform_calls'],22)
        self.assertEqual(work['independent_transform']['transform_calls'],22)
        self.assertEqual(work['independent_selector']['selector_calls'],33)
        self.assertEqual(work['new_model_fits'],0)
        h.preparation_evidence(v[0],v[1],v[2],total=5,width=6,prefixes=(0,1,3,4,5))

    def test_each_selector_prefix_and_arm_resists_balanced_counter_forgery(self):
        for where in ('main','cached','physical'):
            for arm in r.ARMS:
                v=self.values();e=v[0]['main'] if where=='main' else v[0]['cached_prefixes'][1] if where=='cached' else v[0]['physical_prefixes'][1]
                e['independent'][arm]['selector']['work']['partial_events']=0
                if where=='main':alias(v)
                refresh(v)
                with self.subTest(where=where,arm=arm),self.assertRaisesRegex(ValueError,'Every selector prefix'):preparation(v)

    def test_each_transform_prefix_and_arm_resists_balanced_counter_forgery(self):
        for where in ('main','cached','physical'):
            for arm in r.ARMS[1:]:
                for field in ('use_cells','active_rows','consumed_raw_cells','source_rank_checks','affine_evaluations','delta_evaluations'):
                    v=self.values();e=v[0]['main'] if where=='main' else v[0]['cached_prefixes'][3] if where=='cached' else v[0]['physical_prefixes'][2]
                    e['independent'][arm]['transform']['work'][field]+=1
                    if where=='main':alias(v)
                    refresh(v)
                    with self.subTest(where=where,arm=arm,field=field),self.assertRaisesRegex(ValueError,'Every transform prefix'):preparation(v)

    def test_no_future_event_body_in_empty_and_short_selector_prefix(self):
        v=self.values();events=v[1]['arms'][r.ARMS[2]]['selector_events']
        class PrefixOnly(list):
            def __getitem__(self,i):
                if type(i) is not int or not 0<=i<1:raise AssertionError('future event read')
                return super().__getitem__(i)
        x=PrefixOnly(events)
        self.assertEqual(r.selector_work(x,v[3],0,6),dict.fromkeys(r.WORK,0)|{'selector_calls':1})
        self.assertEqual(r.selector_work(x,v[3],1,6),v[0]['cached_prefixes'][1]['independent'][r.ARMS[2]]['selector']['work'])

    def test_exact_transform_raw_rank_affine_endpoint_and_weight_checks(self):
        for role,field in [('pooled','old_rank'),('pooled','new_rank'),('pooled','delta'),('local','delta')]:
            v=self.values();event=v[1]['arms'][r.ARMS[2]]['transform']['events'][1];event[role][field][2]+=.001
            with self.subTest(role=role,field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['transform']['events'][1]['delta'][2]+=.001
        with self.assertRaisesRegex(ValueError,'two-product'):preparation(v)

    def test_map_scalar_type_signedzero_and_nonfinite_guards(self):
        # Use real audit for exact rank/arithmetic, including signed-zero delta.
        actual=self.values()[1]['arms'][r.ARMS[1]]['transform']['events'][1]['pooled']
        for field,value in [('minimum',False),('maximum',float('inf')),('distinct_levels',True)]:
            bad=deepcopy(actual);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.saved_map(bad,6,.0375)
        bad=deepcopy(actual);bad['delta'][0]=-0.
        with self.assertRaisesRegex(ValueError,'float bits'):r.saved_map(bad,6,.0375)

    def test_every_transform_counter_derived_from_events(self):
        for field in r.TRANSFORM_COUNTS:
            v=self.values();v[1]['arms'][r.ARMS[1]]['transform']['counts'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)

    def test_candidate_arm_scope_and_no_new_budget(self):
        for field in r.TRANSFORM_TRUE:
            v=self.values();v[1]['arms'][r.ARMS[2]]['transform'][field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['transform']['events'][1]['local']=deepcopy(v[1]['arms'][r.ARMS[2]]['transform']['events'][1]['local'])
        with self.assertRaisesRegex(ValueError,'Pooled-only'):preparation(v)

    def test_initial_and_fallback_saved_maps_cannot_consume_data(self):
        for arm in r.ARMS[1:]:
            v=self.values();e=v[1]['arms'][arm]['transform']['events'][0];e['branch']='active'
            with self.assertRaisesRegex(ValueError,'precedence'):preparation(v)
            v=self.values();v[1]['arms'][arm]['transform']['events'][0]['pooled']={}
            with self.assertRaisesRegex(ValueError,'Inactive'):preparation(v)

    def test_changed_pre_evidence_cannot_claim_zero_or_outside_u(self):
        for index in (0,6):
            v=self.values();e=v[1]['arms'][r.ARMS[1]]['transform']['events'][1];e['changed_pre_indices']=[index]
            with self.subTest(index=index),self.assertRaises(ValueError):preparation(v)

    def test_selector_orders_map_in_exact_order_and_top2_locked(self):
        v=self.values();e=v[1]['arms'][r.ARMS[1]]['selector_events'][1];e['selected_indices']=e['selected_indices'][::-1]
        with self.assertRaisesRegex(ValueError,'ordered holdings'):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['selector_events'][1]['selected_indices']=[3,4,5]
        with self.assertRaisesRegex(ValueError,'locked top2'):preparation(v)

    def test_selector_boundary_order_and_small_domain(self):
        v=self.values();e=v[1]['arms'][r.ARMS[1]]['selector_events'][1];e['boundary_indices'].append(e['original_rank'][0])
        with self.assertRaises(ValueError):preparation(v)
        args=fixtures.fixture();args[1]['eligible'][1]=False;args[1]['eligible'][2,2:]=False;args=fixtures.rebind(args)
        full,proof,_=fixtures.run(args)
        for arm in r.ARMS:
            got=r.selector_work(full[3]['arms'][arm]['selector_events'],args[2],5,6,full[0][arm])
            self.assertEqual(got,proof['independent'][arm]['selector']['work']);self.assertEqual(got['cash_events'],1);self.assertEqual(got['two_eligible_events'],1)

    def test_zero_prefix_transform_and_source_counts_not_omitted(self):
        v=self.values();preparation(v)
        for arm in r.ARMS[1:]:
            got=r.transform_evidence(v[1]['arms'][arm]['transform'],arm,v[3],5,6)
            self.assertEqual(got[0],dict.fromkeys(r.TRANSFORM_WORK,0)|{'transform_calls':1})
            self.assertEqual(got[1]['consumed_raw_cells'],0)
        self.assertEqual(v[0]['cached_prefixes'][0]['source_audit']['source_rank_work'],dict(pooled_rank_calls=1,joint_rank_calls=1))

    def test_physical_full_alias_and_all123_inventory(self):
        for mutation in ('remove','alias','counter'):
            v=self.values()
            if mutation=='remove':v[0]['cached_prefixes'].pop(2)
            elif mutation=='alias':v[0]['physical_prefixes'][-1]['reuses_main_verified_full']=False
            else:v[0]['physical_prefixes'][-1]['producer_counts']['source_joint_rank_calls']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):preparation(v)

    def test_source_rank_proof_and_new_read_claims_fail(self):
        for field,value in [('source_rank_parity',False),('source_files_authenticated',True),('new_labels_computed',True)]:
            v=self.values();v[0]['cached_prefixes'][3]['source_audit'][field]=value;refresh(v)
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['cached_prefixes'][3]['source_audit']['source_rank_work']['pooled_rank_calls']=0;refresh(v)
        with self.assertRaisesRegex(ValueError,'source rank'):preparation(v)

    def test_independent_flags_mode_source_claims_and_missing_proof_fail(self):
        for part,field,value in [('transform','mode',r.ARMS[0]),('selector','mode',r.ARMS[1]),('transform','actual_coverage_claim',True),('selector','passed',False),('transform','numerical_tolerance',1e-12)]:
            v=self.values();v[0]['cached_prefixes'][3]['independent'][r.ARMS[1]][part][field]=value
            with self.subTest(part=part,field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['cached_prefixes'][3]['independent'][r.ARMS[1]]['transform']={'passed':True}
        with self.assertRaises(ValueError):preparation(v)

    def test_mutually_bound_source_artifact_and_invocation_counts(self):
        v=self.values();v[2]['cached'][3]['audit']['source_rank_parity']=False
        with self.assertRaises(ValueError):preparation(v)
        for field in r.PRODUCER:
            v=self.values();v[0]['cached_prefixes'][3]['producer_counts'][field]+=1;refresh(v)
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)

    def test_typed_accounting_digest_matches_host_and_preserves_zero(self):
        work=preparation(self.values());self.assertEqual(r.structured_digest(work),h.structured_digest(work))
        for value in ({'a':[1,True,None,-0.,'x'],'b':(2,.5)},[0.,-0.],{'nested':{'counter':0}}):self.assertEqual(r.structured_digest(value),h.structured_digest(value))
        self.assertNotEqual(r.structured_digest(0),r.structured_digest(False));self.assertNotEqual(r.structured_digest(0.),r.structured_digest(-0.))
        with self.assertRaises(ValueError):r.structured_digest(np.zeros(1))

    def test_valid_nonweak_notready_and_tied_rows(self):
        args=fixtures.fixture();args[1]['state_weak_narrow'][1]=False;args[1]['raw_local_constant'][2,0]=np.nan
        for key in ('raw_source_log','raw_local_linear'):args[1][key][3]=[1.,1.,1.,2.,2.,2.]
        args=fixtures.rebind(args);full,e,_=fixtures.run(args)
        for arm in r.ARMS[1:]:
            prefix=r.transform_evidence(full[3]['arms'][arm]['transform'],arm,args[2],5,6)
            self.assertEqual(prefix[-1],e['independent'][arm]['transform']['work'])
            self.assertEqual(full[3]['arms'][arm]['transform']['events'][2]['branch'],'not_ready')
            self.assertEqual(full[3]['arms'][arm]['transform']['events'][3]['pooled']['branch'],'old_rank_copy')



class EvidenceTests(unittest.TestCase):
    def terminal(self):
        hashes=dict(prepare='a'*64,select='b'*64,diagnose='c'*64)
        phases={}
        for phase,path in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:phases[phase]=dict(session_id=123,exit_code=0,chunk_id='abc',artifact_path=str(r.OUT/path),artifact_sha256=hashes[phase])
        return dict(plan_sha256='d'*64,phases=phases),hashes

    def test_terminal_success_and_failure_fields(self):
        record,hashes=self.terminal();r.terminals(record,'d'*64,hashes)
        for key,value in [('exit_code',1),('exit_code',False),('session_id',-1),('chunk_id',' '),('artifact_sha256','e'*64),('artifact_path',str(r.OUT/'wrong.json'))]:
            bad=deepcopy(record);bad['phases']['prepare'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.terminals(bad,'d'*64,hashes)
        bad=deepcopy(record);bad['phases'].pop('select')
        with self.assertRaises(ValueError):r.terminals(bad,'d'*64,hashes)

    def test_exact36_account_paths_only(self):
        seen=[];ev=SimpleNamespace(read=lambda p:seen.append(p) or str(p));adapter=r.Accounts(ev)
        self.assertEqual(len(adapter.paths),36)
        for before,after in adapter.paths.items():self.assertEqual(adapter.read(before),str(after))
        self.assertEqual(set(seen),{r.OUT/'results'/(a+'_'+p+'_'+k+'.json') for a in r.ARMS for p in r.PATHS for k in r.KINDS})
        with self.assertRaises(ValueError):adapter.read(r.old.OUT/'results/RESULTS.json')
        with self.assertRaises(ValueError):adapter.read(r.OUT/'results'/(r.ARMS[0]+'_'+r.PATHS[0]+'_DAILY.json'))

    def test_accounting_role_adapter_does_not_change_old_globals(self):
        values={a:object() for a in r.ARMS};old_arms=r.old.ARMS;old_out=r.old.OUT
        def check(ev,result,ind,training):
            self.assertEqual(set(result['arms']),set(old_arms));self.assertEqual(set(ind['paths']),set(old_arms));self.assertEqual(set(training),set(old_arms))
            return {'passed':True},{(a,p):[] for a in old_arms for p in r.PATHS}
        with patch.object(r.old,'accounting',side_effect=check):
            report,exports=r.accounting(SimpleNamespace(),dict(arms=values),dict(paths=values),values)
        self.assertTrue(report['passed']);self.assertEqual(set(exports),{(a,p) for a in r.ARMS for p in r.PATHS})
        self.assertEqual((r.old.ARMS,r.old.OUT),(old_arms,old_out))

    def test_source_scope_claims_with_metadata_only_fixture(self):
        ev,plan,source,meta=source_fixture()
        with patch.object(r,'digest',return_value='a'*64),patch.object(r.old,'full_original_chain'):
            axis,rowwidth=r.source_proof(ev,plan,source,meta)
            self.assertEqual(rowwidth,162);self.assertEqual(axis,['2016-08-17','2016-08-31','2016-09-01'])
            for key in ('training_arrays_decoded','new_features_computed','new_labels_computed','new_price_windows_read','strict_pit'):
                bad=deepcopy(source);bad[key]=True
                with self.subTest(key=key),self.assertRaises(ValueError):r.source_proof(ev,plan,bad,meta)

    def test_source_raw_lineage_calendar_and_cache_pins_reject_mutation(self):
        ev,plan,source,meta=source_fixture()
        with patch.object(r,'digest',return_value='a'*64),patch.object(r.old,'full_original_chain'):
            for key in ('original_training_hash','calendar_sha256','prior_prepared_sha256'):
                bad=deepcopy(source);bad[key]='b'*64
                with self.subTest(key=key),self.assertRaises(ValueError):r.source_proof(ev,plan,bad,meta)
            bad=deepcopy(source);bad['raw_source_hashes']['work/synthetic_source0.json']='b'*64
            with self.assertRaises(ValueError):r.source_proof(ev,plan,bad,meta)

class FormalBindingTests(unittest.TestCase):
    def fixture(self):
        digest='a'*64;context,contract=fixtures.contract_fixture()
        runtime=dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')
        expected={str(r.ROOT/name):digest for name in r.FORMAL}
        tests=dict(passed=True,synthetic_only=True,tests=99,code_sha256=deepcopy(expected),runtime=runtime)
        review=dict(passed=True,unresolved_blockers=[],code_sha256=deepcopy(expected),actual_returns_computed=False,context_sha256=digest,implementation_contract_sha256=digest,input_sha256={'work/synthetic_review_input.json':digest})
        plan=dict(context=context,implementation_contract=contract,context_sha256=digest,implementation_contract_sha256=digest,runtime=runtime)
        records={r.CONTEXT:context,r.CONTRACT:contract,r.TESTS:tests,r.REVIEW:review}
        def pin(path,*args):
            path=Path(path);relative=str(path.relative_to(r.ROOT)) if path.is_absolute() else str(path)
            return r.FIXED.get(relative,digest)
        ev=SimpleNamespace(read=lambda path,*args:deepcopy(records[Path(path)]),registered=lambda plan,path:pin(path),pin=pin)
        return ev,plan,records

    def test_complete_formal8_and_supplemental2_and_fixed_helper_binding(self):
        ev,plan,records=self.fixture()
        with patch.object(r,'digest',return_value='a'*64):
            context,tests=r.formal(ev,plan)
        self.assertEqual(context['arms'],list(r.ARMS));self.assertEqual(tests['tests'],99)
        self.assertEqual(len(r.FORMAL),8);self.assertEqual(len(r.SUPPLEMENTAL),2)
        self.assertEqual(r.FORMAL,tuple('work/'+x for x in h.code_names));self.assertEqual(r.SUPPLEMENTAL,h.SUPPLEMENTAL)

    def test_exact_ordered_inventory_and_each_paper_pin_fail_closed(self):
        for what in ('formal_order','reader_count','context_pin','contract_pin','frozen','authorized','actual_started'):
            ev,plan,records=self.fixture()
            if what=='formal_order':records[r.CONTRACT]['code_files'].reverse()
            elif what=='reader_count':records[r.CONTRACT]['supplemental_input_files'].pop()
            elif what=='context_pin':records[r.CONTEXT]['input_sha256'][r.FORMAL[0]]='b'*64
            elif what=='contract_pin':records[r.CONTRACT]['input_sha256'][r.SUPPLEMENTAL[1]]='b'*64
            elif what=='frozen':records[r.CONTEXT]['frozen']=False
            elif what=='authorized':records[r.CONTRACT]['registration_authorized']=False
            else:records[r.CONTEXT]['actual_data_started']=True
            with self.subTest(what=what),patch.object(r,'digest',return_value='a'*64),self.assertRaises(ValueError):r.formal(ev,plan)

    def test_formal_tests_runtime_and_review_do_not_accept_passed_only(self):
        for what in ('tests_hash','runtime','test_bool','test_zero','synthetic','blocker','review_hash','review_paper','actual_returns'):
            ev,plan,records=self.fixture()
            if what=='tests_hash':records[r.TESTS]['code_sha256'][str(r.ROOT/r.FORMAL[0])]='b'*64
            elif what=='runtime':records[r.TESTS]['runtime']=dict(records[r.TESTS]['runtime'],numpy='9')
            elif what=='test_bool':records[r.TESTS]['tests']=True
            elif what=='test_zero':records[r.TESTS]['tests']=0
            elif what=='synthetic':records[r.TESTS]['synthetic_only']=False
            elif what=='blocker':records[r.REVIEW]['unresolved_blockers']=['x']
            elif what=='review_hash':records[r.REVIEW]['code_sha256']={}
            elif what=='review_paper':records[r.REVIEW]['context_sha256']='b'*64
            else:records[r.REVIEW]['actual_returns_computed']=True
            with self.subTest(what=what),patch.object(r,'digest',return_value='a'*64),self.assertRaises(ValueError):r.formal(ev,plan)

    def test_six_zero_headlines_exact_and_cost_diagnostic_separate(self):
        result=dict(arms={a:{p:dict(metrics=dict(cost_per_side=0.,fees_paid=0.),slices=dict(full122=dict(cost_per_side=0.,fees_paid=0.))) for p in r.PATHS} for a in r.ARMS})
        for a in r.ARMS:result['arms'][a]['next_open_cost']['metrics'].update(cost_per_side=.0005,fees_paid=15.)
        r.zero_headlines(result)
        for a in r.ARMS:
            for p in r.PATHS[:2]:
                for bucket in ('metrics','full122'):
                    for key in ('cost_per_side','fees_paid'):
                        bad=deepcopy(result);item=bad['arms'][a][p]['metrics'] if bucket=='metrics' else bad['arms'][a][p]['slices']['full122'];item[key]=1e-300
                        with self.subTest(a=a,p=p,bucket=bucket,key=key),self.assertRaises(ValueError):r.zero_headlines(bad)
        result['arms'][r.ARMS[0]][r.PATHS[0]]['metrics']['fees_paid']=False
        with self.assertRaises(ValueError):r.zero_headlines(result)



def source_fixture():
    # Metadata dictionaries only: no fake market panel, no file writes or NPZ.
    digest='a'*64;meta=dict(dates=['2016-08-17','2016-08-31'])
    raw={'work/synthetic_source'+str(i)+'.json':digest for i in range(64)}
    names=('AUDIT.json','BASE_AUDIT.json','BASE_DATA.npz','DATA.npz','LABEL_AUDIT.json','META.json','MODEL_AUDIT.json','MONTHLY_META.json','ORDERS.json','TRAIN_INPUTS.npz','WEEKLY_INPUTS.npz','WEEKLY_META.json')
    artifacts=dict.fromkeys(names,digest);closure=dict(chain_sha256={str(r.old.REBUILD/'PARITY.json'):digest},artifacts_sha256=artifacts,source_input_count=1)
    plan=dict(raw_source_hashes=raw,original169_source_closure=closure,calendar_sha256=digest)
    source=dict(raw_source_hashes=deepcopy(raw),original_training_hash=digest,original_identity_hash=digest,saved169_parity_hash=digest,original_arrays_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_eligibility_exact=True,original_forecast_ranks_exact=True,paired_readiness_exact=True,original_gate_before_candidates=True,whole_original_cache_decoded=True,training_arrays_decoded=False,new_features_computed=False,new_labels_computed=False,new_price_windows_read=False,strict_pit=False,new_model_fits=0,original169_source_closure=deepcopy(closure),prior_prepared_sha256=digest,original_array_fingerprints={str(i):dict(dtype='<f8',shape=[122,6],sha256=digest) for i in range(27)},calendar_sha256=digest)
    records={r.old.LOCAL/'PLAN.json':dict(raw_source_hashes=raw),r.old.REBUILD/'PARITY.json':dict(passed=True,returns_computed=False,source_plan_sha256=digest,artifacts_sha256=artifacts),r.old.REBUILD.parent/'PLAN.json':dict(input_sha256={'work/synthetic_input.json':digest}),r.old.REBUILD/'MONTHLY_META.json':dict(monthly=meta),r.old.CALENDAR:dict(sessions=['2016-08-17','2016-08-31','2016-09-01'])}
    ev=SimpleNamespace(read=lambda p,*args:deepcopy(records[Path(p)]),registered=lambda plan,p:digest,pin=lambda p,*args:digest)
    return ev,plan,source,meta

if __name__=="__main__":unittest.main()
