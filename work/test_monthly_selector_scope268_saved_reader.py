"""Synthetic saved-evidence checks; no actual source/price/output loading."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import monthly_selector_scope268_saved_reader as r
import monthly_selector_scope268_host as h
import monthly_selector_scope268_admission as admission
import test_monthly_selector_scope268_host as fixtures


def selection_fixture():
    group=dict(training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3,daily_return_sd=.01,sharpe_status="defined",sharpe_reason=None),early=dict(cagr=.4),recent=dict(cagr=.6))
    metrics={a:{p:deepcopy(group) for p in r.PATHS[:2]} for a in r.ARMS}
    orders={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in r.ARMS}
    for a in r.ARMS[1:]:
        for p in r.PATHS[:2]:metrics[a][p]['training']['cagr']=.52
        for i in (1,54,55):orders[a][i]['symbols']=['A','B','D']
    return metrics,orders


def prepared_fixture():
    args=fixtures.fixture(n=5);full,main,cache=fixtures.run(args);cached=[];physical=[]
    for n in range(6):
        got,e,_=fixtures.run(args,n,cached_pre=cache);h.exact_prefix(full,got,n);cached.append(e)
    for n in (0,1,3,4):
        got,e,_=fixtures.run(args,n);h.exact_prefix(full,got,n);physical.append(e)
    alias=deepcopy(main);alias['reuses_main_verified_full']=True;physical.append(alias)
    proof=dict(schema='selector_scope268_independent_prepare_v1',main=main,cached_prefixes=cached,physical_prefixes=physical,all123_order_prefixes=True,source_arrays_unchanged=True,original_reference_unchanged=True)
    proof['execution_accounting']=h.execution_accounting(proof)
    source=dict(main=main['source_audit'],physical=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in physical],cached=[dict(prefix=x['prefix'],audit=x['source_audit']) for x in cached])
    return proof,full[3],source,args[2]


def preparation(values,orders=None):return r.preparation(*values,total=5,width=6,prefixes=(0,1,3,4,5),orders=orders)


def refresh(proof,source):
    proof['execution_accounting']=h.execution_accounting(proof)
    source.clear();source.update(main=deepcopy(proof['main']['source_audit']),physical=[dict(prefix=x['prefix'],audit=deepcopy(x['source_audit'])) for x in proof['physical_prefixes']],cached=[dict(prefix=x['prefix'],audit=deepcopy(x['source_audit'])) for x in proof['cached_prefixes']])


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
    def setUpClass(cls):cls.original=prepared_fixture()
    def values(self):return deepcopy(self.original)

    def test_host_saved_schema_and_all_prefix_work(self):
        v=self.values();ledger=preparation(v)
        self.assertEqual(ledger['actual_prepare_entries'],11)
        self.assertEqual(ledger['producer']['original_pre_score_calls'],5)
        self.assertEqual(ledger['producer']['original_selector_calls'],11)
        self.assertEqual(ledger['independent']['selector_calls'],33)
        self.assertEqual(ledger['new_model_fits'],0)
        h.preparation_evidence(v[0],v[1],v[2],total=5,width=6,prefixes=(0,1,3,4,5))

    def test_real_host_order_indices_cross_bind_in_order(self):
        orders=fixtures.run(fixtures.fixture(n=5))[0][0]
        values=self.values();preparation(values,orders)
        # Same set and valid domain, but reversed order must now fail binding.
        bad=self.values();event=bad[1]['arms'][r.ARMS[1]]['events'][1]
        event['selected_indices']=list(reversed(event['selected_indices']))
        with self.assertRaisesRegex(ValueError,'ordered holdings'):preparation(bad,orders)
        bad_orders=deepcopy(orders);bad_orders[r.ARMS[2]][1]['symbols'][0]='WRONG'
        with self.assertRaisesRegex(ValueError,'ordered holdings'):preparation(self.values(),bad_orders)

    def test_independent_event_counts_fail_after_balanced_totals_and_alias(self):
        orders=fixtures.run(fixtures.fixture(n=5))[0][0]
        for key in ('partial_events','two_eligible_events','cash_events'):
            v=self.values();work=v[0]['main']['independent'][r.ARMS[1]]['work']
            if key=='partial_events':work[key]=0
            else:
                work[key]=1;work['selection_rank_calls']-=1
            # Keep every aggregate/alias self-consistent: rejection must come
            # from the independent versus actual event-derived cross-check.
            alias=deepcopy(v[0]['main']);alias['reuses_main_verified_full']=True
            v[0]['physical_prefixes'][-1]=alias;refresh(v[0],v[2])
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Independent exact event counter'):
                preparation(v,orders)

    def test_precise_extra_event_counters_and_changed_domain(self):
        for key in ('nonzero_delta_cells','changed_score_cells','boundary_winner_reductions'):
            v=self.values();v[1]['arms'][r.ARMS[1]]['counts'][key]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Recorded total'):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['events'][1]['changed_score_indices'].append(5)
        with self.assertRaisesRegex(ValueError,'Changed cells'):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['counts']['unregistered_counter']=0
        with self.assertRaisesRegex(ValueError,'Exact producer count fields'):preparation(v)

    def test_every_prefix_arm_counter_rejects_balanced_forgery(self):
        cases=[('main',r.ARMS[0])]+[(kind,a) for kind in ('cached','physical') for a in r.ARMS]
        for kind,arm in cases:
            v=self.values()
            entry=v[0]['main'] if kind=='main' else v[0]['cached_prefixes'][1] if kind=='cached' else v[0]['physical_prefixes'][1]
            entry['independent'][arm]['work']['partial_events']=0
            if kind=='main':
                alias=deepcopy(v[0]['main']);alias['reuses_main_verified_full']=True
                v[0]['physical_prefixes'][-1]=alias
            refresh(v[0],v[2])
            with self.subTest(kind=kind,arm=arm),self.assertRaisesRegex(ValueError,'Every prefix/arm actual primitive count'):
                preparation(v)

    def test_prefix_work_empty_partial_and_distinct_domains(self):
        v=self.values();events=v[1]['arms'][r.ARMS[1]]['events']
        for arm in r.ARMS:
            empty=r.prefix_work(events,arm,0)
            self.assertEqual(empty,dict.fromkeys(r.WORK,0)|{'selector_calls':1})
            self.assertEqual(r.prefix_work(events,arm,1),v[0]['cached_prefixes'][1]['independent'][arm]['work'])
            for n in range(6):self.assertEqual(r.prefix_work(events,arm,n),v[0]['cached_prefixes'][n]['independent'][arm]['work'])
        consumed=[r.prefix_work(events,a,5)['support_evaluations'] for a in r.ARMS]
        self.assertLess(consumed[0],consumed[1]);self.assertLess(consumed[1],consumed[2])

    def test_typed_ledger_digest_matches_host_exactly(self):
        ledger=preparation(self.values());self.assertEqual(r.structured_digest(ledger),h.structured_digest(ledger))
        for value in ({'a':[1,True,None,-0.,'x'],'b':(2,.5)},[0.,-0.],{'nested':{'counter':0}}):
            self.assertEqual(r.structured_digest(value),h.structured_digest(value))
        self.assertNotEqual(r.structured_digest(0),r.structured_digest(False))
        self.assertNotEqual(r.structured_digest(0.),r.structured_digest(-0.))
        self.assertNotEqual(r.structured_digest([1]),r.structured_digest((1,)))
        with self.assertRaises(ValueError):r.structured_digest(np.zeros(1))

    def test_prefix_inventory_and_alias_must_be_exact(self):
        v=self.values();v[0]['cached_prefixes'].pop(2)
        with self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['physical_prefixes'][-1]['reuses_main_verified_full']=False
        with self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['physical_prefixes'][-1]['source_audit']['source_integrity_work']['cache_values_compared']+=1
        with self.assertRaises(ValueError):preparation(v)

    def test_source_counters_reject_even_after_total_recomputed(self):
        for field in ('checks','reference_values_compared','reference_order_events_compared','metadata_events_compared'):
            v=self.values();v[0]['cached_prefixes'][3]['source_audit']['source_integrity_work'][field]+=1;refresh(v[0],v[2])
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['cached_prefixes'][3]['source_audit']['source_integrity_work']['cache_values_compared']=0;refresh(v[0],v[2])
        with self.assertRaises(ValueError):preparation(v)

    def test_false_new_reads_or_source_authentication_claims_rejected(self):
        for key in ('source_files_authenticated','new_features_computed','new_labels_computed','new_price_windows_read'):
            v=self.values();v[0]['cached_prefixes'][3]['source_audit'][key]=True;refresh(v[0],v[2])
            with self.subTest(key=key),self.assertRaises(ValueError):preparation(v)
        for key in ('new_model_fits','new_label_values_computed','new_price_values_read'):
            v=self.values();v[0]['cached_prefixes'][3]['producer_counts'][key]=1;refresh(v[0],v[2])
            with self.subTest(key=key),self.assertRaises(ValueError):preparation(v)

    def test_independent_flags_modes_and_coverage_claims_rejected(self):
        for field,value in [('mode',r.ARMS[2]),('actual_coverage_claim',True),('source_provenance_verified',True),('numerical_tolerance',1e-12),('passed',False)]:
            v=self.values();v[0]['cached_prefixes'][3]['independent'][r.ARMS[1]][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['cached_prefixes'][3]['independent'][r.ARMS[1]]['flags'][r.FLAGS[0]]=False
        with self.assertRaises(ValueError):preparation(v)

    def test_independent_event_counter_mutant_rejected(self):
        v=self.values();proof=v[0]['cached_prefixes'][3]['independent'][r.ARMS[1]];proof['work']['delta_evaluations']+=1;refresh(v[0],v[2])
        with self.assertRaises(ValueError):preparation(v)
        v=self.values();v[0]['cached_prefixes'][3]['producer_counts']['original_selector_calls']=True
        with self.assertRaises(ValueError):preparation(v)

    def test_top5_cannot_admit_rank6_from_outside_domain(self):
        v=self.values();event=v[1]['arms'][r.ARMS[1]]['events'][1];event['selected_indices'][-1]=5
        with self.assertRaisesRegex(ValueError,'domain'):preparation(v)

    def test_global_and_narrow_adjustment_scope_cannot_be_swapped(self):
        v=self.values();event=v[1]['arms'][r.ARMS[2]]['events'][1];event['adjustment_indices'].pop()
        with self.assertRaises(ValueError):preparation(v)
        v=self.values();event=v[1]['arms'][r.ARMS[1]]['events'][1];event['selection_indices'].append(5)
        with self.assertRaises(ValueError):preparation(v)

    def test_partial_identity_date_and_top2_evidence(self):
        for field,value in [('initial_partial',False),('effective_mode',r.ARMS[1]),('date','2016-08-18')]:
            v=self.values();v[1]['arms'][r.ARMS[1]]['events'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):preparation(v)
        v=self.values();v[1]['arms'][r.ARMS[1]]['events'][1]['top2_exited_indices']=[]
        with self.assertRaises(ValueError):preparation(v)

    def test_small_universe_saved_events_and_cash(self):
        args=fixtures.fixture(n=5);args[1]['eligible'][1]=False;args[1]['eligible'][2,2:]=False;args=fixtures.rebind(args)
        result,proof,_=fixtures.run(args)
        for a in r.ARMS[1:]:r.decision(result[3]['arms'][a],proof['independent'][a],args[2],a)
        self.assertEqual(result[3]['arms'][r.ARMS[1]]['events'][1]['selected_indices'],[])
        self.assertEqual(len(result[3]['arms'][r.ARMS[1]]['events'][2]['selected_indices']),2)

    def test_decision_counter_and_source_artifact_drift(self):
        v=self.values();v[1]['arms'][r.ARMS[1]]['counts']['top2_exit_events']+=1
        with self.assertRaises(ValueError):preparation(v)
        v=self.values();v[2]['cached'][3]['audit']['original_pre_scores_exact']=False
        with self.assertRaises(ValueError):preparation(v)


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


def source_fixture():
    # Metadata dictionaries only: no fake market panel, no file writes or NPZ.
    digest='a'*64;meta=dict(dates=['2016-08-17','2016-08-31'])
    raw={'work/synthetic_source'+str(i)+'.json':digest for i in range(64)}
    names=('AUDIT.json','BASE_AUDIT.json','BASE_DATA.npz','DATA.npz','LABEL_AUDIT.json','META.json','MODEL_AUDIT.json','MONTHLY_META.json','ORDERS.json','TRAIN_INPUTS.npz','WEEKLY_INPUTS.npz','WEEKLY_META.json')
    artifacts=dict.fromkeys(names,digest);closure=dict(chain_sha256={str(r.old.REBUILD/'PARITY.json'):digest},artifacts_sha256=artifacts,source_input_count=1)
    plan=dict(raw_source_hashes=raw,original169_source_closure=closure,calendar_sha256=digest)
    source=dict(raw_source_hashes=deepcopy(raw),original_training_hash=digest,original_identity_hash=digest,saved169_parity_hash=digest,original_arrays_unchanged=True,original_orders_exact=True,original_scores_exact=True,original_pre_scores_exact=True,original_eligibility_exact=True,original_gate_before_candidates=True,whole_original_cache_decoded=True,training_arrays_decoded=False,new_features_computed=False,new_labels_computed=False,new_price_windows_read=False,strict_pit=False,new_model_fits=0,original169_source_closure=deepcopy(closure),prior_prepared_sha256=digest,original_array_fingerprints={str(i):dict(dtype='<f8',shape=[122,6],sha256=digest) for i in range(27)},calendar_sha256=digest)
    records={r.old.LOCAL/'PLAN.json':dict(raw_source_hashes=raw),r.old.REBUILD/'PARITY.json':dict(passed=True,returns_computed=False,source_plan_sha256=digest,artifacts_sha256=artifacts),r.old.REBUILD.parent/'PLAN.json':dict(input_sha256={'work/synthetic_input.json':digest}),r.old.REBUILD/'MONTHLY_META.json':dict(monthly=meta),r.old.CALENDAR:dict(sessions=['2016-08-17','2016-08-31','2016-09-01'])}
    ev=SimpleNamespace(read=lambda p,*args:deepcopy(records[Path(p)]),registered=lambda plan,p:digest,pin=lambda p,*args:digest)
    return ev,plan,source,meta


if __name__=='__main__':unittest.main()
