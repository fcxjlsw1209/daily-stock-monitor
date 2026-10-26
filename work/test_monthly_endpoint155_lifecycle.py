"""Synthetic integrity/selection/crash-safety tests; never open market arrays."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import monthly_endpoint155_lifecycle as c


def state():
    return dict(completed=[dict(directory='old'+str(i),trials=6) for i in range(153)]+[
        dict(directory='monthly_financial_coverage154',trials=57,completion=dict(sha256=c.PREVIOUS_SHA))],
        registered_queue=[],best_training_selected_candidate=dict(cagr=.676,keep='original'),
        unrelated=dict(preserve=[1,2,3]))


def rows():
    return [dict(name=n,training=dict(cagr=.5,sharpe=1.7,max_drawdown=-.3),
                 early=dict(cagr=.4),recent=dict(cagr=.6)) for n in c.NAMES]


class SelectionTests(unittest.TestCase):
    def test_ties_preserve_incumbent(self):
        self.assertFalse(c.selection(rows(),rows())['admitted'])

    def test_requires_strict_advantage_over_both_close_controls(self):
        close,opening=rows(),rows(); close[2]['training']['cagr']=.51; opening[2]['training']['cagr']=.51
        self.assertTrue(c.selection(close,opening)['admitted'])
        close[1]['training']['cagr']=.51
        self.assertFalse(c.selection(close,opening)['admitted'])

    def test_next_open_not_less_than_incumbent_but_strict_return(self):
        close,opening=rows(),rows(); close[2]['training']['cagr']=.51
        opening[1]['training']['cagr']=.49
        self.assertTrue(c.selection(close,opening)['admitted'])
        opening[2]['training']['cagr']=.499
        self.assertFalse(c.selection(close,opening)['admitted'])

    def test_all_risk_subperiod_guards(self):
        for key,field,value in [('training','sharpe',1.64),('training','max_drawdown',-.34),
                                ('early','cagr',.37),('recent','cagr',.57)]:
            with self.subTest(key=key,field=field):
                close,opening=rows(),rows(); close[2]['training']['cagr']=.51; opening[2]['training']['cagr']=.51
                close[2][key][field]=value
                self.assertFalse(c.selection(close,opening)['admitted'])

    def test_nonfinite_and_boolean_metrics_rejected(self):
        for value in (float('nan'),float('inf'),True,None):
            close=rows(); close[2]['training']['cagr']=value
            self.assertFalse(c.selection(close,rows())['admitted'])

    def test_reordered_or_missing_arms_rejected(self):
        for group in (rows()[:2],list(reversed(rows()))):
            with self.assertRaises(ValueError):c.selection(group,rows())

    def test_model_count_derived_from_distinct_fitted_years(self):
        audit=dict(new_pooled_model_fits=4,annual=[dict(year=2020,status='fit'),dict(year=2021,status='fit'),
                                                 dict(year=2019,status='cold_start')])
        self.assertEqual(c.fitted_models(audit),4)
        audit['new_pooled_model_fits']=2
        with self.assertRaises(ValueError):c.fitted_models(audit)

    def test_duplicate_year_boolean_count_unknown_status_rejected(self):
        for audit in (dict(new_pooled_model_fits=4,annual=[dict(year=2020,status='fit')]*2),
                      dict(new_pooled_model_fits=False,annual=[]),
                      dict(new_pooled_model_fits=0,annual=[dict(year=2020,status='failed')])):
            with self.assertRaises(ValueError):c.fitted_models(audit)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic155-',dir='/private/tmp'); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.out=self.root/'outputs'/c.STUDY; self.out.mkdir(parents=True)
        self.put(c.STATE,state())

    def put(self,name,value):
        p=Path(name)
        if not p.is_absolute():p=self.root/p
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else c.old.encoded(value))
        return p

    def ref(self,p):return dict(path=str(p),sha256=c.sha(p))

    def manifest(self):
        suites=[]
        for name,filename in c.SUITES.items():
            source=self.root/'work'/filename
            if not source.exists():source=self.put(source,'synthetic suite '+name)
            log=self.put('logs/'+name+'.log','Ran 2 tests in 0.01s\n\nOK\n')
            suites.append(dict(name=name,count=2,passed=True,source=self.ref(source),log=self.ref(log)))
        return dict(passed=True,synthetic_only=True,test_count=10,suites=suites)


class StateTests(Fixture):
    def test_predecessor_all_history_and_latest_completion(self):
        c.predecessor(state())
        for mutate in (lambda s:s['completed'].pop(),
                       lambda s:s['completed'][0].update(trials=True),
                       lambda s:s['completed'][0].update(trials=7),
                       lambda s:s['completed'][-1]['completion'].update(sha256='0'*64)):
            doc=state(); mutate(doc)
            with self.assertRaises(ValueError):c.predecessor(doc)

    def test_atomic_update_preserves_exact_original_bytes(self):
        p=self.root/c.STATE; original=p.read_bytes(); digest=c.sha(p); new=state(); new['extra']='update'
        c.update_state(p,digest,new)
        self.assertEqual(c.old.decode(p.read_bytes()),new)
        self.assertEqual((p.parent/('PRE155_'+digest+'.json')).read_bytes(),original)

    def test_stale_update_does_not_overwrite(self):
        p=self.root/c.STATE; digest=c.sha(p); self.put(p,{'newer':True}); original=p.read_bytes()
        with self.assertRaises(ValueError):c.update_state(p,digest,state())
        self.assertEqual(p.read_bytes(),original)

    def test_conflicting_backup_rejected(self):
        p=self.root/c.STATE; digest=c.sha(p); self.put(p.parent/('PRE155_'+digest+'.json'),'wrong')
        with self.assertRaises(ValueError):c.update_state(p,digest,state())

    def test_exclusive_write_preserves_existing(self):
        p=self.put('existing.json',{'old':True})
        with self.assertRaises(FileExistsError):c.write(p,{'new':True})
        self.assertEqual(c.old.decode(p.read_bytes()),{'old':True})

    def test_serialization_failure_leaves_no_final_or_temp(self):
        p=self.root/'incomplete.json'
        with self.assertRaises(ValueError):c.write(p,{'bad':float('nan')})
        self.assertFalse(p.exists());self.assertEqual(list(self.root.glob('.endpoint155-*')),[])

    def test_publish_failure_leaves_no_partial_final(self):
        p=self.root/'incomplete.json'
        with patch.object(c.os,'link',side_effect=OSError('synthetic publish failure')):
            with self.assertRaises(OSError):c.write(p,{'complete':True})
        self.assertFalse(p.exists());self.assertEqual(list(self.root.glob('.endpoint155-*')),[])

    def plan(self):
        p=self.put('source.py','synthetic pinned source')
        return dict(round=155,status='REGISTERED',arms=list(c.NAMES),primary_challenger=c.NAMES[2],
                    output_directory=str(self.out),registered_utc='2020-01-01T00:00:00+00:00',
                    registration_precondition=dict(path=c.STATE,sha256=c.sha(self.root/c.STATE)),
                    hashes={str(p):c.sha(p)})

    def test_enqueue_idempotent(self):
        p=self.put(self.out/'PLAN.json',self.plan()); h=c.sha(p)
        self.assertFalse(c.enqueue(self.root,h)['already_enqueued'])
        first=(self.root/c.STATE).read_bytes()
        self.assertTrue(c.enqueue(self.root,h)['already_enqueued'])
        self.assertEqual(first,(self.root/c.STATE).read_bytes())

    def test_enqueue_stale_state_or_conflicting_queue(self):
        p=self.put(self.out/'PLAN.json',self.plan()); h=c.sha(p)
        for doc in (dict(state(),extra=True),dict(state(),registered_queue=[{'unexpected':True}])):
            self.put(c.STATE,doc); original=(self.root/c.STATE).read_bytes()
            with self.assertRaises(ValueError):c.enqueue(self.root,h)
            self.assertEqual(original,(self.root/c.STATE).read_bytes())

    def test_enqueue_changed_pinned_input(self):
        p=self.put(self.out/'PLAN.json',self.plan()); h=c.sha(p); self.put('source.py','changed')
        with self.assertRaises(ValueError):c.enqueue(self.root,h)

    def test_registration_rejects_existing_real_artifacts_before_inputs(self):
        fake=SimpleNamespace(CACHE_FILES=('DATA.npz',))
        for filename in ('DATA.npz','RESULTS.json','PLAN.json'):
            p=self.put(self.out/filename,'already exists')
            with patch.object(c,'runner',return_value=fake),self.assertRaises(ValueError):
                c.build_plan(self.root,self.out/'TESTS.json','a'*64)
            p.unlink()


class SyntheticEvidenceTests(Fixture):
    def test_complete_suites(self):
        self.assertEqual(sum(c.validate_tests(self.manifest(),c.old.Reader(self.root)).values()),10)

    def test_failed_log_or_spoofed_count(self):
        doc=self.manifest(); row=doc['suites'][0]
        self.put(row['log']['path'],'Ran 2 tests in 0.01s\n\nFAILED\n'); row['log']=self.ref(Path(row['log']['path']))
        with self.assertRaises(ValueError):c.validate_tests(doc,c.old.Reader(self.root))
        doc=self.manifest(); doc['suites'][0]['count']=3
        with self.assertRaises(ValueError):c.validate_tests(doc,c.old.Reader(self.root))

    def test_duplicate_missing_or_wrong_suite(self):
        for mutate in (lambda x:x['suites'].pop(),lambda x:x['suites'].append(x['suites'][0]),
                       lambda x:x['suites'][0].update(name='unknown'),lambda x:x.update(test_count=True)):
            doc=self.manifest(); mutate(doc)
            with self.assertRaises(ValueError):c.validate_tests(doc,c.old.Reader(self.root))

    def test_changed_test_source_hash_rejected(self):
        doc=self.manifest(); self.put(doc['suites'][0]['source']['path'],'changed')
        with self.assertRaises(ValueError):c.validate_tests(doc,c.old.Reader(self.root))

    def test_prereview_binds_current_contracts(self):
        sources={}; implementations={}; inputs={}
        for name in ('monthly_endpoint155_model.py','monthly_endpoint155_runner.py','monthly_endpoint155_lifecycle.py'):
            p=self.put('work/'+name,'synthetic '+name); implementations[str(p)]=c.sha(p)
        p=self.put('work/monthly_endpoint155_prepare_review.py','synthetic independent review'); sources[str(p)]=c.sha(p)
        for name in (c.PROPOSAL,c.DESIGN):
            p=self.put(name,{'synthetic':name}); inputs[str(p)]=c.sha(p)
        doc=dict(passed=True,round=155,stage='preregistration_prepare_review',proposal_sha256='a'*64,
                 synthetic_only=True,real_155_label_fit_or_return_calculations=0,source_hashes=sources,
                 implementation_hashes=implementations,input_hashes=inputs)
        p=self.put('review.json',doc)
        self.assertEqual(c.prereview(c.old.Reader(self.root),p,'prepare','a'*64),sources)
        self.put('work/monthly_endpoint155_runner.py','changed after review')
        with self.assertRaises(ValueError):c.prereview(c.old.Reader(self.root),p,'prepare','a'*64)


class CompletionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.cache=('DATA.npz','SCORES.npz','META.json','PRETRAIN_ORDERS.json','MODEL_AUDIT.json')
        self.frozen=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json',
                     'TRAIN_OPEN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
        fake=SimpleNamespace(CACHE_FILES=self.cache,FROZEN_FILES=self.frozen,
                             Runner=lambda *args:SimpleNamespace(require_prepare_review=lambda:None))
        p=patch.object(c,'runner',return_value=fake);p.start();self.addCleanup(p.stop)
        for module in (c,c.old):
            p=patch.object(module,'PROTECTED_COUNT',3);p.start();self.addCleanup(p.stop)
        self.members={}
        for i in range(3):
            p=self.put('protected/'+str(i),'preserved '+str(i)); self.members[str(p)]=c.sha(p)
        manifest=self.put('protected.json',self.members)
        p=patch.object(c.old,'PROTECTED_SHA',c.sha(manifest));p.start();self.addCleanup(p.stop)
        self.orders={n:[dict(date='synthetic-%03d'%i,symbols=['A','B','C'],scores=[.3,.2,.1])
                         for i in range(122)] for n in c.NAMES}
        original=self.put('original.json',self.orders[c.NAMES[0]])
        sources={}
        for phase in ('prepare','outcome'):
            p=self.put(phase+'.py','synthetic reviewer '+phase);sources[phase]={str(p):c.sha(p)}
        self.plan=dict(round=155,status='REGISTERED',registered_utc='2020-01-01T00:00:00+00:00',
                       output_directory=str(self.out),arms=list(c.NAMES),primary_challenger=c.NAMES[2],
                       inputs=dict(protected_manifest=self.ref(manifest),original_orders=self.ref(original)),
                       hashes={str(original):c.sha(original),str(manifest):c.sha(manifest)},
                       independent_prepare_source_hashes=sources['prepare'],independent_outcome_source_hashes=sources['outcome'])
        for source in (Path(c.__file__),Path(__file__),Path(c.old.__file__)):
            p=self.put('work/'+source.name,source.read_bytes());self.plan['hashes'][str(p)]=c.sha(p)
        for key in ('actual_label_calculations_before_registration','actual_model_fits_before_registration',
                    'actual_feature_calculations_before_registration','actual_returns_before_registration','new_market_downloads'):
            self.plan[key]=0
        self.plan['no_observer_writes']=True
        self.plan_sha=c.sha(self.put(self.out/'PLAN.json',self.plan))
        self.results={n:{m:dict(metrics=dict(cagr=.676,sharpe=1.735,max_drawdown=-.312)) for m in c.MODES} for n in c.NAMES}
        self.close=rows();self.opening=rows();self.prepare_overrides={};self.outcome_overrides={};self.drop=None
        self.rebuild()
        doc=state();doc['registered_queue']=[c.queue_entry(self.plan,self.plan_sha)];self.put(c.STATE,doc)

    def output(self,n,v):return self.put(self.out/n,v)

    def rebuild(self):
        for n in self.cache:self.output(n,{'synthetic':n})
        self.output('PRETRAIN_ORDERS.json',self.orders)
        self.output('MODEL_AUDIT.json',dict(new_pooled_model_fits=2,annual=[dict(year=2020,status='fit')]))
        self.output('CACHE.json',{n:c.sha(self.out/n) for n in self.cache})
        prep=dict(passed=True,stage='prepare',producer_modules_imported=False,returns_or_nav_computed=False,
                  plan_sha256=self.plan_sha,cache_sha256=c.sha(self.out/'CACHE.json'),
                  orders_sha256=c.sha(self.out/'PRETRAIN_ORDERS.json'),source_hashes=self.plan['independent_prepare_source_hashes'],
                  orders_checked=366,candidate_cells_checked=5978,training_rows_checked=8820,months_checked=122,
                  prefixes_checked=123,forecast_arrays_verified=5,new_model_fits=2,
                  artifact_hashes={str(self.out/n):c.sha(self.out/n) for n in (*self.cache,'PLAN.json','CACHE.json')},
                  **{k:True for k in c.PREPARE_CHECKS})
        prep.update(self.prepare_overrides);self.output('INDEPENDENT_PREPARE.json',prep)
        self.output('TRAIN_TRIALS.json',self.close);self.output('TRAIN_OPEN_TRIALS.json',self.opening)
        self.output('WALKFORWARD.json',{}); chosen=c.selection(self.close,self.opening)
        self.output('FROZEN.json',dict(hashes={n:c.sha(self.out/n) for n in self.frozen},selection=chosen,
                                      pristine_oos=False,all_diagnostic_orders_frozen=True))
        self.output('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=c.sha(self.out/'FROZEN.json'),
                                                started_utc='2020-01-01T01:00:00+00:00'))
        self.output('RESULTS.json',self.results);self.output('MONTH_COMPARISON.json',[]);self.output('ATTRIBUTION.json',{})
        self.output('VALIDATION.json',dict(selection=chosen,no_observer_writes=True,no_intramonth_trades=True,new_model_fits=2))
        for n in c.NAMES:
            for m in c.MODES:
                for k in ('DAILY','TRADES','HOLDINGS'):self.output(n+'_'+m+'_'+k+'.json',[])
        artifacts={str(self.out/n):c.sha(self.out/n) for n in c.required_outcome_files(c.runner())}
        if self.drop:artifacts.pop(str(self.out/self.drop))
        metric=self.results[chosen['primary']]['same_close']['metrics']
        outcome=dict(passed=True,stage='evaluate',producer_modules_imported=False,
                     plan_sha256=self.plan_sha,cache_sha256=c.sha(self.out/'CACHE.json'),
                     orders_sha256=c.sha(self.out/'PRETRAIN_ORDERS.json'),source_hashes=self.plan['independent_outcome_source_hashes'],
                     artifact_hashes=artifacts,execution_paths_verified=9,total_monthly_orders_verified=366,
                     endpoint_forecast_arrays_verified=5,models_fit=0,producer_pooled_model_fits=2,
                     selected_name=chosen['primary'],primary_cagr=metric['cagr'],primary_target_reached=metric['cagr']>=.7,
                     frozen_sha256=c.sha(self.out/'FROZEN.json'),diagnose_started_sha256=c.sha(self.out/'DIAGNOSE_STARTED.json'),
                     audit_started_utc='2020-01-01T02:00:00+00:00',**{k:True for k in c.OUTCOME_CHECKS})
        outcome.update(self.outcome_overrides);self.output('INDEPENDENT_OUTCOME.json',outcome)
        self.output('TESTS.json',self.manifest())

    def test_complete_advances_exactly_three_trials_and_preserves_state(self):
        result=c.complete(self.root,self.plan_sha);doc=c.old.decode((self.root/c.STATE).read_bytes())
        self.assertEqual(len(doc['completed']),155);self.assertEqual(sum(x['trials'] for x in doc['completed']),978)
        self.assertEqual(doc['unrelated'],state()['unrelated']);self.assertEqual(doc['registered_queue'],[])
        self.assertFalse(result['target_reached']);self.assertEqual(doc['best_training_selected_candidate']['keep'],'original')

    def test_diagnostic_control_above_target_does_not_promote(self):
        self.results[c.NAMES[1]]['same_close']['metrics']['cagr']=.90;self.rebuild()
        result=c.complete(self.root,self.plan_sha)
        self.assertEqual(result['primary'],c.NAMES[0]);self.assertFalse(result['target_reached'])

    def test_admitted_candidate_can_reach_target(self):
        self.close[2]['training']['cagr']=.52;self.opening[2]['training']['cagr']=.52
        self.results[c.NAMES[2]]['same_close']['metrics']['cagr']=.71;self.rebuild()
        result=c.complete(self.root,self.plan_sha)
        self.assertTrue(result['target_reached']);self.assertEqual(result['primary'],c.NAMES[2])

    def test_failed_independent_prepare(self):
        self.prepare_overrides[c.PREPARE_CHECKS[-1]]=False;self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)
        self.assertFalse((self.out/'COMPLETION.json').exists())

    def test_incomplete_nine_path_scope(self):
        self.outcome_overrides['execution_paths_verified']=8;self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_unaudited_output(self):
        self.drop=c.NAMES[2]+'_next_open_cost_DAILY.json';self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_forecast_or_fit_counts_cannot_be_relabelled_zero(self):
        for key,value in (('endpoint_forecast_arrays_verified',4),('producer_pooled_model_fits',0),('models_fit',1)):
            self.outcome_overrides={key:value};self.rebuild()
            with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_initial_partial_fallback_enforced(self):
        self.orders[c.NAMES[2]][0]['symbols']=['A','B','D'];self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_changed_original_order_rejected(self):
        self.orders[c.NAMES[0]][5]['symbols']=['A','B','D'];self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_protected_member_change_stops_completion(self):
        self.put(next(iter(self.members)),'changed')
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)
        self.assertFalse((self.out/'COMPLETION.json').exists())

    def test_false_target_flag_rejected(self):
        self.outcome_overrides['primary_target_reached']=True;self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_pre_registration_outcome_chronology_rejected(self):
        self.outcome_overrides['audit_started_utc']='2019-12-31T23:00:00+00:00';self.rebuild()
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_changed_frozen_selection_rejected(self):
        path=self.out/'FROZEN.json';doc=c.old.decode(path.read_bytes());doc['selection']['primary']=c.NAMES[1];self.put(path,doc)
        with self.assertRaises(ValueError):c.complete(self.root,self.plan_sha)

    def test_write_before_state_commit_can_resume(self):
        with patch.object(c,'update_state',side_effect=RuntimeError('synthetic crash')):
            with self.assertRaises(RuntimeError):c.complete(self.root,self.plan_sha)
        digest=c.sha(self.out/'COMPLETION.json')
        c.complete(self.root,self.plan_sha)
        self.assertEqual(c.sha(self.out/'COMPLETION.json'),digest)

    def test_completed_retry_revalidates_without_recounting(self):
        c.complete(self.root,self.plan_sha);original=(self.root/c.STATE).read_bytes()
        self.assertTrue(c.complete(self.root,self.plan_sha)['already_completed'])
        self.assertEqual((self.root/c.STATE).read_bytes(),original)

    def test_state_commit_succeeded_but_return_lost_can_resume(self):
        original_update=c.update_state
        def lost(*args):original_update(*args);raise RuntimeError('lost successful response')
        with patch.object(c,'update_state',side_effect=lost):
            with self.assertRaises(RuntimeError):c.complete(self.root,self.plan_sha)
        self.assertTrue(c.complete(self.root,self.plan_sha)['already_completed'])


if __name__=='__main__':unittest.main()
