"""Synthetic lifecycle and evidence tests; no real source/price/return reads."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import tempfile
import unittest

import monthly_financial_coverage153_lifecycle as m


def state():
    rows=[dict(directory='old'+str(i),trials=6) for i in range(151)]
    rows.append(dict(directory='monthly_financial_coverage152',trials=56,
                     completion=dict(sha256=m.PREVIOUS_DONE)))
    return dict(completed=rows,registered_queue=[])


def outcome():
    return dict(passed=True,stage='evaluate',plan_sha256='a'*64,
                execution_paths_verified=18,total_monthly_orders_verified=732,
                financial_coverage_feature_matrices_verified=10,
                coverage_source_admission_verified=True,
                original149_earnings_cache_and_mask_verified=True,
                old44_source_records_and_holds_preserved=True,
                latest_catalog_blocking_independently_verified=True,
                prior150_batch1_earnings_cache_and_mask_verified=True,
                batch1_source_records_and_holds_preserved=True,
                prior151_batch2_earnings_cache_and_mask_verified=True,
                batch2_source_records_and_holds_preserved=True,
                prior152_batch3_earnings_cache_and_mask_verified=True,
                batch3_source_records_and_holds_preserved=True)


def put(root,name,obj):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    m.write(path,obj)
    return dict(path=name,sha256=m.r.sha(path))


class LifecycleTests(unittest.TestCase):
    def test_outcome_scope_exact_and_flags_required(self):
        value=outcome();m.validate_outcome_receipt(value,'a'*64)
        for key in value:
            changed=dict(value);del changed[key]
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_outcome_receipt(changed,'a'*64)

    def test_outcome_scope_rejects_bool_and_wrong_dimensions(self):
        for key,value in [('execution_paths_verified',9),('total_monthly_orders_verified',366),
                          ('financial_coverage_feature_matrices_verified',True),
                          ('old44_source_records_and_holds_preserved',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.validate_outcome_receipt(dict(outcome(),**{key:value}),'a'*64)

    def test_explicit_proposal_required_before_any_reads(self):
        for digest in (None,'','pending','f'*63,'x'*64):
            with self.subTest(digest=digest),self.assertRaises(ValueError):
                m.build_plan(Path('/unread-synthetic'),'unused',digest)

    def test_exact_predecessor(self):
        m.predecessor(state())
        for change in (lambda x:x['completed'].pop(),
                       lambda x:x['completed'][-1].update(trials=57),
                       lambda x:x['completed'][-1].update(directory='wrong'),
                       lambda x:x['completed'][-1]['completion'].update(sha256='wrong'),
                       lambda x:x['completed'][0].update(directory=m.STUDY)):
            value=state();change(value)
            with self.assertRaises(ValueError):m.predecessor(value)

    def test_snapshot_single_read(self):
        raw=m.old.encoded(state())
        with patch.object(Path,'read_bytes',return_value=raw) as reader:
            obj,digest=m.state_snapshot(Path('/never-read'))
        self.assertEqual(reader.call_count,1)
        self.assertEqual(obj,state());self.assertEqual(digest,hashlib.sha256(raw).hexdigest())

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.json';m.write(path,{'a':1})
            with self.assertRaises(FileExistsError):m.write(path,{'a':2})
            self.assertEqual(m.r.read(path),{'a':1})

    def test_state_compare_and_swap_with_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';m.write(path,state())
            raw=path.read_bytes();digest=m.r.sha(path)
            changed=state();changed['registered_queue']=['new']
            m.update_state(path,digest,changed)
            self.assertEqual(m.r.read(path),changed)
            self.assertEqual((path.parent/('PRE153_'+digest+'.json')).read_bytes(),raw)

    def test_state_concurrent_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';m.write(path,state());digest=m.r.sha(path)
            path.write_bytes(b'{"unrelated":"change"}')
            with self.assertRaises(ValueError):m.update_state(path,digest,{})
            self.assertEqual(m.r.read(path),{'unrelated':'change'})

    def test_existing_backup_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';m.write(path,state());digest=m.r.sha(path)
            (path.parent/('PRE153_'+digest+'.json')).write_text('bad')
            with self.assertRaises(ValueError):m.update_state(path,digest,{})
            self.assertEqual(m.r.read(path),state())

    def test_lock_rejects_parallel_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs/monthly_target70_research').mkdir(parents=True)
            with m.state_lock(root):
                with self.assertRaises(BlockingIOError):
                    with m.state_lock(root):pass
            with m.state_lock(root):pass

    def test_partial_completion_reentry_preserves_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'COMPLETION.json'
            prior=dict(completed_utc='prior',primary='control',result=.6)
            m.preserve_completion(path,prior);raw=path.read_bytes()
            self.assertEqual(m.preserve_completion(path,dict(prior,completed_utc='later')),prior)
            self.assertEqual(path.read_bytes(),raw)
            with self.assertRaises(ValueError):m.preserve_completion(path,dict(prior,result=.7))

    def test_required_artifacts_include_all_54_paths_and_prepare(self):
        names=m.required_outcome_files()
        for arm in m.r.NAMES:
            for mode in ('same_close','next_open_zero','next_open_cost'):
                for kind in ('DAILY','TRADES','HOLDINGS'):
                    self.assertIn(arm+'_'+mode+'_'+kind+'.json',names)
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(sum(name.endswith(('_DAILY.json','_TRADES.json','_HOLDINGS.json')) for name in names),54)
        for name in ('PLAN.json','CACHE.json','INDEPENDENT_PREPARE.json','TRAIN_OPEN_TRIALS.json',*m.r.CACHE_FILES):
            self.assertIn(name,names)

    def test_queue_entry_exact(self):
        result=m.queue_entry(Path('/any'),dict(registered_utc='time'),'h')
        self.assertEqual(result,dict(round=153,directory=m.STUDY,registered_utc='time',
            plan=dict(path='outputs/'+m.STUDY+'/PLAN.json',sha256='h')))

    def test_enqueue_idempotent_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=put(root,'input.json',{'a':1})
            prior=put(root,m.STATE,state())
            plan=dict(round=153,status='REGISTERED',registered_utc='time',hashes={source['path']:source['sha256']},
                      registration_precondition={'sha256':prior['sha256']})
            ref=put(root,'outputs/'+m.STUDY+'/PLAN.json',plan)
            first=m.enqueue(root,ref['sha256']);raw=(root/m.STATE).read_bytes()
            second=m.enqueue(root,ref['sha256'])
            self.assertFalse(first['already_enqueued']);self.assertTrue(second['already_enqueued'])
            self.assertEqual((root/m.STATE).read_bytes(),raw)

    def test_enqueue_rejects_unrelated_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=put(root,'input.json',{'a':1})
            prior=state();prior['registered_queue']=['unrelated'];state_ref=put(root,m.STATE,prior)
            ref=put(root,'outputs/'+m.STUDY+'/PLAN.json',dict(round=153,status='REGISTERED',registered_utc='time',
                hashes={source['path']:source['sha256']},registration_precondition={'sha256':state_ref['sha256']}))
            with self.assertRaises(ValueError):m.enqueue(root,ref['sha256'])
            self.assertEqual(m.r.read(root/m.STATE),prior)


class EvidenceTests(unittest.TestCase):
    def make_tests(self,root):
        suites=[]
        for name in sorted(m.TEST_SUITES):
            source=root/(name+'.py');source.write_text('pass\n')
            log=root/(name+'.log');log.write_text('Ran 2 tests in 0.01s\n\nOK\n')
            suites.append(dict(name=name,count=2,
                source=dict(path=str(source),sha256=m.r.sha(source)),
                log=dict(path=str(log),sha256=m.r.sha(log))))
        return dict(passed=True,synthetic_only=True,test_count=12,suites=suites)

    def test_complete_six_suite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);doc=self.make_tests(root)
            m.validate_tests(doc,m.old.Reader(root))
            self.assertNotIn('proposal',m.TEST_SUITES)
            self.assertIn('selection',m.TEST_SUITES)

    def test_incomplete_or_recounted_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);doc=self.make_tests(root)
            for key,value in [('test_count',11),('suites',doc['suites'][:-1]),('passed',False),('synthetic_only',False)]:
                changed=deepcopy(doc);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_tests(changed,m.old.Reader(root))

    def test_duplicate_suite_and_boolean_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);doc=self.make_tests(root)
            for change in (lambda x:x['suites'].append(x['suites'][0]),lambda x:x['suites'][0].update(count=True)):
                changed=deepcopy(doc);change(changed)
                with self.assertRaises(ValueError):m.validate_tests(changed,m.old.Reader(root))

    def test_log_failure_and_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);doc=self.make_tests(root);ref=doc['suites'][0]['log']
            Path(ref['path']).write_text('Ran 2 tests in 0.01s\nFAILED\n');ref['sha256']=m.r.sha(ref['path'])
            with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(root))
            Path(ref['path']).write_text('Ran 2 tests in 0.01s\nOK\n')
            with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(root))

    def test_refs_require_exact_path_key_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=put(root,'right.json',{'a':1});wrong=put(root,'wrong.json',{'a':1})
            self.assertEqual(m.refs(m.old.Reader(root),{'source':ref},{'source':'right.json'}),{'source':{'a':1}})
            for value in ({'source':wrong},{'source':dict(ref,sha256='a'*64)},{'wrong':ref}):
                with self.assertRaises(ValueError):m.refs(m.old.Reader(root),value,{'source':'right.json'})

    def test_reviewer_source_list_and_dict_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=put(root,'a.json',{})
            listpins=m.source_pins(m.old.Reader(root),[root/ref['path']])
            dictpins=m.source_pins(m.old.Reader(root),{ref['path']:ref['sha256']})
            self.assertEqual(listpins,dictpins)

    def test_prereview_requires_exact_source_and_three_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);src=put(root,'reviewer.py',{})
            expected={str((root/src['path']).resolve()):src['sha256']}
            contracts={}
            for name in ('work/monthly_financial_coverage153_runner.py','work/monthly_financial_coverage153_factor.py',m.PROPOSAL):
                ref=put(root,name,{});contracts[name]=ref['sha256']
            good=dict(passed=True,actual_factors_calculated=0,actual_returns_calculated=0,
                source_hashes=expected,reviewed_contract_hashes=contracts)
            put(root,'review.json',good)
            m.prospective_review(m.old.Reader(root),'review.json',expected)
            changed=deepcopy(good);changed['reviewed_contract_hashes'].pop(m.PROPOSAL)
            (root/'review.json').write_bytes(m.old.encoded(changed))
            with self.assertRaises(ValueError):m.prospective_review(m.old.Reader(root),'review.json',expected)

    def test_prereview_actual_work_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);put(root,'review.json',dict(passed=True,actual_factors_calculated=1,actual_returns_calculated=0))
            with self.assertRaises(ValueError):m.prospective_review(m.old.Reader(root),'review.json',{})

    def test_legacy_old_plan_hash_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);supplied={}
            for key,name in m.r.LEGACY_FILES.items():
                supplied[key]=put(root,m.r.LEGACY_OUT+'/'+name,{'synthetic':True})
            with self.assertRaisesRegex(ValueError,'Exact finalized149'):
                m.verify_legacy(m.old.Reader(root),supplied)

    def test_complete_cannot_skip_independent_preparation(self):
        class BadRunner:
            def __init__(self,*args):pass
            def require_prepare_review(self):raise ValueError('missing preparation audit')
        with tempfile.TemporaryDirectory() as directory,patch.object(m.r,'Runner',BadRunner):
            with self.assertRaisesRegex(ValueError,'missing preparation audit'):m.complete(Path(directory),'a'*64)
            self.assertFalse((Path(directory)/m.STATE).exists())



def prior_fixture(root, mutate_plan=None, mutate_review=None, mutate_done=None):
    sources = {ref['path']:ref['sha256'] for ref in
               (put(root, 'raw%02d.json'%i, {'synthetic':i}) for i in range(64))}
    legacy = {key:dict(path=m.r.LEGACY_OUT+'/'+name,sha256='a'*64)
              for key,name in m.r.LEGACY_FILES.items()}
    cache = {name:put(root,m.r.PRIOR_OUT+'/'+name,{'synthetic':name})['sha256']
             for name in m.r.CACHE_FILES}
    cache_ref = put(root,m.r.PRIOR_OUT+'/CACHE.json',cache)
    p = dict(round=152,status='REGISTERED',legacy149=deepcopy(legacy),hashes=dict(sources),raw_source_hashes=dict(sources))
    if mutate_plan:mutate_plan(p)
    plan_ref = put(root,m.r.PRIOR_OUT+'/PLAN.json',p)
    review = dict(passed=True,stage='evaluate',plan_sha256=plan_ref['sha256'],
        cache_sha256=cache_ref['sha256'],orders_sha256=cache['PRETRAIN_ORDERS.json'],
        artifact_hashes={m.r.PRIOR_OUT+'/'+k:v for k,v in cache.items()},
        source_hashes={'raw00.json':sources['raw00.json']})
    if mutate_review:mutate_review(review)
    review_ref = put(root,m.r.PRIOR_OUT+'/INDEPENDENT_OUTCOME.json',review)
    done = dict(round=152,plan_sha256=plan_ref['sha256'],independent_outcome_sha256=review_ref['sha256'])
    if mutate_done:mutate_done(done)
    done_ref = put(root,m.r.PRIOR_OUT+'/COMPLETION.json',done)
    return dict(plan=plan_ref,cache=cache_ref,outcome=review_ref,completion=done_ref),legacy


class PriorChainTests(unittest.TestCase):
    def check_prior(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);refs,legacy=prior_fixture(root,**kwargs)
            with patch.object(m,'PREVIOUS_PLAN',refs['plan']['sha256']),patch.object(m,'PREVIOUS_DONE',refs['completion']['sha256']):
                return m.verify_prior(m.old.Reader(root),refs,legacy)

    def test_full_finalized152_chain_and64_sources(self):
        self.assertEqual(self.check_prior()['round'],152)

    def test_prior_plan_requires_registered_state(self):
        with self.assertRaisesRegex(ValueError,'metadata'):
            self.check_prior(mutate_plan=lambda p:p.update(status='DRAFT'))

    def test_prior_done_plan_binding_required(self):
        with self.assertRaisesRegex(ValueError,'audit/cache'):
            self.check_prior(mutate_done=lambda p:p.update(plan_sha256='b'*64))

    def test_prior_done_outcome_binding_required(self):
        with self.assertRaisesRegex(ValueError,'audit/cache'):
            self.check_prior(mutate_done=lambda p:p.update(independent_outcome_sha256='b'*64))

    def test_prior_review_must_be_successful_evaluation(self):
        for change in (dict(passed=False),dict(stage='prepare'),dict(cache_sha256='b'*64)):
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'audit/cache'):
                self.check_prior(mutate_review=lambda p:p.update(change))

    def test_prior_review_must_cover_each_cache_artifact(self):
        with self.assertRaisesRegex(ValueError,'independently audited'):
            self.check_prior(mutate_review=lambda p:p['artifact_hashes'].pop(next(iter(p['artifact_hashes']))))

    def test_prior_orders_receipt_binding_required(self):
        with self.assertRaisesRegex(ValueError,'orders binding'):
            self.check_prior(mutate_review=lambda p:p.update(orders_sha256='b'*64))

    def test_prior_cannot_rewire_original149(self):
        with self.assertRaisesRegex(ValueError,'legacy149'):
            self.check_prior(mutate_plan=lambda p:p['legacy149']['plan'].update(sha256='b'*64))

    def test_prior_raw_scope_exact64(self):
        with self.assertRaisesRegex(ValueError,'Original64'):
            self.check_prior(mutate_plan=lambda p:p['raw_source_hashes'].pop('raw01.json'))

    def test_four_batches_are_all_validated_then_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            reader=m.old.Reader(Path(directory))
            refs={'batch1':{'dataset':{'path':'first.json','sha256':'a'*64}},
                  'batch2':{'dataset':{'path':'second.json','sha256':'b'*64}},
                  'batch3':{'dataset':{'path':'third.json','sha256':'c'*64}},
                  'batch4':{'dataset':{'path':'fourth.json','sha256':'d'*64}}}
            objects={'batch1':{'dataset':{'tag':'first'}},'batch2':{'dataset':{'tag':'second'}},
                     'batch3':{'dataset':{'tag':'third'}},'batch4':{'dataset':{'tag':'fourth'}}}
            def validate(_reader,supplied,old,batch):
                self.assertIs(supplied,refs[batch]);return objects[batch]
            with patch.object(m,'verify_coverage',side_effect=validate) as verify,patch.object(m.r,'combine_sources') as combine:
                result=m.verify_coverage_sources(reader,refs,{},dict(
                    coverage_sources={b:refs[b] for b in ('batch1','batch2','batch3')}))
            self.assertEqual(result,objects);self.assertEqual(verify.call_count,4)
            combine.assert_called_once_with(objects['batch1']['dataset'],objects['batch2']['dataset'],
                                            objects['batch3']['dataset'],objects['batch4']['dataset'],refs)

    def test_batch1_cannot_drift_from_prior152(self):
        with tempfile.TemporaryDirectory() as directory:
            reader=m.old.Reader(Path(directory));first={'dataset':{'path':'first.json','sha256':'a'*64}}
            changed={'dataset':{'path':'first.json','sha256':'b'*64}}
            with self.assertRaisesRegex(ValueError,'Frozen prior152'):
                m.verify_coverage_sources(reader,{'batch1':changed,'batch2':{},'batch3':{},'batch4':{}},{},
                                          dict(coverage_sources={'batch1':first,'batch2':{},'batch3':{}}))

    def test_batch2_cannot_drift_from_prior152(self):
        with tempfile.TemporaryDirectory() as directory:
            reader=m.old.Reader(Path(directory))
            second={'dataset':{'path':'second.json','sha256':'a'*64}}
            changed={'dataset':{'path':'second.json','sha256':'b'*64}}
            with self.assertRaisesRegex(ValueError,'Frozen prior152'):
                m.verify_coverage_sources(reader,{'batch1':{},'batch2':changed,'batch3':{},'batch4':{}},{},
                                          dict(coverage_sources={'batch1':{},'batch2':second,'batch3':{}}))

    def test_missing_or_extra_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reader=m.old.Reader(Path(directory))
            for refs in ({'batch1':{}},{'batch1':{},'batch2':{},'third':{}}):
                with self.assertRaisesRegex(ValueError,'Exact four'):
                    m.verify_coverage_sources(reader,refs,{},dict(coverage_sources={}))


class ProposalTests(unittest.TestCase):
    def test_builder_is_metadata_only_and_keeps_six_fixed_arms(self):
        import prepare_financial_coverage153_proposal as p
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            legacy={key:put(root,p.experiment.LEGACY_OUT+'/'+name,{'old':key}) for key,name in p.experiment.LEGACY_FILES.items()}
            prior={key:put(root,p.experiment.PRIOR_OUT+'/'+name,{'prior':key}) for key,name in p.experiment.LEGACY_FILES.items()}
            for entries in p.experiment.COVERAGE_REFS.values():
                for key,path in entries.items():
                    value=dict(passed=True,new_strategy_trials=0,actual_factors_calculated=0,actual_returns_calculated=0) if key=='admission' else {'source':key}
                    put(root,path,value)
            concept=put(root,'work/after149_coverage_extension_concept.json',{'synthetic':True})
            st=state();st['completed'][-1]['completion']['sha256']=prior['completion']['sha256']
            put(root,m.STATE,st)
            with patch.object(p,'ROOT',root),patch.object(p,'PREVIOUS_DONE',prior['completion']['sha256']),patch.object(p,'PREVIOUS_PLAN',prior['plan']['sha256']),patch.object(p,'CONCEPT_SHA',concept['sha256']):
                proposal=p.build()
            p.experiment.validate_proposal(proposal)
            self.assertEqual(proposal['arms'],['incumbent_local67','original_earnings','batch1_earnings','batch2_earnings','batch3_earnings','expanded_earnings'])
            self.assertEqual(set(proposal['sources']),{'legacy149','prior152','coverage_sources'})
            self.assertEqual(set(proposal['sources']['coverage_sources']),{'batch1','batch2','batch3','batch4'})
            self.assertFalse((root/'work/monthly_financial_coverage153_proposal.json').exists())
            self.assertEqual(proposal['actual_new_factors'],0)


class SyntheticRunner:
    def __init__(self,*args):pass
    def require_prepare_review(self):pass


def completion_fixture(root, rejected_control=None, missing_output=None):
    """Fabricated evidence only, with real digest/metadata checks and no replay."""
    from test_monthly_financial_coverage153_selection import passing_rows
    out=root/'outputs'/m.STUDY;out.mkdir(parents=True)
    source=put(root,'synthetic-reviewer.py',{'generated':True})
    manifest=put(root,'synthetic-protected.json',{'synthetic-'+str(i):'0'*64 for i in range(21444)})
    plan=dict(round=153,status='REGISTERED',registered_utc='synthetic-time',
              inputs=dict(protected_manifest=manifest),
              independent_outcome_source_hashes={source['path']:source['sha256']})
    pref=put(root,str((out/'PLAN.json').relative_to(root)),plan)
    for name in m.r.CACHE_FILES:put(root,str((out/name).relative_to(root)),{'synthetic':name})
    cache={name:m.r.sha(out/name) for name in m.r.CACHE_FILES}
    put(root,str((out/'CACHE.json').relative_to(root)),cache)
    close=passing_rows();opened=passing_rows()
    if rejected_control is not None:close[rejected_control]['training']['cagr']=.6
    selection=m.r.select(close,opened)
    values={'TRAIN_TRIALS.json':close,'TRAIN_OPEN_TRIALS.json':opened,'WALKFORWARD.json':{},
            'INDEPENDENT_PREPARE.json':{'synthetic':True}}
    for name,value in values.items():put(root,str((out/name).relative_to(root)),value)
    frozen=dict(selection=selection,hashes={name:m.r.sha(out/name) for name in m.r.FROZEN_FILES})
    put(root,str((out/'FROZEN.json').relative_to(root)),frozen)
    put(root,str((out/'VALIDATION.json').relative_to(root)),dict(selection=selection,no_observer_writes=True,no_intramonth_trades=True,new_model_fits=0))
    put(root,str((out/'DIAGNOSE_STARTED.json').relative_to(root)),dict(plan_sha256=pref['sha256']))
    results={name:{mode:dict(metrics=dict(cagr=.81 if i==5 else .61,sharpe=1.5,max_drawdown=-.2))
             for mode in ('same_close','next_open_zero','next_open_cost')} for i,name in enumerate(m.r.NAMES)}
    put(root,str((out/'RESULTS.json').relative_to(root)),results)
    for name in m.required_outcome_files():
        if not (out/name).exists():put(root,str((out/name).relative_to(root)),[])
    receipt=outcome();receipt.update(plan_sha256=pref['sha256'],
        cache_sha256=m.r.sha(out/'CACHE.json'),orders_sha256=m.r.sha(out/'PRETRAIN_ORDERS.json'),
        source_hashes={source['path']:source['sha256']},
        artifact_hashes={str(out/name):m.r.sha(out/name) for name in m.required_outcome_files()},
        selected_name=selection['primary'],primary_cagr=.81 if selection['admitted'] else .61,
        primary_target_reached=selection['admitted'])
    if missing_output:receipt['artifact_hashes'].pop(str(out/missing_output))
    put(root,str((out/'INDEPENDENT_OUTCOME.json').relative_to(root)),receipt)
    st=state();st['registered_queue']=[m.queue_entry(root,plan,pref['sha256'])]
    st['best_training_selected_candidate']=dict(cagr=.67)
    put(root,m.STATE,st)
    return out,pref['sha256']


class CompleteSyntheticTests(unittest.TestCase):
    def test_close_advances_exact_round_and_trial_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,psha=completion_fixture(root)
            with patch.object(m.r,'Runner',SyntheticRunner),patch.object(m.old,'protected_scan') as scan:
                result=m.complete(root,psha)
            scan.assert_called_once()
            st=m.r.read(root/m.STATE);done=m.r.read(out/'COMPLETION.json')
            self.assertEqual((len(st['completed']),sum(x['trials'] for x in st['completed'])),(153,968))
            self.assertEqual((done['trials'],done['independent_execution_paths'],done['protected_files']),(6,18,21444))
            self.assertEqual(st['registered_queue'],[])
            self.assertEqual(result['primary'],m.r.NAMES[5])
            self.assertEqual(set(done['all_arm_results']),set(m.r.NAMES))
            self.assertFalse(st['best_training_selected_candidate']['observer_promotion'])

    def test_every_first88_control_can_veto_even_when_full_period_challenger_is_higher(self):
        for control in range(5):
            with self.subTest(control=control),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out,psha=completion_fixture(root,rejected_control=control)
                with patch.object(m.r,'Runner',SyntheticRunner),patch.object(m.old,'protected_scan'):
                    result=m.complete(root,psha)
                self.assertEqual(result['primary'],m.r.NAMES[0])
                self.assertFalse(result['target_reached'])
                self.assertEqual(m.r.read(root/m.STATE)['best_training_selected_candidate'],{'cagr':.67})

    def test_missing_any_arm_path_audit_stops_before_protection_or_state_commit(self):
        missing=m.r.NAMES[4]+'_next_open_cost_HOLDINGS.json'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,psha=completion_fixture(root,missing_output=missing);before=(root/m.STATE).read_bytes()
            with patch.object(m.r,'Runner',SyntheticRunner),patch.object(m.old,'protected_scan') as scan:
                with self.assertRaisesRegex(ValueError,'Unreviewed outcome'):m.complete(root,psha)
            scan.assert_not_called()
            self.assertEqual((root/m.STATE).read_bytes(),before)
            self.assertFalse((out/'COMPLETION.json').exists())

    def test_partial_close_recovery_preserves_frozen_completion_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,psha=completion_fixture(root)
            with patch.object(m.r,'Runner',SyntheticRunner),patch.object(m.old,'protected_scan'),patch.object(m,'update_state',side_effect=ValueError('interrupted commit')):
                with self.assertRaisesRegex(ValueError,'interrupted commit'):m.complete(root,psha)
            frozen=(out/'COMPLETION.json').read_bytes()
            self.assertEqual(len(m.r.read(root/m.STATE)['completed']),152)
            with patch.object(m.r,'Runner',SyntheticRunner),patch.object(m.old,'protected_scan'):
                m.complete(root,psha)
            self.assertEqual((out/'COMPLETION.json').read_bytes(),frozen)
            self.assertEqual(len(m.r.read(root/m.STATE)['completed']),153)



if __name__=='__main__':
    unittest.main(verbosity=2)
