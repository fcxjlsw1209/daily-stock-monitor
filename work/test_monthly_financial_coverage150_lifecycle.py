"""Synthetic lifecycle and evidence tests; no real source/price/return reads."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import tempfile
import unittest

import monthly_financial_coverage150_lifecycle as m


def state():
    rows=[dict(directory='old'+str(i),trials=6) for i in range(148)]
    rows.append(dict(directory='monthly_stock_comp149',trials=62,
                     completion=dict(sha256=m.PREVIOUS_DONE)))
    return dict(completed=rows,registered_queue=[])


def outcome():
    return dict(passed=True,stage='evaluate',plan_sha256='a'*64,
                execution_paths_verified=9,total_monthly_orders_verified=366,
                financial_coverage_feature_matrices_verified=4,
                coverage_source_admission_verified=True,
                original149_earnings_cache_and_mask_verified=True,
                old44_source_records_and_holds_preserved=True,
                latest_catalog_blocking_independently_verified=True)


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
        for key,value in [('execution_paths_verified',12),('total_monthly_orders_verified',488),
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
                       lambda x:x['completed'][-1].update(trials=63),
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
            self.assertEqual((path.parent/('PRE150_'+digest+'.json')).read_bytes(),raw)

    def test_state_concurrent_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';m.write(path,state());digest=m.r.sha(path)
            path.write_bytes(b'{"unrelated":"change"}')
            with self.assertRaises(ValueError):m.update_state(path,digest,{})
            self.assertEqual(m.r.read(path),{'unrelated':'change'})

    def test_existing_backup_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';m.write(path,state());digest=m.r.sha(path)
            (path.parent/('PRE150_'+digest+'.json')).write_text('bad')
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

    def test_required_artifacts_include_all_27_paths_and_prepare(self):
        names=m.required_outcome_files()
        for arm in m.r.NAMES:
            for mode in ('same_close','next_open_zero','next_open_cost'):
                for kind in ('DAILY','TRADES','HOLDINGS'):
                    self.assertIn(arm+'_'+mode+'_'+kind+'.json',names)
        self.assertEqual(len(names),len(set(names)))
        for name in ('PLAN.json','CACHE.json','INDEPENDENT_PREPARE.json','TRAIN_OPEN_TRIALS.json',*m.r.CACHE_FILES):
            self.assertIn(name,names)

    def test_queue_entry_exact(self):
        result=m.queue_entry(Path('/any'),dict(registered_utc='time'),'h')
        self.assertEqual(result,dict(round=150,directory=m.STUDY,registered_utc='time',
            plan=dict(path='outputs/'+m.STUDY+'/PLAN.json',sha256='h')))

    def test_enqueue_idempotent_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=put(root,'input.json',{'a':1})
            prior=put(root,m.STATE,state())
            plan=dict(round=150,status='REGISTERED',registered_utc='time',hashes={source['path']:source['sha256']},
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
            ref=put(root,'outputs/'+m.STUDY+'/PLAN.json',dict(round=150,status='REGISTERED',registered_utc='time',
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
            for name in ('work/monthly_financial_coverage150_runner.py','work/monthly_financial_coverage150_factor.py',m.PROPOSAL):
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


if __name__=='__main__':
    unittest.main(verbosity=2)
