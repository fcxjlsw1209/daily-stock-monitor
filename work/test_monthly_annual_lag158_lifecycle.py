"""Synthetic-only checks for pilot registration and immutable state handling."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import monthly_annual_lag158_lifecycle as m


def state():
    rows=[dict(directory='old'+str(i),trials=6) for i in range(156)]
    rows.append(dict(directory='monthly_filing_association157',trials=50,
                     completion=dict(sha256=m.PREVIOUS_SHA)))
    return dict(completed=rows,registered_queue=[],current='monthly_filing_association157',
                best_training_selected_candidate=dict(cagr=.6766617372041686))


class Registration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()
    def test_predecessor_accepts_exact_counter_without_mutation(self):
        value=state();before=copy.deepcopy(value);m.predecessor(value);self.assertEqual(value,before)
    def test_wrong_round_count(self):
        value=state();value['completed'].pop()
        with self.assertRaises(ValueError):m.predecessor(value)
    def test_wrong_trial_count(self):
        value=state();value['completed'][0]['trials']+=1
        with self.assertRaises(ValueError):m.predecessor(value)
    def test_bool_is_not_trial_count(self):
        value=state();value['completed'][0]['trials']=True
        with self.assertRaises(ValueError):m.predecessor(value)
    def test_wrong_predecessor_digest(self):
        value=state();value['completed'][-1]['completion']['sha256']='0'*64
        with self.assertRaises(ValueError):m.predecessor(value)
    def test_write_once(self):
        path=self.root/'receipt.json';m.write(path,dict(value=1))
        before=path.read_bytes()
        with self.assertRaises(FileExistsError):m.write(path,dict(value=2))
        self.assertEqual(before,path.read_bytes())
    def test_noncentral_pin_never_uses_archive(self):
        path=self.root/'evidence.json';path.write_text('one')
        with self.assertRaises(ValueError):m.source_pins(m.old.Reader(self.root),{'evidence.json':'0'*64})
    def test_historical_alias_rejects_unknown_digest(self):
        with self.assertRaisesRegex(ValueError,'Only exact historical'):
            m.source_pins(m.old.Reader(self.root),{m.STATE:'0'*64})
    def test_historical_alias_reads_only_exact_archive(self):
        reader=m.old.Reader(self.root);seen=[]
        def digest(path):seen.append(Path(path));return m.HISTORICAL_SHA
        with patch.object(reader,'hash',side_effect=digest):
            result=m.source_pins(reader,{m.STATE:m.HISTORICAL_SHA})
        self.assertEqual(seen,[self.root/m.ARCHIVE])
        self.assertEqual(result,{str(self.root/m.ARCHIVE):m.HISTORICAL_SHA})
    def test_no_empty_source_closure(self):
        with self.assertRaises(ValueError):m.source_pins(m.old.Reader(self.root),{})
    def test_queue_entry_exact(self):
        self.assertEqual(m.queue_entry({'registered_utc':'t'},'a'*64),dict(round=158,
            directory=m.STUDY,plan=dict(path=m.OUT+'/PLAN.json',sha256='a'*64),registered_utc='t'))
    def test_source_review_required(self):
        reader=m.old.Reader(self.root)
        with self.assertRaises((ValueError,FileNotFoundError)):m.validate_source(reader)
    def tests_doc(self):
        rows=[]
        (self.root/'work').mkdir(exist_ok=True)
        for name,filename in m.SUITES.items():
            p=self.root/'work'/filename;p.write_text('# synthetic')
            log=self.root/(name+'.log');log.write_text('Ran 1 test in 0.001s\n\nOK\n')
            rows.append(dict(name=name,count=1,passed=True,
                source=dict(path=str(p),sha256=m.sha(p)),log=dict(path=str(log),sha256=m.sha(log))))
        return dict(passed=True,synthetic_only=True,suites=rows,test_count=5)
    def test_complete_bound_test_logs(self):
        doc=self.tests_doc();self.assertEqual(m.validate_tests(doc,m.old.Reader(self.root)),{k:1 for k in m.SUITES})
    def test_partial_test_manifest_rejected(self):
        doc=self.tests_doc();doc['suites'].pop();doc['test_count']=4
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_replaced_test_source_rejected(self):
        doc=self.tests_doc();Path(doc['suites'][0]['source']['path']).write_text('# altered')
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_same_basename_outside_work_rejected(self):
        doc=self.tests_doc();row=doc['suites'][0]
        wrong=self.root/Path(row['source']['path']).name;wrong.write_text('# synthetic')
        row['source']=dict(path=str(wrong),sha256=m.sha(wrong))
        with self.assertRaisesRegex(ValueError,'Wrong test suite'):
            m.validate_tests(doc,m.old.Reader(self.root))
    def test_non_synthetic_rejected(self):
        doc=self.tests_doc();doc['synthetic_only']=False
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_failed_log_rejected_even_with_matching_hash(self):
        doc=self.tests_doc();row=doc['suites'][0];p=Path(row['log']['path'])
        p.write_text('Ran 1 test in 0.001s\nFAILED (errors=1)\n');row['log']['sha256']=m.sha(p)
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_duplicate_suite_rejected(self):
        doc=self.tests_doc();doc['suites'][1]=copy.deepcopy(doc['suites'][0])
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_count_bool_rejected(self):
        doc=self.tests_doc();doc['suites'][0]['count']=True
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_multiple_log_runs_rejected(self):
        doc=self.tests_doc();row=doc['suites'][0];p=Path(row['log']['path'])
        p.write_text('Ran 1 test in 0.001s\nOK\nRan 1 test in 0.001s\nOK\n');row['log']['sha256']=m.sha(p)
        with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(self.root))
    def test_wrong_receipt_stage(self):
        with self.assertRaises(ValueError):
            m.validate_receipt(m.old.Reader(self.root),dict(passed=True,stage='evaluate'),
                               'prepare',(),{},self.root,'x')
    def test_state_decoder_preserves_original_path_strings(self):
        p=self.root/'state.json';p.write_text('{"old":"/Users/user/OneDrive/example"}')
        value,digest=m.prior.state_snapshot(p)
        self.assertEqual(value['old'],'/Users/user/OneDrive/example')
        self.assertEqual(digest,hashlib.sha256(p.read_bytes()).hexdigest())
    def test_state_compare_and_swap_prevents_concurrent_write(self):
        p=self.root/'state.json';p.write_text('{}')
        with self.assertRaises(ValueError):m.prior.update_state(p,'0'*64,dict(changed=True))
        self.assertEqual(p.read_text(),'{}')


class CompletionRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.out=self.root/m.OUT;self.out.mkdir(parents=True)
        self.plan=dict(registered_utc='2026-01-01T00:00:00+00:00');self.plan_sha='a'*64
        self.state_path=self.root/m.STATE;self.state_path.parent.mkdir(parents=True)
        original=state();original['registered_queue']=[m.queue_entry(self.plan,self.plan_sha)]
        original['unrelated_evidence']={'preserve':'original'}
        m.write(self.state_path,original)
        self.proof=dict(passed=True,checked=21444,verified=21444,changed=[],missing=[])
        self.payload=dict(round=158,directory=m.STUDY,status='completed_independently_verified',
            plan_sha256=self.plan_sha,primary=m.NAMES[0],selection=dict(primary=m.NAMES[0],admitted=False),
            primary_cagr=.6766617372041686,primary_sharpe=1.7348,target_reached=False,trials=3,
            new_model_fits=0,independent_execution_paths=9,protected_files=21444)
        self.marker=self.root/'source.json';m.write(self.marker,{'fixed':True})
    def tearDown(self):self.tmp.cleanup()
    def publish(self):
        reader=m.old.Reader(self.root);reader.hash(self.marker)
        return m.publish_completion(reader,self.plan,self.plan_sha,self.proof,self.payload)
    def test_first_publication_and_idempotent_repeat(self):
        first=self.publish();files=[self.state_path,self.out/'COMPLETION.json',self.out/'PROTECTED_AFTER.json']
        raw=[p.read_bytes() for p in files];second=self.publish()
        self.assertFalse(first['already_published']);self.assertTrue(second['already_published'])
        self.assertEqual(raw,[p.read_bytes() for p in files])
        value,_=m.prior.state_snapshot(self.state_path)
        self.assertEqual(value['unrelated_evidence'],{'preserve':'original'})
        self.assertEqual(len(value['completed']),158);self.assertEqual(value['registered_queue'],[])
    def test_recovery_after_only_protection_receipt(self):
        m.write(self.out/'PROTECTED_AFTER.json',self.proof)
        raw=(self.out/'PROTECTED_AFTER.json').read_bytes();self.publish()
        self.assertEqual(raw,(self.out/'PROTECTED_AFTER.json').read_bytes())
    def test_recovery_after_completion_before_central_commit(self):
        with patch.object(m.prior,'update_state',side_effect=RuntimeError('interrupted CAS')):
            with self.assertRaises(RuntimeError):self.publish()
        raw=(self.out/'COMPLETION.json').read_bytes();self.publish()
        self.assertEqual(raw,(self.out/'COMPLETION.json').read_bytes())
    def test_wrong_queue_fails_before_publishing_any_receipt(self):
        value,_=m.prior.state_snapshot(self.state_path);value['registered_queue']=[]
        self.state_path.write_bytes(m.old.encoded(value))
        with self.assertRaises(ValueError):self.publish()
        self.assertFalse((self.out/'PROTECTED_AFTER.json').exists())
        self.assertFalse((self.out/'COMPLETION.json').exists())
    def test_changed_protection_never_overwritten(self):
        m.write(self.out/'PROTECTED_AFTER.json',dict(self.proof,checked=1))
        raw=(self.out/'PROTECTED_AFTER.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'protection proof differs'):self.publish()
        self.assertEqual(raw,(self.out/'PROTECTED_AFTER.json').read_bytes())
    def test_changed_completion_never_overwritten(self):
        with patch.object(m.prior,'update_state',side_effect=RuntimeError('interrupted CAS')):
            with self.assertRaises(RuntimeError):self.publish()
        self.payload['primary_cagr']=.99;raw=(self.out/'COMPLETION.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'completion differs'):self.publish()
        self.assertEqual(raw,(self.out/'COMPLETION.json').read_bytes())
    def test_completion_cannot_replace_missing_protection(self):
        with patch.object(m.prior,'update_state',side_effect=RuntimeError('interrupted CAS')):
            with self.assertRaises(RuntimeError):self.publish()
        (self.out/'PROTECTED_AFTER.json').unlink()
        with self.assertRaisesRegex(ValueError,'without original protection'):self.publish()
        self.assertFalse((self.out/'PROTECTED_AFTER.json').exists())
    def test_completed_central_entry_must_match(self):
        self.publish();value,_=m.prior.state_snapshot(self.state_path)
        value['completed'][-1]['primary_cagr']=.99;self.state_path.write_bytes(m.old.encoded(value))
        with self.assertRaisesRegex(ValueError,'central binding differs'):self.publish()
    def test_completed_return_detects_concurrent_state_change(self):
        self.publish();reader=m.old.Reader(self.root);reader.hash(self.marker)
        verify=reader.verify
        def changed(mapping):
            result=verify(mapping)
            value,_=m.prior.state_snapshot(self.state_path);value['concurrent_update']=True
            self.state_path.write_bytes(m.old.encoded(value));return result
        with patch.object(reader,'verify',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Concurrent completed state change'):
                m.publish_completion(reader,self.plan,self.plan_sha,self.proof,self.payload)
    def test_recovery_does_not_erase_concurrent_unrelated_state(self):
        with patch.object(m.prior,'update_state',side_effect=RuntimeError('interrupted CAS')):
            with self.assertRaises(RuntimeError):self.publish()
        value,_=m.prior.state_snapshot(self.state_path);value['new_progress']='keep'
        self.state_path.write_bytes(m.old.encoded(value));self.publish()
        current,_=m.prior.state_snapshot(self.state_path);self.assertEqual(current['new_progress'],'keep')


if __name__=='__main__':unittest.main()
