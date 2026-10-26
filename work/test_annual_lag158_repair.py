"""Synthetic temporary-directory checks; no actual market inputs."""
import copy
import tempfile
import unittest
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import annual_lag158_repair as m


class RepairContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        (self.root/m.OUT).mkdir(parents=True);(self.root/'work').mkdir()
        for name in m.FILES:(self.root/name).write_text('# synthetic fixture')
        self.plan=dict(round=158,status='REGISTERED',dose=.01,registered_utc='2026-01-01T00:00:00+00:00')
        m.write_once(self.root/m.OUT/'PLAN.json',self.plan)
        self.digest=m.sha(self.root/m.OUT/'PLAN.json')
        self.patcher=patch.object(m,'PLAN_SHA',self.digest);self.patcher.start()
        self.contract=dict(m.CONTRACT,original_plan_sha256=self.digest)
        self.cp=patch.object(m,'CONTRACT',self.contract);self.cp.start()
        (self.root/m.FAILURE_LOG).write_text('ValueError: Explicit source reason list required: known amendments')
        self.failure=dict(original_plan_sha256=self.digest,exit_code=1,
            strategy_returns_calculated=0,prepared_artifacts_published=0,lag_feature_build_attempted=True,
            log=m.reference(self.root,m.FAILURE_LOG),recorded_utc='2026-01-02T00:00:00+00:00')
        m.write_once(self.root/m.FAILURE,self.failure)
        self.review=dict(passed=True,synthetic_only=True,semantics_preserved=True,
            original_plan_sha256=self.digest,actual_returns_calculated=0,
            implementation_hashes=m.implementation_hashes(self.root),failure_record=m.reference(self.root,m.FAILURE),
            reviewed_utc='2026-01-03T00:00:00+00:00',
            tests={name:dict(passed=True,count=1,source_sha256=m.sha(self.root/path),
                log='Ran 1 test in 0.01s\n\nOK\n') for name,path in m.SUITES.items()})
        m.write_once(self.root/m.PREREVIEW,self.review)
        self.amend=dict(self.contract,registered_utc='2026-01-04T00:00:00+00:00',
            implementation_hashes=m.implementation_hashes(self.root),failure_record=m.reference(self.root,m.FAILURE),
            independent_prereview=m.reference(self.root,m.PREREVIEW))
        m.write_once(self.root/m.AMENDMENT,self.amend);self.amend_sha=m.sha(self.root/m.AMENDMENT)
    def tearDown(self):self.cp.stop();self.patcher.stop();self.temp.cleanup()
    def verify(self):return m.verify_amendment(self.root,self.digest,self.amend_sha)
    def test_bound_amendment_passes(self):self.assertEqual(self.verify(),self.amend)
    def test_wrong_plan_digest(self):
        with self.assertRaises(ValueError):m.verify_amendment(self.root,'0'*64,self.amend_sha)
    def test_wrong_amendment_digest(self):
        with self.assertRaises(ValueError):m.verify_amendment(self.root,self.digest,'0'*64)
    def test_changed_original_plan(self):
        (self.root/m.OUT/'PLAN.json').write_text('{}')
        with self.assertRaises(ValueError):self.verify()
    def test_changed_repair_code(self):
        (self.root/m.FILES[0]).write_text('# different')
        with self.assertRaises(ValueError):self.verify()
    def test_changed_failure_log(self):
        (self.root/m.FAILURE_LOG).write_text('different')
        with self.assertRaises(ValueError):self.verify()
    def test_changed_review(self):
        (self.root/m.PREREVIEW).write_text('{}')
        with self.assertRaises(ValueError):self.verify()
    def test_changed_formula_contract_rejected(self):
        obj=dict(self.amend,formula_changed=True)
        (self.root/m.AMENDMENT).write_text(__import__('json').dumps(obj));self.amend_sha=m.sha(self.root/m.AMENDMENT)
        with self.assertRaises(ValueError):self.verify()
    def test_nonzero_failed_attempt_returns_rejected(self):
        with self.assertRaises(ValueError):m.validate_failure(self.root,dict(self.failure,strategy_returns_calculated=1))
    def test_boolean_failed_attempt_returns_rejected(self):
        with self.assertRaises(ValueError):m.validate_failure(self.root,dict(self.failure,strategy_returns_calculated=False))
    def test_missing_test_suite_rejected(self):
        review=copy.deepcopy(self.review);review['tests'].pop('cell')
        with self.assertRaises(ValueError):m.validate_prereview(self.root,review)
    def test_failed_log_rejected(self):
        review=copy.deepcopy(self.review);review['tests']['cell']['log']='Ran 1 test in 0.01s\nFAILED'
        with self.assertRaises(ValueError):m.validate_prereview(self.root,review)
    def test_repeated_test_logs_rejected(self):
        review=copy.deepcopy(self.review);review['tests']['cell']['log']*=2
        with self.assertRaises(ValueError):m.validate_prereview(self.root,review)
    def test_boolean_count_rejected(self):
        review=copy.deepcopy(self.review);review['tests']['cell']['count']=True
        with self.assertRaises(ValueError):m.validate_prereview(self.root,review)
    def test_bad_reference_path_rejected(self):
        with self.assertRaises(ValueError):m.verify_reference(self.root,dict(path='../x',sha256='0'*64),m.FAILURE)
    def test_write_once_preserves_bytes(self):
        before=(self.root/m.FAILURE).read_bytes()
        with self.assertRaises(FileExistsError):m.write_once(self.root/m.FAILURE,{})
        self.assertEqual(before,(self.root/m.FAILURE).read_bytes())
    def test_unprepared_does_not_count_failure_artifacts(self):m.unprepared(self.root)
    def test_unprepared_rejects_cache(self):
        (self.root/m.OUT/'DATA.npz').write_bytes(b'partial')
        with self.assertRaises(ValueError):m.unprepared(self.root)
    def test_augment_preserves_old_closures_and_input(self):
        report=dict(source_hashes={'old':'a'},artifact_hashes={'oldartifact':'b'},passed=True)
        before=copy.deepcopy(report)
        result=m.augment_report(self.root,report,self.digest,self.amend_sha)
        self.assertEqual(report,before);self.assertEqual(result['source_hashes'],before['source_hashes'])
        self.assertEqual(result['artifact_hashes'],before['artifact_hashes'])
        self.assertEqual(m.verify_repair_report(self.root,result,self.digest,self.amend_sha),self.amend)
    def test_missing_repair_evidence_rejected(self):
        report=m.augment_report(self.root,{},self.digest,self.amend_sha);report['repair_artifact_hashes'].popitem()
        with self.assertRaises(ValueError):m.verify_repair_report(self.root,report,self.digest,self.amend_sha)
    def test_double_augmentation_rejected(self):
        report=m.augment_report(self.root,{},self.digest,self.amend_sha)
        with self.assertRaises(ValueError):m.augment_report(self.root,report,self.digest,self.amend_sha)
    def test_chronology_inversion_rejected(self):
        obj=dict(self.amend,registered_utc='2025-12-01T00:00:00+00:00')
        (self.root/m.AMENDMENT).write_text(__import__('json').dumps(obj));self.amend_sha=m.sha(self.root/m.AMENDMENT)
        with self.assertRaises(ValueError):self.verify()

    def registration_fixture(self, fail_update=False):
        state={'registered_queue':['fixed158']}
        writes=[]
        def snapshot(path):return copy.deepcopy(state),'before-state'
        def update(path,before,value):
            self.assertEqual(before,'before-state')
            if fail_update:raise RuntimeError('simulated interrupted update')
            state.clear();state.update(value);writes.append(copy.deepcopy(value))
        module=SimpleNamespace(STATE='state.json',
            old=SimpleNamespace(Reader=lambda root:SimpleNamespace(verify=lambda hashes:None)),
            prior=SimpleNamespace(state_lock=lambda root:nullcontext(),state_snapshot=snapshot,update_state=update),
            predecessor=lambda state:None,queue_entry=lambda plan,digest:'fixed158')
        plan=dict(self.plan,hashes={})
        return state,writes,module,plan

    def registration_call(self,module,plan):
        with patch.dict(sys.modules,{'monthly_annual_lag158_lifecycle':module}), patch.object(m,'verify_plan',return_value=plan):
            return m.register(self.root)

    def test_register_first_publication_and_idempotent_reentry(self):
        (self.root/m.AMENDMENT).unlink()
        state,writes,module,plan=self.registration_fixture()
        first=self.registration_call(module,plan)
        raw=(self.root/m.AMENDMENT).read_bytes()
        second=self.registration_call(module,plan)
        self.assertEqual(first,second);self.assertEqual(len(writes),1)
        self.assertEqual(raw,(self.root/m.AMENDMENT).read_bytes())
        self.assertEqual(state['active_execution_amendment'],first['amendment'])

    def test_register_recovers_published_amendment_after_failed_update(self):
        (self.root/m.AMENDMENT).unlink()
        state,writes,module,plan=self.registration_fixture(fail_update=True)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):self.registration_call(module,plan)
        raw=(self.root/m.AMENDMENT).read_bytes()
        state,writes,module,plan=self.registration_fixture()
        result=self.registration_call(module,plan)
        self.assertEqual(raw,(self.root/m.AMENDMENT).read_bytes())
        self.assertEqual(len(writes),1)
        self.assertEqual(state['active_execution_amendment'],result['amendment'])

    def test_register_rejects_unbound_amendment_with_changed_state(self):
        state,writes,module,plan=self.registration_fixture()
        with self.assertRaisesRegex(ValueError,'continuation changed'):self.registration_call(module,plan)
        self.assertEqual(writes,[])

    def test_register_rejects_different_active_amendment(self):
        state,writes,module,plan=self.registration_fixture()
        state['active_execution_amendment']={'path':'other','sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'Different active'):self.registration_call(module,plan)
        self.assertEqual(writes,[])

    def test_register_rejects_tampered_published_amendment(self):
        state,writes,module,plan=self.registration_fixture()
        (self.root/m.FILES[0]).write_text('# changed')
        with self.assertRaises(ValueError):self.registration_call(module,plan)
        self.assertEqual(writes,[])

    def test_register_replaces_only_verified_previous_binding(self):
        (self.root/m.AMENDMENT).unlink()
        previous=self.root/'previous.json';m.write_once(previous,{'round':157})
        ref=m.reference(self.root,'previous.json')
        state,writes,module,plan=self.registration_fixture()
        state['active_execution_amendment']=ref
        with patch.object(m,'PREVIOUS_AMENDMENT',ref):
            result=self.registration_call(module,plan)
        self.assertEqual(state['active_execution_amendment'],result['amendment'])
        self.assertEqual(len(writes),1)

    def test_register_rejects_changed_previous_binding_file(self):
        (self.root/m.AMENDMENT).unlink()
        previous=self.root/'previous.json';m.write_once(previous,{'round':157})
        ref=m.reference(self.root,'previous.json');previous.write_text('{}')
        state,writes,module,plan=self.registration_fixture()
        state['active_execution_amendment']=ref
        with patch.object(m,'PREVIOUS_AMENDMENT',ref),self.assertRaisesRegex(ValueError,'digest changed'):
            self.registration_call(module,plan)
        self.assertFalse((self.root/m.AMENDMENT).exists());self.assertEqual(writes,[])


if __name__=='__main__':unittest.main()
