"""Synthetic-only governance and execution-scope tests for the177 repair."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import monthly_breakpoint177_repaired_runner as r


def array_fixture():
    n, width = 122, 49
    raw = np.broadcast_to(np.arange(1., width+1), (n, width)).copy()
    values = dict(baseline=raw, candidate=raw.copy(), control=raw.copy(), available=np.ones((n, width), bool))
    ranks = np.broadcast_to(np.arange(1., width+1)/width-.5, (n, width)).copy()
    arrays = dict(core=.52*ranks, core_weight=np.full((n, 1), .52), state_calm10=np.zeros(n, bool))
    deltas = {arm: np.zeros((n, width)) for arm in r.original.NAMES[1:]}
    return values, arrays, deltas


class TemporaryRepair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.out = self.root/'outputs/monthly_breakpoint177'; self.out.mkdir(parents=True)
        (self.root/'work').mkdir(); self.patches = []
        def replace(obj, name, value):
            p=patch.object(obj, name, value); p.start(); self.patches.append(p)
        replace(r, 'ROOT', self.root); replace(r, 'OUT', self.out)
        for name, filename in [('AMENDMENT','AMENDMENT_01.json'), ('FAILURE','PREPARE_INTERFACE_FAILURE_01.json'),
                ('FAILURE_LOG','PREPARE_INTERFACE_FAILURE_01_EXCERPT.txt'), ('TESTS','REPAIR_TESTS_01.json'),
                ('PREREVIEW','REPAIR_PREREVIEW_01.json'), ('REPAIR_COMPLETION','REPAIR_COMPLETION_01.json')]:
            replace(r,name,self.out/filename)
        files=tuple(self.root/'work'/path.name for path in r.FILES)
        replace(r,'FILES',files)
        for path in files: path.write_text('# synthetic code\n')
        previous=self.root/'outputs/monthly_annual_lag158/AMENDMENT_01.json'
        previous.parent.mkdir(); previous.write_text('{"old":true}')
        replace(r,'PREVIOUS_AMENDMENT',r.reference(previous))
        central=self.root/'outputs/CONTINUATION_STATUS.json';replace(r.original,'CENTRAL',central)
        self.plan=dict(created_utc='2020-01-01T00:00:00+00:00', contract=r.original.contract(),
            runtime=r.original.ev.runtime(),input_sha256={str(p):r.sha(p) for p in files},
            prior_central_sha256=hashlib.sha256(b'state before').hexdigest())
        self.write(self.out/'PLAN.json',self.plan);(self.out/'STATE_BEFORE.json').write_bytes(b'state before')
        replace(r,'PLAN_SHA',r.sha(self.out/'PLAN.json'))
        replace(r,'CONTRACT',dict(r.CONTRACT,original_plan_sha256=r.PLAN_SHA))
        self.state=dict(completed=[dict(trials=6,directory='old') for _ in range(172)]+[
            dict(trials=2,directory='old'),dict(trials=3,directory=r.original.PRIOR.name)],
            best_training_selected_candidate=dict(directory='monthly_local_linear70'), current=self.out.name,
            current_status='registered_before_new_breakpoint_features',
            registered_queue=[dict(directory=self.out.name,plan_sha256=r.PLAN_SHA)],
            active_execution_amendment=r.PREVIOUS_AMENDMENT)
        self.write(central,self.state)
        def gate(digest):
            r.require(digest==r.PLAN_SHA and r.sha(self.out/'PLAN.json')==r.PLAN_SHA,'Original gate failed')
            return self.plan
        replace(r,'ORIGINAL_GATE',gate)
        replace(r.original,'gate',gate)
        replace(r.original,'verify',lambda pins: [r.require(r.sha(Path(path))==sha,'Original pin changed') for path,sha in pins.items()])
        r.FAILURE_LOG.write_text('Retained execution-context excerpt, not full log. monthly_breakpoint177_host.py prepare -> verify_deltas -> independent_deltas: ValueError: Core weight shape/finite')
        self.failure=dict(original_plan_sha256=r.PLAN_SHA,exit_code=1,strategy_returns_calculated=0,
            prepared_artifacts_published=0,feature_build_attempted=True,feature_independent_audit_passed=True,
            original_order_parity_passed=True,original_core_weight_shape=[122,1],excerpt=r.reference(r.FAILURE_LOG),
            artifact_kind='retained_execution_context_excerpt',source='tool_output_excerpt_not_full_log',
            full_raw_log_available=False,recorded_utc='2020-01-02T00:00:00+00:00')
        self.write(r.FAILURE,self.failure)
        self.tests=dict(passed=True,synthetic_only=True,original_plan_sha256=r.PLAN_SHA,created_utc='2020-01-03T00:00:00+00:00',
            runtime=r.original.ev.runtime(),implementation_hashes=r.implementation_hashes(),
            suites={p.name:dict(passed=True,count=1,source_sha256=r.sha(p),log='Ran 1 test in 0.01s\n\nOK\n') for p in (files[1],files[3])})
        self.write(r.TESTS,self.tests)
        self.review=dict(passed=True,synthetic_only=True,semantics_preserved=True,independent_review=True,
            original_frozen_files_unchanged=True,original_plan_sha256=r.PLAN_SHA,
            implementation_hashes=r.implementation_hashes(),actual_returns_calculated=0,
            failure_record=r.reference(r.FAILURE),tests=r.reference(r.TESTS),reviewed_utc='2020-01-04T00:00:00+00:00')
        self.write(r.PREREVIEW,self.review)

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value))

    def register(self):
        return r.register_repair()['amendment']['sha256']


class GovernanceTests(TemporaryRepair):
    def test_register_preserves_original_queue_counts_best_and_previous(self):
        before=copy.deepcopy(self.state); digest=self.register();after=r.read(r.original.CENTRAL)
        self.assertEqual(after['registered_queue'],before['registered_queue'])
        self.assertEqual(after['completed'],before['completed'])
        self.assertEqual(after['best_training_selected_candidate'],before['best_training_selected_candidate'])
        self.assertEqual(r.read(r.AMENDMENT)['previous_active_execution_amendment'],r.PREVIOUS_AMENDMENT)
        self.assertEqual(after['active_execution_amendment'],r.reference(r.AMENDMENT))
        self.assertEqual(r.verify_amendment(r.PLAN_SHA,digest)['additional_strategy_trials'],0)

    def test_register_is_idempotent_without_rewriting_amendment(self):
        digest=self.register();before=r.AMENDMENT.read_bytes();state=r.original.CENTRAL.read_bytes()
        self.assertEqual(self.register(),digest)
        self.assertEqual(r.AMENDMENT.read_bytes(),before);self.assertEqual(r.original.CENTRAL.read_bytes(),state)

    def test_unknown_previous_amendment_rejected(self):
        self.state['active_execution_amendment']={'path':'unknown','sha256':'0'*64}
        self.write(r.original.CENTRAL,self.state)
        with self.assertRaises(ValueError):self.register()
        self.assertFalse(r.AMENDMENT.exists())

    def test_unprepared_rejects_any_real_phase_output(self):
        for name in ('prepared','selected','results','COMPLETION.json'):
            path=self.out/name;path.mkdir()
            with self.assertRaises(ValueError):self.register()
            path.rmdir()

    def test_wrong_plan_amendment_and_active_binding_rejected(self):
        digest=self.register()
        for plan,amend in [('0'*64,digest),(r.PLAN_SHA,'0'*64)]:
            with self.assertRaises(ValueError):r.verify_amendment(plan,amend)
        state=r.read(r.original.CENTRAL);state['active_execution_amendment']=r.PREVIOUS_AMENDMENT
        self.write(r.original.CENTRAL,state)
        with self.assertRaises(ValueError):r.verify_amendment(r.PLAN_SHA,digest)

    def test_repair_source_change_rejected(self):
        digest=self.register();r.FILES[0].write_text('changed')
        with self.assertRaises(ValueError):r.verify_amendment(r.PLAN_SHA,digest)

    def test_closure_mode_still_checks_original_pins(self):
        digest=self.register();self.plan['input_sha256'][str(r.FILES[0])]='wrong'
        with patch.object(r,'read',side_effect=lambda path:self.plan if path==self.out/'PLAN.json' else json.loads(path.read_text())):
            with self.assertRaises(ValueError):r.verify_amendment(r.PLAN_SHA,digest,original_gate=False)

    def test_failure_explicitly_identifies_excerpt_and_zero_returns(self):
        r.validate_failure(self.failure)
        for key,value in [('artifact_kind','full_traceback'),('source','raw_log'),('full_raw_log_available',True),
                ('strategy_returns_calculated',False),('strategy_returns_calculated',1),('prepared_artifacts_published',1),
                ('feature_independent_audit_passed',False),('original_core_weight_shape',[122,49])]:
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_failure(dict(self.failure,**{key:value}))

    def test_missing_independent_review_or_suite_rejected(self):
        for key in ('independent_review','semantics_preserved','original_frozen_files_unchanged'):
            with self.assertRaises(ValueError):r.validate_prereview(dict(self.review,**{key:False}))
        bad=copy.deepcopy(self.tests);bad['suites'].pop(r.FILES[1].name)
        with self.assertRaises(ValueError):r.validate_tests(bad)
        bad=copy.deepcopy(self.tests);bad['suites'][r.FILES[1].name]['log']*=2
        with self.assertRaises(ValueError):r.validate_tests(bad)

    def test_bad_chronology_rejected(self):
        digest=self.register();doc=r.read(r.AMENDMENT);doc['registered_utc']='2019-01-01T00:00:00+00:00'
        self.write(r.AMENDMENT,doc)
        with self.assertRaises(ValueError):r.verify_amendment(r.PLAN_SHA,r.sha(r.AMENDMENT),active=False)

    def test_write_once_and_reference_path(self):
        before=r.FAILURE.read_bytes()
        with self.assertRaises(FileExistsError):r.write_once(r.FAILURE,{})
        self.assertEqual(r.FAILURE.read_bytes(),before)
        with self.assertRaises(ValueError):r.verify_reference(dict(path='../escape',sha256=r.sha(r.FAILURE)),r.FAILURE)

    def test_concurrent_register_can_recover_only_same_before_state(self):
        before=r.original.CENTRAL.read_bytes()
        with patch.object(r.original,'commit_state',side_effect=ValueError('Concurrent central update')):
            with self.assertRaises(ValueError):self.register()
        self.assertEqual(r.original.CENTRAL.read_bytes(),before)
        self.assertTrue(r.AMENDMENT.exists())
        self.register()


class ScopedExecutionTests(TemporaryRepair):
    def setUp(self):
        super().setUp();self.digest=self.register()

    def test_real_frozen_verifier_receives_normalized_copy_only(self):
        values,arrays,deltas=array_fixture();before={key:value.tobytes() for key,value in arrays.items()}
        frozen=r.independent.verify_deltas
        with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
            receipt=r.independent.verify_deltas(values,deltas,arrays,dose=.5)
            r.original.require_delta_audit(receipt,deltas)
            self.assertEqual(receipt['weight_shape_adapter']['original_shape'],[122,1])
        self.assertIs(r.independent.verify_deltas,frozen)
        for key,value in arrays.items():self.assertEqual(value.tobytes(),before[key])

    def test_scoped_patch_restores_on_failure_and_nested_rejection(self):
        frozen=r.independent.verify_deltas
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
                with self.assertRaises(ValueError):
                    with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):pass
                raise RuntimeError('synthetic failure')
        self.assertIs(r.independent.verify_deltas,frozen)
        self.assertIs(r.original.gate,r.ORIGINAL_GATE)
        self.assertIs(r.original.require_delta_audit,r.ORIGINAL_REQUIRE_DELTA)

    def test_unexpected_patch_is_detected_and_restored(self):
        frozen=r.independent.verify_deltas
        with self.assertRaises(ValueError):
            with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
                r.independent.verify_deltas=lambda *a,**kw:None
        self.assertIs(r.independent.verify_deltas,frozen)

    def test_later_phases_do_not_patch_independent_computation(self):
        for phase in ('select','diagnose','close'):
            with r.execution_context(r.PLAN_SHA,self.digest,phase):
                self.assertIs(r.independent.verify_deltas,r.ORIGINAL_VERIFY_DELTAS)
                with self.assertRaises(ValueError):r.original.require_delta_audit({}, {})

    def test_missing_or_changed_saved_provenance_rejected(self):
        values,arrays,deltas=array_fixture()
        with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
            receipt=r.independent.verify_deltas(values,deltas,arrays,dose=.5)
        r.validate_repaired_delta(receipt,self.digest)
        for key in ('execution_amendment','repair_source_hashes','repair_prereview','weight_shape_adapter'):
            bad=copy.deepcopy(receipt);bad.pop(key)
            with self.assertRaises(ValueError):r.validate_repaired_delta(bad,self.digest)
        bad=copy.deepcopy(receipt);bad['weight_shape_adapter']['exact_state_weights']=False
        with self.assertRaises(ValueError):r.validate_repaired_delta(bad,self.digest)

    def test_wrong_delta_still_rejected_by_original_verifier(self):
        values,arrays,deltas=array_fixture();deltas[r.original.NAMES[1]][1,0]=.01
        with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
            with self.assertRaises(ValueError):r.independent.verify_deltas(values,deltas,arrays,dose=.5)

    def test_bad_amendment_prevents_any_real_phase_call(self):
        for phase in ('prepare','select','diagnose','close'):
            with patch.object(r.original,phase) as action:
                with self.assertRaises(ValueError):r.run_phase(phase,r.PLAN_SHA,'0'*64)
                action.assert_not_called()

    def test_phase_runs_original_action_under_both_gates(self):
        for phase in ('prepare','select','diagnose'):
            def action(plan):
                self.assertEqual(plan,r.PLAN_SHA)
                r.original.gate(plan)
            with patch.object(r.original,phase,side_effect=action) as call:
                r.run_phase(phase,r.PLAN_SHA,self.digest)
                call.assert_called_once_with(r.PLAN_SHA)


class CompletionTests(TemporaryRepair):
    def setUp(self):
        super().setUp();self.digest=self.register()
        for name in ('prepared','selected','results'):(self.out/name).mkdir()
        values,arrays,deltas=array_fixture()
        # The shape proof is synthetic; no strategy data or accounting engine is run.
        with r.execution_context(r.PLAN_SHA,self.digest,'prepare'):
            proof=r.independent.verify_deltas(values,deltas,arrays,dose=.5)
        self.write(self.out/'prepared/INDEPENDENT_PREPARE.json',dict(delta=proof))
        self.write(self.out/'prepared/ORDERS.json',{})
        self.write(self.out/'selected/FROZEN.json',{})
        self.write(self.out/'results/INDEPENDENT_OUTCOME.json',{})
        for folder in ('prepared','results'):
            names=[p.name for p in (self.out/folder).iterdir()]
            self.write(self.out/folder/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=r.PLAN_SHA,
                frozen_sha256=r.sha(self.out/'selected/FROZEN.json'),
                artifacts_sha256={name:r.sha(self.out/folder/name) for name in names}))
        self.write(self.out/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256=r.PLAN_SHA,
            result_completion_sha256=r.sha(self.out/'results/COMPLETION.json'),protected_files=21444,observers_modified=False))
        state=r.read(r.original.CENTRAL)
        state['registered_queue']=[];state['current_status']='completed_independently_verified'
        state['completed'].append(dict(trials=3,completion=dict(sha256=r.sha(self.out/'COMPLETION.json'))))
        self.write(r.original.CENTRAL,state)

    def test_close_sidecar_is_bound_and_idempotent(self):
        with patch.object(r.original,'frozen_selection',return_value={}):
            result=r.complete_repair(r.PLAN_SHA,self.digest)
            before=r.REPAIR_COMPLETION.read_bytes()
            self.assertEqual(r.complete_repair(r.PLAN_SHA,self.digest),result)
            self.assertEqual(before,r.REPAIR_COMPLETION.read_bytes())
        self.assertEqual(r.read(r.original.CENTRAL)['latest_execution_repair_receipt'],r.reference(r.REPAIR_COMPLETION))
        self.assertEqual(r.read(r.REPAIR_COMPLETION)['additional_strategy_trials'],0)

    def test_bad_final_state_never_publishes_sidecar(self):
        state=r.read(r.original.CENTRAL);state['completed'].pop();self.write(r.original.CENTRAL,state)
        with patch.object(r.original,'frozen_selection',return_value={}):
            with self.assertRaises(ValueError):r.complete_repair(r.PLAN_SHA,self.digest)
        self.assertFalse(r.REPAIR_COMPLETION.exists())

    def test_missing_repair_provenance_rejected_on_close(self):
        self.write(self.out/'prepared/INDEPENDENT_PREPARE.json',dict(delta={}))
        with patch.object(r.original,'frozen_selection',return_value={}):
            with self.assertRaises(ValueError):r.complete_repair(r.PLAN_SHA,self.digest)
        self.assertFalse(r.REPAIR_COMPLETION.exists())

    def test_frozen_selection_binding_failure_stops_close(self):
        with patch.object(r.original,'frozen_selection',side_effect=ValueError('frozen binding')):
            with self.assertRaisesRegex(ValueError,'frozen binding'):r.complete_repair(r.PLAN_SHA,self.digest)
        self.assertFalse(r.REPAIR_COMPLETION.exists())


if __name__=='__main__':unittest.main()
