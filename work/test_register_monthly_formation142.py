"""Synthetic registration state transitions; never imports any research producer."""
from pathlib import Path
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def dump(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

# A stub prevents even import-time loading of the real runner/factor/data stack.
runner = types.ModuleType('monthly_formation142_runner')
runner.NAMES = ('incumbent_local67', 'formation_common_delete_mean', 'formation_common_delete_worst')
runner.sha, runner.require, runner.dump = sha, require, dump
runner.read = lambda p: json.loads(Path(p).read_text())
spec = importlib.util.spec_from_file_location('registration_under_test', HERE/'register_monthly_formation142.py')
registration = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'monthly_formation142_runner': runner}):
    spec.loader.exec_module(registration)


class RegistrationTransition(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-registration142-', dir='/private/tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_path = self.root/registration.STATE_RELATIVE
        self.plan_path = self.root/'outputs'/registration.EXPERIMENT/'PLAN.json'
        self.state_path.parent.mkdir(parents=True)
        self.plan_path.parent.mkdir(parents=True)
        self.source_path = self.root/'source.txt'
        self.source_path.write_text('synthetic prerequisite, no market data\n')
        self.state = dict(completed=[dict(trials=6, directory='synthetic_%03d' % i) for i in range(140)] +
                         [dict(trials=83, directory='monthly_profitability141')], registered_queue=[],
                         current='synthetic_141', current_status='completed_independently_verified',
                         unchanged_metadata={'nested': ['original', 17]}, updated_utc='before')
        self.state_path.write_text(json.dumps(self.state, indent=2)+'\n')
        self.original_state = self.state_path.read_bytes()
        self.plan = dict(round=142, status='REGISTERED', registered_utc='2040-01-02T03:04:05+00:00',
                         arms=list(runner.NAMES), primary_challenger=runner.NAMES[2],
                         output_directory=str(self.plan_path.parent),
                         registration_precondition=dict(path=registration.STATE_RELATIVE,
                                                        sha256=sha(self.state_path)),
                         hashes={str(self.source_path): sha(self.source_path)})

    def stage(self, **changes):
        self.plan.update(changes)
        dump(self.plan_path, self.plan)
        self.original_plan = self.plan_path.read_bytes()
        return sha(self.plan_path)

    def enqueue(self, digest=None):
        return registration.enqueue_registered_plan(self.root, self.plan_path, digest or sha(self.plan_path))

    def test_success_preserves_completed_history_and_metadata(self):
        digest = self.stage()
        self.assertTrue(self.enqueue(digest))
        actual = runner.read(self.state_path)
        expected = deepcopy(self.state)
        expected.update(registered_queue=[dict(round=142, directory=registration.EXPERIMENT,
                        plan=dict(path=str(self.plan_path.relative_to(self.root)), sha256=digest),
                        registered_utc=self.plan['registered_utc'])],
                        current=registration.EXPERIMENT, current_status='registered_before_actual_preparation',
                        updated_utc=self.plan['registered_utc'])
        self.assertEqual(actual, expected)
        self.assertEqual(self.plan_path.read_bytes(), self.original_plan)
        self.assertNotIn(str(self.state_path), self.plan['hashes'])

    def test_idempotent_retry_does_not_write_state_or_plan(self):
        self.stage()
        self.enqueue()
        before = self.state_path.read_bytes()
        with patch.object(registration.os, 'replace', side_effect=AssertionError('must not write')):
            self.assertFalse(self.enqueue())
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual(self.original_plan, self.plan_path.read_bytes())

    def test_atomic_replace_failure_preserves_original_then_can_resume(self):
        self.stage()
        with patch.object(registration.os, 'replace', side_effect=OSError('synthetic failure')):
            with self.assertRaisesRegex(OSError, 'synthetic failure'):
                self.enqueue()
        self.assertEqual(self.original_state, self.state_path.read_bytes())
        self.assertEqual(self.original_plan, self.plan_path.read_bytes())
        self.assertEqual(list(self.state_path.parent.glob('.CONTINUATION_STATUS.*.tmp')), [])
        self.assertTrue(self.enqueue())

    def test_state_changed_after_plan_is_not_overwritten(self):
        self.stage()
        self.state_path.write_bytes(self.original_state+b' ')
        changed = self.state_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'changed since PLAN'):
            self.enqueue()
        self.assertEqual(changed, self.state_path.read_bytes())

    def test_conflicting_queue_is_not_overwritten(self):
        self.stage()
        self.state['registered_queue'] = [dict(round=142)]
        self.state_path.write_text(json.dumps(self.state))
        before = self.state_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Conflicting registered queue'):
            self.enqueue()
        self.assertEqual(before, self.state_path.read_bytes())

    def test_wrong_predecessor_count_or_trial_total_fails(self):
        self.stage()
        for field in ('count', 'trials'):
            bad = deepcopy(self.state)
            if field == 'count':
                bad['completed'].pop()
            else:
                bad['completed'][-1]['trials'] += 1
            self.state_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'predecessor'):
                self.enqueue()

    def assert_rejects_history_without_writing(self, history):
        bad = deepcopy(self.state)
        bad['completed'] = history
        self.state_path.write_text(json.dumps(bad))
        before = self.state_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            self.enqueue()
        self.assertEqual(before, self.state_path.read_bytes())
        self.assertEqual(self.original_plan, self.plan_path.read_bytes())

    def test_invalid_history_shape_is_rejected_before_commit(self):
        self.stage()
        for history in (None, {}, 'invalid', [None]*141):
            with self.subTest(history_type=type(history).__name__):
                self.assert_rejects_history_without_writing(history)

    def test_invalid_trial_types_and_negative_counts_cannot_offset_total(self):
        self.stage()
        for value in (True, 6.0, -1, None, '6'):
            with self.subTest(value=value):
                history = deepcopy(self.state['completed'])
                history[0]['trials'] = value
                if isinstance(value, (int, float)):
                    history[-1]['trials'] = 89 - value
                self.assert_rejects_history_without_writing(history)

    def test_wrong_predecessor_order_or_duplicate_experiment_is_rejected(self):
        self.stage()
        for directory, index in (('wrong_predecessor', -1), (registration.EXPERIMENT, 0)):
            with self.subTest(directory=directory):
                history = deepcopy(self.state['completed'])
                history[index]['directory'] = directory
                self.assert_rejects_history_without_writing(history)

    def test_wrong_plan_digest_fails(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, 'Changed registration PLAN'):
            self.enqueue('0'*64)
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_wrong_plan_path_fails(self):
        self.stage()
        other = self.root/'other.json'
        other.write_bytes(self.original_plan)
        with self.assertRaisesRegex(ValueError, 'PLAN path'):
            registration.enqueue_registered_plan(self.root, other, sha(other))

    def test_wrong_round_status_arm_and_output_directory_fail(self):
        cases = [dict(round=138), dict(status='DRAFT'), dict(arms=list(reversed(runner.NAMES))),
                 dict(primary_challenger=runner.NAMES[1]), dict(output_directory=str(self.root))]
        for change in cases:
            candidate = dict(self.plan, **change)
            self.plan_path.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError, 'Wrong registered experiment'):
                self.enqueue()
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_wrong_state_reference_fails(self):
        self.plan['registration_precondition']['path'] = 'somewhere_else.json'
        self.stage()
        with self.assertRaisesRegex(ValueError, 'Wrong continuation precondition'):
            self.enqueue()

    def test_missing_precondition_fails_closed(self):
        del self.plan['registration_precondition']
        self.stage()
        with self.assertRaises(KeyError):
            self.enqueue()
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_prerequisite_hash_change_fails(self):
        self.stage()
        self.source_path.write_text('changed prerequisite')
        with self.assertRaisesRegex(ValueError, 'Changed registered input'):
            self.enqueue()
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_empty_hash_map_fails(self):
        self.stage(hashes={})
        with self.assertRaisesRegex(ValueError, 'Missing registration input hashes'):
            self.enqueue()

    def test_relative_hash_paths_are_supported(self):
        self.stage(hashes={'source.txt': sha(self.source_path)})
        self.assertTrue(self.enqueue())

    def test_change_during_staging_detected_without_clobbering_other_writer(self):
        self.stage()
        def racing_sha(path):
            if Path(path) == self.state_path:
                self.state_path.write_text(json.dumps(dict(self.state, external_update=True)))
            return sha(path)
        with patch.object(runner, 'sha', side_effect=racing_sha):
            with self.assertRaisesRegex(ValueError, 'changed before queue commit'):
                self.enqueue()
        self.assertTrue(runner.read(self.state_path)['external_update'])
        self.assertEqual(list(self.state_path.parent.glob('.CONTINUATION_STATUS.*.tmp')), [])

    def test_plan_changed_during_staging_does_not_write_state(self):
        self.stage()
        def racing_sha(path):
            if Path(path) == self.state_path:
                self.plan_path.write_bytes(self.original_plan+b' ')
            return sha(path)
        with patch.object(runner, 'sha', side_effect=racing_sha):
            with self.assertRaisesRegex(ValueError, 'PLAN changed before queue commit'):
                self.enqueue()
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_main_new_registration_commits_queue(self):
        with patch.object(registration, 'ROOT', self.root), patch.object(registration, 'build', return_value=self.plan), \
             patch.object(sys, 'argv', ['register', '--prepare-review', 'synthetic_pre.json',
                                       '--outcome-review', 'synthetic_out.json']), redirect_stdout(io.StringIO()) as out:
            registration.main()
        receipt = json.loads(out.getvalue())
        self.assertTrue(receipt['queue_committed'])
        self.assertEqual(receipt['registered_queue_contains_round'], 142)
        self.assertEqual(runner.read(self.state_path)['registered_queue'][0]['plan']['sha256'], sha(self.plan_path))

    def test_main_resume_does_not_build_or_rewrite_plan(self):
        digest = self.stage()
        with patch.object(registration, 'ROOT', self.root), \
             patch.object(registration, 'build', side_effect=AssertionError('must not rebuild')), \
             patch.object(sys, 'argv', ['register', '--resume-plan-sha', digest]), redirect_stdout(io.StringIO()):
            registration.main()
        self.assertEqual(self.plan_path.read_bytes(), self.original_plan)

    def test_build_refuses_existing_plan_before_any_prerequisite_read(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            registration.build(self.root, 'does_not_exist', 'does_not_exist')
        self.assertEqual(self.original_plan, self.plan_path.read_bytes())

    def test_prereview_failure_does_not_publish_plan_or_state(self):
        with patch.object(registration, 'ROOT', self.root), \
             patch.object(registration, 'build', side_effect=ValueError('synthetic prereview failure')), \
             patch.object(sys, 'argv', ['register', '--prepare-review', 'pre', '--outcome-review', 'out']):
            with self.assertRaisesRegex(ValueError, 'prereview failure'):
                registration.main()
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_cli_requires_both_prereviews_and_disallows_mixed_resume(self):
        for args in ([], ['--prepare-review', 'pre'], ['--resume-plan-sha', 'f'*64, '--outcome-review', 'out']):
            with patch.object(registration, 'ROOT', self.root), patch.object(sys, 'argv', ['register']+args), \
                 patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit) as cm:
                registration.main()
            self.assertEqual(cm.exception.code, 2)
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(self.original_state, self.state_path.read_bytes())



class ProspectiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-register-evidence142-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suites = []
        for name, filename in registration.SUITE_SOURCES.items():
            source = self.root/filename;source.write_text('synthetic test source '+name)
            log = self.root/(name+'.log');log.write_text('Ran 3 tests in 0.01s\n\nOK\n')
            self.suites.append(dict(name=name,count=3,passed=True,
                source=dict(path=str(source),sha256=sha(source)),log=dict(path=str(log),sha256=sha(log))))
        self.tests = dict(passed=True,suites=self.suites,test_count=21)
        self.bound = {}

    def add(self, name, expected):
        path=Path(name)
        require(sha(path)==expected,'Changed input')
        self.bound[str(path)]=expected
        return path

    def verify(self):return registration.validate_tests(self.tests,self.add)

    def test_all_suites_bind_both_source_and_log(self):
        self.assertEqual(self.verify(),dict.fromkeys(registration.SUITE_SOURCES,3))
        self.assertEqual(len(self.bound),14)

    def test_missing_completion_suite_rejected(self):
        self.suites.pop();self.tests['test_count']=18
        with self.assertRaisesRegex(ValueError,'All new'):self.verify()

    def test_duplicate_suite_rejected(self):
        self.suites.append(deepcopy(self.suites[0]));self.tests['test_count']=24
        with self.assertRaisesRegex(ValueError,'Unique passed'):self.verify()

    def test_wrong_suite_source_identity_rejected(self):
        self.suites[0]['source']=self.suites[1]['source']
        with self.assertRaisesRegex(ValueError,'source identity'):self.verify()

    def test_source_or_log_mutation_rejected(self):
        for field in ('source','log'):
            ref=self.suites[0][field];path=Path(ref['path']);before=path.read_bytes()
            path.write_bytes(before+b'changed')
            with self.assertRaisesRegex(ValueError,'Changed input'):self.verify()
            path.write_bytes(before)

    def test_log_count_and_success_must_match_manifest(self):
        self.suites[0]['count']=4;self.tests['test_count']=22
        with self.assertRaisesRegex(ValueError,'does not report success'):self.verify()
        self.suites[0]['count']=3;self.tests['test_count']=21
        ref=self.suites[0]['log'];Path(ref['path']).write_text('Ran 3 tests in 0.01s\nFAILED\n')
        ref['sha256']=sha(ref['path'])
        with self.assertRaisesRegex(ValueError,'does not report success'):self.verify()

    def test_boolean_and_wrong_total_counts_rejected(self):
        self.suites[0]['count']=True
        with self.assertRaisesRegex(ValueError,'Positive'):self.verify()
        self.suites[0]['count']=3;self.tests['test_count']=20
        with self.assertRaisesRegex(ValueError,'total count'):self.verify()

    def test_missing_features_or_selector_suite_rejected(self):
        for absent in ('features', 'selector'):
            original = self.tests['suites']
            self.tests['suites'] = [x for x in original if x['name'] != absent]
            self.tests['test_count'] = sum(x['count'] for x in self.tests['suites'])
            with self.assertRaisesRegex(ValueError, 'All new'):
                self.verify()
            self.tests['suites'] = original

    def test_exact_seven_lifecycle_source_names(self):
        self.assertEqual(set(registration.SUITE_SOURCES),
            {'features','selector','runner','prepare_review','outcome_review','registration','completion'})
        self.assertEqual(registration.SUITE_SOURCES['features'], 'test_monthly_formation142_features.py')
        self.assertEqual(registration.SUITE_SOURCES['selector'], 'test_monthly_formation142_selector.py')

    def test_additional_passed_bound_suite_allowed(self):
        extra=deepcopy(self.suites[0]);extra['name']='contract'
        self.suites.append(extra);self.tests['test_count']=24
        self.assertEqual(self.verify()['contract'],3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
