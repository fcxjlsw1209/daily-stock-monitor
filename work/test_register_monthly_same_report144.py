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
runner = types.ModuleType('monthly_same_report144_runner')
runner.NAMES = ('incumbent_local67', 'same_report_gross_growth', 'same_report_profitability')
runner.sha, runner.require, runner.dump = sha, require, dump
runner.read = lambda p: json.loads(Path(p).read_text())
spec = importlib.util.spec_from_file_location('registration_under_test', HERE/'register_monthly_same_report144.py')
registration = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'monthly_same_report144_runner': runner}):
    spec.loader.exec_module(registration)


class RegistrationTransition(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-registration144-', dir='/private/tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_path = self.root/registration.STATE_RELATIVE
        self.plan_path = self.root/'outputs'/registration.EXPERIMENT/'PLAN.json'
        self.state_path.parent.mkdir(parents=True)
        self.plan_path.parent.mkdir(parents=True)
        self.source_path = self.root/'source.txt'
        self.source_path.write_text('synthetic prerequisite, no market data\n')
        self.state = dict(completed=[dict(trials=6, directory='synthetic_%03d' % i) for i in range(142)] +
                         [dict(trials=77, directory='monthly_paired_target143')], registered_queue=[],
                         current='synthetic_143', current_status='completed_independently_verified',
                         unchanged_metadata={'nested': ['original', 17]}, updated_utc='before')
        self.state_path.write_text(json.dumps(self.state, indent=2)+'\n')
        self.original_state = self.state_path.read_bytes()
        self.plan = dict(round=144, status='REGISTERED', registered_utc='2040-01-02T03:04:05+00:00',
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
        expected.update(registered_queue=[dict(round=144, directory=registration.EXPERIMENT,
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
        self.state['registered_queue'] = [dict(round=144)]
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
        for history in (None, {}, 'invalid', [None]*143):
            with self.subTest(history_type=type(history).__name__):
                self.assert_rejects_history_without_writing(history)

    def test_invalid_trial_types_and_negative_counts_cannot_offset_total(self):
        self.stage()
        for value in (True, 6.0, -1, None, '6'):
            with self.subTest(value=value):
                history = deepcopy(self.state['completed'])
                history[0]['trials'] = value
                if isinstance(value, (int, float)):
                    history[-1]['trials'] = 83 - value
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
                                       '--outcome-review', 'synthetic_out.json', '--admitted-dataset', 'synthetic_dataset.json', '--source-admission','synthetic_admission.json']), redirect_stdout(io.StringIO()) as out:
            registration.main()
        receipt = json.loads(out.getvalue())
        self.assertTrue(receipt['queue_committed'])
        self.assertEqual(receipt['registered_queue_contains_round'], 144)
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

    def test_unfrozen_proposal_stops_before_reading_dependencies(self):
        runner.PROPOSAL_SHA='UNREGISTERED_DRAFT'
        with self.assertRaisesRegex(ValueError,'Final prospective proposal'):
            registration.build(self.root,'pre','out','dataset','admission')
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(self.original_state,self.state_path.read_bytes())

    def test_prereview_failure_does_not_publish_plan_or_state(self):
        with patch.object(registration, 'ROOT', self.root), \
             patch.object(registration, 'build', side_effect=ValueError('synthetic prereview failure')), \
             patch.object(sys, 'argv', ['register', '--prepare-review', 'pre', '--outcome-review', 'out', '--admitted-dataset', 'dataset','--source-admission','admission']):
            with self.assertRaisesRegex(ValueError, 'prereview failure'):
                registration.main()
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(self.original_state, self.state_path.read_bytes())

    def test_cli_requires_both_prereviews_and_disallows_mixed_resume(self):
        for args in ([], ['--prepare-review', 'pre'], ['--prepare-review','pre','--outcome-review','out'], ['--resume-plan-sha', 'f'*64, '--outcome-review', 'out'], ['--resume-plan-sha','f'*64,'--admitted-dataset','dataset']):
            with patch.object(registration, 'ROOT', self.root), patch.object(sys, 'argv', ['register']+args), \
                 patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit) as cm:
                registration.main()
            self.assertEqual(cm.exception.code, 2)
        self.assertFalse(self.plan_path.exists())
        self.assertEqual(self.original_state, self.state_path.read_bytes())



class ProspectiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-register-evidence144-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suites = []
        for name, filename in registration.SUITE_SOURCES.items():
            source = self.root/filename;source.write_text('synthetic test source '+name)
            log = self.root/(name+'.log');log.write_text('Ran 3 tests in 0.01s\n\nOK\n')
            self.suites.append(dict(name=name,count=3,passed=True,
                source=dict(path=str(source),sha256=sha(source)),log=dict(path=str(log),sha256=sha(log))))
        self.tests = dict(passed=True,suites=self.suites,test_count=18)
        self.bound = {}

    def add(self, name, expected):
        path=Path(name)
        require(sha(path)==expected,'Changed input')
        self.bound[str(path)]=expected
        return path

    def verify(self):return registration.validate_tests(self.tests,self.add)

    def test_all_suites_bind_both_source_and_log(self):
        self.assertEqual(self.verify(),dict.fromkeys(registration.SUITE_SOURCES,3))
        self.assertEqual(len(self.bound),12)

    def test_missing_completion_suite_rejected(self):
        self.suites.pop();self.tests['test_count']=15
        with self.assertRaisesRegex(ValueError,'All new'):self.verify()

    def test_duplicate_suite_rejected(self):
        self.suites.append(deepcopy(self.suites[0]));self.tests['test_count']=21
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
        self.suites[0]['count']=4;self.tests['test_count']=19
        with self.assertRaisesRegex(ValueError,'does not report success'):self.verify()
        self.suites[0]['count']=3;self.tests['test_count']=18
        ref=self.suites[0]['log'];Path(ref['path']).write_text('Ran 3 tests in 0.01s\nFAILED\n')
        ref['sha256']=sha(ref['path'])
        with self.assertRaisesRegex(ValueError,'does not report success'):self.verify()

    def test_boolean_and_wrong_total_counts_rejected(self):
        self.suites[0]['count']=True
        with self.assertRaisesRegex(ValueError,'Positive'):self.verify()
        self.suites[0]['count']=3;self.tests['test_count']=17
        with self.assertRaisesRegex(ValueError,'total count'):self.verify()

    def test_additional_passed_bound_suite_allowed(self):
        extra=deepcopy(self.suites[0]);extra['name']='contract'
        self.suites.append(extra);self.tests['test_count']=21
        self.assertEqual(self.verify()['contract'],3)



class BuildAdmissionTests(unittest.TestCase):
    """Build a synthetic dependency closure without loading market inputs."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic-profit144-build-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.bound_files={}
        def put(name,obj):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            if isinstance(obj,str):path.write_text(obj)
            else:dump(path,obj)
            self.bound_files[str(path)]=sha(path)
            return path
        self.put=put
        self.state=put(registration.STATE_RELATIVE,dict(
            completed=[dict(trials=6,directory='synthetic_%d'%i) for i in range(142)]+
                      [dict(trials=77,directory='monthly_paired_target143')],registered_queue=[]))
        self.old_source=put('sources/original.txt','original synthetic source')
        raw={str(put('sources/raw_%d.txt'%i,'synthetic raw '+str(i))):None for i in range(64)}
        raw={path:sha(path) for path in raw}
        inputs={name:put('sources/'+name+'.json',{'synthetic':name}) for name in
                ('meta','policy','original_orders','prices','protected_manifest')}
        self.antecedent=put('outputs/monthly_paired_target143/COMPLETION.json',{'synthetic':'completed140'})
        self.old_plan=put('outputs/monthly_paired_target143/PLAN.json',dict(
            hashes={str(self.old_source):sha(self.old_source)},raw_source_hashes=raw,
            inputs={key:dict(path=str(path),sha256=sha(path)) for key,path in inputs.items()}))
        self.proposal=put('work/monthly_same_report144_proposal.json',{'synthetic':'proposal144'})
        self.calendar=put('outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json',
                          dict(bindings={str(self.old_source):sha(self.old_source)}))
        self.sessions=put('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json',
                          dict(plan_sha256=sha(self.calendar),sessions=['2040-01-02']))
        self.novelty=put('work/monthly_same_report144_novelty.json',dict(
            proposal_sha256=sha(self.proposal),source_sha256={str(self.old_source):sha(self.old_source)}))
        suites=[]
        for name,filename in registration.SUITE_SOURCES.items():
            source=put('work/'+filename,'synthetic source '+name)
            log=put('logs/'+name+'.log','Ran 3 tests in 0.01s\n\nOK\n')
            suites.append(dict(name=name,count=3,passed=True,
                source=dict(path=str(source),sha256=sha(source)),log=dict(path=str(log),sha256=sha(log))))
        self.tests=put('outputs/monthly_same_report144/TESTS.json',dict(
            passed=True,suites=suites,test_count=18))
        self.implementation=put('work/monthly_same_report144_runner.py','synthetic implementation')
        put('work/complete_monthly_same_report144.py','synthetic closer')
        self.reviews={}
        for kind in ('prepare','outcome'):
            source=put('work/monthly_same_report144_'+kind+'_review.py','synthetic independent '+kind)
            self.reviews[kind]=put('reviews/'+kind+'.json',dict(
                passed=True,round=144,stage='preregistration_'+kind+'_review',
                proposal_sha256=sha(self.proposal),synthetic_only=True,
                real_144_factor_or_return_calculations=0,
                source_hashes={str(source):sha(source)}))
        self.evidence=put('sources/issuer_original.txt','synthetic financial source')
        folder='outputs/monthly_same_report144_admission/'
        self.dataset=put(folder+'DATA.json',dict(schema_version=1,events=[]))
        self.source_policy=put(folder+'POLICY.json',{'policy':'synthetic same-report'})
        self.source_code=put('work/synthetic_source_adapter.py','synthetic source adapter')
        codes={str(self.source_code):sha(self.source_code)}
        sources={str(self.evidence):sha(self.evidence),str(self.sessions):sha(self.sessions)}
        self.source_tests=put(folder+'TESTS.json',dict(passed=True,synthetic_only=True,
            tests_run=3,actual_financial_calculations=0,errors=0,failures=0,
            log='Ran 3 tests in 0.01s\n\nOK\n',source_code_hashes=codes))
        self.source_plan=put(folder+'PLAN.json',dict(source_hashes=sources,source_code_hashes=codes))
        self.receipt=put(folder+'AUDIT.json',dict(stage='conditional_source_admission',passed=True,
            data_sha256=sha(self.dataset),policy_sha256=sha(self.source_policy),plan_sha256=sha(self.source_plan),
            source_hashes=sources,source_code_hashes=codes))
        self.source_manifest=put(folder+'MANIFEST.json',dict(plan_sha256=sha(self.source_plan),
            output_hashes={name:sha(self.root/folder/name) for name in ('DATA.json','AUDIT.json')}))
        proposal=runner.read(self.proposal)
        proposal['source_contract']=dict(dataset_sha256=sha(self.dataset),audit_sha256=sha(self.receipt))
        self.proposal.write_text(json.dumps(proposal))
        for review in self.reviews.values():
            body=runner.read(review);body['proposal_sha256']=sha(self.proposal);review.write_text(json.dumps(body))
        self.original_state=self.state.read_bytes()
        self.checked=[]
        def validate(receipt,**bindings):
            self.checked.append((deepcopy(receipt),dict(bindings)))
            require(receipt.get('passed') is True,'synthetic admission rejected')
            for key,value in bindings.items():
                target={'dataset_sha':'data_sha256','meta_sha':'meta_sha256',
                        'sessions_sha':'sessions_sha256','proposal_sha':'proposal_sha256'}[key]
                require(receipt[target]==value,'synthetic admission binding')
        for patcher in (
            patch.multiple(registration,PROPOSAL_SHA=sha(self.proposal),BASE_PLAN_SHA=sha(self.old_plan),
                           LATEST_COMPLETION_SHA=sha(self.antecedent),CALENDAR_SHA=sha(self.calendar)),
            patch.multiple(runner,create=True,PROPOSAL_SHA=sha(self.proposal),
                INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest',
                        'admitted_dataset','source_admission'),TOLERANCES={'synthetic':0},
                source_files=lambda root:{str(self.implementation):sha(self.implementation)},
                validate_source_admission=validate,
                np=types.SimpleNamespace(__version__='synthetic'),pd=types.SimpleNamespace(__version__='synthetic'))):
            patcher.start();self.addCleanup(patcher.stop)

    def build(self):
        return registration.build(self.root,self.reviews['prepare'],self.reviews['outcome'],
                                  self.dataset,self.receipt)

    def test_build_binds_dataset_admission_sources_and_all_reviews(self):
        plan=self.build()
        self.assertEqual(plan['round'],144)
        self.assertEqual(set(plan['fundamental_source_hashes']),{str(x) for x in (self.evidence,self.source_code,self.source_policy,self.source_plan,self.source_tests,self.source_manifest,self.dataset,self.receipt,self.sessions)})
        self.assertEqual(plan['inputs']['admitted_dataset']['sha256'],sha(self.dataset))
        self.assertEqual(plan['inputs']['source_admission']['sha256'],sha(self.receipt))
        for path in (self.dataset,self.receipt,self.evidence):self.assertEqual(plan['hashes'][str(path)],sha(path))
        self.assertEqual(len(self.checked),1)
        self.assertEqual(plan['registration_precondition']['sha256'],sha(self.state))
        self.assertFalse((self.root/'outputs/monthly_same_report144/PLAN.json').exists())
        self.assertEqual(self.original_state,self.state.read_bytes())

    def test_rejected_admission_never_builds_a_plan(self):
        obj=runner.read(self.receipt);obj['passed']=False;self.receipt.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
        self.assertEqual(self.original_state,self.state.read_bytes())

    def test_changed_admitted_dataset_does_not_rebind_itself(self):
        self.dataset.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()

    def test_changed_evidence_does_not_rebind_itself(self):
        self.evidence.write_text('changed issuer statement')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()

    def test_missing_admission_arguments_stop_before_source_reads(self):
        for dataset,admission in ((None,self.receipt),(self.dataset,None)):
            with self.subTest(dataset=dataset,admission=admission),self.assertRaisesRegex(ValueError,'Explicit admitted'):
                registration.build(self.root,self.reviews['prepare'],self.reviews['outcome'],dataset,admission)

    def test_changed_source_plan_rejected(self):
        obj=runner.read(self.source_plan);obj['extra']='tampered';self.source_plan.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'policy/plan binding'):self.build()

    def test_changed_source_policy_rejected(self):
        self.source_policy.write_text('changed')
        with self.assertRaisesRegex(ValueError,'policy/plan binding'):self.build()

    def test_source_code_mismatch_rejected(self):
        obj=runner.read(self.source_tests);obj['source_code_hashes']={};self.source_tests.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'closure mismatch'):self.build()

    def test_manifest_omission_rejected(self):
        obj=runner.read(self.source_manifest);obj['output_hashes'].pop('DATA.json');self.source_manifest.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'manifest output binding'):self.build()

    def test_source_test_error_rejected(self):
        obj=runner.read(self.source_tests);obj['errors']=1;self.source_tests.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'Source synthetic tests'):self.build()

    def test_source_test_false_count_rejected(self):
        obj=runner.read(self.source_tests);obj['tests_run']=True;self.source_tests.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'Source synthetic tests'):self.build()

    def test_source_adapter_tampering_rejected(self):
        self.source_code.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()

    def test_wrong_source_directory_rejected(self):
        with self.assertRaisesRegex(ValueError,'Exact admitted source paths'):
            registration.admitted_source_closure(self.root,self.receipt,self.root/'other.json',lambda x,*a:x)

    def test_consistent_source_replacement_still_violates_proposal(self):
        self.dataset.write_text('{"schema_version":1,"events":["replacement"]}')
        body=runner.read(self.receipt);body['data_sha256']=sha(self.dataset);self.receipt.write_text(json.dumps(body))
        body=runner.read(self.source_manifest);body['output_hashes']={name:sha(path) for name,path in [('DATA.json',self.dataset),('AUDIT.json',self.receipt)]};self.source_manifest.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
        self.assertEqual(self.original_state,self.state.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
