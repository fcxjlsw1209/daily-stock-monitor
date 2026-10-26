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
from test_complete_monthly_same_report145 import synthetic_sources, factor_reuse_fixture

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
runner = types.ModuleType('monthly_same_report145_runner')
runner.NAMES = ('incumbent_local67', 'expanded_same_report_gross_growth', 'expanded_same_report_profitability')
runner.sha, runner.require, runner.dump = sha, require, dump
runner.read = lambda p: json.loads(Path(p).read_text())
spec = importlib.util.spec_from_file_location('registration_under_test', HERE/'register_monthly_same_report145.py')
registration = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'monthly_same_report145_runner': runner}):
    spec.loader.exec_module(registration)


class RegistrationTransition(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic-registration145-', dir='/private/tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_path = self.root/registration.STATE_RELATIVE
        self.plan_path = self.root/'outputs'/registration.EXPERIMENT/'PLAN.json'
        self.state_path.parent.mkdir(parents=True)
        self.plan_path.parent.mkdir(parents=True)
        self.source_path = self.root/'source.txt'
        self.source_path.write_text('synthetic prerequisite, no market data\n')
        self.state = dict(completed=[dict(trials=6, directory='synthetic_%03d' % i) for i in range(143)] +
                         [dict(trials=74, directory='monthly_same_report144')], registered_queue=[],
                         current='synthetic_143', current_status='completed_independently_verified',
                         unchanged_metadata={'nested': ['original', 17]}, updated_utc='before')
        self.state_path.write_text(json.dumps(self.state, indent=2)+'\n')
        self.original_state = self.state_path.read_bytes()
        self.plan = dict(round=145, status='REGISTERED', registered_utc='2040-01-02T03:04:05+00:00',
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
        expected.update(registered_queue=[dict(round=145, directory=registration.EXPERIMENT,
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
        self.state['registered_queue'] = [dict(round=145)]
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
        self.assertEqual(receipt['registered_queue_contains_round'], 145)
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
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-evidence145-',dir='/private/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bound={}
        def put(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(value if isinstance(value,str) else json.dumps(value));return path
        self.suites=[factor_reuse_fixture(self.root,put)]
        for name,filename in registration.SUITE_SOURCES.items():
            if name=='factor':continue
            source=put(self.root/filename,'synthetic suite');log=put(self.root/(name+'.log'),'Ran 3 tests in 0.01s\n\nOK\n')
            self.suites.append(dict(name=name,count=3,passed=True,source=dict(path=str(source),sha256=sha(source)),log=dict(path=str(log),sha256=sha(log))))
        self.tests=dict(passed=True,suites=self.suites,test_count=48)
    def add(self,name,expected):
        path=Path(name);require(sha(path)==expected,'Changed input');self.bound[str(path)]=expected;return path
    def verify(self):return registration.validate_tests(self.tests,self.add,self.root)
    def test_all_suites_bound_with_honest_original_factor_receipt(self):
        self.assertEqual(self.verify()['factor'],33);self.assertEqual(len(self.bound),13)
    def test_factor_cannot_be_claimed_as_new_run(self):
        self.suites[0].pop('reused_from_round')
        with self.assertRaisesRegex(ValueError,'explicitly reused'):self.verify()
    def test_original_factor_receipt_hash_checked(self):
        self.suites[0]['original_receipt']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'receipt binding'):self.verify()
    def test_original_factor_count_cannot_be_relabelled(self):
        self.suites[0]['count']=32
        with self.assertRaises(ValueError):self.verify()
    def test_missing_suite_rejected(self):
        self.suites.pop();self.tests['test_count']-=3
        with self.assertRaisesRegex(ValueError,'All new'):self.verify()
    def test_duplicate_suite_rejected(self):
        self.suites.append(deepcopy(self.suites[1]));self.tests['test_count']+=3
        with self.assertRaisesRegex(ValueError,'Unique'):self.verify()
    def test_bad_log_rejected(self):
        ref=self.suites[1]['log'];Path(ref['path']).write_text('FAILED');ref['sha256']=sha(ref['path'])
        with self.assertRaisesRegex(ValueError,'does not report'):self.verify()
    def test_total_count_checked(self):
        self.tests['test_count']=49
        with self.assertRaisesRegex(ValueError,'total count'):self.verify()


class BuildAdmissionTests(unittest.TestCase):
    """Build a synthetic dependency closure without loading market inputs."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic-profit145-build-',dir='/private/tmp')
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
            completed=[dict(trials=6,directory='synthetic_%d'%i) for i in range(143)]+
                      [dict(trials=74,directory='monthly_same_report144')],registered_queue=[]))
        self.old_source=put('sources/original.txt','original synthetic source')
        raw={str(put('sources/raw_%d.txt'%i,'synthetic raw '+str(i))):None for i in range(64)}
        raw={path:sha(path) for path in raw}
        inputs={name:put('sources/'+name+'.json',{'synthetic':name}) for name in
                ('meta','policy','original_orders','prices','protected_manifest')}
        self.antecedent=put('outputs/monthly_same_report144/COMPLETION.json',{'synthetic':'completed140'})
        self.old_plan=put('outputs/monthly_same_report144/PLAN.json',dict(
            round=144,status='REGISTERED',hashes={str(self.old_source):sha(self.old_source)},raw_source_hashes=raw,
            inputs={key:dict(path=str(path),sha256=sha(path)) for key,path in inputs.items()}))
        self.proposal=put('work/monthly_same_report145_proposal.json',{'synthetic':'proposal144'})
        self.calendar=put('outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json',
                          dict(bindings={str(self.old_source):sha(self.old_source)}))
        self.sessions=put('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json',
                          dict(plan_sha256=sha(self.calendar),sessions=['2040-01-02']))
        self.novelty=put('work/monthly_same_report145_novelty.json',dict(
            proposal_sha256=sha(self.proposal),source_sha256={str(self.old_source):sha(self.old_source)}))
        suites=[factor_reuse_fixture(self.root,put)]
        for name,filename in registration.SUITE_SOURCES.items():
            if name=='factor':continue
            source=put('work/'+filename,'synthetic source '+name)
            log=put('logs/'+name+'.log','Ran 3 tests in 0.01s\n\nOK\n')
            suites.append(dict(name=name,count=3,passed=True,
                source=dict(path=str(source),sha256=sha(source)),log=dict(path=str(log),sha256=sha(log))))
        self.tests=put('outputs/monthly_same_report145/TESTS.json',dict(
            passed=True,suites=suites,test_count=48))
        self.implementation=put('work/monthly_same_report145_runner.py','synthetic implementation')
        put('work/complete_monthly_same_report145.py','synthetic closer')
        self.reviews={}
        for kind in ('prepare','outcome'):
            source=put('work/monthly_same_report145_'+kind+'_review.py','synthetic independent '+kind)
            self.reviews[kind]=put('reviews/'+kind+'.json',dict(
                passed=True,round=145,stage='preregistration_'+kind+'_review',
                proposal_sha256=sha(self.proposal),synthetic_only=True,
                real_145_factor_or_return_calculations=0,
                source_hashes={str(source):sha(source)}))
        self.folder,self.data,self.source_audit,self.source_hashes=synthetic_sources(self.root,put,self.sessions)
        self.dataset=self.folder/'DATA.json';self.receipt=self.folder/'AUDIT.json'
        self.reference_orders=put('outputs/monthly_same_report144/PRETRAIN_ORDERS.json',{'same_report_profitability':[]})
        self.reference_training=put('outputs/monthly_same_report144/TRAIN_OPEN_TRIALS.json',[])
        self.references={'PLAN.json':sha(self.old_plan),'PRETRAIN_ORDERS.json':sha(self.reference_orders),'TRAIN_OPEN_TRIALS.json':sha(self.reference_training)}
        proposal=runner.read(self.proposal)
        proposal['sources']=dict(directory='outputs/monthly_same_report145_union',dataset_sha256=sha(self.dataset),audit_sha256=sha(self.receipt),hashes={n:sha(self.folder/n) for n in ('DATA.json','AUDIT.json','PLAN.json','MANIFEST.json')})
        proposal['reference144']={'hashes':self.references}
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
                           LATEST_COMPLETION_SHA=sha(self.antecedent),CALENDAR_SHA=sha(self.calendar),
                           UNION_REVIEW_HASHES={n:sha(self.root/n) for n in registration.closure.UNION_REVIEW_FILES}),
            patch.multiple(runner,create=True,PROPOSAL_SHA=sha(self.proposal),
                INPUTS=('proposal','meta','policy','original_orders','prices','sessions','protected_manifest',
                        'admitted_dataset','source_admission','reference144_plan','reference144_orders','reference144_training_open'),TOLERANCES={'synthetic':0},
                source_files=lambda root:{str(self.implementation):sha(self.implementation)},
                validate_source_admission=validate,
                np=types.SimpleNamespace(__version__='synthetic'),pd=types.SimpleNamespace(__version__='synthetic'))):
            patcher.start();self.addCleanup(patcher.stop)

    def build(self):
        return registration.build(self.root,self.reviews['prepare'],self.reviews['outcome'],
                                  self.dataset,self.receipt)

    def test_build_binds_complete_union_and_reference_closures(self):
        plan=self.build();self.assertEqual(plan['round'],145)
        self.assertEqual(plan['fundamental_source_hashes'],self.source_hashes)
        self.assertEqual(plan['reference144_hashes'],self.references)
        for path in (self.dataset,self.receipt,self.reference_orders,self.reference_training):
            self.assertEqual(plan['hashes'][str(path)],sha(path))
        self.assertEqual(self.original_state,self.state.read_bytes())
        self.assertFalse((self.root/'outputs/monthly_same_report145/PLAN.json').exists())
    def test_changed_union_dataset_rejected(self):
        self.dataset.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
    def test_changed_union_source_rejected(self):
        original=self.root/'aapl_amd_original.html';original.write_text('mutated')
        with self.assertRaises(ValueError):self.build()
    def test_changed_reference_orders_rejected(self):
        self.reference_orders.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
    def test_changed_reference_training_rejected(self):
        self.reference_training.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
    def test_missing_explicit_source_arguments_rejected(self):
        with self.assertRaisesRegex(ValueError,'Explicit admitted'):
            registration.build(self.root,self.reviews['prepare'],self.reviews['outcome'])
    def test_wrong_source_directory_rejected(self):
        with self.assertRaisesRegex(ValueError,'Exact source admission paths'):
            registration.admitted_source_closure(self.root,self.receipt,self.root/'other.json',lambda x,*a:x)
    def test_final_proposal_cannot_replace_fixed_reference_hashes(self):
        obj=runner.read(self.proposal);obj['reference144']['hashes']['PRETRAIN_ORDERS.json']='0'*64
        self.proposal.write_text(json.dumps(obj))
        with patch.object(registration,'PROPOSAL_SHA',sha(self.proposal)),patch.object(runner,'PROPOSAL_SHA',sha(self.proposal)):
            with self.assertRaisesRegex(ValueError,'Changed input'):self.build()
    def test_final_proposal_must_bind_union_manifest(self):
        obj=runner.read(self.proposal);obj['sources']['hashes']['MANIFEST.json']='0'*64
        self.proposal.write_text(json.dumps(obj))
        with patch.object(registration,'PROPOSAL_SHA',sha(self.proposal)),patch.object(runner,'PROPOSAL_SHA',sha(self.proposal)):
            with self.assertRaisesRegex(ValueError,'complete union binding'):self.build()


if __name__=='__main__':unittest.main(verbosity=2)
