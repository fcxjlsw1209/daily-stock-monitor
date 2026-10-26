"""Only generated metadata/byte fixtures; never loads real research artifacts."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('closer144', Path(__file__).with_name('complete_monthly_same_report144.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
FIXTURE_SUITES = dict(factor=13, runner=31, prepare_review=51, outcome_review=62, registration=21, completion=37)


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-closer144-', dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'outputs'/c.EXPERIMENT
        self.out.mkdir(parents=True)
        self.state_path = self.root/c.STATE
        self.state_path.parent.mkdir(parents=True)
        self.write('work/complete_monthly_same_report144.py', Path(c.__file__).read_bytes())
        self.write('work/test_complete_monthly_same_report144.py', Path(__file__).read_bytes())
        for name in ('prepare', 'outcome'):
            self.write(name+'.py', 'synthetic reviewer source only')
        self.members = {}
        for i in range(3):
            path = self.write('protected/'+str(i), 'original protected bytes '+str(i))
            self.members[str(path)] = c.sha(path)
        self.manifest = self.write('manifest.json', self.members)
        self.proposal = self.write('proposal.json', dict(limitations=['Synthetic fixture limitations.']))
        for name in ('PROTECTED_COUNT', 'PROTECTED_SHA', 'PROPOSAL_SHA'):
            value = {'PROTECTED_COUNT':3, 'PROTECTED_SHA':c.sha(self.manifest), 'PROPOSAL_SHA':c.sha(self.proposal)}[name]
            p = patch.object(c, name, value);p.start();self.addCleanup(p.stop)
        self.raw = {}
        for i in range(64):
            path = self.write('raw/source%02d.txt'%i, 'synthetic raw '+str(i))
            self.raw[str(path)] = c.sha(path)
        self.orders = {name:[dict(date='synthetic-%03d'%i, symbols=['A'], scores=[.1]) for i in range(122)] for name in c.NAMES}
        original = self.write('original_orders.json', self.orders[c.NAMES[0]])
        self.inputs = {'proposal':self.proposal, 'protected_manifest':self.manifest, 'original_orders':original}
        for key in ('meta', 'policy', 'prices', 'sessions'):
            self.inputs[key] = self.write(key+'.json', {'synthetic':key})
        self.source_dir=self.root/'outputs/monthly_same_report144_admission'
        self.source_dir.mkdir(parents=True)
        self.source_code={}
        for name in ('same_report144_admission.py','test_same_report144_admission.py'):
            path=self.write('work/'+name,'synthetic source mapper '+name);self.source_code[str(path)]=c.sha(path)
        source=self.write('issuer_source.html','synthetic original issuer source')
        self.source_hashes={str(source):c.sha(source)}
        common=dict(policy_id='same_report144_source_v1',**{k:False for k in c.CERTIFICATIONS})
        self.source_policy=dict(common,method='same_current_release_annual_comparison',assumptions={'vintage':'Not certified.'})
        self.source_data=dict(common,method='same_current_release_annual_comparison',model_admitted=False,strategy_admitted=False,
            counts=deepcopy(c.SOURCE_COUNTS),ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0)
        self.source_tests=dict(passed=True,synthetic_only=True,failures=0,errors=0,actual_financial_calculations=0,
            tests_run=1,log='Ran 1 test in 0.01s\n\nOK\n',source_code_hashes=self.source_code)
        self.admission=dict(common,stage='conditional_source_admission',passed=True,checks={k:True for k in c.ADMISSION_CHECKS},
            counts=deepcopy(c.SOURCE_COUNTS),financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,
            limitations=self.source_policy['assumptions'],source_hashes=self.source_hashes,source_code_hashes=self.source_code)
        self.sync_admission()
        self.write(self.proposal,dict(interpretation_limits=['Synthetic fixture limitations.'],source_contract=dict(
            directory='outputs/monthly_same_report144_admission',method='same_current_release_annual_comparison',
            dataset_sha256=c.sha(self.inputs['admitted_dataset']),audit_sha256=c.sha(self.inputs['source_admission']))))
        c.PROPOSAL_SHA=c.sha(self.proposal)
        self.trials = [dict(name=name, training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                           early=dict(cagr=.2), recent=dict(cagr=.2)) for name in c.NAMES]
        self.open_trials=deepcopy(self.trials)
        self.open_trials[2]['training']['cagr']=.21
        self.results = {name:{mode:dict(metrics=dict(cagr=.3, sharpe=1., max_drawdown=-.2),
                                        slices=dict(later34=dict(cagr=.3))) for mode in c.MODES} for name in c.NAMES}
        self.features = dict(new_model_fits=0, latest_event_before_validity_filter=True,same_current_release_for_both_periods=True, sign_method='exact_decimal_cross_products_after_declared_scale_normalization',common_available_cells=0,
            original_auxiliary_states_and_core_coefficients_preserved=True, original_arrays_unchanged=True,
            returns_or_nav_computed=False, original_months=122, original_width=49, cells=[{'available':False} for _ in range(5978)])
        self.suites = []
        for name,count in FIXTURE_SUITES.items():
            path = self.write('logs/'+name+'.log', 'Ran %d tests in 0.01s\n\nOK\n'%count)
            source=self.write('suite_sources/'+c.SUITE_SOURCES[name], 'synthetic source '+name)
            self.suites.append(dict(name=name,count=count,passed=True,log=dict(path=str(path),sha256=c.sha(path)),
                                    source=dict(path=str(source),sha256=c.sha(source))))
        self.proof = dict(passed=True, checked=3, verified=3, changed=[], missing=[],
                          read_or_identity_errors=[], manifest_sha256=c.PROTECTED_SHA)
        self.prepare_overrides = {};self.outcome_overrides = {};self.drop_artifact = None
        self.setup_evidence()
        self.state = dict(completed=[dict(directory='old%03d'%i,trials=6) for i in range(142)]+[
            dict(directory='monthly_paired_target143',trials=77)],
            registered_queue=c.expected_queue(self.plan,self.plan_sha),current='registered144',
            best_training_selected_candidate=dict(cagr=.5,spec='original-best',nested={'preserved':True}),
            untouched={'array':[1,2,3],'text':'do not replace'},unexpected_metadata=42)
        self.write(self.state_path,self.state)
        self.before = self.state_path.read_bytes()

    def write(self, name, value):
        path = Path(name)
        if not path.is_absolute():path=self.root/path
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else c.encoded(value))
        return path

    def output(self, name, value):return self.write(self.out/name,value)

    def sync_admission(self):
        base=self.source_dir
        self.write(base/'POLICY.json',self.source_policy);self.write(base/'DATA.json',self.source_data)
        self.write(base/'TESTS.json',self.source_tests)
        source_plan=dict(stage='source_admission_plan',policy_id='same_report144_source_v1',method='same_current_release_annual_comparison',
            financial_computations_permitted=False,network_requests_permitted=False,source_hashes=self.source_hashes,
            source_code_hashes=self.source_code,outputs=['DATA.json','AUDIT.json','MANIFEST.json'])
        self.write(base/'PLAN.json',source_plan)
        self.admission.update(data_sha256=c.sha(base/'DATA.json'),plan_sha256=c.sha(base/'PLAN.json'),policy_sha256=c.sha(base/'POLICY.json'))
        self.write(base/'AUDIT.json',self.admission)
        self.write(base/'MANIFEST.json',dict(stage='conditional_source_admission_complete',plan_sha256=c.sha(base/'PLAN.json'),
            output_hashes={name:c.sha(base/name) for name in ('DATA.json','AUDIT.json')}))
        self.inputs['source_admission']=base/'AUDIT.json';self.inputs['admitted_dataset']=base/'DATA.json'
        self.fundamental=dict(self.source_hashes,**self.source_code)
        self.fundamental.update({str(base/name):c.sha(base/name) for name in c.ADMISSION_FILES})

    def setup_evidence(self):

        sources = {str(self.root/(name+'.py')):c.sha(self.root/(name+'.py')) for name in ('prepare','outcome')}
        pins=dict(self.raw,**sources);pins.update(self.fundamental)
        for path in self.inputs.values():pins[str(path)]=c.sha(path)
        for filename in ('complete_monthly_same_report144.py','test_complete_monthly_same_report144.py'):
            path=self.root/'work'/filename;pins[str(path)]=c.sha(path)
        for suite in self.suites:
            for field in ('source','log'):
                ref=suite[field];pins[str(Path(ref['path']).resolve())]=ref['sha256']
        tests=self.output('TESTS.json',dict(passed=True,suites=self.suites,test_count=sum(x['count'] for x in self.suites)))
        pins[str(tests)]=c.sha(tests)
        self.plan=dict(round=144,status='REGISTERED',registered_utc='2020-01-01T00:00:00+00:00',
            arms=list(c.NAMES),primary_challenger=c.NAMES[2],output_directory=str(self.out),hashes=pins,
            raw_source_hashes=self.raw, fundamental_source_hashes=self.fundamental,
            independent_prepare_source_hashes={str(self.root/'prepare.py'):sources[str(self.root/'prepare.py')]},
            independent_outcome_source_hashes={str(self.root/'outcome.py'):sources[str(self.root/'outcome.py')]},
            actual_model_fits_before_registration=0,actual_feature_calculations_before_registration=0,
            actual_returns_before_registration=0,
            inputs={key:dict(path=str(path),sha256=c.sha(path)) for key,path in self.inputs.items()})
        self.plan_sha=c.sha(self.output('PLAN.json',self.plan))
        self.output('DATA.npz',b'opaque synthetic bytes; not a market archive')
        self.output('SCORES.npz',b'opaque synthetic score bytes')
        self.output('META.json',{})
        self.output('PRICE_PARITY.json',{})
        self.features.update(source_admission_sha256=c.sha(self.inputs['source_admission']),admitted_dataset_sha256=c.sha(self.inputs['admitted_dataset']))
        self.output('FEATURE_AUDIT.json',self.features)
        self.output('PRETRAIN_ORDERS.json',self.orders)
        cached={name:c.sha(self.out/name) for name in c.CACHE}
        self.output('CACHE.json',cached)
        prep=dict(passed=True,stage='prepare',producer_modules_imported=False,
            returns_or_nav_computed=False,
            orders_checked=366,source_cells_checked=5978,stock_cells_checked=3904,monthly_label_models_fit=0,new_model_fits=0,
            plan_sha256=self.plan_sha,cache_sha256=c.sha(self.out/'CACHE.json'),
            orders_sha256=c.sha(self.out/'PRETRAIN_ORDERS.json'),
            source_hashes=self.plan['independent_prepare_source_hashes'],
            artifact_hashes=dict(pins,**{str(self.out/name):c.sha(self.out/name) for name in (*c.CACHE,'PLAN.json','CACHE.json')}),
            **{key:True for key in c.PREPARE_CHECKS})
        prep.update(self.prepare_overrides)
        self.output('INDEPENDENT_PREPARE.json',prep)
        self.output('TRAIN_TRIALS.json',self.trials)
        self.output('WALKFORWARD.json',{})
        self.output('TRAIN_OPEN_TRIALS.json',self.open_trials)
        chosen=c.selection(self.trials,self.open_trials)
        self.output('FROZEN.json',dict(selection=chosen,pristine_oos=False,all_diagnostic_orders_frozen=True,
                                      hashes={name:c.sha(self.out/name) for name in c.FROZEN}))
        self.output('DIAGNOSE_STARTED.json',dict(plan_sha256=self.plan_sha,frozen_sha256=c.sha(self.out/'FROZEN.json'),
                                                started_utc='2020-01-01T01:00:00+00:00'))
        self.output('RESULTS.json',self.results)
        for name in ('MONTH_COMPARISON.json','ATTRIBUTION.json','VALIDATION.json'):self.output(name,{})
        for name in c.NAMES:
            for mode in c.MODES:
                for kind in ('DAILY','TRADES','HOLDINGS'):self.output(name+'_'+mode+'_'+kind+'.json',[])
        files=(*c.CACHE,*c.FROZEN,'CACHE.json','FROZEN.json','DIAGNOSE_STARTED.json','RESULTS.json',
               'MONTH_COMPARISON.json','ATTRIBUTION.json','VALIDATION.json')
        files+=tuple(name+'_'+mode+'_'+kind+'.json' for name in c.NAMES for mode in c.MODES for kind in ('DAILY','TRADES','HOLDINGS'))
        artifacts=dict(pins,**{str(self.out/name):c.sha(self.out/name) for name in files})
        if self.drop_artifact:del artifacts[str(self.out/self.drop_artifact)]
        metrics=self.results[chosen['primary']]['same_close']['metrics']
        outcome=dict(passed=True,stage='evaluate',producer_modules_imported=False,available_feature_cells=0,unavailable_feature_cells=5978,source_admission_sha256=c.sha(self.inputs['source_admission']),admitted_dataset_sha256=c.sha(self.inputs['admitted_dataset']),
            execution_paths_verified=9,total_monthly_orders_verified=366,profitability_feature_matrices_verified=3,
            models_fit=0,producer_model_fits=0,plan_sha256=self.plan_sha,cache_sha256=c.sha(self.out/'CACHE.json'),
            orders_sha256=c.sha(self.out/'PRETRAIN_ORDERS.json'),frozen_sha256=c.sha(self.out/'FROZEN.json'),
            diagnose_started_sha256=c.sha(self.out/'DIAGNOSE_STARTED.json'),audit_started_utc='2020-01-01T02:00:00+00:00',
            selected_name=chosen['primary'],primary_cagr=metrics['cagr'],primary_target_reached=metrics['cagr']>=.7,
            source_hashes=self.plan['independent_outcome_source_hashes'],artifact_hashes=artifacts,
            max_nav_absolute_difference=0.,**{key:True for key in c.OUTCOME_CHECKS})
        outcome.update(self.outcome_overrides)
        self.output('INDEPENDENT_OUTCOME.json',outcome)
        self.output('PROTECTED_AFTER.json',self.proof)

    def build(self):return c.build(self.root,self.plan_sha)

    def publish(self):
        result=self.build();path=self.out/'COMPLETION.json';c.publish(path,result)
        return result,c.sha(path)

    def test_complete_preserves_history_metadata_and_adds_exactly_three(self):
        result,digest=self.publish()
        self.assertTrue(c.commit(self.root,digest))
        after=c.decode(self.state_path.read_bytes())
        self.assertEqual(after['completed'][:-1],self.state['completed'])
        self.assertEqual(len(after['completed']),144)
        self.assertEqual(sum(x['trials'] for x in after['completed']),932)
        self.assertEqual(after['untouched'],self.state['untouched'])
        self.assertEqual(after['unexpected_metadata'],42)
        self.assertEqual(after['registered_queue'],[])
        self.assertEqual(after['best_training_selected_candidate'],self.state['best_training_selected_candidate'])
        self.assertFalse(result['verification']['live_cloud_account_state_certified'])
        self.assertTrue(result['verification']['protected_members_rehashed'])

    def test_highest_control_cannot_promote_or_trigger_target(self):
        self.results[c.NAMES[1]]['same_close']['metrics']['cagr']=1.8
        self.setup_evidence();result=self.build()
        self.assertEqual(result['primary'],c.NAMES[0]);self.assertFalse(result['target_reached'])

    def test_admitted_candidate_target_uses_actual_selected_same_close(self):
        self.trials[2]['training']['cagr']=.25
        self.results[c.NAMES[2]]['same_close']['metrics']['cagr']=.7
        self.setup_evidence();result,digest=self.publish();c.commit(self.root,digest)
        self.assertTrue(result['target_reached']);self.assertEqual(result['new_admitted'],1)
        self.assertEqual(c.decode(self.state_path.read_bytes())['best_training_selected_candidate']['spec'],c.NAMES[2])

    def test_second_reference_rejects_otherwise_winning_candidate(self):
        self.trials[2]['training']['cagr']=.25;self.trials[1]['training']['cagr']=.3
        self.setup_evidence();self.assertEqual(self.build()['primary'],c.NAMES[0])

    def test_nonfinite_or_boolean_metric_cannot_admit(self):
        for value in (float('nan'),float('inf'),True,None):
            rows=deepcopy(self.trials);rows[2]['training']['cagr']=value
            self.assertFalse(c.selection(rows,self.open_trials)['admitted'])

    def test_strict_cagr_and_guard_boundaries(self):
        rows=deepcopy(self.trials);rows[2]['training']['cagr']=.2+1e-12
        self.assertFalse(c.selection(rows,self.open_trials)['admitted'])
        rows[2]['training']['cagr']=.2+2e-12;rows[2]['training']['sharpe']=.95
        self.assertTrue(c.selection(rows,self.open_trials)['admitted'])
        rows[2]['training']['sharpe']=.95-1e-12;self.assertFalse(c.selection(rows,self.open_trials)['admitted'])

    def test_partial_prepare_flag_fails(self):
        self.prepare_overrides['zero_and_missing_exact_fallback']=False;self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Incomplete independent prepare'):self.build()

    def test_all_three_population_counts_are_required(self):
        for field,value in [('execution_paths_verified',8),('total_monthly_orders_verified',365),('profitability_feature_matrices_verified',2)]:
            with self.subTest(field=field):
                self.outcome_overrides={field:value};self.setup_evidence()
                with self.assertRaisesRegex(ValueError,'Wrong count'):self.build()

    def test_boolean_counts_and_nonzero_fits_rejected(self):
        for overrides in ({'orders_checked':True},{'monthly_label_models_fit':1},{'new_model_fits':1}):
            self.prepare_overrides=overrides;self.setup_evidence()
            with self.assertRaisesRegex(ValueError,'Wrong count'):self.build()

    def test_missing_execution_artifact_coverage_fails(self):
        self.drop_artifact=c.NAMES[2]+'_next_open_cost_HOLDINGS.json';self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'required independent artifacts'):self.build()

    def test_changed_reviewer_source_fails(self):
        self.write('prepare.py','changed')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.build()

    def test_wrong_independent_source_set_fails(self):
        self.outcome_overrides['source_hashes']=self.plan['independent_prepare_source_hashes'];self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'source set mismatch'):self.build()

    def test_changed_plan_dependency_fails(self):
        self.write(next(iter(self.raw)),'changed')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.build()

    def test_changed_frozen_selection_fails_even_after_new_outcome_artifact_hash(self):
        frozen=c.decode((self.out/'FROZEN.json').read_bytes());frozen['selection']['primary']=c.NAMES[1]
        self.output('FROZEN.json',frozen)
        outcome=c.decode((self.out/'INDEPENDENT_OUTCOME.json').read_bytes())
        outcome['artifact_hashes'][str(self.out/'FROZEN.json')]=c.sha(self.out/'FROZEN.json')
        self.output('INDEPENDENT_OUTCOME.json',outcome)
        with self.assertRaisesRegex(ValueError,'selection mismatch'):self.build()

    def test_protected_receipt_cannot_hide_changed_member(self):
        self.write(next(iter(self.members)),'changed protected bytes')
        with self.assertRaisesRegex(ValueError,'Protected member changed'):self.build()

    def test_protected_missing_member_rejected(self):
        Path(next(iter(self.members))).unlink()
        with self.assertRaises(FileNotFoundError):self.build()

    def test_protected_symlink_rejected(self):
        path=Path(next(iter(self.members)));data=path.read_bytes();path.unlink()
        other=self.write('other',data);path.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'without symlinks'):self.build()

    def test_protected_receipt_requires_verified_count_and_no_read_errors(self):
        for updates in ({'verified':2},{'checked':2},{'read_or_identity_errors':['failed']},{'changed':['x']}):
            self.output('PROTECTED_AFTER.json',dict(self.proof,**updates))
            with self.assertRaisesRegex(ValueError,'Wrong count|incomplete'):self.build()

    def test_bad_synthetic_counts_and_duplicate_suites_fail(self):
        self.suites[0]['count']=12;self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'does not report success'):self.build()
        self.suites[0]['count']=13;self.suites[-1]=deepcopy(self.suites[0]);self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Unique passed'):self.build()

    def test_new_authenticated_suite_count_is_not_hardcoded(self):
        suite=next(x for x in self.suites if x['name']=='outcome_review');suite['count']=65
        self.write(suite['log']['path'],'Ran 65 tests in 0.01s\n\nOK\n')
        suite['log']['sha256']=c.sha(suite['log']['path']);self.setup_evidence()
        self.state['registered_queue']=c.expected_queue(self.plan,self.plan_sha);self.write(self.state_path,self.state)
        self.assertEqual(self.build()['verification']['synthetic_tests']['outcome_review'],65)

    def test_test_manifest_change_after_registration_fails(self):
        self.output('TESTS.json',dict(passed=True,suites=self.suites,test_count=999))
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.build()

    def test_changed_test_log_fails(self):
        self.write(self.suites[0]['log']['path'],'FAILED\n')
        with self.assertRaisesRegex(ValueError,'Hash mismatch|Unbound synthetic'):self.build()

    def test_wrong_predecessor_and_extra_queue_fail(self):
        bad=deepcopy(self.state);bad['completed'][0]['trials']+=1;self.write(self.state_path,bad)
        with self.assertRaisesRegex(ValueError,'929 trials'):self.build()
        bad=deepcopy(self.state);bad['registered_queue'].append(dict(round=144));self.write(self.state_path,bad)
        with self.assertRaisesRegex(ValueError,'Exact registered144'):self.build()

    def test_wrong_queue_plan_hash_fails(self):
        bad=deepcopy(self.state);bad['registered_queue'][0]['plan']['sha256']='0'*64;self.write(self.state_path,bad)
        with self.assertRaisesRegex(ValueError,'Exact registered144'):self.build()

    def test_exclusive_completion_never_overwrites(self):
        result,digest=self.publish();before=(self.out/'COMPLETION.json').read_bytes()
        with self.assertRaises(FileExistsError):c.publish(self.out/'COMPLETION.json',result)
        self.assertEqual(before,(self.out/'COMPLETION.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'resume-completion'):self.build()
        self.assertEqual(list(self.out.glob('.COMPLETION.*.tmp')),[])

    def test_atomic_commit_failure_then_exact_resume_and_idempotency(self):
        result,digest=self.publish()
        with patch.object(c.os,'replace',side_effect=OSError('synthetic failure')):
            with self.assertRaisesRegex(OSError,'synthetic failure'):c.commit(self.root,digest)
        self.assertEqual(self.before,self.state_path.read_bytes())
        self.assertEqual(c.sha(self.out/'COMPLETION.json'),digest)
        self.assertEqual(list(self.state_path.parent.glob('.CONTINUATION_STATUS.*.tmp')),[])
        self.assertTrue(c.commit(self.root,digest));after=self.state_path.read_bytes()
        with patch.object(c.os,'replace',side_effect=AssertionError('no write on retry')):
            self.assertFalse(c.commit(self.root,digest))
        self.assertEqual(after,self.state_path.read_bytes())

    def test_resume_rejects_wrong_completion_hash(self):
        self.publish()
        with self.assertRaisesRegex(ValueError,'Exact completion SHA'):c.commit(self.root,'0'*64)

    def test_resume_rechecks_source_and_protected_bytes(self):
        result,digest=self.publish();self.write(next(iter(self.members)),'later modification')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):c.commit(self.root,digest)
        self.assertEqual(self.before,self.state_path.read_bytes())

    def test_resume_rejects_symlink_even_with_same_protected_bytes(self):
        result,digest=self.publish();path=Path(next(iter(self.members)));data=path.read_bytes();path.unlink()
        other=self.write('other',data);path.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'Protected resume member path'):c.commit(self.root,digest)

    def test_resume_rejects_omitted_protected_member(self):
        result,digest=self.publish();del result['input_hashes'][next(iter(self.members))]
        self.output('COMPLETION.json',result)
        with self.assertRaisesRegex(ValueError,'omitted a protected member'):c.commit(self.root,c.sha(self.out/'COMPLETION.json'))

    def test_wrong_immediate_predecessor_rejected(self):
        bad=deepcopy(self.state);bad['completed'][-1]['directory']='wrong-round';self.write(self.state_path,bad)
        with self.assertRaisesRegex(ValueError,'immediate predecessor'):self.build()

    def test_resume_never_overwrites_changed_central_metadata(self):
        result,digest=self.publish();changed=deepcopy(self.state);changed['untouched']['text']='new user metadata'
        self.write(self.state_path,changed);before=self.state_path.read_bytes()
        with self.assertRaisesRegex(ValueError,'refusing to overwrite'):c.commit(self.root,digest)
        self.assertEqual(before,self.state_path.read_bytes())

    def test_conflict_immediately_before_atomic_commit_fails(self):
        result,digest=self.publish();original=c.sha
        def race(path):
            if Path(path)==self.state_path:self.state_path.write_bytes(self.before+b' ')
            return original(path)
        with patch.object(c,'sha',side_effect=race):
            with self.assertRaisesRegex(ValueError,'before commit'):c.commit(self.root,digest)
        self.assertEqual(self.state_path.read_bytes(),self.before+b' ')

    def test_resume_rejects_modified_selection_even_with_new_completion_digest(self):
        result,digest=self.publish();result['primary']=c.NAMES[1]
        self.output('COMPLETION.json',result)
        with self.assertRaisesRegex(ValueError,'selected arm'):c.commit(self.root,c.sha(self.out/'COMPLETION.json'))

    def test_module_has_no_producer_numpy_or_market_imports(self):
        self.assertNotIn('numpy',c.__dict__);self.assertNotIn('pandas',c.__dict__)
        self.assertEqual(set(c.SUITE_NAMES),set(FIXTURE_SUITES))

    def test_cli_creation_and_exact_resume_are_synthetic_and_idempotent(self):
        buffer=io.StringIO()
        with patch.object(sys,'argv',['closer','--root',str(self.root),'--plan-sha',self.plan_sha]),redirect_stdout(buffer):
            c.main()
        result=json.loads(buffer.getvalue());self.assertEqual(result['strategy_evaluations'],932)
        digest=result['completion']['sha256'];before=self.state_path.read_bytes();buffer=io.StringIO()
        with patch.object(sys,'argv',['closer','--root',str(self.root),'--resume-completion-sha',digest]),redirect_stdout(buffer):
            c.main()
        self.assertFalse(json.loads(buffer.getvalue())['state_committed'])
        self.assertEqual(before,self.state_path.read_bytes())


    def test_nonzero_feature_fit_count_rejected(self):
        self.features['new_model_fits']=1;self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Wrong count'):self.build()

    def test_missing_source_commonmask_or_prefix_proof_rejected(self):
        for name in ('source_admission_bound','common_availability_verified','all_signal_prefixes_verified'):
            self.prepare_overrides={name:False};self.setup_evidence()
            with self.assertRaisesRegex(ValueError,'Incomplete independent prepare'):self.build()

    def test_failed_admission_rejected_even_after_rebinding_plan(self):
        self.admission['checks']['causal_identity_windows']=False;self.sync_admission();self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'source admission checks'):self.build()

    def test_admission_input_mismatch_rejected_even_after_rebinding_plan(self):
        self.sync_admission();self.admission['data_sha256']='0'*64;self.write(self.inputs['source_admission'],self.admission);self.fundamental[str(self.inputs['source_admission'])]=c.sha(self.inputs['source_admission']);self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Admission input binding'):self.build()

    def test_outcome_admission_hash_and_common_mask_required(self):
        for name,value in [('source_admission_sha256','0'*64),('admitted_dataset_sha256','0'*64),('available_feature_cells',1)]:
            self.outcome_overrides={name:value};self.setup_evidence()
            with self.assertRaisesRegex(ValueError,'admission binding|Wrong count'):self.build()

    def test_changed_test_source_rejected(self):
        self.write(self.suites[0]['source']['path'],'different suite implementation')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.build()

    def test_wrong_source_identity_even_if_bound_rejected(self):
        self.suites[0]['source']=self.suites[1]['source'];self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'source identity'):self.build()

    def test_missing_completion_suite_rejected(self):
        self.suites=[x for x in self.suites if x['name']!='completion'];self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'All new'):self.build()

    def test_extra_bound_passed_suite_allowed(self):
        extra=deepcopy(self.suites[0]);extra['name']='contract';self.suites.append(extra)
        self.setup_evidence();self.state['registered_queue']=c.expected_queue(self.plan,self.plan_sha)
        self.write(self.state_path,self.state)
        self.assertEqual(self.build()['verification']['synthetic_tests']['contract'],13)

    def rebind_source_bytes(self):
        """Reseal deliberately bad synthetic documents so semantic checks must act."""
        base=self.source_dir
        audit=c.decode((base/'AUDIT.json').read_bytes())
        audit.update(data_sha256=c.sha(base/'DATA.json'),plan_sha256=c.sha(base/'PLAN.json'),policy_sha256=c.sha(base/'POLICY.json'))
        self.write(base/'AUDIT.json',audit)
        manifest=c.decode((base/'MANIFEST.json').read_bytes())
        manifest.update(plan_sha256=c.sha(base/'PLAN.json'),output_hashes={name:c.sha(base/name) for name in ('DATA.json','AUDIT.json')})
        self.write(base/'MANIFEST.json',manifest)
        self.fundamental.update({str(base/name):c.sha(base/name) for name in c.ADMISSION_FILES})
        self.setup_evidence()

    def verify_source(self):
        return c.verify_admission(c.Reader(self.root),self.plan,self.plan['hashes'])

    def test_all_six_source_artifacts_required(self):
        for name in c.ADMISSION_FILES:
            with self.subTest(name=name):
                path=self.source_dir/name;raw=path.read_bytes();path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError):self.verify_source()
                finally:path.write_bytes(raw)

    def test_source_closure_cannot_omit_or_add_member(self):
        original=deepcopy(self.plan['fundamental_source_hashes'])
        for name in c.ADMISSION_FILES:
            with self.subTest(name=name):
                self.plan['fundamental_source_hashes']=deepcopy(original)
                del self.plan['fundamental_source_hashes'][str(self.source_dir/name)]
                with self.assertRaisesRegex(ValueError,'Exact admitted source closure'):self.verify_source()
        extra=self.write('unrelated_source','not part of source contract')
        self.plan['fundamental_source_hashes']=dict(original,**{str(extra):c.sha(extra)})
        with self.assertRaisesRegex(ValueError,'Exact admitted source closure'):self.verify_source()

    def test_source_zero_counters_reject_bool_and_nonzero(self):
        for key in ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests'):
            for value in (True,1):
                with self.subTest(key=key,value=value):
                    bad=deepcopy(self.admission);bad[key]=value;self.write(self.source_dir/'AUDIT.json',bad)
                    self.rebind_source_bytes()
                    with self.assertRaisesRegex(ValueError,'Wrong count'):self.verify_source()

    def test_source_first_version_and_security_master_cannot_be_certified(self):
        originals={name:(self.source_dir/name).read_bytes() for name in ('AUDIT.json','POLICY.json','DATA.json')}
        for name in originals:
            for key in c.CERTIFICATIONS:
                with self.subTest(name=name,key=key):
                    for filename,raw in originals.items():self.write(self.source_dir/filename,raw)
                    bad=c.decode(originals[name]);bad[key]=True;self.write(self.source_dir/name,bad)
                    self.rebind_source_bytes()
                    with self.assertRaisesRegex(ValueError,'certification limitations'):self.verify_source()

    def test_source_plan_and_audit_closures_must_match(self):
        path=self.source_dir/'PLAN.json';bad=c.decode(path.read_bytes())
        extra=self.write('unlisted-original.html','source not checked by AUDIT')
        bad['source_hashes']=dict(bad['source_hashes'],**{str(extra):c.sha(extra)})
        self.write(path,bad);self.rebind_source_bytes()
        with self.assertRaisesRegex(ValueError,'PLAN/AUDIT closure mismatch'):self.verify_source()

    def test_source_manifest_mismatched_output_rejected(self):
        path=self.source_dir/'MANIFEST.json';bad=c.decode(path.read_bytes());bad['output_hashes']['DATA.json']='0'*64
        self.write(path,bad)
        with self.assertRaisesRegex(ValueError,'Source manifest bindings'):self.verify_source()

    def test_source_plan_cannot_allow_financial_or_network_execution(self):
        path=self.source_dir/'PLAN.json';original=c.decode(path.read_bytes())
        for key in ('financial_computations_permitted','network_requests_permitted'):
            with self.subTest(key=key):
                bad=deepcopy(original);bad[key]=True;self.write(path,bad);self.rebind_source_bytes()
                with self.assertRaisesRegex(ValueError,'source-only plan'):self.verify_source()

    def test_source_counts_and_actual_data_computation_rejected(self):
        path=self.source_dir/'DATA.json';original=c.decode(path.read_bytes())
        for key,value in (('model_admitted',True),('strategy_admitted',True),('ratios_computed',1),('signs_computed',True)):
            with self.subTest(key=key):
                bad=deepcopy(original);bad[key]=value;self.write(path,bad);self.rebind_source_bytes()
                with self.assertRaisesRegex(ValueError,'Source-only dataset|Wrong count'):self.verify_source()
        bad=deepcopy(original);bad['counts']['rejected']=3;self.write(path,bad);self.rebind_source_bytes()
        with self.assertRaisesRegex(ValueError,'Wrong count: rejected'):self.verify_source()

    def test_source_tests_must_be_successful_bound_and_synthetic(self):
        path=self.source_dir/'TESTS.json';original=c.decode(path.read_bytes())
        for key,value in (('synthetic_only',False),('failures',1),('tests_run',True),('log','FAILED')):
            with self.subTest(key=key):
                bad=deepcopy(original);bad[key]=value;self.write(path,bad);self.rebind_source_bytes()
                with self.assertRaises(ValueError):self.verify_source()

    def test_source_assumptions_cannot_disappear(self):
        path=self.source_dir/'AUDIT.json';bad=c.decode(path.read_bytes());bad['limitations']={}
        self.write(path,bad);self.rebind_source_bytes()
        with self.assertRaisesRegex(ValueError,'Source assumptions'):self.verify_source()

    def test_dual_gate_rejects_close_winner_losing_open_control(self):
        self.trials[2]['training']['cagr']=.25;self.open_trials[2]['training']['cagr']=.1
        self.setup_evidence();result=self.build()
        self.assertFalse(result['selection']['admitted']);self.assertFalse(result['selection']['mechanism_confirmed'])
        self.assertEqual(result['primary'],c.NAMES[0])

    def test_open_incumbent_noninferiority_and_growth_strict_boundary(self):
        close=deepcopy(self.trials);close[2]['training']['cagr']=.3
        opened=deepcopy(self.open_trials);opened[0]['training']['cagr']=.21
        self.assertTrue(c.selection(close,opened)['admitted'])
        opened[2]['training']['cagr']=.21-2e-12
        self.assertFalse(c.selection(close,opened)['admitted'])
        opened[0]['training']['cagr']=.19;opened[2]['training']['cagr']=.2+1e-12
        self.assertFalse(c.selection(close,opened)['admitted'])
        opened[2]['training']['cagr']=.2+2e-12
        self.assertTrue(c.selection(close,opened)['admitted'])

    def test_open_risk_and_subsegment_guards_apply_to_both_references(self):
        close=deepcopy(self.trials);close[2]['training']['cagr']=.3
        for section,field,delta in (('training','sharpe',.05),('training','max_drawdown',.03),('early','cagr',.025),('recent','cagr',.025)):
            for index in (0,1):
                with self.subTest(section=section,field=field,index=index):
                    opened=deepcopy(self.open_trials)
                    opened[index][section][field]=opened[2][section][field]+delta+1e-10
                    self.assertFalse(c.selection(close,opened)['admitted'])

    def test_dual_selection_report_flag_is_required(self):
        self.outcome_overrides['first88_dual_execution_selection_verified']=False;self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Incomplete independent evaluate'):self.build()

    def test_open_training_required_in_frozen_and_independent_artifacts(self):
        self.assertIn('TRAIN_OPEN_TRIALS.json',c.FROZEN)
        self.drop_artifact='TRAIN_OPEN_TRIALS.json';self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'required independent artifacts'):self.build()

    def test_diagnostic_marker_must_predate_outcome(self):
        self.outcome_overrides['audit_started_utc']='2020-01-01T00:00:00+00:00';self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'chronology'):self.build()

    def test_same_report_feature_identity_is_required(self):
        self.features['same_current_release_for_both_periods']=False;self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'feature contract'):self.build()

    def test_complete_preserves_source_assumptions(self):
        result=self.build()
        self.assertTrue(result['source_admission_is_conditional'])
        self.assertEqual(result['source_assumptions'],self.admission['limitations'])
        for key in c.CERTIFICATIONS:self.assertIs(result[key],False)
        self.assertEqual(result['first88_next_open_training'],self.open_trials)

    def test_resume_rechecks_next_open_selected_gate(self):
        result,digest=self.publish();result['selection']['next_open_guards_against'][c.NAMES[0]]=False
        self.output('COMPLETION.json',result)
        with self.assertRaisesRegex(ValueError,'selected arm'):c.commit(self.root,c.sha(self.out/'COMPLETION.json'))
        self.assertEqual(self.before,self.state_path.read_bytes())


    def test_resume_cannot_certify_unverified_source_history(self):
        result,digest=self.publish();result['first_public_version_verified']=True
        self.output('COMPLETION.json',result)
        with self.assertRaisesRegex(ValueError,'conditional source assumptions'):
            c.commit(self.root,c.sha(self.out/'COMPLETION.json'))
        self.assertEqual(self.before,self.state_path.read_bytes())

    def test_resume_cannot_drop_source_assumptions(self):
        result,digest=self.publish();result['source_assumptions']={}
        self.output('COMPLETION.json',result)
        with self.assertRaisesRegex(ValueError,'conditional source assumptions'):
            c.commit(self.root,c.sha(self.out/'COMPLETION.json'))


    def test_proposal_source_contract_rejects_different_registered_source(self):
        bad=c.decode(self.proposal.read_bytes());bad['source_contract']['dataset_sha256']='0'*64
        self.write(self.proposal,bad);c.PROPOSAL_SHA=c.sha(self.proposal);self.setup_evidence()
        with self.assertRaisesRegex(ValueError,'Final proposal source contract binding'):self.build()


class ProductionConstants(unittest.TestCase):
    def test_protected_count_and_original_authority_are_not_relaxed(self):
        self.assertEqual(c.PROTECTED_COUNT,21444)
        self.assertEqual(c.PROTECTED_SHA,'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')


if __name__=='__main__':unittest.main(verbosity=2)
