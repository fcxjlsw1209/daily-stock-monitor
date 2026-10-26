"""Synthetic metadata regressions for explicit157 code repair provenance."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import tempfile
import unittest
from unittest.mock import patch
import monthly_filing_association157_repair as r


class RepairContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=r.ROOT/'work',prefix='test-repair157-')
        self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        (self.root/r.OUT).mkdir(parents=True);(self.root/'work').mkdir()
        for name in r.FILES:(self.root/name).write_text('# synthetic dependency\n')
        plan=dict(round=157,status='REGISTERED',dose=.01,registered_utc='2026-01-01T00:00:00+00:00')
        self.put(r.OUT+'/PLAN.json',plan);self.plan_sha=r.sha(self.root/r.OUT/'PLAN.json')
        context=patch.object(r,'PLAN_SHA',self.plan_sha);context.start();self.addCleanup(context.stop)
        failure=dict(original_plan_sha256=self.plan_sha,exit_code=1,recorded_utc='2026-01-02T00:00:00+00:00',
            activity=dict(real_features_calculated_in_memory=True,real_nuisance_estimates_attempted=True,
                exact_estimate_counts_persisted=False,strategy_returns_calculated=0,prepared_artifacts_published=0))
        self.put(r.FAILURE,failure)
        tests={}
        for name,index in [('runner',1),('lifecycle',3),('independent',5)]:
            tests[name]=dict(count=2,passed=True,source_sha256=r.sha(self.root/r.FILES[index]),
                             log='Ran 2 tests in 0.001s\n\nOK\n')
        review=dict(passed=True,synthetic_only=True,semantics_preserved=True,
            original_plan_sha256=self.plan_sha,actual_returns_calculated=0,
            implementation_hashes=r.implementation_hashes(self.root),failure_record=r.reference(self.root,r.FAILURE),
            tests=tests,reviewed_utc='2026-01-03T00:00:00+00:00')
        self.put(r.PREREVIEW,review)
        self.doc=dict(schema_version=1,round=157,kind='zero_budget_adapter_dummy_only',
            original_plan_sha256=self.plan_sha,same_experiment=True,additional_strategy_trials=0,
            formula_changed=False,parameters_changed=False,registered_utc='2026-01-04T00:00:00+00:00',
            implementation_hashes=r.implementation_hashes(self.root),failure_record=r.reference(self.root,r.FAILURE),
            independent_prereview=r.reference(self.root,r.PREREVIEW))
        self.put(r.AMENDMENT,self.doc)

    def put(self,path,value):
        (self.root/path).write_text(json.dumps(value))

    def verify(self):
        return r.verify_amendment(self.root,self.plan_sha,r.sha(self.root/r.AMENDMENT))

    def test_complete_synthetic_provenance_passes(self):
        self.assertEqual(self.verify(),self.doc)

    def test_original_plan_mutation_rejected(self):
        self.put(r.OUT+'/PLAN.json',dict(round=158))
        with self.assertRaisesRegex(ValueError,'PLAN'):self.verify()

    def test_every_new_dependency_is_bound(self):
        for name in r.FILES:
            p=self.root/name;before=p.read_bytes();p.write_bytes(before+b'# changed')
            with self.assertRaisesRegex(ValueError,'implementation changed'):self.verify()
            p.write_bytes(before)

    def test_cannot_relabel_parameter_or_formula_change(self):
        for field in ('formula_changed','parameters_changed'):
            modified=dict(self.doc);modified[field]=True;self.put(r.AMENDMENT,modified)
            with self.assertRaisesRegex(ValueError,'contract'):self.verify()

    def test_same_experiment_not_extra_trials(self):
        modified=dict(self.doc,additional_strategy_trials=4);self.put(r.AMENDMENT,modified)
        with self.assertRaisesRegex(ValueError,'contract'):self.verify()

    def test_failure_activity_must_not_be_erased(self):
        failure=r.read(self.root/r.FAILURE);failure['activity']['real_features_calculated_in_memory']=False
        with self.assertRaisesRegex(ValueError,'actual failed'):r.validate_failure(failure)

    def test_no_ambiguous_numeric_flags(self):
        self.doc['same_experiment']=1;self.put(r.AMENDMENT,self.doc)
        with self.assertRaisesRegex(ValueError,'contract'):self.verify()

    def test_amendment_timestamp_must_follow_review(self):
        self.doc['registered_utc']='2026-01-02T23:00:00+00:00';self.put(r.AMENDMENT,self.doc)
        with self.assertRaisesRegex(ValueError,'chronology'):self.verify()

    def test_review_all_three_suites_required(self):
        review=r.read(self.root/r.PREREVIEW);review['tests'].pop('runner')
        with self.assertRaisesRegex(ValueError,'three'):r.validate_prereview(self.root,review)

    def test_test_success_requires_actual_ok_log(self):
        review=r.read(self.root/r.PREREVIEW);review['tests']['runner']['log']='Ran 2 tests in 0.001s\nFAILED'
        with self.assertRaisesRegex(ValueError,'test evidence'):r.validate_prereview(self.root,review)

    def test_phase_entry_rejects_already_published_actual_work(self):
        r.unprepared(self.root)
        for name in r.REAL_ARTIFACTS:
            p=self.root/r.OUT/name;p.write_text('present')
            with self.assertRaisesRegex(ValueError,'already exist'):r.unprepared(self.root)
            p.unlink()

    def test_exclusive_publication_preserves_original(self):
        p=self.root/'work/once.json';r.write_once(p,{'a':1});before=p.read_bytes()
        with self.assertRaises(FileExistsError):r.write_once(p,{'a':2})
        self.assertEqual(p.read_bytes(),before)

    def test_local_paths_and_symlinks_cannot_escape(self):
        for value in ('../bad','/tmp/bad'):
            with self.assertRaises(ValueError):r.local(self.root,value)
        (self.root/'escape').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):r.local(self.root,'escape/outside')

    def test_audit_cannot_omit_new_code_or_amendment(self):
        report=dict(original_plan_sha256=self.plan_sha,amendment_sha256=r.sha(self.root/r.AMENDMENT),
            repair_source_hashes=r.implementation_hashes(self.root),repair_review_sha256=r.sha(self.root/r.FILES[4]),
            artifact_hashes={str(self.root/n):r.sha(self.root/n) for n in (r.AMENDMENT,r.FAILURE,r.PREREVIEW)})
        r.verify_repair_report(self.root,report,self.plan_sha,report['amendment_sha256'])
        for name in (r.AMENDMENT,r.FAILURE,r.PREREVIEW):
            broken=deepcopy(report);broken['artifact_hashes'].pop(str(self.root/name))
            with self.assertRaisesRegex(ValueError,'Missing repair evidence'):
                r.verify_repair_report(self.root,broken,self.plan_sha,report['amendment_sha256'])

    def test_no_top_level_producer_imports(self):
        tree=ast.parse(Path(r.__file__).read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any('monthly_' in (name or '') or 'after156' in (name or '') for name in names))


if __name__=='__main__':unittest.main(verbosity=2)
