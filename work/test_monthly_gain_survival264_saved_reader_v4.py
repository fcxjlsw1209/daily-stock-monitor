"""Synthetic metadata integration only; stop before any prepared/result phase.

Uses the pinned stdlib Evidence implementation with fabricated PLAN/receipts and
seven source files. No actual round264 artifacts, prices, arrays or outcomes.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import monthly_gain_survival264_saved_reader as legacy
import monthly_gain_survival264_saved_reader_v4 as reader

BASE_SHA='e1cb3f142c068762eff25593f1ee9b85e3ab60f08369d1a284cfbf62c7b50cdf'
MAPPING_OLD="path=ROOT/('work/monthly_gain_survival264_'+name+'.json')"
MAPPING_NEW="path=ROOT/('work/monthly_gain_survival264_'+{'TESTS':'TESTS_V2','CODE_REVIEW':'CODE_REVIEW'}[name]+'.json')"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhaseBoundary(Exception):
    pass


class RegisteredEvidenceIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gain264-reader-v4-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'outputs/monthly_gain_survival264'
        self.work=self.root/'work';self.work.mkdir();self.out.mkdir(parents=True)
        self.parent=reader.load_parent()  # Pinned stdlib source only, no data read.
        for relative in ('work/monthly_weak_tree261_saved_reader.py',
                         'work/monthly_tracking_risk262_saved_reader_v2.py',
                         'outputs/monthly_factor_search_v2/validate.py'):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(reader.ROOT/relative,target)
        for module in (reader,legacy):
            shutil.copyfile(module.__file__,self.work/Path(module.__file__).name)
        self.code={}
        for name in ('host','test_host','kernel','test_kernel','independent','test_independent','sources'):
            file=self.work/('synthetic_'+name+'.py');file.write_text('# synthetic '+name+'\n')
            self.code[str(file.relative_to(self.root))]=sha(file)
        self.tests_path=self.work/'monthly_gain_survival264_TESTS_V2.json'
        self.old_path=self.work/'monthly_gain_survival264_TESTS.json'
        self.review_path=self.work/'monthly_gain_survival264_CODE_REVIEW.json'
        self.context_path=self.work/'monthly_gain_survival264_context.json'
        self.contract_path=self.work/'monthly_gain_survival264_IMPLEMENTATION_CONTRACT.json'
        self.runtime=dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3')
        self.context=dict(frozen=True,arms=list(reader.ARMS))
        self.write(self.context_path,self.context)
        self.write(self.contract_path,dict(frozen=True,context=dict(sha256=sha(self.context_path))))
        self.tests=dict(passed=True,synthetic_only=True,tests=141,runtime=self.runtime,code_sha256=self.code)
        self.review=dict(passed=True,unresolved_blockers=[],actual_returns_computed=False,code_sha256=self.code)
        self.write(self.tests_path,self.tests);self.write(self.review_path,self.review)
        self.write(self.old_path,dict(passed=True,tests=139,code_sha256={k:'0'*64 for k in self.code}))
        self.plan=dict(context=self.context,context_sha256=sha(self.context_path),
            implementation_contract_sha256=sha(self.contract_path),runtime=self.runtime,
            input_sha256={str(p.relative_to(self.root)):sha(p) for p in (self.tests_path,self.review_path)})
        self.instances=[];self.phase_calls=[]
        owner=self
        class Evidence(self.parent.Evidence):
            def __init__(self):
                super().__init__();owner.instances.append(self)
            def phase(self,*args):
                owner.phase_calls.append(args)
                raise PhaseBoundary('No phase artifacts are read in this synthetic test')
        self.evidence_type=Evidence

    def write(self,path,value):
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))

    def refresh(self,path,value):
        self.write(path,value);self.plan['input_sha256'][str(path.relative_to(self.root))]=sha(path)

    def run_prefix(self,module=reader):
        self.write(self.out/'PLAN.json',self.plan)
        args=SimpleNamespace(plan_sha256=sha(self.out/'PLAN.json'),prepared_sha256='1'*64,
            frozen_sha256='2'*64,results_sha256='3'*64,terminals_sha256='4'*64)
        with patch.object(self.parent,'ROOT',self.root),patch.object(self.parent,'Evidence',self.evidence_type),\
             patch.object(module,'ROOT',self.root),patch.object(module,'OUT',self.out),\
             patch.object(module,'PARENT',self.work/'monthly_weak_tree261_saved_reader.py'),\
             patch.object(module,'__file__',str(self.work/Path(module.__file__).name)),\
             patch.object(module,'load_parent',return_value=self.parent):
            module.run(args)

    def test_v4_is_exactly_two_authorized_byte_replacements(self):
        base=Path(legacy.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(base).hexdigest(),BASE_SHA)
        self.assertEqual(base.count(MAPPING_OLD.encode()),1)
        expected=base.replace(MAPPING_OLD.encode(),MAPPING_NEW.encode()).replace(
            b"schema='monthly_gain_survival264_root_saved_review_v3'",
            b"schema='monthly_gain_survival264_root_saved_review_v4'")
        self.assertEqual(Path(reader.__file__).read_bytes(),expected)

    def test_legacy_path_absent_from_plan_reproduces_original_failure(self):
        self.assertNotIn(str(self.old_path.relative_to(self.root)),self.plan['input_sha256'])
        with self.assertRaisesRegex(ValueError,'Missing/conflicting registered pin: .*TESTS.json'):
            self.run_prefix(legacy)
        self.assertEqual(self.phase_calls,[])

    def test_registered_v2_and_all_seven_current_source_pins_reach_phase_boundary(self):
        with self.assertRaises(PhaseBoundary):self.run_prefix()
        self.assertEqual(len(self.phase_calls),1)
        self.assertEqual(self.phase_calls[0][0],self.out/'prepared')
        pins=self.instances[-1].pins
        self.assertEqual(pins[str(self.tests_path.relative_to(self.root))],sha(self.tests_path))
        self.assertEqual(pins[str(self.review_path.relative_to(self.root))],sha(self.review_path))
        for path,digest in self.code.items():self.assertEqual(pins[path],digest)
        self.assertNotIn(str(self.old_path.relative_to(self.root)),pins)

    def test_registered_legacy_receipt_cannot_replace_missing_v2_pin(self):
        self.plan['input_sha256'].pop(str(self.tests_path.relative_to(self.root)))
        self.plan['input_sha256'][str(self.old_path.relative_to(self.root))]=sha(self.old_path)
        with self.assertRaisesRegex(ValueError,'Missing/conflicting registered pin: .*TESTS_V2.json'):
            self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_registered_v2_digest_mismatch_rejected_before_phase(self):
        self.plan['input_sha256'][str(self.tests_path.relative_to(self.root))]='0'*64
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*TESTS_V2.json'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_each_current_source_pin_is_enforced(self):
        for relative in self.code:
            path=self.root/relative;before=path.read_bytes();path.write_bytes(before+b'# changed\n')
            try:
                with self.subTest(path=relative),self.assertRaisesRegex(ValueError,'Digest mismatch: .*synthetic_'):
                    self.run_prefix()
            finally:path.write_bytes(before)
        self.assertEqual(self.phase_calls,[])

    def test_old_source_digest_in_registered_v2_cannot_pass(self):
        stale=dict(self.tests,code_sha256=dict(self.code));stale['code_sha256'][next(iter(self.code))]='0'*64
        self.refresh(self.tests_path,stale)
        with self.assertRaisesRegex(ValueError,'Digest mismatch: .*synthetic_'):self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_synthetic_and_runtime_guards_unchanged(self):
        for changes in (dict(synthetic_only=False),dict(runtime=dict(self.runtime,python='wrong'))):
            self.refresh(self.tests_path,dict(self.tests,**changes))
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Formal synthetic/runtime evidence'):
                self.run_prefix()
        self.assertEqual(self.phase_calls,[])

    def test_code_review_guards_unchanged(self):
        for changes in (dict(passed=False),dict(unresolved_blockers=['blocked']),dict(actual_returns_computed=True)):
            self.refresh(self.review_path,dict(self.review,**changes))
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Formal CODE_REVIEW|Prospective code review'):
                self.run_prefix()
        self.assertEqual(self.phase_calls,[])


def load_tests(loader, standard_tests, pattern):
    # Preserve the registered reader's 21 synthetic checks, against V4 only.
    # The frozen test file remains byte-for-byte unchanged.
    import test_monthly_gain_survival264_saved_reader as previous_tests
    previous_tests.reader=reader
    standard_tests.addTests(loader.loadTestsFromModule(previous_tests))
    return standard_tests


if __name__=='__main__':unittest.main()
