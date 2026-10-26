"""Synthetic amendment gates and temporary cell substitution; no real phases."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import monthly_annual_lag158_repaired_runner as r


class RepairedRunner(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'out';self.out.mkdir()
        self.runner=object.__new__(r.Runner)
        self.runner.root=self.root;self.runner.out=self.out
        self.runner.plan_path=self.out/'PLAN.json';self.runner.plan_path.write_text('{}')
        self.runner.plan_sha=r.original.sha(self.runner.plan_path);self.runner.amendment_sha='a'*64
        self.code=self.root/'synthetic.py';self.code.write_text('# original')
        self.runner.pins={str(self.code):r.original.sha(self.code)}

    def test_context_restores_after_success(self):
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)
        with r.repaired_cells():self.assertIs(r.original.model._cell,r.cell.repaired_cell)
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)

    def test_context_restores_after_failure(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with r.repaired_cells():raise RuntimeError('synthetic failure')
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)

    def test_context_refuses_nested_or_existing_replacement(self):
        with r.repaired_cells():
            with self.assertRaisesRegex(ValueError,'no nested patch'):
                with r.repaired_cells():pass
            self.assertIs(r.original.model._cell,r.cell.repaired_cell)
        foreign=lambda *args:None
        with patch.object(r.original.model,'_cell',foreign):
            with self.assertRaises(ValueError):
                with r.repaired_cells():pass
            self.assertIs(r.original.model._cell,foreign)

    def test_context_restores_and_rejects_midphase_replacement(self):
        with self.assertRaisesRegex(ValueError,'Unexpected cell replacement'):
            with r.repaired_cells():r.original.model._cell=lambda *args:None
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)

    def test_constructor_checks_amendment_before_original_constructor(self):
        with patch.object(r.Runner,'_verify_amendment',side_effect=ValueError('not registered')),\
                patch.object(r.original.Runner,'__init__') as original:
            with self.assertRaisesRegex(ValueError,'not registered'):
                r.Runner(self.root,'PLAN.json','p'*64,'a'*64)
            original.assert_not_called()

    def test_constructor_retains_original_plan_and_backend_checks(self):
        backend=object()
        with patch.object(r.Runner,'_verify_amendment') as verify,\
                patch.object(r.original.Runner,'__init__',return_value=None) as original:
            instance=r.Runner(self.root,'PLAN.json','p'*64,'a'*64,backend_factory=backend)
        verify.assert_called_once_with()
        original.assert_called_once_with(self.root,'PLAN.json','p'*64,backend_factory=backend)
        self.assertEqual(instance.amendment_sha,'a'*64)

    def test_lazy_verifier_exact_api(self):
        verify=Mock(return_value={'verified':True})
        with patch.dict(sys.modules,{'annual_lag158_repair':SimpleNamespace(verify_amendment=verify)}):
            self.assertEqual(self.runner._verify_amendment(),{'verified':True})
        verify.assert_called_once_with(self.root,self.runner.plan_sha,self.runner.amendment_sha)

    def test_write_checks_amendment_once_and_original_pins_before_write(self):
        with patch.object(self.runner,'_verify_amendment') as verify:
            self.runner.write('synthetic.json',{'pass':True})
        verify.assert_called_once_with()
        self.code.write_text('# changed')
        with patch.object(self.runner,'_verify_amendment'),self.assertRaisesRegex(ValueError,'Changed dependency'):
            self.runner.write('rejected.json',{})
        self.assertFalse((self.out/'rejected.json').exists())

    def test_failed_amendment_prevents_write_and_prepare(self):
        with patch.object(self.runner,'_verify_amendment',side_effect=ValueError('amendment changed')),\
                patch.object(r.original.Runner,'prepare') as prepare:
            with self.assertRaises(ValueError):self.runner.write('rejected.json',{})
            with self.assertRaises(ValueError):self.runner.prepare()
            prepare.assert_not_called()
        self.assertFalse((self.out/'rejected.json').exists())
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)

    def test_only_prepare_substitutes_cell_and_restores_on_failure(self):
        def phase(self):
            if r.original.model._cell is not r.cell.repaired_cell:raise AssertionError('patch absent')
            raise RuntimeError('synthetic prepare failure')
        with patch.object(self.runner,'_verify_amendment'),patch.object(r.original.Runner,'prepare',phase):
            with self.assertRaisesRegex(RuntimeError,'synthetic prepare failure'):self.runner.prepare()
        self.assertIs(r.original.model._cell,r.ORIGINAL_CELL)
        self.assertIs(r.Runner.select_first88,r.original.Runner.select_first88)
        self.assertIs(r.Runner.diagnose_later34,r.original.Runner.diagnose_later34)

    def test_prepare_review_runs_original_guard_then_separate_repair_guard(self):
        report={'synthetic':True};(self.out/'INDEPENDENT_PREPARE.json').write_text(json.dumps(report))
        calls=[]
        module=SimpleNamespace(verify_repair_report=Mock(side_effect=lambda *a:calls.append('repair')))
        with patch.dict(sys.modules,{'annual_lag158_repair':module}),\
                patch.object(r.original.Runner,'require_prepare_review',side_effect=lambda:calls.append('original')):
            self.runner.require_prepare_review()
        self.assertEqual(calls,['original','repair'])
        module.verify_repair_report.assert_called_once_with(self.root,report,self.runner.plan_sha,self.runner.amendment_sha)

    def test_failed_original_prepare_review_cannot_be_bypassed(self):
        module=SimpleNamespace(verify_repair_report=Mock())
        with patch.dict(sys.modules,{'annual_lag158_repair':module}),\
                patch.object(r.original.Runner,'require_prepare_review',side_effect=ValueError('original gate')):
            with self.assertRaisesRegex(ValueError,'original gate'):self.runner.require_prepare_review()
        module.verify_repair_report.assert_not_called()


if __name__=='__main__': unittest.main(verbosity=2)
