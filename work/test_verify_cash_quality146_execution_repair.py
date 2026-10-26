"""Synthetic files only; never executes preparation, factors, or completion."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_cash_quality146_execution_repair as v


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.now = datetime(2026, 10, 20, 10, tzinfo=timezone.utc)
        self.plan = dict(
            round=146, status='REGISTERED', registered_utc='2026-10-20T01:00:00Z',
            hashes={v.ADAPTER_TEST: self.put(v.ADAPTER_TEST, b'synthetic fixed source')},
            independent_prepare_source_hashes={'work/prepare.py': self.put('work/prepare.py', b'prepare')},
            independent_outcome_source_hashes={'work/outcome.py': self.put('work/outcome.py', b'outcome')})
        self.plan_sha = self.put(v.OUT+'/PLAN.json', self.plan)
        failure = b'Traceback (most recent call last):\nValueError: Original pre-score shape\n'
        self.failure_sha = self.put(v.OUT+'/PREPARE.log', failure)
        direct = {v.OUT+'/PLAN.json': self.plan_sha, v.OUT+'/PREPARE.log': self.failure_sha,
                  v.ADAPTER: self.put(v.ADAPTER, b'new prefix adapter'),
                  v.ADAPTER_TEST: self.hash(v.ADAPTER_TEST)}
        self.adapter_review = dict(
            passed=True, plan_sha256=self.plan_sha,
            repair='full_pre_score_to_requested_prefix', selection_or_factor_changes=False,
            reviewed_utc='2026-10-20T01:30:00Z', input_hashes=dict(direct),
            checks=dict(full_122_pre_score_bytes_unchanged=True,
                        same_n_passed_to_original_backend=True, original_baseline_method_not_overridden=True))
        direct[v.ADAPTER_REVIEW] = self.put(v.ADAPTER_REVIEW, self.adapter_review)
        direct[v.OUT+'/PREFIX_ADAPTER_TESTS.log'] = self.put(
            v.OUT+'/PREFIX_ADAPTER_TESTS.log', b'Ran 4 tests in 0.01s\n\nOK\n')
        self.amendment = dict(plan_sha256=self.plan_sha, repair='full_pre_score_to_requested_prefix',
                              selection_or_factor_changes=False, bindings=direct,
                              created_utc='2026-10-20T02:00:00Z')
        self.amend_sha = self.put(v.AMENDMENT, self.amendment)
        self.started = dict(plan_sha256=self.plan_sha, amendment_sha256=self.amend_sha,
                            started_utc='2026-10-20T03:00:00Z')
        self.put(v.OUT+'/PREFIX_ADAPTER_STARTED.json', self.started)
        self.cache = {name: self.put(v.OUT+'/'+name, ('fake '+name).encode()) for name in v.CACHE_FILES}
        self.cache_sha = self.put(v.OUT+'/CACHE.json', self.cache)
        self.execution = dict(
            plan_sha256=self.plan_sha, amendment_sha256=self.amend_sha,
            cache_sha256=self.cache_sha, full_horizon_pre_score_unchanged=True,
            selection_or_factor_changes=False)
        self.put(v.OUT+'/PREFIX_ADAPTER_EXECUTION.json', self.execution)
        self.artifacts = {v.OUT+'/'+name: sha for name, sha in self.cache.items()}
        self.artifacts.update({v.OUT+'/CACHE.json': self.cache_sha, v.OUT+'/PLAN.json': self.plan_sha})
        self.prepare = dict(
            passed=True, stage='prepare', plan_sha256=self.plan_sha, cache_sha256=self.cache_sha,
            orders_sha256=self.cache['PRETRAIN_ORDERS.json'], producer_modules_imported=False,
            returns_or_nav_computed=False, orders_checked=488, source_cells_checked=5978,
            stock_cells_checked=3904, new_model_fits=0, monthly_label_models_fit=0,
            source_hashes=self.plan['independent_prepare_source_hashes'],
            artifact_hashes=dict(self.artifacts), **{name: True for name in v.PREPARE_CHECKS})
        self.put(v.OUT+'/INDEPENDENT_PREPARE.json', self.prepare)
        self.prepare_started = dict(plan_sha256=self.plan_sha, amendment_sha256=self.amend_sha,
                                    started_utc='2026-10-20T04:00:00Z')
        self.put(v.OUT+'/INDEPENDENT_PREPARE_STARTED.json', self.prepare_started)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else v.encoded(value)
        path.write_bytes(raw)
        return v.digest(raw)

    def hash(self, name):
        return v.digest((self.root/name).read_bytes())

    def check(self, phase='prepare'):
        return v.validate(self.root, self.amend_sha, phase,
                          expected_plan=self.plan_sha, expected_failure=self.failure_sha, now=self.now)

    def complete_fixture(self):
        prior = self.check()
        self.put(v.OUT+'/EXECUTION_REPAIR_PREPARE_REVIEW.json', prior)
        artifacts = dict(self.artifacts)
        artifacts[v.OUT+'/INDEPENDENT_PREPARE.json'] = self.hash(v.OUT+'/INDEPENDENT_PREPARE.json')
        self.outcome = dict(
            passed=True, stage='evaluate', plan_sha256=self.plan_sha, cache_sha256=self.cache_sha,
            execution_paths_verified=12, total_monthly_orders_verified=488,
            cash_quality_feature_matrices_verified=4, models_fit=0, producer_model_fits=0,
            producer_modules_imported=False, source_hashes=self.plan['independent_outcome_source_hashes'],
            artifact_hashes=artifacts, audit_started_utc='2026-10-20T05:00:00Z',
            **{name: True for name in v.OUTCOME_CHECKS})
        outcome_sha = self.put(v.OUT+'/INDEPENDENT_OUTCOME.json', self.outcome)
        protected_sha = self.put(v.OUT+'/PROTECTED_AFTER.json', dict(passed=True))
        self.completion = dict(
            round=146, directory='monthly_cash_quality146', plan_sha256=self.plan_sha,
            independent_outcome_sha256=outcome_sha, completed_utc='2026-10-20T06:00:00Z',
            trials=4, independent_execution_paths=12, protected_files=21444,
            protected_sha256=protected_sha)
        self.put(v.OUT+'/COMPLETION.json', self.completion)

    def test_valid_prepare_chain(self):
        got = self.check()
        self.assertTrue(got['passed'])
        self.assertEqual(got['phase'], 'prepare')
        self.assertTrue(all(got['checks'].values()))
        self.assertEqual(got['verifier_performed_factor_calculations'], 0)
        self.assertFalse((self.root/v.OUT/'EXECUTION_REPAIR_PREPARE_REVIEW.json').exists())

    def test_valid_complete_chain(self):
        self.complete_fixture()
        got = self.check('complete')
        self.assertTrue(got['checks']['completion_bound_to_review_and_protected_receipt'])
        self.assertIn(v.OUT+'/EXECUTION_REPAIR_PREPARE_REVIEW.json', got['input_hashes'])

    def test_frozen_plan_and_source_bytes_must_match(self):
        self.put(v.ADAPTER_TEST, b'modified frozen source')
        with self.assertRaises(ValueError):
            self.check()

    def test_original_failure_cannot_be_replaced(self):
        self.put(v.OUT+'/PREPARE.log', b'No failure')
        with self.assertRaises(ValueError):
            self.check()

    def test_amendment_sha_is_mandatory(self):
        self.amend_sha = '0'*64
        with self.assertRaises(ValueError):
            self.check()

    def test_scope_changes_forbidden(self):
        self.amendment['selection_or_factor_changes'] = True
        self.amend_sha = self.put(v.AMENDMENT, self.amendment)
        with self.assertRaises(ValueError):
            self.check()

    def test_all_critical_repair_dependencies_bound(self):
        self.amendment['bindings'].pop(v.ADAPTER_REVIEW)
        self.amend_sha = self.put(v.AMENDMENT, self.amendment)
        with self.assertRaises(ValueError):
            self.check()

    def test_start_must_match_amendment(self):
        self.started['amendment_sha256'] = '0'*64
        self.put(v.OUT+'/PREFIX_ADAPTER_STARTED.json', self.started)
        with self.assertRaises(ValueError):
            self.check()

    def test_amendment_requires_its_own_time_before_execution(self):
        for value in (None, '2026-10-20T03:30:00Z'):
            if value is None:
                self.amendment.pop('created_utc')
            else:
                self.amendment['created_utc'] = value
            self.amend_sha = self.put(v.AMENDMENT, self.amendment)
            self.started['amendment_sha256'] = self.amend_sha
            self.put(v.OUT+'/PREFIX_ADAPTER_STARTED.json', self.started)
            with self.assertRaises(ValueError):
                self.check()

    def test_start_precedes_independent_review_and_is_not_future(self):
        for value in ('2026-10-20T04:00:00Z', '2026-10-21T03:00:00Z'):
            self.started['started_utc'] = value
            self.put(v.OUT+'/PREFIX_ADAPTER_STARTED.json', self.started)
            with self.assertRaises(ValueError):
                self.check()

    def test_prepare_time_evidence_cannot_be_omitted(self):
        (self.root/v.OUT/'INDEPENDENT_PREPARE_STARTED.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_no_naive_or_non_utc_time(self):
        for value in ('2026-10-20T03:00:00', '2026-10-20T03:00:00+01:00'):
            with self.assertRaises(ValueError):
                v.utc(value)

    def test_cache_mutation_detected(self):
        self.put(v.OUT+'/DATA.npz', b'changed cache')
        with self.assertRaises(ValueError):
            self.check()

    def test_execution_receipt_links_exact_cache(self):
        self.execution['cache_sha256'] = '0'*64
        self.put(v.OUT+'/PREFIX_ADAPTER_EXECUTION.json', self.execution)
        with self.assertRaises(ValueError):
            self.check()

    def test_prepare_requires_exact_count_and_checks(self):
        for key, value in [('orders_checked', 366), ('new_model_fits', False),
                           ('all_signal_prefixes_verified', False), ('producer_modules_imported', True)]:
            previous = self.prepare[key]
            self.prepare[key] = value
            self.put(v.OUT+'/INDEPENDENT_PREPARE.json', self.prepare)
            with self.assertRaises(ValueError):
                self.check()
            self.prepare[key] = previous

    def test_unreviewed_cache_file_rejected(self):
        self.prepare['artifact_hashes'].pop(v.OUT+'/DATA.npz')
        self.put(v.OUT+'/INDEPENDENT_PREPARE.json', self.prepare)
        with self.assertRaises(ValueError):
            self.check()

    def test_complete_needs_preparation_repair_review(self):
        self.complete_fixture()
        (self.root/v.OUT/'EXECUTION_REPAIR_PREPARE_REVIEW.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.check('complete')

    def test_complete_cannot_use_wrong_outcome_or_early_time(self):
        self.complete_fixture()
        for key, value in [('independent_outcome_sha256', '0'*64),
                           ('completed_utc', '2026-10-20T04:30:00Z'),
                           ('independent_execution_paths', 9)]:
            previous = self.completion[key]
            self.completion[key] = value
            self.put(v.OUT+'/COMPLETION.json', self.completion)
            with self.assertRaises(ValueError):
                self.check('complete')
            self.completion[key] = previous

    def test_complete_source_change_detected(self):
        self.complete_fixture()
        self.put('work/outcome.py', b'changed independent reviewer')
        with self.assertRaises(ValueError):
            self.check('complete')

    def test_only_exact_external_interpreter_is_allowed_and_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory).resolve()/'python'
            runtime.write_bytes(b'synthetic interpreter')
            sha = v.digest(runtime.read_bytes())
            with patch.object(v, 'RUNTIME', str(runtime)), patch.object(v, 'RUNTIME_SHA', sha):
                evidence = v.PlanEvidence(self.root)
                self.assertEqual(evidence.hash(str(runtime)), sha)
                with self.assertRaises(ValueError):
                    evidence.hash(str(runtime), '0'*64)
                with self.assertRaises(ValueError):
                    evidence.path(str(runtime.parent/'other'))
                runtime.write_bytes(b'changed interpreter')
                with self.assertRaises(ValueError):
                    v.PlanEvidence(self.root).hash(str(runtime))

    def test_external_interpreter_cannot_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()/'target'
            target.write_bytes(b'synthetic interpreter')
            link = target.parent/'python'
            link.symlink_to(target)
            with patch.object(v, 'RUNTIME', str(link)):
                with self.assertRaises(ValueError):
                    v.PlanEvidence(self.root).path(str(link))


if __name__ == '__main__':
    unittest.main()
