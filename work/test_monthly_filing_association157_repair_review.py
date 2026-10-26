"""Generated arrays and mocked audit I/O only; no actual prices or returns."""
import ast
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import monthly_filing_association157_repair_review as review
import monthly_filing_association157_prepare_review as independent


def scoring_fixture():
    n, width = 122, 49
    names = ['S%02d' % j for j in range(32)] + ['ETF%02d' % j for j in range(17)]
    base = np.tile(np.linspace(.19, -.19, width), (n, 1))
    base += .001 * np.sin(np.arange(n))[:, None]
    shape = base.shape
    zero = np.zeros(shape)
    a = dict(baseline=base, core=base*.5, eligible=np.ones(shape, bool),
        core_weight=np.where(np.arange(n) % 2, .52, .5),
        state_weak_narrow=np.arange(n) % 3 == 0, state_calm10=np.arange(n) % 3 == 1,
        pre_score=base.copy(), effects=.24*np.sin(np.arange(n*width).reshape(shape)),
        forecast_available=np.zeros(shape, bool), rank_log_excess=zero.copy(),
        old_ml4=zero.copy(), extension_available=np.zeros(shape, bool),
        model_available=np.zeros(shape, bool), raw_local_constant=zero.copy(),
        raw_local_linear=zero.copy(), raw_source_log=zero.copy())
    # Preserve 27 independent input arrays, as in the actual prepared contract.
    for j in range(27-len(a)):
        a['unused_original_%02d' % j] = zero.copy()
    a['eligible'][1, 1] = False
    a['eligible'][2] = False; a['eligible'][2, :2] = True
    a['eligible'][3] = False; a['eligible'][3, 0] = True
    days = ['synthetic-%03d' % i for i in range(n)]
    meta = dict(symbols=[names[:] for _ in days], etfs=names[32:], dates=days, ends=days)
    hand = np.full(shape, np.nan)
    hand[:, :32] = .123
    hand[:, :8] = np.array([-1., 1., .75, -.75, .500001, -.500001, 0., np.nan])
    return a, meta, hand


def load_adapter():
    path = review.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
    spec = importlib.util.spec_from_file_location('repair_test_frozen_adapter', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_episode():
    # Execute the exact frozen function without importing unrelated data loaders.
    path = review.ROOT/'outputs/monthly_daily_episodes/episodes.py'
    nodes = ast.parse(path.read_text()).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == 'select_boundary']
    if len(selected) != 1:
        raise AssertionError('One frozen episode implementation required')
    namespace = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['select_boundary']


def backend(meta, proxied=True):
    repaired = importlib.import_module('monthly_filing_association157_repaired_runner')
    adapter, selector = load_adapter(), load_episode()
    helper = independent.old.original_helper()
    b = SimpleNamespace(adapter=repaired.AdapterProxy(adapter, selector) if proxied else adapter,
        selector=selector)
    b.pre = lambda a, n: helper.original_prefix(a, meta, n)['local']
    b.baseline = lambda a, m, n, p: helper.episode(helper.original_prefix(a, m, n)['local'], helper.original_prefix(a, m, n))[:2]
    b.make_orders = lambda choices, scores, m: [dict(date=m['dates'][i], signal_date=m['dates'][i],
        end=m['ends'][i], symbols=[m['symbols'][i][j] for j in picked],
        scores=[float(scores[i, j]) for j in picked]) for i, picked in enumerate(choices)]
    return repaired, b


class NumericalRepairTests(unittest.TestCase):
    def setUp(self):
        self.a, self.meta, self.hand = scoring_fixture()
        self.repaired, self.backend = backend(self.meta)
        self.policy = independent.old.original_helper().POLICY

    def test_real_legacy_adapter_rejects_original_dummy(self):
        _, original = backend(self.meta, False)
        with self.assertRaisesRegex(ValueError, 'bounded ranks'):
            self.repaired.corrected(original.pre(self.a, 122), self.hand, self.a,
                self.meta, 122, original, self.policy)

    def test_repair_matches_independent_point01_h_and_real_episode(self):
        before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in self.a.items()}
        scores, choices, _ = independent.score_oracle(self.a, self.meta, dict.fromkeys(independent.HANDS, self.hand))
        actual_choices, actual_scores = self.repaired.corrected(self.backend.pre(self.a, 122), self.hand,
            self.a, self.meta, 122, self.backend, self.policy)
        self.assertEqual(actual_choices, choices[independent.NAMES[1]])
        np.testing.assert_array_equal(actual_scores, scores[independent.NAMES[1]])
        self.assertEqual(before, {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in self.a.items()})

    def test_all123_prefixes_and_zero_missing_fallback(self):
        scores, choices, _ = independent.score_oracle(self.a, self.meta, dict.fromkeys(independent.HANDS, self.hand))
        for n in range(123):
            c, s = self.repaired.corrected(self.backend.pre(self.a, n), self.hand,
                self.a, self.meta, n, self.backend, self.policy)
            self.assertEqual(c, choices[independent.NAMES[1]][:n])
            np.testing.assert_array_equal(s, scores[independent.NAMES[1]][:n])
        for fill in (0., np.nan):
            h = np.full(self.hand.shape, np.nan); h[:, :32] = fill
            c, s = self.repaired.corrected(self.backend.pre(self.a, 122), h,
                self.a, self.meta, 122, self.backend, self.policy)
            self.assertEqual(c, choices[independent.NAMES[0]])
            np.testing.assert_array_equal(s, scores[independent.NAMES[0]])

    def test_all_four_panels_and_488_orders_match_independent_oracle(self):
        expected_s, expected_c, _ = independent.score_oracle(self.a, self.meta, dict.fromkeys(independent.HANDS, self.hand))
        for name in independent.HANDS:
            self.a[name] = self.hand.copy()
        self.a['common_available'] = np.isfinite(self.hand)
        orders, scores = self.repaired.panels(self.a, self.meta, self.backend, self.policy)
        self.assertEqual(sum(map(len, orders.values())), 488)
        self.assertEqual(orders, independent.order_oracle(self.meta, expected_s, expected_c))
        for name in independent.NAMES:
            np.testing.assert_array_equal(scores[name], expected_s[name])

    def test_true_hand_domain_and_etf_ineligibility_not_relaxed(self):
        for value, column in ((1.0001, 0), (-1.0001, 0), (np.inf, 0), (.75, 48)):
            h = self.hand.copy(); h[0, column] = value
            with self.subTest(value=value, column=column), self.assertRaises(ValueError):
                self.repaired.corrected(self.backend.pre(self.a, 122), h,
                    self.a, self.meta, 122, self.backend, self.policy)

    def test_nonzero_budget_and_wrong_selector_are_not_whitelisted(self):
        proxy = self.backend.adapter
        kwargs = dict(pre_score=self.backend.pre(self.a, 122), effects=self.a['effects'],
            episode_policy=self.policy, boundary_selector=self.backend.selector)
        for budget in (.01, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                proxy.score_and_choose('incumbent66', budget, self.hand, self.a, self.meta, 122, **kwargs)
        kwargs['boundary_selector'] = lambda *args: None
        with self.assertRaises(ValueError):
            proxy.score_and_choose('incumbent66', 0., self.hand, self.a, self.meta, 122, **kwargs)
        kwargs.pop('pre_score')
        with self.assertRaises(ValueError):
            proxy.score_and_choose('incumbent66', 0., self.hand, self.a, self.meta, 122, **kwargs)


class AuditWrapperTests(unittest.TestCase):
    def receipt(self):
        return dict(passed=True, producer_modules_imported=False, stage='prepare',
            source_hashes={'old-reviewer.py': 'o'*64}, artifact_hashes={'old-artifact': 'p'*64})

    def amendment(self):
        return dict(original_plan_sha256=review.PLAN_SHA,
            implementation_hashes={review.SELF: 'b'*64},
            independent_prereview=dict(path=review.PREREVIEW, sha256='c'*64))

    def test_added_provenance_preserves_all_old_fields_and_source_scope(self):
        old = self.receipt(); before = deepcopy(old)
        result = review.augment(old, self.amendment(), 'a'*64, {'new-artifact': 'n'*64})
        self.assertEqual(old, before)
        self.assertEqual(result['source_hashes'], before['source_hashes'])
        for key in before:
            if key != 'artifact_hashes':
                self.assertEqual(result[key], before[key])
        self.assertEqual(result['repair_review_sha256'], 'b'*64)
        self.assertEqual(result['artifact_hashes']['old-artifact'], 'p'*64)
        self.assertEqual(result['artifact_hashes']['new-artifact'], 'n'*64)

    def test_forged_prior_repair_or_conflicting_artifact_fails(self):
        for old, artifacts in ((dict(self.receipt(), amendment_sha256='bad'), {}),
                               (self.receipt(), {'old-artifact': 'wrong'}),
                               (dict(self.receipt(), producer_modules_imported=True), {})):
            with self.assertRaises(ValueError):
                review.augment(old, self.amendment(), 'a'*64, artifacts)

    def test_prepare_and_outcome_call_unchanged_reviewers(self):
        for stage in ('prepare', 'outcome'):
            calls = []
            original = SimpleNamespace(audit=lambda *args: calls.append(args) or self.receipt())
            module = SimpleNamespace(verify_amendment=lambda *args: self.amendment())
            with patch.object(review, 'independent_imports'), patch.object(review, 'helper', return_value=module), \
                 patch.object(review, 'verify_repair_evidence', return_value={}), \
                 patch.object(review.importlib, 'import_module', return_value=original):
                result = review.audit(stage, '/synthetic', review.PLAN_SHA, 'a'*64)
            expected = (Path('/synthetic'), Path('/synthetic')/review.OUT/'PLAN.json', review.PLAN_SHA) if stage == 'prepare' else (Path('/synthetic'), review.PLAN_SHA)
            self.assertEqual(calls, [expected])
            self.assertEqual(result['source_hashes'], self.receipt()['source_hashes'])

    def test_amendment_checked_before_original_audit_and_rechecked_after(self):
        changed = self.amendment(); changed['changed'] = True
        module = SimpleNamespace(verify_amendment=None)
        with patch.object(review, 'independent_imports'), patch.object(review, 'helper', return_value=module), \
             patch.object(module, 'verify_amendment', side_effect=ValueError('bad amendment')), \
             patch.object(review.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'bad amendment'):
                review.audit('prepare', '/synthetic', review.PLAN_SHA, 'a'*64)
            imported.assert_not_called()
        with patch.object(review, 'independent_imports'), patch.object(review, 'helper', return_value=module), \
             patch.object(module, 'verify_amendment', side_effect=[self.amendment(), changed]), \
             patch.object(review, 'verify_repair_evidence', return_value={}), \
             patch.object(review.importlib, 'import_module', return_value=SimpleNamespace(audit=lambda *a: self.receipt())):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                review.audit('prepare', '/synthetic', review.PLAN_SHA, 'a'*64)

    def test_actual_audit_rejects_imported_old_or_new_producer(self):
        for name in review.FORBIDDEN:
            with self.subTest(module=name), patch.dict(sys.modules, {name: SimpleNamespace()}):
                with self.assertRaises(ValueError):
                    review.independent_imports()

    def test_bad_plan_and_stage_fail_before_io(self):
        with patch.object(review, 'helper', side_effect=AssertionError('Unexpected I/O')):
            for stage, plan in (('bad', review.PLAN_SHA), ('prepare', 'b'*64)):
                with self.assertRaises(ValueError):
                    review.audit(stage, '/synthetic', plan, 'a'*64)

    def test_publish_is_write_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'receipt.json'
            review.publish(path, {'synthetic': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                review.publish(path, {'different': True})
            self.assertEqual(path.read_bytes(), before)

    def test_existing_actual_receipt_stops_before_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/review.OUT/'INDEPENDENT_PREPARE.json'
            path.parent.mkdir(parents=True); path.write_text('{}')
            with patch.object(review, 'independent_imports'), \
                 patch.object(review, 'helper', side_effect=AssertionError('Unexpected audit')):
                with self.assertRaisesRegex(ValueError, 'already published'):
                    review.audit('prepare', temporary, review.PLAN_SHA, 'a'*64)


if __name__ == '__main__':
    unittest.main()
