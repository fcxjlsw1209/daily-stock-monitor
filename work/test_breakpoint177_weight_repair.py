"""Synthetic-only regressions for round177's audit-shape amendment."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

import breakpoint177_weight_repair as repair


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'work/monthly_breakpoint177_independent.py'
FROZEN_SHA = '4e36fb95bba0b3e7a9708db14e5cfa7e5b01a640b473ddcff052abde2d40339f'


def frozen():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA
    spec = importlib.util.spec_from_file_location('repair177_frozen_independent', FROZEN)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture(shape='vector'):
    baseline = np.array([[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]])
    candidate = baseline[:, ::-1].copy()
    values = dict(baseline=baseline, candidate=candidate, control=baseline.copy(),
                  available=np.ones(baseline.shape, dtype=bool))
    calm = np.array([False, False, True])
    weight = np.where(calm, .5, .52)
    # Ascending average percentile minus 0.5: 1/2, 1/6, -1/6.
    rank = np.array([.5, 2. / 3. - .5, 1. / 3. - .5])
    original = dict(core=np.tile(.52 * rank, (3, 1)), core_weight=weight,
                    state_calm10=calm, eligible=np.ones((3, 3), dtype=bool))
    original['core'][2] = [.3, -.1, .2]  # Calm's different core is left untouched.
    if shape == 'column': original['core_weight'] = weight[:, None].copy()
    elif shape == 'matrix': original['core_weight'] = np.repeat(weight[:, None], 3, axis=1)
    elif shape != 'vector': raise ValueError(shape)
    return values, original


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.old = frozen()

    def test_three_legal_shapes_equal_frozen_vector_receipt(self):
        values, vector = fixture()
        deltas = self.old.independent_deltas(values, vector)
        expected = self.old.verify_deltas(values, deltas, vector)
        for shape in ('vector', 'column', 'matrix'):
            _, original = fixture(shape)
            before = {k: v.copy() for k, v in original.items()}
            receipt = repair.adapted_verify_deltas(values, deltas, original, self.old.verify_deltas)
            evidence = receipt.pop('weight_shape_adapter')
            self.assertEqual(receipt, expected)
            self.assertEqual(evidence['original_shape'], list(original['core_weight'].shape))
            self.assertEqual(evidence['normalized_shape'], [3])
            self.assertEqual(evidence['validated_weight_cells'], original['core_weight'].size)
            for key in before: np.testing.assert_array_equal(before[key], original[key])

    def test_only_weight_is_a_new_object(self):
        values, original = fixture('column')
        def callback(v, d, view, dose):
            self.assertIs(v, values); self.assertIsNot(view, original)
            self.assertEqual(dose, .5)
            self.assertFalse(np.shares_memory(view['core_weight'], original['core_weight']))
            for key in original:
                if key != 'core_weight': self.assertIs(view[key], original[key])
            return dict(passed=True, marker='frozen receipt preserved')
        receipt = repair.adapted_verify_deltas(values, {}, original, callback)
        self.assertEqual(receipt['marker'], 'frozen receipt preserved')
        self.assertTrue(receipt['weight_shape_adapter']['original_arrays_unchanged'])

    def test_nonfirst_column_mismatch_rejected(self):
        values, original = fixture('matrix'); original['core_weight'][1, -1] = .50
        with self.assertRaisesRegex(ValueError, 'Every original weight'):
            repair.adapted_verify_deltas(values, {}, original, self.old.verify_deltas)

    def test_every_state_weight_is_checked_including_calm(self):
        for row in (0, 1, 2):
            values, original = fixture(); original['core_weight'][row] += .01
            with self.assertRaises(ValueError): repair.adapted_verify_deltas(values, {}, original, self.old.verify_deltas)

    def test_finite_numeric_and_shape_validation(self):
        for bad in (np.array([.52, np.nan, .5]), np.array([.52, np.inf, .5]),
                    np.array(['.52', '.52', '.5']), np.ones((3, 2)),
                    np.ones((3, 1, 1)), np.array([.52, .52]), np.array(.52),
                    np.array([.52 + 0j, .52 + 0j, .5 + 0j])):
            values, original = fixture(); original['core_weight'] = bad
            with self.assertRaises(ValueError): repair.adapted_verify_deltas(values, {}, original, self.old.verify_deltas)

    def test_state_must_be_boolean_n_vector(self):
        for bad in (np.array([0., 0., 1.]), np.array([False, True]), np.array([[False], [False], [True]])):
            values, original = fixture(); original['state_calm10'] = bad
            with self.assertRaises(ValueError): repair.adapted_verify_deltas(values, {}, original, self.old.verify_deltas)

    def test_wrong_delta_still_rejected_by_frozen_checker(self):
        values, vector = fixture(); deltas = self.old.independent_deltas(values, vector)
        _, original = fixture('column'); deltas['breakpoint_core'][1, 0] += .001
        before = repair._snapshot(original)
        with self.assertRaisesRegex(ValueError, 'Half-core delta differs'):
            repair.adapted_verify_deltas(values, deltas, original, self.old.verify_deltas)
        self.assertEqual(before, repair._snapshot(original))

    def test_no_change_to_other_original_gates(self):
        values, vector = fixture(); deltas = self.old.independent_deltas(values, vector)
        _, original = fixture('matrix'); original['core'][1, 0] += .0001
        with self.assertRaisesRegex(ValueError, 'Original noncalm core rank parity'):
            repair.adapted_verify_deltas(values, deltas, original, self.old.verify_deltas)

    def test_callback_and_dose_requirements(self):
        values, original = fixture('column')
        for callback in (None, repair.adapted_verify_deltas):
            with self.assertRaises(ValueError): repair.adapted_verify_deltas(values, {}, original, callback)
        for dose in (.25, 1., True):
            with self.assertRaises(ValueError): repair.adapted_verify_deltas(values, {}, original, self.old.verify_deltas, dose)

    def test_callback_exception_propagates_without_original_changes(self):
        values, original = fixture('column'); before = repair._snapshot(original)
        def callback(*args, **kwargs): raise RuntimeError('expected verifier failure')
        with self.assertRaisesRegex(RuntimeError, 'expected verifier failure'):
            repair.adapted_verify_deltas(values, {}, original, callback)
        self.assertEqual(before, repair._snapshot(original))

    def test_callback_result_must_pass_and_not_duplicate_shape_receipt(self):
        values, original = fixture('column')
        for result in (None, {'passed': False}, {'passed': True, 'weight_shape_adapter': {}}):
            with self.assertRaises(ValueError):
                repair.adapted_verify_deltas(values, {}, original, lambda *a, **k: result)

    def test_callback_mutations_cannot_receive_success_receipt(self):
        for mutation in ('original_content', 'original_identity', 'normalized_content', 'normalized_identity'):
            for raise_after in (False, True):
                with self.subTest(mutation=mutation, raise_after=raise_after):
                    values, original = fixture('column')
                    def callback(v, d, view, dose):
                        if mutation == 'original_content':
                            view['core'][1, 0] += .01
                        elif mutation == 'original_identity':
                            original['eligible'] = original['eligible'].copy()
                        elif mutation == 'normalized_content':
                            view['core_weight'][0] += .01
                        else:
                            view['core_weight'] = view['core_weight'].copy()
                        if raise_after:
                            raise RuntimeError('callback failed after mutation')
                        return {'passed': True}
                    expected = ('Original array identity' if mutation.startswith('original_')
                                else 'Verifier changed normalized weights')
                    with self.assertRaisesRegex(ValueError, expected):
                        repair.adapted_verify_deltas(values, {}, original, callback)

    def test_narrow_module_has_no_loader_or_producer_dependencies(self):
        source = Path(repair.__file__).read_text(); tree = ast.parse(source)
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [v.name for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names]
        self.assertEqual(names, ['numpy'])
        self.assertNotIn('__main__', source)
        self.assertEqual(hashlib.sha256(FROZEN.read_bytes()).hexdigest(), FROZEN_SHA)


if __name__ == '__main__':
    unittest.main()
