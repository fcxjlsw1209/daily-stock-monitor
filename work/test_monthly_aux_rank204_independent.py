"""Synthetic-only scalar, original-formula, prefix, and artifact tests."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import monthly_aux_rank204_independent as oracle


def fixture():
    calm = np.array([False, True, True, True, False, False, False, True])
    weak = np.array([False, False, False, True, False, True, False, False])
    n, width = len(calm), 5
    support = np.array([[-.4, -.1, 0., .02, .3], [-.4, -.4, 0., .4, .4],
        [.125] * 5, [-.25, -.25, .125, .125, .125], [-.4, -.1, .01, .02, .3],
        [-.4, -.2, .01, .015, .2], [-.3, -.1, 0., .05, .2], [-.4, -.1, 0., .02, .3]])
    cw = np.where(calm, .5, .52)[:, None]
    arrays = dict(baseline=(1-cw)*support, core=np.zeros((n, width)), core_weight=cw,
        state_calm10=calm, state_weak_narrow=weak, extension_available=np.ones((n, width), bool),
        forecast_available=np.ones((n, width), bool), eligible=np.ones((n, width), bool),
        rank_log_excess=np.full((n, width), .2))
    arrays['extension_available'][1, 0] = False
    arrays['forecast_available'][1, 1] = False
    arrays['eligible'][1, 2] = False
    arrays['rank_log_excess'][1, 3] = np.nan
    arrays['eligible'][2, 0] = False
    arrays['extension_available'][7] = False
    arrays['rank_log_excess'][7, 1:3] = [np.inf, -np.inf]
    arrays['unused_frozen_model'] = np.full((n, width), 'not consumed', dtype=object)
    return arrays


def deltas(values):
    return {arm: values[key].copy() for arm, key in oracle.DELTA_KEYS.items()}


def pure_functions(path, names, namespace):
    nodes = [x for x in ast.parse(path.read_text()).body
             if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Tests(unittest.TestCase):
    def test_hand_count_ranks_and_tied_endpoint_affine(self):
        q, a, levels = oracle.row_targets([-.4, -.4, 0., .4, .4])
        self.assertEqual(levels, 3)
        self.assertEqual(q, [-.375, -.375, 0., .375, .375])
        self.assertEqual(a, [-.375, -.375, -.375+(0.-(-.4))*(.375-(-.375))/(.4-(-.4)), .375, .375])
        q, a, _ = oracle.row_targets([-.4, -.1, 0., .02, .3])
        self.assertEqual(q, [-.5, -.25, 0., .25, .5])
        self.assertEqual(a[0], q[0]); self.assertEqual(a[-1], q[-1])
        self.assertNotEqual(a[2], q[2])
        self.assertAlmostEqual(sum(q), 0., places=16)

    def test_two_levels_and_singleton_are_exact(self):
        for s in ([-.3, -.3, .2, .2, .2], [0., .2], [.2, 0., 0., 0.]):
            q, a, levels = oracle.row_targets(s)
            self.assertEqual(levels, 2)
            self.assertEqual(np.asarray(q).tobytes(), np.asarray(a).tobytes())
        for s in ([.2], [.125] * 49):
            q, a, levels = oracle.row_targets(s)
            self.assertEqual(levels, 1); self.assertEqual(q, [0.] * len(s)); self.assertEqual(a, q)

    def test_exact_near_ties_are_not_merged(self):
        q, a, levels = oracle.row_targets([0., np.nextafter(0., 1.), .2])
        self.assertEqual(levels, 3)
        self.assertEqual(q, [-.5, 0., .5])
        self.assertNotEqual(a, q)

    def test_full_fixture_audit_and_no_model_claim(self):
        arrays = fixture()
        values, audit = oracle.independent_features(arrays)
        receipt = oracle.verify_features(arrays, values, audit)
        self.assertTrue(all(receipt[key] for key in oracle.FEATURE_FLAGS))
        self.assertEqual(receipt['checked_cells'], 40)
        self.assertEqual((audit['constant_rows'], audit['two_level_rows']), (1, 1))
        self.assertFalse(receipt['computes_returns'])
        self.assertEqual(receipt['predictive_model_fits'], 0)
        proof = oracle.verify_deltas(values, arrays, deltas(values))
        self.assertTrue(all(proof[key] for key in oracle.DELTA_FLAGS))
        self.assertLessEqual(proof['observed_max_absolute_delta'], .5)

    def test_constant_nonzero_support_active_heterogeneity_is_noop(self):
        arrays = fixture()
        values, audit = oracle.independent_features(arrays)
        self.assertTrue(audit['event_history'][2]['constant'])
        self.assertGreater(np.ptp(values['weight'][2]), 0.)
        for key in ('candidate', 'control', 'candidate_delta', 'control_delta'):
            np.testing.assert_array_equal(values[key][2], np.zeros(5))
        for key in ('candidate_delta', 'control_delta'):
            np.testing.assert_array_equal(values[key][0], np.zeros(5))
        self.assertGreater(np.ptp(values['candidate'][0]), 0.)

    def test_rank_domain_and_delta_not_eligibility_filtered(self):
        arrays = fixture()
        values, _ = oracle.independent_features(arrays)
        arrays['eligible'][4] = False
        other, _ = oracle.independent_features(arrays)
        for key in oracle.VALUE_KEYS:
            np.testing.assert_array_equal(other[key][4], values[key][4])
        self.assertTrue(np.any(other['candidate_delta'][4] != 0.))
        arrays['baseline'][4, 0] = .48 * .49
        changed, _ = oracle.independent_features(arrays)
        self.assertNotEqual(changed['candidate'][4, 1], other['candidate'][4, 1])

    def test_all_inherited_active_conditions_and_no_extra_regime(self):
        for key in ('extension_available', 'forecast_available', 'eligible'):
            arrays = fixture()
            baseline, _ = oracle.independent_features(arrays)
            arrays[key][1, 4] = False
            changed, _ = oracle.independent_features(arrays)
            self.assertEqual(baseline['weight'][1, 4], .5-(.025+.01))
            self.assertEqual(changed['weight'][1, 4], .5-.025)
        for invalid in (np.nan, np.inf, -np.inf):
            arrays = fixture(); arrays['rank_log_excess'][1, 4] = invalid
            values, _ = oracle.independent_features(arrays)
            self.assertEqual(values['weight'][1, 4], .475)
        arrays = fixture(); values, _ = oracle.independent_features(arrays)
        self.assertTrue(np.all(values['weight'][3] == .5-(.025+.05)))
        self.assertTrue(np.all(values['weight'][5] == .48-.05))

    def test_original_functions_prove_effective_weight_and_full_remap(self):
        arrays = fixture(); arrays['residual'] = np.full((8, 5), .12)
        arrays['old_ml'] = np.full((8, 5), -.15)
        root = oracle.ROOT
        state = pure_functions(root/'outputs/monthly_state_rank_research/state_rank.py', ['score'], {'np': np})
        adapter = pure_functions(root/'outputs/monthly_path70_research/incumbent_adapter.py',
            ['_matrix', '_state'], {'np': np})
        policy = [dict(key='residual', state='calm10', amount=.025),
                  dict(key='old_ml', state='weak_narrow', amount=.05)]
        scope_spec = dict(id=3, base='incumbent67', scope='calm_nonweak', budget=.01)
        upstream = SimpleNamespace(configs=lambda: [{}] * 6,
            modified_pre_score=lambda spec, a, n: state['score'](policy, a, n))
        scope = pure_functions(root/'outputs/monthly_model_scope70/study.py', ['modified_pre_score'],
            {'np': np, 'configs': lambda: [scope_spec], 'adapter': SimpleNamespace(**adapter), 'upstream': upstream})
        before = scope['modified_pre_score'](scope_spec, arrays, 8)
        bumped = copy.deepcopy(arrays); bumped['baseline'] += (1-arrays['core_weight']) * .125
        after = scope['modified_pre_score'](scope_spec, bumped, 8)
        values, audit = oracle.independent_features(arrays)
        np.testing.assert_allclose((after-before)/.125, values['weight'], rtol=0., atol=2e-15)
        for target in ('candidate', 'control'):
            changed = copy.deepcopy(arrays)
            changed['baseline'] = arrays['core'] + (1-arrays['core_weight']) * values[target]
            replacement = scope['modified_pre_score'](scope_spec, changed, 8)
            for i, row in enumerate(audit['event_history']):
                if i and not row['constant']:
                    np.testing.assert_allclose(replacement[i], before[i]+values[target+'_delta'][i], rtol=0., atol=2e-16)

    def test_weight_three_shapes_and_empty_rows(self):
        arrays = fixture(); expected, _ = oracle.independent_features(arrays)
        for weight in (arrays['core_weight'][:, 0],
                       np.broadcast_to(arrays['core_weight'], arrays['baseline'].shape).copy()):
            other = copy.deepcopy(arrays); other['core_weight'] = weight
            values, _ = oracle.independent_features(other)
            for key in values: np.testing.assert_array_equal(values[key], expected[key])
        empty = {k: v[:0] for k, v in arrays.items()}
        values, audit = oracle.independent_features(empty)
        self.assertEqual(values['support'].shape, (0, 5)); self.assertEqual(audit['checked_cells'], 0)
        oracle.verify_features(empty, values, audit)
        oracle.verify_deltas(values, empty, deltas(values))

    def test_prefixes_physical_truncation_and_future_poison(self):
        arrays = fixture(); expected, audit = oracle.independent_features(arrays)
        for n in range(9):
            for a, limit in ((arrays, n), ({k: v[:n] for k, v in arrays.items()}, None)):
                values, aa = oracle.independent_features(a, limit)
                for key in values: np.testing.assert_array_equal(values[key], expected[key][:n])
                self.assertEqual(aa['event_history'], audit['event_history'][:n])
                oracle.verify_features(a, values, aa, limit)
                oracle.verify_deltas(values, a, deltas(values), limit)
        for n in (0, 3):
            poison = {k: v.astype(object) for k, v in arrays.items()}
            for v in poison.values(): v[n:] = 'DO NOT READ FUTURE'
            values, _ = oracle.independent_features(poison, n)
            for key in values: np.testing.assert_array_equal(values[key], expected[key][:n])

    def test_inputs_are_not_mutated(self):
        arrays = fixture(); before = copy.deepcopy(arrays)
        values, audit = oracle.independent_features(arrays)
        oracle.verify_features(arrays, values, audit)
        oracle.verify_deltas(values, arrays, deltas(values))
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], before[key])
        values['support'][:] = 99.
        for key in arrays: np.testing.assert_array_equal(arrays[key], before[key])

    def test_boolean_numeric_shape_prefix_and_state_errors(self):
        for key, value in (('baseline', True), ('core', '0'), ('core_weight', .51),
                           ('rank_log_excess', 1+0j), ('state_calm10', 1), ('eligible', 1)):
            a = fixture(); a[key] = a[key].astype(object); a[key].flat[1] = value
            with self.assertRaises(ValueError): oracle.independent_features(a)
        for key in ('core', 'core_weight', 'state_calm10', 'forecast_available'):
            a = fixture(); a[key] = a[key][:-1]
            with self.assertRaises(ValueError): oracle.independent_features(a)
        for n in (-1, 9, True, 1.5):
            with self.assertRaises(ValueError): oracle.independent_features(fixture(), n)
        a = fixture(); a['core_weight'] = np.broadcast_to(a['core_weight'], (8, 5)).copy()
        a['core_weight'][1, 4] = .52
        with self.assertRaises(ValueError): oracle.independent_features(a)

    def test_nonfinite_arithmetic_and_support_bound_not_clipping(self):
        for key in ('baseline', 'core', 'core_weight'):
            for value in (np.nan, np.inf, -np.inf):
                a = fixture(); a[key].flat[3] = value
                with self.assertRaises(ValueError): oracle.independent_features(a)
        a = fixture(); a['baseline'][1, 0] = 1e308; a['core'][1, 0] = -1e308
        with self.assertRaises(FloatingPointError): oracle.independent_features(a)
        a = fixture(); a['baseline'][4, 0] = .48*(.5+5e-13)
        values, _ = oracle.independent_features(a)
        self.assertGreater(values['support'][4, 0], .5)
        a['baseline'][4, 0] = .48*(.5+2e-12)
        with self.assertRaises(ValueError): oracle.independent_features(a)

    def test_artifact_and_delta_single_ulp_corruption_rejected(self):
        arrays = fixture(); values, audit = oracle.independent_features(arrays)
        for key in oracle.VALUE_KEYS:
            bad = copy.deepcopy(values); bad[key][4, 1] = np.nextafter(bad[key][4, 1], np.inf)
            with self.assertRaises(ValueError): oracle.verify_features(arrays, bad, audit)
        for path in ('top', 'row'):
            bad = copy.deepcopy(audit)
            if path == 'top': bad['active_forecast_cells'] += 1
            else: bad['event_history'][1]['distinct_levels'] += 1
            with self.assertRaises(ValueError): oracle.verify_features(arrays, values, bad)
        bad = deltas(values); bad['aux_rank_full'][0, 0] = 1e-16
        with self.assertRaises(ValueError): oracle.verify_deltas(values, arrays, bad)
        bad = deltas(values); bad['aux_affine_control'][4, 1] *= -1
        with self.assertRaises(ValueError): oracle.verify_deltas(values, arrays, bad)

    def test_accounting_adapter_pins_and_arm_isolation(self):
        helper = oracle.frozen_helper()
        self.assertNotEqual(helper.ARMS, oracle.ARMS)
        account = oracle.accounting()
        self.assertTrue(callable(account.metrics))
        fake = SimpleNamespace(ARMS=('old',), audit_outcomes=lambda *args: dict(
            checked_paths=9, checked_slices=45, paths={arm: {} for arm in oracle.ARMS}))
        with patch.object(oracle, 'frozen_helper', return_value=fake):
            proof = oracle.audit_outcomes(None, None, None, None, None, None)
        self.assertEqual(proof['outcome_adapter_sha256'], oracle.FROZEN_HELPER_SHA)
        self.assertNotEqual(oracle.frozen_helper().ARMS, oracle.ARMS)
        with patch.object(oracle, 'FROZEN_HELPER_SHA', '0'*64):
            with self.assertRaises(ValueError): oracle.accounting()

    def test_both_producer_and_independent_fixtures_all_prefix_audits(self):
        import monthly_aux_rank204_kernel as production
        import test_monthly_aux_rank204_kernel as producer_test
        for arrays in (fixture(), producer_test.fixture()):
            expected, _ = oracle.independent_features(arrays)
            for n in range(len(arrays['baseline']) + 1):
                actual, audit = production.build_features(arrays, n)
                oracle.verify_features(arrays, actual, audit, n)
                oracle.verify_deltas(actual, arrays, deltas(actual), n)
                for key in expected: np.testing.assert_array_equal(actual[key], expected[key][:n])


if __name__ == '__main__':
    unittest.main(verbosity=2)
