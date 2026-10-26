"""Synthetic-only independent group ranks, original funding and prefix checks."""
import ast
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import monthly_asset_aux210_independent as oracle


def fixture():
    calm = np.array([False, True, True, True, False, False, False, True])
    weak = np.array([False, False, False, True, False, True, False, False])
    support = np.array([
        [-.4, -.1, 0., .3, -.3, .01, .2],
        [-.4, -.4, .2, .2, -.4, 0., .3],
        [.125, .125, .125, .125, -.25, 0., .25],
        [-.4, -.1, .01, .3, .125, .125, .125],
        [-.4, -.1, .01, .3, -.25, -.25, .125],
        [-.4, -.2, -.1, 0., .015, .2, .4],
        [-.3, -.1, 0., .01, .05, .2, .3],
        [.2, -.4, -.1, 0., .01, .02, .3]])
    n, width = support.shape
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
    symbols = [['S0', 'S1', 'S2', 'S3', 'E0', 'E1', 'E2'] for _ in range(n)]
    symbols[5] = ['S' + str(j) for j in range(width)]
    symbols[6] = ['E' + str(j) for j in range(width)]
    symbols[7] = ['S0'] + ['E' + str(j) for j in range(width-1)]
    meta = dict(symbols=symbols, etfs=['E' + str(j) for j in range(width)] + ['ABSENT_ETF'])
    return arrays, meta


def deltas(values):
    return {arm: values[key].copy() for arm, key in oracle.DELTA_KEYS.items()}


def truncate(arrays, meta, n):
    return {k: v[:n] for k, v in arrays.items()}, dict(symbols=meta['symbols'][:n], etfs=meta['etfs'].copy())


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

    def test_two_levels_singleton_and_exact_near_ties(self):
        for s in ([-.3, -.3, .2, .2, .2], [0., .2], [.2, 0., 0., 0.]):
            q, a, levels = oracle.row_targets(s)
            self.assertEqual(levels, 2)
            self.assertEqual(np.asarray(q).tobytes(), np.asarray(a).tobytes())
        for s in ([.2], [.125] * 49):
            q, a, levels = oracle.row_targets(s)
            self.assertEqual(levels, 1); self.assertEqual(q, [0.] * len(s)); self.assertEqual(a, q)
        q, a, levels = oracle.row_targets([0., np.nextafter(0., 1.), .2])
        self.assertEqual(levels, 3); self.assertEqual(q, [-.5, 0., .5]); self.assertNotEqual(a, q)

    def test_complete_audit_counts_and_flags(self):
        arrays, meta = fixture(); values, audit = oracle.independent_features(arrays, meta)
        receipt = oracle.verify_features(arrays, meta, values, audit)
        self.assertTrue(all(receipt[key] for key in oracle.FEATURE_FLAGS))
        self.assertEqual(receipt['checked_cells'], 56)
        self.assertEqual((audit['rank_rows'], audit['grouped_rank_rows'], audit['empty_groups']), (14, 14, 2))
        self.assertEqual((audit['stock_cells'], audit['etf_cells']), (28, 28))
        self.assertEqual((audit['constant_rows'], audit['two_level_rows']), (3, 2))
        self.assertFalse(receipt['computes_returns']); self.assertEqual(receipt['new_model_fits'], 0)
        proof = oracle.verify_deltas(values, arrays, meta, deltas(values))
        self.assertTrue(all(proof[key] for key in oracle.DELTA_FLAGS))
        self.assertLessEqual(proof['observed_max_absolute_delta'], .5)

    def test_group_constant_and_singleton_preserve_nonzero_support(self):
        arrays, meta = fixture(); values, audit = oracle.independent_features(arrays, meta)
        self.assertGreater(np.ptp(values['weight'][2, :4]), 0.)
        for i, group in ((2, 0), (3, 1), (7, 0)):
            detail = audit['event_history'][i]['groups'][group]
            self.assertTrue(detail['constant'])
            for key in ('candidate', 'control', 'candidate_delta', 'control_delta'):
                np.testing.assert_array_equal(values[key][i, detail['indices']], 0.)
        self.assertTrue(np.any(values['candidate_delta'][2, 4:] != 0.))
        for key in ('candidate_delta', 'control_delta'):
            np.testing.assert_array_equal(values[key][0], 0.)
        self.assertGreater(np.ptp(values['candidate'][0]), 0.)

    def test_empty_groups_and_single_width(self):
        arrays, meta = fixture(); values, audit = oracle.independent_features(arrays, meta)
        for i, g in ((5, 1), (6, 0)):
            empty = audit['event_history'][i]['groups'][g]
            self.assertEqual(empty['size'], 0); self.assertFalse(empty['constant'])
            self.assertIsNone(empty['support_min']); self.assertIsNone(empty['candidate_max'])
            self.assertEqual(empty['indices'], []); self.assertEqual(empty['distinct_levels'], 0)
        one = {k: v[:, :1] if v.ndim == 2 else v.copy() for k, v in arrays.items()}
        om = dict(symbols=[[row[0]] for row in meta['symbols']], etfs=meta['etfs'])
        ov, oa = oracle.independent_features(one, om)
        self.assertEqual(oa['rank_rows'], 8); self.assertEqual(oa['empty_groups'], 8)
        self.assertEqual(oa['constant_rows'], 8)
        for key in ('candidate', 'control', 'candidate_delta', 'control_delta'):
            np.testing.assert_array_equal(ov[key], 0.)
        em = copy.deepcopy(meta); em['etfs'] = []
        _, ea = oracle.independent_features(arrays, em)
        self.assertEqual(ea['etf_cells'], 0); self.assertEqual(ea['rank_rows'], 8)

    def test_explicit_identity_and_within_group_independence(self):
        arrays, meta = fixture(); values, _ = oracle.independent_features(arrays, meta)
        # An ETF-like ticker is still a stock unless explicitly listed.
        renamed = copy.deepcopy(meta)
        renamed['symbols'][4][0] = 'ETF_LOOKING_NAME'
        other, _ = oracle.independent_features(arrays, renamed)
        for key in values: np.testing.assert_array_equal(other[key], values[key])
        changed = copy.deepcopy(arrays); changed['baseline'][4, 4:] = .48 * np.array([.4, -.1, -.2])
        result, _ = oracle.independent_features(changed, meta)
        for key in values: np.testing.assert_array_equal(result[key][4, :4], values[key][4, :4])
        reassigned = copy.deepcopy(meta); reassigned['etfs'].append('S0')
        result, _ = oracle.independent_features(arrays, reassigned)
        self.assertFalse(np.array_equal(result['candidate'][4], values['candidate'][4]))

    def test_symbol_column_permutation_equivariance(self):
        arrays, meta = fixture(); expected, _ = oracle.independent_features(arrays, meta)
        order = [5, 2, 0, 6, 1, 4, 3]
        permuted = {k: v[:, order] if v.ndim == 2 and v.shape[1] == 7 else v.copy() for k, v in arrays.items()}
        pm = dict(symbols=[[row[j] for j in order] for row in meta['symbols']], etfs=meta['etfs'])
        values, _ = oracle.independent_features(permuted, pm)
        for key in values: np.testing.assert_array_equal(values[key], expected[key][:, order])

    def test_rank_and_delta_not_eligibility_filtered(self):
        arrays, meta = fixture(); expected, _ = oracle.independent_features(arrays, meta)
        arrays['eligible'][4] = False
        values, _ = oracle.independent_features(arrays, meta)
        for key in values: np.testing.assert_array_equal(values[key][4], expected[key][4])
        self.assertTrue(np.any(values['candidate_delta'][4] != 0.))
        arrays['baseline'][4, 0] = .48 * .49
        changed, _ = oracle.independent_features(arrays, meta)
        self.assertNotEqual(changed['candidate'][4, 1], values['candidate'][4, 1])

    def test_inherited_funding_masks(self):
        for key in ('extension_available', 'forecast_available', 'eligible'):
            arrays, meta = fixture(); baseline, _ = oracle.independent_features(arrays, meta)
            arrays[key][1, 4] = False
            changed, _ = oracle.independent_features(arrays, meta)
            self.assertEqual(baseline['weight'][1, 4], .5-(.025+.01))
            self.assertEqual(changed['weight'][1, 4], .5-.025)
        for invalid in (np.nan, np.inf, -np.inf):
            arrays, meta = fixture(); arrays['rank_log_excess'][1, 4] = invalid
            values, _ = oracle.independent_features(arrays, meta)
            self.assertEqual(values['weight'][1, 4], .475)
        arrays, meta = fixture(); values, _ = oracle.independent_features(arrays, meta)
        self.assertTrue(np.all(values['weight'][3] == .5-(.025+.05)))
        self.assertTrue(np.all(values['weight'][5] == .48-.05))

    def test_original_scope_functions_prove_funding_and_full_remap(self):
        arrays, meta = fixture(); arrays['residual'] = np.full((8, 7), .12)
        arrays['old_ml'] = np.full((8, 7), -.15)
        root = oracle.ROOT
        state = pure_functions(root/'outputs/monthly_state_rank_research/state_rank.py', ['score'], {'np': np})
        adapter = pure_functions(root/'outputs/monthly_path70_research/incumbent_adapter.py',
            ['_matrix', '_state'], {'np': np})
        policy = [dict(key='residual', state='calm10', amount=.025),
                  dict(key='old_ml', state='weak_narrow', amount=.05)]
        spec = dict(id=3, base='incumbent67', scope='calm_nonweak', budget=.01)
        upstream = SimpleNamespace(configs=lambda: [{}] * 6,
            modified_pre_score=lambda spec, a, n: state['score'](policy, a, n))
        scope = pure_functions(root/'outputs/monthly_model_scope70/study.py', ['modified_pre_score'],
            {'np': np, 'configs': lambda: [spec], 'adapter': SimpleNamespace(**adapter), 'upstream': upstream})
        before = scope['modified_pre_score'](spec, arrays, 8)
        bumped = copy.deepcopy(arrays); bumped['baseline'] += (1-arrays['core_weight']) * .125
        after = scope['modified_pre_score'](spec, bumped, 8)
        values, audit = oracle.independent_features(arrays, meta)
        np.testing.assert_allclose((after-before)/.125, values['weight'], rtol=0., atol=2e-15)
        for target in ('candidate', 'control'):
            replacement_arrays = copy.deepcopy(arrays)
            effective_target = values[target].copy()
            effective_target[0] = values['support'][0]
            for i, row in enumerate(audit['event_history']):
                for group in row['groups']:
                    if group['constant']:
                        effective_target[i, group['indices']] = values['support'][i, group['indices']]
            replacement_arrays['baseline'] = arrays['core'] + (1-arrays['core_weight']) * effective_target
            replacement = scope['modified_pre_score'](spec, replacement_arrays, 8)
            np.testing.assert_allclose(replacement, before+values[target+'_delta'], rtol=0., atol=2e-16)

    def test_core_weight_shapes_and_zero_prefix(self):
        arrays, meta = fixture(); expected, _ = oracle.independent_features(arrays, meta)
        for cw in (arrays['core_weight'][:, 0], np.broadcast_to(arrays['core_weight'], (8, 7)).copy()):
            other = copy.deepcopy(arrays); other['core_weight'] = cw
            values, _ = oracle.independent_features(other, meta)
            for key in values: np.testing.assert_array_equal(values[key], expected[key])
        empty, em = truncate(arrays, meta, 0)
        values, audit = oracle.independent_features(empty, em)
        self.assertEqual(values['support'].shape, (0, 7)); self.assertEqual(audit['empty_groups'], 0)
        oracle.verify_features(empty, em, values, audit)
        oracle.verify_deltas(values, empty, em, deltas(values))

    def test_all_prefixes_and_unread_future_numeric_and_identity_poison(self):
        arrays, meta = fixture(); expected, audit = oracle.independent_features(arrays, meta)
        for n in range(9):
            ta, tm = truncate(arrays, meta, n)
            for a, m, limit in ((arrays, meta, n), (ta, tm, None)):
                values, aa = oracle.independent_features(a, m, limit)
                for key in values: np.testing.assert_array_equal(values[key], expected[key][:n])
                self.assertEqual(aa['event_history'], audit['event_history'][:n])
                oracle.verify_features(a, m, values, aa, limit)
                oracle.verify_deltas(values, a, m, deltas(values), limit)
        for n in (0, 3):
            poison = {k: v.astype(object) for k, v in arrays.items()}
            for v in poison.values(): v[n:] = 'DO NOT READ FUTURE'
            pm = copy.deepcopy(meta); pm['symbols'][n:] = [None] * (8-n)
            pm['dates'] = object()
            values, _ = oracle.independent_features(poison, pm, n)
            for key in values: np.testing.assert_array_equal(values[key], expected[key][:n])

    def test_inputs_and_metadata_unchanged_no_shared_output_storage(self):
        arrays, meta = fixture(); before = copy.deepcopy(arrays); mb = copy.deepcopy(meta)
        values, audit = oracle.independent_features(arrays, meta)
        oracle.verify_features(arrays, meta, values, audit)
        oracle.verify_deltas(values, arrays, meta, deltas(values))
        values['support'][:] = 99.; audit['event_history'][0]['symbols'][0] = 'changed'
        for key in arrays: np.testing.assert_array_equal(arrays[key], before[key])
        self.assertEqual(meta, mb)

    def test_metadata_missing_ambiguous_and_duplicate_identity_rejected(self):
        arrays, meta = fixture()
        bads = [None, {}, dict(symbols=meta['symbols']), dict(symbols=meta['symbols'], etfs=('E0',)),
                dict(symbols=meta['symbols'], etfs=['E0', 'E0']), dict(symbols=meta['symbols'], etfs=[True]),
                dict(symbols=meta['symbols'], etfs=['']), dict(symbols=meta['symbols'][:-1], etfs=[])]
        for bad in bads:
            with self.assertRaises(ValueError): oracle.independent_features(arrays, bad)
        for row in (['X']*7, ['']+meta['symbols'][0][1:], [True]+meta['symbols'][0][1:],
                    tuple(meta['symbols'][0]), meta['symbols'][0][:-1]):
            bad = copy.deepcopy(meta); bad['symbols'][0] = row
            with self.assertRaises(ValueError): oracle.independent_features(arrays, bad)

    def test_numeric_boolean_shape_and_state_errors(self):
        for key, value in (('baseline', True), ('core', '0'), ('core_weight', .51),
                           ('rank_log_excess', 1+0j), ('state_calm10', 1), ('eligible', 1)):
            a, m = fixture(); a[key] = a[key].astype(object); a[key].flat[1] = value
            with self.assertRaises(ValueError): oracle.independent_features(a, m)
        for key in ('core', 'core_weight', 'state_calm10', 'forecast_available'):
            a, m = fixture(); a[key] = a[key][:-1]
            with self.assertRaises(ValueError): oracle.independent_features(a, m)
        for n in (-1, 9, True, 1.5):
            with self.assertRaises(ValueError): oracle.independent_features(*fixture(), limit=n)
        a, m = fixture(); a['core_weight'] = np.broadcast_to(a['core_weight'], (8, 7)).copy()
        a['core_weight'][1, 4] = .52
        with self.assertRaises(ValueError): oracle.independent_features(a, m)

    def test_nonfinite_arithmetic_and_support_tolerance_not_clipping(self):
        for key in ('baseline', 'core', 'core_weight'):
            for value in (np.nan, np.inf, -np.inf):
                a, m = fixture(); a[key].flat[3] = value
                with self.assertRaises(ValueError): oracle.independent_features(a, m)
        a, m = fixture(); a['baseline'][1, 0] = 1e308; a['core'][1, 0] = -1e308
        with self.assertRaises(FloatingPointError): oracle.independent_features(a, m)
        a, m = fixture(); a['baseline'][4, 0] = .48*(.5+5e-13)
        values, _ = oracle.independent_features(a, m)
        self.assertGreater(values['support'][4, 0], .5)
        a['baseline'][4, 0] = .48*(.5+2e-12)
        with self.assertRaises(ValueError): oracle.independent_features(a, m)

    def test_each_feature_ulp_and_group_audit_corruption_rejected(self):
        arrays, meta = fixture(); values, audit = oracle.independent_features(arrays, meta)
        for key in oracle.VALUE_KEYS:
            bad = copy.deepcopy(values); bad[key][4, 1] = np.nextafter(bad[key][4, 1], np.inf)
            with self.assertRaises(ValueError): oracle.verify_features(arrays, meta, bad, audit)
        for key in oracle.COUNT_FIELDS:
            bad = copy.deepcopy(audit); bad[key] += 1
            with self.assertRaises(ValueError): oracle.verify_features(arrays, meta, values, bad)
        bad = copy.deepcopy(audit); bad['event_history'][1]['groups'][0]['asset_type'] = 'etf'
        with self.assertRaises(ValueError): oracle.verify_features(arrays, meta, values, bad)
        bad = copy.deepcopy(audit); bad['event_history'][5]['groups'][1]['support_min'] = 0.
        with self.assertRaises(ValueError): oracle.verify_features(arrays, meta, values, bad)
        for arm in oracle.DELTA_KEYS:
            bad = deltas(values); bad[arm][0, 0] = 1e-16
            with self.assertRaises(ValueError): oracle.verify_deltas(values, arrays, meta, bad)
            bad = deltas(values); bad[arm][2, 0] = .1
            with self.assertRaises(ValueError): oracle.verify_deltas(values, arrays, meta, bad)

    def test_accounting_pin_and_arm_isolation(self):
        self.assertNotEqual(oracle.frozen_helper().ARMS, oracle.ARMS)
        self.assertTrue(callable(oracle.accounting().metrics))
        fake = SimpleNamespace(ARMS=('old',), audit_outcomes=lambda *args: dict(
            checked_paths=9, checked_slices=45, paths={arm: {} for arm in oracle.ARMS}))
        with patch.object(oracle, 'frozen_helper', return_value=fake):
            proof = oracle.audit_outcomes(None, None, None, None, None, None)
        self.assertEqual(proof['outcome_adapter_sha256'], oracle.FROZEN_HELPER_SHA)
        self.assertNotEqual(oracle.frozen_helper().ARMS, oracle.ARMS)
        with patch.object(oracle, 'FROZEN_HELPER_SHA', '0'*64):
            with self.assertRaises(ValueError): oracle.accounting()

    def test_both_fixtures_full_audits_and_all_physical_prefixes(self):
        import monthly_asset_aux210_kernel as production
        import test_monthly_asset_aux210_kernel as producer_test
        for arrays, meta in (fixture(), producer_test.fixture()):
            expected, _ = oracle.independent_features(arrays, meta)
            for n in range(len(arrays['baseline']) + 1):
                for a, m, limit in ((arrays, meta, n), (*truncate(arrays, meta, n), None)):
                    actual, audit = production.build_features(a, m, limit)
                    oracle.verify_features(a, m, actual, audit, limit)
                    oracle.verify_deltas(actual, a, m, deltas(actual), limit)
                    for key in expected: np.testing.assert_array_equal(actual[key], expected[key][:n])


if __name__ == '__main__':
    unittest.main(verbosity=2)
