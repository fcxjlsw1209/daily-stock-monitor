"""Offline invented scores only; no source loader, actual cache or performance."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import monthly_reference_gain250_adapter as adapter


def fixture():
    n, width = 4, 6
    calm = np.array([False, True, False, True])
    weak = np.array([False, False, True, True])
    cw = np.where(calm, .50, .52)
    core = np.tile([.24, .22, .10, .09, .11, .12], (n, 1))
    support = np.tile([-.3, .25, 0., .05, -.1, .2], (n, 1))
    baseline = core + (1. - cw[:, None]) * support
    pre = np.tile([.4, .3, .2, .195, .19, .189], (n, 1))
    effects = np.tile([0., 0., -.1, .3, .5, .4], (n, 1))
    targets = np.tile([.5, -.5, .3, -.2, .1, 0.], (n, 1))
    return dict(pre_score=pre, core=core, core_weight=cw, baseline=baseline,
                calm=calm, weak=weak, eligible=np.ones((n, width), bool),
                symbols=[list('ABCDEF') for _ in range(n)], effects=effects,
                target=targets, active=np.ones(n, bool))


def prefix(args, n):
    return {k:deepcopy(v[:n]) for k,v in args.items()}


def original(args):
    n, width = args['pre_score'].shape
    weight = np.asarray(args['core_weight'])
    if weight.ndim == 1:weight = weight[:, None]
    weight = np.broadcast_to(weight, (n, width)).copy()
    source = dict(baseline=args['baseline'].copy(), core=args['core'].copy(),
                  core_weight=weight, state_calm10=args['calm'].copy(),
                  state_weak_narrow=args['weak'].copy())
    return adapter.ep.select_boundary(args['pre_score'].copy(), args['eligible'].copy(),
        deepcopy(args['symbols']), args['effects'].copy(), source,
        dict(budget=.05, ceiling=5, gap=.02))


def assert_same_inputs(case, left, right):
    case.assertEqual(set(left), set(right))
    for key in left:
        if isinstance(left[key], np.ndarray):
            case.assertEqual(left[key].shape, right[key].shape, key)
            case.assertEqual(left[key].dtype, right[key].dtype, key)
            case.assertEqual(left[key].tobytes(), right[key].tobytes(), key)
        else:case.assertEqual(left[key], right[key], key)


class AdapterTests(unittest.TestCase):
    def test_schema_one_original_call_and_plain_audit(self):
        args = fixture()
        with patch.object(adapter.ep, 'select_boundary', wraps=adapter.ep.select_boundary) as hook:
            ch, final, pre, audit = adapter.score_and_choose(**args)
        self.assertEqual(hook.call_count, 1)
        self.assertEqual(tuple(audit['counts']), adapter.COUNT_FIELDS)
        self.assertTrue(all(tuple(e) == adapter.EVENT_KEYS for e in audit['events']))
        self.assertEqual(audit['counts']['original_episode_calls'], 1)
        self.assertEqual(audit['counts']['consumed_target_cells'], 24)
        self.assertEqual(audit['counts']['new_model_fits'], 0)
        self.assertTrue(audit['source_free']);self.assertFalse(audit['actual_data_loaded'])
        self.assertEqual(pre.shape, (4, 6));self.assertEqual(final.shape, (4, 6))
        self.assertEqual(len(ch), 4);json.dumps(audit, allow_nan=False)

    def test_inactive_all_four_states_exact_original_scalar_policy(self):
        args = fixture();args['active'][:] = False;args['target'][:] = np.nan
        expected_ch, expected_final = original(args)
        ch, final, pre, audit = adapter.score_and_choose(**args)
        self.assertEqual(ch, expected_ch)
        self.assertEqual(final.tobytes(), expected_final.tobytes())
        self.assertEqual(pre.tobytes(), args['pre_score'].tobytes())
        for row in audit['events']:
            self.assertIsNone(row['target']);self.assertFalse(row['target_consumed'])
            self.assertEqual(row['requested_episode_budget'], .05)
            self.assertEqual(row['alpha'], 1.)

    def test_mixed_active_inactive_exact_bypass(self):
        args = fixture();args['active'] = np.array([True, False, True, False])
        args['target'][~args['active']] = np.nan
        old_ch, old_final = original(args)
        ch, final, pre, audit = adapter.score_and_choose(**args)
        for i in [1, 3]:
            self.assertEqual(ch[i], old_ch[i])
            self.assertEqual(final[i].tobytes(), old_final[i].tobytes())
            self.assertEqual(pre[i].tobytes(), args['pre_score'][i].tobytes())
        self.assertEqual(audit['counts']['consumed_target_cells'], 12)

    def test_canonical_pre_addition_and_all_auxiliary_ml_shrinkage(self):
        args = fixture();before = deepcopy(args)
        ch, final, pre, audit = adapter.score_and_choose(**args)
        coefficient = .02 / (1. - args['core_weight'])
        delta = .02 * args['target'] - coefficient[:, None] * (args['pre_score'] - args['core'])
        expected = args['pre_score'].copy();nz = delta != 0.;expected[nz] += delta[nz]
        self.assertEqual(pre.tobytes(), expected.tobytes())
        wrong_baseline_only = args['pre_score'] + .02*args['target'] - coefficient[:, None]*(args['baseline']-args['core'])
        self.assertFalse(np.array_equal(pre, wrong_baseline_only))
        for i,row in enumerate(audit['events']):self.assertEqual(row['correction'], delta[i].tolist())
        assert_same_inputs(self, before, args)

    def test_episode_funding_exact_order_all_states_and_row_vector(self):
        args = fixture();e0 = np.minimum(.05, np.maximum(0., .10-(.025*args['calm']+.05*args['weak'])))
        alpha = 1. - .02/(1.-args['core_weight'])
        with patch.object(adapter.ep, 'select_boundary', wraps=adapter.ep.select_boundary) as hook:
            _, _, _, audit = adapter.score_and_choose(**args)
        policy = hook.call_args.args[-1]
        self.assertEqual(policy['budget'].shape, (4,))
        self.assertEqual(policy['budget'].tobytes(), (alpha*e0).tobytes())
        self.assertEqual((policy['ceiling'], policy['gap']), (5, .02))
        self.assertEqual([x['original_episode_budget'] for x in audit['events']], e0.tolist())
        self.assertEqual([x['effective_episode_budget'] for x in audit['events']], (alpha*e0).tolist())
        np.testing.assert_allclose(args['core_weight'] + alpha*(1.-args['core_weight'])+.02, 1., atol=2e-16, rtol=0.)

    def test_original_support_and_effects_are_passed_unchanged(self):
        args = fixture()
        with patch.object(adapter.ep, 'select_boundary', wraps=adapter.ep.select_boundary) as hook:
            adapter.score_and_choose(**args)
        supplied = hook.call_args.args
        self.assertEqual(supplied[1].tobytes(), args['eligible'].tobytes())
        self.assertEqual(supplied[2], args['symbols'])
        self.assertEqual(supplied[3].tobytes(), args['effects'].tobytes())
        self.assertEqual(supplied[4]['baseline'].tobytes(), args['baseline'].tobytes())
        self.assertEqual(supplied[4]['core'].tobytes(), args['core'].tobytes())
        self.assertEqual(supplied[4]['core_weight'].shape, (4, 6))

    def test_core_weight_three_shapes_identical(self):
        args = fixture();expected = adapter.score_and_choose(**args)
        for cw in [args['core_weight'][:, None], np.broadcast_to(args['core_weight'][:, None], (4, 6)).copy()]:
            new = deepcopy(args);new['core_weight'] = cw
            actual = adapter.score_and_choose(**new)
            self.assertEqual(expected[0], actual[0]);self.assertEqual(expected[3], actual[3])
            self.assertEqual(expected[1].tobytes(), actual[1].tobytes())
            self.assertEqual(expected[2].tobytes(), actual[2].tobytes())

    def test_row_weight_broadcast_when_events_equal_width(self):
        args = fixture()
        for key in ['pre_score','core','baseline','eligible','effects','target']:args[key] = args[key][:, :4]
        args['symbols'] = [row[:4] for row in args['symbols']]
        with patch.object(adapter.ep, 'select_boundary', wraps=adapter.ep.select_boundary) as hook:
            adapter.score_and_choose(**args)
        actual = hook.call_args.args[4]['core_weight']
        np.testing.assert_array_equal(actual, np.broadcast_to(args['core_weight'][:, None], (4, 4)))
        self.assertFalse(np.array_equal(actual, np.broadcast_to(args['core_weight'], (4, 4))))

    def test_empty_prefix_retains_width_and_calls_once(self):
        args = prefix(fixture(), 0)
        with patch.object(adapter.ep, 'select_boundary', wraps=adapter.ep.select_boundary) as hook:
            ch, final, pre, audit = adapter.score_and_choose(**args)
        self.assertEqual(ch, []);self.assertEqual(final.shape, (0, 6));self.assertEqual(pre.shape, (0, 6))
        self.assertEqual(hook.call_count, 1);self.assertEqual(hook.call_args.args[-1]['budget'].shape, (0,))
        self.assertEqual(audit['counts']['symbols_per_event'], 6);self.assertEqual(audit['events'], [])

    def test_every_physical_prefix_exact(self):
        args = fixture();args['active'] = np.array([False, True, False, True])
        full = adapter.score_and_choose(**args)
        for n in range(5):
            cut = adapter.score_and_choose(**prefix(args, n))
            self.assertEqual(cut[0], full[0][:n]);self.assertEqual(cut[3]['events'], full[3]['events'][:n])
            self.assertEqual(cut[1].tobytes(), full[1][:n].tobytes());self.assertEqual(cut[2].tobytes(), full[2][:n].tobytes())

    def test_inactive_target_not_converted_even_poison_object(self):
        class Poison:
            def __float__(self):raise AssertionError('inactive target read')
        args = fixture();args['active'] = np.array([True, False, True, False])
        args['target'] = args['target'].astype(object)
        args['target'][1] = Poison();args['target'][3] = 'not numeric'
        _, _, _, audit = adapter.score_and_choose(**args)
        self.assertIsNone(audit['events'][1]['target']);self.assertIsNone(audit['events'][3]['target'])

    def test_inactive_nested_list_float_subclass_is_not_converted(self):
        class PoisonFloat(float):
            def __float__(self):raise AssertionError('inactive nested target conversion')
        args = fixture();args['active'][1] = False
        args['target'] = args['target'].tolist();args['target'][1][0] = PoisonFloat(.2)
        _, _, _, audit = adapter.score_and_choose(**args)
        self.assertIsNone(audit['events'][1]['target'])
        self.assertFalse(audit['events'][1]['target_consumed'])

    def test_active_zero_target_still_funds(self):
        args = fixture();args['target'][:] = 0.
        _, _, pre, audit = adapter.score_and_choose(**args)
        self.assertFalse(np.array_equal(pre, args['pre_score']))
        self.assertGreater(audit['counts']['nonzero_correction_cells'], 0)
        self.assertTrue(audit['constant_target_is_not_neutral'])

    def test_constant_all_tied_rank_target_not_noop(self):
        args = fixture();args['target'][:] = 1./98.
        _, _, pre, audit = adapter.score_and_choose(**args)
        self.assertFalse(np.array_equal(pre, args['pre_score']))
        self.assertTrue(all(row['funding_budget'] == .02 for row in audit['events']))

    def test_computed_zero_correction_skips_addition_preserves_signed_zero(self):
        args = prefix(fixture(), 1)
        args['pre_score'][:] = 0.;args['pre_score'][0, 0] = -0.
        args['core'][:] = 0.;args['baseline'][:] = 0.;args['effects'][:] = 0.;args['target'][:] = 0.
        _, _, pre, audit = adapter.score_and_choose(**args)
        self.assertEqual(pre.tobytes(), args['pre_score'].tobytes())
        self.assertEqual(audit['counts']['nonzero_correction_cells'], 0)

    def test_hook_is_rebuilt_from_funded_pre_only_once(self):
        args = prefix(fixture(), 1)
        args['pre_score'][0] = [.4, .3, .2, .199, .198, .197]
        args['core'][:] = 0.;args['baseline'][:] = 0.
        args['target'][0] = [0., 0., 0., 0., -.5, .5]
        args['effects'][0] = [0., 0., 0., 0., 0., .5]
        old_choices, old_final = original(args)
        ch, final, pre, audit = adapter.score_and_choose(**args)
        self.assertNotIn(5, [2, 3, 4]);self.assertIn(5, audit['events'][0]['boundary_indices'])
        self.assertNotIn(4, audit['events'][0]['boundary_indices']);self.assertIn(5, ch[0])
        self.assertNotEqual(ch, old_choices)
        changed_final_from_old_hook = pre + (1.-.02/.48)*(old_final-args['pre_score'])
        self.assertFalse(np.array_equal(final, changed_final_from_old_hook))

    def test_returned_choice_order_is_not_global_final_rerank(self):
        args = prefix(fixture(), 1);args['active'][:] = False
        args['pre_score'][0] = [.205, .204, .2, .199, .198, .1]
        args['core'][:] = 0.;args['baseline'][:] = 0.
        args['effects'][0] = [0., 0., .5, .5, .5, 0.]
        ch, final, _, _ = adapter.score_and_choose(**args)
        global_order = sorted(range(6), key=lambda j:(-final[0,j], args['symbols'][0][j]))[:3]
        self.assertNotEqual(ch[0], global_order);self.assertEqual(ch[0], [2, 0, 1])

    def test_symbol_ties_exact_original_order(self):
        args = prefix(fixture(), 1);args['active'][:] = False
        args['symbols'] = [list('FCBAED')];args['pre_score'][:] = 0.
        args['core'][:] = 0.;args['baseline'][:] = 0.;args['effects'][:] = 0.
        ch, _, _, _ = adapter.score_and_choose(**args)
        self.assertEqual(ch, [[3, 2, 1]])

    def test_zero_one_two_eligible_preserve_cash_half_rule(self):
        for count, want in [(0, []), (1, []), (2, [0, 1])]:
            args = prefix(fixture(), 1);args['eligible'][:] = False;args['eligible'][0, :count] = True
            ch, _, pre, audit = adapter.score_and_choose(**args)
            self.assertEqual(ch, [want]);self.assertEqual(audit['events'][0]['boundary_indices'], [])

    def test_eligibility_does_not_limit_target_funding_domain(self):
        args = prefix(fixture(), 1);args['eligible'][0, -1] = False
        _, _, pre, audit = adapter.score_and_choose(**args)
        self.assertNotEqual(pre[0, -1], args['pre_score'][0, -1])
        self.assertEqual(audit['counts']['consumed_target_cells'], 6)

    def test_source_arrays_and_names_do_not_mutate(self):
        args = fixture();before = deepcopy(args)
        for value in args.values():
            if isinstance(value, np.ndarray):value.setflags(write=False)
        adapter.score_and_choose(**args)
        assert_same_inputs(self, before, args)

    def test_invalid_active_targets_fail_before_episode(self):
        for value in [np.nan, np.inf, -np.inf, True, '0.1', .5000000000000001, -.5000000000000001]:
            with self.subTest(value=value):
                args = fixture();args['target'] = args['target'].astype(object);args['target'][0, 0] = value
                with patch.object(adapter.ep, 'select_boundary') as hook:
                    with self.assertRaises(ValueError):adapter.score_and_choose(**args)
                    hook.assert_not_called()

    def test_mixed_boolean_container_source_not_coerced_valid(self):
        args = fixture();args['pre_score'] = args['pre_score'].tolist();args['pre_score'][0][0] = True
        with self.assertRaisesRegex(ValueError, 'nonboolean'):adapter.score_and_choose(**args)
        args = fixture();args['target'] = args['target'].tolist();args['target'][0][0] = True
        with self.assertRaisesRegex(ValueError, 'nonboolean'):adapter.score_and_choose(**args)

    def test_bad_source_values_fail(self):
        for key in ['pre_score', 'core', 'baseline', 'effects']:
            for value in [True, '0.1', np.nan, np.inf, 10**1000]:
                with self.subTest(key=key, value=repr(value)[:40]):
                    args = fixture();args[key] = args[key].astype(object);args[key][0, 0] = value
                    with self.assertRaises(ValueError):adapter.score_and_choose(**args)

    def test_strict_boolean_row_masks_and_eligibility(self):
        for key in ['calm', 'weak', 'active', 'eligible']:
            for kind in ['integer', 'object', 'extra_axis']:
                args = fixture()
                if kind == 'integer':args[key] = args[key].astype(int)
                elif kind == 'object':args[key] = args[key].astype(object)
                else:args[key] = args[key][..., None]
                with self.assertRaises(ValueError):adapter.score_and_choose(**args)

    def test_shapes_symbols_and_weight_policy_fail(self):
        cases=[]
        a=fixture();a['target']=a['target'][:, :5];cases.append(a)
        a=fixture();a['effects']=a['effects'][:3];cases.append(a)
        a=fixture();a['symbols'][0][1]='A';cases.append(a)
        a=fixture();a['symbols'][0][1]=' B';cases.append(a)
        a=fixture();a['core_weight'][0]=.5;cases.append(a)
        a=fixture();a['core_weight']=np.broadcast_to(a['core_weight'][:,None],(4,6)).copy();a['core_weight'][0,1]=.5;cases.append(a)
        for args in cases:
            with self.assertRaises(ValueError):adapter.score_and_choose(**args)

    def test_support_and_core_bounds_are_validation_only(self):
        args = fixture();args['baseline'][0, 0] = args['core'][0, 0] + .48*.6
        with self.assertRaisesRegex(ValueError, 'support bounds'):adapter.score_and_choose(**args)
        args = fixture();args['core'][0, 0] = .4;args['baseline'][0, 0] = .4
        with self.assertRaisesRegex(ValueError, 'core bounds'):adapter.score_and_choose(**args)

    def test_nonfinite_derived_support_fails_as_arithmetic(self):
        args = fixture();args['baseline'][0, 0] = np.finfo(float).max
        with self.assertRaises(ArithmeticError):adapter.score_and_choose(**args)

    def test_overflow_inside_original_episode_propagates(self):
        args = prefix(fixture(), 1);args['active'][:] = False
        m=np.finfo(float).max;args['pre_score'][0] = [m,m,m,-m,-m,-m]
        with self.assertRaises(ArithmeticError):adapter.score_and_choose(**args)

    def test_nonfinite_episode_output_is_not_source_missing(self):
        args = fixture();bad=np.full((4,6),np.inf)
        with patch.object(adapter.ep,'select_boundary',return_value=([[]]*4,bad)):
            with self.assertRaises(ArithmeticError):adapter.score_and_choose(**args)

    def test_dependency_is_exact_frozen_episode(self):
        import hashlib
        self.assertEqual(adapter.dependency_files(), (adapter.EPISODE_PATH,))
        self.assertEqual(hashlib.sha256(adapter.EPISODE_PATH.read_bytes()).hexdigest(), adapter.EPISODE_SHA256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
