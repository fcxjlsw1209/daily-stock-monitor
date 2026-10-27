"""Synthetic selector tests; no data cache, prices, fitted models or returns."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

import numpy as np
import monthly_selector_scope268_kernel as k


OLD = Path(__file__).resolve().parents[1] / 'outputs/monthly_daily_episodes/episodes.py'
OLD_SHA = '8ae91e55e0d662cb6dee9cba6d1422aa3713b779976de69b4560194d311ae16a'


def old_selector():
    """Compile only the pinned pure function; never import its research driver."""
    text = OLD.read_bytes()
    if hashlib.sha256(text).hexdigest() != OLD_SHA:
        raise AssertionError('Frozen selector source changed')
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == 'select_boundary')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OLD), 'exec'), namespace)
    return namespace['select_boundary']


def fixture(n=4, width=6):
    p = np.tile(np.array([.20, .19, .18, .175, .169, .16])[:width], (n, 1))
    cw = np.full((n, width), .52)
    core = cw * np.linspace(.4, -.4, width)
    support = np.tile(np.linspace(-.3, .3, width), (n, 1))
    return dict(pre=p, eligible=np.ones((n, width), bool),
        symbols=[[chr(65+j) for j in range(width)] for _ in range(n)],
        effects=np.tile(np.linspace(.25, -.25, width), (n, 1)),
        baseline=core+(1-cw)*support, core=core, core_weight=cw,
        state_calm10=np.resize(np.array([False, True, False, True]), n),
        state_weak_narrow=np.resize(np.array([False, False, True, True]), n),
        dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'][:n])


def call(a, mode=k.ARMS[0]):
    return k.select(**a, mode=mode)


def prefix(a, n):
    return {key: value[:n] for key, value in a.items()}


def old_call(a):
    return old_selector()(a['pre'], a['eligible'], a['symbols'], a['effects'],
        {key: a[key] for key in ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow')}, k.POLICY)


def equal_array(test, got, want):
    test.assertEqual(got.dtype, want.dtype)
    test.assertEqual(got.shape, want.shape)
    test.assertEqual(got.tobytes(), want.tobytes())


class CanonicalTests(unittest.TestCase):
    def test_incumbent_exact_pinned_function(self):
        a = fixture(); got = call(a); want = old_call(a)
        self.assertEqual(got[0], want[0]); equal_array(self, got[1], want[1])

    def test_random_synthetic_incumbent_exact(self):
        rng = np.random.default_rng(268)
        for _ in range(40):
            a = fixture()
            a['pre'] = rng.uniform(-.5, .5, (4, 6))
            a['eligible'] = rng.random((4, 6)) > .25
            a['effects'] = rng.uniform(-1, 1, (4, 6))
            a['core_weight'] = rng.choice([.5, .52], (4, 6))
            a['core'] = a['core_weight'] * rng.uniform(-.5, .5, (4, 6))
            a['baseline'] = a['core'] + (1-a['core_weight'])*rng.uniform(-.5, .5, (4, 6))
            got, want = call(a), old_call(a)
            self.assertEqual(got[0], want[0]); equal_array(self, got[1], want[1])

    def test_top_two_can_exit_in_new_scope(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        a['pre'][0] = [.10, .09, .08, .079, .078, .077]
        a['core'][:] = 0.; a['baseline'][:] = 0.
        a['effects'][0] = [-.5, -.5, .5, .5, .5, .5]
        self.assertEqual(call(a)[0], [[2, 0, 1]])
        self.assertEqual(call(a, k.ARMS[1])[0], [[2, 3, 4]])
        self.assertEqual(call(a, k.ARMS[2])[0], [[2, 3, 4]])
        self.assertEqual(call(a, k.ARMS[1])[2]['events'][0]['top2_exited_indices'], [0, 1])

    def test_all_can_take_rank_six_but_top5_gap_cannot(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        a['pre'][0] = [.10, .099, .098, .097, .096, .095]
        a['core'][:] = 0.; a['baseline'][:] = 0.; a['effects'][:] = -.5
        a['effects'][0, 5] = .5
        self.assertNotIn(5, call(a, k.ARMS[1])[0][0])
        self.assertIn(5, call(a, k.ARMS[2])[0][0])

    def test_original_gap_precedes_adjustment(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        a['pre'][0] = [.10, .099, .098, .05, .04, .03]
        a['effects'][0, 3:] = 100.
        self.assertEqual(call(a, k.ARMS[1])[2]['events'][0]['adjustment_indices'], [0, 1, 2])

    def test_gap_threshold_no_epsilon(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        a['pre'][0] = [.3, .2, 0., -.02, -np.nextafter(.02, np.inf), -.1]
        event = call(a, k.ARMS[1])[2]['events'][0]
        self.assertEqual(event['original_boundary_indices'], [2, 3])

    def test_symbol_ties_original_and_final(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        a['pre'][:] = .2; a['core'][:] = 0.; a['baseline'][:] = 0.; a['effects'][:] = 0.
        a['symbols'] = [['Z', 'Y', 'X', 'C', 'B', 'A']]
        for mode in k.ARMS: self.assertEqual(call(a, mode)[0], [[5, 4, 3]])

    def test_budget_same_four_states(self):
        got = call(fixture())
        expected = np.minimum(.05, np.maximum(0., .10 - (.025*np.array([False, True, False, True])+.05*np.array([False, False, True, True]))))
        self.assertEqual([e['episode_budget'] for e in got[2]['events']], expected.tolist())

    def test_partial_new_modes_exact_original(self):
        a = fixture(); a['effects'][0] = [-.5, -.5, .5, .5, .5, .5]
        want = call(a)
        for mode in k.ARMS[1:]:
            got = call(a, mode)
            self.assertEqual(got[0][0], want[0][0]); equal_array(self, got[1][:1], want[1][:1])
            self.assertEqual(got[2]['events'][0]['effective_mode'], k.ARMS[0])

    def test_partial_does_not_first_consume_unlocked_top_two(self):
        a = prefix(fixture(), 1)
        for key in ('effects', 'baseline', 'core', 'core_weight'): a[key][0, :2] = np.nan
        want = call(a)
        for mode in k.ARMS[1:]: equal_array(self, call(a, mode)[1], want[1])

    def test_zero_effect_and_bounded_support_has_zero_delta(self):
        a = fixture(); a['effects'][:] = 0.
        for mode in k.ARMS:
            got = call(a, mode)
            self.assertEqual(got[2]['counts']['nonzero_delta_cells'], 0)
            equal_array(self, got[1], a['pre'])

    def test_zero_effect_can_clip_out_of_range_support(self):
        a = prefix(fixture(), 1); a['effects'][:] = 0.; a['core'][:] = 0.
        a['baseline'][:] = .75*(1-a['core_weight'])
        got, want = call(a), old_call(a)
        self.assertGreater(got[2]['counts']['nonzero_delta_cells'], 0)
        equal_array(self, got[1], want[1])

    def test_zero_delta_addition_signed_zero_matches_original(self):
        a = prefix(fixture(), 1); a['pre'][0] = [.2, .1, -.0, -.1, -.2, -.3]
        a['core'][:] = 0.; a['baseline'][:] = 0.; a['effects'][:] = 0.
        got, want = call(a), old_call(a)
        equal_array(self, got[1], want[1])
        self.assertEqual(got[2]['counts']['nonzero_delta_cells'], 0)
        self.assertEqual(got[2]['counts']['changed_score_cells'], 1)


class BoundariesTests(unittest.TestCase):
    def test_empty_prefix_no_numeric_consumption(self):
        a = fixture()
        for key, value in a.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == 'f': value[:] = np.nan
        for mode in k.ARMS:
            ch, sc, audit = call(prefix(a, 0), mode)
            self.assertEqual(ch, []); self.assertEqual(sc.shape, (0, 6))
            self.assertEqual(audit['counts']['selector_calls'], 1)

    def test_cash_and_two_names_ignore_effects_support(self):
        for count in range(3):
            a = prefix(fixture(), 1); a['eligible'][:] = False; a['eligible'][0, :count] = True
            for key in ('effects', 'baseline', 'core', 'core_weight'): a[key][:] = np.nan
            for mode in k.ARMS:
                got = call(a, mode)
                self.assertEqual(got[0], [[0, 1] if count == 2 else []])
                equal_array(self, got[1], a['pre'])
                self.assertEqual(got[2]['counts']['adjusted_cells'], 0)

    def test_ineligible_and_unused_bits_preserved(self):
        a = prefix(fixture(), 1); a['eligible'][0, 5] = False; a['pre'][0, 5] = np.nan
        for key in ('effects', 'baseline', 'core', 'core_weight'): a[key][0, 5] = np.nan
        for mode in k.ARMS:
            got = call(a, mode); self.assertEqual(got[1][0, 5].tobytes(), a['pre'][0, 5].tobytes())

    def test_original_scope_does_not_consume_top_two_support(self):
        a = prefix(fixture(), 1); a['dates'] = ['2020-01-31']
        for key in ('effects', 'baseline', 'core', 'core_weight'): a[key][0, 0] = np.nan
        call(a)
        for mode in k.ARMS[1:]:
            with self.assertRaisesRegex(ValueError, 'Finite consumed'): call(a, mode)

    def test_eligible_nonfinite_p_rejected_before_sort(self):
        for x in (np.nan, np.inf, -np.inf):
            a = fixture(); a['pre'][0, 0] = x
            with self.assertRaisesRegex(ValueError, 'Finite eligible'): call(a)

    def test_required_nonfinite_source_rejected(self):
        for key in ('effects', 'baseline', 'core', 'core_weight'):
            for x in (np.nan, np.inf, -np.inf):
                a = fixture(); a[key][0, 2] = x
                with self.assertRaisesRegex(ValueError, 'Finite consumed'): call(a)

    def test_required_core_weight_limits(self):
        for x in (0., .49, 1., 1.01):
            a = fixture(); a['core_weight'][0, 2] = x
            with self.assertRaisesRegex(ValueError, 'core weight'): call(a)

    def test_nonfinite_intermediate_rejected(self):
        a = fixture(); a['baseline'][0, 2] = np.finfo(float).max; a['core'][0, 2] = -np.finfo(float).max
        with self.assertRaises((ValueError, FloatingPointError)): call(a)

    def test_prefix_scores_choices_and_events(self):
        a = fixture()
        for mode in k.ARMS:
            full = call(a, mode)
            for n in range(5):
                got = call(prefix(a, n), mode)
                self.assertEqual(got[0], full[0][:n]); equal_array(self, got[1], full[1][:n])
                self.assertEqual(got[2]['events'], full[2]['events'][:n])

    def test_future_body_poison_removed_by_caller_prefix(self):
        a = fixture(); past = call(prefix(a, 2), k.ARMS[2])
        for key, value in a.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == 'f': value[2:] = np.nan
        a['symbols'][2:] = [object(), object()]; a['dates'][2:] = [object(), object()]
        got = call(prefix(a, 2), k.ARMS[2])
        self.assertEqual(got[0], past[0]); equal_array(self, got[1], past[1])
        self.assertEqual(got[2], past[2])

    def test_inputs_and_continuous_core_unchanged(self):
        a = fixture(); before = deepcopy(a)
        for mode in k.ARMS: call(a, mode)
        for key in a:
            if isinstance(a[key], np.ndarray): equal_array(self, a[key], before[key])
            else: self.assertEqual(a[key], before[key])

    def test_readonly_inputs_and_fresh_result(self):
        a = fixture()
        for value in a.values():
            if isinstance(value, np.ndarray): value.setflags(write=False)
        result = call(a)
        self.assertFalse(np.shares_memory(result[1], a['pre']))


class SchemaTests(unittest.TestCase):
    def test_schema_and_work_counts(self):
        a = fixture(); out = call(a, k.ARMS[2]); au = out[2]
        self.assertEqual(tuple(au), k.AUDIT_KEYS); self.assertEqual(tuple(au['counts']), k.COUNT_FIELDS)
        self.assertTrue(all(tuple(event) == k.EVENT_KEYS for event in au['events']))
        c = au['counts']; self.assertEqual(c['eligible_cells'], 24)
        self.assertEqual(c['adjusted_cells'], 3+3*6)
        self.assertEqual(c['original_rankings'], 4); self.assertEqual(c['final_rankings'], 4)
        self.assertEqual(c['boundary_winner_reductions'], 1)
        self.assertEqual(c['new_model_fits'], 0); self.assertEqual(c['predictive_model_fits'], 0)
        self.assertEqual(c['selector_calls'], 1)

    def test_metadata_and_shape_rejections(self):
        mutations = [lambda a: a.update(mode='bad'), lambda a: a.update(pre=a['pre'].astype('float32')),
            lambda a: a.update(effects=a['effects'][:, :5]), lambda a: a.update(eligible=a['eligible'].astype(int)),
            lambda a: a.update(core_weight=a['core_weight'][:, :1]),
            lambda a: a.update(state_calm10=np.ones((4, 1), bool)),
            lambda a: a.update(state_weak_narrow=np.ones(4, int)),
            lambda a: a['symbols'][0].__setitem__(0, 'B'), lambda a: a['symbols'][0].__setitem__(0, ''),
            lambda a: a['dates'].__setitem__(1, '2016-08-17'),
            lambda a: a['dates'].__setitem__(1, '2016-8-31'),
            lambda a: a['dates'].__setitem__(1, '2016-02-30'),
            lambda a: a['dates'].__setitem__(1, 123)]
        for change in mutations:
            a = fixture(); change(a)
            with self.assertRaises(ValueError): k.select(**a) if 'mode' in a else call(a)

    def test_partial_date_in_later_position_rejected(self):
        a = prefix(fixture(), 2); a['dates'] = ['2016-07-29', '2016-08-17']
        with self.assertRaisesRegex(ValueError, 'index zero'): call(a)

    def test_no_data_or_model_imports(self):
        tree = ast.parse(Path(k.__file__).read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertEqual(set(modules), {'datetime', 'numpy'})


class BilateralCross(unittest.TestCase):
    def test_three_modes_random_prefixes_scope_and_preserved_bits(self):
        import monthly_selector_scope268_independent as independent
        self.assertEqual(hashlib.sha256(Path(independent.__file__).read_bytes()).hexdigest(),
                         '12ec572b1378bbf16fc698890b689523904dd4b7858060a5c4d35619d348d8d9')
        examples = [fixture()]
        rng = np.random.default_rng(268268)
        for _ in range(20):
            a = fixture()
            a['pre'] = rng.uniform(-.5, .5, (4, 6))
            a['eligible'] = rng.random((4, 6)) > .25
            a['effects'] = rng.uniform(-1, 1, (4, 6))
            a['core'] = rng.uniform(-.26, .26, (4, 6))
            a['baseline'] = a['core'] + .48*rng.uniform(-.6, .6, (4, 6))
            examples.append(a)
        unused = fixture()
        unused['eligible'][:, 5] = False
        for key in ('pre', 'effects', 'baseline', 'core', 'core_weight'):
            unused[key][:, 5] = np.nan
        unused['pre'][0, 2] = -0.
        examples.append(unused)
        zero = fixture(); zero['effects'][:] = 0.; zero['baseline'][:] = 0.; zero['core'][:] = 0.
        zero['pre'][0] = [.2, .1, -0., -.1, -.2, -.3]
        examples.append(zero)
        domain = fixture(); domain['dates'][0] = '2016-08-16'
        domain['pre'][:] = [.50, .49, .48, .479, .478, .477]
        domain['baseline'][:] = 0.; domain['core'][:] = 0.; domain['effects'][:] = -.5
        examples.append(domain)
        self.assertNotIn(5, call(domain, k.ARMS[1])[0][0])
        for a in examples:
            for n in range(5):
                p = prefix(a, n)
                for mode in k.ARMS:
                    choices, scores, audit = call(p, mode)
                    receipt = independent.verify(**p, choices=choices, scores=scores, mode=mode)
                    self.assertTrue(receipt['passed'])
                    self.assertEqual(receipt['work']['delta_evaluations'], audit['counts']['adjusted_cells'])
                    self.assertEqual(receipt['work']['source_rank_calls'], audit['counts']['original_rankings'])
                    self.assertEqual(receipt['work']['selection_rank_calls'], audit['counts']['final_rankings'])
                    self.assertEqual(receipt['work']['selector_calls'], 1)


if __name__ == '__main__':
    unittest.main()
