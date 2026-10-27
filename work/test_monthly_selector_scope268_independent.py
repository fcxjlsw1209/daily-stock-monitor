"""Synthetic-only checks; no actual source loading or producer import."""
from copy import deepcopy
import unittest
import numpy as np
import monthly_selector_scope268_independent as ind


def fixture(n=1, width=6):
    base = np.array([.50, .49, .48, .479, .478, .477], dtype=np.float64)[:width]
    return dict(pre=np.tile(base, (n, 1)), eligible=np.ones((n, width), dtype=bool),
                symbols=[list('ABCDEF'[:width]) for _ in range(n)],
                effects=np.tile(np.array([-.5, -.5, 0., .5, .5, .5])[:width], (n, 1)),
                baseline=np.zeros((n, width)), core=np.zeros((n, width)),
                core_weight=np.full((n, width), .52),
                state_calm10=np.zeros(n, dtype=bool),
                state_weak_narrow=np.zeros(n, dtype=bool),
                dates=['2020-%02d-28' % (i + 1) for i in range(n)])


def supplied(x, mode):
    """Closed-form fixture outputs, deliberately no general selector clone."""
    scores = x['pre'].copy()
    orders = []
    for i in range(len(x['dates'])):
        actual_mode = ind.MODES[0] if x['dates'][i] == ind.INITIAL_PARTIAL else mode
        indices = ([2, 3, 4] if actual_mode == ind.MODES[0]
                   else [0, 1, 2, 3, 4] if actual_mode == ind.MODES[1]
                   else list(range(6)))
        budget = .025 if x['state_calm10'][i] and x['state_weak_narrow'][i] else .05
        for j in indices:
            scores[i, j] += budget * x['effects'][i, j]
        orders.append([3, 0, 1] if actual_mode == ind.MODES[0]
                      else [3, 4, 2] if actual_mode == ind.MODES[1] else [3, 4, 5])
    return orders, scores


def check(x, mode, choices=None, scores=None):
    if choices is None:
        choices, scores = supplied(x, mode)
    return ind.verify(**x, choices=choices, scores=scores, mode=mode)


class SelectorChecks(unittest.TestCase):
    def test_three_modes_and_actual_work(self):
        for mode, cells in zip(ind.MODES, (3, 5, 6)):
            result = check(fixture(), mode)
            self.assertEqual(set(result['flags']), set(ind.FLAGS))
            self.assertTrue(all(result['flags'].values()))
            self.assertEqual(tuple(result['work']), ind.WORK_FIELDS)
            self.assertEqual(result['work']['delta_evaluations'], cells)
            self.assertEqual(result['work']['support_evaluations'], cells)
            self.assertEqual(result['work']['source_rank_calls'], 1)
            self.assertEqual(result['work']['selection_rank_calls'], 1)
            self.assertEqual(result['work']['new_model_fits'], 0)
            self.assertFalse(result['source_provenance_verified'])

    def test_top2_locked_mutant_rejected(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[0])
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], orders, scores)

    def test_wrong_global_scope_rejected(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[2])
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], orders, scores)

    def test_unlocked_top5_cannot_choose_unadjusted_rank6(self):
        x = fixture(); x['effects'][:] = -.5
        scores = x['pre'].copy(); scores[0, :5] += .05 * -.5
        check(x, ind.MODES[1], [[0, 1, 2]], scores)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], [[5, 0, 1]], scores)

    def test_gap_exact_boundary_and_above(self):
        x = fixture(); x['pre'][0] = [.50, .49, .02, 0., -np.spacing(.02), -.1]
        x['effects'][:] = .5
        scores = x['pre'].copy(); scores[0, [2, 3]] += .025
        result = check(x, ind.MODES[0], [[0, 1, 2]], scores)
        self.assertEqual(result['work']['delta_evaluations'], 2)
        bad = scores.copy(); bad[0, 4] += .025
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0], [[0, 1, 2]], bad)

    def test_gap_origin_is_original_rank3(self):
        x = fixture(); x['pre'][0] = [.50, .49, .1, .085, .07, .06]
        x['effects'][:] = .5
        scores = x['pre'].copy(); scores[0, [2, 3]] += .025
        check(x, ind.MODES[0], [[0, 1, 2]], scores)
        bad = scores.copy(); bad[0, 4] += .025
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0], [[0, 1, 2]], bad)

    def test_both_states_reduce_budget(self):
        x = fixture(); x['state_calm10'][:] = True; x['state_weak_narrow'][:] = True
        orders, scores = supplied(x, ind.MODES[2])
        # The declared arithmetic is .10-(.025+.05), not literal .025.
        scores = x['pre'] + (.10 - (.025 + .05)) * x['effects']
        check(x, ind.MODES[2], orders, scores)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], orders, x['pre'] + .05 * x['effects'])

    def test_single_states_leave_full_budget(self):
        for state in ('state_calm10', 'state_weak_narrow'):
            x = fixture(); x[state][:] = True
            check(x, ind.MODES[2])

    def test_clip_and_support_use_existing_formula(self):
        x = fixture(); x['baseline'][:] = .2; x['core'][:] = .1
        x['core_weight'][:] = .5; x['effects'][:] = .5
        scores = x['pre'].copy(); scores[0, [2, 3, 4]] += .05 * (.5 - .2)
        check(x, ind.MODES[0], [[0, 2, 1]], scores)
        bad = x['pre'].copy(); bad[0, [2, 3, 4]] += .05 * .5
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0], [[0, 2, 1]], bad)

    def test_named_partial_matches_incumbent_all_modes(self):
        x = fixture(); x['dates'] = [ind.INITIAL_PARTIAL]
        for mode in ind.MODES:
            result = check(x, mode)
            self.assertEqual(result['work']['partial_events'], 1)
            self.assertEqual(result['work']['delta_evaluations'], 3)
        orders, scores = supplied(fixture(), ind.MODES[2])
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], orders, scores)

    def test_ordinary_first_row_is_not_partial(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[0])
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], orders, scores)

    def test_two_eligible_unadjusted(self):
        x = fixture(); x['eligible'][:] = False; x['eligible'][0, [1, 4]] = True
        for key in ('effects', 'baseline', 'core', 'core_weight'):
            x[key][:] = np.nan
        for mode in ind.MODES:
            result = check(x, mode, [[1, 4]], x['pre'].copy())
            self.assertEqual(result['work']['two_eligible_events'], 1)
            self.assertEqual(result['work']['delta_evaluations'], 0)

    def test_zero_or_one_eligible_cash(self):
        for size in (0, 1):
            x = fixture(); x['eligible'][:] = False; x['eligible'][0, :size] = True
            result = check(x, ind.MODES[2], [[]], x['pre'].copy())
            self.assertEqual(result['work']['cash_events'], 1)

    def test_empty_prefix_no_numeric_consumption(self):
        x = fixture(n=0)
        result = check(x, ind.MODES[2], [], x['pre'].copy())
        self.assertEqual(result['work']['selector_calls'], 1)
        self.assertTrue(all(v == 0 for k, v in result['work'].items() if k != 'selector_calls'))

    def test_prefix_equals_same_consumed_rows(self):
        x = fixture(n=3)
        orders, scores = supplied(x, ind.MODES[2])
        check(x, ind.MODES[2], orders, scores)
        y = {k: v[:1] for k, v in x.items()}
        x['effects'][1:] = np.nan
        check(y, ind.MODES[2], orders[:1], scores[:1])

    def test_inputs_unchanged(self):
        x = fixture(); before = deepcopy(x)
        for mode in ind.MODES:
            check(x, mode)
        for key, value in x.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(value.tobytes(), before[key].tobytes())
            else:
                self.assertEqual(value, before[key])

    def test_symbol_ties_exact_and_final_order_exact(self):
        x = fixture(); x['pre'][:] = .1; x['effects'][:] = 0.
        x['symbols'] = [list('FEDCBA')]
        check(x, ind.MODES[2], [[5, 4, 3]], x['pre'].copy())
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], [[3, 4, 5]], x['pre'].copy())

    def test_score_one_ulp_rejected(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[2])
        scores[0, 3] = np.nextafter(scores[0, 3], np.inf)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], orders, scores)

    def test_unconsumed_nan_payload_preserved(self):
        x = fixture(); x['eligible'][0, 5] = False
        x['pre'].view(np.uint64)[0, 5] = 0x7ff8000000000123
        for key in ('effects', 'baseline', 'core', 'core_weight'):
            x[key][0, 5] = np.nan
        orders, scores = supplied(x, ind.MODES[1])
        check(x, ind.MODES[1], orders, scores)
        scores.view(np.uint64)[0, 5] = 0x7ff8000000000456
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], orders, scores)

    def test_unconsumed_signed_zero_preserved(self):
        x = fixture(); x['eligible'][0, 5] = False; x['pre'][0, 5] = -0.
        orders, scores = supplied(x, ind.MODES[1]); check(x, ind.MODES[1], orders, scores)
        scores[0, 5] = 0.
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], orders, scores)

    def test_zero_delta_keeps_original_addition_semantics(self):
        x = fixture(); x['pre'][:] = -0.; x['effects'][:] = 0.
        scores = x['pre'].copy(); scores[:] = 0.
        check(x, ind.MODES[2], [[0, 1, 2]], scores)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[2], [[0, 1, 2]], x['pre'].copy())

    def test_nonconsumed_top2_support_is_not_read(self):
        x = fixture()
        for key in ('effects', 'baseline', 'core', 'core_weight'):
            x[key][0, :2] = np.nan
        orders, scores = supplied(x, ind.MODES[0]); check(x, ind.MODES[0], orders, scores)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[1], orders, scores)

    def test_nonfinite_eligible_pre_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            x = fixture(); x['pre'][0, 0] = bad
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0])

    def test_required_nonfinite_and_bad_denominator_rejected(self):
        for key in ('effects', 'baseline', 'core', 'core_weight'):
            for bad in (np.nan, np.inf, -np.inf):
                x = fixture(); x[key][0, 3] = bad
                with self.assertRaises(ValueError):
                    check(x, ind.MODES[0])
        for bad in (.49, 1., 1.1):
            x = fixture(); x['core_weight'][0, 3] = bad
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0])

    def test_intermediate_overflow_rejected(self):
        x = fixture(); x['baseline'][0, 3] = np.finfo(float).max
        x['core'][0, 3] = -np.finfo(float).max
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])

    def test_exact_types_shapes(self):
        x = fixture(width=0)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0], [[]], x['pre'].copy())
        for key in ('pre', 'effects', 'baseline', 'core', 'core_weight'):
            x = fixture(); x[key] = x[key].astype(np.float32)
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0])
        x = fixture(); x['eligible'] = x['eligible'].astype(int)
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])
        x = fixture(); x['core_weight'] = x['core_weight'][:, :1]
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])
        x = fixture(); x['state_calm10'] = np.array([1])
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])

    def test_score_dtype_shape_strict(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[0])
        for bad in (scores.astype(np.float32), scores[:, :5], scores.tolist(), scores.astype('>f8')):
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0], orders, bad)

    def test_metadata_failures(self):
        mutations = [('symbols', [['A'] * 6]), ('dates', ['2020-2-28']),
                     ('dates', ['2020-02-30']), ('dates', [None]), ('symbols', [['A'] * 5])]
        for key, value in mutations:
            x = fixture(); x[key] = value
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0], [[3, 0, 1]], x['pre'].copy())
        x = fixture(n=2); x['dates'] = ['2016-01-31', ind.INITIAL_PARTIAL]
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])
        x = fixture(n=2); x['dates'][1] = x['dates'][0]
        with self.assertRaises(ValueError):
            check(x, ind.MODES[0])

    def test_choice_types_and_mode_fail_closed(self):
        x = fixture(); orders, scores = supplied(x, ind.MODES[0])
        for bad in ([[True, 0, 1]], [[3., 0, 1]], [[3, 0, 9]], [[3, 0, 0]], [tuple(orders[0])]):
            with self.assertRaises(ValueError):
                check(x, ind.MODES[0], bad, scores)
        with self.assertRaises(ValueError):
            check(x, 'unlocked_top5_gap_control', orders, scores)


if __name__ == '__main__':
    unittest.main()
