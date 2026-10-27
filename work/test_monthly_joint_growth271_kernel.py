"""Pure synthetic tests only: no market files, model fitting, or replay."""
import copy
from fractions import Fraction as F
from itertools import combinations
import math
import pickle
import sys
import unittest

import numpy as np
import monthly_basket_core251_math as legacy
import monthly_joint_growth271_kernel as k


def path(end=2.0):
    return [1.0 + (end - 1.0) * j / 252 for j in range(253)]


def powers(exponents):
    bounds = [1.0]
    for exponent in exponents:
        bounds.append(bounds[-1] * 2.0**exponent)
    return [bounds[j // 21] for j in range(253)]


def event(width=6, day='2016-08-31'):
    return dict(date=day, symbols=[f'S{i:02d}' for i in range(width)],
                eligible=[True] * width, prices253=[path(2 + i / 8) for i in range(width)],
                incumbent_choices=list(range(min(width, 3))) if width >= 2 else [])


class Poison:
    def __len__(self):
        raise AssertionError('Unused body read')

    def __iter__(self):
        raise AssertionError('Unused body read')

    def __getitem__(self, key):
        raise AssertionError('Unused body read')


class IndexedOnly:
    def __init__(self, size, values):
        self.size, self.values, self.reads = size, values, []

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if key not in self.values:
            raise AssertionError(f'Unconsumed index {key}')
        self.reads.append(key)
        return self.values[key]


class MeasureTests(unittest.TestCase):
    def test_exact_251_momentum_and_twelve_endpoints(self):
        values = path(3.75)
        got, old = k.measure(values), legacy.measure(values)
        self.assertEqual(got['momentum'], old['momentum'])
        self.assertEqual((got['price_count'], got['block_count']), (253, 12))
        for b, gross in enumerate(got['block_gross']):
            expected = F.from_float(values[21 * (b + 1)]) / F.from_float(values[21 * b])
            self.assertEqual(k.rational(gross), expected)
            self.assertEqual(k.rational(gross) - 1, legacy._rational(old['block_returns'][b]))

    def test_nonendpoint_value_cannot_change_math(self):
        values = path()
        before = k.measure(values)
        values[1], values[250] = 1e200, 1e-200
        self.assertEqual(before, k.measure(values))

    def test_bad_interior_still_rejected(self):
        values = path()
        values[1] = 0.0
        with self.assertRaisesRegex(ValueError, 'Price 1'):
            k.measure(values)

    def test_invalid_prices(self):
        for invalid in (None, float('nan'), float('inf'), -float('inf'), 0.0,
                        -0.0, -1, True, np.bool_(True), '1', 1j):
            values = path()
            values[87] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                k.measure(values)

    def test_binary64_conversion_edges(self):
        for invalid in (10**1000, F(1, 10**1000)):
            values = path()
            values[0] = invalid
            with self.assertRaises(ValueError):
                k.measure(values)
        values = [float.fromhex('0x0.0000000000001p-1022')] * 253
        values[-1] *= 2
        self.assertEqual(k.rational(k.measure(values)['momentum']), 1)

    def test_all_price_errors_are_found(self):
        values = path()
        values[0], values[-1] = None, False
        with self.assertRaisesRegex(ValueError, 'Price 0.*Price 252'):
            k.measure(values)

    def test_shapes(self):
        for invalid in ([], path()[:-1], path() + [1], np.ones((253, 1)),
                        set(path()), 'prices', {'values': path()}):
            with self.subTest(type=type(invalid)), self.assertRaises(ValueError):
                k.measure(invalid)

    def test_scale_invariance_exact_power_of_two(self):
        values = path(2.125)
        self.assertEqual(k.measure(values), k.measure([v * 128 for v in values]))

    def test_zero_and_negative_momentum_measured_not_filtered(self):
        self.assertEqual(k.rational(k.measure([1.0] * 253)['momentum']), 0)
        self.assertEqual(k.rational(k.measure(path(0.5))['momentum']), F(-1, 2))

    def test_slice_exact_253_and_no_recent42_or_earlier_reads(self):
        values = {i: float(i) for i in range(105, 358)}
        history = IndexedOnly(400, values)
        self.assertEqual(k.slice_history(history), [float(i) for i in range(105, 358)])
        self.assertEqual(history.reads, list(range(105, 358)))

    def test_skip42_changes_do_not_change_measurement(self):
        history = path() + [999.0] * 42
        a = k.measure(k.slice_history(history))
        history[-42:] = [float('nan')] * 42
        self.assertEqual(a, k.measure(k.slice_history(history)))
        history[252] *= 2
        self.assertNotEqual(a, k.measure(k.slice_history(history)))

    def test_short_history_and_shape(self):
        for invalid in ([1] * 294, np.ones((295, 1)), {}, 'x'):
            with self.assertRaises(ValueError):
                k.slice_history(invalid)

    def test_canonical_rational_rejection(self):
        for invalid in (1.0, True, {'numerator': '01', 'denominator': '1'},
                        {'numerator': '-0', 'denominator': '1'},
                        {'numerator': '2', 'denominator': '2'},
                        {'numerator': '1', 'denominator': '0'}):
            with self.assertRaises(ValueError):
                k.rational(invalid)


class BasketTests(unittest.TestCase):
    def rows(self):
        return [[F(101 + ((b + i) % 5), 100) for b in range(12)] for i in range(3)]

    def test_exact_all_values_recomputed(self):
        rows = self.rows()
        got = k.basket(rows)
        means = [sum(row[b] for row in rows) / 3 for b in range(12)]
        annual = [math.prod(row) for row in rows]
        self.assertEqual([k.rational(x) for x in got['mean_block_gross']], means)
        self.assertEqual([k.rational(x) for x in got['constituent_annual_gross']], annual)
        self.assertEqual(k.rational(got['control_product']), math.prod(annual))
        self.assertEqual(k.rational(got['joint_product']), math.prod(means))
        for b, actual in enumerate(got['amgm_block_gaps']):
            self.assertEqual(k.rational(actual), means[b]**3 - math.prod(row[b] for row in rows))
        self.assertEqual(k.rational(got['amgm_total_gap']), math.prod(means)**3 - math.prod(annual))

    def test_log_definitions_and_bounded_display_roundoff(self):
        rows = self.rows()
        got = k.basket(rows)
        c = math.fsum(math.log(float(x)) for row in rows for x in row) / 36
        j = math.fsum(math.log(float(sum(row[b] for row in rows) / 3)) for b in range(12)) / 12
        # A display-only ULP envelope for this synthetic fixture, not an
        # implementation gate, rank tolerance, or financial admission epsilon.
        bound = 48 * math.ulp(max(abs(c), abs(j)))
        self.assertLessEqual(abs(got['control_log'] - c), bound)
        self.assertLessEqual(abs(got['joint_log'] - j), bound)
        self.assertGreaterEqual(got['direct_log_gap'], -bound)
        self.assertLessEqual(abs(got['direct_log_gap'] - got['log_gap_from_exact_ratios']), bound)
        self.assertFalse(got['log_displays_used_for_selection'])

    def test_amgm_strict_and_equality_cases(self):
        row = [F(101, 100)] * 12
        same = k.basket([row, row, row])
        self.assertEqual(k.rational(same['amgm_total_gap']), 0)
        self.assertEqual(same['log_gap_from_exact_ratios'], 0.0)
        varied = k.basket([row, [F(102, 100)] * 12, [F(103, 100)] * 12])
        self.assertGreater(k.rational(varied['amgm_total_gap']), 0)
        self.assertGreater(varied['log_gap_from_exact_ratios'], 0)

    def test_block_and_name_permutations(self):
        rows = self.rows()
        a, b = k.basket(rows), k.basket(rows[::-1])
        for key in ('control_product', 'joint_product', 'amgm_block_gaps', 'amgm_total_gap'):
            self.assertEqual(a[key], b[key])
        c = k.basket([row[::-1] for row in rows])
        for key in ('control_product', 'joint_product', 'amgm_total_gap'):
            self.assertEqual(a[key], c[key])

    def test_invalid_basket_domains(self):
        for invalid in ([], [[F(2)] * 12] * 2, [[F(2)] * 11] * 3,
                        [[F(0)] * 12] * 3, [[F(-1)] * 12] * 3,
                        [[F(1)] * 12] * 3, [[F(1, 2)] * 12] * 3):
            with self.assertRaises(ValueError):
                k.basket(invalid)

    def test_exact_products_separate_float_log_ties(self):
        a = F.from_float(1e100)
        b = F.from_float(math.nextafter(1e100, math.inf))
        low = k.basket([[a] + [F(1)] * 11] * 3)
        high = k.basket([[b] + [F(1)] * 11] * 3)
        self.assertEqual(low['control_log'], high['control_log'])
        self.assertGreater(k.rational(high['control_product']), k.rational(low['control_product']))

    def test_extreme_valid_binary64_ratios_remain_exact(self):
        low, high = float.fromhex('0x0.0000000000001p-1022'), sys.float_info.max
        a = [low if b % 2 == 0 else high for b in range(13)]
        a[-1] = high
        b = [low] + [low if j % 2 else high for j in range(1, 13)]
        p = [[bounds[i // 21] for i in range(253)] for bounds in (a, b, a)]
        got = k.basket([k.measure(values)['block_gross'] for values in p])
        self.assertGreater(len(got['joint_product']['numerator']), 4300)
        self.assertEqual(k.pair(k.rational(got['joint_product'])), got['joint_product'])
        self.assertGreaterEqual(k.rational(got['amgm_total_gap']), 0)
        self.assertTrue(math.isfinite(got['joint_log']))


class SelectionTests(unittest.TestCase):
    def test_control_top3_raw_annual_momentum(self):
        value = event(8)
        chosen, audit = k.select([value], k.ARMS[1])
        self.assertEqual(chosen, [[7, 6, 5]])
        proof = audit['events'][0]
        self.assertEqual(proof['shortlist_indices'], [7, 6, 5, 4, 3, 2])
        self.assertEqual(len(proof['trios']), 20)
        self.assertTrue(proof['control_top3_equivalence_checked'])
        momenta = {x['column_index']: k.rational(x['measurement']['annual_gross']) for x in proof['measurements']}
        values = []
        for trio in proof['trios']:
            expected = math.prod(momenta[i] for i in trio['indices'])
            self.assertEqual(k.rational(trio['statistics']['control_product']), expected)
            values.append(expected)
        self.assertEqual(values[proof['winning_trio_index']], max(values))

    def test_joint_objective_can_choose_complementary_paths(self):
        value = event()
        steady = powers([0] * 11 + [2])
        complement1 = powers([3, -3] * 5 + [3, -2])
        complement2 = powers([-3, 3] * 5 + [-2, 3])
        value['prices253'] = [steady, steady, steady, complement1, complement2, powers([0] * 11 + [1])]
        control, _ = k.select([value], k.ARMS[1])
        joint, proof = k.select([value], k.ARMS[2])
        self.assertEqual(control, [[0, 1, 2]])
        self.assertEqual(joint, [[0, 3, 4]])
        chosen = proof['events'][0]['trios'][proof['events'][0]['winning_trio_index']]
        self.assertGreater(k.rational(chosen['statistics']['amgm_total_gap']), 0)

    def test_lexicographic_ticker_ties_and_momentum_output_order(self):
        value = event()
        value['symbols'] = ['Z', 'D', 'B', 'A', 'C', 'Y']
        value['prices253'] = [path()] * 6
        for mode in k.ARMS[1:]:
            chosen, proof = k.select([value], mode)
            self.assertEqual(chosen, [[3, 2, 4]])
            self.assertEqual(proof['events'][0]['tied_trio_indices'], list(range(20)))

    def test_top6_ignores_seventh_even_if_highly_complementary(self):
        value = event(7)
        value['prices253'] = [path(4)] * 6 + [powers([6, -6] * 5 + [6, -5])]
        chosen, proof = k.select([value], k.ARMS[2])
        self.assertEqual(proof['events'][0]['shortlist_indices'], list(range(6)))
        self.assertNotIn(6, chosen[0])

    def test_all_original_E_checked_even_if_momentum_excluded(self):
        value = event(7)
        value['prices253'][6] = path(0.5)
        value['prices253'][6][17] = 0
        with self.assertRaisesRegex(ValueError, 'Column 6: Price 17'):
            k.select([value], k.ARMS[1])

    def test_all_required_columns_checked_before_ranking(self):
        value = event(7)
        value['prices253'][0][5] = None
        value['prices253'][6][252] = 0
        with self.assertRaisesRegex(ValueError, 'Column 0:.*Column 6:'):
            k.select([value], k.ARMS[2])

    def test_Efalse_is_not_read_or_promoted(self):
        value = event(7)
        value['eligible'][6] = False
        value['prices253'][6] = Poison()
        _, proof = k.select([value], k.ARMS[2])
        self.assertEqual(proof['counts']['required_cells'], 6)
        self.assertNotIn(6, proof['events'][0]['shortlist_indices'])

    def test_positive_gate_and_cash_halves(self):
        for positive in range(4):
            value = event(5)
            value['prices253'] = [path(2)] * positive + [path(0.5)] * (5 - positive)
            for mode in k.ARMS[1:]:
                chosen, audit = k.select([value], mode)
                expected = list(range(positive)) if positive >= 2 else []
                self.assertEqual(chosen, [expected])
                self.assertEqual(audit['events'][0]['equal_weights'], [1 / positive] * positive if positive >= 2 else [])

    def test_original_small_universe(self):
        for width in (1, 2, 3):
            value = event(width)
            for mode in k.ARMS:
                chosen, _ = k.select([value], mode)
                self.assertEqual(len(chosen[0]), width if width >= 2 else 0)

    def test_constant_positive_growth_no_variance_fallback(self):
        value = event()
        value['prices253'] = [powers([1] * 12)] * 6
        for mode in k.ARMS[1:]:
            choices, proof = k.select([value], mode)
            self.assertEqual(choices, [[0, 1, 2]])
            self.assertEqual(proof['events'][0]['branch'], 'optimized')

    def test_partial_and_incumbent_do_not_read_price_body(self):
        for day, modes in ((k.INITIAL_PARTIAL, k.ARMS), ('2016-08-31', k.ARMS[:1])):
            value = event(day=day)
            value['incumbent_choices'] = [2, 0, 1]
            value['prices253'] = Poison()
            for mode in modes:
                choices, proof = k.select([value], mode)
                self.assertEqual(choices, [[2, 0, 1]])
                self.assertEqual(proof['counts']['price_observations'], 0)
                self.assertEqual(proof['events'][0]['trios'], [])

    def test_empty_and_prefix_future_poison(self):
        values = IndexedOnly(4, {})
        choices, proof = k.select(values, k.ARMS[2], n=0)
        self.assertEqual(choices, [])
        self.assertEqual(proof['counts']['selection_calls'], 1)
        self.assertEqual(values.reads, [])
        partial = event(day=k.INITIAL_PARTIAL)
        partial['prices253'] = Poison()
        values = IndexedOnly(4, {0: partial})
        self.assertEqual(k.select(values, k.ARMS[2], n=1)[0], [[0, 1, 2]])
        self.assertEqual(values.reads, [0])

    def test_all_prefixes_match_full(self):
        values = [event(4, day) for day in (k.INITIAL_PARTIAL, '2016-08-31', '2016-09-30', '2016-10-31')]
        for mode in k.ARMS:
            full, audit = k.select(values, mode)
            for n in range(5):
                chosen, proof = k.select(values, mode, n=n)
                self.assertEqual(chosen, full[:n])
                self.assertEqual(proof['events'], audit['events'][:n])

    def test_no_mutation_and_independent_selected_recomputation(self):
        values = [event()]
        before = pickle.dumps(values)
        for mode in k.ARMS:
            _, proof = k.select(values, mode)
            for row in proof['events']:
                if row['winning_trio_index'] is None:
                    continue
                trio = row['trios'][row['winning_trio_index']]
                # Recreate directly from synthetic binary64 input endpoints.
                rows = [[F.from_float(values[0]['prices253'][i][21 * (b + 1)]) /
                         F.from_float(values[0]['prices253'][i][21 * b]) for b in range(12)]
                        for i in trio['indices']]
                exact_joint = math.prod(sum(r[b] for r in rows) / 3 for b in range(12))
                exact_control = math.prod(math.prod(r) for r in rows)
                self.assertEqual(k.rational(trio['statistics']['joint_product']), exact_joint)
                self.assertEqual(k.rational(trio['statistics']['control_product']), exact_control)
        self.assertEqual(before, pickle.dumps(values))

    def test_counts_trace_actual_work(self):
        _, audit = k.select([event()], k.ARMS[2])
        counts = audit['counts']
        self.assertEqual(tuple(counts), k.COUNT_FIELDS)
        expected = dict(events=1, symbols_per_event=6, incumbent_copy_events=0,
                        initial_partial_events=0, formal_events=1, required_cells=6,
                        price_observations=1518, momentum_evaluations=6,
                        block_gross_evaluations=72, positive_momentum_cells=6,
                        excluded_momentum_cells=0, source_rankings=1, shortlist_cells=6,
                        trio_evaluations=20, objective_comparisons=19,
                        exact_amgm_block_checks=240, exact_amgm_total_checks=20,
                        cash_events=0, two_name_events=0, optimized_events=1,
                        selection_calls=1, new_model_fits=0, predictive_model_fits=0)
        self.assertEqual(counts, expected)
        for flag in ('source_files_authenticated', 'original_incumbent_authenticated',
                     'official_clock_authenticated', 'log_displays_used_for_selection',
                     'computes_strategy_returns', 'computes_predictive_labels'):
            self.assertIs(audit[flag], False)

    def test_malformed_metadata_and_prefix(self):
        mutators = [lambda e: e.update(date='2016-8-31'),
                    lambda e: e['symbols'].__setitem__(0, e['symbols'][1]),
                    lambda e: e['eligible'].__setitem__(0, 1),
                    lambda e: e.update(incumbent_choices=[0, 0, 1]),
                    lambda e: e.update(incumbent_choices=[0, 1]),
                    lambda e: e.update(prices253=e['prices253'][:-1]),
                    lambda e: e.update(extra=True)]
        for mutate in mutators:
            value = event()
            mutate(value)
            with self.assertRaises(ValueError):
                k.select([value], k.ARMS[2])
        with self.assertRaises(ValueError):
            k.select([event(), event()], k.ARMS[2])
        for n in (-1, 2, True, 0.5):
            with self.assertRaises(ValueError):
                k.select([event()], k.ARMS[2], n=n)
        with self.assertRaises(ValueError):
            k.select([event()], 'other')

    def test_numeric_log_ties_do_not_become_selection_ties(self):
        value = event()
        ends = [1e100]
        for _ in range(5):
            ends.append(math.nextafter(ends[-1], math.inf))
        value['prices253'] = [[1.0] * 252 + [x] for x in ends]
        chosen, proof = k.select([value], k.ARMS[1])
        self.assertEqual(chosen, [[5, 4, 3]])
        self.assertEqual(len(proof['events'][0]['tied_trio_indices']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
