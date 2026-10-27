"""Synthetic-only independent oracle tests; never load research data.

This test caller imports the producer solely to supply outputs to the verifier.
The oracle itself does not import it. Analytic fixtures establish expectations
without consulting producer outputs; mutations test that supplied evidence is
actually recomputed rather than trusted.
"""
import ast
import copy
from decimal import Decimal
from fractions import Fraction
import inspect
import math
import random
import unittest

import numpy as np
import monthly_joint_growth271_independent as oracle
import monthly_joint_growth271_kernel as producer


def prices(boundaries):
    assert len(boundaries) == 13
    return [float(boundaries[j // 21]) for j in range(253)]


def rising(end=2):
    return prices([1] + [end] * 12)


def event(names=('A', 'B', 'C', 'D'), ends=None, day='2016-08-31'):
    return {'date': day, 'symbols': list(names), 'eligible': [True] * len(names),
            'prices253': [rising(x) for x in (ends or [2] * len(names))],
            'incumbent_choices': list(range(min(3, len(names)))) if len(names) >= 2 else []}


def ratio(wire):
    return Fraction(int(Decimal(wire['numerator'])), int(Decimal(wire['denominator'])))


class Poison:
    def __iter__(self):
        raise AssertionError('unused body accessed')

    def __getitem__(self, key):
        raise AssertionError('unused body accessed')

    def __len__(self):
        raise AssertionError('unused body accessed')


class PrefixOnly:
    def __init__(self, values, length):
        self.values, self.length, self.reads = values, length, []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.reads.append(index)
        if index not in self.values:
            raise AssertionError('unconsumed event accessed')
        return self.values[index]


class IndependentTests(unittest.TestCase):
    def check(self, events, mode='joint_log_growth', n=None):
        chosen, audit = producer.select(events, mode, n=n)
        result = oracle.verify_selection(events, mode, chosen, audit, n=n)
        self.assertTrue(result['passed'])
        self.assertFalse(result['source_files_authenticated'])
        self.assertFalse(result['official_clock_authenticated'])
        self.assertFalse(result['original_incumbent_authenticated'])
        self.assertFalse(result['computes_strategy_returns'])
        return chosen, audit, result

    def reject_both(self, events, mode='joint_log_growth', n=None):
        with self.assertRaises(ValueError):
            producer.select(events, mode, n=n)
        with self.assertRaises(ValueError):
            oracle.verify_selection(events, mode, [], {}, n=n)

    def test_independence_imports(self):
        tree = ast.parse(inspect.getsource(oracle))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertFalse(any('monthly_' in x or x in ('fractions', 'itertools') for x in imports))

    def test_exact_analytic_equal_path_and_amgm_equality(self):
        chosen, audit, checks = self.check([event(('C', 'A', 'B'))])
        self.assertEqual(chosen, [[1, 2, 0]])
        stats = audit['events'][0]['trios'][0]['statistics']
        self.assertEqual(ratio(stats['control_product']), 8)
        self.assertEqual(ratio(stats['joint_product']), 2)
        self.assertEqual([ratio(x) for x in stats['amgm_block_gaps']], [0] * 12)
        self.assertEqual(ratio(stats['amgm_total_gap']), 0)
        self.assertAlmostEqual(stats['control_log'], math.log(2) / 12)
        self.assertEqual(checks['log_display_checks'], 4)

    def test_control_top3_tie_at_boundary(self):
        e = event(('F', 'B', 'A', 'D', 'C', 'E', 'H', 'G'),
                  ends=[4, 3, 3, 3, 3, 2, 2, 2])
        chosen, audit, _ = self.check([e], 'member_log_growth')
        self.assertEqual(chosen, [[0, 2, 1]])
        self.assertTrue(audit['events'][0]['control_top3_equivalence_checked'])
        self.assertEqual(len(audit['events'][0]['tied_trio_indices']), 6)

    def test_all_exact_ties_use_symbol_tuple_and_rank6_boundary(self):
        e = event(('H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'))
        for mode in oracle.ARMS[1:]:
            with self.subTest(mode=mode):
                chosen, audit, _ = self.check([e], mode)
                row = audit['events'][0]
                self.assertEqual(row['shortlist_indices'], [7, 6, 5, 4, 3, 2])
                self.assertEqual(chosen, [[7, 6, 5]])
                self.assertEqual(row['tied_trio_indices'], list(range(20)))
                self.assertEqual(row['winning_trio_index'], 0)

    def test_joint_differs_from_control_with_known_fraction(self):
        e = event()
        e['prices253'] = [prices([1, 8] + [2] * 11)] * 3 + [prices([1, 1 / 8] + [1.5] * 11)]
        c, ca, _ = self.check([e], 'member_log_growth')
        j, ja, _ = self.check([e], 'joint_log_growth')
        self.assertEqual(c, [[0, 1, 2]])
        self.assertEqual(j, [[0, 1, 3]])
        selected = ja['events'][0]['trios'][1]['statistics']
        self.assertEqual(ratio(selected['control_product']), 6)
        self.assertEqual(ratio(selected['joint_product']), Fraction(1075, 48))
        self.assertGreater(ratio(selected['amgm_total_gap']), 0)
        self.assertEqual(ja['events'][0]['tied_trio_indices'], [1, 2, 3])

    def test_strict_positive_new_momentum_is_separate_from_old_E(self):
        e = event(('A', 'B', 'C', 'D', 'E'), [2, 1, 0.5, 4, math.nextafter(1.0, 2.0)])
        chosen, audit, _ = self.check([e])
        row = audit['events'][0]
        self.assertEqual(row['original_eligible_indices'], list(range(5)))
        self.assertEqual(row['new_eligible_indices'], [0, 3, 4])
        self.assertEqual(row['excluded_momentum_indices'], [1, 2])
        self.assertEqual(chosen, [[3, 0, 4]])

    def test_cash_zero_one_and_exact_two(self):
        for ends, selected in [([1, .5, 1], []), ([2, .5, 1], []), ([2, 3, 1], [1, 0])]:
            with self.subTest(ends=ends):
                chosen, audit, _ = self.check([event(('A', 'B', 'C'), ends)])
                self.assertEqual(chosen, [selected])
                self.assertEqual(audit['events'][0]['trios'], [])
                self.assertEqual(audit['events'][0]['equal_weights'], [.5, .5] if selected else [])

    def test_original_domain_zero_and_one(self):
        for flags in ([False, False, False], [True, False, False]):
            e = event(('A', 'B', 'C'))
            e['eligible'], e['incumbent_choices'] = flags, []
            e['prices253'] = [rising(2) if x else Poison() for x in flags]
            chosen, audit, _ = self.check([e])
            self.assertEqual(chosen, [[]])
            self.assertEqual(audit['counts']['required_cells'], sum(flags))

    def test_partial_and_incumbent_bypass_even_outer_price_body(self):
        for mode in oracle.ARMS:
            e = event(day=oracle.PARTIAL_DATE)
            e['prices253'], e['incumbent_choices'] = Poison(), [2, 0, 1]
            chosen, audit, _ = self.check([e], mode)
            self.assertEqual(chosen, [[2, 0, 1]])
            self.assertEqual(audit['counts']['required_cells'], 0)
        e = event()
        e['prices253'] = Poison()
        self.check([e], oracle.ARMS[0])

    def test_ineligible_inner_price_body_is_not_read(self):
        e = event()
        e['eligible'][3], e['prices253'][3] = False, Poison()
        self.check([e])

    def test_prefix_and_zero_do_not_read_later_events(self):
        for n in (0, 1):
            source = PrefixOnly({0: event()}, 9)
            chosen, audit, _ = self.check(source, n=n)
            self.assertEqual(source.reads, [0, 0] if n else [])
            self.assertEqual(audit['counts']['symbols_per_event'], 4 if n else 0)
            self.assertEqual(len(chosen), n)
            self.assertEqual(audit['counts']['selection_calls'], 1)

    def test_all_required_interior_values_checked_even_outside_shortlist(self):
        e = event(tuple('ABCDEFGH'), [8, 7, 6, 5, 4, 3, 2, .5])
        for column in (6, 7):
            bad = copy.deepcopy(e)
            bad['prices253'][column][111] = float('nan')
            self.reject_both([bad])

    def test_aggregates_required_errors_across_columns(self):
        e = event()
        e['prices253'][0][1] = -1
        e['prices253'][3][222] = 0
        with self.assertRaises(ValueError) as ctx:
            oracle.verify_selection([e], oracle.ARMS[2], [], {})
        self.assertIn('price[0][1]', str(ctx.exception))
        self.assertIn('price[3][222]', str(ctx.exception))

    def test_all_253_checked_but_only_boundaries_enter_score(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        e['prices253'][0][1] = 1e150
        e['prices253'][1][251] = 1e-150
        oracle.verify_selection([e], oracle.ARMS[2], chosen, audit)

    def test_invalid_prices(self):
        for value in (None, float('nan'), float('inf'), -float('inf'), 0., -0.,
                      -1, True, np.bool_(True), '2', complex(2), 10**1000,
                      Fraction(1, 10**1000), Decimal('1')):
            with self.subTest(kind=type(value).__name__):
                e = event()
                e['prices253'][0][100] = value
                self.reject_both([e])

    def test_invalid_price_shape(self):
        for value in ([1.] * 252, [1.] * 254, [[1.]] * 253, set([1.]), '1' * 253,
                      np.ones((253, 1)), {'x': 1}):
            e = event()
            e['prices253'][0] = value
            self.reject_both([e])
        e = event()
        e['prices253'].pop()
        self.reject_both([e])

    def test_binary64_real_conversion_and_numpy(self):
        e = event(('A', 'B', 'C'))
        e['prices253'][0] = [Fraction(x) for x in rising(2)]
        e['prices253'][1] = np.array(rising(3), dtype=np.float64)
        e['prices253'][2] = [int(x) for x in rising(4)]
        self.check([e])

    def test_extreme_binary64_and_huge_exact_rationals(self):
        low, high = math.nextafter(0., 1.), float.fromhex('0x1.fffffffffffffp+1023')
        e = event(('A', 'B', 'C'))
        e['prices253'] = [prices([low] + [high if b % 2 else low for b in range(1, 12)] + [high]),
                          prices([low] + [high if not b % 2 else low for b in range(1, 12)] + [high]),
                          prices([low] * 12 + [high])]
        chosen, audit, _ = self.check([e])
        stats = audit['events'][0]['trios'][0]['statistics']
        self.assertGreater(len(stats['joint_product']['numerator']), 4300)
        self.assertEqual(chosen, [[0, 1, 2]])

    def test_random_dyadic_boundaries_cross_all_arms(self):
        rng = random.Random(271)
        for case in range(9):
            names = list('ABCDEFGH')
            rng.shuffle(names)
            e = event(names)
            e['prices253'] = [prices([1.] + [2. ** rng.randrange(-4, 5) for _ in range(11)] +
                                    [2. ** rng.randrange(-1, 4)]) for _ in names]
            e['eligible'] = [rng.random() > .2 for _ in names]
            old = [i for i, f in enumerate(e['eligible']) if f]
            e['incumbent_choices'] = old[:3] if len(old) >= 2 else []
            for mode in oracle.ARMS:
                with self.subTest(case=case, mode=mode):
                    self.check([e], mode)

    def test_multi_event_counts_and_prefix(self):
        events = [event(day=oracle.PARTIAL_DATE),
                  event(ends=[1, .5, 1, 1], day='2016-08-31'),
                  event(ends=[3, 2, 1, .5], day='2016-09-30'),
                  event(day='2016-10-31')]
        _, audit, checks = self.check(events)
        counts = audit['counts']
        self.assertEqual((counts['initial_partial_events'], counts['formal_events']), (1, 3))
        self.assertEqual((counts['cash_events'], counts['two_name_events'], counts['optimized_events']), (1, 1, 1))
        self.assertEqual((counts['required_cells'], counts['price_observations']), (12, 3036))
        self.assertEqual((counts['trio_evaluations'], counts['objective_comparisons']), (4, 3))
        self.assertEqual(checks['checked_trios'], 4)
        self.check(events, n=3)

    def test_no_mutations(self):
        events = [event()]
        before = copy.deepcopy(events)
        chosen, audit = producer.select(events, oracle.ARMS[2])
        saved = copy.deepcopy((chosen, audit))
        oracle.verify_selection(events, oracle.ARMS[2], chosen, audit)
        self.assertEqual(events, before)
        self.assertEqual((chosen, audit), saved)

    def test_invalid_metadata(self):
        changes = [dict(date='2016-8-31'), dict(date='2016-02-30'), dict(date=3),
                   dict(symbols=['A', 'A', 'C', 'D']), dict(symbols=['A', ' B', 'C', 'D']),
                   dict(symbols=['A', '', 'C', 'D']), dict(symbols=set('ABCD')),
                   dict(symbols=(x for x in 'ABCD')), dict(eligible=[True, 1, True, True]),
                   dict(eligible=[True, np.bool_(True), True, True]), dict(eligible=[True]),
                   dict(incumbent_choices=[True, 1, 2]), dict(incumbent_choices=[0, 0, 2]),
                   dict(incumbent_choices=[0, 1, 4]), dict(incumbent_choices=[0, 1]),
                   dict(incumbent_choices={0, 1, 2})]
        for change in changes:
            with self.subTest(fields=list(change)):
                e = event()
                e.update(change)
                self.reject_both([e])
        e = event()
        e['extra'] = 1
        self.reject_both([e])

    def test_date_order_and_stable_width(self):
        for second in (event(day='2016-08-31'), event(day='2016-08-30'),
                       event(('A', 'B', 'C'), day='2016-09-30')):
            self.reject_both([event(), second])

    def test_bad_modes_prefix_and_event_container(self):
        for mode in ('', 'joint', 1, True, None):
            self.reject_both([event()], mode)
        for n in (-1, 2, True, 1., np.int64(1)):
            self.reject_both([event()], n=n)
        for events in ('x', {'x': event()}, set(), iter([event()])):
            self.reject_both(events)

    def test_reject_holding_tie_or_rank_manipulation(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        for change in ('choices', 'selected', 'ties', 'shortlist', 'winner', 'weight', 'measure_order', 'trio_order'):
            with self.subTest(change=change):
                c, a = copy.deepcopy((chosen, audit))
                row = a['events'][0]
                if change == 'choices': c[0] = [0, 1, 3]
                if change == 'selected': row['selected_indices'] = [0, 1, 3]
                if change == 'ties': row['tied_trio_indices'].pop()
                if change == 'shortlist': row['shortlist_indices'].reverse()
                if change == 'winner': row['winning_trio_index'] = 1
                if change == 'weight': row['equal_weights'][0] = .34
                if change == 'measure_order': row['measurements'].reverse()
                if change == 'trio_order': row['trios'].reverse()
                with self.assertRaises(ValueError):
                    oracle.verify_selection([e], oracle.ARMS[2], c, a)

    def test_reject_every_counter_and_authentication_claim(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        for key in audit['counts']:
            a = copy.deepcopy(audit)
            a['counts'][key] += 1
            with self.subTest(counter=key), self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], chosen, a)
        for key in ('source_files_authenticated', 'original_incumbent_authenticated',
                    'official_clock_authenticated', 'log_displays_used_for_selection',
                    'computes_strategy_returns', 'computes_predictive_labels'):
            a = copy.deepcopy(audit)
            a[key] = True
            with self.subTest(flag=key), self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], chosen, a)

    def test_reject_exact_feature_and_amgm_mutations(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        for field in ('annual_gross', 'momentum', 'block_gross'):
            a = copy.deepcopy(audit)
            wire = a['events'][0]['measurements'][0]['measurement'][field]
            if isinstance(wire, list): wire = wire[0]
            wire['numerator'] = '123'
            with self.subTest(field=field), self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], chosen, a)
        for field in ('constituent_annual_gross', 'mean_block_gross', 'control_product',
                      'joint_product', 'amgm_block_gaps', 'amgm_total_gap'):
            a = copy.deepcopy(audit)
            wire = a['events'][0]['trios'][0]['statistics'][field]
            if isinstance(wire, list): wire = wire[0]
            wire['numerator'] = '999'
            with self.subTest(field=field), self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], chosen, a)

    def test_reject_noncanonical_rational_even_equal_value(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        for replacement in ({'numerator': '4', 'denominator': '2'},
                            {'numerator': '+2', 'denominator': '1'},
                            {'numerator': '02', 'denominator': '1'},
                            {'numerator': 2, 'denominator': '1'}):
            a = copy.deepcopy(audit)
            a['events'][0]['measurements'][0]['measurement']['annual_gross'] = replacement
            with self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], chosen, a)

    def test_log_display_tolerance_cannot_change_exact_selection(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        a = copy.deepcopy(audit)
        a['events'][0]['trios'][0]['statistics']['control_log'] += 1e-13
        oracle.verify_selection([e], oracle.ARMS[2], chosen, a)
        for field in ('control_log', 'joint_log', 'direct_log_gap', 'log_gap_from_exact_ratios'):
            for replacement in (1., float('nan'), float('inf'), 0):
                a = copy.deepcopy(audit)
                a['events'][0]['trios'][0]['statistics'][field] = replacement
                with self.subTest(field=field), self.assertRaises(ValueError):
                    oracle.verify_selection([e], oracle.ARMS[2], chosen, a)

    def test_exact_output_fields_and_types(self):
        e = event()
        chosen, audit = producer.select([e], oracle.ARMS[2])
        for change in ('extra', 'missing', 'int_bool', 'tuple_choice', 'list_tuple'):
            c, a = copy.deepcopy((chosen, audit))
            if change == 'extra': a['extra'] = None
            if change == 'missing': del a['events'][0]['branch']
            if change == 'int_bool': a['counts']['events'] = True
            if change == 'tuple_choice': c[0] = tuple(c[0])
            if change == 'list_tuple': a['events'] = tuple(a['events'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                oracle.verify_selection([e], oracle.ARMS[2], c, a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
