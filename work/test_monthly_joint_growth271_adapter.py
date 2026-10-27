"""Synthetic source/control/271 seams only; no saved research inputs."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_joint_growth271_adapter as a
import test_monthly_basket_core251_kernel as old_fixture


def fixture():
    args = old_fixture.fixture()
    args[1]['eligible'][:, 6:] = False
    return args


def run(args, n=None, **kwargs):
    return a.prepare(*args, n=n, **kwargs)


class Poison:
    def __deepcopy__(self, memo):
        raise AssertionError('future object copied')
    def __float__(self):
        raise AssertionError('future value consumed')


def same(case, left, right):
    for key in ('source_features', 'events', 'choices', 'orders', 'audits', 'source_proof',
                'incumbent_proof', 'selection_proofs', 'work', 'scope'):
        case.assertEqual(left[key], right[key], key)
    for key in ('scores', 'pre_scores'):
        for arm in a.ARMS:
            case.assertEqual(left[key][arm].dtype, right[key][arm].dtype)
            case.assertEqual(left[key][arm].shape, right[key][arm].shape)
            case.assertEqual(left[key][arm].tobytes(), right[key][arm].tobytes())


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = fixture()
        cls.full = run(cls.args)

    def test_full_source_incumbent_then_pure271_and_orders(self):
        r = self.full
        self.assertEqual(tuple(r), a.RESULT_KEYS)
        self.assertEqual(tuple(r['orders']), a.ARMS)
        self.assertTrue(r['source_proof']['passed'])
        self.assertFalse(r['source_proof']['candidate_selection_computed'])
        self.assertTrue(r['incumbent_proof']['passed'])
        s, aa, mm = self.args[:3]
        chosen, post = s.choices(s.configs()[4], aa, mm, 3, policy=dict(s.POLICY_FIELDS))
        self.assertEqual(r['orders'][a.ARMS[0]], s.rp.make_orders(chosen, post, mm))
        self.assertEqual(r['scores'][a.ARMS[0]].tobytes(), post.tobytes())
        self.assertEqual(r['pre_scores'][a.ARMS[0]].tobytes(), s.modified_pre_score(s.configs()[4], aa, 3).tobytes())
        for arm in a.ARMS[1:]:
            self.assertEqual(r['orders'][arm][0], r['orders'][a.ARMS[0]][0])
            self.assertEqual(r['scores'][arm][0].tobytes(), r['scores'][a.ARMS[0]][0].tobytes())
            self.assertEqual(r['pre_scores'][arm][0].tobytes(), r['pre_scores'][a.ARMS[0]][0].tobytes())
            for i in (1, 2):
                ids = r['choices'][arm][i]
                self.assertEqual(r['orders'][arm][i]['scores'], list(map(float, range(len(ids), 0, -1))))
        json.dumps({k: v for k, v in r.items() if k not in ('scores', 'pre_scores')}, allow_nan=False)

    def test_forbidden251_candidate_objectives_never_called(self):
        with patch.object(a.source, '_finalize', side_effect=AssertionError('forbidden 251 finalizer')), \
             patch.object(a.source.exact, 'basket', side_effect=AssertionError('forbidden old basket')), \
             patch.object(a.oracle.exactmath, 'basket', side_effect=AssertionError('forbidden independent basket')):
            result = run(self.args, 2)
        self.assertTrue(result['scope']['source251_candidate_finalizer_called'] is False)

    def test_two_early_gates_precede_any271_selection(self):
        marks = []
        real_source, real_inc, real_select = a.oracle.verify_source_features, a.oracle.verify_incumbent, a.kernel.select
        def source(*args, **kwargs):
            self.assertEqual(marks, [])
            result = real_source(*args, **kwargs); marks.append('source'); return result
        def inc(*args, **kwargs):
            self.assertEqual(marks, ['source'])
            result = real_inc(*args, **kwargs); marks.append('inc'); return result
        def select(*args, **kwargs):
            self.assertEqual(marks[:2], ['source', 'inc']); marks.append('selection')
            return real_select(*args, **kwargs)
        with patch.object(a.oracle, 'verify_source_features', side_effect=source), \
             patch.object(a.oracle, 'verify_incumbent', side_effect=inc), patch.object(a.kernel, 'select', side_effect=select):
            run(self.args, 1)
        self.assertEqual(marks, ['source', 'inc', 'selection', 'selection', 'selection'])

    def test_failed_source_stops_original_episode_and_selection(self):
        with patch.object(a.oracle, 'verify_source_features', return_value={'passed': False}), \
             patch.object(self.args[0].ep, 'select_boundary', side_effect=AssertionError('too early')), \
             patch.object(a.kernel, 'select', side_effect=AssertionError('too early')), \
             self.assertRaisesRegex(ValueError, 'source gate'):
            run(self.args, 1)

    def test_failed_incumbent_stops_window_reads_and_new_selection(self):
        with patch.object(a.oracle, 'verify_incumbent', return_value={'passed': False}), \
             patch.object(a, '_windows', side_effect=AssertionError('too early')), \
             patch.object(a.kernel, 'select', side_effect=AssertionError('too early')), \
             self.assertRaisesRegex(ValueError, 'incumbent gate'):
            run(self.args, 1)

    def test_passed_true_incomplete_source_scope_or_counts_rejected_early(self):
        valid = deepcopy(self.full['source_proof']); valid['checked_events'] = 1
        bad = [{'passed': True}]
        for key in a.oracle.SOURCE_GATE_FLAGS:
            if key != 'passed':
                check = deepcopy(valid); del check[key]; bad.append(check)
        for key, value in (('checked_events', 0), ('checked_events', True), ('episode_calls', 1),
                           ('episode_calls', False), ('new_target_ranks_computed', True),
                           ('new_basket_selection_computed', True), ('producer_modules_imported', True),
                           ('computes_returns', True), ('candidate_selection_computed', True),
                           ('C_source_atol', 1e-10), ('C_source_rtol', 1e-10),
                           ('continuous_atol', 1e-8), ('continuous_rtol', 1e-8), ('rank_tolerance', 1e-12)):
            check = deepcopy(valid); check[key] = value; bad.append(check)
        check = deepcopy(valid); del check['continuous_atol']; bad.append(check)
        for check in bad:
            with self.subTest(check=check), patch.object(a.oracle, 'verify_source_features', return_value=check), \
                 patch.object(a, '_incumbent', side_effect=AssertionError('original episode before source proof')), \
                 patch.object(a.kernel, 'select', side_effect=AssertionError('new selection before source proof')), \
                 self.assertRaisesRegex(ValueError, 'source gate'):
                run(self.args, 1)

    def test_passed_true_incomplete_incumbent_scope_or_counts_rejected_early(self):
        valid = deepcopy(self.full['incumbent_proof'])
        valid['checked_events'] = valid['ordered_choices_checked'] = 1
        bad = [{'passed': True}]
        for key in a.oracle.INCUMBENT_GATE_FLAGS:
            if key != 'passed':
                check = deepcopy(valid); del check[key]; bad.append(check)
        for key, value in (('checked_events', 0), ('checked_events', True), ('ordered_choices_checked', 2),
                           ('ordered_choices_checked', True), ('numerical_tolerance', 1e-12),
                           ('numerical_tolerance', False), ('producer_modules_imported', True),
                           ('computes_returns', True), ('candidate_selection_computed', True)):
            check = deepcopy(valid); check[key] = value; bad.append(check)
        check = deepcopy(valid); del check['ordered_choices_checked']; bad.append(check)
        for check in bad:
            with self.subTest(check=check), patch.object(a.oracle, 'verify_incumbent', return_value=check), \
                 patch.object(a, '_windows', side_effect=AssertionError('new window before incumbent proof')), \
                 patch.object(a.kernel, 'select', side_effect=AssertionError('new selection before incumbent proof')), \
                 self.assertRaisesRegex(ValueError, 'incumbent gate'):
                run(self.args, 1)

    def test_real251_gate_receipts_remain_compatible(self):
        a.source_gate_check(self.full['source_proof'], 3)
        a.incumbent_gate_check(self.full['incumbent_proof'], 3)
        for n in (0, 1):
            result = run(self.args, n)
            a.source_gate_check(result['source_proof'], n)
            a.incumbent_gate_check(result['incumbent_proof'], n)

    def test_expected_incumbent_mismatch_stops_before_new_work(self):
        expected = deepcopy(self.full['orders'][a.ARMS[0]])
        expected[0]['symbols'].reverse()
        with patch.object(a.kernel, 'select', side_effect=AssertionError('too early')), self.assertRaises(ValueError):
            run(self.args, 1, expected_incumbent_orders=expected)

    def test_all_fake_prefixes_physical_equal_fresh(self):
        for n in range(4):
            with self.subTest(n=n):
                fresh = run(self.args, n)
                shortened = a.physical_inputs(*self.args[1:], n=n)
                actual = a.prepare(self.args[0], *shortened)
                same(self, fresh, actual)
                self.assertEqual(fresh['choices'], {k: self.full['choices'][k][:n] for k in a.ARMS})

    def test_future_bodies_unconsumed_and_input_bytes_preserved(self):
        s, arrays, meta, panel, calendar, review = fixture()
        expected = run((s, arrays, meta, panel, calendar, review), 2)
        for k in ('dates', 'ends', 'symbols'):
            meta[k][2] = Poison()
        review['clock_events'][2] = Poison()
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].loc['2016-09-01':, :] = Poison()
        for k in ('baseline', 'core', 'effects', 'raw_local_linear'):
            arrays[k] = arrays[k].astype(object); arrays[k][2] = Poison()
        before = {k: v.tobytes() for k, v in arrays.items()}
        actual = run((s, arrays, meta, panel, calendar, review), 2)
        same(self, expected, actual)
        self.assertEqual(before, {k: v.tobytes() for k, v in arrays.items()})

    def test_empty_and_partial_no_source_numbers_or271_measure(self):
        s, arrays, meta, panel, calendar, review = fixture()
        panel['adj'] = panel['adj'].astype(object); panel['adj'].loc[:, :] = Poison()
        for n in (0, 1):
            with patch.object(a.kernel, 'measure', side_effect=AssertionError('partial source numeric')), \
                 patch.object(a.oracle, 'required_prices', side_effect=AssertionError('partial source window')):
                actual = run((s, arrays, meta, panel, calendar, review), n)
            for k in ('producer_original_history_price_observations', 'independent_original_history_price_observations',
                      'adapter_required_price_observations'):
                self.assertEqual(actual['work'][k], 0)
            self.assertEqual(actual['work']['producer_original_episode_calls'], 1)

    def test_253_window_is_exact_E_and_official_formation_clock(self):
        for i in (1, 2):
            e = self.full['events'][i]; f = self.full['source_features'][i]
            t = self.args[4].index(e['date'])
            for j in range(49):
                if e['eligible'][j]:
                    expected = self.args[3]['adj'].loc[self.args[4][t - 294]:self.args[4][t - 42], e['symbols'][j]].to_numpy()
                    self.assertEqual(np.asarray(e['prices253'][j], dtype=float).tobytes(), expected.tobytes())
                else:
                    self.assertIsNone(e['prices253'][j])
            self.assertEqual(f['required_indices'], list(range(6)))

    def test_independent_window_difference_rejected_before271(self):
        events = self.full['source_features']
        real = a.oracle.required_prices
        def changed(*args, **kwargs):
            result = real(*args, **kwargs); result[10] = np.nextafter(result[10], np.inf); return result
        with patch.object(a.oracle, 'required_prices', side_effect=changed), self.assertRaisesRegex(ValueError, 'required-price window'):
            a._windows(events, self.args[3], self.args[4], self.args[5], self.full['choices'][a.ARMS[0]])

    def test_source_clock_and_core_tamper_rejected(self):
        s, arrays, meta, panel, calendar, review = fixture()
        review['clock_events'][1]['price_start'] = calendar[0]
        with self.assertRaisesRegex(ValueError, 'Source clock'):
            run((s, arrays, meta, panel, calendar, review), 2)
        review = self.args[5]
        arrays['core'][1, 0] += .01
        with self.assertRaisesRegex(ValueError, 'core source mismatch'):
            run((s, arrays, meta, panel, calendar, review), 2)

    def test_visible_listing_bound_cannot_be_bypassed(self):
        s, arrays, meta, panel, calendar, _ = fixture()
        bound = dict(symbol='S00', issuer_key='synthetic:S00', not_before='2016-01-01',
                     available_from='2016-01-04', evidence_id='synthetic', kind='new_issue')
        review = old_fixture.make_review(meta, calendar, calendar[0], [bound])
        with self.assertRaisesRegex(ValueError, 'listing contradiction'):
            run((s, arrays, meta, panel, calendar, review), 2)

    def test_required_interior_nan_fails_no_positive_filter_escape(self):
        s, arrays, meta, panel, calendar, review = fixture()
        t = calendar.index(meta['dates'][1])
        panel['adj'].at[pd.Timestamp(calendar[t - 140]), 'S00'] = np.nan
        with patch.object(a.kernel, 'select', side_effect=AssertionError('new objective')), self.assertRaises(ValueError):
            run((s, arrays, meta, panel, calendar, review), 2)

    def test_work_discloses_reused251_measurements_and271_work(self):
        w = self.full['work']; H = 12
        for k in ('producer251_required_price_observations', 'independent251_required_price_observations',
                  'adapter_required_price_observations', 'independent_window_price_observations',
                  'exact_window_value_comparisons'):
            self.assertEqual(w[k], 253 * H)
        self.assertEqual(w['producer251_block_return_evaluations'], 12 * H)
        self.assertEqual(w['producer_original_episode_calls'], 1)
        self.assertEqual(w['independent_original_episode_calls'], 1)
        self.assertEqual(w['candidate_episode_calls'], 0)
        for arm in a.ARMS:
            count = w['producer_selection_counts'][arm]
            self.assertEqual(count['selection_calls'], 1)
            self.assertEqual(count['required_cells'], 0 if arm == a.ARMS[0] else H)
            self.assertEqual(w['independent_selection_counts'][arm]['checked_required_cells'], count['required_cells'])

    def test_cash_two_names_and_original_arrays_unchanged(self):
        for size in (0, 1, 2):
            args = fixture(); args[1]['eligible'][:] = False; args[1]['eligible'][:, :size] = True
            before = a.source.old._snapshot(args[1])
            result = run(args, 2)
            self.assertEqual(before, a.source.old._snapshot(args[1]))
            for arm in a.ARMS:
                self.assertEqual(len(result['choices'][arm][1]), 2 if size == 2 else 0)

    def test_strict_prefix_and_27_axis(self):
        for n in (-1, True, 4, 1.0):
            with self.assertRaises(ValueError):
                run(self.args, n)
        args = fixture(); del args[1]['synthetic_unchanged_0']
        with self.assertRaisesRegex(ValueError, '27'):
            run(args, 0)


if __name__ == '__main__':
    unittest.main()
