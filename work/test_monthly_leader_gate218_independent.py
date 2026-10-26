"""Invented-input oracle checks; no actual218 securities, returns or cache reads."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import monthly_leader_gate218_independent as oracle


def fixture():
    dates = ['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31',
             '2016-11-30', '2016-12-30', '2017-01-31', '2017-02-28', '2017-03-31']
    ends = dates[1:] + ['2017-04-28']
    names = list('ABCDEFGHI')
    baseline = np.tile(np.arange(1., 10.), (9, 1))
    core = np.tile(np.arange(9., 0., -1.), (9, 1))
    core[1] = [9, 8, 7, 6, 5, 4, 4, 4, 1]
    core[2] = 0.
    core[8] = 0.
    baseline[8] = 1.
    eligible = np.ones((9, 9), dtype=bool)
    for row, count in ((3, 6), (4, 2), (5, 1), (6, 0), (7, 3)):
        eligible[row, count:] = False
    arrays = dict(baseline=baseline, core=core, core_weight=np.full((9, 1), .5), eligible=eligible)
    meta = dict(dates=dates, ends=ends, symbols=[names.copy() for _ in dates], etfs=['A', 'B', 'C'])
    return arrays, meta


def physical_prefix(args, n):
    arrays, meta = copy.deepcopy(args)
    arrays = {k: v[:n].copy() for k, v in arrays.items()}
    for key in ('dates', 'ends', 'symbols'):
        meta[key] = meta[key][:n]
    return arrays, meta


class Tests(unittest.TestCase):
    def test_hand_computed_decisions_counts_and_records(self):
        arrays, meta = fixture()
        result = oracle.reconstruct(arrays, meta, 9)
        self.assertEqual(result['orders']['leader6_gate'][0]['symbols'], ['F', 'E', 'D'])
        self.assertEqual(result['orders']['baseline_score'][0]['symbols'], ['I', 'H', 'G'])
        self.assertEqual(result['orders']['leader6_gate'][1]['symbols'], ['H', 'G', 'F'])
        self.assertEqual(result['orders']['leader6_gate'][8]['symbols'], ['A', 'B', 'C'])
        self.assertEqual(result['audit']['counts'], dict(query_months=9, eligible_cells=48,
            shortlisted_cells=44, gated_months=4, tie_expansion_months=3, tie_added_cells=8,
            changed_holding_months=2, changed_ordered_months=2, cash_months=2,
            two_holding_months=1, three_holding_months=6))
        proof = oracle.audit_preparation(arrays, meta, 9, result)
        self.assertTrue(all(proof[k] is True for k in oracle.PREP_FLAGS))
        self.assertFalse(proof['computes_returns'])
        self.assertFalse(proof['producer_modules_imported'])
        self.assertEqual(proof['orders_checked'], dict(leader6_gate=9, baseline_score=9))

    def test_cutoff_all_exact_ties_and_all_neutral_core(self):
        arrays, meta = fixture(); result = oracle.reconstruct(arrays, meta, 9)
        record = result['audit']['records'][1]
        self.assertEqual(record['sixth_core_threshold'], 8.)
        self.assertEqual(record['sixth_tie_count'], 3)
        self.assertEqual(record['tie_added_count'], 2)
        self.assertEqual(np.flatnonzero(result['shortlist'][1]).tolist(), list(range(8)))
        self.assertTrue(result['shortlist'][2].all())
        self.assertEqual(result['audit']['records'][2]['sixth_tie_count'], 9)

    def test_exact_ties_have_no_epsilon(self):
        arrays, meta = fixture()
        arrays['core'][0] = [9., 8., 7., 6., 5., 4., np.nextafter(4., 0.), 3., 2.]
        result = oracle.reconstruct(arrays, meta, 1)
        self.assertEqual(np.flatnonzero(result['shortlist'][0]).tolist(), list(range(6)))
        self.assertEqual(result['audit']['records'][0]['sixth_tie_count'], 1)

    def test_six_or_fewer_keep_all_two_halves_or_cash(self):
        result = oracle.reconstruct(*fixture(), 9)
        for row in range(3, 8):
            np.testing.assert_array_equal(result['shortlist'][row], fixture()[0]['eligible'][row])
            self.assertIsNone(result['audit']['records'][row]['sixth_core_threshold'])
            self.assertEqual(result['audit']['records'][row]['sixth_tie_count'], 0)
            self.assertEqual(result['orders']['leader6_gate'][row], result['orders']['baseline_score'][row])
        self.assertEqual(result['orders']['leader6_gate'][4]['symbols'], ['B', 'A'])
        self.assertEqual(result['orders']['leader6_gate'][5]['symbols'], [])
        self.assertEqual(result['orders']['leader6_gate'][6]['symbols'], [])

    def test_same_sector_and_etfs_have_no_extra_caps(self):
        arrays, meta = fixture()
        meta['sectors'] = [['same-sector'] * 9 for _ in range(9)]
        result = oracle.reconstruct(arrays, meta, 9)
        self.assertEqual(result['orders']['leader6_gate'][7]['symbols'], ['C', 'B', 'A'])
        self.assertEqual(set(result['orders']['leader6_gate'][7]['symbols']), set(meta['etfs']))

    def test_current_eligibility_precedes_cutoff_and_score(self):
        arrays, meta = fixture(); arrays['eligible'][0, 0] = False
        arrays['baseline'][0, 0] = arrays['core'][0, 0] = 1e308
        result = oracle.reconstruct(arrays, meta, 1)
        self.assertFalse(result['shortlist'][0, 0])
        self.assertNotIn('A', result['orders']['baseline_score'][0]['symbols'])
        self.assertEqual(np.flatnonzero(result['shortlist'][0]).tolist(), list(range(1, 7)))

    def test_initial_partial_uses_gate_not_special_copied_local67(self):
        arrays, meta = fixture(); result = oracle.reconstruct(arrays, meta, 1)
        self.assertNotEqual(result['orders']['leader6_gate'][0], result['orders']['baseline_score'][0])
        self.assertEqual(result['orders']['leader6_gate'][0]['signal_date'], '2016-08-17')

    def test_exact_identity_and_lexical_ties_independent_of_columns(self):
        arrays, meta = fixture(); arrays['baseline'][0] = 0.; arrays['core'][0] = 0.
        meta['symbols'][0] = ['z', 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd']
        result = oracle.reconstruct(arrays, meta, 1)
        self.assertEqual(result['orders']['leader6_gate'][0]['symbols'], ['A', 'B', 'C'])
        perm = np.array([8, 6, 4, 2, 0, 7, 5, 3, 1])
        for key in ('baseline', 'core', 'eligible'): arrays[key][0] = arrays[key][0, perm]
        meta['symbols'][0] = [meta['symbols'][0][j] for j in perm]
        other = oracle.reconstruct(arrays, meta, 1)
        self.assertEqual(result['orders'], other['orders'])

    def test_common_positive_scale_preserves_decisions(self):
        arrays, meta = fixture(); expected = oracle.reconstruct(arrays, meta, 9)
        arrays['core'] *= 8.; arrays['core_weight'] *= 8.
        result = oracle.reconstruct(arrays, meta, 9)
        oracle.record_check(result['orders'], expected['orders'])
        oracle.record_check(result['audit'], expected['audit'])
        np.testing.assert_array_equal(result['shortlist'], expected['shortlist'])

    def test_no_inputs_mutated_and_unrelated_arrays_not_consumed(self):
        arrays, meta = fixture(); original = copy.deepcopy((arrays, meta))
        arrays['future_labels'] = object(); arrays['raw_local_linear'] = object()
        oracle.reconstruct(arrays, meta, 9)
        for key in original[0]: np.testing.assert_array_equal(arrays[key], original[0][key])
        self.assertEqual(meta, original[1])

    def test_all_physical_prefixes_match_full_outputs(self):
        args = fixture(); full = oracle.reconstruct(*args, 9)
        for n in range(10):
            for arrays, meta in (args, physical_prefix(args, n)):
                result = oracle.reconstruct(arrays, meta, n)
                self.assertEqual(result['orders'], {k: v[:n] for k, v in full['orders'].items()})
                self.assertEqual(result['audit']['records'], full['audit']['records'][:n])
                np.testing.assert_array_equal(result['shortlist'], full['shortlist'][:n])
                oracle.audit_preparation(arrays, meta, n, result)

    def test_future_poison_is_not_consumed(self):
        arrays, meta = fixture(); expected = oracle.reconstruct(arrays, meta, 4)
        for key in ('baseline', 'core', 'core_weight'): arrays[key][4:] = np.nan
        for key in ('dates', 'ends', 'symbols'): meta[key][4:] = [object()] * 5
        result = oracle.reconstruct(arrays, meta, 4)
        self.assertEqual(result['orders'], expected['orders'])
        self.assertEqual(result['audit'], expected['audit'])
        np.testing.assert_array_equal(result['shortlist'], expected['shortlist'])

    def test_zero_prefix_checks_shapes_but_not_values(self):
        arrays, meta = fixture()
        for key in ('baseline', 'core', 'core_weight'): arrays[key][:] = np.nan
        for key in ('dates', 'ends', 'symbols'): meta[key][:] = [object()] * 9
        result = oracle.reconstruct(arrays, meta, 0)
        self.assertEqual(result['shortlist'].shape, (0, 9))
        self.assertEqual(result['audit']['counts'], dict.fromkeys(oracle.COUNT_FIELDS, 0))

    def test_nonfinite_requested_cache_rejected_even_outside_E(self):
        for key in ('baseline', 'core'):
            for bad in (np.nan, np.inf, -np.inf):
                arrays, meta = fixture(); arrays['eligible'][0, 0] = False; arrays[key][0, 0] = bad
                with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 1)

    def test_weight_positive_and_shape_contract(self):
        for bad in (0., -1., np.nan, np.inf):
            arrays, meta = fixture(); arrays['eligible'][0] = False; arrays['core_weight'][0, 0] = bad
            with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 1)
        for shape in ((9,), (9, 9), (1, 1)):
            arrays, meta = fixture(); arrays['core_weight'] = np.ones(shape)
            with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 1)

    def test_nonfinite_ratio_rejected_but_unused_ratio_not_evaluated(self):
        arrays, meta = fixture(); arrays['core'][0, 0] = 1e308; arrays['core_weight'][0, 0] = 1e-308
        with self.assertRaises(FloatingPointError): oracle.reconstruct(arrays, meta, 1)
        arrays['eligible'][0] = False
        result = oracle.reconstruct(arrays, meta, 1)
        self.assertEqual(result['orders']['leader6_gate'][0]['symbols'], [])

    def test_mask_dtype_numeric_dtype_and_width(self):
        for key, value in (('eligible', np.ones((9, 9))), ('baseline', np.ones((9, 9), bool)),
                           ('core', np.ones((9, 9), complex)), ('core', np.full((9, 9), '1'))):
            arrays, meta = fixture(); arrays[key] = value
            with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 1)
        arrays, meta = fixture(); arrays['baseline'] = np.empty((9, 0))
        with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 0)

    def test_prefix_must_be_nonboolean_integral_and_in_range(self):
        for n in (True, False, 1., -1, 10, None, '1'):
            with self.assertRaises(ValueError): oracle.reconstruct(*fixture(), n)
        self.assertEqual(oracle.reconstruct(*fixture(), np.int64(1))['audit']['counts']['query_months'], 1)

    def test_invalid_metadata_identity_and_date_axes(self):
        alterations = [('symbols', 0, ['A'] * 9), ('symbols', 0, [' A'] + list('BCDEFGHI')),
                       ('dates', 0, '2016-8-17'), ('ends', 0, '2016-09-30'),
                       ('dates', 1, '2016-08-17'), ('ends', 2, '2016-12-30')]
        for key, row, value in alterations:
            arrays, meta = fixture(); meta[key][row] = value
            with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 9)
        arrays, meta = fixture(); meta['symbols'].pop()
        with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 1)

    def test_duplicate_formal_month_and_cross_year(self):
        arrays, meta = fixture(); meta['dates'][2] = '2016-08-30'
        with self.assertRaises(ValueError): oracle.reconstruct(arrays, meta, 9)
        arrays, meta = fixture()
        result = oracle.reconstruct(arrays, meta, 9)
        self.assertEqual(result['orders']['baseline_score'][5]['end'], '2017-01-31')

    def test_signed_zero_score_is_copied(self):
        arrays, meta = fixture(); arrays['baseline'][0] = -0.; arrays['core'][0] = 0.
        result = oracle.reconstruct(arrays, meta, 1)
        self.assertTrue(all(np.signbit(v) for v in result['orders']['leader6_gate'][0]['scores']))
        changed = copy.deepcopy(result); changed['orders']['leader6_gate'][0]['scores'][0] = 0.
        with self.assertRaises(ValueError): oracle.audit_preparation(arrays, meta, 1, changed)

    def test_tampered_outputs_audit_and_mask_rejected(self):
        arrays, meta = fixture(); result = oracle.reconstruct(arrays, meta, 9)
        variants = []
        z = copy.deepcopy(result); z['shortlist'][0, 8] = True; variants.append(z)
        z = copy.deepcopy(result); z['shortlist'] = z['shortlist'].astype(int); variants.append(z)
        z = copy.deepcopy(result); z['orders']['leader6_gate'][0]['scores'][0] += .01; variants.append(z)
        z = copy.deepcopy(result); z['audit']['counts']['eligible_cells'] += 1; variants.append(z)
        z = copy.deepcopy(result); z['audit']['new_parameters'] = 1; variants.append(z)
        z = copy.deepcopy(result); z['audit']['records'][1]['sixth_tie_count'] = 2; variants.append(z)
        for z in variants:
            with self.assertRaises(ValueError): oracle.audit_preparation(arrays, meta, 9, z)

    def test_accounting_dependency_pinned_no_producer_import(self):
        self.assertEqual(oracle.sha(oracle.ACCOUNTING), oracle.ACCOUNTING_SHA)
        helper = oracle.accounting()
        self.assertTrue(callable(helper.audit_path))
        with patch.object(oracle, 'sha', return_value='bad'):
            with self.assertRaises(ValueError): oracle.accounting()
        with self.assertRaises(ValueError): oracle.audit_outcomes(None, None, {}, {}, {}, {})

    def test_unique_blackbox_producer_cross_all_prefixes_both_fixtures(self):
        import monthly_leader_gate218_model as producer
        from test_monthly_leader_gate218_model import fixture as producer_fixture
        for original in (fixture(), producer_fixture()):
            arrays, meta = original
            for n in range(len(meta['dates']) + 1):
                for args in (original, physical_prefix(original, n)):
                    got = producer.build(*args, n)
                    oracle.audit_preparation(*args, n, got)


if __name__ == '__main__':
    unittest.main()
