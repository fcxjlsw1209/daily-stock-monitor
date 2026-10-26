"""Synthetic contract checks; no real research data is loaded."""
import copy
import json
import unittest

import numpy as np

import monthly_leader_gate218_model as model


def fixture():
    """Four small monthly rows: normal gate, boundary tie, two holdings, cash."""
    names = list('ABCDEFGHI')
    weights = np.array([[.5], [.52], [.5], [.52]])
    cores = np.array([[9., 8., 7., 6., 5., 4., 3., 2., 1.],
                      [9., 8., 7., 6., 5., 4., 4., 4., 1.],
                      [0.] * 9, [0.] * 9])
    eligible = np.ones((4, 9), dtype=bool)
    eligible[2, 2:] = False
    eligible[3, 1:] = False
    arrays = dict(baseline=np.tile(np.arange(1., 10.), (4, 1)),
                  core=cores*weights, core_weight=weights, eligible=eligible)
    meta = dict(dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'],
                ends=['2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30'],
                symbols=[names[:] for _ in range(4)])
    return arrays, meta


def physical_prefix(args, n):
    arrays, meta = args
    return ({key: np.array(value[:n], copy=True) for key, value in arrays.items()},
            {key: copy.deepcopy(value[:n]) for key, value in meta.items()})


def one_row(width=9):
    arrays, meta = physical_prefix(fixture(), 1)
    for key in ('baseline', 'core', 'eligible'):
        arrays[key] = arrays[key][:, :width]
    meta['symbols'][0] = meta['symbols'][0][:width]
    return arrays, meta


class BuildTests(unittest.TestCase):
    def test_fixture_exact_choices_and_scores(self):
        out = model.build(*fixture(), 4)
        self.assertEqual([r['symbols'] for r in out['orders']['leader6_gate']],
                         [['F', 'E', 'D'], ['H', 'G', 'F'], ['B', 'A'], []])
        self.assertEqual([r['symbols'] for r in out['orders']['baseline_score']],
                         [['I', 'H', 'G'], ['I', 'H', 'G'], ['B', 'A'], []])
        self.assertEqual(out['orders']['leader6_gate'][0], dict(
            date='2016-08-17', end='2016-08-31', signal_date='2016-08-17',
            symbols=['F', 'E', 'D'], scores=[6., 5., 4.]))
        self.assertEqual(set(out), {'orders', 'shortlist', 'audit'})
        self.assertEqual(out['shortlist'].shape, (4, 9))
        self.assertEqual(out['shortlist'].dtype.kind, 'b')

    def test_counts_and_boundary_record(self):
        out = model.build(*fixture(), 4)
        self.assertEqual(out['audit']['counts'], dict(
            query_months=4, eligible_cells=21, shortlisted_cells=17, gated_months=2,
            tie_expansion_months=1, tie_added_cells=2, changed_holding_months=2,
            changed_ordered_months=2, cash_months=1, two_holding_months=1,
            three_holding_months=2))
        self.assertEqual(out['audit']['records'][1], dict(
            row=1, date='2016-08-31', eligible_count=9, shortlist_count=8,
            sixth_core_threshold=4., sixth_tie_count=3, tie_added_count=2,
            leader6_symbols=['H', 'G', 'F'], baseline_symbols=['I', 'H', 'G'],
            changed_holdings=True, changed_order=True))
        self.assertEqual(out['audit']['method'], 'eligible_core_top6_then_baseline_top3')
        for key in ('new_model_fits', 'predictive_model_fits', 'new_ridge_fits',
                    'new_tree_fits', 'new_parameters'):
            self.assertEqual(out['audit'][key], 0)
        json.dumps(out['audit'], allow_nan=False)

    def test_six_or_fewer_keeps_every_eligible_name(self):
        for width in range(1, 7):
            with self.subTest(width=width):
                args = one_row(width)
                out = model.build(*args, 1)
                self.assertTrue(out['shortlist'].all())
                self.assertEqual(out['orders']['leader6_gate'], out['orders']['baseline_score'])
                rec = out['audit']['records'][0]
                self.assertIsNone(rec['sixth_core_threshold'])
                self.assertEqual(rec['sixth_tie_count'], 0)
                self.assertEqual(rec['tie_added_count'], 0)
                expected = min(width, 3) if width >= 2 else 0
                self.assertEqual(len(out['orders']['leader6_gate'][0]['symbols']), expected)

    def test_no_eligible_produces_cash_without_ratio_calculation(self):
        arrays, meta = one_row()
        arrays['eligible'][:] = False
        arrays['core'][:] = 1.e308
        arrays['core_weight'][:] = 1.e-308
        out = model.build(arrays, meta, 1)
        self.assertFalse(out['shortlist'].any())
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'], [])
        self.assertEqual(out['orders']['baseline_score'][0]['scores'], [])
        self.assertEqual(out['audit']['counts']['cash_months'], 1)

    def test_neutral_constant_core_preserves_ties_and_baseline(self):
        arrays, meta = one_row()
        arrays['core'][:] = 0.
        out = model.build(arrays, meta, 1)
        self.assertTrue(out['shortlist'].all())
        self.assertEqual(out['orders']['leader6_gate'], out['orders']['baseline_score'])
        rec = out['audit']['records'][0]
        self.assertEqual((rec['sixth_core_threshold'], rec['sixth_tie_count'], rec['tie_added_count']),
                         (0., 9, 3))
        self.assertEqual(out['audit']['counts']['gated_months'], 1)

    def test_sixth_exact_ties_are_expanded_not_approximately_equal(self):
        arrays, meta = one_row(8)
        arrays['core_weight'][:] = 1.
        arrays['core'][0] = [1., .9, .8, .7, .6, .5, np.nextafter(.5, -np.inf), .1]
        out = model.build(arrays, meta, 1)
        self.assertEqual(out['shortlist'][0].tolist(), [True]*6+[False, False])
        arrays['core'][0, 6] = .5
        out2 = model.build(arrays, meta, 1)
        self.assertEqual(out2['shortlist'][0].tolist(), [True]*7+[False])
        self.assertEqual(out2['audit']['records'][0]['sixth_tie_count'], 2)

    def test_gate_rank_is_within_original_eligibility(self):
        arrays, meta = one_row()
        arrays['eligible'][0, 0] = False
        out = model.build(arrays, meta, 1)
        self.assertEqual(out['shortlist'][0].tolist(), [False]+[True]*6+[False]*2)
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'], ['G', 'F', 'E'])
        self.assertEqual(out['audit']['records'][0]['eligible_count'], 8)

    def test_baseline_ties_are_symbol_alphabetical_not_column_order(self):
        arrays, meta = one_row()
        meta['symbols'][0] = list('IHGFEDCBA')
        arrays['baseline'][:] = 1.
        arrays['core'][:] = 0.
        out = model.build(arrays, meta, 1)
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'], ['A', 'B', 'C'])
        self.assertEqual(out['orders']['baseline_score'][0]['symbols'], ['A', 'B', 'C'])
        self.assertEqual(out['orders']['leader6_gate'][0]['scores'], [1., 1., 1.])

    def test_negative_core_keeps_best_six_without_an_upperhalf_gate(self):
        arrays, meta = one_row()
        arrays['core'][0] = -np.arange(1., 10.)
        out = model.build(arrays, meta, 1)
        self.assertEqual(out['shortlist'][0].tolist(), [True]*6+[False]*3)
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'], ['F', 'E', 'D'])

    def test_initial_partial_is_not_incumbent_copy_or_special_cased(self):
        arrays, meta = one_row()
        out = model.build(arrays, meta, 1)
        self.assertNotEqual(out['orders']['leader6_gate'][0]['symbols'],
                            out['orders']['baseline_score'][0]['symbols'])
        meta['dates'][0], meta['ends'][0] = '2020-11-30', '2020-12-31'
        later = model.build(arrays, meta, 1)
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'],
                         later['orders']['leader6_gate'][0]['symbols'])

    def test_independent_months_have_no_retention_or_prior_column_state(self):
        arrays, meta = fixture()
        arrays['eligible'][:] = True
        arrays['core'][:] = 0.
        arrays['baseline'][1] = -arrays['baseline'][0]
        meta['symbols'][1] = list('IHGFEDCBA')
        out = model.build(arrays, meta, 4)
        self.assertEqual(out['orders']['leader6_gate'][0]['symbols'], ['I', 'H', 'G'])
        self.assertEqual(out['orders']['leader6_gate'][1]['symbols'], ['I', 'H', 'G'])
        arrays['baseline'][0] = -arrays['baseline'][0]
        changed = model.build(arrays, meta, 4)
        self.assertEqual(out['orders']['leader6_gate'][1:], changed['orders']['leader6_gate'][1:])

    def test_all_physical_prefixes_preserve_orders_shortlists_and_record_counts(self):
        args = fixture()
        full = model.build(*args, 4)
        for n in range(5):
            with self.subTest(n=n):
                actual = model.build(*args, n)
                physical = model.build(*physical_prefix(args, n), n)
                self.assertEqual(actual['orders'], physical['orders'])
                self.assertEqual(actual['audit'], physical['audit'])
                np.testing.assert_array_equal(actual['shortlist'], physical['shortlist'])
                np.testing.assert_array_equal(actual['shortlist'], full['shortlist'][:n])
                self.assertEqual(actual['audit']['records'], full['audit']['records'][:n])
                for arm in model.ARMS:
                    self.assertEqual(actual['orders'][arm], full['orders'][arm][:n])

    def test_future_numeric_and_metadata_poison_does_not_change_prefix(self):
        arrays, meta = fixture()
        before = model.build(arrays, meta, 3)
        arrays['baseline'][3] = np.nan
        arrays['core'][3] = np.inf
        arrays['core_weight'][3] = -1.
        meta['dates'][3] = object()
        meta['ends'][3] = 'not-a-date'
        meta['symbols'][3] = ['duplicate']*9
        after = model.build(arrays, meta, 3)
        self.assertEqual(before['orders'], after['orders'])
        self.assertEqual(before['audit'], after['audit'])
        np.testing.assert_array_equal(before['shortlist'], after['shortlist'])
        with self.assertRaises(ValueError):
            model.build(arrays, meta, 4)

    def test_empty_prefix_never_reads_any_row(self):
        arrays, meta = fixture()
        arrays['baseline'][:] = np.nan
        arrays['core'][:] = np.inf
        arrays['core_weight'][:] = -1.
        meta['dates'] = [None]*4
        meta['ends'] = [None]*4
        meta['symbols'] = [None]*4
        out = model.build(arrays, meta, 0)
        self.assertEqual(out['orders'], {'leader6_gate': [], 'baseline_score': []})
        self.assertEqual(out['shortlist'].shape, (0, 9))
        self.assertEqual(out['audit']['records'], [])
        self.assertTrue(all(v == 0 for v in out['audit']['counts'].values()))

    def test_bad_consumed_values_fail_including_ineligible_cells(self):
        for key, value in [('baseline', np.nan), ('baseline', np.inf),
                           ('core', np.nan), ('core', -np.inf)]:
            with self.subTest(key=key, value=value):
                arrays, meta = one_row()
                arrays['eligible'][0, 8] = False
                arrays[key][0, 8] = value
                with self.assertRaises(ValueError):
                    model.build(arrays, meta, 1)

    def test_bad_core_weight_values_shapes_and_normalized_overflow(self):
        for value in (0., -1., np.nan, np.inf):
            with self.subTest(value=value):
                arrays, meta = one_row()
                arrays['core_weight'][0, 0] = value
                with self.assertRaises(ValueError):
                    model.build(arrays, meta, 1)
        for shape in ((1,), (1, 9), (1, 0), (1, 1, 1)):
            arrays, meta = one_row()
            arrays['core_weight'] = np.ones(shape)
            with self.assertRaises(ValueError):
                model.build(arrays, meta, 1)
        arrays, meta = one_row()
        arrays['core'][0, 0] = 1.e308
        arrays['core_weight'][0, 0] = 1.e-308
        with self.assertRaises(ValueError):
            model.build(arrays, meta, 1)

    def test_exact_boolean_eligibility_and_real_numeric_types(self):
        arrays, meta = one_row()
        for dtype in (int, float, object):
            bad = copy.deepcopy(arrays)
            bad['eligible'] = bad['eligible'].astype(dtype)
            with self.assertRaises(ValueError):
                model.build(bad, meta, 1)
        for key in ('baseline', 'core', 'core_weight'):
            for dtype in (bool, complex, str, object):
                bad = copy.deepcopy(arrays)
                bad[key] = bad[key].astype(dtype)
                with self.subTest(key=key, dtype=dtype), self.assertRaises(ValueError):
                    model.build(bad, meta, 1)

    def test_input_shapes_required_fields_and_prefix_length(self):
        args = fixture()
        for n in (-1, 5, True, False, np.bool_(True), 1., '1', None):
            with self.subTest(n=n), self.assertRaises(ValueError):
                model.build(*args, n)
        model.build(*args, np.int64(1))
        for key in ('baseline', 'core', 'core_weight', 'eligible'):
            arrays, meta = fixture()
            del arrays[key]
            with self.assertRaises(ValueError):
                model.build(arrays, meta, 1)
        for key in ('baseline', 'core', 'eligible'):
            arrays, meta = fixture()
            arrays[key] = arrays[key][:, :8]
            with self.assertRaises(ValueError):
                model.build(arrays, meta, 1)
        arrays, meta = fixture()
        for key in ('baseline', 'core', 'eligible'):
            arrays[key] = arrays[key][:, :0]
        with self.assertRaises(ValueError):
            model.build(arrays, meta, 1)

    def test_metadata_axes_symbols_and_identity_validation(self):
        for change in ('missing', 'short_dates', 'short_symbols', 'width', 'duplicate',
                       'blank', 'whitespace', 'not_string'):
            arrays, meta = fixture()
            if change == 'missing': del meta['ends']
            elif change == 'short_dates': meta['dates'].pop()
            elif change == 'short_symbols': meta['symbols'].pop()
            elif change == 'width': meta['symbols'][0].pop()
            elif change == 'duplicate': meta['symbols'][0][1] = 'A'
            elif change == 'blank': meta['symbols'][0][0] = ''
            elif change == 'whitespace': meta['symbols'][0][0] = ' A'
            else: meta['symbols'][0][0] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.build(arrays, meta, 4)

    def test_metadata_dates_intervals_and_duplicate_months(self):
        for change in ('noncanonical', 'invalid', 'not_string', 'end_before',
                       'same_month', 'skip_end_month', 'partial_wrong_end',
                       'partial_not_first', 'duplicate_day', 'duplicate_month'):
            arrays, meta = fixture()
            if change == 'noncanonical': meta['dates'][1] = '2016-8-31'
            elif change == 'invalid': meta['dates'][1] = '2016-02-31'
            elif change == 'not_string': meta['dates'][1] = None
            elif change == 'end_before': meta['ends'][1] = '2016-07-31'
            elif change == 'same_month': meta['ends'][1] = '2016-08-31'
            elif change == 'skip_end_month': meta['ends'][1] = '2016-10-31'
            elif change == 'partial_wrong_end': meta['ends'][0] = '2016-09-30'
            elif change == 'partial_not_first': meta['dates'][1] = '2016-08-17'
            elif change == 'duplicate_day': meta['dates'][2] = meta['dates'][1]
            else:
                meta['dates'][2] = '2016-09-01'
                meta['dates'][3] = '2016-09-30'
                meta['ends'][3] = '2016-10-31'
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.build(arrays, meta, 4)

    def test_inputs_remain_unmodified_and_unrelated_fields_unread(self):
        arrays, meta = fixture()
        before_a, before_m = copy.deepcopy(arrays), copy.deepcopy(meta)
        for value in arrays.values():
            value.setflags(write=False)
        arrays['labels'] = object()
        arrays['episode'] = object()
        meta['future_diagnostics'] = object()
        model.build(arrays, meta, 4)
        for key in before_a:
            np.testing.assert_array_equal(arrays[key], before_a[key])
        self.assertEqual({k: meta[k] for k in before_m}, before_m)

    def test_current_column_permutation_preserves_symbol_orders(self):
        arrays, meta = fixture()
        out = model.build(arrays, meta, 4)
        permutation = np.array([8, 1, 6, 3, 4, 5, 2, 7, 0])
        for key in ('baseline', 'core', 'eligible'):
            arrays[key] = arrays[key][:, permutation]
        meta['symbols'] = [[ss[j] for j in permutation] for ss in meta['symbols']]
        permuted = model.build(arrays, meta, 4)
        self.assertEqual(out['orders'], permuted['orders'])
        self.assertEqual(out['audit'], permuted['audit'])
        np.testing.assert_array_equal(out['shortlist'][:, permutation], permuted['shortlist'])

    def test_signed_zero_scores_are_copied(self):
        arrays, meta = one_row(2)
        arrays['baseline'][0] = [-0., 0.]
        out = model.build(arrays, meta, 1)
        order = out['orders']['leader6_gate'][0]
        self.assertEqual(order['symbols'], ['A', 'B'])
        self.assertTrue(np.signbit(order['scores'][0]))
        self.assertFalse(np.signbit(order['scores'][1]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
