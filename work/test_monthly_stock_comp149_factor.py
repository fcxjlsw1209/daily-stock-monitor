"""Synthetic-only149 validation. No real SBC, market or return inputs."""
from copy import deepcopy
from decimal import localcontext
from fractions import Fraction
import unittest

import numpy as np
import monthly_stock_comp149_factor as f
import test_monthly_cash_quality146_factor as fixtures


def candidate(original, amount='5', scale=None):
    value = deepcopy(original['fields']['current']['operating_cash_flow'])
    value.update(value_decimal=amount, source_sha256=original['source']['sha256'])
    if scale is not None:
        value['scale'] = scale
    result = {key: deepcopy(original[key]) for key in f.METADATA}
    result.update(
        stock_compensation=dict(field='stock_compensation_addback', value=value,
                                gaps=[], candidate_only=True, fact_admitted=False),
        stock_compensation_admission=dict(admitted=True, reasons=[], definition=f.DEFINITION),
        fact_admitted=False, model_admitted=False, strategy_admitted=False)
    return result


def fixture(dates=None, amount='5'):
    meta, original = fixtures.fixture(dates)
    sbc = dict(schema_version=1, method=f.MEASUREMENT,
               events=[candidate(original['events'][0], amount)])
    return meta, original, sbc


def run(meta=None, original=None, sbc=None):
    if meta is None:
        meta, original, sbc = fixture()
    return f.build_features(meta, original, sbc)


class ArithmeticTests(unittest.TestCase):
    def test_exact_definition_and_controls(self):
        meta, original, sbc = fixture()
        got = run(meta, original, sbc)
        old = f.old.build_features(meta, original)
        self.assertEqual(tuple(got['arrays']), f.EXTRA)
        self.assertEqual(got['arrays']['comp_adjusted_cash_flow_bounded'][0, 0], float(Fraction(1, 5)))
        for key in ('earnings_bounded', 'cash_flow_bounded'):
            np.testing.assert_array_equal(got['arrays'][key], old['arrays'][key])
        self.assertEqual(got['arrays']['common_available'].shape, (2, 49))
        self.assertEqual(got['audit']['new_model_fits'], 0)
        self.assertFalse(got['audit']['returns_or_nav_computed'])
        self.assertNotIn('low_accrual_bounded', got['audit']['cells'][0])

    def test_zero_sbc_is_exact_cashflow_control(self):
        got = run(*fixture(amount='0'))
        np.testing.assert_array_equal(got['arrays']['cash_flow_bounded'],
                                      got['arrays']['comp_adjusted_cash_flow_bounded'])

    def test_signed_sbc_zero_crossing_and_no_absolute_value(self):
        for amount in ('5', '-5', '30', '35', '0'):
            with self.subTest(amount=amount):
                got = run(*fixture(amount=amount))['arrays']
                expected = Fraction(30-Fraction(amount), 100)
                self.assertEqual(got['comp_adjusted_cash_flow_bounded'][0, 0], f.old.bounded(expected))
        self.assertGreater(run(*fixture(amount='-5'))['arrays']['comp_adjusted_cash_flow_bounded'][0, 0],
                           run(*fixture(amount='5'))['arrays']['comp_adjusted_cash_flow_bounded'][0, 0])

    def test_exact_large_cancellation_before_float(self):
        meta, original, sbc = fixture()
        huge = 10**80
        event = original['events'][0]
        event['fields']['current']['operating_cash_flow']['value_decimal'] = str(huge+1)
        sbc['events'][0] = candidate(event, str(huge))
        with localcontext() as context:
            context.prec = 3
            got = run(meta, original, sbc)
            self.assertEqual(context.prec, 3)
        self.assertEqual(got['arrays']['comp_adjusted_cash_flow_bounded'][0, 0], float(Fraction(1, 101)))

    def test_bounded_after_subtraction_not_difference_of_bounds(self):
        got = run()['arrays']
        self.assertNotEqual(got['comp_adjusted_cash_flow_bounded'][0, 0],
                            f.old.bounded(Fraction(30, 100))-f.old.bounded(Fraction(5, 100)))

    def test_coherent_unit_rescaling_invariant(self):
        meta, original, sbc = fixture()
        expected = run(meta, original, sbc)
        for side, field in f.old.REQUIRED:
            cell = original['events'][0]['fields'][side][field]
            cell['value_decimal'] = str(Fraction(cell['value_decimal'])/1000)
            # Use terminating decimal spellings, not rational slash syntax.
            cell['value_decimal'] = {'3/100': '.03', '1/100': '.01',
                                     '3/20': '.15', '1/20': '.05'}[cell['value_decimal']]
            cell['scale'] = '1000'
        sbc['events'][0] = candidate(original['events'][0], '.005')
        got = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], expected['arrays'][key])

    def test_huge_scale_invariant(self):
        meta, original, sbc = fixture()
        expected = run(meta, original, sbc)
        for side, field in f.old.REQUIRED:
            original['events'][0]['fields'][side][field]['scale'] = '1e1000'
        sbc['events'][0] = candidate(original['events'][0])
        got = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], expected['arrays'][key])


class ContractTests(unittest.TestCase):
    def assert_all_missing(self, got):
        self.assertFalse(got['arrays']['common_available'].any())
        for key in f.EXTRA[:-1]:
            self.assertTrue(np.isnan(got['arrays'][key]).all())
            self.assertIsNone(got['audit']['cells'][0][key])

    def test_new_admission_required_despite_raw_fact_flag(self):
        meta, original, sbc = fixture()
        del sbc['events'][0]['stock_compensation_admission']
        sbc['events'][0]['fact_admitted'] = True
        self.assert_all_missing(run(meta, original, sbc))

    def test_admission_exact_schema_bool_definition_reasons(self):
        for change in ({'admitted': False}, {'admitted': 1}, {'admitted': 'true'},
                       {'reasons': ['review']}, {'reasons': {}},
                       {'definition': 'different_measure'}, {'extra': True}):
            with self.subTest(change=change):
                meta, original, sbc = fixture()
                sbc['events'][0]['stock_compensation_admission'].update(change)
                self.assert_all_missing(run(meta, original, sbc))

    def test_raw_flags_false_do_not_override_new_admission(self):
        self.assertTrue(run()['arrays']['common_available'][0, 0])

    def test_original_cash_admission_and_all_holds_preserved(self):
        for change in ('admission', 'legacy', 'missing_asset'):
            meta, original, sbc = fixture()
            event = original['events'][0]
            if change == 'admission':
                event['cash_quality_admission'].update(admitted=False, reasons=['hold'])
            elif change == 'legacy':
                event['inherited_rejection_codes'] = ['old_rejection']
            else:
                event['fields']['comparison']['total_assets'] = None
            self.assert_all_missing(run(meta, original, sbc))

    def test_missing_event_value_field_or_parser_gap(self):
        for change in ('event', 'value', 'field', 'gap'):
            meta, original, sbc = fixture()
            if change == 'event':
                sbc['events'] = []
            else:
                result = sbc['events'][0]['stock_compensation']
                if change == 'value': result['value'] = None
                if change == 'field': result['field'] = 'some_other_addback'
                if change == 'gap': result['gaps'] = [{'reason': 'combined'}]
            self.assert_all_missing(run(meta, original, sbc))

    def test_metadata_mismatch_does_not_join_different_reports(self):
        for key in f.METADATA:
            meta, original, sbc = fixture()
            sbc['events'][0][key] = ('other' if key in ('event_id', 'symbol') else None)
            self.assert_all_missing(run(meta, original, sbc))

    def test_cell_source_sha_period_and_availability_must_match(self):
        for key, value in [('source_sha256', 'b'*64), ('source', {}), ('period', {}),
                           ('available_from', '2026-02-03')]:
            meta, original, sbc = fixture()
            sbc['events'][0]['stock_compensation']['value'][key] = value
            self.assert_all_missing(run(meta, original, sbc))

    def test_scale_and_currency_must_match_cashflow_statement(self):
        for key, value in [('scale', '1000'), ('scale', '0'), ('scale', '-1'),
                           ('currency_literal', 'USD'), ('currency_literal', 'EUR'),
                           ('currency_literal', None)]:
            meta, original, sbc = fixture()
            sbc['events'][0]['stock_compensation']['value'][key] = value
            self.assert_all_missing(run(meta, original, sbc))

    def test_invalid_numeric_values_never_guessed(self):
        for value in ('NaN', 'Infinity', '-Infinity', '1,000', '1/2', ' 1', None, True, 1):
            meta, original, sbc = fixture()
            sbc['events'][0]['stock_compensation']['value']['value_decimal'] = value
            self.assert_all_missing(run(meta, original, sbc))

    def test_conflicting_duplicate_blocks_but_exact_duplicate_is_harmless(self):
        meta, original, sbc = fixture()
        expected = run(meta, original, sbc)
        sbc['events'].append(deepcopy(sbc['events'][0]))
        got = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], expected['arrays'][key])
        sbc['events'][1]['stock_compensation']['value']['value_decimal'] = '6'
        self.assert_all_missing(run(meta, original, sbc))

    def test_original_inputs_unchanged_and_all_etfs_unavailable(self):
        inputs = fixture()
        saved = deepcopy(inputs)
        got = run(*inputs)
        self.assertEqual(inputs, saved)
        self.assertFalse(got['arrays']['common_available'][:, 32:].any())
        self.assertEqual(got['audit']['common_available_cells'], 2)

    def test_dataset_schema_requires_explicit_version_and_events(self):
        for replacement in (None, {}, {'schema_version': True, 'events': []},
                            {'schema_version': 1, 'events': None}):
            meta, original, _ = fixture()
            with self.assertRaises(ValueError):
                run(meta, original, replacement)


class CausalityTests(unittest.TestCase):
    assert_all_missing = ContractTests.assert_all_missing

    def test_future_append_does_not_change_prefix(self):
        meta, original, sbc = fixture()
        expected = run(meta, original, sbc)
        future = deepcopy(original['events'][0])
        future.update(event_id='future', available_from='2026-05-01')
        original['events'].append(future)
        new = candidate(future, '999')
        new['stock_compensation_admission']['admitted'] = False
        sbc['events'].append(new)
        got = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], expected['arrays'][key])
        self.assertEqual(got['audit']['cells'], expected['audit']['cells'])

    def test_latest_without_sbc_blocks_older_valid_report(self):
        meta, original, sbc = fixture()
        future = deepcopy(original['events'][0])
        future.update(event_id='latest', release_date='2026-03-10', available_from='2026-03-11')
        future['identity']['identity_valid_from'] = future['available_from']
        for side in f.old.SIDES:
            for value in future['fields'][side].values():
                value['available_from'] = future['available_from']
        original['events'].append(future)
        got = run(meta, original, sbc)
        self.assertTrue(got['arrays']['common_available'][0, 0])
        self.assertFalse(got['arrays']['common_available'][1, 0])
        self.assertTrue(np.isnan(got['arrays']['earnings_bounded'][1, 0]))
        self.assertEqual(got['audit']['cells'][49]['current_event_id'], 'latest')
        self.assertEqual(got['audit']['cells'][49]['reason'], 'current_stock_compensation_event_missing')

    def test_latest_new_gate_invalid_does_not_fallback(self):
        meta, original, sbc = fixture()
        latest = deepcopy(original['events'][0])
        latest.update(event_id='latest', release_date='2026-02-15', available_from='2026-02-16')
        latest['identity']['identity_valid_from'] = latest['available_from']
        for side in f.old.SIDES:
            for value in latest['fields'][side].values():
                value['available_from'] = latest['available_from']
        original['events'].append(latest)
        new = candidate(latest)
        new['stock_compensation_admission'].update(admitted=False, reasons=['source_review'])
        sbc['events'].append(new)
        self.assert_all_missing(run(meta, original, sbc))

    def test_prefix_matches_longer_axis(self):
        meta, original, sbc = fixture()
        full = run(meta, original, sbc)
        meta['dates'] = meta['dates'][:1]
        meta['symbols'] = meta['symbols'][:1]
        prefix = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(prefix['arrays'][key], full['arrays'][key][:1])

    def test_before_availability_and_after_causal_expiry_unavailable(self):
        for day in ('2026-01-30', '2027-07-05'):
            self.assert_all_missing(run(*fixture(dates=[day])))

    def test_comparison_flows_not_required(self):
        meta, original, sbc = fixture()
        expected = run(meta, original, sbc)
        for field in ('net_income', 'operating_cash_flow'):
            original['events'][0]['fields']['comparison'][field] = None
        got = run(meta, original, sbc)
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], expected['arrays'][key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
