"""Synthetic-only tests: never load actual financial facts or portfolio outcomes."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import localcontext
import json
import unittest

import numpy as np
import pandas as pd

import monthly_profitability141_factor as f


def event(identifier, start, end, available, gp, assets='1000', currency='USD'):
    return dict(event_id=identifier, available_from=available, admitted=True,
                original_annual=True, measurement_eligible=True,
                issuer_id='issuer_s0', security_id='common_s0',
                identity_valid_from='2010-01-01', identity_valid_through='2035-12-31',
                period_start=start, period_end=end, currency_key=currency,
                gross_profit={'value': gp, 'scale': '1'}, assets={'value': assets, 'scale': '1'})


def pair():
    prior = event('prior', '2024-01-01', '2024-12-31', '2025-02-03', '36600')
    current = event('current', '2025-01-01', '2025-12-31', '2026-02-02', '36500')
    return current, prior


def fixture(dates=None):
    names = ['S'+str(i) for i in range(32)] + ['E'+str(i) for i in range(17)]
    dates = dates or ['2026-03-02', '2026-04-01']
    meta = dict(dates=dates, symbols=[names[:] for _ in dates], etfs=names[32:])
    cur, prev = pair()
    return meta, {'schema_version': 1, 'events': {'S0': [prev, cur]}}


class PairTests(unittest.TestCase):
    def evaluate(self, current=None, prior=None, signal='2026-03-02'):
        cur, prev = pair()
        return f.evaluate_pair(current or cur, prior or prev, signal)

    def test_exact_equality_after_annual_day_normalization(self):
        got = self.evaluate()
        self.assertTrue(got['available'])
        self.assertEqual((got['current_days'], got['prior_days']), (365, 366))
        self.assertEqual(got['gross_profit_growth_sign'], 0.)
        self.assertEqual(got['gross_profitability_improvement_sign'], 0.)

    def test_profit_growth_and_asset_efficiency_are_distinct(self):
        cur, prev = pair()
        cur['gross_profit']['value'] = '73000'
        cur['assets']['value'] = '3000'
        got = self.evaluate(cur, prev)
        self.assertEqual(got['gross_profit_growth_sign'], .5)
        self.assertEqual(got['gross_profitability_improvement_sign'], -.5)
        cur['gross_profit']['value'] = '18250'
        cur['assets']['value'] = '250'
        got = self.evaluate(cur, prev)
        self.assertEqual(got['gross_profit_growth_sign'], -.5)
        self.assertEqual(got['gross_profitability_improvement_sign'], .5)

    def test_negative_and_zero_gross_profit_are_valid(self):
        cur, prev = pair()
        prev['gross_profit']['value'] = '-36600'
        for value in ['-18250', '0', '36500']:
            with self.subTest(value=value):
                cur['gross_profit']['value'] = value
                got = self.evaluate(cur, prev)
                self.assertTrue(got['available'])
                self.assertEqual(got['gross_profitability_improvement_sign'], .5)

    def test_declared_scales_normalize_each_amount_independently(self):
        cur, prev = pair()
        cur['gross_profit'] = {'value': '.0365', 'scale': '1000000'}
        prev['gross_profit'] = {'value': '36.6', 'scale': '1000'}
        cur['assets'] = {'value': '1', 'scale': '1000'}
        prev['assets'] = {'value': '.001', 'scale': '1000000'}
        got = self.evaluate(cur, prev)
        self.assertTrue(got['available'])
        self.assertEqual(got['gross_profit_growth_sign'], 0.)
        self.assertEqual(got['gross_profitability_improvement_sign'], 0.)

    def test_literal_non_usd_currency_is_accepted_but_not_relabelled(self):
        cur, prev = pair()
        cur['currency_key'] = prev['currency_key'] = 'issuer_reported_YEN'
        got = self.evaluate(cur, prev)
        self.assertTrue(got['available'])
        self.assertEqual(got['currency_key'], 'issuer_reported_YEN')
        prev['currency_key'] = 'JPY'
        self.assertEqual(self.evaluate(cur, prev)['reason'], 'different_currency')

    def test_very_small_changes_do_not_round_at_ambient_precision(self):
        prior = event('p', '2022-01-01', '2022-12-31', '2023-02-01', '1')
        current = event('c', '2023-01-01', '2023-12-31', '2024-02-01',
                        '1.0000000000000000000000000000000000000000000000000001')
        with localcontext() as ctx:
            ctx.prec = 3
            got = self.evaluate(current, prior, '2024-03-01')
            self.assertEqual(ctx.prec, 3)
        self.assertEqual(got['gross_profit_growth_sign'], .5)
        self.assertEqual(got['gross_profitability_improvement_sign'], .5)

    def test_huge_finite_decimal_scales_remain_exact(self):
        cur, prev = pair()
        for record in [cur, prev]:
            record['gross_profit']['scale'] = '1e1000'
            record['assets']['scale'] = '1e1000'
        got = self.evaluate(cur, prev)
        self.assertTrue(got['available'])
        self.assertEqual(got['gross_profitability_improvement_sign'], 0.)

    def test_nonpositive_assets_block_both_arms(self):
        for which in ['current', 'prior']:
            for value in ['0', '-1', '-0']:
                cur, prev = pair()
                (cur if which == 'current' else prev)['assets']['value'] = value
                got = self.evaluate(cur, prev)
                self.assertFalse(got['available'])
                self.assertEqual(got['reason'], which + '_nonpositive_assets')
                self.assertIsNone(got['gross_profit_growth_sign'])

    def test_nonfinite_implicit_float_or_bad_scales_are_rejected(self):
        for bad in ['NaN', 'sNaN', 'Infinity', '-Infinity', 123.4, True, None, ' 1']:
            cur, prev = pair()
            cur['gross_profit']['value'] = bad
            self.assertEqual(self.evaluate(cur, prev)['reason'], 'current_invalid_amount_or_scale')
        for bad in ['0', '-1', 'NaN', 1, None]:
            cur, prev = pair()
            cur['assets']['scale'] = bad
            self.assertEqual(self.evaluate(cur, prev)['reason'], 'current_invalid_amount_or_scale')

    def test_inclusive_annual_duration_bounds(self):
        for current_days, prior_days in [(330, 330), (400, 400), (364, 371), (371, 364)]:
            end = date(2025, 12, 31)
            start = end - timedelta(days=current_days - 1)
            prev_end = start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=prior_days - 1)
            cur = event('c', str(start), str(end), '2026-02-02', str(current_days))
            prev = event('p', str(prev_start), str(prev_end), str(prev_end + timedelta(days=20)), str(prior_days))
            got = self.evaluate(cur, prev)
            self.assertTrue(got['available'], got)
            self.assertEqual(got['gross_profitability_improvement_sign'], 0.)
        for days in [329, 401]:
            cur, prev = pair()
            cur['period_start'] = str(date(2025, 12, 31) - timedelta(days=days - 1))
            self.assertEqual(self.evaluate(cur, prev)['reason'], 'current_nonannual_duration')

    def test_noncontiguous_and_overlapping_annual_periods_rejected(self):
        for start in ['2025-01-02', '2024-12-31']:
            cur, prev = pair()
            cur['period_start'] = start
            self.assertEqual(self.evaluate(cur, prev)['reason'], 'noncontiguous_annual_periods')

    def test_550_calendar_days_is_inclusive_then_stale(self):
        cur, prev = pair()
        end = date.fromisoformat(cur['period_end'])
        self.assertTrue(self.evaluate(cur, prev, str(end + timedelta(days=550)))['available'])
        self.assertEqual(self.evaluate(cur, prev, str(end + timedelta(days=551)))['reason'], 'stale_current_report')

    def test_unavailable_report_and_impossible_publication(self):
        cur, prev = pair()
        self.assertEqual(self.evaluate(cur, prev, '2026-01-30')['reason'], 'report_not_yet_available')
        cur['available_from'] = '2025-12-31'
        self.assertEqual(self.evaluate(cur, prev)['reason'], 'current_availability_not_after_period')

    def test_each_source_flag_is_explicit_not_truthy(self):
        for field in ['admitted', 'original_annual', 'measurement_eligible']:
            for value in [False, None, 1, 'true']:
                cur, prev = pair()
                cur[field] = value
                self.assertFalse(self.evaluate(cur, prev)['available'])

    def test_same_report_and_different_identity_cannot_pair(self):
        cur, prev = pair()
        prev['event_id'] = cur['event_id']
        self.assertEqual(self.evaluate(cur, prev)['reason'], 'same_report_not_independent')
        for field in ['issuer_id', 'security_id']:
            cur, prev = pair()
            prev[field] = 'different'
            self.assertEqual(self.evaluate(cur, prev)['reason'], 'different_issuer_or_security')

    def test_current_identity_covers_signal_prior_covers_own_availability(self):
        cur, prev = pair()
        prev['identity_valid_from'] = prev['identity_valid_through'] = prev['available_from']
        cur['identity_valid_from'] = cur['identity_valid_through'] = '2026-03-02'
        self.assertTrue(self.evaluate(cur, prev)['available'])
        self.assertEqual(self.evaluate(cur, prev, '2026-03-03')['reason'], 'current_identity_outside_interval')
        prev['identity_valid_through'] = '2025-02-02'
        self.assertEqual(self.evaluate(cur, prev)['reason'], 'prior_identity_outside_interval')

    def test_missing_identity_bounds_is_neutral(self):
        for which in ['current', 'prior']:
            for field in ['identity_valid_from', 'identity_valid_through']:
                cur, prev = pair()
                del (cur if which == 'current' else prev)[field]
                self.assertFalse(self.evaluate(cur, prev)['available'])


class BuilderTests(unittest.TestCase):
    def test_arrays_common_mask_order_and_audit(self):
        meta, dataset = fixture()
        cur, prev = pair()
        dataset['events']['S1'] = deepcopy([prev, cur])
        dataset['events']['S1'][1]['gross_profit']['value'] = '73000'
        meta['symbols'][1][0], meta['symbols'][1][1] = meta['symbols'][1][1], meta['symbols'][1][0]
        old_meta, old_data = deepcopy(meta), deepcopy(dataset)
        got = f.build_features(meta, dataset)
        self.assertEqual(tuple(got['arrays']), f.EXTRA)
        for key in f.EXTRA:
            self.assertEqual(got['arrays'][key].shape, (2, 49))
        np.testing.assert_array_equal(got['arrays'][f.EXTRA[0]][:, :2], [[0., .5], [.5, 0.]])
        for key in f.EXTRA[:-1]:
            np.testing.assert_array_equal(np.isfinite(got['arrays'][key]), got['arrays']['common_available'])
        self.assertEqual(got['audit']['common_available_cells'], 4)
        self.assertEqual(len(got['audit']['cells']), 98)
        json.dumps(got['audit'], allow_nan=False)
        self.assertEqual(meta, old_meta)
        self.assertEqual(dataset, old_data)

    def test_122_by_49_original_shape_with_no_admitted_data(self):
        dates = [str(d.date()) for d in pd.date_range('2016-08-31', periods=122, freq='BME')]
        meta, dataset = fixture(dates)
        dataset['events'] = {}
        got = f.build_features(meta, dataset)
        self.assertEqual(got['arrays'][f.EXTRA[0]].shape, (122, 49))
        self.assertEqual(len(got['audit']['cells']), 122 * 49)
        self.assertFalse(got['arrays']['common_available'].any())
        self.assertTrue(np.isnan(got['arrays'][f.EXTRA[0]]).all())
        self.assertTrue(np.isnan(got['arrays'][f.EXTRA[1]]).all())

    def test_etf_always_nan_even_with_apparently_valid_reports(self):
        meta, dataset = fixture()
        dataset['events']['E0'] = deepcopy(dataset['events']['S0'])
        got = f.build_features(meta, dataset)
        self.assertTrue(np.isnan(got['arrays'][f.EXTRA[0]][:, 32:]).all())
        self.assertFalse(got['arrays']['common_available'][:, 32:].any())

    def test_latest_invalid_event_blocks_older_valid_pair(self):
        meta, dataset = fixture()
        dataset['events']['S0'].append({'event_id': 'new_invalid', 'available_from': '2026-02-20', 'admitted': False})
        got = f.build_features(meta, dataset)
        self.assertFalse(got['arrays']['common_available'][:, 0].any())
        self.assertEqual(got['audit']['cells'][0]['current_event_id'], 'new_invalid')
        self.assertEqual(got['audit']['cells'][0]['reason'], 'current_not_admitted')

    def test_invalid_prior_revision_is_not_replaced_by_older_valid_prior(self):
        meta, dataset = fixture()
        revision = deepcopy(dataset['events']['S0'][0])
        revision.update(event_id='prior_amended', available_from='2025-06-01', original_annual=False)
        dataset['events']['S0'].append(revision)
        got = f.build_features(meta, dataset)
        self.assertFalse(got['arrays']['common_available'][:, 0].any())
        self.assertEqual(got['audit']['cells'][0]['reason'], 'prior_not_original_annual')
        self.assertEqual(got['audit']['cells'][0]['prior_event_id'], 'prior_amended')

    def test_future_events_and_restated_values_do_not_change_past_prefix(self):
        meta, dataset = fixture(['2026-03-02'])
        before = f.build_features(meta, dataset)
        future = deepcopy(dataset['events']['S0'][1])
        future.update(event_id='future_restatement', available_from='2026-03-20', original_annual=False)
        future['gross_profit'] = {'value': 'NaN', 'scale': False}
        dataset['events']['S0'].append(future)
        after = f.build_features(meta, dataset)
        for key in f.EXTRA:
            np.testing.assert_array_equal(before['arrays'][key], after['arrays'][key])
        self.assertEqual(before['audit'], after['audit'])
        meta['dates'].append('2026-04-01')
        meta['symbols'].append(meta['symbols'][0][:])
        extended = f.build_features(meta, dataset)
        for key in f.EXTRA:
            np.testing.assert_array_equal(before['arrays'][key], extended['arrays'][key][:1])
        self.assertFalse(extended['arrays']['common_available'][1, 0])
        self.assertEqual(extended['audit']['cells'][49]['reason'], 'current_not_original_annual')

    def test_known_competing_same_session_events_are_ambiguous(self):
        meta, dataset = fixture()
        other = deepcopy(dataset['events']['S0'][1])
        other['event_id'] = 'other_source_same_day'
        dataset['events']['S0'].append(other)
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit']['cells'][0]['reason'], 'current_ambiguous_latest_event')
        self.assertFalse(got['arrays']['common_available'][:, 0].any())

    def test_identical_duplicates_do_not_double_count_conflicts_block(self):
        meta, dataset = fixture()
        dataset['events']['S0'].append(deepcopy(dataset['events']['S0'][1]))
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit']['common_available_cells'], 2)
        dataset['events']['S0'][-1]['gross_profit']['value'] = '10'
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit']['cells'][0]['reason'], 'conflicting_event_id')

    def test_event_order_is_not_used_as_publication_order(self):
        meta, dataset = fixture()
        original = f.build_features(meta, dataset)
        dataset['events']['S0'].reverse()
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit'], original['audit'])
        for key in f.EXTRA:
            np.testing.assert_array_equal(got['arrays'][key], original['arrays'][key])

    def test_missing_prior_does_not_use_current_comparative_amount(self):
        meta, dataset = fixture()
        current = dataset['events']['S0'][1]
        current['comparative_gross_profit'] = '36600'
        dataset['events']['S0'] = [current]
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit']['cells'][0]['reason'], 'prior_missing_event')

    def test_unknown_intervening_prior_period_blocks_fallback(self):
        meta, dataset = fixture()
        dataset['events']['S0'].append({'event_id': 'unknown_period', 'available_from': '2025-10-01', 'admitted': False})
        got = f.build_features(meta, dataset)
        self.assertEqual(got['audit']['cells'][0]['reason'], 'unresolved_intervening_annual_period')

    def test_missing_event_chronology_fails_explicitly(self):
        meta, dataset = fixture()
        for bad in [None, '2026-02-01T00:00Z', '2026-02-30']:
            dataset['events']['S0'][1]['available_from'] = bad
            with self.assertRaises(ValueError):
                f.build_features(meta, dataset)

    def test_invalid_schema_axes_and_pool_contracts(self):
        meta, dataset = fixture()
        for schema in [True, '1', 2]:
            dataset['schema_version'] = schema
            with self.assertRaises(ValueError):
                f.build_features(meta, dataset)
        dataset['schema_version'] = 1
        for bad_dates in [['2026-04-01', '2026-03-02'], ['2026-03-02', '2026-03-02'],
                          ['2026-03-02T00:00Z', '2026-04-01T00:00Z']]:
            broken = deepcopy(meta)
            broken['dates'] = bad_dates
            with self.assertRaises(ValueError):
                f.build_features(broken, dataset)
        for bad in ['duplicate', 'width', 'etfs']:
            broken = deepcopy(meta)
            if bad == 'duplicate': broken['symbols'][0][0] = broken['symbols'][0][1]
            if bad == 'width': broken['symbols'][0].pop()
            if bad == 'etfs': broken['etfs'].pop()
            with self.assertRaises(ValueError):
                f.build_features(broken, dataset)

    def test_latest_excluded_industry_is_neutral_without_dropping_stock(self):
        meta, dataset = fixture()
        dataset['events']['S0'][1]['measurement_eligible'] = False
        got = f.build_features(meta, dataset)
        self.assertEqual(got['arrays'][f.EXTRA[0]].shape, (2, 49))
        self.assertEqual(got['audit']['cells'][0]['symbol'], 'S0')
        self.assertEqual(got['audit']['cells'][0]['reason'], 'current_measurement_excluded')
        self.assertFalse(got['arrays']['common_available'][:, 0].any())


if __name__ == '__main__':
    unittest.main()
