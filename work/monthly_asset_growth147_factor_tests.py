"""Generated inputs only; no real factors or portfolio calculations."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import localcontext
from fractions import Fraction
import json
import unittest
import numpy as np
import pandas as pd
import monthly_asset_growth147_factor as f
import monthly_cash_quality146_factor as old
from test_monthly_cash_quality146_factor import event, fixture


class AssetGrowthTests(unittest.TestCase):
    def result(self, current='150', prior='50'):
        meta, data = fixture()
        for side, value in [('current', current), ('comparison', prior)]:
            data['events'][0]['fields'][side]['total_assets']['value_decimal'] = value
        return f.build_features(meta, data)

    def test_expansion_contraction_zero_and_direction(self):
        self.assertEqual(self.result()['arrays']['low_asset_growth_bounded'][0, 0], -2/3)
        self.assertEqual(self.result('50', '150')['arrays']['low_asset_growth_bounded'][0, 0], .4)
        self.assertEqual(self.result('50', '50')['arrays']['low_asset_growth_bounded'][0, 0], 0.)
        self.assertFalse(np.signbit(self.result('50', '50')['arrays']['low_asset_growth_bounded'][0, 0]))

    def test_control_bytes_and_mask_identical_to146(self):
        meta, data = fixture()
        expected, got = old.build_features(meta, data), f.build_features(meta, data)
        for key in ('earnings_bounded', 'low_accrual_bounded', 'common_available'):
            self.assertEqual(got['arrays'][key].tobytes(), expected['arrays'][key].tobytes())
        self.assertEqual(set(got['arrays']), set(f.EXTRA))

    def test_each_scale_is_normalized(self):
        meta, data = fixture()
        expected = f.build_features(meta, data)
        data['events'][0]['fields']['current']['total_assets'].update(value_decimal='.15', scale='1000')
        data['events'][0]['fields']['comparison']['total_assets'].update(value_decimal='.00005', scale='1000000')
        got = f.build_features(meta, data)
        for key in f.EXTRA:
            self.assertEqual(got['arrays'][key].tobytes(), expected['arrays'][key].tobytes())

    def test_exact_small_difference_before_float_and_decimal_context(self):
        big = 10**80
        with localcontext() as ctx:
            ctx.prec = 3
            got = self.result(str(big+1), str(big))['arrays']['low_asset_growth_bounded'][0, 0]
        self.assertEqual(got, float(Fraction(-1, big+1)))
        self.assertLess(got, 0)

    def test_contraction_near_minus_one_and_huge_expansion(self):
        self.assertEqual(self.result('1e-1000', '1')['arrays']['low_asset_growth_bounded'][0, 0], .5)
        self.assertEqual(self.result('1e1000', '1')['arrays']['low_asset_growth_bounded'][0, 0], -1.)

    def test_nonpositive_or_nonfinite_endpoint_blocks_common_mask(self):
        for side in ('current', 'comparison'):
            for bad in ('0', '-1', 'NaN', 'Infinity', True, 0):
                meta, data = fixture()
                data['events'][0]['fields'][side]['total_assets']['value_decimal'] = bad
                got = f.build_features(meta, data)
                self.assertFalse(got['arrays']['common_available'].any())
                self.assertTrue(all(np.isnan(got['arrays'][key]).all() for key in f.EXTRA[:-1]))

    def test_only_assets_does_not_relax_four_field_mask(self):
        for name in ('net_income', 'operating_cash_flow'):
            meta, data = fixture()
            data['events'][0]['fields']['current'][name] = None
            self.assertFalse(f.build_features(meta, data)['arrays']['common_available'].any())

    def test_comparison_flows_still_unused(self):
        meta, data = fixture()
        expected = f.build_features(meta, data)
        for name in ('net_income', 'operating_cash_flow'):
            data['events'][0]['fields']['comparison'][name] = None
        got = f.build_features(meta, data)
        for key in f.EXTRA:
            self.assertEqual(got['arrays'][key].tobytes(), expected['arrays'][key].tobytes())

    def test_missing_and_etfs_exact_neutral_nan_features(self):
        meta, data = fixture()
        got = f.build_features(meta, data)
        self.assertFalse(got['arrays']['common_available'][:, 1:].any())
        self.assertTrue(np.isnan(got['arrays']['low_asset_growth_bounded'][:, 1:]).all())

    def test_latest_invalid_blocks_older_valid(self):
        meta, data = fixture(['2026-03-02', '2026-04-01'])
        later = deepcopy(data['events'][0])
        later.update(event_id='later', available_from='2026-03-15')
        later['cash_quality_admission'].update(admitted=False, reasons=['hold'])
        data['events'].append(later)
        got = f.build_features(meta, data)
        self.assertTrue(got['arrays']['common_available'][0, 0])
        self.assertFalse(got['arrays']['common_available'][1, 0])

    def test_future_conflicting_same_id_does_not_change_past(self):
        meta, data = fixture()
        expected = f.build_features(meta, data)
        future = deepcopy(data['events'][0])
        future['available_from'] = '2030-01-01'
        future['fields']['current']['total_assets']['value_decimal'] = '999999999'
        data['events'].append(future)
        got = f.build_features(meta, data)
        for key in f.EXTRA:
            self.assertEqual(got['arrays'][key].tobytes(), expected['arrays'][key].tobytes())

    def test_duplicate_same_event_accepted_conflicting_known_event_rejected(self):
        meta, data = fixture()
        data['events'].append(deepcopy(data['events'][0]))
        self.assertTrue(f.build_features(meta, data)['arrays']['common_available'][0, 0])
        data['events'][-1]['fields']['current']['total_assets']['value_decimal'] = '3'
        self.assertFalse(f.build_features(meta, data)['arrays']['common_available'].any())

    def test_legacy_hold_cannot_be_reopened(self):
        meta, data = fixture()
        data['events'][0]['inherited_rejection_codes'] = ['prior_period_unknown']
        self.assertFalse(f.build_features(meta, data)['arrays']['common_available'].any())

    def test_source_period_and_available_mismatch_blocks(self):
        for key, value in [('source', {}), ('period', {}), ('available_from', '2026-01-01')]:
            meta, data = fixture()
            data['events'][0]['fields']['comparison']['total_assets'][key] = value
            self.assertFalse(f.build_features(meta, data)['arrays']['common_available'].any())

    def test_release_and_550_day_expiry(self):
        last = date(2025, 12, 31)+timedelta(days=550)
        dates = ['2026-01-30', '2026-02-02', str(last), str(last+timedelta(days=1))]
        meta, data = fixture(dates)
        self.assertEqual(f.build_features(meta, data)['arrays']['common_available'][:, 0].tolist(),
                         [False, True, True, False])

    def test_all123_prefixes_identical_and_input_unchanged(self):
        dates = pd.date_range('2026-01-01', periods=122).strftime('%Y-%m-%d').tolist()
        meta, data = fixture(dates)
        before = deepcopy((meta, data))
        full = f.build_features(meta, data)
        for n in range(123):
            prefix = dict(meta, dates=dates[:n], symbols=meta['symbols'][:n])
            got = f.build_features(prefix, data)
            for key in f.EXTRA:
                self.assertEqual(got['arrays'][key].tobytes(), full['arrays'][key][:n].tobytes())
        self.assertEqual((meta, data), before)

    def test_audit_has_new_measurement_no_cash_control_or_nonfinite_values(self):
        meta, data = fixture()
        got = f.build_features(meta, data)
        self.assertEqual(got['audit']['measurement'], f.MEASUREMENT)
        self.assertTrue(got['audit']['common_mask_identical_to146'])
        self.assertNotIn('cash_flow_bounded', got['audit']['cells'][0])
        self.assertIn('low_asset_growth_bounded', got['audit']['cells'][0])
        json.dumps(got['audit'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
