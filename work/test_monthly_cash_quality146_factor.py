"""Generated inputs only: arithmetic, source contract, causal masks and axes."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import localcontext
from fractions import Fraction
import json
import unittest

import numpy as np
import pandas as pd
import monthly_cash_quality146_factor as f


def period(start, end):
    return dict(period_start=start, period_end=end,
                days_inclusive=(date.fromisoformat(end)-date.fromisoformat(start)).days+1)


def event(identifier='generated', symbol='AAPL'):
    periods = dict(current=period('2025-01-01', '2025-12-31'),
                   comparison=period('2024-01-01', '2024-12-31'))
    source = dict(path='synthetic/report.html', sha256='a'*64, url='https://example.invalid/report')
    record = dict(
        event_id=identifier, symbol=symbol, fiscal_year=2025,
        release_date='2026-01-30', available_from='2026-02-02',
        identity=dict(issuer_id='synthetic_'+symbol, security_id=symbol,
                      identity_valid_from='2026-02-02',
                      identity_valid_through=str(date(2025, 12, 31)+timedelta(days=550))),
        source=source, periods=periods, inherited_rejection_codes=[],
        fact_admitted=False, model_admitted=False, strategy_admitted=False,
        cash_quality_admission=dict(admitted=True, reasons=[], definition=f.DEFINITION))
    record['fields'] = {side: {
        name: dict(value_decimal=value, scale='1', currency_literal='$', currency_iso=None,
                   source=deepcopy(source), period=deepcopy(periods[side]),
                   available_from=record['available_from'], fact_admitted=False)
        for name, value in [('operating_cash_flow', '30'), ('net_income', '10'),
                            ('total_assets', '150' if side == 'current' else '50')]
    } for side in ('current', 'comparison')}
    return record


def fixture(dates=None):
    names = ['AAPL', 'AMD', 'MSFT', 'NVDA']+['S'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
    dates = dates if dates is not None else ['2026-03-02', '2026-04-01']
    return (dict(dates=dates, symbols=[names[:] for _ in dates], etfs=names[32:]),
            dict(schema_version=1, method=f.METHOD, events=[event()]))


def change_periods(record, current_days, prior_days):
    end = date(2025, 12, 31)
    start = end-timedelta(days=current_days-1)
    previous_end = start-timedelta(days=1)
    record['periods'] = dict(current=period(str(start), str(end)),
                            comparison=period(str(previous_end-timedelta(days=prior_days-1)), str(previous_end)))
    for side in f.SIDES:
        for cell in record['fields'][side].values():
            cell['period'] = deepcopy(record['periods'][side])


class ReportTests(unittest.TestCase):
    def evaluate(self, record=None, signal='2026-03-02'):
        return f.evaluate_report(event() if record is None else record, signal)

    def test_exact_average_assets_and_three_definitions(self):
        got = self.evaluate()
        self.assertTrue(got['available'])
        self.assertEqual(got['earnings_bounded'], float(Fraction(1, 11)))
        self.assertEqual(got['cash_flow_bounded'], float(Fraction(3, 13)))
        self.assertEqual(got['low_accrual_bounded'], float(Fraction(1, 6)))

    def test_quality_is_h_of_difference_not_difference_of_h(self):
        got = self.evaluate()
        self.assertNotEqual(got['low_accrual_bounded'], got['cash_flow_bounded']-got['earnings_bounded'])

    def test_old_raw_fact_flags_remain_false_but_new_admission_can_pass(self):
        record = event()
        self.assertFalse(record['fact_admitted'])
        self.assertTrue(self.evaluate(record)['available'])

    def test_raw_complete_or_old_admission_never_substitutes_new_admission(self):
        record = event()
        del record['cash_quality_admission']
        record['fact_admitted'] = True
        self.assertEqual(self.evaluate(record)['reason'], 'cash_quality_admission_required')

    def test_strict_admission_schema_and_definition(self):
        for value in (False, None, 1, 'true'):
            record = event()
            record['cash_quality_admission']['admitted'] = value
            self.assertFalse(self.evaluate(record)['available'])
        for change in ({'extra': True}, {'definition': 'old_gp'}, {'reasons': ['hold']}, {'reasons': {}}):
            record = event()
            record['cash_quality_admission'].update(change)
            self.assertFalse(self.evaluate(record)['available'])

    def test_old_rejection_cannot_be_silently_cleared_by_mask(self):
        record = event()
        record['inherited_rejection_codes'] = ['prior_period_unknown']
        self.assertEqual(self.evaluate(record)['reason'], 'inherited_rejection_not_cleared')
        del record['inherited_rejection_codes']
        self.assertFalse(self.evaluate(record)['available'])

    def test_comparison_flows_not_required_or_used(self):
        record = event()
        expected = self.evaluate(record)
        record['fields']['comparison']['net_income'] = None
        record['fields']['comparison']['operating_cash_flow'] = {'ignored': 'never used'}
        self.assertEqual(self.evaluate(record), expected)

    def test_signed_and_zero_current_flows(self):
        for ni, cf in [('-10', '-30'), ('0', '0'), ('-10', '30'), ('30', '-10')]:
            record = event()
            record['fields']['current']['net_income']['value_decimal'] = ni
            record['fields']['current']['operating_cash_flow']['value_decimal'] = cf
            got = self.evaluate(record)
            self.assertTrue(got['available'])
            expected = (Fraction(ni)/100, Fraction(cf)/100, (Fraction(cf)-Fraction(ni))/100)
            for name, value in zip(f.EXTRA[:-1], expected):
                self.assertEqual(got[name], float(value/(1+abs(value))))

    def test_separate_scales_normalize_exactly(self):
        record = event()
        expected = self.evaluate(record)
        for side, name, text, scale in [('current', 'net_income', '.01', '1000'),
                                        ('current', 'operating_cash_flow', '30', '1'),
                                        ('current', 'total_assets', '.00015', '1000000'),
                                        ('comparison', 'total_assets', '.05', '1000')]:
            record['fields'][side][name].update(value_decimal=text, scale=scale)
        self.assertEqual(self.evaluate(record), expected)

    def test_exact_cancellation_before_float_conversion(self):
        record = event()
        huge = '1'+'0'*80
        record['fields']['current']['net_income']['value_decimal'] = huge
        record['fields']['current']['operating_cash_flow']['value_decimal'] = str(int(huge)+1)
        for side in f.SIDES:
            record['fields'][side]['total_assets']['value_decimal'] = huge
        with localcontext() as ctx:
            ctx.prec = 3
            got = self.evaluate(record)
            self.assertEqual(ctx.prec, 3)
        self.assertEqual(got['earnings_bounded'], got['cash_flow_bounded'])
        self.assertEqual(got['low_accrual_bounded'], float(Fraction(1, int(huge)+1)))
        self.assertGreater(got['low_accrual_bounded'], 0)

    def test_huge_scales_do_not_overflow(self):
        record = event()
        for side, name in f.REQUIRED:
            record['fields'][side][name]['scale'] = '1e1000'
        self.assertEqual(self.evaluate(record), self.evaluate())

    def test_mapping_is_signed_monotone_and_bounded(self):
        values = [Fraction(-10**1000), Fraction(-10), Fraction(-1), Fraction(0),
                  Fraction(1), Fraction(10), Fraction(10**1000)]
        mapped = [f.bounded(x) for x in values]
        self.assertEqual(mapped, sorted(mapped))
        self.assertTrue(all(-1 <= x <= 1 for x in mapped))
        self.assertEqual(mapped, [-x for x in reversed(mapped)])
        with self.assertRaises(ValueError):
            f.bounded(0.5)

    def test_asset_endpoints_must_both_be_positive(self):
        for side in f.SIDES:
            for value in ('0', '-1', '-0'):
                record = event()
                record['fields'][side]['total_assets']['value_decimal'] = value
                self.assertEqual(self.evaluate(record)['reason'], 'nonpositive_asset_endpoint')

    def test_missing_required_cell_blocks_all_three(self):
        for side, name in f.REQUIRED:
            record = event()
            record['fields'][side][name] = None
            got = self.evaluate(record)
            self.assertFalse(got['available'])
            self.assertTrue(all(got[key] is None for key in f.EXTRA[:-1]))

    def test_decimal_and_scale_contract(self):
        bad = [True, 1, 1.5, None, 'NaN', 'sNaN', 'Infinity', '-Infinity', ' 1', '1,000', '1_000', '1/2']
        for value in bad:
            for field in ('value_decimal', 'scale'):
                record = event()
                record['fields']['current']['net_income'][field] = value
                self.assertFalse(self.evaluate(record)['available'], (field, value))
        for value in ('0', '-1'):
            record = event()
            record['fields']['current']['net_income']['scale'] = value
            self.assertFalse(self.evaluate(record)['available'])

    def test_no_iso_currency_substitution(self):
        for value in ('USD', 'EUR', None):
            record = event()
            record['fields']['current']['net_income']['currency_literal'] = value
            self.assertFalse(self.evaluate(record)['available'])
        self.assertEqual(self.evaluate()['currency_key'], 'same-report-dollar:generated')

    def test_cell_source_period_and_availability_exact(self):
        for side, name in f.REQUIRED:
            for field in ('source', 'period', 'available_from'):
                record = event()
                cell = record['fields'][side][name]
                cell[field] = deepcopy(cell[field])
                if field == 'source':
                    cell[field]['sha256'] = 'b'*64
                elif field == 'period':
                    cell[field]['period_end'] = '2023-12-31'
                else:
                    cell[field] = '2026-02-03'
                self.assertFalse(self.evaluate(record)['available'], (side, name, field))

    def test_assets_do_not_accept_other_instant_date(self):
        record = event()
        record['fields']['comparison']['total_assets']['period']['period_end'] = '2024-12-30'
        self.assertFalse(self.evaluate(record)['available'])

    def test_annual_range_without_duration_normalization(self):
        for current_days, prior_days in [(330, 330), (400, 400), (364, 371), (371, 364)]:
            record = event()
            change_periods(record, current_days, prior_days)
            got = self.evaluate(record)
            self.assertTrue(got['available'])
            for name in f.EXTRA[:-1]:
                self.assertEqual(got[name], self.evaluate()[name])
        for current_days, prior_days in [(329, 365), (401, 365), (365, 329), (365, 401)]:
            record = event()
            change_periods(record, current_days, prior_days)
            self.assertFalse(self.evaluate(record)['available'])

    def test_unknown_duration_and_declared_duration_mismatch(self):
        for field, value in [('period_start', None), ('period_end', '2024-02-30'),
                             ('days_inclusive', True), ('days_inclusive', 1), ('days_inclusive', '366')]:
            record = event()
            record['periods']['comparison'][field] = value
            self.assertFalse(self.evaluate(record)['available'])

    def test_gap_overlap_and_chronology(self):
        for previous_end in ('2024-12-30', '2025-01-01'):
            record = event()
            record['periods']['comparison'] = period('2024-01-01', previous_end)
            self.assertEqual(self.evaluate(record)['reason'], 'noncontiguous_annual_periods')
        record = event()
        record['release_date'] = '2025-12-30'
        self.assertEqual(self.evaluate(record)['reason'], 'release_before_period_end')

    def test_availability_and_identity_intervals(self):
        self.assertFalse(self.evaluate(signal='2026-01-30')['available'])
        self.assertTrue(self.evaluate(signal='2026-02-02')['available'])
        record = event()
        record['release_date'] = record['available_from']
        self.assertEqual(self.evaluate(record)['reason'], 'availability_not_after_release')
        record = event()
        record['current_release_available_from'] = '2026-02-03'
        self.assertEqual(self.evaluate(record)['reason'], 'current_release_availability_mismatch')

    def test_causal_550_day_expiry_inclusive(self):
        end = date(2025, 12, 31)
        self.assertTrue(self.evaluate(signal=str(end+timedelta(days=550)))['available'])
        self.assertEqual(self.evaluate(signal=str(end+timedelta(days=551)))['reason'], 'stale_current_report')
        for field, value in [('identity_valid_from', '2026-02-03'), ('identity_valid_through', '2030-01-01')]:
            record = event()
            record['identity'][field] = value
            self.assertFalse(self.evaluate(record)['available'])

    def test_identity_and_source_required(self):
        for field in ('issuer_id', 'security_id', 'identity_valid_from', 'identity_valid_through'):
            record = event()
            del record['identity'][field]
            self.assertFalse(self.evaluate(record)['available'])
        record = event()
        record['identity']['security_id'] = 'OTHER'
        self.assertEqual(self.evaluate(record)['reason'], 'security_symbol_mismatch')
        record = event(symbol='UNREGISTERED')
        self.assertFalse(self.evaluate(record)['available'])
        record = event()
        record['source']['sha256'] = 'bad'
        self.assertFalse(self.evaluate(record)['available'])

    def test_exact_raw_field_shape(self):
        for key in ('periods', 'fields'):
            record = event()
            record[key]['extra'] = {}
            self.assertFalse(self.evaluate(record)['available'])
        record = event()
        del record['fields']['comparison']['net_income']
        self.assertFalse(self.evaluate(record)['available'])


class BuilderTests(unittest.TestCase):
    def test_axes_masks_nonmutation_and_json_audit(self):
        meta, data = fixture()
        data['events'].append(event('amd', 'AMD'))
        meta['symbols'][1][0], meta['symbols'][1][1] = meta['symbols'][1][1], meta['symbols'][1][0]
        saved = deepcopy((meta, data))
        got = f.build_features(meta, data)
        self.assertEqual((meta, data), saved)
        for name in f.EXTRA[:-1]:
            np.testing.assert_array_equal(np.isfinite(got['arrays'][name]), got['arrays']['common_available'])
        self.assertEqual(got['audit']['common_available_cells'], 4)
        self.assertEqual(len(got['audit']['cells']), 98)
        self.assertFalse(got['audit']['comparison_flow_values_used'])
        self.assertFalse(got['audit']['annual_duration_normalization'])
        self.assertEqual(got['audit']['new_model_fits'], 0)
        json.dumps(got['audit'], allow_nan=False)

    def test_122_months_and_all_123_prefixes(self):
        meta, data = fixture([str(d.date()) for d in pd.date_range('2016-08-31', periods=122, freq='BME')])
        full = f.build_features(meta, data)
        self.assertEqual(full['arrays']['common_available'].shape, (122, 49))
        self.assertEqual(len(full['audit']['cells']), 5978)
        for n in range(123):
            prefix = f.build_features(dict(meta, dates=meta['dates'][:n], symbols=meta['symbols'][:n]), data)
            for name in f.EXTRA:
                np.testing.assert_array_equal(prefix['arrays'][name], full['arrays'][name][:n])
            self.assertEqual(prefix['audit']['cells'], full['audit']['cells'][:n*49])

    def test_empty_census_means_no_feature_not_zero_financial_amount(self):
        meta, data = fixture()
        data['events'] = []
        got = f.build_features(meta, data)
        self.assertFalse(got['arrays']['common_available'].any())
        self.assertTrue(all(np.isnan(got['arrays'][name]).all() for name in f.EXTRA[:-1]))

    def test_etfs_always_unavailable(self):
        meta, data = fixture()
        data['events'].append(event('etf', 'ETF0'))
        got = f.build_features(meta, data)
        self.assertFalse(got['arrays']['common_available'][:, 32:].any())
        self.assertEqual(got['audit']['cells'][32]['reason'], 'etf_not_applicable')

    def test_latest_rejected_blocks_older_valid(self):
        meta, data = fixture()
        data['events'].append(dict(event_id='later', symbol='AAPL', available_from='2026-02-20',
                                   cash_quality_admission=dict(admitted=False, reasons=['hold'], definition=f.DEFINITION)))
        got = f.build_features(meta, data)
        self.assertFalse(got['arrays']['common_available'][:, 0].any())
        self.assertEqual(got['audit']['cells'][0]['current_event_id'], 'later')
        self.assertEqual(got['audit']['cells'][0]['reason'], 'current_not_cash_quality_admitted')

    def test_latest_missing_cell_no_old_fallback(self):
        meta, data = fixture()
        later = event('later')
        later['available_from'] = later['identity']['identity_valid_from'] = '2026-02-20'
        for side in f.SIDES:
            for cell in later['fields'][side].values():
                cell['available_from'] = '2026-02-20'
        later['fields']['current']['net_income'] = None
        data['events'].append(later)
        got = f.build_features(meta, data)
        self.assertFalse(got['arrays']['common_available'][:, 0].any())
        self.assertEqual(got['audit']['cells'][0]['reason'], 'missing_current_net_income')

    def test_future_append_preserves_every_old_array_and_audit_cell(self):
        meta, data = fixture(['2026-03-02'])
        old = f.build_features(meta, data)
        data['events'].append(dict(event_id='future', symbol='AAPL', available_from='2026-03-20', poison=True))
        new = f.build_features(meta, data)
        self.assertEqual(new['audit'], old['audit'])
        for name in f.EXTRA:
            np.testing.assert_array_equal(new['arrays'][name], old['arrays'][name])
        meta['dates'].append('2026-04-01')
        meta['symbols'].append(meta['symbols'][0][:])
        self.assertFalse(f.build_features(meta, data)['arrays']['common_available'][1, 0])

    def test_duplicate_or_competing_event_rules(self):
        meta, data = fixture()
        data['events'].append(deepcopy(data['events'][0]))
        self.assertEqual(f.build_features(meta, data)['audit']['common_available_cells'], 2)
        data['events'][1]['fields']['current']['net_income']['value_decimal'] = '99'
        self.assertEqual(f.build_features(meta, data)['audit']['cells'][0]['reason'], 'conflicting_event_id')
        meta, data = fixture()
        data['events'].append(event('competing'))
        self.assertEqual(f.build_features(meta, data)['audit']['cells'][0]['reason'], 'current_ambiguous_latest_event')

    def test_event_order_does_not_select_different_report(self):
        meta, data = fixture()
        data['events'].append(dict(event_id='old', symbol='AAPL', available_from='2025-02-01'))
        old = f.build_features(meta, data)
        data['events'].reverse()
        self.assertEqual(f.build_features(meta, data)['audit'], old['audit'])

    def test_unreadable_event_time_fails_not_drops(self):
        for value in (None, '2026-02-30', '2026-02-01T00:00Z'):
            meta, data = fixture()
            data['events'][0]['available_from'] = value
            with self.assertRaises(ValueError):
                f.build_features(meta, data)

    def test_wrong_dataset_contract(self):
        for changes in ({'schema_version': True}, {'schema_version': 2},
                        {'method': 'old_gp'}, {'events': None}):
            meta, data = fixture()
            data.update(changes)
            with self.assertRaises(ValueError):
                f.build_features(meta, data)

    def test_bad_axes_and_etf_pool(self):
        for dates in (['2026-04-01', '2026-03-02'], ['2026-03-02']*2,
                      ['2026-03-02T00:00Z'], ['2026-03-02T01:00']):
            meta, data = fixture(dates)
            with self.assertRaises(ValueError):
                f.build_features(meta, data)
        for mode in ('width', 'duplicate', 'etf', 'length'):
            meta, data = fixture()
            if mode == 'width':
                meta['symbols'][0].pop()
            elif mode == 'duplicate':
                meta['symbols'][0][0] = meta['symbols'][0][1]
            elif mode == 'etf':
                meta['etfs'].pop()
            else:
                meta['symbols'].pop()
            with self.assertRaises(ValueError):
                f.build_features(meta, data)


if __name__ == '__main__':
    unittest.main()
