"""Source metadata and positivity checks on generated records only."""
from copy import deepcopy
import unittest
import monthly_asset_growth147_source_use as s


def fixture():
    periods = {
        'current': dict(period_start='2025-01-01', period_end='2025-12-31', days_inclusive=365),
        'comparison': dict(period_start='2024-01-01', period_end='2024-12-31', days_inclusive=366)}
    source = dict(path='generated.html', sha256='a'*64)
    event = dict(event_id='synthetic', symbol='AAPL', fiscal_year=2025,
        release_date='2026-01-30', available_from='2026-02-02', source=source, periods=periods,
        identity=dict(issuer_id='synthetic', security_id='AAPL',
                      identity_valid_from='2026-02-02', identity_valid_through='2027-07-04'),
        inherited_rejection_codes=[], prior_field_admitted=True, metadata_gaps=[],
        cash_quality_admission=dict(admitted=True, reasons=[], definition=s.prior.DEFINITION))
    event['fields'] = {}
    for side in periods:
        event['fields'][side] = {}
        for name in ('total_assets', 'net_income', 'operating_cash_flow'):
            locator = dict(period_end=periods[side]['period_end'], currency_literal='$', scale='1',
                ni_scope='indirect_method_consolidated_start', format='html', statement_title='CASH FLOWS',
                table_index_zero_based=1, logical_columns=[1])
            event['fields'][side][name] = dict(source=deepcopy(source), period=deepcopy(periods[side]),
                available_from='2026-02-02', currency_literal='$', currency_iso=None,
                value_decimal='100', scale='1', fact_admitted=False, locator=locator)
    return event, ['2026-01-30', '2026-02-02', '2026-02-03']


class SourceUseTests(unittest.TestCase):
    def test_positive_endpoints_and_no_new_growth_fields(self):
        event, sessions = fixture()
        before = deepcopy(event)
        got = s.review_asset_event(event, sessions)
        self.assertTrue(got['conditional_asset_use'])
        self.assertEqual(event, before)
        self.assertNotIn('growth', got)
        self.assertNotIn('value_decimal', str(got))

    def test_nonpositive_or_nonfinite_endpoint_rejected(self):
        for bad in ('0', '-1', 'NaN', 'Infinity'):
            event, sessions = fixture()
            event['fields']['comparison']['total_assets']['value_decimal'] = bad
            with self.assertRaises(ValueError):
                s.review_asset_event(event, sessions)

    def test_same_source_date_and_period_required(self):
        for key, bad in [('source', {}), ('period', {}), ('available_from', '2026-02-03')]:
            event, sessions = fixture()
            event['fields']['current']['total_assets'][key] = bad
            with self.assertRaises(ValueError):
                s.review_asset_event(event, sessions)

    def test_next_session_and_identity_expiry(self):
        for key, value in [('available_from', '2026-02-03'), ('release_date', '2026-02-02')]:
            event, sessions = fixture()
            event[key] = value
            with self.assertRaises(ValueError):
                s.review_asset_event(event, sessions)
        event, sessions = fixture()
        event['identity']['identity_valid_through'] = '2027-07-05'
        with self.assertRaises(ValueError):
            s.review_asset_event(event, sessions)

    def test_old_hold_is_preserved_even_with_both_assets(self):
        event, sessions = fixture()
        event['prior_field_admitted'] = False
        event['cash_quality_admission'].update(admitted=False, reasons=['preserved_inherited_source_hold'])
        self.assertFalse(s.review_asset_event(event, sessions)['conditional_asset_use'])
        event['cash_quality_admission'].update(admitted=True, reasons=[])
        with self.assertRaises(ValueError):
            s.review_asset_event(event, sessions)

    def test_missing_current_flow_still_held(self):
        event, sessions = fixture()
        event['fields']['current']['net_income'] = None
        event['cash_quality_admission'].update(admitted=False, reasons=['missing_required:current.net_income'])
        self.assertFalse(s.review_asset_event(event, sessions)['conditional_asset_use'])

    def test_comparison_flow_missing_does_not_add_hold(self):
        event, sessions = fixture()
        event['fields']['comparison']['net_income'] = None
        event['fields']['comparison']['operating_cash_flow'] = None
        self.assertTrue(s.review_asset_event(event, sessions)['conditional_asset_use'])

    def test_unknown_prior_duration_requires_existing_hold(self):
        event, sessions = fixture()
        event['periods']['comparison']['period_start'] = None
        for cell in event['fields']['comparison'].values():
            cell['period'] = deepcopy(event['periods']['comparison'])
        event['cash_quality_admission'].update(admitted=False, reasons=['period_identity_or_clock:unknown'])
        self.assertFalse(s.review_asset_event(event, sessions)['conditional_asset_use'])

    def test_currency_scale_and_year_end_locator_required(self):
        for key, bad in [('currency_literal', 'EUR'), ('scale', '-1'), ('period_end', '2024-12-30')]:
            event, sessions = fixture()
            event['fields']['comparison']['total_assets']['locator'][key] = bad
            with self.assertRaises(ValueError):
                s.review_asset_event(event, sessions)


if __name__ == '__main__':
    unittest.main()
