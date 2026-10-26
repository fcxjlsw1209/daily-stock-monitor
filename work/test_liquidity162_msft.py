"""Synthetic-only tests: no issuer data or frozen output reads."""
import copy
from html import escape
from pathlib import Path
import unittest
from unittest import mock

import liquidity162_msft as m
import profitability141_issuer_statements as helpers


EVENT = {'ticker': 'MSFT', 'fiscal_year': 2025, 'fiscal_quarter': 4,
         'fiscal_period_end': '2025-06-30'}
LIQUID = 'Total cash, cash equivalents, and short-term investments'


def fixture(caption='BALANCE SHEETS (In millions)', missing=(), values=None,
            spacer=False, reverse=False, split_dollar=False, extra=(),
            current='June 30, 2025', previous='June 30, 2024', split_header=False):
    values = values or {}
    data = [
        ('Assets', '', ''),
        ('Current assets:', '', ''),
        ('Cash and cash equivalents', '$20', '$19'),
        ('Short-term investments', '81', '80'),
        (LIQUID, '101', '99'),
        ('Total current assets', '140', '138'),
        ('Property and equipment, net', '60', '57'),
        ('Total assets', '$200', '$195'),
        ("Liabilities and stockholders' equity", '', ''),
        ('Current liabilities:', '', ''),
        ('Short-term debt', '12', '11'),
        ('Current portion of long-term debt', '8', '7'),
        ('Accounts payable', '30', '29'),
    ] + list(extra) + [
        ('Total current liabilities', '50', '47'),
        ('Long-term debt', '70', '68'),
        ("Stockholders' equity:", '', ''),
        ('Total stockholders equity', '80', '80'),
        ("Total liabilities and stockholders' equity", '$200', '$195'),
    ]
    first, second = (previous, current) if reverse else (current, previous)
    span = ' colspan="2"' if split_dollar else ''
    middle = '<th></th>' if spacer else ''
    if split_header:
        header = ('<tr><th rowspan="2"></th><th>June 30,</th><th>June 30,</th></tr>'
                  '<tr><th>2025</th><th>2024</th></tr>')
    else:
        header = '<tr><th></th><th%s>%s</th>%s<th%s>%s</th></tr>' % (
            span, first, middle, span, second)
    rows = []
    for label, a, b in data:
        if label in missing:
            continue
        a = values.get(label, a)
        pairs = (b, a) if reverse else (a, b)
        raw_cells = []
        for i, value in enumerate(pairs):
            if i and spacer:
                raw_cells.append('')
            if split_dollar:
                raw_cells.extend(('$' if value.startswith('$') else '', value.lstrip('$')))
            else:
                raw_cells.append(value)
        rows.append('<tr><td>' + escape(label) + '</td>' + ''.join(
            '<td>' + escape(value) + '</td>' for value in raw_cells) + '</tr>')
    return '<table><caption>' + caption + '</caption>' + header + ''.join(rows) + '</table>'


class MsftExtractionTests(unittest.TestCase):
    def extract(self, html=None, event=None):
        return m.extract(fixture() if html is None else html, EVENT if event is None else event)

    def assertRejected(self, result, substring):
        self.assertEqual(result['status'], 'rejected')
        self.assertTrue(any(substring in reason for reason in result['reasons']), result['reasons'])

    def test_reported_aggregate_and_components_no_sum(self):
        result = self.extract()
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        self.assertEqual(result['fields']['liquid_financial_assets']['value_reported'], '101')
        self.assertEqual(result['current_debt_components'],
                         ['short_term_debt', 'current_portion_long_term_debt'])
        self.assertEqual(result['noncurrent_debt_components'], ['long_term_debt'])
        self.assertEqual(result['fields']['total_assets']['section'], 'all_assets')
        self.assertEqual(result['fields']['long_term_debt']['section'], 'noncurrent_liabilities')
        self.assertEqual(result['statement']['currency_status'], 'dollar_symbol_only')
        self.assertEqual(result['statement']['header_rows'], [['', 'June 30, 2025', 'June 30, 2024']])
        self.assertEqual(result['statement']['scale'], 1000000)
        self.assertEqual(result['fields']['total_assets']['value_reported'], '200')

    def test_old_spacer_and_reversed_period_order(self):
        for options, column in [({'spacer': True}, 1), ({'reverse': True}, 2),
                                ({'spacer': True, 'reverse': True}, 3)]:
            with self.subTest(options=options):
                result = self.extract(fixture(**options))
                self.assertEqual(result['status'], 'candidate', result['reasons'])
                field = result['fields']['short_term_debt']
                self.assertEqual(field['value_reported'], '12')
                self.assertEqual(field['selected_columns'], [column])
                self.assertEqual(field['selected_cells'], [column])

    def test_split_currency_cells_and_colspan_header(self):
        result = self.extract(fixture(split_dollar=True))
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        field = result['fields']['total_assets']
        self.assertEqual(field['raw_cells'], ['Total assets', '$', '200', '$', '195'])
        self.assertEqual(field['raw_value'], '$ 200')
        self.assertEqual(field['selected_columns'], [1, 2])
        self.assertEqual(field['selected_cells'], [1, 2])
        self.assertEqual(field['value_reported'], '200')

    def test_split_dates_with_rowspan_label(self):
        result = self.extract(fixture(split_header=True))
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        self.assertEqual(result['statement']['selected_period'], '2025-06-30')
        self.assertEqual(len(result['statement']['header_rows']), 2)

    def test_blank_and_dash_are_unavailable_but_explicit_zero_valid(self):
        for token in ['', '-', '\u2013', '\u2014', '$ -']:
            with self.subTest(token=token):
                result = self.extract(fixture(values={'Short-term debt': token}))
                self.assertRejected(result, 'short_term_debt:unavailable')
                self.assertIsNone(result['fields']['short_term_debt'])
                self.assertEqual(result['current_debt_components'], ['current_portion_long_term_debt'])
        result = self.extract(fixture(values={'Short-term debt': '0'}))
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['fields']['short_term_debt']['value_reported'], '0')

    def test_all_required_missing_fields_remain_none(self):
        for key, aliases in m.FIELD_LABELS.items():
            label = next(label for label in (LIQUID, 'Short-term debt',
                'Current portion of long-term debt', 'Long-term debt', 'Total assets')
                if m._label(label) in aliases)
            with self.subTest(key=key):
                result = self.extract(fixture(missing=(label,)))
                self.assertEqual(result['status'], 'rejected')
                self.assertIsNone(result['fields'][key])

    def test_current_portion_cannot_replace_missing_short_debt(self):
        result = self.extract(fixture(missing=('Short-term debt',)))
        self.assertRejected(result, 'short_term_debt:missing')
        self.assertEqual(result['fields']['current_portion_long_term_debt']['value_reported'], '8')

    def test_liquid_components_not_used_as_fallback(self):
        result = self.extract(fixture(missing=(LIQUID,)))
        self.assertRejected(result, 'liquid_financial_assets:missing')
        self.assertIsNone(result['fields']['liquid_financial_assets'])

    def test_duplicate_field_is_ambiguous(self):
        result = self.extract(fixture(extra=[('Short-term debt', '4', '3')]))
        self.assertRejected(result, 'short_term_debt:missing_or_ambiguous')
        self.assertIsNone(result['fields']['short_term_debt'])

    def test_invalid_numeric_tokens_and_negative_balances(self):
        for token in ['10%', '1,23', 'NaN', 'Infinity', '1 2', '(12)', '-1']:
            with self.subTest(token=token):
                result = self.extract(fixture(values={'Short-term debt': token}))
                self.assertRejected(result, 'short_term_debt:')
                self.assertIsNone(result['fields']['short_term_debt'])
        result = self.extract(fixture(values={'Short-term debt': '1,234.50'}))
        self.assertEqual(result['fields']['short_term_debt']['value_reported'], '1234.50')

    def test_future_and_missing_current_headers_rejected(self):
        for options in [dict(current='June 30, 2024', previous='June 30, 2023'),
                        dict(previous='June 30, 2026')]:
            with self.subTest(options=options):
                self.assertRejected(self.extract(fixture(**options)), 'current_period')

    def test_duplicate_current_columns_do_not_choose_first(self):
        result = self.extract(fixture(previous='June 30, 2025'))
        self.assertRejected(result, 'unavailable_or_ambiguous_current_value')

    def test_current_cell_spanning_comparative_column_rejected(self):
        html = fixture().replace('<td>12</td><td>11</td>', '<td colspan="2">12</td>')
        result = self.extract(html)
        self.assertRejected(result, 'short_term_debt:current_cell_crosses_period_boundary')

    def test_unknown_or_conflicting_units_rejected(self):
        for caption in ['BALANCE SHEETS', 'BALANCE SHEETS (In billions)',
                        'BALANCE SHEETS (In millions and in thousands)']:
            with self.subTest(caption=caption):
                self.assertRejected(self.extract(fixture(caption=caption)), 'scale')
        result = self.extract(fixture(caption='BALANCE SHEETS (In thousands)'))
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['statement']['scale'], 1000)
        self.assertEqual(result['fields']['total_assets']['value_reported'], '200')

    def test_conflicting_unit_in_body_rejected(self):
        result = self.extract(fixture(extra=[('Other liability (in thousands)', '2', '1')]))
        self.assertRejected(result, 'scale')

    def test_no_dollar_symbol_and_explicit_usd(self):
        self.assertRejected(self.extract(fixture().replace('$', '')), 'dollar_symbol_missing')
        result = self.extract(fixture(caption='BALANCE SHEETS (In millions of U.S. dollars)'))
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['statement']['currency_status'], 'explicit_usd')

    def test_explicit_competing_and_mixed_currencies_rejected(self):
        for unit in ['CAD', 'AUD', 'Canadian dollars', 'Australian dollars',
                     'C$', 'A$', 'CA$', 'AU$', 'NZD', 'HK$', 'EUR',
                     'USD and CAD', 'U.S. dollars and Australian dollars', '£']:
            with self.subTest(unit=unit):
                result = self.extract(fixture(caption='BALANCE SHEETS (In millions of ' + unit + ')'))
                self.assertRejected(result, 'explicit_non_usd_or_mixed_currency')
                self.assertEqual(result['statement']['currency_status'], 'explicit_non_usd_or_mixed')
                self.assertTrue(all(value is None for value in result['fields'].values()))
                self.assertTrue(any(note['code'] == 'explicit_competing_currency'
                                    for note in result['notes']))

    def test_competing_currency_in_same_table_body_rejected(self):
        result = self.extract(fixture(extra=[('Loan denomination: CAD', '2', '1')]))
        self.assertRejected(result, 'explicit_non_usd_or_mixed_currency')

    def test_currency_evidence_does_not_leak_across_tables(self):
        html = ('<p>Canadian dollars are discussed elsewhere.</p>'
                '<table><tr><td>CAD</td><td>1</td></tr></table>' + fixture())
        result = self.extract(html)
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        self.assertEqual(result['statement']['currency_status'], 'dollar_symbol_only')
        result = self.extract(fixture(caption='BALANCE SHEETS (In millions of US$)'))
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        self.assertEqual(result['statement']['currency_status'], 'explicit_usd')

    def test_explicit_zero_liquid_assets_and_total_assets_are_source_values(self):
        result = self.extract(fixture(values={LIQUID: '0', 'Total assets': '$0'}))
        self.assertEqual(result['status'], 'candidate', result['reasons'])
        self.assertEqual(result['fields']['liquid_financial_assets']['value_reported'], '0')
        self.assertEqual(result['fields']['total_assets']['value_reported'], '0')

    def test_cashflow_nongaap_and_duration_tables_rejected(self):
        for caption in ['STATEMENTS OF CASH FLOWS (In millions)',
                        'Non-GAAP BALANCE SHEETS (In millions)',
                        'BALANCE SHEETS as adjusted (In millions)',
                        'BALANCE SHEETS STATEMENTS OF CASH FLOWS (In millions)']:
            with self.subTest(caption=caption):
                self.assertRejected(self.extract(fixture(caption=caption)), 'balance_sheet')
        self.assertRejected(self.extract(fixture(current='Twelve Months Ended June 30, 2025')),
                            'duration_header')

    def test_cashflow_decoy_ignored_and_multiple_gaap_rejected(self):
        result = self.extract(fixture(caption='CASH FLOWS (In millions)') + fixture())
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['statement']['table_index'], 1)
        self.assertRejected(self.extract(fixture() + fixture()), 'ambiguous_gaap')

    def test_missing_boundaries_and_wrong_section_rejected(self):
        self.assertRejected(self.extract(fixture(missing=('Total current liabilities',))),
                            'missing_or_ambiguous_section')
        short = '<tr><td>Short-term debt</td><td>12</td><td>11</td></tr>'
        html = fixture().replace(short, '')
        html = html.replace('<tr><td>Accounts payable', short + '<tr><td>Accounts payable')
        # Move short debt into the asset block, leaving total-assets unchanged.
        html = html.replace(short, '').replace('<tr><td>Total current assets',
                                               short + '<tr><td>Total current assets')
        result = self.extract(html)
        self.assertRejected(result, 'short_term_debt:wrong_statement_section')
        self.assertIsNone(result['fields']['short_term_debt'])

    def test_long_term_debt_cannot_be_in_current_section(self):
        long = '<tr><td>Long-term debt</td><td>70</td><td>68</td></tr>'
        html = fixture().replace(long, '').replace('<tr><td>Total current liabilities',
                                                  long + '<tr><td>Total current liabilities')
        self.assertRejected(self.extract(html), 'long_term_debt:wrong_statement_section')

    def test_securities_lending_and_restricted_cash_require_review(self):
        for label in ['Securities lending payable',
                      'Short-term investments (including securities loaned)',
                      'Restricted cash']:
            with self.subTest(label=label):
                result = self.extract(fixture(extra=[(label, '2', '1')]))
                self.assertRejected(result, 'scope_requires_review')
                self.assertTrue(any(note.get('row_index') is not None for note in result['notes']))
                self.assertEqual(result['fields']['liquid_financial_assets']['value_reported'], '101')

    def test_unmapped_debt_rejected_and_leases_not_added(self):
        for label in ['Commercial paper', 'Other current borrowings', 'Current debt']:
            with self.subTest(label=label):
                self.assertRejected(self.extract(fixture(extra=[(label, '2', '1')])),
                                    'unmapped_debt_scope_requires_review')
        result = self.extract(fixture(extra=[('Current operating lease liabilities', '2', '1')]))
        self.assertEqual(result['status'], 'candidate')
        self.assertTrue(any(note['code'] == 'lease_row_not_added' for note in result['notes']))
        self.assertEqual(result['current_debt_components'],
                         ['short_term_debt', 'current_portion_long_term_debt'])

    def test_event_scope_and_period_validation(self):
        for change in [{'ticker': 'NVDA'}, {'fiscal_quarter': 3}, {'fiscal_year': '2025'},
                       {'fiscal_year': 2014}, {'fiscal_period_end': '2025-03-31'},
                       {'fiscal_period_end': None}]:
            with self.subTest(change=change):
                event = dict(EVENT, **change)
                self.assertEqual(self.extract(event=event)['status'], 'rejected')
        self.assertEqual(m.extract(fixture(), None)['status'], 'rejected')
        self.assertRejected(self.extract(''), 'empty_or_nontext')

    def test_label_hyphens_and_raw_physical_text(self):
        html = fixture().replace('Short-term debt', 'Short-<b>term</b> debt')
        result = self.extract(html)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['fields']['short_term_debt']['label'], 'Short- term debt')

    def test_external_old_style_statement_context(self):
        html = fixture().replace('<caption>BALANCE SHEETS (In millions)</caption>', '')
        html = '<p>BALANCE SHEETS</p><p>(In millions)</p><div class="c-table">' + html + '</div>'
        self.assertEqual(self.extract(html)['status'], 'candidate')

    def test_pure_function_and_import_have_no_data_io(self):
        html, event = fixture(), copy.deepcopy(EVENT)
        source = Path(m.__file__).read_text()
        with mock.patch('builtins.open', side_effect=AssertionError('no data I/O')), \
             mock.patch.object(Path, 'read_text', side_effect=AssertionError('no data I/O')), \
             mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('no data I/O')):
            result = m.extract(html, event)
            with mock.patch.object(helpers, 'exact_amount', side_effect=AssertionError('no import extraction')):
                namespace = {'__name__': 'synthetic_import', '__file__': m.__file__}
                exec(compile(source, m.__file__, 'exec'), namespace)
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(event, EVENT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
