"""Synthetic-only NVDA balance-sheet parser regressions; no source file reads."""
from copy import deepcopy
import unittest

import liquidity162_nvda as parser


def fixture(short='Short-term debt', amount='12', units='In millions', year=2026, extra='', late=False):
    pad = '<td></td>' if late else ''
    width = 9 if late else 8
    def line(label, current='21', prior='91'):
        return ('<tr><td colspan="3">' + label + '</td>' + pad +
                '<td>$</td><td>' + current + '</td><td></td><td>$</td><td>' + prior + '</td></tr>')
    def full(text):
        return '<tr><td colspan="%d">%s</td></tr>' % (width, text)
    dates = ('<tr><td colspan="3"></td>' + pad + '<td colspan="2">January 31</td><td></td>'
             '<td colspan="2">January 31</td></tr>')
    years = ('<tr><td colspan="3"></td>' + pad + '<td colspan="2">%d</td><td></td>'
             '<td colspan="2">%d</td></tr>') % (year, year - 1)
    html = ('<table>' + full('NVIDIA CORPORATION') + full('CONDENSED CONSOLIDATED BALANCE SHEETS') +
            full(units) + full('(Unaudited)') + dates + years + full('ASSETS') +
            full('Current assets:') + line('Cash, cash equivalents and marketable securities', '1,234') +
            line('Total current assets', '1,500') + line('Total assets', '2,000') +
            full('LIABILITIES AND SHAREHOLDERS’ EQUITY') + full('Current liabilities:') +
            line('Accounts payable') + (line(short, amount) if short else '') + extra +
            line('Total current liabilities', '200') + line('Long-term debt', '345') +
            line('Long-term operating lease liabilities', '50') + line('Total liabilities', '600') +
            line('Convertible debt conversion obligation', '77') + line('Shareholders’ equity', '1,400') +
            line('Total liabilities and shareholders’ equity', '2,000') + '</table>')
    return html, dict(ticker='NVDA', fiscal_year=year, fiscal_quarter=4, fiscal_period_end='%d-01-31' % year), line


class ParserTests(unittest.TestCase):
    def test_current_column_and_exact_decimal(self):
        html, event, _ = fixture()
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['fields']['short_term_debt']['value_reported'], '12')
        self.assertEqual(got['fields']['liquid_financial_assets']['value_reported'], '1234')
        self.assertEqual(got['fields']['total_assets']['section'], 'all_assets')
        self.assertEqual(got['statement']['scale'], 1000000)
        self.assertEqual(got['statement']['currency_status'], 'dollar_symbol_only')
        self.assertEqual(got['statement']['selected_period'], event['fiscal_period_end'])
        self.assertEqual(got['current_debt_components'], ['short_term_debt'])
        self.assertEqual(got['noncurrent_debt_components'], ['long_term_debt'])
        self.assertIsNone(got['fields']['current_portion_long_term_debt'])
        self.assertEqual(got['fields']['short_term_debt']['selected_cells'], [2])
        self.assertEqual(got['fields']['short_term_debt']['selected_columns'], [4])
    def test_late_padding_geometry(self):
        html, event, _ = fixture(late=True)
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['fields']['short_term_debt']['selected_columns'], [5])
    def test_convertible_current_debt_and_thousands(self):
        html, event, _ = fixture(short='Convertible short-term debt', units='(In thousands)', year=2016)
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['statement']['scale'], 1000)
        self.assertEqual(got['fields']['short_term_debt']['label'], 'Convertible short-term debt')
    def test_convertible_scope_note_and_excluded_conversion_obligation(self):
        html, event, _ = fixture(short='Convertible short-term debt', year=2016)
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertIn('Convertible short-term debt is the reported current carrying amount; '
                      'it is not certified contractual principal due within one year.', got['notes'])
        self.assertTrue(any('Excluded financing row' in note and
                            'Convertible debt conversion obligation' in note for note in got['notes']))
        self.assertEqual(got['current_debt_components'], ['short_term_debt'])
        self.assertEqual(got['noncurrent_debt_components'], ['long_term_debt'])
    def test_missing_short_debt_not_zero(self):
        for year in (2015, 2020):
            html, event, _ = fixture(short=None, year=year)
            got = parser.extract(html, event)
            self.assertEqual(got['status'], 'rejected')
            self.assertIsNone(got['fields']['short_term_debt'])
            self.assertIn('missing_field:short_term_debt', got['reasons'])
            self.assertEqual(got['current_debt_components'], [])
    def test_all_dash_tokens_unavailable_but_raw_retained(self):
        for token in ('-', '—', '–', '(—)', 'N/A'):
            html, event, _ = fixture(amount=token)
            got = parser.extract(html, event)
            self.assertEqual(got['status'], 'rejected')
            self.assertIsNone(got['fields']['short_term_debt']['value_reported'])
            self.assertEqual(got['fields']['short_term_debt']['raw_value'], token)
    def test_explicit_zero_retained(self):
        html, event, _ = fixture(amount='0')
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['fields']['short_term_debt']['value_reported'], '0')
    def test_negative_balance_rejected_but_preserved(self):
        for token in ('-12', '(12)'):
            html, event, _ = fixture(amount=token)
            got = parser.extract(html, event)
            self.assertEqual(got['status'], 'rejected')
            self.assertEqual(got['fields']['short_term_debt']['value_reported'], '-12')
            self.assertIn('negative_balance_requires_review:short_term_debt', got['reasons'])
    def test_malformed_input_rejected_without_side_effect(self):
        html, event, _ = fixture()
        for bad in (None, [], 'NVDA'):
            self.assertEqual(parser.extract(html, bad)['reasons'], ['invalid_input'])
        self.assertEqual(parser.extract(None, event)['reasons'], ['invalid_input'])
    def test_blank_not_prior_value(self):
        html, event, _ = fixture(amount='')
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'rejected')
        self.assertIsNone(got['fields']['short_term_debt'])
    def test_duplicate_short_row_is_ambiguous(self):
        html, event, line = fixture()
        html = html.replace(line('Short-term debt', '12'), line('Short-term debt', '12') * 2)
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'rejected')
        self.assertIn('ambiguous_or_wrong_section:short_term_debt', got['reasons'])
    def test_current_portion_overlap_pauses(self):
        _, _, line = fixture()
        html, event, _ = fixture(extra=line('Current portion of long-term debt', '3'))
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'rejected')
        self.assertIn('current_portion_overlap_requires_review', got['reasons'])
        self.assertEqual(got['fields']['current_portion_long_term_debt']['value_reported'], '3')
        self.assertNotIn('current_portion_long_term_debt', got['current_debt_components'])
    def test_wrong_liability_section_rejected(self):
        html, event, line = fixture()
        html = html.replace(line('Short-term debt', '12'), '')
        html = html.replace(line('Long-term debt', '345'), line('Short-term debt', '12') + line('Long-term debt', '345'))
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'rejected')
        self.assertIn('ambiguous_or_wrong_section:short_term_debt', got['reasons'])
    def test_unknown_financing_not_silently_excluded(self):
        _, _, line = fixture()
        html, event, _ = fixture(extra=line('Commercial paper', '9'))
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'rejected')
        self.assertTrue(any(x.startswith('unclassified_financing_row:') for x in got['reasons']))
    def test_lease_and_after_total_conversion_excluded(self):
        html, event, _ = fixture()
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['noncurrent_debt_components'], ['long_term_debt'])
        self.assertTrue(any('lease liabilities' in x for x in got['notes']))
        self.assertTrue(any('conversion obligation' in x for x in got['notes']))
    def test_old_equity_boundary_without_total_liabilities(self):
        html, event, line = fixture()
        html = html.replace(line('Total liabilities', '600'), '').replace(line('Convertible debt conversion obligation', '77'), '')
        got = parser.extract(html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
    def test_no_wrong_cashflow_or_non_gaap_fallback(self):
        html, event, _ = fixture()
        cf = html.replace('BALANCE SHEETS', 'STATEMENTS OF CASH FLOWS').replace('1,234', '9,999')
        got = parser.extract(cf + html, event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['statement']['table_index'], 1)
        self.assertEqual(got['fields']['liquid_financial_assets']['value_reported'], '1234')
        self.assertEqual(parser.extract(html.replace('BALANCE SHEETS', 'NON-GAAP BALANCE SHEETS'), event)['status'], 'rejected')
    def test_ambiguous_tables_no_best_table_selection(self):
        html, event, _ = fixture()
        got = parser.extract(html + html, event)
        self.assertEqual(got['reasons'], ['missing_or_ambiguous_balance_sheet'])
    def test_wrong_or_ambiguous_current_period(self):
        html, event, _ = fixture()
        event['fiscal_period_end'] = '2026-01-30'
        self.assertEqual(parser.extract(html, event)['status'], 'rejected')
        event['fiscal_period_end'] = '2026-01-31'
        self.assertEqual(parser.extract(html.replace('2025', '2026'), event)['status'], 'rejected')
    def test_missing_and_conflicting_units(self):
        for units in ('Unaudited', 'In millions and in thousands'):
            html, event, _ = fixture(units=units)
            self.assertEqual(parser.extract(html, event)['status'], 'rejected')
    def test_no_unstated_usd(self):
        html, event, _ = fixture(units='In millions of U.S. dollars')
        self.assertEqual(parser.extract(html, event)['statement']['currency_status'], 'explicit_usd')
        html, event, _ = fixture(units='In millions of Canadian dollars')
        self.assertEqual(parser.extract(html, event)['status'], 'rejected')
    def test_header_and_raw_cells_preserved_no_input_mutation(self):
        html, event, _ = fixture()
        before = deepcopy(event)
        got = parser.extract(html, event)
        self.assertEqual(event, before)
        self.assertEqual(got['statement']['header_rows'][4], ['', 'January 31', '', 'January 31'])
        self.assertEqual(got['fields']['short_term_debt']['raw_cells'], ['Short-term debt', '$', '12', '', '$', '91'])
    def test_layout_parent_not_double_counted(self):
        html, event, _ = fixture()
        got = parser.extract('<table><tr><td>' + html + '</td></tr></table>', event)
        self.assertEqual(got['status'], 'candidate', got['reasons'])
        self.assertEqual(got['statement']['table_index'], 1)
    def test_missing_current_assets_and_wrong_scope(self):
        html, event, _ = fixture()
        self.assertEqual(parser.extract(html.replace('Current assets:', 'Assets:'), event)['status'], 'rejected')
        for key, value in (('ticker', 'MSFT'), ('fiscal_quarter', 3), ('fiscal_year', '2026')):
            changed = dict(event);changed[key] = value
            self.assertEqual(parser.extract(html, changed)['status'], 'rejected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
