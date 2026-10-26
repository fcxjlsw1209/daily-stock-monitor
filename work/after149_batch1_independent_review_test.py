"""Pure synthetic tests for the independent audit; no producer imports."""
import sys
import unittest
from unittest.mock import patch
from bs4 import BeautifulSoup
import after149_batch1_independent_review as audit


def fixture(value='$ (1,234.50)', duration='Twelve Months Ended', extra='', span=1):
    raw = '''<table><tr><td colspan="3">CONSOLIDATED STATEMENTS OF CASH FLOWS</td></tr>
    <tr><td colspan="3">(In millions)</td></tr>
    <tr><td></td><td colspan="2">%s</td></tr>
    <tr><td></td><td>July 26, 2025</td><td>July 27, 2024</td></tr>
    <tr><td>Net income</td><td rowspan="%d">%s</td><td>$ 987</td></tr>
    <tr><td>Adjustments to reconcile net income to operating activities</td></tr>
    <tr><td>Net cash provided by operating activities</td><td>$ 555</td></tr>%s</table>''' % (duration, span, value, extra)
    context = {'periods': {'current': {'period_end': '2025-07-26'}, 'comparison': {'period_end': '2024-07-27'}},
               'tables': [{'table': 0, 'title': ['CONSOLIDATED STATEMENTS OF CASH FLOWS'],
                           'labels': [{'row': 4, 'label': 'Net income'}],
                           'columns': [{'column': 1, 'dates': ['2025-07-26'], 'annual': True, 'quarter': False},
                                       {'column': 2, 'dates': ['2024-07-27'], 'annual': True, 'quarter': False}]}]}
    return BeautifulSoup(raw, 'html.parser'), context


class ReviewTests(unittest.TestCase):
    def test_exact_large_signed_amount(self):
        text = '123456789012345678901234567890123456789.1234'
        self.assertEqual(audit.signed_text('($ ' + text + ')'), '-' + text)

    def test_signed_variants(self):
        for raw, wanted in [('$ (1,234.50)', '-1234.50'), ('− 12', '-12'), ('$ 0', '0'), ('1 , 234', '1234')]:
            with self.subTest(raw=raw): self.assertEqual(audit.signed_text(raw), wanted)

    def test_invalid_sign_or_number(self):
        for raw in ('(-1)', '(2', '2)', '1 234', '$ $ 2', 'NaN', 'Infinity', '3%', '-', '', '1,23'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): audit.signed_text(raw)

    def test_dates(self):
        self.assertEqual(audit.parsed_dates('Jan. 27, 2011 and July 30, 2011'), {'2011-01-27', '2011-07-30'})
        self.assertEqual(audit.parsed_dates('February 30, 2024'), set())

    def test_grid_spans(self):
        soup = BeautifulSoup('<table><tr><td rowspan="2">x</td><td colspan="2">y</td></tr><tr><td>a</td><td>b</td></tr></table>', 'html.parser')
        cells, grid = audit.grid_from_html(soup.table)
        self.assertEqual(cells[1, 0]['columns'], [1])
        self.assertEqual(grid[1, 0]['row'], 0)
        self.assertEqual(cells[0, 1]['columns'], [1, 2])

    def test_invalid_span(self):
        soup = BeautifulSoup('<table><tr><td colspan="0">x</td></tr></table>', 'html.parser')
        with self.assertRaises(ValueError): audit.grid_from_html(soup.table)

    def read(self, **kwargs):
        soup, context = fixture(**kwargs)
        return audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')

    def test_independent_complete_field(self):
        value = self.read()
        self.assertEqual(value['value_decimal'], '-1234.50')
        self.assertEqual(value['scale'], '1000000')
        self.assertEqual(value['row'], 4)
        self.assertEqual(value['physical_cells'][0]['physical_cell'], 1)

    def test_zero(self):
        self.assertEqual(self.read(value='$ 0')['value_decimal'], '0')

    def test_all_short_periods_rejected(self):
        for duration in ('Three Months Ended', 'Six Months Ended', 'Nine Months Ended'):
            with self.subTest(duration=duration), self.assertRaises(ValueError): self.read(duration=duration)

    def test_wrong_year(self):
        soup, context = fixture()
        with self.assertRaises(ValueError): audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2024-07-27')

    def test_duplicate_row(self):
        with self.assertRaises(ValueError): self.read(extra='<tr><td>Net income</td><td>$ 1</td></tr>')

    def test_rowspan_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cross-row'): self.read(span=2)

    def test_cross_year_span_rejected(self):
        soup, context = fixture()
        row = soup.table.find_all('tr')[4]
        row.find_all('td')[1]['colspan'] = '2'
        with self.assertRaisesRegex(ValueError, 'Cross-year'): audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')

    def test_missing_reconciliation(self):
        soup, context = fixture()
        soup.find(string='Adjustments to reconcile net income to operating activities').replace_with('Other items')
        with self.assertRaises(ValueError): audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')

    def test_scope_not_attributable(self):
        soup, context = fixture()
        soup.find(string='Net income').replace_with('Net income attributable to common shareholders')
        with self.assertRaises(ValueError): audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')

    def test_currency_conflict(self):
        with self.assertRaises(ValueError): self.read(extra='<tr><td>Amounts in EUR</td></tr>')

    def test_indented_leftmost_nonempty_label(self):
        soup, context = fixture()
        row = soup.table.find_all('tr')[4]
        blank = soup.new_tag('td'); row.insert(0, blank)
        for item in context['tables'][0]['columns']: item['column'] += 1
        # Align the corresponding date and duration rows with the indentation.
        for r in soup.table.find_all('tr')[:4]: r.insert(0, soup.new_tag('td'))
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['value_decimal'], '-1234.50')

    def test_combined_issuer_title_and_units(self):
        soup, context = fixture()
        soup.find(string='CONSOLIDATED STATEMENTS OF CASH FLOWS').replace_with('CISCO SYSTEMS, INC. CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions) (Unaudited)')
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['scale'], '1000000')

    def test_external_formal_heading(self):
        soup, context = fixture()
        soup.find(string='CONSOLIDATED STATEMENTS OF CASH FLOWS').replace_with('')
        soup.find(string='(In millions)').replace_with('')
        heading=soup.new_tag('p');heading.string='CONSOLIDATED STATEMENTS OF CASH FLOWS (In millions)'
        soup.table.insert_before(heading)
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['scale'], '1000000')

    def test_unrelated_non_gaap_notice_not_inline_title(self):
        soup, context = fixture()
        heading=soup.new_tag('p');heading.string='Additional reconciliations between GAAP and non-GAAP financial measures follow.'
        soup.table.insert_before(heading)
        self.assertEqual(audit.independently_read_field(soup, context, 'CSCO', 'annual_net_income', '2025-07-26')['value_decimal'], '-1234.50')

    def test_no_io_or_producer_import(self):
        with patch('builtins.open', side_effect=AssertionError('No I/O')): self.read()
        self.assertNotIn('after149_amzn_extract', sys.modules)
        self.assertNotIn('after149_csco_extract', sys.modules)
        self.assertNotIn('after149_source_batch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
