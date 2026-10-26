"""Synthetic-only source parser checks; no real HTMLs or market files are read."""
from copy import deepcopy
import json
import unittest

from bs4 import BeautifulSoup
import operating_margin172_html_sources as subject


def fixture(issuer='MSFT', split=False, current_oi='(120)', prior_oi='90'):
    year = 2024
    width = 2 if split else 1
    if issuer == 'MSFT':
        title = '<h2>INCOME STATEMENTS</h2><div class="c-table">'
        close = '</div>'
        unit = '<tr><td colspan="%s">(In millions, except per share amounts)</td></tr>' % (1 + 4 * width)
        heading = '<tr><td></td><td colspan="%s">Three Months Ended June 30,</td><td colspan="%s">Twelve Months Ended June 30,</td></tr>' % (2 * width, 2 * width)
        revenue_label, oi_label, cost_label = 'Total revenue', 'Operating income', 'Cost of revenue'
        release, known = '2024-07-30', '2024-07-31T20:00:00+00:00'
        ends, starts = ['2024-06-30', '2023-06-30'], ['2023-07-01', '2022-07-01']
        url = 'https://www.microsoft.com/en-us/Investor/earnings/FY-2024-Q4/press-release-webcast'
    else:
        title, close = '', ''
        unit = '<tr><td colspan="%s">AbbVie Inc. Consolidated Statements of Earnings (In millions, except per share data)</td></tr>' % (1 + 4 * width)
        heading = '<tr><td></td><td colspan="%s">Fourth Quarter Ended December 31</td><td colspan="%s">Twelve Months Ended December 31</td></tr>' % (2 * width, 2 * width)
        revenue_label, oi_label, cost_label = 'Net revenues', 'Operating earnings', 'Cost of products sold'
        release, known = '2025-01-31', '2025-02-03T21:00:00+00:00'
        ends, starts = ['2024-12-31', '2023-12-31'], ['2024-01-01', '2023-01-01']
        url = 'https://news.abbvie.com/2025-01-31-test-full-year-2024'
    years = '<tr><td></td>' + ''.join('<td colspan="%s">%s</td>' % (width, x) for x in (year, year - 1, year, year - 1)) + '</tr>'
    def amounts(values, currency=False):
        if split:
            return ''.join('<td>%s</td><td>%s</td>' % ('$' if currency else '', v) for v in values)
        return ''.join('<td>%s%s</td>' % ('$ ' if currency else '', v) for v in values)
    revenue = '<tr><td>%s</td>%s</tr>' % (revenue_label, amounts(['250', '200', '1,000.50', '900'], True))
    cost = '<tr><td>%s</td>%s</tr>' % (cost_label, amounts(['10', '20', '400', '390']))
    oi = '<tr><td>%s</td>%s</tr>' % (oi_label, amounts(['777', '888', current_oi, prior_oi]))
    html = '<html>' + title + '<table>' + unit + heading + years + revenue + cost + oi + '</table>' + close + '</html>'
    raw = html.encode()
    soup = BeautifulSoup(raw, 'html.parser')
    rows = soup.table.find_all('tr')
    cells = rows[3].find_all('td')
    cur_pi, cmp_pi = (6, 8) if split else (3, 4)
    def loc(pi):
        return dict(table_index_zero_based=0, row_index_zero_based=3,
                    physical_cell_index_zero_based=pi, logical_columns=[pi])
    source = dict(raw=dict(path='synthetic.html', sha256=subject.digest(raw), bytes=len(raw)),
                  receipt=dict(path='synthetic.receipt.json', sha256='0' * 64))
    if issuer == 'MSFT':
        source['expected_url'] = url
    else:
        source['metadata'] = dict(source_url=url)
    report = dict(issuer=issuer, fiscal_year=year, report_id='synthetic-' + issuer,
        source=source, current='1000.50', comparative='900', scale='1000000', currency='reported_dollar',
        release_date=release, known_by=known, period_start=starts[0], period_end=ends[0],
        comparative_period_start=starts[1], comparative_period_end=ends[1],
        current_locator=loc(cur_pi), comparative_locator=loc(cmp_pi),
        current_cell_html_sha256=subject.digest(str(cells[cur_pi]).encode()),
        comparative_cell_html_sha256=subject.digest(str(cells[cmp_pi]).encode()),
        row_html_sha256=subject.digest(str(rows[3]).encode()),
        original_holds_preserved=['synthetic_original_hold'])
    receipt = dict(url=url, http_status=200, sha256=subject.digest(raw), bytes=len(raw),
                   retrieved_at='2026-01-01T00:00:00+00:00')
    return report, raw, receipt


def alter(raw, report, receipt, old, new):
    changed = raw.replace(old.encode(), new.encode())
    assert changed != raw
    report, receipt = deepcopy(report), deepcopy(receipt)
    report['source']['raw'].update(sha256=subject.digest(changed), bytes=len(changed))
    receipt.update(sha256=subject.digest(changed), bytes=len(changed))
    return report, changed, receipt


class SourceTests(unittest.TestCase):
    def checked(self, args):
        result = subject.extract_report(*args)
        self.assertEqual(result['status'], 'extracted', result.get('extraction_errors'))
        return result

    def held(self, args, message):
        result = subject.extract_report(*args)
        self.assertEqual(result['status'], 'hold')
        self.assertIn(message, ' '.join(result['extraction_errors']))
        self.assertFalse(result['source_fields_admitted'])
        return result

    def test_annual_columns_negative_oi_and_exact_decimals(self):
        row = self.checked(fixture())
        self.assertEqual(row['cells']['operating_income']['current']['amount_decimal'], '-120')
        self.assertEqual(row['cells']['operating_income']['comparative']['amount_decimal'], '90')
        self.assertEqual(row['cells']['revenue']['current']['amount_decimal'], '1000.50')
        self.assertEqual(row['periods']['current']['end'], '2024-06-30')

    def test_both_issuers_split_currency_columns(self):
        for issuer in ('MSFT', 'ABBV'):
            row = self.checked(fixture(issuer, split=True))
            self.assertEqual(row['cells']['operating_income']['current']['locator']['logical_columns'], [6])
            self.assertIsNone(row['cells']['operating_income']['current']['currency_iso'])

    def test_current_report_time_is_shared_and_inputs_unchanged(self):
        args = fixture('ABBV')
        before = deepcopy(args)
        row = self.checked(args)
        for measure in row['cells'].values():
            for cell in measure.values():
                self.assertEqual(cell['known_by'], args[0]['known_by'])
                self.assertEqual(cell['release_date'], args[0]['release_date'])
        self.assertEqual(args, before)
        self.assertEqual(row['inherited_record']['original_holds_preserved'], ['synthetic_original_hold'])
        self.assertEqual(row['financial_ratios_computed'], 0)
        json.dumps(row, allow_nan=False)

    def test_distinguish_other_operating_income(self):
        report, raw, receipt = fixture('ABBV')
        changed = alter(raw, report, receipt, '</table>', '<tr><td>Other operating income</td><td>9</td><td>9</td><td>9</td><td>9</td></tr></table>')
        self.assertEqual(self.checked(changed)['cells']['operating_income']['current']['amount_decimal'], '-120')

    def test_reject_adjusted_heading(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'INCOME STATEMENTS', 'Non-GAAP INCOME STATEMENTS'), 'Non-GAAP')

    def test_reject_duplicate_oi(self):
        report, raw, receipt = fixture()
        changed = alter(raw, report, receipt, '</table>', '<tr><td>Operating income</td><td>1</td></tr></table>')
        self.held(changed, 'Exactly one')

    def test_missing_oi_is_hold_not_skipped_or_zero(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'Operating income', 'Unclassified earnings'), 'Exactly one')

    def test_reject_quarterly_header(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'Twelve Months Ended', 'Three Months Ended'), 'annual cells')

    def test_reject_ambiguous_numeric_cells(self):
        for value in ('—', '-', '(12)-', '12%', '12(a)', '1,23', '(-12)', '$$12'):
            self.held(fixture(current_oi=value), 'amount' if value != '$$12' else 'dollar')

    def test_zero_and_explicit_negative_oi_are_valid(self):
        for value, expected in [('0', '0'), ('-2.500', '-2.500'), ('(2.500)', '-2.500'), ('−2', '-2')]:
            row = self.checked(fixture(current_oi=value))
            self.assertEqual(row['cells']['operating_income']['current']['amount_decimal'], expected)

    def test_reject_wrong_unit(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'In millions', 'In thousands'), 'scale mismatch')

    def test_reject_missing_formal_title(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'INCOME STATEMENTS', 'Segment snapshot'), 'title absent')

    def test_reject_missing_cost_context(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, 'Cost of revenue', 'Expenses'), 'cost row absent')

    def test_bound_revenue_amount_digest_and_locator(self):
        for key, value, message in [
            ('current', '1000.51', 'amount mismatch'),
            ('current_cell_html_sha256', 'a' * 64, 'digest mismatch')]:
            args = fixture()
            args[0][key] = value
            self.held(args, message)
        args = fixture()
        args[0]['current_locator']['logical_columns'] = [2]
        self.held(args, 'locator mismatch')

    def test_source_digest_and_receipt_failures(self):
        args = fixture()
        args[0]['source']['raw']['sha256'] = 'b' * 64
        self.held(args, 'digest mismatch')
        args = fixture()
        args[2]['http_status'] = 403
        self.held(args, 'HTTP 200')

    def test_future_or_naive_publication_rejected(self):
        args = fixture()
        args[0]['known_by'] = '2024-07-30T00:00:00+00:00'
        self.held(args, 'source clock mismatch')
        args = fixture()
        args[0]['known_by'] = '2024-07-31T20:00:00'
        self.held(args, 'Timezone-aware')

    def test_comparative_date_not_replaced_with_current(self):
        args = fixture()
        args[0]['comparative_period_end'] = '2024-06-30'
        self.held(args, 'period')

    def test_cross_column_span_rejected(self):
        report, raw, receipt = fixture()
        self.held(alter(raw, report, receipt, '<td>(120)</td><td>90</td>', '<td colspan="2">(120)</td>'), 'crosses annual')

    def test_nonfinite_and_zero_revenue_rejected(self):
        for value in ('0', 'NaN'):
            report, raw, receipt = fixture()
            self.held(alter(raw, report, receipt, '$ 1,000.50', '$ ' + value), 'revenue' if value == '0' else 'amount')

    def test_source_notes_not_accounting_approval(self):
        report, raw, receipt = fixture('ABBV')
        row = self.checked(alter(raw, report, receipt, '</html>', '<p>Prior financial results were reclassified.</p></html>'))
        self.assertEqual(len(row['accounting_note_leads']), 1)
        self.assertIn('accounting_comparability_not_admitted', row['holds'])
        self.assertFalse(row['source_fields_admitted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
