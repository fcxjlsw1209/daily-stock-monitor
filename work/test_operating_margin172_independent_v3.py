"""Synthetic-only tests. Never load a financial report, price, or strategy."""
from copy import deepcopy
import unittest

import operating_margin172_independent_v3 as audit


def html(issuer='ABBV', year=2024, income='(30)', annual=True):
    abb = issuer == 'ABBV'
    title = 'AbbVie Inc. Consolidated Statements of Earnings' if abb else 'CONSOLIDATED STATEMENTS OF INCOME'
    endpoint = 'December 31' if abb else 'June 30'
    duration = 'Twelve Months Ended' if annual else 'Three Months Ended'
    revenue = ('Net sales' if year <= 2014 else 'Net revenues') if abb else 'Total revenues'
    # Quarter and annual columns have different numbers deliberately.
    return f'''<html><p>{'AbbVie' if abb else 'Microsoft'}</p><h2>{title}</h2>
    <table><tr><th colspan="5">{title}</th></tr>
    <tr><td colspan="5">(In millions, except per share data)</td></tr>
    <tr><th rowspan="2"></th><th colspan="2">Three Months Ended {endpoint}</th><th colspan="2">{duration} {endpoint}</th></tr>
    <tr><th>{year}</th><th>{year-1}</th><th>{year}</th><th>{year-1}</th></tr>
    <tr><td>{revenue}</td><td>$10</td><td>$9</td><td>$1,200</td><td>$1,100</td></tr>
    <tr><td>{'Cost of products sold' if abb else 'Cost of revenue'}</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>
    <tr><td>{'Operating earnings' if abb else 'Operating income'}</td><td>8</td><td>7</td><td>{income}</td><td>220</td></tr>
    <tr><td>Diluted earnings per share</td><td>1</td><td>2</td><td>3</td><td>4</td></tr></table></html>'''.encode()


def apple(income='(300)', total=False):
    sales = ('Net sales: Products $10 $9 $100 $90 Services 20 19 200 190 Total net sales (1) 30 28 300 280'
             if total else 'Net sales $30 $28 $300 $280')
    footnote = '(1) Net sales by reportable segment' if total else ''
    return [f'''Apple Inc. CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)
    (In millions, except number of shares and per share amounts) Three Months Ended Twelve Months Ended
    September 28, 2024 September 30, 2023 September 28, 2024 September 30, 2023
    {sales} Cost of sales 1 2 3 4 Operating income 9 8 {income} 270
    Other income 1 1 1 1 Net income $3 $4 $5 $6 {footnote}''',
    '''Apple Inc. CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)
    September 28, 2024 September 30, 2023 ASSETS: cash 1 2''']


def html_produced(parsed):
    cells = deepcopy(parsed['cells'])
    for roles in cells.values():
        for cell in roles.values():
            cell['locator']['logical_columns'] = cell['locator'].pop('logical_columns_zero_based')
            cell.update(unit_literal='In millions', release_date='2025-01-31', known_by='2025-02-03T21:00:00+00:00',
                        source_raw_sha256=parsed['source_sha256'], currency_literal=parsed['currency_literal'], currency_iso=None,
                        currency_iso_newly_certified=False, fact_admitted=False)
    periods = {role: dict(start=cells['revenue'][role]['period_start'], end=cells['revenue'][role]['period_end'],
                         days_inclusive=366 if role == 'current' else 365) for role in ('current', 'comparative')}
    return dict(status='extracted', issuer=parsed['issuer'], fiscal_year=parsed['fiscal_year'],
                release_date='2025-01-31', known_by='2025-02-03T21:00:00+00:00', cells=cells, periods=periods,
                statement={k: parsed[k] for k in ('table_index_zero_based', 'table_html_sha256', 'scale')})


def apple_produced(parsed):
    fields = deepcopy(parsed['cells'])
    for name, roles in fields.items():
        for role, cell in roles.items():
            cell.update(period=dict(start=cell['period_start'], end=cell['period_end'],
                                    days_inclusive=364 if role == 'current' else None, duration_literal='Twelve Months Ended'),
                        date=cell['period_end'], all_row_cells=cell['all_amount_decimals'], unit_literal=parsed['unit_literal'],
                        release_date='2024-10-31', known_at_session='2024-11-01', iso_currency=None,
                        ISO_currency_verified=False, fact_admitted=False, strategy_admitted=False,
                        vintage_role='own_year_original' if role == 'current' else 'same_current_release_prior_year_comparative',
                        row_text='synthetic row', row_text_sha256=audit.digest(b'synthetic row'))
    return dict(status='EXTRACTED_SOURCE_CANDIDATE', symbol='AAPL', fiscal_year=2024, fields=fields,
                statement=dict(header_dates=parsed['header_dates'], page_1based=1),
                publication=dict(release_date='2024-10-31'), known_at_session='2024-11-01',
                availability=dict(candidate_first_usable_session='2024-11-01'), holds=['comparative_period_start_unknown'])


class FakeTextPage:
    def __init__(self):
        self.text, self.coords = '', []

    def add(self, text, x, y):
        index = len(self.text)
        self.text += text + '\n'
        self.coords.extend((x + i, y, x + i + 1, y + 2) for i in range(len(text) + 1))
        return dict(index=index, count=len(text), text=text, bbox=[x, y, x + len(text), y + 2])

    def get_charbox(self, j): return self.coords[j]
    def get_text_range(self, index, count): return self.text[index:index + count]

    def search(self, term, match_case=False):
        pairs, start = [], 0
        while True:
            index = self.text.lower().find(term.lower(), start)
            if index < 0: break
            pairs.append((index, len(term))); start = index + len(term)
        class Search:
            def __init__(self): self.iterator = iter(pairs)
            def get_next(self): return next(self.iterator, None)
            def close(self): pass
        return Search()


class IndependentTests(unittest.TestCase):
    def test_signed_money_not_absolute(self):
        self.assertEqual(audit.amount('(1,200.50)'), '-1200.50')
        self.assertEqual(audit.amount('−300'), '-300')
        self.assertEqual(audit.amount('$ 42'), '42')

    def test_missing_or_bad_money_not_zero(self):
        for bad in ['—', '', 'N/A', '(−10)', '1,23', '1e3', '10%']:
            with self.subTest(bad=bad), self.assertRaises(ValueError): audit.amount(bad)

    def test_abbv_annual_not_quarter(self):
        got = audit.parse_html_statement(html(), 'ABBV', 2024)
        self.assertEqual(got['cells']['revenue']['current']['amount_decimal'], '1200')
        self.assertEqual(got['cells']['operating_income']['current']['amount_decimal'], '-30')
        self.assertEqual(got['cells']['revenue']['comparative']['period_start'], '2023-01-01')

    def test_msft_fiscal_year(self):
        got = audit.parse_html_statement(html('MSFT'), 'MSFT', 2024)
        self.assertEqual(got['cells']['revenue']['current']['period_start'], '2023-07-01')
        self.assertEqual(got['cells']['revenue']['current']['period_end'], '2024-06-30')

    def test_caption_owns_formal_title_and_units(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html('MSFT'), 'html.parser')
        table = soup.table
        for n in soup.find_all('h2'): n.decompose()
        rows = table.find_all('tr')
        rows[0].decompose(); rows[1].decompose()
        caption = soup.new_tag('caption')
        caption.string = 'MICROSOFT CORPORATION INCOME STATEMENTS (In millions, except per share amounts)'
        table.insert(0, caption)
        got = audit.parse_html_statement(str(soup).encode(), 'MSFT', 2024)
        self.assertEqual(got['cells']['revenue']['current']['amount_decimal'], '1200')

    def test_reconciliation_candidate_before_formal_table_is_not_fatal(self):
        earlier = html('MSFT').replace(b'CONSOLIDATED STATEMENTS OF INCOME', b'Financial Performance Constant Currency Reconciliation')
        got = audit.parse_html_statement(earlier + html('MSFT'), 'MSFT', 2024)
        self.assertEqual(got['table_index_zero_based'], 1)

    def test_cannot_borrow_title_across_another_table(self):
        bad = html('MSFT').replace(b'<h2>CONSOLIDATED STATEMENTS OF INCOME</h2>', b'<h2>INCOME STATEMENTS</h2><table><tr><td>unrelated</td></tr></table>')
        bad = bad.replace(b'<th colspan="5">CONSOLIDATED STATEMENTS OF INCOME</th>', b'<th colspan="5">Summary</th>')
        with self.assertRaises(ValueError): audit.parse_html_statement(bad, 'MSFT', 2024)

    def test_adjacent_c_table_title_and_unit(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html('MSFT'), 'html.parser')
        table = soup.table
        rows = table.find_all('tr'); rows[0].decompose(); rows[1].decompose()
        wrapper = soup.new_tag('div'); wrapper['class'] = 'c-table'
        table.wrap(wrapper)
        unit = soup.new_tag('p'); unit.string = 'In millions'
        wrapper.insert_before(unit)
        got = audit.parse_html_statement(str(soup).encode(), 'MSFT', 2024)
        self.assertEqual(got['cells']['revenue']['current']['amount_decimal'], '1200')

    def test_formal_candidate_amount_failure_is_not_skipped(self):
        with self.assertRaises(ValueError): audit.parse_html_statement(html('MSFT', income='N/A') + html('MSFT'), 'MSFT', 2024)

    def test_abbv_old_sales_label(self):
        self.assertEqual(audit.parse_html_statement(html(year=2013), 'ABBV', 2013)['cells']['revenue']['current']['amount_decimal'], '1200')

    def test_no_annual_table(self):
        with self.assertRaises(ValueError): audit.parse_html_statement(html(annual=False), 'ABBV', 2024)

    def test_reject_duplicate_tables(self):
        with self.assertRaises(ValueError): audit.parse_html_statement(html() + html(), 'ABBV', 2024)

    def test_reject_other_operating_income(self):
        with self.assertRaises(ValueError): audit.parse_html_statement(html().replace(b'Operating earnings', b'Other operating earnings'), 'ABBV', 2024)

    def test_reject_segment_or_adjusted(self):
        for title in (b'Adjusted', b'Non-GAAP', b'Segment'):
            bad = html().replace(b'AbbVie Inc. Consolidated Statements of Earnings', title)
            with self.subTest(title=title), self.assertRaises(ValueError): audit.parse_html_statement(bad, 'ABBV', 2024)

    def test_require_same_table(self):
        bad = html().replace(b'<tr><td>Operating earnings', b'</table><table><tr><td>Operating earnings')
        with self.assertRaises(ValueError): audit.parse_html_statement(bad, 'ABBV', 2024)

    def test_require_scale(self):
        with self.assertRaises(ValueError): audit.parse_html_statement(html().replace(b'In millions', b'In thousands'), 'ABBV', 2024)

    def test_absent_dollar_is_unknown_not_rejection(self):
        parsed = audit.parse_html_statement(html('MSFT').replace(b'$', b''), 'MSFT', 2024)
        self.assertIsNone(parsed['currency_literal'])
        self.assertIsNone(parsed['currency_iso'])
        self.assertFalse(parsed['currency_iso_certified'])
        self.assertEqual(parsed['cells']['revenue']['current']['amount_decimal'], '1200')

    def test_currency_literal_must_match_raw_evidence(self):
        for original, invented in [(html().replace(b'$', b''), '$'), (html(), None)]:
            parsed = audit.parse_html_statement(original, 'ABBV', 2024)
            good = html_produced(parsed)
            self.assertTrue(audit.compare_html_record(parsed, good, good['release_date'], good['known_by']))
            bad = deepcopy(good); bad['cells']['operating_income']['current']['currency_literal'] = invented
            with self.assertRaises(ValueError): audit.compare_html_record(parsed, bad, good['release_date'], good['known_by'])

    def test_dollar_elsewhere_does_not_certify_statement(self):
        parsed = audit.parse_html_statement(b'<p>Stock traded at $100</p>' + html().replace(b'$', b''), 'ABBV', 2024)
        self.assertIsNone(parsed['currency_literal'])

    def test_ambiguous_cell_straddles_years(self):
        bad = html().replace(b'<td>(30)</td><td>220</td>', b'<td colspan="2">(30)</td>')
        with self.assertRaises(ValueError): audit.parse_html_statement(bad, 'ABBV', 2024)

    def test_separate_dollar_cells_and_colspan_years(self):
        raw = b'''<p>AbbVie</p><table><tr><td colspan="9">AbbVie Inc. Consolidated Statements of Earnings</td></tr>
        <tr><td colspan="9">In millions</td></tr><tr><td rowspan="2"></td><th colspan="4">Three Months Ended December 31</th><th colspan="4">Twelve Months Ended December 31</th></tr>
        <tr><th colspan="2">2024</th><th colspan="2">2023</th><th colspan="2">2024</th><th colspan="2">2023</th></tr>
        <tr><td>Net revenues</td><td>$</td><td>10</td><td>$</td><td>9</td><td>$</td><td>1200</td><td>$</td><td>1100</td></tr>
        <tr><td>Cost of products sold</td></tr>
        <tr><td>Operating earnings</td><td></td><td>1</td><td></td><td>2</td><td></td><td>(30)</td><td></td><td>220</td></tr></table>'''
        parsed = audit.parse_html_statement(raw, 'ABBV', 2024)
        self.assertEqual(parsed['cells']['operating_income']['current']['amount_decimal'], '-30')
        self.assertEqual(parsed['cells']['revenue']['current']['locator']['physical_cell_index_zero_based'], 6)

    def test_html_output_comparison_and_mutations(self):
        parsed = audit.parse_html_statement(html(), 'ABBV', 2024)
        good = html_produced(parsed)
        self.assertTrue(audit.compare_html_record(parsed, good, good['release_date'], good['known_by']))
        for key, value in [('amount_decimal', '30'), ('period_end', '2023-12-31'), ('known_by', '2024-02-01'),
                           ('cell_html_sha256', 'bad'), ('currency_iso', 'USD')]:
            bad = deepcopy(good); bad['cells']['operating_income']['current'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit.compare_html_record(parsed, bad, good['release_date'], good['known_by'])

    def test_pdf_annual_columns_and_signed_income(self):
        got = audit.parse_aapl_statements(apple(), 2024)
        self.assertEqual(got['cells']['revenue']['current']['amount_decimal'], '300')
        self.assertEqual(got['cells']['operating_income']['current']['amount_decimal'], '-300')
        self.assertEqual(got['current_period']['start'], '2023-10-01')
        self.assertIsNone(got['comparative_period']['start'])

    def test_pdf_product_dollars_and_total_footnote(self):
        got = audit.parse_aapl_statements(apple(total=True), 2024)
        self.assertEqual(got['cells']['revenue']['current']['amount_decimal'], '300')

    def test_pdf_bad_boundaries_headers_scopes(self):
        original = apple()
        for a, b in [('Operating income', 'Other operating income'), ('Twelve Months Ended', 'Three Months Ended'),
                     ('Cost of sales', 'No boundary'), ('September 28, 2024', 'September 27, 2024'), ('(Unaudited)', 'NON-GAAP')]:
            bad = [original[0].replace(a, b), original[1]]
            with self.subTest(a=a), self.assertRaises(ValueError): audit.parse_aapl_statements(bad, 2024)

    def test_pdf_duplicate_or_missing_oi(self):
        for bad in [apple()[0].replace('Operating income', 'Missing income'), apple()[0] + ' Operating income 1 2 3 4']:
            with self.assertRaises(ValueError): audit.parse_aapl_statements([bad, apple()[1]], 2024)

    def test_pdf_dollar_evidence_not_inferred(self):
        with self.assertRaises(ValueError): audit.parse_aapl_statements([x.replace('$', '') for x in apple()], 2024)

    def test_pdf_output_comparison_and_mutations(self):
        parsed = audit.parse_aapl_statements(apple(), 2024)
        good = apple_produced(parsed)
        self.assertTrue(audit.compare_aapl_record(parsed, good, '2024-10-31', '2024-11-01'))
        for key, value in [('amount_decimal', '300'), ('known_at_session', '2023-11-01'), ('iso_currency', 'USD'),
                           ('all_row_cells', ['9', '8', '-30', '270'])]:
            bad = deepcopy(good); bad['fields']['operating_income']['current'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit.compare_aapl_record(parsed, bad, '2024-10-31', '2024-11-01')

    def test_pdf_unknown_start_not_fabricated(self):
        parsed = audit.parse_aapl_statements(apple(), 2024); bad = apple_produced(parsed)
        bad['fields']['operating_income']['comparative']['period']['start'] = '2022-10-01'
        with self.assertRaises(ValueError): audit.compare_aapl_record(parsed, bad, '2024-10-31', '2024-11-01')

    def test_comparative_boundary_requires_earlier_original(self):
        parsed = audit.parse_aapl_statements(apple(), 2024); good = apple_produced(parsed)
        prior = dict(fiscal_year=2023, release_date='2023-11-02', current_period=dict(start='2022-09-25', end='2023-09-30'))
        for roles in good['fields'].values():
            roles['comparative']['period'].update(start='2022-09-25', days_inclusive=371)
        self.assertTrue(audit.compare_aapl_record(parsed, good, '2024-10-31', '2024-11-01', prior))
        prior['release_date'] = '2025-01-01'
        with self.assertRaises(ValueError): audit.compare_aapl_record(parsed, good, '2024-10-31', '2024-11-01', prior)

    def test_pdf_geometry_source_characters_and_annual_columns(self):
        parsed = audit.parse_aapl_statements(apple(), 2024); produced = apple_produced(parsed)
        page = FakeTextPage()
        page.add('CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', 0, 100)
        page.add('Cost of sales', 0, 60); page.add('Net income', 0, 20)
        for name, label, y in [('revenue', 'Net sales', 80), ('operating_income', 'Operating income', 40)]:
            label_box = page.add(label, 0, y)
            row = [page.add(v, 100 + i * 50, y) for i, v in enumerate(parsed['cells'][name]['current']['all_amount_decimals'])]
            for role, col in [('current', 2), ('comparative', 3)]:
                cell = produced['fields'][name][role]
                cell.update(measure_label=label, pdfium_all_row_cells=row)
                cell['locator'].update(pdfium_char_index=row[col]['index'], pdfium_char_count=row[col]['count'],
                    pdfium_cell_bbox=row[col]['bbox'], pdfium_label_bbox=label_box['bbox'])
        self.assertTrue(audit.verify_pdf_geometry(page, parsed, produced))
        bad = deepcopy(produced)
        bad['fields']['operating_income']['current']['locator']['pdfium_char_index'] -= 1
        with self.assertRaises(ValueError): audit.verify_pdf_geometry(page, parsed, bad)

    def test_publication_independent_visible_and_structured(self):
        raw = b'<span class="category-eyebrow__date">October 31, 2024</span><script>{"datePublished":"2024-10-31T12:00:00"}</script>'
        self.assertEqual(audit.publication_date(raw, 'AAPL'), '2024-10-31')
        with self.assertRaises(ValueError): audit.publication_date(raw.replace(b'2024-10-31T', b'2024-10-30T'), 'AAPL')
        self.assertEqual(audit.publication_date(b'<p>REDMOND, Wash. July 30, 2024</p>', 'MSFT'), '2024-07-30')

    def test_calendar_strict_next_session_and_scope(self):
        sessions = ['2024-10-30', '2024-10-31', '2024-11-01', '2024-11-04']
        self.assertEqual(audit.next_session('2024-10-31', sessions), '2024-11-01')
        self.assertEqual(audit.next_session('2024-11-02', sessions), '2024-11-04')
        for bad in [sessions + [sessions[-1]], sessions[::-1], ['2024-10-31', 'bad']]:
            with self.assertRaises(ValueError): audit.next_session('2024-10-31', bad)
        with self.assertRaises(ValueError): audit.next_session('2020-01-01', sessions)

    def test_receipt_original_bytes_bound(self):
        raw = b'%PDF-test'; url = 'https://www.apple.com/test.pdf'
        receipt = dict(status=200, url=url, final_url=url, response_sha256=audit.digest(raw), decoded_sha256=audit.digest(raw),
                       response_bytes=len(raw), decoder='identity', content_type='application/pdf',
                       public_unauthenticated=True, retrieved_at='2026-10-21T18:00:00+00:00')
        audit.verify_receipt(raw, receipt, url, 'AAPL', pdf=True)
        for key, value in [('response_sha256', 'bad'), ('decoder', 'transformed'), ('status', 404), ('final_url', 'https://other.com')]:
            bad = dict(receipt); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_receipt(raw, bad, url, 'AAPL', pdf=True)

    def test_scope_never_silently_omit_or_duplicate(self):
        expected = [('MSFT', 2024), ('AAPL', 2024)]
        rows = [dict(issuer='MSFT', fiscal_year=2024), dict(symbol='AAPL', fiscal_year=2024)]
        self.assertTrue(audit.verify_scope(rows, expected))
        for bad in [rows[:1], rows + rows[:1], rows + [dict(issuer='ABBV', fiscal_year=2024)]]:
            with self.assertRaises(ValueError): audit.verify_scope(bad, expected)


if __name__ == '__main__':
    unittest.main()
