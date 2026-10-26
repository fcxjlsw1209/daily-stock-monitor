"""Synthetic-only adversarial tests for independent HTML/source auditing."""
import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from bs4 import BeautifulSoup
import review_msft_nvda_same_report_preflight as r


def fixture():
    html = '''<h2>MICROSOFT CORPORATION INCOME STATEMENTS</h2><p>(In millions)</p>
<table><tr><th></th><th colspan="2">Twelve Months Ended June 30,</th></tr>
<tr><th></th><th>2016</th><th>2015</th></tr><tr><td>Revenue</td><td>$ 20</td><td>$ 19</td></tr>
<tr><td>Gross margin</td><td>12</td><td>11</td></tr></table>
<h2>MICROSOFT CORPORATION BALANCE SHEETS</h2><p>(In millions)</p>
<table><tr><th></th><th>June 30, 2016</th><th>June 30, 2015</th></tr>
<tr><td>Assets</td><td></td><td></td></tr><tr><td>Total assets</td><td>$ 100</td><td>$ 90</td></tr></table>'''
    soup = BeautifulSoup(html, 'html.parser')
    source = {'raw_path': 'current.html'}
    event = {'symbol': 'MSFT', 'available_from': '2016-07-21', 'source': source}
    spec = {'rows': []}
    for field, ti, ri, label in [('gross_profit', 0, 3, 'Gross margin'), ('assets', 1, 2, 'Total assets')]:
        spec['rows'].append({'kind': 'annual_gross_amount' if field == 'gross_profit' else 'year_end_assets',
                            'row_label': label, 'table_index_zero_based': ti, 'row_index_zero_based': ri,
                            'units': {'literal': '(In millions)'}})
        for which, pi, end, amount in [('current', 1, '2016-06-30', '12' if field == 'gross_profit' else '100'),
                                      ('comparison', 2, '2015-06-30', '11' if field == 'gross_profit' else '90')]:
            td = soup.find_all('table')[ti].find_all('tr')[ri].find_all(['td', 'th'])[pi]
            event.setdefault(which, {})['period'] = {'end': end}
            event[which][field] = {'locator': {'table_index_zero_based': ti, 'row_index_zero_based': ri,
                                             'physical_cell_index_zero_based': pi, 'logical_columns': [pi]},
                'parsed_cell_sha256': hashlib.sha256(str(td).encode()).hexdigest(), 'displayed_amount': td.get_text(' ', strip=True),
                'value_decimal': amount, 'header_literals': ['Twelve Months Ended June 30,', end[:4]] if ti == 0 else ['June 30, ' + end[:4]],
                'unit_literal': '(In millions)', 'scale': '1000000', 'currency_literal': '$', 'currency_iso': None,
                'source': source, 'available_from': event['available_from'], 'period_end': end}
    return soup, spec, event


class CellTests(unittest.TestCase):
    def test_four_physical_cells(self):
        soup, spec, event = fixture()
        for side in ('current', 'comparison'):
            for field in ('gross_profit', 'assets'):
                self.assertTrue(r.verify_cell(soup, spec, event, side, field)['verified'])

    def test_cell_claim_tampering(self):
        changes = {'value_decimal': '13', 'displayed_amount': '13', 'parsed_cell_sha256': '0' * 64,
                   'scale': '1000', 'currency_literal': 'USD', 'currency_iso': 'USD',
                   'unit_literal': '(In thousands)', 'source': {'raw_path': 'older.html'},
                   'available_from': '2015-07-22', 'period_end': '2015-06-30',
                   'header_literals': ['Three Months Ended', '2016']}
        for field, value in changes.items():
            with self.subTest(field=field):
                soup, spec, event = fixture()
                event['current']['gross_profit'][field] = value
                with self.assertRaises(ValueError):
                    r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_caption_is_its_own_statement_context(self):
        soup, spec, event = fixture()
        caption = soup.new_tag('caption')
        caption.string = 'MICROSOFT CORPORATION INCOME STATEMENTS (In millions)'
        soup.find('table').insert(0, caption)
        soup.find('h2').string = 'Unrelated non-GAAP analysis'
        self.assertTrue(r.verify_cell(soup, spec, event, 'current', 'gross_profit')['verified'])

    def test_embedded_heading_does_not_borrow_prior_non_gaap_note(self):
        soup = BeautifulSoup('<p>Reconciliation to non-GAAP</p><table><tr><th>CONSOLIDATED STATEMENTS OF INCOME</th></tr><tr><th>(In millions)</th></tr></table>', 'html.parser')
        self.assertNotIn('non-GAAP', r.nearby_context(soup.table))
        self.assertIn('STATEMENTS OF INCOME', r.nearby_context(soup.table))

    def test_thousands_are_preserved_without_normalization(self):
        soup, spec, event = fixture()
        soup.find('p').string = '(In thousands)'
        spec['rows'][0]['units']['literal'] = '(In thousands)'
        event['current']['gross_profit']['unit_literal'] = '(In thousands)'
        event['current']['gross_profit']['scale'] = '1000'
        result = r.verify_cell(soup, spec, event, 'current', 'gross_profit')
        self.assertEqual((result['value_decimal'], result['scale']), ('12', '1000'))

    def test_quarter_column_is_not_annual(self):
        soup, spec, event = fixture()
        soup.find('th', colspan='2').string = 'Three Months Ended June 30,'
        event['current']['gross_profit']['header_literals'][0] = 'Three Months Ended June 30,'
        with self.assertRaisesRegex(ValueError, 'annual/date'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_wrong_date_even_if_claims_match(self):
        soup, spec, event = fixture()
        soup.find_all('table')[0].find_all('tr')[1].find_all('th')[1].string = '2014'
        event['current']['gross_profit']['header_literals'][1] = '2014'
        with self.assertRaisesRegex(ValueError, 'annual/date'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_duplicate_amount_column_is_ambiguous(self):
        soup, spec, event = fixture()
        soup.find_all('table')[0].find_all('tr')[1].find_all('th')[2].string = '2016'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_wrong_row_label(self):
        soup, spec, event = fixture()
        soup.find_all('table')[0].find_all('tr')[3].find('td').string = 'Operating income'
        with self.assertRaisesRegex(ValueError, 'row label'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_percentage_row(self):
        soup, spec, event = fixture()
        soup.find_all('table')[0].find_all('tr')[3].find_all('td')[2].string = '91%'
        with self.assertRaisesRegex(ValueError, 'percentage'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_non_gaap_statement(self):
        soup, spec, event = fixture()
        soup.find('h2').string = 'NON-GAAP INCOME STATEMENTS'
        with self.assertRaisesRegex(ValueError, 'non-GAAP'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_unit_must_exist_in_source(self):
        soup, spec, event = fixture()
        soup.find('p').string = '(In thousands)'
        with self.assertRaisesRegex(ValueError, 'unit not in source'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_logical_location_tampering(self):
        soup, spec, event = fixture()
        event['current']['gross_profit']['locator']['logical_columns'] = [2]
        with self.assertRaisesRegex(ValueError, 'logical column'):
            r.verify_cell(soup, spec, event, 'current', 'gross_profit')

    def test_invalid_span(self):
        soup, _, _ = fixture()
        soup.find('th')['colspan'] = '0'
        with self.assertRaisesRegex(ValueError, 'span'):
            r.expanded_table(soup.find('table'))

    def test_span_overlap(self):
        soup = BeautifulSoup('<table><tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">C</td></tr></table>', 'html.parser')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            r.expanded_table(soup.table)

    def test_exact_string_parsing(self):
        self.assertEqual(r.raw_decimal('$ (1,234.50)'), '-1234.50')
        for invalid in ('-', '1%', '12 34%', 'USD100', '1e5', 'NaN', '∞'):
            with self.assertRaises(ValueError):
                r.raw_decimal(invalid)


class ProvenanceTests(unittest.TestCase):
    def test_first_strictly_later_session(self):
        self.assertEqual(r.strict_next_session('2020-07-02', ['2020-07-02', '2020-07-06', '2020-07-07']), '2020-07-06')
        self.assertEqual(r.strict_next_session('2020-07-04', ['2020-07-02', '2020-07-06', '2020-07-07']), '2020-07-06')

    def test_invalid_sessions(self):
        for sessions in ([], ['2020-07-02', '2020-07-02'], ['2020-07-03', '2020-07-02']):
            with self.assertRaises(ValueError):
                r.strict_next_session('2020-07-02', sessions)

    def test_outside_calendar(self):
        for day in ('2019-01-01', '2020-07-07'):
            with self.assertRaises(ValueError):
                r.strict_next_session(day, ['2020-07-02', '2020-07-07'])

    def test_duplicate_json_and_nonfinite(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                r.parse_json(raw)

    def test_path_and_hash_protection(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT / 'work') as tmp:
            src = r.Sources(tmp)
            Path(tmp, 'x').write_text('one')
            src.read('x', r.digest(b'one'))
            Path(tmp, 'x').write_text('two')
            with self.assertRaisesRegex(ValueError, 'changed'):
                src.read('x')
            for bad in ('../x', '/x', 'CloudStorage/x', 'OneDrive/x'):
                with self.assertRaises(ValueError):
                    src.read(bad)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                src.read('x', '0' * 64)

    def test_previous_boundary_and_initial_sentinel(self):
        class MockSources:
            def read(self, *args): return b''
        raw = 'prior.html'
        current = {'start': '2015-07-01', 'end': '2016-06-30', 'days_inclusive': 366,
                   'derived_not_direct_quote': True, 'evidence': [{'path': 'current.html', 'sha256': 'x', 'release_date': '2016-07-20'}]}
        comp = {'start': '2014-07-01', 'end': '2015-06-30', 'days_inclusive': 365,
                'derived_not_direct_quote': True, 'evidence': [{'path': raw, 'sha256': 'x', 'release_date': '2015-07-20'}]}
        event = {'symbol': 'MSFT', 'fiscal_year': 2016, 'release_date': '2016-07-20', 'current': {'period': current}, 'comparison': {'period': comp}}
        previous = {'symbol': 'MSFT', 'release_date': '2015-07-20', 'source': {'raw_path': raw},
                    'current': {'period': {'end': '2015-06-30'}}, 'comparison': {'period': {'end': '2014-06-30'}}}
        spec = {'current_period': current, 'comparison_period': comp}
        self.assertTrue(r.verify_periods(spec, event, previous, MockSources()))
        broken = copy.deepcopy(previous)
        broken['release_date'] = '2017-01-01'
        with self.assertRaisesRegex(ValueError, 'future'):
            r.verify_periods(spec, event, broken, MockSources())
        broken = copy.deepcopy(previous)
        broken['comparison']['period']['end'] = '2014-05-31'
        with self.assertRaisesRegex(ValueError, 'prior duration'):
            r.verify_periods(spec, event, broken, MockSources())
        # Initial unknown duration must remain an explicit sentinel.
        event['fiscal_year'] = 2015
        current.update(start='2014-07-01', end='2015-06-30', days_inclusive=365)
        comp.update(start=None, end='2014-06-30', days_inclusive=None)
        self.assertFalse(r.verify_periods(spec, event, None, MockSources()))
        comp.update(start='2013-07-01', days_inclusive=365)
        with self.assertRaisesRegex(ValueError, 'must stay unknown'):
            r.verify_periods(spec, event, None, MockSources())


class AccountingTests(unittest.TestCase):
    def event(self, year):
        event = {'symbol': 'MSFT', 'fiscal_year': year, 'source': {'raw_path': 'current.html', 'raw_sha256': 'x'},
                 'current': {'period': {'start': str(year - 1) + '-07-01'}}, 'comparison': {}}
        previous = {'source': {'raw_path': 'prior.html', 'raw_sha256': 'y'}, 'current': {}}
        for f in ('gross_profit', 'assets'):
            event['current'][f] = {'locator': {'table_index_zero_based': 9}}
            event['comparison'][f] = {'value_decimal': '12', 'scale': '1000000', 'displayed_amount': '12', 'locator': {}}
            previous['current'][f] = {'value_decimal': '12', 'scale': '1000000', 'displayed_amount': '12', 'locator': {}}
        return event, previous

    def test_2018_requires_explicit_whole_report_restatement(self):
        event, previous = self.event(2018)
        for f in ('gross_profit', 'assets'):
            previous['current'][f]['value_decimal'] = '11'
        text = 'We adopted new accounting standards related to revenue recognition and leases effective July 1, 2017. The prior periods presented here have been restated to reflect adoption of these new standards.'
        self.assertEqual(r.accounting_scope(event, previous, BeautifulSoup('<p>' + text + '</p>', 'html.parser'))['status'], 'resolved')
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup('<p>We recast segment data</p>', 'html.parser'))
        previous['current']['assets']['value_decimal'] = '12'
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup('<p>' + text + '</p>', 'html.parser'))

    def test_cashflow_note_scope_and_parity(self):
        event, previous = self.event(2026)
        text = '<h2>CASH FLOWS</h2><table><tr><td>Operating activities</td></tr></table><p>We have recast certain prior period amounts to conform to the current period presentation.</p>'
        self.assertEqual(r.accounting_scope(event, previous, BeautifulSoup(text, 'html.parser'))['status'], 'resolved')
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup(text.replace('CASH FLOWS', 'BALANCE SHEETS'), 'html.parser'))
        previous['current']['assets']['value_decimal'] = '11'
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup(text, 'html.parser'))

    def test_segment_note_and_2023_caveat(self):
        event, previous = self.event(2023)
        block = '<h2>SEGMENT RESULTS</h2><table><tr><td>Productivity and Business Processes Intelligent Cloud More Personal Computing</td></tr></table><p>We have recast certain prior period amounts to conform to the way we internally manage and monitor our business.</p>'
        caveat = '<p>a reduction in depreciation expense due to the change in accounting estimate for the useful lives of our server and network equipment</p>'
        result = r.accounting_scope(event, previous, BeautifulSoup(block + block + caveat, 'html.parser'))
        self.assertIn('material', result['scope'])
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup(block + block, 'html.parser'))
        with self.assertRaises(ValueError):
            r.accounting_scope(event, previous, BeautifulSoup(block + caveat, 'html.parser'))

    def test_unflagged_is_not_certification(self):
        event, previous = self.event(2024)
        self.assertEqual(r.accounting_scope(event, previous, BeautifulSoup('', 'html.parser'))['status'], 'no_flag')


if __name__ == '__main__':
    unittest.main(verbosity=2)
