"""Synthetic-only regression tests. No cached issuer statement is opened."""
import copy
import unittest
from unittest import mock
from bs4 import BeautifulSoup
import after141_amd_source_extract as m


def _date_text(iso):
    d = m.date.fromisoformat(iso)
    return d.strftime('%B') + ' ' + str(d.day) + ',<br>' + str(d.year)


def fixture(year=2022, end='2022-12-31', prior='2021-12-25', quarters=2,
            gross='1,234.50', assets='9,876', external=False, split=False,
            unit='Millions', gross_label=None, extra='', interim_balance_date=None):
    label = gross_label or ('Gross margin' if year <= 2018 else 'Gross profit')
    qdates = [end, prior] if quarters == 2 else [end, '2022-09-24', prior]
    def row(label, values, dollars=False):
        result = '<tr><td>' + label + '</td>'
        for index, value in enumerate(values):
            if index:
                result += '<td></td>'
            if split and value.startswith('('):
                result += '<td>(</td><td>' + value[1:-1] + '</td><td>)</td>'
            else:
                result += '<td>' + ('$' if dollars else '') + '</td><td colspan="2">' + value + '</td>'
        return result + '</tr>'
    def statement(title, ds, body, periods=''):
        heading = '<div>' + title + '</div><div>(' + unit + ' except per share amounts and percentages)</div><div>Unaudited</div>'
        width = len(ds) * 4
        table = '<table>'
        if not external:
            table += '<tr><td colspan="' + str(width) + '">' + heading + '</td></tr>'
        table += periods
        table += '<tr><td></td>' + '<td></td>'.join('<td colspan="3">' + _date_text(day) + '</td>' for day in ds) + '</tr>'
        table += body + '</table>'
        return heading + '<div><div>' + table + '</div></div>' if external else table
    ds = qdates + [end, prior]
    periods = ('<tr><td></td><td colspan="' + str(quarters * 4 - 1) + '">Three Months Ended</td>'
               '<td></td><td colspan="7">Year Ended</td></tr>')
    income = statement('CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', ds,
                       row('Net revenue', ['100'] * len(ds), True) +
                       row('Cost of sales', ['50'] * len(ds)) +
                       row(label, ['11'] * quarters + [gross, '987']) +
                       row('Gross margin %', ['42%'] * len(ds)), periods)
    balance_dates = [end, prior] if interim_balance_date is None else [end, interim_balance_date, prior]
    balance = statement('CONDENSED CONSOLIDATED BALANCE SHEETS', balance_dates,
                        '<tr><td>Assets</td></tr>' + row('Cash and cash equivalents', ['500'] * len(balance_dates), True) +
                        row('Total Assets', [assets] + ['8,000'] * (len(balance_dates) - 1)))
    page = ('<html><p>January 31, ' + str(year + 1) + '</p>'
            '<p>Consolidated Financial Statements prepared in accordance with GAAP.</p>' +
            income + balance + extra + '</html>')
    spec = {'fiscal_year': year, 'fiscal_year_end': end, 'release_date': str(year + 1) + '-01-31',
            'source_url': 'https://ir.amd.com/synthetic', 'raw_html': {'path': 'synthetic.html'},
            'receipt': {'path': 'synthetic.meta.json'}}
    return page.encode(), spec


class SourceTests(unittest.TestCase):
    def check_pair(self, **kwargs):
        raw, spec = fixture(**kwargs)
        return m.extract(raw, spec)

    def test_current_annual_not_quarter_or_comparative(self):
        result = self.check_pair()
        self.assertEqual(result['gross_amount']['amount_decimal'], '1234.50')
        self.assertEqual(result['total_assets']['amount_decimal'], '9876')
        self.assertEqual(result['annual_period']['days_inclusive'], 371)
        self.assertEqual(result['annual_period']['start'], '2021-12-26')
        self.assertFalse(result['annual_period']['start_is_direct_quote'])
        self.assertEqual(result['gross_amount']['scale'], 1000000)
        self.assertEqual(result['gross_amount']['currency_literal'], '$')
        self.assertIsNone(result['gross_amount']['currency_iso'])
        self.assertTrue(result['gaap_evidence']['issuer_policy_literal'])

    def test_three_quarter_columns_shift_current_annual(self):
        result = self.check_pair(quarters=3)
        self.assertEqual(result['gross_amount']['amount_decimal'], '1234.50')
        self.assertEqual(result['gross_amount']['locator']['logical_columns_zero_based'], [13, 14, 15])

    def test_external_heading_in_nested_wrappers(self):
        self.assertEqual(self.check_pair(external=True)['gross_amount']['amount_decimal'], '1234.50')

    def test_interim_balance_comparison_is_not_annual_start(self):
        result = self.check_pair(interim_balance_date='2022-09-24')
        self.assertEqual(result['annual_period']['prior_year_end'], '2021-12-25')
        self.assertTrue(result['annual_period']['prior_year_end_balance_header_cells'])
        self.assertIn('2022-09-24', result['total_assets']['all_statement_period_dates'])

    def test_older_monetary_margin(self):
        result = self.check_pair(year=2015, end='2015-12-26', prior='2014-12-27')
        self.assertEqual(result['gross_amount']['measure_label'], 'Gross margin')
        self.assertEqual(result['annual_period']['days_inclusive'], 364)
        self.assertIn('monetary_Gross_margin_semantic_review_required', result['review_flags'])

    def test_2025_always_semantic_review(self):
        result = self.check_pair(year=2025, end='2025-12-27', prior='2024-12-28',
                                 extra='<p>Manufacturing results are classified as discontinued operations.</p>')
        self.assertTrue(result['semantic_review_required'])
        self.assertIn('discontinued operations', result['discontinued_operations_literal'])
        self.assertIn('discontinued_operations_measurement_compatibility_review_required', result['review_flags'])

    def test_2025_review_even_without_discontinued_literal(self):
        result = self.check_pair(year=2025, end='2025-12-27', prior='2024-12-28')
        self.assertIn('discontinued_operations_measurement_compatibility_review_required', result['review_flags'])

    def test_no_admission_or_timing_overclaim(self):
        result = self.check_pair()
        for key in ('fact_admitted', 'strategy_admitted', 'factor_or_strategy_admitted'):
            self.assertFalse(result[key])
        for key in ('gross_amount', 'total_assets'):
            self.assertFalse(result[key]['fact_admitted'])
            self.assertTrue(result[key]['context_independent_review_required'])
        self.assertFalse(result['availability']['release_date_verified'])
        self.assertFalse(result['availability']['first_public_version_verified'])
        self.assertFalse(result['source']['receipt_and_event_verified'])

    def test_split_parentheses(self):
        self.assertEqual(self.check_pair(gross='(1,234.50)', split=True)['gross_amount']['amount_decimal'], '-1234.50')

    def test_decimal_zero_is_preserved(self):
        self.assertEqual(self.check_pair(gross='0.00')['gross_amount']['amount_decimal'], '0.00')

    def test_amount_rejections(self):
        for value in ('42%', '42 percent', '', '—', '1,23', 'NaN', 'Inf', '1 234', '12 34', '(12', '12)', '(-12)', '1e3'):
            with self.subTest(value=value), self.assertRaises(m.ExtractionError):
                m.exact_amount(value)

    def test_amount_accepted_forms(self):
        for value, expected in (('$ (1,234.50)', '-1234.50'), ('−12', '-12'), ('  $ 0.00 ', '0.00')):
            self.assertEqual(m.exact_amount(value), expected)

    def test_percentage_current_cell_rejected(self):
        with self.assertRaisesRegex(m.ExtractionError, 'Percentage'):
            self.check_pair(gross='42%')

    def test_non_gaap_table_rejected(self):
        raw, spec = fixture()
        raw = raw.replace(b'CONDENSED CONSOLIDATED STATEMENTS', b'Non-GAAP CONDENSED CONSOLIDATED STATEMENTS')
        with self.assertRaisesRegex(m.ExtractionError, 'Non-GAAP'):
            m.extract(raw, spec)

    def test_adjusted_or_proforma_rows_rejected(self):
        for text in (b'As adjusted cost of sales', b'Pro forma cost of sales'):
            raw, spec = fixture()
            with self.subTest(text=text), self.assertRaisesRegex(m.ExtractionError, 'adjusted'):
                m.extract(raw.replace(b'Cost of sales', text), spec)

    def test_duplicate_statement_is_conflict_even_equal(self):
        raw, spec = fixture()
        table = BeautifulSoup(raw, 'html.parser').table
        with self.assertRaisesRegex(m.ExtractionError, 'found 2'):
            m.extract(raw + str(table).encode(), spec)

    def test_invalid_additional_formal_statement_cannot_be_ignored(self):
        raw, spec = fixture()
        duplicate = str(BeautifulSoup(raw, 'html.parser').table).replace('1,234.50', '42%')
        with self.assertRaisesRegex(m.ExtractionError, 'conflicting additional formal statement'):
            m.extract(raw + duplicate.encode(), spec)

    def test_separate_non_gaap_statement_is_excluded(self):
        raw, spec = fixture()
        duplicate = str(BeautifulSoup(raw, 'html.parser').table).replace('CONDENSED CONSOLIDATED', 'Non-GAAP CONDENSED CONSOLIDATED')
        self.assertEqual(m.extract(raw + duplicate.encode(), spec)['gross_amount']['amount_decimal'], '1234.50')

    def test_duplicate_target_row_is_conflict(self):
        raw, spec = fixture()
        soup = BeautifulSoup(raw, 'html.parser')
        row = next(row for row in soup.table.find_all('tr') if 'Gross profit' in row.get_text())
        soup.table.append(copy.copy(row))
        with self.assertRaisesRegex(m.ExtractionError, 'duplicate'):
            m.extract(str(soup).encode(), spec)

    def test_wrong_statement_date_rejected(self):
        raw, spec = fixture()
        spec['fiscal_year_end'] = '2022-12-24'
        with self.assertRaises(m.ExtractionError):
            m.extract(raw, spec)

    def test_prior_column_not_backfilled(self):
        raw, spec = fixture(year=2023, end='2023-12-30', prior='2022-12-31')
        spec.update(fiscal_year=2022, fiscal_year_end='2022-12-31', release_date='2023-01-31')
        with self.assertRaisesRegex(m.ExtractionError, 'not the latest'):
            m.extract(raw, spec)

    def test_quarter_header_cannot_substitute_annual(self):
        raw, spec = fixture()
        with self.assertRaises(m.ExtractionError):
            m.extract(raw.replace(b'Year Ended', b'Three Months Ended'), spec)

    def test_missing_or_conflicting_scale(self):
        for unit in ('Dollars', 'Millions and thousands'):
            with self.subTest(unit=unit), self.assertRaisesRegex(m.ExtractionError, 'scale'):
                self.check_pair(unit=unit)

    def test_pair_scale_conflict(self):
        raw, spec = fixture()
        soup = BeautifulSoup(raw, 'html.parser')
        soup.find_all('table')[1].find_all('div')[1].string = '(Thousands)'
        with self.assertRaisesRegex(m.ExtractionError, 'conflicting scales'):
            m.extract(str(soup).encode(), spec)

    def test_missing_dollar_evidence(self):
        raw, spec = fixture()
        with self.assertRaisesRegex(m.ExtractionError, 'dollar evidence'):
            m.extract(raw.replace(b'$', b''), spec)

    def test_other_currency_rejected(self):
        raw, spec = fixture()
        with self.assertRaisesRegex(m.ExtractionError, 'currency'):
            m.extract(raw.replace(b'Cost of sales', b'Cost of sales CAD'), spec)

    def test_missing_title_rejected(self):
        raw, spec = fixture()
        with self.assertRaisesRegex(m.ExtractionError, 'title'):
            m.extract(raw.replace(b'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', b'Financial summary'), spec)

    def test_missing_cost_context_rejected(self):
        raw, spec = fixture()
        with self.assertRaisesRegex(m.ExtractionError, 'cost-of-sales'):
            m.extract(raw.replace(b'Cost of sales', b'Expense'), spec)

    def test_current_boundary_requires_52_or_53_weeks(self):
        with self.assertRaisesRegex(m.ExtractionError, '52 or 53'):
            self.check_pair(prior='2022-09-24')

    def test_2019_gap_is_explicit(self):
        raw, spec = fixture()
        spec.update(fiscal_year=2019, fiscal_year_end='2019-12-28', release_date='2020-01-28')
        with self.assertRaisesRegex(m.ExtractionError, 'FY2019 remains missing'):
            m.extract(raw, spec)

    def test_spec_aliases_and_conflicts(self):
        raw, spec = fixture()
        aliases = {'year': spec['fiscal_year'], 'end': spec['fiscal_year_end'], 'release': spec['release_date']}
        self.assertEqual(m.extract(raw, aliases)['fiscal_year'], 2022)
        spec['year'] = 2023
        with self.assertRaisesRegex(m.ExtractionError, 'Conflicting spec aliases'):
            m.extract(raw, spec)

    def test_spec_year_symbol_and_date_checks(self):
        for update in ({'fiscal_year': '2022'}, {'symbol': 'OTHER'}, {'release_date': '2022-01-01'},
                       {'fiscal_year_end': 'invalid'}, {'numerator_label': 'Revenue'}, {'expected_label': 'Revenue'}):
            raw, spec = fixture()
            spec.update(update)
            with self.subTest(update=update), self.assertRaises(m.ExtractionError):
                m.extract(raw, spec)

    def test_bound_source_digest(self):
        raw, spec = fixture()
        spec['raw_html']['sha256'] = m.digest(raw)
        self.assertEqual(m.extract(raw, spec)['source']['raw_sha256'], m.digest(raw))
        with self.assertRaisesRegex(m.ExtractionError, 'SHA-256'):
            m.extract(raw + b' ', spec)

    def test_source_evidence_cell_coordinates_and_hash(self):
        raw, spec = fixture()
        result = m.extract(raw, spec)
        value = result['gross_amount']
        self.assertEqual(value['locator']['logical_columns_zero_based'], [9, 10, 11])
        self.assertTrue(all(len(cell['parsed_cell_markup_sha256']) == 64 for cell in value['amount_cells']))
        self.assertTrue(any('Year Ended' == cell['text'] for cell in value['header_cells']))
        self.assertEqual(result['source']['raw_path'], 'synthetic.html')
        self.assertEqual(result['source']['receipt']['path'], 'synthetic.meta.json')

    def test_rowspan_and_overlap_guards(self):
        soup = BeautifulSoup('<table><tr><td rowspan="2">Label</td><td colspan="2">Header</td></tr>'
                             '<tr><td>A</td><td>B</td></tr></table>', 'html.parser')
        _, grid, _ = m.table_grid(soup.table)
        self.assertEqual(grid[(1, 0)]['text'], 'Label')
        for span in ('0', '101', 'no'):
            soup = BeautifulSoup('<table><tr><td colspan="' + span + '">X</td></tr></table>', 'html.parser')
            with self.subTest(span=span), self.assertRaises(m.ExtractionError):
                m.table_grid(soup.table)

    def test_no_io_no_spec_mutation(self):
        raw, spec = fixture()
        previous = copy.deepcopy(spec)
        with mock.patch('builtins.open', side_effect=AssertionError('file I/O forbidden')), \
             mock.patch('socket.socket', side_effect=AssertionError('network forbidden')), \
             mock.patch('subprocess.Popen', side_effect=AssertionError('subprocess forbidden')):
            self.assertEqual(m.extract(raw, spec)['gross_amount']['amount_decimal'], '1234.50')
        self.assertEqual(spec, previous)


def frozen_source_smoke():
    # Explicit opt-in, after the parent froze this source-only extraction plan.
    # Never emits amounts or writes outputs. The pure extractor remains I/O-free.
    import json
    from pathlib import Path
    plan_path = Path('outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json')
    raw_plan = plan_path.read_bytes()
    if m.digest(raw_plan) != 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50':
        raise AssertionError('Frozen source-plan hash mismatch')
    specs = [s for s in json.loads(raw_plan)['sources'] if s['symbol'] == 'AMD']
    if len(specs) != 10:
        raise AssertionError('Unexpected registered AMD source count')
    failed = []
    for spec in specs:
        path = Path(spec['raw_path'])
        if path.is_absolute() or '..' in path.parts or path.resolve() != Path.cwd() / path:
            raise AssertionError('Nonlocal source path')
        try:
            result = m.extract(path.read_bytes(), spec)
            assert result['fact_admitted'] is False
            print(json.dumps({'year': spec['fiscal_year'], 'status': 'EXTRACTED_NOT_ADMITTED',
                              'gross_locator': result['gross_amount']['locator'],
                              'assets_locator': result['total_assets']['locator'],
                              'days': result['annual_period']['days_inclusive'],
                              'flags': result['review_flags']}, sort_keys=True))
        except (m.ExtractionError, AssertionError) as exc:
            failed.append(spec['fiscal_year'])
            print(json.dumps({'year': spec['fiscal_year'], 'status': 'BLOCKED', 'reason': str(exc)}, sort_keys=True))
    if failed:
        raise SystemExit('Source-only smoke blocked for years: ' + str(failed))


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--frozen-source-smoke']:
        frozen_source_smoke()
    else:
        unittest.main()
