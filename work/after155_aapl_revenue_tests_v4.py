"""Fabricated boundary cases only; no real report or monetary data read."""
from copy import deepcopy
import io
from pathlib import Path
import unittest
import after155_aapl_revenue_source_v4 as m


def fixture():
    texts = ['''Apple Inc.
CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)
(In millions, except shares and per share amounts)
Three Months Ended                  Twelve Months Ended
September 28, September 30, September 28, September 30,
2024 2023 2024 2023
Net sales $ 5.01 $ 4.00 $ 20,000.50 $ 18,000.25
Cost of sales 3 3 12 11
Gross margin 2 1 8 7
''', '''Apple Inc.
CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)
(In millions, except shares and par value)
September 28, September 30,
2024 2023
Cash $ 4 $ 3
''']
    return texts, {'fiscal_year': 2024, 'fiscal_year_end': '2024-09-28', 'release_date': '2024-10-31'}


class BoundaryTests(unittest.TestCase):
    def test_annual_columns_decimals_scale_and_roles(self):
        texts, spec = fixture(); row = m.parse_revenue(texts, spec)
        a, b = row['revenue']['current'], row['revenue']['comparative']
        self.assertEqual(a['displayed_amount'], '20,000.50'); self.assertEqual(a['amount_decimal'], '20000.50')
        self.assertEqual(b['amount_decimal'], '18000.25'); self.assertEqual(a['scale'], 1000000)
        self.assertEqual(a['locator']['financial_column_index_0based'], 2)
        self.assertEqual(b['locator']['financial_column_index_0based'], 3)
        self.assertFalse(b['historical_original_event_created']); self.assertFalse(a['fact_admitted'])

    def test_total_label_and_component_decoys(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales $', 'Net sales:\nProducts $ 3 $ 2 $ 10 $ 9\nServices $ 2 $ 2 $ 10 $ 9\nTotal net sales $')
        self.assertEqual(m.parse_revenue(texts, spec)['revenue']['current']['measure_label'], 'Total net sales')

    def test_duplicate_total_and_competing_labels_rejected(self):
        for label in ['Net sales', 'Total net sales']:
            texts, spec = fixture(); texts[0] = texts[0].replace('Cost of sales', label + ' $ 1 $ 2 $ 3 $ 4\nCost of sales')
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'Unique exact'):
                m.parse_revenue(texts, spec)

    def test_missing_total_not_reconstructed(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales', 'Products')
        with self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_quarter_only_and_reversed_duration_rejected(self):
        for replacement in ['Three Months Ended', 'Six Months Ended']:
            texts, spec = fixture(); texts[0] = texts[0].replace('Twelve Months Ended', replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError): m.parse_revenue(texts, spec)
        texts, spec = fixture(); texts[0] = texts[0].replace('Three Months Ended                  Twelve Months Ended', 'Twelve Months Ended    Three Months Ended')
        with self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_wrong_annual_columns_rejected(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('2024 2023 2024 2023', '2024 2023 2023 2024')
        with self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_wrong_balance_endpoint_rejected(self):
        texts, spec = fixture(); texts[1] = texts[1].replace('September 30', 'September 29')
        with self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_52_and_53_week_periods(self):
        self.assertEqual(m.period_from_boundaries('2023-09-30', '2024-09-28')['days_inclusive'], 364)
        x = m.period_from_boundaries('2022-09-24', '2023-09-30')
        self.assertEqual((x['start'], x['days_inclusive'], x['weeks']), ('2022-09-25', 371, 53))

    def test_calendar_or_nonannual_end_rejected(self):
        for p, e in [('2023-09-28', '2024-09-28'), ('2023-12-31', '2024-12-31'), ('2022-09-24', '2024-09-28')]:
            with self.subTest(prior=p, end=e), self.assertRaises(ValueError): m.period_from_boundaries(p, e)

    def test_unknown_comparative_start_preserved(self):
        texts, spec = fixture(); x = m.parse_revenue(texts, spec)
        self.assertIsNone(x['revenue']['comparative']['period']['start'])
        self.assertIsNone(x['revenue']['comparative']['period']['days_inclusive'])

    def test_earlier_original_duration_only_no_value_copy(self):
        texts, spec = fixture(); row = m.parse_revenue(texts, spec)
        row.update(fiscal_year=2024, publication={'release_date': '2024-10-31'})
        earlier = {'fiscal_year': 2023, 'publication': {'release_date': '2023-11-02'}, 'source': {'path': 'earlier.pdf', 'sha256': 'a'*64},
            'revenue': {'current': {'amount_decimal': '9999', 'period': dict(m.period_from_boundaries('2022-09-24', '2023-09-30'), boundary_locator={'page_1based': 2})}}}
        m.attach_prior_duration(row, earlier)
        self.assertEqual(row['revenue']['comparative']['amount_decimal'], '18000.25')
        self.assertEqual(row['revenue']['comparative']['period']['weeks'], 53)
        earlier['publication']['release_date'] = '2025-01-01'
        with self.assertRaises(ValueError): m.attach_prior_duration(row, earlier)

    def test_currency_literal_is_not_iso(self):
        texts, spec = fixture(); a = m.parse_revenue(texts, spec)['revenue']['current']
        self.assertEqual(a['currency_evidence_class'], 'dollar_literal_only')
        self.assertIsNone(a['iso_currency']); self.assertFalse(a['ISO_currency_verified'])

    def test_missing_row_currency_or_scale_rejected(self):
        for old, new in [('$', ''), ('In millions', 'In thousands')]:
            texts, spec = fixture(); texts[0] = texts[0].replace(old, new)
            with self.subTest(old=old), self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_dollarsign_and_split_glyphs_supported(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('$', '/dollarsign').replace('Net sales', 'N e t s a l e s').replace('Twelve Months Ended', 'Twe lve M o n t h s E n d e d')
        self.assertEqual(m.parse_revenue(texts, spec)['revenue']['current']['amount_decimal'], '20000.50')

    def test_invalid_monetary_formats_rejected(self):
        for value in ['NaN', 'Infinity', '1,23', '20%', '(2', '(-2)', '2e5', '2 000']:
            texts, spec = fixture(); texts[0] = texts[0].replace('20,000.50', value)
            with self.subTest(value=value), self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_zero_negative_exact_values_retained(self):
        for value, result in [('0.00', '0.00'), ('(2.50)', '-2.50'), ('-2.50', '-2.50')]:
            texts, spec = fixture(); texts[0] = texts[0].replace('20,000.50', value)
            self.assertEqual(m.parse_revenue(texts, spec)['revenue']['current']['amount_decimal'], result)

    def test_non_gaap_and_wrong_issuer_rejected(self):
        for old, new in [('Apple Inc.', 'Other Inc.'), ('Unaudited', 'NON-GAAP Unaudited')]:
            texts, spec = fixture(); texts[0] = texts[0].replace(old, new)
            with self.subTest(old=old), self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_no_input_mutation(self):
        texts, spec = fixture(); before = deepcopy((texts, spec)); m.parse_revenue(texts, spec)
        self.assertEqual((texts, spec), before)

    def test_current_release_availability_and_weekend(self):
        sessions = ['2023-11-03', '2024-10-31', '2024-11-01', '2024-11-04']
        self.assertEqual(m.first_session('2024-10-31', sessions), '2024-11-01')
        self.assertEqual(m.first_session('2024-11-02', sessions), '2024-11-04')

    def test_bad_calendar_rejected(self):
        for release, sessions in [('2024-11-01', []), ('2024-11-01', ['2024-11-04', '2024-11-01']), ('2024-11-01', ['2024-11-01', '2024-11-01']), ('2024-11-05', ['2024-11-01', '2024-11-04'])]:
            with self.subTest(sessions=sessions), self.assertRaises(ValueError): m.first_session(release, sessions)

    def test_future_note_knowledge_retained(self):
        notes = m.note_contexts('The company adopted an accounting standard and retrospectively recast revenue.', {'path': 'fake'}, '2024-11-01', 1)
        self.assertGreaterEqual(len(notes), 3)
        self.assertTrue(all(x['known_at_session'] == '2024-11-01' and not x['confirmed_accounting_change'] for x in notes))

    def test_receipt_digest_and_host(self):
        raw = b'%PDF-fabricated'; url = 'https://www.apple.com/fake.pdf'
        meta = {'status': 200, 'url': url, 'final_url': url, 'public_unauthenticated': True,
            'response_sha256': m.sha(raw), 'response_bytes': len(raw), 'retrieved_at': '2026-01-01T00:00:00Z',
            'content_type': 'application/pdf', 'decoder': 'identity', 'decoded_sha256': m.sha(raw)}
        self.assertTrue(m.receipt_check(meta, raw, url, 'pdf')['verified'])
        bad = dict(meta, response_sha256='0'*64)
        with self.assertRaises(ValueError): m.receipt_check(bad, raw, url, 'pdf')
        with self.assertRaises(ValueError): m.receipt_check(meta, raw, 'https://www.apple.com.evil/fake.pdf', 'pdf')

    def test_parent_date_title_and_direct_link(self):
        raw = b'<h1>Apple reports fourth quarter results</h1><span class="category-eyebrow__date">October 31, 2024</span><script type="application/ld+json">{"datePublished":"2024-10-31"}</script><a href="/fake.pdf">PDF</a>'
        spec = {'release_date': '2024-10-31', 'source_url': 'https://www.apple.com/fake.pdf'}
        self.assertTrue(m.publication(raw, spec, 'https://www.apple.com/release')['direct_pdf_link_verified'])
        for change in [raw.replace(b'2024-10-31', b'2024-10-30'), raw.replace(b'/fake.pdf', b'/other.pdf'), raw.replace(b'fourth quarter', b'third quarter')]:
            with self.assertRaises(ValueError): m.publication(change, spec, 'https://www.apple.com/release')

    def test_footnote_with_same_sales_block_currency(self):
        texts, spec = fixture()
        texts[0] = texts[0].replace('Net sales $ 5.01 $ 4.00 $ 20,000.50 $ 18,000.25',
            'Net sales:\nProducts $ 3 $ 2 $ 10 $ 9\nServices 2 2 10 9\nTotal net sales (1) 5.01 4.00 20,000.50 18,000.25\n(1) Net sales by category:')
        result = m.parse_revenue(texts, spec)['revenue']['current']
        self.assertEqual(result['amount_decimal'], '20000.50')
        self.assertEqual(result['label_footnote']['literal'], '(1)')
        self.assertEqual(result['currency_evidence']['scope'], 'same net-sales block Products row')
        self.assertEqual(result['currency_evidence']['locator']['line_1based'], 8)
        self.assertIsNone(result['iso_currency'])

    def test_unsupported_or_unresolved_footnote_held(self):
        for marker in ['(1)', '(2)']:
            texts, spec = fixture(); texts[0] = texts[0].replace('Net sales $', 'Total net sales ' + marker + ' $')
            with self.subTest(marker=marker), self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_four_cell_negative_quarter_not_stripped_as_annotation(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales $ 5.01', 'Total net sales $ (1)')
        result = m.parse_revenue(texts, spec)['revenue']['current']
        self.assertEqual(result['all_row_cells'][0], '(1)'); self.assertIsNone(result['label_footnote'])

    def test_currency_from_unrelated_block_rejected(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales $ 5.01 $ 4.00 $ 20,000.50 $ 18,000.25',
            'Cost of sales:\nProducts $ 3 $ 2 $ 10 $ 9\nTotal net sales 5.01 4.00 20,000.50 18,000.25')
        with self.assertRaises(ValueError): m.parse_revenue(texts, spec)

    def test_later_segment_totals_excluded(self):
        texts, spec = fixture(); texts[0] += '\n(1) Net sales by category:\nTotal net sales $ 5 $ 4 $ 21 $ 19\nNet sales $ 9 $ 8 $ 22 $ 20\n'
        self.assertEqual(m.parse_revenue(texts, spec)['revenue']['current']['amount_decimal'], '20000.50')

    def test_missing_formal_cost_boundary_rejected(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Cost of sales 3 3 12 11\n', '')
        with self.assertRaisesRegex(ValueError, 'boundary required'): m.parse_revenue(texts, spec)

    def test_no_total_in_formal_block_not_filled_from_note(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales', 'Products') + '\nTotal net sales $ 5 $ 4 $ 21 $ 19\n'
        with self.assertRaisesRegex(ValueError, 'Unique exact'): m.parse_revenue(texts, spec)

    def test_fabricated_geometry_scope_and_cross_engine_disagreement(self):
        texts, spec = fixture(); row = m.parse_revenue(texts, spec)['revenue']['current']
        def box(text, x, y, index):
            return {'text': text, 'bbox': [x, y, x + 5, y + 2], 'index': index, 'count': len(text)}
        cells = row['all_row_cells']
        values = {'Cost of sales': [box('Cost of sales', 5, 500, 10)],
            'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS': [box('title', 5, 700, 1)],
            'Net sales': [box('Net sales', 5, 600, 5), box('Net sales', 5, 100, 500)]}
        for i, cell in enumerate(cells):
            values[cell] = [box(cell, 30 + i * 10, 600, 20 + i), box(cell, 30 + i * 10, 100, 600 + i)]
        finder = lambda page, term: values.get(term, [])
        m.attach_statement_geometry(row, None, finder)
        self.assertEqual(row['locator']['pdfium_char_index'], 22)
        self.assertTrue(row['independent_PDFium_cell_text_match'])
        values[cells[2]][0]['text'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'disagree'): m.attach_statement_geometry(row, None, finder)
        values[cells[2]][0]['text'] = cells[2]
        values['Net sales'].append(box('Net sales', 5, 610, 30))
        with self.assertRaisesRegex(ValueError, 'Unique formal'): m.attach_statement_geometry(row, None, finder)

    def test_currency_witness_spaced_nontarget_amount_is_not_parsed(self):
        texts, spec = fixture(); texts[0] = texts[0].replace('Net sales $ 5.01 $ 4.00 $ 20,000.50 $ 18,000.25',
            'Net sales:\nProducts $ 1 2 3 $ 2 $ 10 $ 9\nTotal net sales 5.01 4.00 20,000.50 18,000.25')
        row = m.parse_revenue(texts, spec)['revenue']['current']
        self.assertEqual(row['amount_decimal'], '20000.50')
        self.assertIn('1 2 3', row['currency_evidence']['text'])
        texts[0] = texts[0].replace('Products $ 1 2 3 $ 2 $ 10 $ 9', 'Products $ 1 2 3 2 $ 10 $ 9')
        with self.assertRaisesRegex(ValueError, 'Four explicit'): m.parse_revenue(texts, spec)


if __name__ == '__main__':
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests))
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(m.write_new('TESTS.json', {'created_utc': m.now(), 'passed': True, 'synthetic_only': True,
        'tests_run': result.testsRun, 'source_sha256': m.sha(Path(m.__file__).read_bytes()),
        'tests_sha256': m.sha(Path(__file__).read_bytes()), 'log': stream.getvalue(),
        'real_financial_sources_parsed': 0}))
