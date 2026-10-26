"""Fabricated source-only fixtures; never reads original issuer PDFs."""
from copy import deepcopy
import json
import unittest

import operating_margin172_aapl_sources as m


def fixture():
    op = '\n'.join([
        'Apple Inc.', m.TITLE, '(In millions, except shares)', '(Unaudited)',
        'Three Months Ended                  Twelve Months Ended',
        'September 24, September 26, September 24, September 26,',
        '2016 2015 2016 2015',
        'Net sales $ 10 $ 20 $ 100 $ 200',
        'Cost of sales 4 8 40 80', 'Gross margin 6 12 60 120',
        'Operating expenses: 1 2 10 20', 'Operating income 5 10 50 (100)',
        'Other income/(expense), net 1 1 2 2',
        'Income before provision for income taxes 6 11 52 (98)',
        'Net income $ 4 $ 9 $ 45 $ (90)',
    ])
    bs = '\n'.join(['Apple Inc.', 'CONDENSED CONSOLIDATED BALANCE SHEETS', '(Unaudited)',
                     'September 24, September 26,', '2016 2015'])
    dates = ['2016-09-24', '2015-09-26', '2016-09-24', '2015-09-26']
    prior = dict(symbol='AAPL', fiscal_year=2016,
        metadata=dict(header_dates=dates,
            current_period=dict(start='2015-09-27', end=dates[2], duration_literal='Twelve Months Ended'),
            comparative_period=dict(start=None, end=dates[3], duration_literal='Twelve Months Ended')),
        publication=dict(release_date='2016-10-25'),
        availability=dict(candidate_first_usable_session='2016-10-26'),
        source_url='https://www.apple.com/newsroom/pdfs/example.pdf',
        revenue={role: dict(amount_decimal=amount, all_row_cells=['10', '20', '100', '200'],
            date=dates[col], scale=1000000) for role, amount, col in [('current', '100', 2), ('comparative', '200', 3)]})
    return [op, bs], prior


def geometry(parsed):
    found, index = {}, 0
    def add(term, text, x, y, width=10):
        nonlocal index
        index += 20
        found.setdefault(term, []).append(dict(index=index, count=len(text), text=text,
                                               bbox=[x, y - 1, x + width, y + 1]))
    add(m.TITLE, m.TITLE, 20, 750, 400)
    add('In millions', 'In millions', 20, 730, 80)
    add('Twelve Months Ended', 'Twelve Months Ended', 350, 710, 120)
    add('Cost of sales', 'Cost of sales', 20, 650, 90)
    add('Net income', 'Net income', 20, 400, 80)
    for name, y in [('revenue', 680), ('operating_income', 500)]:
        row = parsed['fields'][name]['current']
        add(row['measure_label'], row['measure_label'], 20, y, 100)
        for col, text in enumerate(row['all_row_cells']):
            add(text, text, 200 + col * 70, y, 30)
    return found


class AppleTests(unittest.TestCase):
    def test_exact_annual_four_cells_and_signed_income(self):
        texts, prior = fixture()
        result = m.parse_layout(texts, prior)
        self.assertEqual(result['fields']['revenue']['current']['amount_decimal'], '100')
        self.assertEqual(result['fields']['operating_income']['current']['amount_decimal'], '50')
        negative = result['fields']['operating_income']['comparative']
        self.assertEqual(negative['displayed_amount'], '(100)')
        self.assertEqual(negative['amount_decimal'], '-100')
        self.assertEqual(negative['sign'], -1)
        self.assertEqual(negative['scale'], 1000000)
        self.assertIsNone(negative['currency_literal'])
        self.assertEqual(negative['statement_currency_evidence']['literal'], '$')
        self.assertIsNone(negative['iso_currency'])
        self.assertFalse(negative['fact_admitted'])

    def test_comparative_current_release_clock_and_unknown_start(self):
        texts, prior = fixture()
        result = m.parse_layout(texts, prior)
        for roles in result['fields'].values():
            self.assertEqual(roles['current']['known_at_session'], '2016-10-26')
            self.assertEqual(roles['comparative']['known_at_session'], '2016-10-26')
            self.assertIsNone(roles['comparative']['period']['start'])
            self.assertFalse(roles['comparative']['historical_original_event_created'])

    def test_bad_or_double_sign_rejected(self):
        for value in ['(5', '5)', '(-5)', '(1,234.50', '1,23', 'NaN', '5%']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.decimal_cell(value)
        self.assertEqual(m.decimal_cell('-4'), '-4')
        self.assertEqual(m.decimal_cell('(1,234.50)'), '-1234.50')

    def test_quarter_only_and_unordered_duration_rejected(self):
        texts, prior = fixture()
        for replacement in ['Three Months Ended', 'Twelve Months Ended Three Months Ended']:
            bad = [texts[0].replace('Three Months Ended                  Twelve Months Ended', replacement), texts[1]]
            with self.assertRaises(ValueError):
                m.parse_layout(bad, prior)

    def test_income_missing_duplicated_or_outside_scope_rejected(self):
        texts, prior = fixture()
        row = 'Operating income 5 10 50 (100)'
        for value in [texts[0].replace(row, ''), texts[0].replace(row, row + '\n' + row),
                      texts[0].replace(row, '') + '\n' + row]:
            with self.assertRaises(ValueError):
                m.parse_layout([value, texts[1]], prior)

    def test_non_gaap_and_other_operating_income_not_accepted(self):
        texts, prior = fixture()
        for value in [texts[0] + '\nNON-GAAP', texts[0].replace('Operating income ', 'Other operating income ')]:
            with self.assertRaises(ValueError):
                m.parse_layout([value, texts[1]], prior)

    def test_no_quarter_substitution_for_bad_annual_cell(self):
        texts, prior = fixture()
        bad = texts[0].replace('Operating income 5 10 50 (100)', 'Operating income 5 10 -- (100)')
        with self.assertRaises(ValueError):
            m.parse_layout([bad, texts[1]], prior)

    def test_revenue_parity_required(self):
        texts, prior = fixture()
        prior['revenue']['current']['amount_decimal'] = '999'
        with self.assertRaisesRegex(ValueError, 'pinned revenue'):
            m.parse_layout(texts, prior)

    def test_year_end_and_balance_header_binding(self):
        texts, prior = fixture()
        for badprior in [dict(prior, fiscal_year=2017), dict(prior, publication=dict(release_date='2016-09-01'))]:
            with self.assertRaises(ValueError):
                m.parse_layout(texts, badprior)
        with self.assertRaises(ValueError):
            m.parse_layout([texts[0], texts[1].replace('September 26', 'September 25')], prior)

    def test_unit_not_inferred(self):
        texts, prior = fixture()
        with self.assertRaises(ValueError):
            m.parse_layout([texts[0].replace('(In millions, except shares)', ''), texts[1]], prior)

    def test_sales_footnote_and_products_currency_preserved(self):
        texts, prior = fixture()
        value = texts[0].replace('Net sales $ 10 $ 20 $ 100 $ 200',
            'Net sales:\nProducts $ 8 $ 17 $ 90 $ 180\nServices 2 3 10 20\nTotal net sales (1) 10 20 100 200')
        value += '\n(1) Net sales by geographical segment'
        result = m.parse_layout([value, texts[1]], prior)
        self.assertEqual(result['fields']['revenue']['current']['label_footnote']['literal'], '(1)')
        self.assertIn('Products', result['statement']['currency_evidence']['scope'])
        with self.assertRaisesRegex(ValueError, 'footnote'):
            m.parse_layout([value.replace('(1) Net sales by geographical segment', ''), texts[1]], prior)

    def test_geometry_signed_columns_and_exact_match(self):
        texts, prior = fixture()
        result = m.parse_layout(texts, prior)
        found = geometry(result)
        m.attach_geometry(result, None, lambda _, term: found.get(term, []))
        self.assertEqual(result['fields']['operating_income']['comparative']['locator']['pdfium_cell_bbox'][0], 410)
        self.assertTrue(result['fields']['revenue']['current']['independent_PDFium_cell_text_match'])
        found['50'][0]['text'] = '51'
        with self.assertRaises(ValueError):
            m.attach_geometry(result, None, lambda _, term: found.get(term, []))

    def test_geometry_duplicate_formal_label_rejected(self):
        texts, prior = fixture()
        result = m.parse_layout(texts, prior)
        found = geometry(result)
        found['Operating income'].append(deepcopy(found['Operating income'][0]))
        with self.assertRaises(ValueError):
            m.attach_geometry(result, None, lambda _, term: found.get(term, []))

    def test_next_session_strictly_after_release(self):
        self.assertEqual(m.first_session('2016-10-25', ['2016-10-24', '2016-10-25', '2016-10-26']), '2016-10-26')
        with self.assertRaises(ValueError):
            m.first_session('2016-10-26', ['2016-10-25', '2016-10-26'])

    def test_parent_visible_structured_and_link_agree(self):
        _, prior = fixture()
        raw = ('<span class="category-eyebrow__date">October 25, 2016</span>'
            '<h1>Apple Reports Fourth Quarter Results</h1>'
            '<script type="application/ld+json">' + json.dumps({'datePublished': '2016-10-25'}) + '</script>'
            '<a href="/newsroom/pdfs/example.pdf">Financials</a>').encode()
        result = m.publication(raw, prior, 'https://www.apple.com/newsroom/results/')
        self.assertFalse(result['historical_web_snapshot_verified'])
        for replacement in [raw.replace(b'2016-10-25', b'2016-10-26'), raw.replace(b'example.pdf', b'wrong.pdf')]:
            with self.assertRaises(ValueError):
                m.publication(replacement, prior, 'https://www.apple.com/newsroom/results/')

    def test_receipt_exact_bytes_and_original_identity(self):
        raw = b'%PDF-fabricated'
        url = 'https://www.apple.com/newsroom/pdfs/test.pdf'
        receipt = dict(status=200, url=url, final_url=url, public_unauthenticated=True,
            response_sha256=m.digest(raw), response_bytes=len(raw), retrieved_at='2026-01-01T00:00:00Z',
            content_type='application/pdf', decoder='identity', decoded_sha256=m.digest(raw))
        m.receipt_check(receipt, raw, url, pdf=True)
        with self.assertRaises(ValueError):
            m.receipt_check(receipt, raw + b'changed', url, pdf=True)
        with self.assertRaises(ValueError):
            m.receipt_check(dict(receipt, decoder='base64'), raw, url, pdf=True)


if __name__ == '__main__':
    unittest.main()
