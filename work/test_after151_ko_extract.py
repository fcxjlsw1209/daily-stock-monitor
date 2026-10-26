"""Synthetic-only KO parser tests; no original source documents are opened."""
import copy
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
import after151_ko_extract as m


def td(text='', span=1, rowspan=1):
    return f'<td colspan="{span}" rowspan="{rowspan}">{text}</td>'


def tr(*cells):
    return '<tr>'+''.join(cells)+'</tr>'


def group(value, dollar=True):
    if value.startswith('(') and value.endswith(')'):
        return td('$ (' if dollar else '(')+td(value[1:-1])+td(')')
    return td('$' if dollar else '')+td(value)+td()


def fixture(year=2016, reverse=False, narrow=False, ni='(1,234.50)', assets='8,000', prior='7,000',
            unit='millions', dollar=True):
    ys = [year-1, year] if reverse else [year, year-1]
    ci = ys.index(year)
    ih = [td()]
    for y in ys+ys:
        ih += ([td(), td(f'December 31, {y}'), td()] if narrow
               else [td(f'December 31, {y}', 3)])+[td()]
    bh = [td()]
    for y in ys:
        bh += ([td(), td(f'December 31, {y}'), td()] if narrow
               else [td(f'December 31, {y}', 3)])+[td()]
    income = [tr(td('Condensed Consolidated Statements of Income', 17)),
              tr(td(f'(In {unit} except per share data)', 17)),
              tr(td(), td('Three Months Ended', 7), td(), td('Year Ended', 7), td()),
              tr(*ih), tr(td('Net Operating Revenues'), td('66', 16)),
              tr(td('Consolidated Net Income'), group('999', dollar), td(), group('888', dollar), td(),
                 *sum(([group(ni if y == year else '111', dollar), td()] for y in ys), [])),
              tr(td('Net Income Attributable to Shareowners of The Coca-Cola Company'), td('555', 16))]
    balance = [tr(td('Condensed Consolidated Balance Sheets', 9)),
               tr(td(f'(In {unit} except par value)', 9)), tr(*bh), tr(td('ASSETS', 9)),
               tr(td('Total Assets'),
                  *sum(([group(assets if y == year else prior, dollar), td()] for y in ys), []))]
    raw = ('<html><p>The Coca-Cola Company (NYSE: KO)</p><table>'+''.join(income)+
           '</table><table>'+''.join(balance)+'</table></html>').encode()
    source = {'path': 'synthetic/ko.html', 'sha256': m.sha256(raw),
              'url': 'https://investors.coca-colacompany.com/synthetic/annual',
              'receipt_path': 'synthetic/ko.html.meta.json', 'receipt_sha256': 'a'*64}
    context = {'symbol': 'KO', 'event_id': f'KO_{year+1}-02-01', 'fiscal_year': year,
               'context_status': m.STATUS, 'source': source,
               'evidence_path': source['path'], 'evidence_sha256': source['sha256'], 'source_url': source['url'],
               'receipt': {'path': source['receipt_path'], 'sha256': source['receipt_sha256']},
               'identity_anchors': ['The Coca-Cola Company', 'NYSE: KO'],
               'publication_date': f'{year+1}-02-01', 'available_session': f'{year+1}-02-02',
               'official_actual_period_end': f'{year}-12-31',
               'context_holds': ['unverified_money_geometry', 'future_purpose_review', 'future_purpose_review'],
               'accounting_terms': ['synthetic accounting note'],
               'accounting_context_numbers_redacted': ['No numeric source facts here'],
               'periods': {'current_end': f'{year}-12-31', 'prior_end': f'{year-1}-12-31',
                           'proposed_current_start': f'{year}-01-01', 'start_is_source_quoted': False,
                           'start_assumption_approved': False,
                           'start_assumption': 'same_current_report_prior_annual_end_plus_one_calendar_day'}}
    tables = BeautifulSoup(raw, 'html.parser').find_all('table')
    fields = {}
    for name, ti, row, base, yi, end in (
            ('annual_net_income', 0, 5, 9, ci, f'{year}-12-31'),
            ('current_total_assets', 1, 4, 1, ci, f'{year}-12-31'),
            ('prior_total_assets', 1, 4, 1, 1-ci, f'{year-1}-12-31')):
        _, _, physical = m.table_grid(tables[ti])
        headers = [dict({k: cell[k] for k in m.GEOMETRY}, text=cell['text'])
                   for cell in physical.values() if cell['row'] < (4 if ti == 0 else 3)]
        geometry = [{k: cell[k] for k in m.GEOMETRY} for cell in physical.values() if cell['row'] == row]
        start = base + 4*yi
        fields[name] = {'table': ti, 'row': row, 'label_cell': 0,
                        'label': 'Consolidated Net Income' if ti == 0 else 'Total Assets',
                        'logical_columns': [start+1] if narrow else list(range(start, start+3)),
                        'as_of': end, 'scale': '1000000' if unit == 'millions' else '1000',
                        'currency_literal': '$', 'currency_iso': None, 'available_from': context['available_session'],
                        'source': copy.deepcopy(source), 'header_evidence': headers,
                        'row_geometry_without_values': geometry}
    context['fields'] = fields
    return raw, context


def changed(raw, context, before, after):
    raw = raw.replace(before.encode(), after.encode())
    c = copy.deepcopy(context)
    c['source']['sha256'] = c['evidence_sha256'] = m.sha256(raw)
    for spec in c['fields'].values():
        spec['source'] = copy.deepcopy(c['source'])
    return raw, c


class ExtractionTests(unittest.TestCase):
    def out(self, **kw):
        return m.extract_document(*fixture(**kw))

    def test_exact_three_fields_and_consolidated_not_parent_or_quarter(self):
        out = self.out()
        self.assertEqual({k: v['value_decimal'] for k, v in out['fields'].items()},
                         {'annual_net_income': '-1234.50', 'current_total_assets': '8000', 'prior_total_assets': '7000'})
        self.assertTrue(out['complete_required_fields'])
        self.assertEqual(out['gaps'], [])

    def test_all_six_synthetic_years_supported(self):
        for year in m.YEARS:
            self.assertTrue(self.out(year=year)['complete_required_fields'])

    def test_year_order_not_assumed(self):
        for reverse in (False, True):
            self.assertEqual(self.out(reverse=reverse)['fields']['annual_net_income']['value_decimal'], '-1234.50')

    def test_narrow_anchor_includes_owned_parentheses(self):
        out = self.out(narrow=True)
        self.assertEqual(out['fields']['annual_net_income']['value_decimal'], '-1234.50')
        self.assertEqual(len(out['fields']['annual_net_income']['locator']['physical_cells']), 3)

    def test_negative_zero_and_unicode_minus(self):
        for value, expected in [('0', '0'), ('-15', '-15'), ('−15', '-15'), ('(0)', '-0')]:
            with self.subTest(value=value):
                self.assertEqual(self.out(ni=value)['fields']['annual_net_income']['value_decimal'], expected)

    def test_large_decimal_does_not_round(self):
        number = '123456789012345678901234567890123456.987654321'
        self.assertEqual(self.out(ni='('+number+')')['fields']['annual_net_income']['value_decimal'], '-'+number)

    def test_original_text_and_markup_hash_retained(self):
        raw, c = changed(*fixture(ni='1,234'), '1,234', '1&nbsp;,&nbsp;234')
        field = m.extract_document(raw, c)['fields']['annual_net_income']
        self.assertEqual(field['value_decimal'], '1234')
        cells = field['locator']['physical_cells']
        self.assertTrue(any('\xa0' in x['original_text'] for x in cells))
        self.assertTrue(all(len(x['parsed_cell_markup_sha256']) == 64 for x in cells))

    def test_missing_ambiguous_or_conflicting_amounts_not_zero(self):
        for value in ('', '-', '—', 'N/A', '123 456', '(-123)', '1,23', '$ 2 $'):
            with self.subTest(value=value):
                self.assertIsNone(self.out(ni=value)['fields']['annual_net_income'])

    def test_source_and_holds_exactly_preserved(self):
        raw, c = fixture()
        out = m.extract_document(raw, c)
        self.assertEqual(out['source'], c['source'])
        self.assertEqual(out['context_holds'], c['context_holds'])
        self.assertEqual(out['accounting_context_numbers_redacted'], c['accounting_context_numbers_redacted'])
        for f in out['fields'].values():
            self.assertEqual(f['source'], c['source'])
            self.assertFalse(f['fact_admitted'])
            self.assertIsNone(f['currency_iso'])

    def test_no_mutation_or_io(self):
        raw, c = fixture()
        before = copy.deepcopy(c)
        with patch('builtins.open', side_effect=AssertionError('I/O forbidden')):
            m.extract_document(raw, c)
        self.assertEqual(c, before)

    def test_period_and_admission_schema(self):
        out = self.out()
        cur = {'period_start': '2016-01-01', 'period_end': '2016-12-31', 'days_inclusive': 366}
        self.assertEqual(out['periods']['current'], cur)
        self.assertEqual(out['periods']['comparison'], {'period_end': '2015-12-31'})
        self.assertFalse(out['periods']['start_assumption_certified'])
        self.assertEqual(out['fields']['annual_net_income']['period'], cur)
        self.assertEqual(out['fields']['annual_net_income']['as_of'], '2016-12-31')
        self.assertIsNone(out['fields']['current_total_assets']['period'])
        self.assertEqual(out['fields']['prior_total_assets']['as_of'], '2015-12-31')
        self.assertTrue(all(out[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')))

    def test_raw_hash_and_identity_fail_hard(self):
        raw, c = fixture()
        with self.assertRaises(m.SourceError):
            m.extract_document(raw+b' ', c)
        raw, c = changed(raw, c, 'Coca-Cola Company', 'Another Company')
        with self.assertRaises(m.SourceError):
            m.extract_document(raw, c)

    def test_source_reference_mismatch_fails_hard(self):
        for key, value in [('evidence_path', 'other.html'), ('evidence_sha256', 'f'*64),
                           ('source_url', 'https://other.example'), ('event_id', 'KO_2010-01-01')]:
            raw, c = fixture(); c[key] = value
            with self.subTest(key=key), self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_wrong_scope_and_status_fail_hard(self):
        for key, value in [('symbol','PEP'), ('fiscal_year',2017), ('fiscal_year',True),
                           ('context_status','pdf_text_leads_only_geometry_pending')]:
            raw, c = fixture(); c[key] = value
            with self.subTest(key=key), self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_source_url_and_path_constraints(self):
        for key, value in [('url','http://investors.coca-colacompany.com/a'),
                           ('url','https://evil.example/a'), ('path','../a'), ('receipt_path','/tmp/a')]:
            raw, c = fixture(); c['source'][key] = value
            with self.subTest(key=key), self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_context_dates_invalid_all_fields_gap(self):
        for key, value in [('current_end','2016-11-30'), ('prior_end',None),
                           ('proposed_current_start','2016-01-02'), ('start_is_source_quoted',True),
                           ('start_assumption_approved',True)]:
            raw, c = fixture(); c['periods'][key] = value
            out = m.extract_document(raw,c)
            self.assertEqual(len(out['gaps']),3)
            self.assertEqual(out['context_holds'], c['context_holds'])

    def test_publication_and_availability_order(self):
        for pub, available in [('2016-12-30','2017-01-02'),('2017-02-01','2017-02-01'),('invalid','2017-02-02')]:
            raw,c=fixture();c['publication_date']=pub;c['event_id']='KO_'+pub;c['available_session']=available
            self.assertEqual(len(m.extract_document(raw,c)['gaps']),3)

    def test_field_source_date_and_currency_mismatch(self):
        for key,value in [('source',{}),('as_of','2015-12-31'),('available_from','2017-02-03'),
                          ('currency_literal','EUR'),('currency_iso','USD'),('scale','1000')]:
            raw,c=fixture();c['fields']['annual_net_income'][key]=value
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_quarterly_column_is_not_annual(self):
        raw,c=fixture();c['fields']['annual_net_income']['logical_columns']=[1,2,3]
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_wrong_year_columns(self):
        raw,c=fixture();c['fields']['annual_net_income']['logical_columns']=[13,14,15]
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_parent_attributable_label_is_rejected(self):
        raw,c=fixture();spec=c['fields']['annual_net_income'];spec['row']=6
        spec['label']='Net Income Attributable to Shareowners of The Coca-Cola Company'
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_duplicate_consolidated_ni_row_rejected(self):
        raw,c=changed(*fixture(),'Net Income Attributable to Shareowners of The Coca-Cola Company','Consolidated Net Income')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_non_gaap_or_wrong_formal_title_rejected(self):
        for title in ('Non-GAAP Statements of Income','Consolidated Statements of Cash Flows',
                      'Non-GAAP Condensed Consolidated Statements of Income'):
            raw,c=changed(*fixture(),'Condensed Consolidated Statements of Income',title)
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_missing_conflicting_units_and_currency_rejected(self):
        self.assertIsNone(self.out(dollar=False)['fields']['annual_net_income'])
        for unit in ('unspecified', 'millions) (in thousands'):
            self.assertIsNone(self.out(unit=unit)['fields']['annual_net_income'])
        raw,c=changed(*fixture(),'In millions except per share data','In millions EUR except per share data')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_tampered_fixed_header_evidence_rejected(self):
        raw,c=fixture();c['fields']['annual_net_income']['header_evidence'][0]['text']='Other title'
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_row_geometry_tamper_rejected(self):
        raw,c=fixture();c['fields']['annual_net_income']['row_geometry_without_values'][1]['columns']=[999]
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_rowspan_amount_rejected(self):
        raw,c=changed(*fixture(ni='123'),td('123'),td('123',rowspan=2))
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_cross_year_colspan_rejected(self):
        raw,c=fixture()
        before=tr(td('Total Assets'),group('8,000'),td(),group('7,000'),td())
        after=tr(td('Total Assets'),td('$ 8,000',7),td())
        raw,c=changed(raw,c,before,after)
        out=m.extract_document(raw,c)
        self.assertIsNone(out['fields']['current_total_assets'])
        self.assertIsNone(out['fields']['prior_total_assets'])

    def test_adjacent_year_parenthesis_cannot_be_borrowed(self):
        physical={(2,0):{'text':'Total Assets','columns':[0],'row':2,'physical_cell':0,'rowspan':1},
                  (2,1):{'text':'(123','columns':[1],'row':2,'physical_cell':1,'rowspan':1},
                  (2,2):{'text':')','columns':[2],'row':2,'physical_cell':2,'rowspan':1}}
        with self.assertRaisesRegex(m.FieldGap,'crosses_year'):
            m.old._value(physical,physical,2,[1],{1})

    def test_nested_statement_table_rejected(self):
        raw,c=changed(*fixture(),'Net Operating Revenues','Net Operating Revenues<table><tr><td>nested</td></tr></table>')
        self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])

    def test_negative_assets_preserved_not_absoluted(self):
        self.assertEqual(self.out(assets='(1)')['fields']['current_total_assets']['value_decimal'],'-1')

    def test_field_missing_does_not_remove_event_or_other_fields(self):
        raw,c=fixture();del c['fields']['annual_net_income']
        out=m.extract_document(raw,c)
        self.assertEqual(out['event_id'],c['event_id'])
        self.assertEqual(len(out['gaps']),1)
        self.assertIsNotNone(out['fields']['current_total_assets'])

    def test_invalid_column_shapes_rejected(self):
        for cols in ([True],[1,1],[],[999],None):
            raw,c=fixture();c['fields']['annual_net_income']['logical_columns']=cols
            self.assertIsNone(m.extract_document(raw,c)['fields']['annual_net_income'])


if __name__=='__main__':
    unittest.main(verbosity=2)
