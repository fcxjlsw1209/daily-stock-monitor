"""Synthetic-only KLAC source parsing checks; no real report files are read."""
from copy import deepcopy
from decimal import localcontext
import ast
from pathlib import Path
import unittest
from bs4 import BeautifulSoup
import after150_klac_extract as r


def cell(text='', span=1, rowspan=1):
    return '<td colspan="'+str(span)+'" rowspan="'+str(rowspan)+'">'+text+'</td>'


def row(*values):
    return '<tr>'+''.join(values)+'</tr>'


def wide_values(label,values,merged=False):
    items=[cell(label)]
    for i,value in enumerate(values):
        if i:items.append(cell())
        items.extend([cell(value,2),cell()] if merged else [cell('$'),cell(value),cell()])
    return row(*items)


def compact_values(label,values,gap):
    items=[cell(label)]
    for i,value in enumerate(values):
        if i:items.extend(cell() for _ in range(gap))
        items.append(cell('$ '+value))
    return row(*items)


def fixture(layout='wide',year=2020,ni='10',assets=('150','50'),merged=False):
    if layout=='wide':
        bh=row(cell('( In thousands )'),cell('June 30, '+str(year),3),cell(),cell('June 30, '+str(year-1),3))
        ih=row(cell('( In thousands, except per share amounts )'),cell('June 30, '+str(year),3),cell(),cell('June 30, '+str(year-1),3),cell(),cell('June 30, '+str(year),3),cell(),cell('June 30, '+str(year-1),3))
        groups=row(cell(),cell('Three months ended',7),cell(),cell('Twelve months ended',7))
        bsvalues=lambda label,vals:wide_values(label,vals)
        invalues=lambda label,vals:wide_values(label,vals,merged and label=='Net income')
        cols=([1,2,3],[5,6,7],[9,10,11],[13,14,15])
    else:
        gap=2 if layout=='compact_gap2' else 1
        group=2+gap
        between=gap+1
        bsitems=[cell('( In thousands )'),cell('June 30, '+str(year))]+[cell() for _ in range(gap)]+[cell('June 30, '+str(year-1))]
        bh=row(*bsitems)
        groups=row(cell(),cell('Three months ended June 30,',group),*[cell() for _ in range(between)],cell('Twelve months ended June 30,',group))
        ihitems=[cell('( In thousands, except per share amounts )'),cell(str(year))]+[cell() for _ in range(gap)]+[cell(str(year-1))]+[cell() for _ in range(between)]+[cell(str(year))]+[cell() for _ in range(gap)]+[cell(str(year-1))]
        ih=row(*ihitems)
        def iv(label,values):
            items=[cell(label)]
            for i,v in enumerate(values):
                if i:items.extend(cell() for _ in range(between if i==2 else gap))
                items.append(cell('$ '+v))
            return row(*items)
        bsvalues=lambda label,vals:compact_values(label,vals,gap)
        invalues=iv
        start=1+group+between
        cols=([1],[2+gap],[start],[start+1+gap])
    balance='<table>'+row(cell('Condensed Consolidated Unaudited Balance Sheets'))+bh+row(cell('Assets'))+bsvalues('Cash and cash equivalents',('20','30'))+bsvalues('Total assets',assets)+'</table>'
    income='<table>'+row(cell('Condensed Consolidated Unaudited Statements of Operations',4))+groups+ih+invalues('Revenues',('7','8','9','10'))+invalues('Net income',('1','2',ni,'4'))+invalues('Net income attributable to KLA',('21','22','23','24'))+'</table>'
    raw=('<html>KLA-Tencor Corporation (NASDAQ: KLAC)<table><tr><td>navigation</td></tr></table>'+balance+income+'</html>').encode()
    c=dict(event_id='KLAC_'+str(year)+'-07-30',symbol='KLAC',fiscal_year=year,fiscal_quarter=4,
           publication_date=str(year)+'-07-30',available_session=str(year)+'-07-31',official_actual_period_end=str(year)+'-06-30',
           evidence_path='synthetic/report.html',evidence_sha256=r.digest(raw),source_url='https://ir.kla.com/news-events/press-releases/detail/999/generated',
           source_hash_verified=True,catalog_claims_issuer_full_body=True,html_context_reviewed=True,format='html_or_text',source_kind='issuer_original_release_full_bytes')
    soup=BeautifulSoup(raw,'html.parser');tables=soup.find_all('table');frozen=[]
    for ti,stop in ((1,2),(2,3)):
        _,_,p=r.table_grid(tables[ti])
        frozen.append(dict(table=ti,header_stop_row=stop,headers=[{k:deepcopy(v[k]) for k in ('row','physical_cell','columns','rowspan','text')} for (ri,_),v in p.items() if ri<stop]))
    annual=next(h for h in frozen[1]['headers'] if 'Twelve months ended' in h['text'])
    c['context']=dict(source_sha_verified=True,financial_values_extracted=0,receipt=dict(status=200,url_and_raw_sha_match=True),statement_tables=frozen,accounting_contexts_numbers_redacted=[])
    c['context_summary']=dict(status='formal_annual_NI_and_two_assets_endpoints_located_no_values_extracted',fact_admitted=False,
        current_annual_period=dict(start=str(year-1)+'-07-01',end=str(year)+'-06-30',basis='Interpreted from explicit Twelve months ended June30 header'),
        comparison_assets_date=str(year-1)+'-06-30',
        units=dict(scale='1000',currency_literal='$',currency_iso=None),accounting_attention=[],
        net_income_locator=dict(table=2,row=4,label_cell=0,label='Net income',annual_header=annual,current_header_columns=cols[2],comparison_header_columns=cols[3]),
        assets_locator=dict(table=1,row=4,label_cell=0,label='Total assets',current_header_columns=cols[0],comparison_header_columns=cols[1]))
    return raw,c


def changed(raw,c,old,new):
    raw=raw.replace(old.encode(),new.encode());c=deepcopy(c);c['evidence_sha256']=r.digest(raw)
    return raw,c


def fields(raw,c):
    return r.extract_document(raw,c)['fields']


class LayoutTests(unittest.TestCase):
    def test_wide_three_column_groups(self):
        raw,c=fixture();out=r.extract_document(raw,c)
        self.assertEqual([out['fields'][k]['value_decimal'] for k in r.FIELDS],['10','150','50'])
        self.assertEqual(out['gaps'],[])
    def test_compact_single_amount_cells(self):
        raw,c=fixture('compact');v=fields(raw,c)
        self.assertEqual([v[k]['value_decimal'] for k in r.FIELDS],['10','150','50'])
    def test_compact_two_blank_separator_cells(self):
        raw,c=fixture('compact_gap2');v=fields(raw,c)
        self.assertEqual([v[k]['value_decimal'] for k in r.FIELDS],['10','150','50'])
    def test_wide_amount_colspan_two_inside_date(self):
        raw,c=fixture(merged=True);v=fields(raw,c)
        self.assertEqual(v[r.FIELDS[0]]['value_decimal'],'10')
    def test_parent_attributable_not_selected(self):
        raw,c=fixture();v=fields(raw,c)
        self.assertEqual(v[r.FIELDS[0]]['value_decimal'],'10')
    def test_quarter_cashflow_not_substituted(self):
        raw,c=fixture();extra='<table>'+row(cell('Condensed Consolidated Statements of Cash Flows'))+row(cell(),cell('Three months ended June 30, 2020'))+row(cell('Net income'),cell('$ 888'))+'</table>'
        raw,c=changed(raw,c,'</html>',extra+'</html>')
        self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'10')
    def test_substantial_tax_subtotals_do_not_replace_total(self):
        raw,c=fixture();raw,c=changed(raw,c,'Revenues','Income before income taxes')
        # The frozen header boundary must not silently extend into monetary rows.
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_duplicate_income_table_rejected(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser');extra=str(soup.find_all('table')[2]);raw,c=changed(raw,c,'</html>',extra+'</html>')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_duplicate_exact_income_row_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'Net income attributable to KLA','Net income')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_only_attributable_income_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'>Net income<','>Net income attributable to KLA<')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_per_share_income_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'>Net income<','>Net income per share<')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_non_gaap_formal_title_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'Condensed Consolidated Unaudited Statements of Operations','Non-GAAP Condensed Consolidated Unaudited Statements of Operations')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_unrelated_prior_non_gaap_notice_ignored(self):
        raw,c=fixture();raw,c=changed(raw,c,'<html>','<html><p>Non-GAAP reconciliation follows below.</p>')
        self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'10')
    def test_wrong_type_cashflow_table_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'Statements of Operations','Statements of Cash Flows')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_no_source_amount_hardcoded(self):
        raw,c=fixture(ni='317',assets=('842','621'))
        self.assertEqual([fields(raw,c)[k]['value_decimal'] for k in r.FIELDS],['317','842','621'])


class MonetaryTests(unittest.TestCase):
    def test_negative_parentheses_exact(self):
        raw,c=fixture(ni='(1,234)');v=fields(raw,c)[r.FIELDS[0]]
        self.assertEqual(v['value_decimal'],'-1234')
    def test_unicode_minus(self):
        raw,c=fixture('compact',ni='−17');self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'-17')
    def test_negative_assets_retained_not_filtered(self):
        raw,c=fixture(assets=('(150)','0'))
        self.assertEqual([fields(raw,c)[k]['value_decimal'] for k in r.FIELDS[1:]],['-150','0'])
    def test_zero_not_missing(self):
        raw,c=fixture(ni='0');self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'0')
    def test_comma_whitespace(self):
        raw,c=fixture(ni='1 , 234');self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'1234')
    def test_low_precision_large_negative_not_rounded(self):
        raw,c=fixture(ni='(12345678901234567890123456789)')
        with localcontext() as d:
            d.prec=3
            self.assertEqual(fields(raw,c)[r.FIELDS[0]]['value_decimal'],'-12345678901234567890123456789')
    def test_dash_is_gap_not_zero(self):
        raw,c=fixture(ni='—');self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_two_values_are_gap(self):
        raw,c=fixture(ni='10 20');self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_unbalanced_parentheses(self):
        raw,c=fixture(ni='(10');self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_double_sign(self):
        raw,c=fixture(ni='(-10)');self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_invalid_grouping(self):
        raw,c=fixture(ni='12,34');self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_nonfinite(self):
        for value in ('NaN','Infinity','100%','10*'):
            raw,c=fixture(ni=value);self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_currency_before_or_after_parenthesis(self):
        for parts in (['$','(123)'],['(','$','123',')']):
            self.assertEqual(r._amount(parts)[2],'-123')
    def test_missing_dollar_evidence(self):
        raw,c=fixture();raw,c=changed(raw,c,'$','')
        self.assertTrue(all(v is None for v in fields(raw,c).values()))
    def test_currency_conflict(self):
        raw,c=fixture();raw,c=changed(raw,c,'Revenues','Revenues CAD')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_conflicting_scale_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'In thousands, except per share amounts','In millions, except per share amounts')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])


class GeometryTests(unittest.TestCase):
    def test_quarter_columns_not_annual(self):
        raw,c=fixture();c['context_summary']['net_income_locator']['current_header_columns']=[1,2,3]
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_prior_column_not_current(self):
        raw,c=fixture();loc=c['context_summary']['net_income_locator'];loc['current_header_columns'],loc['comparison_header_columns']=loc['comparison_header_columns'],loc['current_header_columns']
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_noncontiguous_columns_rejected(self):
        raw,c=fixture();c['context_summary']['net_income_locator']['current_header_columns']=[9,11]
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_changed_date_header_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'June 30, 2020','July 1, 2020')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_three_six_nine_months_rejected(self):
        for period in ('Three','Six','Nine'):
            raw,c=fixture();raw,c=changed(raw,c,'Twelve months ended',period+' months ended')
            self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_amount_crosses_selected_group(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser');cells=soup.find_all('table')[2].find_all('tr')[4].find_all('td');cells[10]['colspan']='4'
        raw=str(soup).encode();c['evidence_sha256']=r.digest(raw)
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_amount_rowspan_rejected(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser');cells=soup.find_all('table')[2].find_all('tr')[4].find_all('td');cells[10]['rowspan']='2'
        raw=str(soup).encode();c['evidence_sha256']=r.digest(raw)
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_outside_open_parenthesis_not_ignored(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser');cells=soup.find_all('table')[2].find_all('tr')[4].find_all('td');cells[8].string='('
        raw=str(soup).encode();c['evidence_sha256']=r.digest(raw)
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_nested_source_statement_rejected(self):
        raw,c=fixture();raw,c=changed(raw,c,'>Net income<','><table><tr><td>Net income</td></tr></table><')
        self.assertIsNone(fields(raw,c)[r.FIELDS[0]])
    def test_headers_and_cells_have_hash_proof(self):
        raw,c=fixture();loc=fields(raw,c)[r.FIELDS[0]]['locator']
        self.assertTrue(all(len(x['markup_sha256'])==64 for key in ('header_cells','physical_cells','scale_cells','dollar_evidence') for x in loc[key]))
        self.assertEqual(loc['logical_columns'],[9,10,11])


class ContextTests(unittest.TestCase):
    def test_preserves_all_false_flags_and_input(self):
        raw,c=fixture();before=deepcopy(c);out=r.extract_document(raw,c)
        self.assertEqual(c,before);self.assertEqual(out['original_context'],before)
        for flag in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','currency_iso_verified','availability_calendar_certified'):self.assertIs(out[flag],False)
        self.assertTrue(all(v['fact_admitted'] is False for v in out['fields'].values()))
    def test_same_source_and_current_availability(self):
        raw,c=fixture();out=r.extract_document(raw,c)
        for v in out['fields'].values():
            self.assertEqual(v['source'],out['source']);self.assertEqual(v['available_from'],c['available_session'])
        self.assertEqual(out['fields'][r.FIELDS[2]]['as_of'],'2019-06-30')
    def test_annual_period_leap_year(self):
        raw,c=fixture(year=2020);out=r.extract_document(raw,c)
        self.assertEqual(out['fields'][r.FIELDS[0]]['period'],dict(period_start='2019-07-01',period_end='2020-06-30',days_inclusive=366))
    def test_unknown_prior_start_not_fabricated(self):
        raw,c=fixture();out=r.extract_document(raw,c)
        self.assertIsNone(out['periods']['comparison']['period_start'])
    def test_2019_accounting_hold_retained(self):
        raw,c=fixture(year=2019);c['context_summary']['accounting_attention']=['ASC606 original note']
        out=r.extract_document(raw,c);self.assertIn(r.ASC_HOLD,out['context_holds']);self.assertEqual(out['accounting_review_flags'],['ASC606 original note'])
    def test_2011_returns_three_explicit_gaps(self):
        raw,c=fixture(year=2011);c['context_summary']=dict(status='preformatted_original_requires_separate_column_geometry_review',fact_admitted=False)
        out=r.extract_document(raw,c);self.assertEqual(out['fields'],{k:None for k in r.FIELDS});self.assertEqual(len(out['gaps']),3);self.assertIn(r.PRE_HOLD,out['context_holds'])
    def test_source_hash_mismatch_fatal(self):
        raw,c=fixture();c['evidence_sha256']='f'*64
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_out_of_scope_year_fatal(self):
        raw,c=fixture(year=2027)
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_issuer_or_ticker_mismatch_fatal(self):
        for before,after in (('KLA-Tencor Corporation','Other Corporation'),('NASDAQ: KLAC','NASDAQ: OTHER')):
            raw,c=fixture();raw,c=changed(raw,c,before,after)
            with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_wrong_url_fatal(self):
        raw,c=fixture();c['source_url']='https://example.com/forged'
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_not_full_source_fatal(self):
        raw,c=fixture();c['catalog_claims_issuer_full_body']=False
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_same_day_availability_fatal(self):
        raw,c=fixture();c['available_session']=c['publication_date']
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_wrong_fiscal_end_fatal(self):
        raw,c=fixture();c['official_actual_period_end']='2020-07-01'
        with self.assertRaises(ValueError):r.extract_document(raw,c)
    def test_pure_no_file_network_or_strategy_import(self):
        tree=ast.parse(Path(r.__file__).read_text());names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module)
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,('open','eval','exec'))
        self.assertFalse(any('runner' in n or 'factor' in n or 'request' in n or 'pandas' in n or 'numpy' in n for n in names))


class AdditionalGuards(unittest.TestCase):
    def test_declared_year_range_uses_generated_values_only(self):
        for year in range(2012,2027):
            raw,c=fixture('wide' if year<2022 else 'compact',year=year,ni=str(year-1900))
            out=r.extract_document(raw,c)
            self.assertEqual(out['fields']['annual_net_income']['value_decimal'],str(year-1900))
            self.assertEqual(out['fields']['current_total_assets']['as_of'],str(year)+'-06-30')
    def test_split_parentheses_owned_by_same_header(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser')
        cells=soup.find_all('table')[2].find_all('tr')[4].find_all('td')
        cells[9].string='$ (';cells[10].string='10';cells[11].string=')'
        raw=str(soup).encode();c['evidence_sha256']=r.digest(raw)
        self.assertEqual(fields(raw,c)['annual_net_income']['value_decimal'],'-10')
    def test_total_label_rowspan_rejected(self):
        raw,c=fixture();soup=BeautifulSoup(raw,'html.parser')
        cells=soup.find_all('table')[2].find_all('tr')[4].find_all('td');cells[0]['rowspan']='2'
        raw=str(soup).encode();c['evidence_sha256']=r.digest(raw)
        self.assertIsNone(fields(raw,c)['annual_net_income'])
    def test_pre_gap_does_not_invent_period_start(self):
        raw,c=fixture(year=2011);c['context_summary']=dict(status='preformatted_original_requires_separate_column_geometry_review',fact_admitted=False)
        raw=b'<html>KLA-Tencor Corporation (NASDAQ: KLAC)<pre>Net income 7 8\nNet income per share 1 2</pre></html>'
        c['evidence_sha256']=r.digest(raw)
        out=r.extract_document(raw,c)
        self.assertEqual(out['fields'],{k:None for k in r.FIELDS})
        self.assertIsNone(out['periods']['current']['period_start'])
    def test_period_interpretation_not_quoted_start(self):
        raw,c=fixture();out=r.extract_document(raw,c)
        self.assertEqual(out['current_period_basis'],c['context_summary']['current_annual_period']['basis'])


if __name__=='__main__':
    unittest.main(verbosity=2)
