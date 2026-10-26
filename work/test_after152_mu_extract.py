"""Synthetic-only MU parser tests. No real source, price or accounting files."""
import copy
import datetime as dt
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
import after152_mu_extract as m


def td(text='', span=1, rowspan=1):
    return f'<td colspan="{span}" rowspan="{rowspan}">{text}</td>'


def tr(*cells):
    return '<tr>'+''.join(cells)+'</tr>'


def value_cells(value, dollar=True):
    if value.startswith('(') and value.endswith(')'):
        return td('$ (' if dollar else '(')+td(value[1:-1])+td(')')
    return td('$' if dollar else '')+td(value)+td()


def fixture(module=m, year=None, reverse=False, narrow=False, ni='(1,234.50)', assets='8,000', prior='7,000',
            unit='millions', dollar=True, slash=False, caption=True, short_week=False):
    profile=module.PROFILE; symbol=profile['symbol']; year=year or profile['years'][-1]
    if symbol=='MU':
        current=dt.date(year,8,28);previous=dt.date(year-1,8,29);middle=dt.date(year,5,29)
        pub=dt.date(year,9,23);qcount=3
    else:
        current=dt.date(year,12,31 if not short_week else 24);previous=dt.date(year-1,12,26)
        middle=None;pub=dt.date(year+1,2,15);qcount=2
    previous=current-dt.timedelta(days=364 if symbol=="MU" or short_week else 371)
    seq=[current,previous] if not reverse else [previous,current]
    qseq=([current,middle,previous] if symbol=='MU' else seq)
    bsseq=([current,middle,previous] if symbol=='MU' else seq)
    if reverse and symbol=='MU':bsseq=list(reversed(bsseq))
    gap=1
    def label(day):
        return day.strftime('%m/%d/%y') if slash else day.strftime('%B %d, %Y').replace(' 0',' ')
    def heads(days):
        result=[]
        for day in days:
            result += ([td(),td(label(day)),td()] if narrow else [td(label(day),3)])+[td()]
        return result
    width=1+4*(qcount+2)
    title='CONSOLIDATED STATEMENTS OF OPERATIONS' if symbol=='MU' else 'Condensed Consolidated Statement of Income'
    caption_text=profile['issuer']+' '+title+' (In '+unit+', except per share amounts'
    if caption:caption_text+='; unaudited except year-ended '+previous.strftime('%m/%d/%Y')+' amounts'
    caption_text+=')'
    income=[tr(td(caption_text,width)),
        tr(td(),td('Three Months Ended' if symbol=='MU' else 'Quarter Ended',4*qcount),td('Year Ended',8)),
        tr(td(),*heads(qseq+seq)),tr(td('Revenue'),td('10',width-1)),
        tr(td('Net income (loss)' if symbol=='MU' else 'Net income'),
           *sum(([value_cells('999',dollar),td()] for _ in qseq),[]),
           *sum(([value_cells(ni if day==current else '111',dollar),td()] for day in seq),[])),
        tr(td('Net income attributable to common shareholders'),td('555',width-1))]
    bwidth=1+4*len(bsseq)
    balance=[tr(td(profile['issuer']+' Condensed Consolidated Balance Sheet (In '+unit+')',bwidth)),
        tr(td(),*heads(bsseq)),tr(td('Assets'),td('',bwidth-1)),
        tr(td('Total assets'),*sum(([value_cells(assets if day==current else prior if day==previous else '6000',dollar),td()] for day in bsseq),[]))]
    exchange='Nasdaq' if symbol=='MU' else 'NYSE'
    raw=('<html><p>'+profile['issuer']+' ('+exchange+': '+symbol+')</p><table>'+''.join(income)+
         '</table><table>'+''.join(balance)+'</table></html>').encode()
    source=dict(path='synthetic/'+symbol+'.html',sha256=m.sha256(raw),url='https://'+profile['hosts'][0]+'/synthetic',
                receipt_path='synthetic/'+symbol+'.meta.json',receipt_sha256='a'*64)
    c=dict(symbol=symbol,event_id=symbol+'_'+str(pub),fiscal_year=year,format='html_or_text',source_kind='HTML_original',
           original_report_bytes_verified=True,context_status=m.STATUS,source=source,evidence_path=source['path'],
           evidence_sha256=source['sha256'],source_url=source['url'],receipt=dict(path=source['receipt_path'],sha256='a'*64),
           identity_anchors=[profile['issuer'],exchange+': '+symbol],publication_date=str(pub),available_session=str(pub+dt.timedelta(days=1)),
           official_actual_period_end=str(current),context_holds=['future_accounting_review','early_clock_hold','early_clock_hold'],
           accounting_terms=['synthetic reclassification'],accounting_context_numbers_redacted=['reclassified [number]'],
           periods=dict(current_end=str(current),prior_end=str(previous),proposed_current_start=str(previous+dt.timedelta(days=1)),
                        annual_date_binding='same_report_full_date_headers',start_assumption='same_current_report_prior_annual_end_plus_one_calendar_day',
                        start_is_source_quoted=False,start_assumption_approved=False))
    fields={}
    for name,ti,row,index,day,base in (
            ('annual_net_income',0,4,seq.index(current),current,1+4*qcount),
            ('current_total_assets',1,3,bsseq.index(current),current,1),
            ('prior_total_assets',1,3,bsseq.index(previous),previous,1)):
        start=base+4*index
        fields[name]=dict(table=ti,row=row,label_cell=0,label=('Net income (loss)' if symbol=='MU' else 'Net income') if ti==0 else 'Total assets',
            logical_columns=[start+1] if narrow else list(range(start,start+3)),as_of=str(day),
            scale='1000000' if unit=='millions' else '1000',currency_literal='$',currency_iso=None,
            available_from=c['available_session'],source=copy.deepcopy(source),external_metadata=[])
    c['fields']=fields
    return raw,refresh(raw,c)


def refresh(raw,c,geometry=True):
    c=copy.deepcopy(c);c['source']['sha256']=c['evidence_sha256']=m.sha256(raw)
    tables=BeautifulSoup(raw,'html.parser').find_all('table')
    for spec in c['fields'].values():
        if spec is None:continue
        spec['source']=copy.deepcopy(c['source'])
        if geometry:
            _,_,physical=m.table_grid(tables[spec['table']]);stop=3 if spec['table']==0 else 2
            spec['header_evidence']=[dict({k:cell[k] for k in m.GEOMETRY},text=cell['text']) for cell in physical.values() if cell['row']<stop]
            spec['row_geometry_without_values']=[{k:cell[k] for k in m.GEOMETRY} for cell in physical.values() if cell['row']==spec['row']]
    return c


def mutate(raw,c,before,after,geometry=True):
    raw=raw.replace(before.encode(),after.encode())
    return raw,refresh(raw,c,geometry)


class ExtractionChecks:
    MODULE=m
    def make(self,**kw):return fixture(self.MODULE,**kw)
    def out(self,**kw):return self.MODULE.extract_document(*self.make(**kw))
    def extract(self,pair):return self.MODULE.extract_document(*pair)
    def assert_gap(self,out,field='annual_net_income'):
        self.assertIsNone(out['fields'][field]);self.assertTrue(any(g['field']==field for g in out['gaps']))

    def test_three_exact_values_annual_not_quarter_or_parent(self):
        out=self.out();self.assertTrue(out['complete_required_fields'],out['gaps'])
        self.assertEqual({k:v['value_decimal'] for k,v in out['fields'].items()},dict(annual_net_income='-1234.50',current_total_assets='8000',prior_total_assets='7000'))

    def test_all_supported_synthetic_years(self):
        for year in self.MODULE.YEARS:
            out=self.out(year=year);self.assertTrue(out['complete_required_fields'],(year,out['gaps']))

    def test_current_prior_order_not_assumed(self):
        self.assertEqual(self.out(reverse=True)['fields']['annual_net_income']['value_decimal'],'-1234.50')

    def test_full_dates_and_slash_headers_supported(self):
        for slash in (False,True):self.assertTrue(self.out(slash=slash)['complete_required_fields'])

    def test_caption_audit_date_is_not_column_date(self):
        self.assertTrue(self.out(caption=True,slash=True)['complete_required_fields'])

    def test_narrow_header_owns_split_symbols(self):
        out=self.out(narrow=True);self.assertTrue(out['complete_required_fields'],out['gaps'])
        self.assertEqual(len(out['fields']['annual_net_income']['locator']['physical_cells']),3)

    def test_negative_zero_and_unicode_minus(self):
        for value,expected in [('0','0'),('(0)','-0'),('-15','-15'),('−15','-15')]:
            self.assertEqual(self.out(ni=value)['fields']['annual_net_income']['value_decimal'],expected)

    def test_exact_large_decimal_no_rounding(self):
        n='123456789012345678901234567890123456789.987654321'
        self.assertEqual(self.out(ni='('+n+')')['fields']['annual_net_income']['value_decimal'],'-'+n)

    def test_thousands_whitespace_and_original_markup(self):
        out=self.extract(mutate(*self.make(ni='1,234'),'1,234','1&nbsp;,&nbsp;234'))
        field=out['fields']['annual_net_income'];self.assertEqual(field['value_decimal'],'1234')
        self.assertTrue(any('\xa0' in c['original_text'] for c in field['locator']['physical_cells']))
        self.assertTrue(all(len(c['markup_sha256'])==64 for c in field['locator']['physical_cells']))

    def test_missing_or_malformed_never_zero(self):
        for value in ('','-','—','N/A','123 456','(-123)','1,23','$ 2 $','1.2.3'):
            self.assert_gap(self.out(ni=value))

    def test_no_io_no_mutation_and_preserved_holds(self):
        raw,c=self.make();before=copy.deepcopy(c)
        with patch('builtins.open',side_effect=AssertionError('No I/O')):out=self.MODULE.extract_document(raw,c)
        self.assertEqual(c,before);self.assertEqual(out['source'],c['source']);self.assertEqual(out['context_holds'],c['context_holds'])
        self.assertEqual(out['accounting_context_numbers_redacted'],c['accounting_context_numbers_redacted'])
        for key in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','first_public_time_verified','certified_daily_security_master'):
            self.assertIs(out[key],False)
        for f in out['fields'].values():
            self.assertFalse(f['fact_admitted']);self.assertIsNone(f['currency_iso']);self.assertEqual(f['source'],c['source'])

    def test_period_asof_and_current_availability(self):
        raw,c=self.make();out=self.MODULE.extract_document(raw,c);p=out['periods']
        start=dt.date.fromisoformat(c['periods']['prior_end'])+dt.timedelta(days=1)
        self.assertEqual(p['current']['period_start'],str(start));self.assertFalse(p['start_assumption_certified'])
        self.assertEqual(out['fields']['annual_net_income']['period'],p['current'])
        self.assertEqual(out['fields']['annual_net_income']['as_of'],c['periods']['current_end'])
        self.assertIsNone(out['fields']['prior_total_assets']['period'])
        self.assertEqual(out['fields']['prior_total_assets']['available_from'],c['available_session'])

    def test_raw_digest_identity_and_scope_fail_hard(self):
        raw,c=self.make()
        with self.assertRaises(m.SourceError):self.MODULE.extract_document(raw+b' ',c)
        for key,value in [('symbol','OTHER'),('fiscal_year',True),('fiscal_year',2000),('event_id','OTHER'),('context_status','index_only'),('source_kind','index'),('original_report_bytes_verified',False)]:
            x=copy.deepcopy(c);x[key]=value
            with self.subTest(key=key),self.assertRaises(m.SourceError):self.MODULE.extract_document(raw,x)
        x=copy.deepcopy(c);x['identity_anchors']=['Not in source']
        with self.assertRaises(m.SourceError):self.MODULE.extract_document(raw,x)

    def test_bad_reference_receipt_and_host_fail_hard(self):
        raw,c=self.make()
        for change in (lambda x:x.update(evidence_path='other'),lambda x:x['receipt'].update(sha256='b'*64),
                       lambda x:x['source'].update(url='https://evil.example/x'),lambda x:x['source'].update(path='../x')):
            x=copy.deepcopy(c);change(x)
            with self.assertRaises(m.SourceError):self.MODULE.extract_document(raw,x)

    def test_period_unknown_or_upgraded_retains_all_three_null(self):
        for key,value in [('prior_end',None),('current_end','2025-01-01'),('proposed_current_start','2001-01-01'),
                           ('start_is_source_quoted',True),('start_assumption_approved',True),('annual_date_binding','year_only')]:
            raw,c=self.make();c['periods'][key]=value;out=self.MODULE.extract_document(raw,c)
            self.assertEqual(len(out['gaps']),3);self.assertEqual(out['context_holds'],c['context_holds'])

    def test_non_week_annual_boundaries_rejected(self):
        for days in (330, 350, 365, 366, 370, 372):
            raw,c=self.make()
            end=dt.date.fromisoformat(c["periods"]["current_end"])
            previous=end-dt.timedelta(days=days)
            c["periods"]["prior_end"]=str(previous)
            c["periods"]["proposed_current_start"]=str(previous+dt.timedelta(days=1))
            out=self.MODULE.extract_document(raw,c)
            self.assertEqual(len(out["gaps"]),3,days)
            self.assertTrue(all(g["reason"] in ("derived_start_mismatch","context_year_end_mismatch") for g in out["gaps"]),out["gaps"])

    def test_publication_and_availability_order(self):
        raw,c=self.make();c['available_session']=c['publication_date']
        self.assertEqual(len(self.MODULE.extract_document(raw,c)['gaps']),3)
        raw,c=self.make();c['publication_date']=c['periods']['current_end'];c['event_id']=c['symbol']+'_'+c['publication_date']
        self.assertEqual(len(self.MODULE.extract_document(raw,c)['gaps']),3)

    def test_field_source_units_clock_and_endpoint_mismatch(self):
        for key,value in [('source',{}),('as_of','2000-01-01'),('available_from','2000-01-01'),('currency_literal','EUR'),('currency_iso','USD'),('scale','1000')]:
            raw,c=self.make();c['fields']['annual_net_income'][key]=value
            self.assert_gap(self.MODULE.extract_document(raw,c))

    def test_quarter_same_current_date_cannot_replace_annual(self):
        raw,c=self.make();c['fields']['annual_net_income']['logical_columns']=[1,2,3]
        self.assert_gap(self.MODULE.extract_document(raw,c))

    def test_wrong_year_endpoint_and_duplicate_annual_date(self):
        raw,c=self.make();spec=c['fields']['annual_net_income'];spec['logical_columns']=[x+4 for x in spec['logical_columns']]
        self.assert_gap(self.MODULE.extract_document(raw,c))
        raw,c=self.make(slash=True);prev=dt.date.fromisoformat(c['periods']['prior_end']).strftime('%m/%d/%y')
        cur=dt.date.fromisoformat(c['periods']['current_end']).strftime('%m/%d/%y')
        self.assert_gap(self.extract(mutate(raw,c,prev,cur)))

    def test_parent_only_and_duplicate_net_income_labels_rejected(self):
        raw,c=self.make();label=c['fields']['annual_net_income']['label']
        self.assert_gap(self.extract(mutate(raw,c,label,'Net income attributable to parent')))
        raw,c=self.make();self.assert_gap(self.extract(mutate(raw,c,'Net income attributable to common shareholders',c['fields']['annual_net_income']['label'])))

    def test_non_gaap_title_and_unrelated_table_rejected(self):
        for title in ('Non-GAAP Consolidated Statements of Income','Consolidated Statements of Cash Flows'):
            raw,c=self.make();soup=BeautifulSoup(raw,'html.parser');soup.find('table').find('td').string=title+' (In millions)'
            changed=str(soup).encode();self.assert_gap(self.MODULE.extract_document(changed,refresh(changed,c)))

    def test_duplicate_title_and_external_context_rejected(self):
        raw,c=self.make();soup=BeautifulSoup(raw,'html.parser');cell=soup.find('table').find_all('tr')[1].find('td');cell.string='Consolidated Statements of Income'
        raw=str(soup).encode();self.assert_gap(self.MODULE.extract_document(raw,refresh(raw,c)))
        raw,c=self.make();c['fields']['annual_net_income']['external_metadata']=['unsupported caption']
        self.assert_gap(self.MODULE.extract_document(raw,c))

    def test_missing_or_mixed_scale_currency(self):
        self.assert_gap(self.out(dollar=False))
        for unit in ('unspecified','millions) (in thousands'):
            self.assert_gap(self.out(unit=unit))
        raw,c=self.make();self.assert_gap(self.extract(mutate(raw,c,'In millions','In millions EUR')))
        self.assertEqual(self.out(unit='thousands')['fields']['annual_net_income']['scale'],'1000')

    def test_frozen_header_and_row_geometry_tamper(self):
        raw,c=self.make();c['fields']['annual_net_income']['header_evidence'][0]['text']='Other'
        self.assert_gap(self.MODULE.extract_document(raw,c))
        raw,c=self.make();c['fields']['annual_net_income']['row_geometry_without_values'][1]['columns']=[999]
        self.assert_gap(self.MODULE.extract_document(raw,c))

    def test_cross_row_amount_rejected_even_if_geometry_refreshed(self):
        raw,c=self.make();soup=BeautifulSoup(raw,'html.parser');row=soup.find_all('table')[0].find_all('tr')[4]
        q=self.MODULE.PROFILE['income_groups']-2;row.find_all('td',recursive=False)[2+4*q]['rowspan']='2'
        soup.find_all('table')[0].find_all('tr')[5].clear()
        raw=str(soup).encode();self.assert_gap(self.MODULE.extract_document(raw,refresh(raw,c)))

    def test_cross_year_amount_colspan_rejected(self):
        raw,c=self.make();soup=BeautifulSoup(raw,'html.parser');row=soup.find_all('table')[0].find_all('tr')[4]
        q=self.MODULE.PROFILE['income_groups']-2;row.find_all('td',recursive=False)[2+4*q]['colspan']='5'
        raw=str(soup).encode();self.assert_gap(self.MODULE.extract_document(raw,refresh(raw,c)))

    def test_neighbor_period_parenthesis_not_borrowed(self):
        raw,c=self.make(narrow=True,ni='1234');soup=BeautifulSoup(raw,'html.parser');row=soup.find_all('table')[0].find_all('tr')[4]
        q=self.MODULE.PROFILE['income_groups']-2;cells=row.find_all('td',recursive=False)
        cells[1+4*q].string='';cells[4*q].string='('
        raw=str(soup).encode();out=self.MODULE.extract_document(raw,refresh(raw,c))
        # The neighboring punctuation must never change selected amount sign.
        f=out['fields']['annual_net_income'];self.assertTrue(f is None or f['value_decimal']=='1234')

    def test_extra_numeric_cell_in_owned_group_rejected(self):
        raw,c=self.make();soup=BeautifulSoup(raw,'html.parser');row=soup.find_all('table')[0].find_all('tr')[4]
        q=self.MODULE.PROFILE['income_groups']-2;row.find_all('td',recursive=False)[4+4*q].string='777'
        raw=str(soup).encode();self.assert_gap(self.MODULE.extract_document(raw,refresh(raw,c)))

    def test_invalid_date_header_rejected(self):
        raw,c=self.make(slash=True);day=dt.date.fromisoformat(c['periods']['current_end']).strftime('%m/%d/%y')
        self.assert_gap(self.extract(mutate(raw,c,day,'99/99/25')))

    def test_absent_field_context_is_explicit_null(self):
        raw,c=self.make();c['fields']['annual_net_income']=None
        out=self.MODULE.extract_document(raw,c);self.assert_gap(out)
        self.assertIsNotNone(out['fields']['current_total_assets'])


class MUTests(ExtractionChecks,unittest.TestCase):
    MODULE=m
    def test_intermediate_balance_period_cannot_replace_endpoint(self):
        raw,c=self.make();c['fields']['current_total_assets']['logical_columns']=[5,6,7]
        self.assert_gap(self.MODULE.extract_document(raw,c),'current_total_assets')


if __name__=='__main__':unittest.main(verbosity=2)
