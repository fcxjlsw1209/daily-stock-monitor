"""Synthetic-only independent review tests; no original reports or producers."""
import ast
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from bs4 import BeautifulSoup
import after152_batch4_independent_review as m

def td(s,n=1):return '<td'+(' colspan="%s"'%n if n!=1 else '')+'>'+s+'</td>'
def tr(xs):return '<tr>'+''.join(xs)+'</tr>'
def label(d):return date.fromisoformat(d).strftime('%B %d, %Y')
class Reader:
    def __init__(self,files):self.files=files;self.hashes={}
    def read(self,p,h=None):
        p=m.relative(p);b=self.files[p];m.require(h is None or m.sha(b)==h,'SHA mismatch');self.hashes[p]=m.sha(b);return b
    def load(self,p,h=None):return json.loads(self.read(p,h))
def rebuild(c,raw):
    soup=BeautifulSoup(raw,'html.parser')
    for st in c['statement_contexts']:
        cells,_=m.grid_from_html(soup.find_all('table')[st['table']]);st['headers']=[{k:x[k] for k in ('row','physical_cell','columns','rowspan','text')} for x in cells.values() if x['row']<st['header_stop_row']]
    for s in c['fields'].values():
        cells,_=m.grid_from_html(soup.find_all('table')[s['table']]);s['header_evidence']=deepcopy(c['statement_contexts'][s['table']]['headers']);s['row_geometry_without_values']=[{k:x[k] for k in ('row','physical_cell','columns','rowspan')} for x in cells.values() if x['row']==s['row']]
    return soup

def fixture(symbol='MU',ni='(31)',days=364,year=2024):
    end=date(year,8,29) if symbol=='MU' else date(year,12,27);prior=end-timedelta(days=days);q=end-timedelta(days=91);pub=end+timedelta(days=27);available=pub+timedelta(days=1)
    current,prev,mid,release,available=map(str,(end,prior,q,pub,available));fmt=(lambda s:date.fromisoformat(s).strftime('%m/%d/%Y')) if symbol=='PEP' else label
    issuer='Micron Technology, Inc.' if symbol=='MU' else 'PepsiCo, Inc.'
    prefix=issuer+(' and Subsidiaries' if symbol=='PEP' else '')
    title=prefix+' Consolidated Statement of '+('Income' if symbol=='PEP' else 'Operations')+' (in millions; unaudited except year-ended '+fmt(prev)+' amounts)'
    income='<table>'+tr([td(title,7)])+tr([td(''),td('Quarter Ended',2),td('Year Ended',4)])+tr([td(''),td(fmt(current)),td(fmt(prev)),td(fmt(current),2),td(fmt(prev),2)])+tr([td('Revenue'),td('1'),td('2'),td('$'),td('3'),td('$'),td('4')])+tr([td('Net income'),td('100'),td('101'),td('$'),td(ni),td('$'),td('20')])+tr([td('Net income attributable to parent'),td('1'),td('1'),td(''),td('2'),td(''),td('3')])+'</table>'
    balance='<table>'+tr([td(prefix+' Consolidated Balance Sheet (in millions)',7)])+tr([td(''),td(fmt(current),2),td(fmt(mid),2),td(fmt(prev),2)])+tr([td('Current assets'),td('$'),td('5'),td('$'),td('6'),td('$'),td('7')])+tr([td('Total assets'),td('$'),td('1,234'),td('$'),td('888'),td('$'),td('987')])+'</table>'
    # Put the dollar inside parentheses for a signed-dollar value.
    income=income.replace('<td>$</td><td>(31)</td>','<td></td><td>($31)</td>')
    raw=('<html><body>'+issuer+' (NASDAQ: '+symbol+') '+label(release)+income+balance+'</body></html>').encode()
    source=dict(path='work/synthetic.html',sha256=m.sha(raw),url=('https://micron.gcs-web.com/' if symbol=='MU' else 'https://www.prnewswire.com/')+'synthetic',receipt_path='work/synthetic.receipt.json')
    receipt=dict(sha256=m.sha(raw),url=source['url'],http_status=200,bytes=len(raw),retrieved_at='2026-10-20T00:00:00+00:00');rr=json.dumps(receipt).encode();source['receipt_sha256']=m.sha(rr)
    c=dict(event_id=symbol+'_'+release,symbol=symbol,fiscal_year=year,publication_date=release,available_session=available,available_session_verified=True,official_actual_period_end=current,source=source,source_url=source['url'],evidence_path=source['path'],evidence_sha256=source['sha256'],receipt=dict(path=source['receipt_path'],sha256=source['receipt_sha256'],retrieved_at=receipt['retrieved_at']),identity_anchors=[issuer,'NASDAQ: '+symbol],source_kind='HTML_original',original_report_bytes_verified=True,context_status=m.STATUS,context_holds=['accounting_not_admitted'],accounting_terms=['reclassification'],accounting_context_numbers_redacted=['Synthetic [number] note'],periods=dict(current_end=current,prior_end=prev,proposed_current_start=str(prior+timedelta(days=1)),start_is_source_quoted=False,start_assumption_approved=False,start_assumption='same_current_report_prior_annual_end_plus_one_calendar_day'),statement_contexts=[dict(table=0,kind='income',header_stop_row=3,scale='1000000'),dict(table=1,kind='balance',header_stop_row=2,scale='1000000')],fields={})
    for f,ti,ri,cols,day in [(m.FIELDS[0],0,4,[3,4],current),(m.FIELDS[1],1,3,[1,2],current),(m.FIELDS[2],1,3,[5,6],prev)]:c['fields'][f]=dict(table=ti,row=ri,label_cell=0,label='Net income' if ti==0 else 'Total assets',logical_columns=cols,as_of=day,scale='1000000',currency_literal='$',currency_iso=None)
    soup=rebuild(c,raw);period=dict(period_start=str(prior+timedelta(days=1)),period_end=current,days_inclusive=days)
    e=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,publication_date=release,available_from=available,source=deepcopy(source),identity=dict(issuer=issuer,symbol=symbol,source_anchors=deepcopy(c['identity_anchors']),certified_daily_security_master=False),fields={},periods=dict(current=period,comparison=dict(period_end=prev),assumption_id=symbol+'_consecutive_annual_boundaries_v1',start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False),context_holds=deepcopy(c['context_holds']),accounting_review_flags=deepcopy(c['accounting_terms']),accounting_context_numbers_redacted=deepcopy(c['accounting_context_numbers_redacted']),gaps=[],fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_time_verified=False,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
    for f,v in zip(m.FIELDS,[ni,'1234','987']):
        s=c['fields'][f];cells,_=m.grid_from_html(soup.find_all('table')[s['table']]);chosen=[x for x in cells.values() if x['row']==s['row'] and set(x['columns'])&set(s['logical_columns'])];display=' '.join(x['text'] for x in chosen if x['text']);value='-31' if ni=='(31)' and f==m.FIELDS[0] else v
        e['fields'][f]=dict(value_decimal=value,displayed_amount=display,scale='1000000',currency_literal='$',currency_iso=None,source=deepcopy(source),available_from=available,as_of=s['as_of'],period=deepcopy(period) if f==m.FIELDS[0] else None,fact_admitted=False,locator=dict(table=s['table'],row=s['row'],header_columns=s['logical_columns'],physical_cells=chosen))
    policy=dict(periods=dict(boundary_assumptions={s:dict(enabled=True,not_a_quoted_fact=True,assumption_id=s+'_consecutive_annual_boundaries_v1',days_inclusive=[364,371]) for s in ('MU','PEP')}))
    row=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,source_path=source['path'],source_sha256=source['sha256'])
    return raw,c,e,row,Reader({source['path']:raw,source['receipt_path']:rr}),[release,available,'2028-01-01'],policy

class SourceTests(unittest.TestCase):
    def read(self,parts=None,field=m.FIELDS[0]):
        raw,c,*_=parts or fixture();return m.source_field(BeautifulSoup(raw,'html.parser'),c,field)
    def mutate(self,old,new,symbol='MU',headers=False):
        p=list(fixture(symbol));p[0]=p[0].replace(old,new)
        if headers:rebuild(p[1],p[0])
        return p
    def test_mu(self):self.assertEqual([self.read(fixture(),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_pep(self):self.assertEqual([self.read(fixture('PEP'),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_quarter_punctuation_cannot_leak_into_annual(self):
        self.assertEqual(self.read(self.mutate(b"<td>101</td>",b"<td>)</td>"))["value_decimal"],"-31")
    def test_caption_date_excluded(self):self.assertEqual(self.read(fixture('PEP'))['columns'],[3,4])
    def test_quarter_assets_excluded(self):self.assertEqual(self.read(field=m.FIELDS[2])['value_decimal'],'987')
    def test_parent_total_is_not_selected(self):self.assertEqual(self.read()['value_decimal'],'-31')
    def test_parent_only_rejected(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>Net income</td>',b'<td>Net income attributable to parent</td>'))
    def test_duplicate_total_rejected(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Net income attributable to parent',b'Net income'))
    def test_negative_zero_positive(self):
        for value in ['-31','0','31','123456789012345.123456789']:
            with localcontext() as c:c.prec=3;self.assertEqual(self.read(fixture(ni=value))['value_decimal'],value)
    def test_invalid_syntax(self):
        for value in [b'NaN',b'1 2',b'1,23',b'(-31)',b'(31',b'31)',b'31%',b'31*',b'--31',b'$ $31']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'($31)',value))
    def test_bad_scale(self):
        for value in [b'in thousands',b'in billions',b'without scale',b'in millions and in thousands']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'in millions',value,headers=True))
    def test_currency(self):
        for old,new in [(b'$',b''),(b'$',b'EUR '),(b'($31)',b'(GBP31)')]:
            with self.assertRaises(ValueError):self.read(self.mutate(old,new))
    def test_bad_title(self):
        for value in [b'Non-GAAP Consolidated',b'Pro forma Consolidated',b'Reconciliation Consolidated']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated',value,headers=True))
    def test_non_gaap_elsewhere_allowed(self):self.assertEqual(self.read(self.mutate(b'</body>',b'<p>Non-GAAP supplemental reconciliation</p></body>'))['value_decimal'],'-31')
    def test_wrong_period(self):
        for value in [b'Three Months Ended',b'Six Months Ended',b'Nine Months Ended',b'Quarter Ended']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'Year Ended',value,headers=True))
    def test_wrong_context_headers(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Year Ended',b'Quarter Ended'))
    def test_cross_row(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>($31)</td>',b'<td rowspan="2">($31)</td>',headers=True))
    def test_cross_date(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>($31)</td>',b'<td colspan="3">($31)</td>',headers=True))
    def test_assets_positive(self):
        for value in [b'0',b'(1234)']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'1,234',value),m.FIELDS[1])
    def test_header_columns_tamper(self):
        p=fixture();p[1]['fields'][m.FIELDS[0]]['logical_columns']=[1]
        with self.assertRaises(ValueError):self.read(p)
    def test_date_parse_two_digits(self):self.assertEqual(m.dates('12/25/10'),{'2010-12-25'})
    def test_invalid_date_rejected(self):
        with self.assertRaises(ValueError):m.date_groups([dict(text='02/31/2024',columns=[1],rowspan=1)], [dict(columns=[0,1])],False)
    def test_duplicate_annual_date(self):
        h=[dict(text='Year Ended',columns=[1,2]),dict(text='12/31/2024',columns=[1],rowspan=1),dict(text='12/31/2024',columns=[2],rowspan=1)]
        with self.assertRaises(ValueError):m.date_groups(h,[dict(columns=[0,1,2])],True)

class EventTests(unittest.TestCase):
    def review(self,p):return m.review_event(p[4],p[3],p[2],p[1],p[5],p[6])
    def test_both_period_lengths(self):
        for s in ('MU','PEP'):
            for days in (364,371):self.assertTrue(self.review(fixture(s,days=days))['boundary_assumption_verified'])
    def test_other_lengths_rejected(self):
        for days in (330,350,365,366,370,372):
            with self.assertRaises(ValueError):self.review(fixture(days=days))
    def test_policy_assumption(self):
        p=fixture();p[6]['periods']['boundary_assumptions']['MU']['enabled']=False
        with self.assertRaises(ValueError):self.review(p)
    def test_source_quotation_upgrade(self):
        p=fixture();p[2]['periods']['start_assumption_certified']=True
        with self.assertRaises(ValueError):self.review(p)
    def test_old_clock_preserved(self):
        p=list(fixture('PEP',year=2011));p[5]=['2015-01-02','2028-01-01'];p[1]['available_session_verified']=False;r=self.review(p);self.assertFalse(r['clock_verified']);self.assertTrue(r['metadata_verified'])
    def test_bad_clock(self):
        p=fixture();p[5].insert(1,str(date.fromisoformat(p[1]['publication_date'])+timedelta(days=2)));p[5].sort()
        p[1]['available_session_verified']=False
        with self.assertRaises(ValueError):self.review(p)
    def test_retained_holds(self):
        p=fixture();p[2]['context_holds']=[]
        with self.assertRaises(ValueError):self.review(p)
    def test_published_amount_tamper(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]['value_decimal']='31'
        with self.assertRaises(ValueError):self.review(p)
    def test_published_sign_tamper(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]['displayed_amount']='31'
        with self.assertRaises(ValueError):self.review(p)
    def test_published_date_tamper(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]['as_of']='2024-08-28'
        with self.assertRaises(ValueError):self.review(p)
    def test_published_markup_tamper(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]['locator']['physical_cells'][0]['markup_sha256']='a'*64
        with self.assertRaises(ValueError):self.review(p)
    def test_source_hash_tamper(self):
        p=fixture();p[3]['source_sha256']='a'*64
        with self.assertRaises(ValueError):self.review(p)
    def test_iso_not_certified(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]['currency_iso']='USD'
        with self.assertRaises(ValueError):self.review(p)
    def test_index_cannot_enter(self):
        p=fixture();p[1]['source_kind']='index_observation_not_original_report'
        with self.assertRaises(ValueError):self.review(p)
    def test_unquoted_identity(self):
        p=fixture();p[2]['identity']['issuer']='Other'
        with self.assertRaises(ValueError):self.review(p)
    def test_null_preserved(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]=None;p[2]['gaps']=[dict(field=m.FIELDS[0],reason='parser_gap')];r=self.review(p);self.assertTrue(r['fields'][m.FIELDS[0]]['null_reason_verified']);self.assertIsNone(p[2]['fields'][m.FIELDS[0]]);self.assertEqual(m.position_counts([r]),(2,1))
    def test_null_cannot_silently_pass(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]=None
        with self.assertRaises(ValueError):self.review(p)

class AccountingTests(unittest.TestCase):
    def data(self):
        raw,c,e,*_=fixture();text=m.norm(BeautifulSoup(raw,'html.parser').get_text(' ',strip=True));a=dict(event_id=e['event_id'],source=e['source'],status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis='Synthetic direct source assessment.',field_checks=dict.fromkeys(m.FIELDS,True),current_prior_common_basis=True,evidence=[dict(path=e['source']['path'],sha256=m.sha(raw),locator=dict(kind='document_text',start=0,end=len(text)),quote=text)]);return a,e,raw
    def test_exact_evidence(self):self.assertEqual(m.verify_accounting(*self.data())['status'],'resolved')
    def test_wrong_quote(self):
        a,e,r=self.data();a['evidence'][0]['quote']='invented'
        with self.assertRaises(ValueError):m.verify_accounting(a,e,r)
    def test_missing_proof(self):
        a,e,r=self.data();a['evidence']=[]
        with self.assertRaises(ValueError):m.verify_accounting(a,e,r)
    def test_all_fields_required(self):
        for f in m.FIELDS:
            a,e,r=self.data();a['field_checks'][f]=False
            with self.assertRaises(ValueError):m.verify_accounting(a,e,r)
    def test_common_basis_required(self):
        a,e,r=self.data();a['current_prior_common_basis']=False
        with self.assertRaises(ValueError):m.verify_accounting(a,e,r)
    def test_unresolved_retained(self):
        a,e,r=self.data();a.update(status='unresolved',reasons=['Unknown comparability'],field_checks=dict.fromkeys(m.FIELDS,False),current_prior_common_basis=False,evidence=[]);self.assertEqual(m.verify_accounting(a,e,r)['status'],'unresolved')

class PopulationTests(unittest.TestCase):
    def data(self):
        docs=[]
        for s,years in [('MU',range(2019,2026)),('PEP',range(2010,2026))]:
            for y in years:
                chosen=(s=='MU' and y>=2024) or (s=='PEP' and y<=2016);original=s=='MU' or y<=2018
                docs.append(dict(event_id=s+'_'+str(y),symbol=s,fiscal_year=y,publication_date=str(y+1)+'-02-01',available_session=str(y+1)+'-02-02',available_session_verified=True,official_actual_period_end=str(y)+'-12-31',source={},source_kind='HTML_original' if original else 'index_observation_not_original_report',original_report_bytes_verified=original,context_status=m.STATUS if chosen else 'unprocessed',context_holds=['not_admitted']))
        chosen=[e for e in docs if e['context_status']==m.STATUS];ids=[e['event_id'] for e in docs];proj=[{k:e[k] for k in m.CATALOG_KEYS} for e in docs]
        return dict(records=chosen,catalog_ids=ids,catalog_events=23,catalog_metadata=deepcopy(proj)),dict(events=[dict(e,fields=dict.fromkeys(m.FIELDS,{})) for e in chosen],catalog_ids=ids,catalog_metadata=deepcopy(proj),counts=dict(events=9,by_symbol={'MU':2,'PEP':7},nonnull_fields=27,unprocessed_catalog_events=14)),dict(events=docs),dict(required_fields=list(m.FIELDS),fixed_catalog_metadata=proj)
    def test_complete(self):self.assertEqual(len(m.verify_population(*self.data())),9)
    def test_catalog_missing(self):
        p,d,c,y=self.data();c['events'].pop()
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_catalog_metadata_changed(self):
        for key in m.CATALOG_KEYS:
            p,d,c,y=self.data();p['catalog_metadata'][0][key]='changed'
            with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_event_order_changed(self):
        p,d,c,y=self.data();d['events'].reverse()
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_counts_changed(self):
        p,d,c,y=self.data();d['counts']['nonnull_fields']=26
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_no_producer_import(self):
        names=[]
        for node in ast.walk(ast.parse(Path(m.__file__).read_text())):
            if isinstance(node,ast.Import):names.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any('extract' in n or 'source_batch' in n or 'admission' in n for n in names))
    def test_hash_before_io(self):
        with patch.object(m,'Reader') as r:
            with self.assertRaises(ValueError):m.audit(None,None)
            r.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
