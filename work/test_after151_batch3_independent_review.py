"""Pure synthetic source tests; no real report or producer imports."""
import ast
from copy import deepcopy
from datetime import date,timedelta
from decimal import localcontext
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from bs4 import BeautifulSoup
import after151_batch3_independent_review as m


def td(text,width=1):return '<td'+(' colspan="%d"'%width if width!=1 else '')+'>'+text+'</td>'
def tr(values):return '<tr>'+''.join(values)+'</tr>'
def label(d):return date.fromisoformat(d).strftime('%B %d, %Y')


class MemoryReader:
    def __init__(self,files):self.files=files;self.hashes={}
    def read(self,path,expected=None):
        path=m.relative(path);raw=self.files[path];digest=m.sha(raw)
        m.require(expected is None or digest==expected,'SHA mismatch');self.hashes[path]=digest;return raw
    def load(self,path,expected=None):return json.loads(self.read(path,expected))


def rebuild_context(c,raw):
    soup=BeautifulSoup(raw,'html.parser')
    for st in c['statement_contexts']:
        cells,_=m.grid_from_html(soup.find_all('table')[st['table']])
        st['headers']=[{k:x[k] for k in ('row','physical_cell','columns','rowspan','text')} for x in cells.values() if x['row']<st['header_stop_row']]
    for spec in c['fields'].values():
        cells,_=m.grid_from_html(soup.find_all('table')[spec['table']])
        spec['header_evidence']=deepcopy(c['statement_contexts'][spec['table']]['headers'])
        spec['row_geometry_without_values']=[{k:x[k] for k in ('row','physical_cell','columns','rowspan')} for x in cells.values() if x['row']==spec['row']]
    return soup


def fixture(symbol='KO',ni='(31)',current=None,prior=None):
    current=current or ('2016-12-31' if symbol=='KO' else '2016-06-26')
    prior=prior or ('2015-12-31' if symbol=='KO' else '2015-06-28')
    start=(date.fromisoformat(prior)+timedelta(days=1)).isoformat();year=int(current[:4]);quarter=(date.fromisoformat(current)-timedelta(days=90)).isoformat()
    released=(date.fromisoformat(current)+timedelta(days=30)).isoformat();available=(date.fromisoformat(released)+timedelta(days=1)).isoformat()
    unit,scale=('millions','1000000') if symbol=='KO' else ('thousands','1000')
    title='Consolidated Statements of Income' if symbol=='KO' else 'Condensed Consolidated Statements of Operations'
    net='Consolidated Net Income' if symbol=='KO' else 'Net income'
    annual='Year Ended' if symbol=='KO' else 'Twelve Months Ended'
    ni_table='<table>'+tr([td(title,9)])+tr([td(''),td('Three Months Ended',3),td(''),td(annual,3),td('')])+tr([td('(In '+unit+', except per share)'),td(label(current)),td(''),td(label(prior)),td(''),td(label(current)),td(''),td(label(prior)),td('')])+tr([td('Revenues'),td('$ 1'),td(''),td('$ 1'),td(''),td('$ 1'),td(''),td('$ 1'),td('')])+tr([td(net),td('9'),td(''),td('8'),td(''),td(ni),td(''),td('7'),td('')])+'</table>'
    bcols=[current,prior] if symbol=='KO' else [current,quarter,prior]
    n=1+2*len(bcols)
    bs='<table>'+tr([td('Consolidated Balance Sheets',n)])+tr([td('(In '+unit+')')]+sum(([td(label(d)),td('')] for d in bcols),[]))+tr([td('Assets')]+sum(([td('$ 1'),td('')] for d in bcols),[]))+tr([td('Total assets')]+sum(([td('1,234' if i==0 else '987' if d==prior else '444'),td('')] for i,d in enumerate(bcols)),[]))+'</table>'
    issuer='The Coca-Cola Company (NYSE: KO)' if symbol=='KO' else 'Lam Research Corporation (NASDAQ: LRCX)'
    raw=('<html><p>'+issuer+' '+label(released)+'</p>'+ni_table+bs+'</html>').encode()
    path='work/synthetic_'+symbol+'.html';rpath=path+'.meta.json';url='https://'+('investors.coca-colacompany.com' if symbol=='KO' else 'newsroom.lamresearch.com')+'/release'
    receipt=dict(sha256=m.sha(raw),url=url,status=200,bytes=len(raw),retrieved_at='2026-10-20T00:00:00+00:00');rr=json.dumps(receipt).encode()
    source=dict(path=path,sha256=m.sha(raw),url=url,receipt_path=rpath,receipt_sha256=m.sha(rr))
    holds=['currency_ISO_and_first_public_version_not_certified','proposed_annual_start_boundary_assumption_not_yet_approved']
    c=dict(event_id=symbol+'_'+released,symbol=symbol,fiscal_year=year,publication_date=released,available_session=available,official_actual_period_end=current,evidence_path=path,evidence_sha256=m.sha(raw),source_url=url,source=source,receipt=dict(path=rpath,sha256=m.sha(rr),retrieved_at=receipt['retrieved_at']),identity_anchors=[issuer],context_status=m.STATUS,context_holds=holds,accounting_terms=[],accounting_context_numbers_redacted=[],periods=dict(current_end=current,prior_end=prior,proposed_current_start=start,start_assumption='same_current_report_prior_annual_end_plus_one_calendar_day',start_assumption_approved=False,start_is_source_quoted=False),fields={},statement_contexts=[dict(table=0,kind='income',header_stop_row=3,headers=[],scale=scale),dict(table=1,kind='balance',header_stop_row=2,headers=[],scale=scale)])
    p=dict(period_start=start,period_end=current,days_inclusive=(date.fromisoformat(current)-date.fromisoformat(start)).days+1)
    e=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,publication_date=released,available_from=available,source=deepcopy(source),fields={},periods=dict(current=p,comparison=dict(period_end=prior),start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False),gaps=[],context_holds=deepcopy(holds),accounting_review_flags=[],accounting_context_numbers_redacted=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)
    values=[('annual_net_income',0,4,5,m.signed_text(ni),ni),('current_total_assets',1,3,1,'1234','1,234'),('prior_total_assets',1,3,n-2,'987','987')]
    soup=BeautifulSoup(raw,'html.parser')
    for f,ti,row,col,value,display in values:
        c['fields'][f]=dict(table=ti,row=row,label_cell=0,label=net if ti==0 else 'Total assets',logical_columns=[col],scale=scale,currency_literal='$',currency_iso=None,as_of=prior if f==m.FIELDS[2] else current,header_evidence=[],row_geometry_without_values=[])
        cells,_=m.grid_from_html(soup.find_all('table')[ti]);cell=cells[row,col]
        e['fields'][f]=dict(value_decimal=value,displayed_amount=display,scale=scale,currency_literal='$',currency_iso=None,source=deepcopy(source),available_from=available,as_of=prior if f==m.FIELDS[2] else current,period=deepcopy(p) if f==m.FIELDS[0] else None,fact_admitted=False,locator=dict(table=ti,row=row,header_columns=[col],physical_cells=[{k:cell[k] for k in ('row','physical_cell','columns','rowspan','text','markup_sha256')}]))
    rebuild_context(c,raw)
    policy=dict(periods=dict(LRCX_52_53_week_assumption=dict(enabled=True,not_a_quoted_fact=True,assumption_id=m.ASSUMPTION)))
    row=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,source_path=path,source_sha256=m.sha(raw))
    return raw,c,e,row,MemoryReader({path:raw,rpath:rr}),[released,available,'2028-01-01'],policy


class SourceTests(unittest.TestCase):
    def read(self,parts=None,f='annual_net_income'):
        raw,c,*_=parts or fixture();return m.source_field(BeautifulSoup(raw,'html.parser'),c,f)
    def mutate(self,old,new,symbol='KO',headers=False):
        f=list(fixture(symbol));f[0]=f[0].replace(old,new)
        if headers:rebuild_context(f[1],f[0])
        return f
    def test_ko_three_fields(self):self.assertEqual([self.read(fixture(),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_lrcx_three_fields(self):self.assertEqual([self.read(fixture('LRCX'),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_lrcx_quarter_asset_not_used(self):self.assertEqual(self.read(fixture('LRCX'),'prior_total_assets')['value_decimal'],'987')
    def test_negative_zero_positive(self):
        for s,v in [('(31)','-31'),('-31','-31'),('0','0'),('31','31')]:self.assertEqual(self.read(fixture(ni=s))['value_decimal'],v)
    def test_high_precision_exact(self):
        with localcontext() as c:
            c.prec=3;self.assertEqual(self.read(fixture(ni='123456789012345.123456789'))['value_decimal'],'123456789012345.123456789')
    def test_bad_numbers(self):
        for value in ['—','N/A','1 2','(-31)','(31','31)','1,23','31%','31*','$ $31','NaN']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'(31)',value.encode()))
    def test_parent_ni_rejected(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated Net Income',b'Net Income Attributable to Shareowners'))
    def test_ko_generic_ni_rejected(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated Net Income',b'Net Income'))
    def test_lrcx_parent_rejected(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Net income',b'Net income attributable to shareholders','LRCX'))
    def test_duplicate_total(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'</table>',b'<tr><td>Consolidated Net Income</td><td>2</td></tr></table>'))
    def test_quarter_six_nine_rejected(self):
        for x in [b'Three',b'Six',b'Nine']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'Year Ended',x+b' Months Ended',headers=True))
    def test_non_gaap_title(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated Statements of Income',b'Non-GAAP Consolidated Statements of Income',headers=True))
    def test_cashflow_is_not_income(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated Statements of Income',b'Consolidated Statements of Cash Flows',headers=True))
    def test_no_title(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Consolidated Statements of Income',b'Summary',headers=True))
    def test_wrong_scale(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'In millions',b'In thousands',headers=True))
    def test_missing_dollar(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'$',b''))
    def test_currency_conflict(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'$ 1',b'EUR 1'))
    def test_frozen_header_mismatch(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'Year Ended',b'Three Months Ended'))
    def test_wrong_columns(self):
        f=fixture();f[1]['fields'][m.FIELDS[0]]['logical_columns']=[1]
        with self.assertRaises(ValueError):self.read(f)
    def test_cross_period_colspan(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>(31)</td>',b'<td colspan="3">(31)</td>',headers=True))
    def test_cross_row_value(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>(31)</td>',b'<td rowspan="2">(31)</td>',headers=True))
    def test_negative_or_zero_assets(self):
        for x in [b'(1234)',b'0']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'1,234',x),'current_total_assets')
    def test_ambiguous_neighbor_sign(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'<td>(31)</td><td></td>',b'<td>31</td><td>)</td>'))
    def test_nested_table(self):
        with self.assertRaises(ValueError):self.read(self.mutate(b'(31)',b'<table><tr><td>(31)</td></tr></table>'))
    def test_space_before_date_comma(self):
        f=self.mutate(b'June 26,',b'June 26 ,','LRCX',headers=True)
        self.assertEqual(self.read(f)['value_decimal'],'-31')
    def test_split_date_header(self):
        f=list(fixture());f[0]=f[0].replace(b'Year Ended',b'Year Ended December 31,');a,b=f[0].split(b'</table>',1);a=a.replace(b'December 31, 2016',b'2016').replace(b'December 31, 2015',b'2015');f[0]=a+b'</table>'+b;rebuild_context(f[1],f[0]);self.assertEqual(self.read(f)['value_decimal'],'-31')
    def test_date_group_tie_not_owned(self):
        h=[dict(columns=[1],text='June 30, 2015'),dict(columns=[3],text='June 29, 2014')]
        _,o=m.date_groups(h,[dict(columns=[0,1,2,3])],False);self.assertIsNone(o[2])


class EventTests(unittest.TestCase):
    def review(self,f):return m.review_event(f[4],f[3],f[2],f[1],f[5],f[6])
    def test_ko_review_leap(self):
        r=self.review(fixture());self.assertTrue(r['passed']);self.assertFalse(r['start_is_source_quoted']);self.assertFalse(r['source_admitted'])
    def test_ko_365(self):self.assertTrue(self.review(fixture(current='2015-12-31',prior='2014-12-31'))['periods_verified'])
    def test_lrcx_364(self):self.assertTrue(self.review(fixture('LRCX'))['boundary_assumption_verified'])
    def test_lrcx_371(self):self.assertTrue(self.review(fixture('LRCX',current='2013-06-30',prior='2012-06-24'))['boundary_assumption_verified'])
    def test_lrcx_bad_duration(self):
        with self.assertRaises(ValueError):self.review(fixture('LRCX',current='2016-06-30'))
    def test_ko_wrong_boundary(self):
        with self.assertRaises(ValueError):self.review(fixture(current='2016-12-30'))
    def test_policy_disabled(self):
        f=fixture('LRCX');f[6]['periods']['LRCX_52_53_week_assumption']['enabled']=False
        with self.assertRaises(ValueError):self.review(f)
    def test_policy_wrong_id(self):
        f=fixture('LRCX');f[6]['periods']['LRCX_52_53_week_assumption']['assumption_id']='other'
        with self.assertRaises(ValueError):self.review(f)
    def test_source_quote_upgrade(self):
        f=fixture();f[2]['periods']['start_assumption_certified']=True
        with self.assertRaises(ValueError):self.review(f)
    def test_context_quote_upgrade(self):
        f=fixture();f[1]['periods']['start_is_source_quoted']=True
        with self.assertRaises(ValueError):self.review(f)
    def test_old_calendar_is_not_admitted(self):
        f=list(fixture(current='2012-12-31',prior='2011-12-31'));f[5]=['2015-01-02','2028-01-01'];self.assertFalse(self.review(f)['clock_verified'])
    def test_available_first_session(self):
        f=fixture();f[5].insert(1,f[1]['publication_date']+'0')
        with self.assertRaises(ValueError):self.review(f)
    def test_context_holds_preserved(self):
        f=fixture();f[2]['context_holds']=[]
        with self.assertRaises(ValueError):self.review(f)
    def test_accounting_notes_preserved(self):
        f=fixture();f[1]['accounting_context_numbers_redacted']=['Retrospective change [number]'];f[2]['accounting_context_numbers_redacted']=deepcopy(f[1]['accounting_context_numbers_redacted']);r=self.review(f);self.assertFalse(r['accounting_comparability_admitted'])
    def test_accounting_flags_lost(self):
        f=fixture();f[2]['accounting_review_flags']=['restatement']
        with self.assertRaises(ValueError):self.review(f)
    def test_wrong_source_hash(self):
        f=fixture();f[3]['source_sha256']='a'*64
        with self.assertRaises(ValueError):self.review(f)
    def test_wrong_published_amount(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]['value_decimal']='31'
        with self.assertRaises(ValueError):self.review(f)
    def test_wrong_published_sign(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]['displayed_amount']='31'
        with self.assertRaises(ValueError):self.review(f)
    def test_wrong_published_markup(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]['locator']['physical_cells'][0]['markup_sha256']='a'*64
        with self.assertRaises(ValueError):self.review(f)
    def test_null_not_silently_filled(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]=None
        with self.assertRaises(ValueError):self.review(f)
    def test_iso_currency_not_upgraded(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]['currency_iso']='USD'
        with self.assertRaises(ValueError):self.review(f)
    def test_flags_not_upgraded(self):
        f=fixture();f[2]['fact_admitted']=True
        with self.assertRaises(ValueError):self.review(f)
    def test_wrong_event(self):
        f=fixture();f[2]['event_id']='KO_other'
        with self.assertRaises(ValueError):self.review(f)


class PopulationTests(unittest.TestCase):
    def data(self):
        docs=[dict(event_id=s+'_'+str(y),symbol=s,fiscal_year=y,publication_date=str(y+1)+'-02-01',available_session=str(y+1)+'-02-02',official_actual_period_end=str(y)+'-12-31',source=dict(path='synthetic.html',sha256='a'*64),context_status=m.STATUS if (s=='KO' and y<=2016) or (s=='LRCX' and 2013<=y<=2016) else 'not_processed') for s in ('KO','LRCX') for y in range(2011,2027)]
        chosen=[x for x in docs if x['context_status']==m.STATUS];events=[dict(x,fields=dict.fromkeys(m.FIELDS,{})) for x in chosen];ids=[x['event_id'] for x in docs]
        return dict(records=chosen,catalog_ids=ids,catalog_events=32),dict(events=events,catalog_ids=ids,counts=dict(events=10,by_symbol={'KO':6,'LRCX':4},nonnull_fields=30,unprocessed_catalog_events=22)),dict(events=docs),dict(required_fields=list(m.FIELDS),fixed_catalog_metadata=[{k:e[k] for k in ('event_id','symbol','fiscal_year','publication_date','available_session','official_actual_period_end','source','context_status')} for e in docs])
    def test_complete(self):self.assertEqual(len(m.verify_population(*self.data())),10)
    def test_unprocessed_cannot_disappear(self):
        p,d,c,y=self.data();c['events'].pop()
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_catalog_order(self):
        p,d,c,y=self.data();d['catalog_ids']=d['catalog_ids'][::-1]
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_event_order(self):
        p,d,c,y=self.data();d['events']=d['events'][::-1]
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_extra_new_year(self):
        p,d,c,y=self.data();c['events'][6]['context_status']=m.STATUS
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_counts(self):
        p,d,c,y=self.data();d['counts']['nonnull_fields']=29
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)
    def test_no_producer_import(self):
        tree=ast.parse(Path(m.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names += [x.name for x in n.names]
            elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('extract' in n or 'source_batch' in n or 'admission' in n for n in names))
    def test_missing_sha_prevents_io(self):
        with patch.object(m,'Reader') as r:
            for a,b in [(None,None),('x'*64,'a'*64),('a'*64,'')]:
                with self.assertRaises(ValueError):m.audit(a,b)
            r.assert_not_called()


class AccountingTests(unittest.TestCase):
    def fixture(self):
        raw,c,e,*_=fixture();text=m.norm(BeautifulSoup(raw,'html.parser').get_text(' ',strip=True))
        a=dict(event_id=e['event_id'],source=deepcopy(e['source']),status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],field_checks=dict.fromkeys(m.FIELDS,True),current_prior_common_basis=True,basis='Synthetic manual assessment of common consolidated statement basis.',evidence=[dict(path=e['source']['path'],sha256=e['source']['sha256'],locator=dict(kind='document_text',start=0,end=len(text)),quote=text)])
        return a,e,raw
    def test_exact_source_assessment(self):
        a,e,raw=self.fixture();self.assertEqual(m.verify_accounting(a,e,raw),a)
    def test_unresolved_preserved(self):
        a,e,raw=self.fixture();a.update(status='unresolved',reasons=['unresolved same-basis issue'],field_checks=dict.fromkeys(m.FIELDS,False),current_prior_common_basis=False,evidence=[])
        self.assertEqual(m.verify_accounting(a,e,raw)['status'],'unresolved')
    def test_resolved_without_field_proof(self):
        for name in m.FIELDS:
            a,e,raw=self.fixture();a['field_checks'][name]=False
            with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_unresolved_needs_reason(self):
        a,e,raw=self.fixture();a['status']='unresolved'
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_resolved_needs_common_basis(self):
        a,e,raw=self.fixture();a['current_prior_common_basis']=False
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_quote_must_match(self):
        a,e,raw=self.fixture();a['evidence'][0]['quote']='invented'
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_evidence_source_sha(self):
        a,e,raw=self.fixture();a['evidence'][0]['sha256']='a'*64
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_no_keyword_clearance(self):
        a,e,raw=self.fixture();a['evidence']=[];a['basis']='No restatement keyword found'
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_unrelated_scope_rejected(self):
        a,e,raw=self.fixture();a['scope']='cashflow_only'
        with self.assertRaises(ValueError):m.verify_accounting(a,e,raw)
    def test_exact_source_local_identity(self):
        f=fixture();r=m.review_event(f[4],f[3],f[2],f[1],f[5],f[6]);self.assertEqual((r['issuer_id'],r['security_id']),('The Coca-Cola Company','KO'))
        self.assertTrue(r['boundary_assumption_verified']);self.assertEqual(r['assumption_id'],'KO_calendar_annual_boundaries_v1')
    def test_indented_labels_and_columns(self):
        raw,c,*_=fixture('LRCX');raw=raw.replace(b'<tr>',b'<tr><td></td><td></td>')
        for spec in c['fields'].values():spec['label_cell']+=2;spec['logical_columns']=[n+2 for n in spec['logical_columns']]
        soup=rebuild_context(c,raw)
        self.assertEqual([m.source_field(soup,c,f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_policy_metadata_cannot_change(self):
        p,d,c,y=PopulationTests().data();y['fixed_catalog_metadata'][0]['source']={'changed':True}
        with self.assertRaises(ValueError):m.verify_population(p,d,c,y)


class NullTests(unittest.TestCase):
    def parts(self):
        f=fixture();f[2]['fields'][m.FIELDS[0]]=None;f[2]['gaps']=[dict(field=m.FIELDS[0],reason='producer monetary cell unresolved')];return f
    def test_null_kept_if_independently_readable(self):
        f=self.parts();r=m.review_event(f[4],f[3],f[2],f[1],f[5],f[6]);v=r['fields'][m.FIELDS[0]]
        self.assertTrue(v['is_null']);self.assertTrue(v['source_value_present']);self.assertTrue(v['null_reason_verified']);self.assertIsNone(f[2]['fields'][m.FIELDS[0]])
        self.assertEqual(m.position_counts([r]),(2,1))
    def test_unparseable_amount_is_not_absence(self):
        f=self.parts();raw=f[0].replace(b'(31)',b'N/A');soup=BeautifulSoup(raw,'html.parser')
        v,_=m.review_position(soup,f[1],m.FIELDS[0],f[2],f[2]['periods']['current'])
        self.assertIsNone(v['source_value_present']);self.assertTrue(v['null_reason_verified'])
    def test_null_cannot_bypass_title_mismatch(self):
        f=self.parts();raw=f[0].replace(b'Consolidated Statements of Income',b'Non-GAAP')
        with self.assertRaises(ValueError):m.review_position(BeautifulSoup(raw,'html.parser'),f[1],m.FIELDS[0],f[2],f[2]['periods']['current'])
    def test_null_needs_precise_frozen_gap(self):
        f=self.parts();f[2]['gaps']=[]
        with self.assertRaises(ValueError):m.review_position(BeautifulSoup(f[0],'html.parser'),f[1],m.FIELDS[0],f[2],f[2]['periods']['current'])
    def test_counts_are_null_aware(self):
        self.assertEqual(m.position_counts([dict(fields={'a':dict(is_null=True),'b':dict(is_null=False)})]),(1,1))
    def test_boolean_null_flag_required(self):
        with self.assertRaises(ValueError):m.position_counts([dict(fields={'a':dict(is_null=0)})])


class RetrospectiveTests(unittest.TestCase):
    def adjusted(self,header='As Adjusted 1'):
        raw,c,e,*_=fixture()
        raw=raw.replace(b'Year Ended',('Year Ended</td><td>'+header).encode())
        # Use an extra header inside the existing title cell, preserving columns.
        raw,c,e,*_=fixture()
        raw=raw.replace(b'(In millions, except per share)',('(In millions, except per share)</td></tr><tr><td>'+header).encode())
        # Rebuild row positions after the deliberately inserted header row.
        c['statement_contexts'][0]['header_stop_row']=4
        c['fields'][m.FIELDS[0]]['row']=5
        rebuild_context(c,raw)
        return raw,c,e
    def test_adjusted_requires_explicit_manual_decision(self):
        raw,c,e=self.adjusted()
        with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0])
    def test_adjusted_formal_header_explained(self):
        raw,c,e=self.adjusted()
        r=m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='resolved',gaap_retrospective_adjustment_verified=True))
        self.assertEqual(r['value_decimal'],'-31')
    def test_non_gaap_not_exempted(self):
        for text in ['Non-GAAP As Adjusted 1','Reconciliation','Pro forma','As Adjusted forecast']:
            raw,c,e=self.adjusted(text)
            with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='resolved',gaap_retrospective_adjustment_verified=True))
    def test_unresolved_does_not_exempt(self):
        raw,c,e=self.adjusted()
        with self.assertRaises(ValueError):m.source_field(BeautifulSoup(raw,'html.parser'),c,m.FIELDS[0],dict(status='unresolved',gaap_retrospective_adjustment_verified=True))
    def test_claim_without_quoted_source_proof_rejected(self):
        raw,c,e,*_=fixture()
        item=dict(event_id=e['event_id'],source=e['source'],status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis='Reviewed',field_checks={f:True for f in m.FIELDS},current_prior_common_basis=True,evidence=[],gaap_retrospective_adjustment_verified=True)
        with self.assertRaisesRegex(ValueError,'Retrospective'):m.verify_accounting(item,e,raw)


if __name__=='__main__':unittest.main(verbosity=2)
