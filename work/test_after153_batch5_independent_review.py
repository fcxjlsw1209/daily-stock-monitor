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
import after153_batch5_independent_review as m

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

def fixture(symbol='PLTR',ni='(31)',days=None,year=None,compound=False,percent=False,external=False,split=False):
    year=year or {'PLTR':2024,'PYPL':2018,'TMO':2020}[symbol]
    end=date(year,12,31);prior=date(year-1,12,31) if days is None else end-timedelta(days=days)
    days=(end-prior).days;pub=date(year+1,2,1);av=pub+timedelta(days=1)
    current,prev,release,available=map(str,(end,prior,pub,av));issuer=m.ISSUERS[symbol]
    scale=m.SCALES[symbol];unit='thousands' if scale=='1000' else 'millions'
    width=9 if percent else 7;prior_col=6 if percent else 5
    income_title='Condensed Consolidated Statement of Income'
    if symbol=='PYPL':income_title='Unaudited Condensed Combined and Consolidated Statement of Income'
    rows=[]
    if not external:
        if split:rows += [tr([td('Condensed Consolidated',width)]),tr([td('Statement of Income',width)])]
        else:rows.append(tr([td(income_title,width)]))
        rows.append(tr([td('(In '+unit+')',width)]))
    rows.append(tr([td(''),td('Three Months Ended'+(' December 31,' if compound else ''),2),td('Year Ended'+(' December 31,' if compound else ''),width-3)]))
    h=lambda d: str(date.fromisoformat(d).year) if compound else label(d)
    row=[td(''),td(h(current)),td(h(prev)),td(h(current),2)]
    if percent:row.append(td('% Revenues'))
    row.append(td(h(prev),2))
    if percent:row.append(td('% Revenues'))
    rows.append(tr(row));stop=len(rows)
    values=lambda label_,cur,prv: [td(label_),td('100'),td('101'),td(''),td(cur)]+([td('17%')] if percent else [])+[td('$'),td(prv)]+([td('13%')] if percent else [])
    rows.append(tr(values('Revenue','$ 9','8')))
    literal='($31)' if ni=='(31)' else '$ '+ni
    rows.append(tr(values('Net income',literal,'20')));ni_row=len(rows)-1
    rows.append(tr(values('Net income attributable to parent','$ 7','6')))
    before='<p>'+income_title+'</p><p>(In '+unit+')</p>' if external else ''
    income=before+'<table>'+''.join(rows)+'</table>'
    bs_rows=[tr([td('Condensed Consolidated Balance Sheet',5)]),tr([td('(In '+unit+')',5)])]
    if compound:
        bs_rows.append(tr([td(''),td('As of December 31,',4)]))
        bs_rows.append(tr([td(''),td(str(year),2),td(str(prior.year),2)]))
    else:bs_rows.append(tr([td(''),td(label(current),2),td(label(prev),2)]))
    bs_stop=len(bs_rows)
    bs_rows.append(tr([td('Current assets'),td('$'),td('1'),td('$'),td('2')]))
    bs_rows.append(tr([td('Total assets'),td('$'),td('1,234'),td('$'),td('987')]))
    asset_row=len(bs_rows)-1
    raw=('<html><body>'+issuer+' (NASDAQ: '+symbol+') '+label(release)+income+'<table>'+''.join(bs_rows)+'</table></body></html>').encode()
    source=dict(path='work/synthetic.html',sha256=m.sha(raw),url='https://'+next(iter(m.HOSTS[symbol]))+'/synthetic',receipt_path='work/synthetic.receipt.json')
    receipt=dict(sha256=m.sha(raw),url=source['url'],http_status=200,bytes=len(raw),retrieved_at='2026-10-20T00:00:00+00:00');rr=json.dumps(receipt).encode();source['receipt_sha256']=m.sha(rr)
    c=dict(event_id=symbol+'_'+release,symbol=symbol,fiscal_year=year,publication_date=release,available_session=available,available_session_verified=True,official_actual_period_end=current,source=source,source_url=source['url'],evidence_path=source['path'],evidence_sha256=source['sha256'],receipt=dict(path=source['receipt_path'],sha256=source['receipt_sha256'],retrieved_at=receipt['retrieved_at']),identity_anchors=[issuer,'NASDAQ: '+symbol],source_kind='HTML_original',original_report_bytes_verified=True,context_status=m.STATUS,context_holds=['accounting_not_admitted'],accounting_terms=['reclassification'],accounting_context_numbers_redacted=['Synthetic [number] note'],periods=dict(current_end=current,prior_end=prev,proposed_current_start=str(prior+timedelta(days=1)),start_is_source_quoted=False,start_assumption_approved=False,start_assumption='same_current_report_prior_annual_end_plus_one_calendar_day',days_inclusive=days,annual_date_binding='same_report_full_date_headers'),statement_contexts=[dict(table=0,kind='income',header_stop_row=stop,scale=scale,external_metadata=[income_title,'(In '+unit+')'] if external else []),dict(table=1,kind='balance',header_stop_row=bs_stop,scale=scale,external_metadata=[])],fields={})
    for f,ti,ri,cols,day in [(m.FIELDS[0],0,ni_row,[3,4],current),(m.FIELDS[1],1,asset_row,[1,2],current),(m.FIELDS[2],1,asset_row,[3,4],prev)]:c['fields'][f]=dict(table=ti,row=ri,label_cell=0,label='Net income' if ti==0 else 'Total assets',logical_columns=cols,as_of=day,scale=scale,currency_literal='$',currency_iso=None)
    soup=rebuild(c,raw);period=dict(period_start=str(prior+timedelta(days=1)),period_end=current,days_inclusive=days)
    e=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,publication_date=release,available_from=available,source=deepcopy(source),identity=dict(issuer=issuer,symbol=symbol,source_anchors=deepcopy(c['identity_anchors']),certified_daily_security_master=False),fields={},periods=dict(current=period,comparison=dict(period_end=prev),assumption_id=symbol+'_consecutive_annual_boundaries_v1',start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False),context_holds=deepcopy(c['context_holds']),accounting_review_flags=deepcopy(c['accounting_terms']),accounting_context_numbers_redacted=deepcopy(c['accounting_context_numbers_redacted']),gaps=[],fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_time_verified=False,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,financial_calculations=0,complete_required_fields=True)
    for f,v in zip(m.FIELDS,[ni,'1234','987']):
        spec=c['fields'][f];cells,_=m.grid_from_html(soup.find_all('table')[spec['table']]);chosen=[x for x in cells.values() if x['row']==spec['row'] and set(x['columns'])&set(spec['logical_columns'])];display=' '.join(x['text'] for x in chosen if x['text']);value='-31' if ni=='(31)' and f==m.FIELDS[0] else v
        e['fields'][f]=dict(value_decimal=value,displayed_amount=display,scale=scale,currency_literal='$',currency_iso=None,source=deepcopy(source),available_from=available,as_of=spec['as_of'],period=deepcopy(period) if f==m.FIELDS[0] else None,fact_admitted=False,locator=dict(table=spec['table'],row=spec['row'],header_columns=spec['logical_columns'],physical_cells=chosen))
    policy=dict(periods=dict(allowed_days=[365,366],required_year_end='12-31',boundary_assumptions={s:dict(not_a_quoted_fact=True,requires_independent_header_verification=True,assumption_id=s+'_consecutive_annual_boundaries_v1') for s in m.ISSUERS}))
    row=dict(event_id=c['event_id'],symbol=symbol,fiscal_year=year,source_path=source['path'],source_sha256=source['sha256'])
    return raw,c,e,row,Reader({source['path']:raw,source['receipt_path']:rr}),[release,available,'2028-01-01'],policy

class SourceTests(unittest.TestCase):
    def read(self,parts=None,field=m.FIELDS[0]):
        raw,c,*_=parts or fixture();return m.source_field(BeautifulSoup(raw,'html.parser'),c,field)
    def mutate(self,old,new,symbol='PLTR',headers=False):
        p=list(fixture(symbol));p[0]=p[0].replace(old,new)
        if headers:rebuild(p[1],p[0])
        return p
    def test_pltr(self):self.assertEqual([self.read(fixture(),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_tmo(self):self.assertEqual([self.read(fixture('TMO'),f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_compound_headers_all_publishers(self):
        for symbol in m.ISSUERS:
            for f,value in zip(m.FIELDS,['-31','1234','987']):
                self.assertEqual(self.read(fixture(symbol,compound=True),f)['value_decimal'],value)
    def test_external_title_and_scale(self):
        for f,value in zip(m.FIELDS,['-31','1234','987']):
            self.assertEqual(self.read(fixture(external=True,compound=True),f)['value_decimal'],value)
    def test_external_units_cannot_cross_prior_table(self):
        p=list(fixture(external=True));p[0]=p[0].replace(b'</p><table>',b'</p><table><tr><td>Unrelated table</td></tr></table><table>',1)
        # The frozen table locator deliberately remains at the original position.
        with self.assertRaises(ValueError):self.read(p)
    def test_external_title_missing_is_not_inferred(self):
        p=list(fixture(external=True));p[0]=p[0].replace(b'Condensed Consolidated Statement of Income',b'Operating overview')
        with self.assertRaises(ValueError):self.read(p)
    def test_split_title(self):
        self.assertEqual(self.read(fixture('TMO',split=True,compound=True))['value_decimal'],'-31')
    def test_pypl_combined_and_consolidated(self):
        self.assertIn('Combined and Consolidated',self.read(fixture('PYPL'))['statement_title'])
    def test_percentage_columns_are_never_monetary(self):
        p=fixture('TMO',compound=True,percent=True)
        for f,value in zip(m.FIELDS,['-31','1234','987']):
            self.assertEqual(self.read(p,f)['value_decimal'],value)
        self.assertEqual(self.read(p)['columns'],[3,4])
    def test_percentage_header_removed_fails_closed(self):
        p=list(fixture(compound=True,percent=True));p[0]=p[0].replace(b'% Revenues',b'')
        rebuild(p[1],p[0])
        with self.assertRaises(ValueError):self.read(p)
    def test_conflicting_compound_monthday(self):
        h=[dict(text='Year Ended December 31,',columns=[1,2],rowspan=1),
           dict(text='December 30,',columns=[1,2],rowspan=1),
           dict(text='2024',columns=[1],rowspan=1),dict(text='2023',columns=[2],rowspan=1)]
        with self.assertRaises(ValueError):m.date_groups(h,[dict(columns=[0,1,2])],True)
    def test_duplicate_compound_year(self):
        h=[dict(text='Year Ended December 31,',columns=[1,2],rowspan=1),
           dict(text='2024',columns=[1],rowspan=1),dict(text='2024',columns=[1],rowspan=1),
           dict(text='2023',columns=[2],rowspan=1)]
        with self.assertRaises(ValueError):m.date_groups(h,[dict(columns=[0,1,2])],True)
    def test_full_and_compound_date_disagree(self):
        h=[dict(text='Year Ended December 31,',columns=[1,2],rowspan=1),
           dict(text='2024',columns=[1],rowspan=1),dict(text='December 30, 2024',columns=[1],rowspan=1),
           dict(text='2023',columns=[2],rowspan=1)]
        with self.assertRaises(ValueError):m.date_groups(h,[dict(columns=[0,1,2])],True)
    def test_wide_label_is_not_a_monetary_domain(self):
        h=[dict(text='December 31, 2024',columns=[3,4],rowspan=1),
           dict(text='December 31, 2023',columns=[6,7],rowspan=1)]
        groups,owners=m.date_groups(h,[dict(columns=[0,1,2]),dict(columns=[3,4]),dict(columns=[5]),dict(columns=[6,7])],False)
        self.assertEqual(groups,{'2024-12-31':[3,4],'2023-12-31':[6,7]})
        self.assertFalse(set(owners)&{0,1,2})
    def test_multiline_date_header_within_header_only(self):
        h=[dict(text='Year Ended',columns=[1,2],row=1,rowspan=1),
           dict(text='December 31, 2024',columns=[1],row=2,rowspan=2),
           dict(text='December 31, 2023',columns=[2],row=2,rowspan=2)]
        self.assertEqual(m.date_groups(h,[dict(columns=[0]),dict(columns=[1]),dict(columns=[2])],True,4)[0],
                         {'2024-12-31':[1],'2023-12-31':[2]})
        for stop in (None,3):
            with self.assertRaises(ValueError):m.date_groups(h,[dict(columns=[0]),dict(columns=[1]),dict(columns=[2])],True,stop)
    def test_quarter_punctuation_cannot_leak_into_annual(self):
        self.assertEqual(self.read(self.mutate(b"<td>101</td>",b"<td>)</td>"))["value_decimal"],"-31")
    def test_caption_date_excluded(self):self.assertEqual(self.read(fixture('TMO'))['columns'],[3,4])
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
        for value in [b'in millions',b'in billions',b'without scale',b'in millions and in thousands']:
            with self.assertRaises(ValueError):self.read(self.mutate(b'In thousands',value,headers=True))
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
        for s in m.ISSUERS:
            for y in (2023,2024):self.assertTrue(self.review(fixture(s,year=y))['boundary_assumption_verified'])
    def test_other_lengths_rejected(self):
        for days in (330,350,364,367,370,371,372):
            with self.assertRaises(ValueError):self.review(fixture(days=days))
    def test_policy_assumption(self):
        p=fixture();p[6]['periods']['boundary_assumptions']['PLTR']['not_a_quoted_fact']=False
        with self.assertRaises(ValueError):self.review(p)
    def test_source_quotation_upgrade(self):
        p=fixture();p[2]['periods']['start_assumption_certified']=True
        with self.assertRaises(ValueError):self.review(p)
    def test_old_clock_preserved(self):
        p=list(fixture('TMO',year=2011));p[5]=['2015-01-02','2028-01-01'];p[1]['available_session_verified']=False;r=self.review(p);self.assertFalse(r['clock_verified']);self.assertTrue(r['metadata_verified'])
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
        p=fixture();p[2]['fields'][m.FIELDS[0]]=None;p[2]['complete_required_fields']=False;p[2]['gaps']=[dict(field=m.FIELDS[0],reason='parser_gap')];r=self.review(p);self.assertTrue(r['fields'][m.FIELDS[0]]['null_reason_verified']);self.assertIsNone(p[2]['fields'][m.FIELDS[0]]);self.assertEqual(m.position_counts([r]),(2,1))
    def test_null_cannot_silently_pass(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]=None
        with self.assertRaises(ValueError):self.review(p)
    def test_source_counter_is_exact_zero_integer(self):
        for value in (False,True,1,None):
            p=fixture();p[2]['financial_calculations']=value
            with self.assertRaises(ValueError):self.review(p)
    def test_completeness_cannot_be_claimed_for_null(self):
        p=fixture();p[2]['fields'][m.FIELDS[0]]=None;p[2]['gaps']=[dict(field=m.FIELDS[0],reason='parser_gap')]
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
        for s,y in m.FULL:
            chosen=(s,y) in m.SELECTED;original=not(s=='TMO' and y==2022)
            docs.append(dict(event_id=s+'_'+str(y),symbol=s,fiscal_year=y,publication_date=str(y+1)+'-02-01',available_session=str(y+1)+'-02-02',available_session_verified=True,official_actual_period_end=str(y)+'-12-31',source={},source_kind='HTML_original' if original else 'index_or_unresolved_not_original_report',original_report_bytes_verified=original,context_status=m.STATUS if chosen else 'unprocessed',context_holds=['not_admitted']))
        chosen=[e for e in docs if e['context_status']==m.STATUS];ids=[e['event_id'] for e in docs];proj=[{k:e[k] for k in m.CATALOG_KEYS} for e in docs]
        return dict(records=chosen,catalog_ids=ids,catalog_events=28,catalog_metadata=deepcopy(proj)),dict(events=[dict(e,fields=dict.fromkeys(m.FIELDS,{})) for e in chosen],catalog_ids=ids,catalog_metadata=deepcopy(proj),counts=dict(events=21,by_symbol={'PLTR':6,'PYPL':5,'TMO':10},nonnull_fields=63,unprocessed_catalog_events=7)),dict(events=docs),dict(required_fields=list(m.FIELDS),fixed_catalog_metadata=proj)
    def test_complete(self):self.assertEqual(len(m.verify_population(*self.data())),21)
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
