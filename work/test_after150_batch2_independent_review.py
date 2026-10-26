"""Pure synthetic tests; no real reports, producer imports or strategy I/O."""
import ast
from copy import deepcopy
from datetime import date, timedelta
from decimal import localcontext
import json
from pathlib import Path
import unittest
from bs4 import BeautifulSoup
import after150_batch2_independent_review as m


def td(text, width=1):
    return '<td%s>%s</td>' % ((' colspan="%s"'%width) if width != 1 else '',text)


def tr(items):
    return '<tr>'+''.join(items)+'</tr>'


def day_label(iso):
    d = date.fromisoformat(iso)
    return d.strftime('%B')+' '+str(d.day)+', '+str(d.year)


class MemoryReader:
    def __init__(self, files):
        self.files = files
        self.hashes = {}
    def read(self,path,expected=None):
        raw = self.files[m.relative(path)]
        digest = m.sha(raw)
        m.require(expected is None or digest == expected,'SHA mismatch')
        self.hashes[m.relative(path)] = digest
        return raw
    def load(self,path,expected=None):
        return json.loads(self.read(path,expected))


def fixture(symbol='INTC', ni='(31)', year=None, current=None, prior=None):
    year = year or (2021 if symbol == 'INTC' else 2023)
    current = current or ('2021-12-25' if symbol == 'INTC' else '%d-06-30'%year)
    prior = prior or ('2020-12-26' if symbol == 'INTC' else '%d-06-30'%(year-1))
    start = (date.fromisoformat(prior)+timedelta(days=1)).isoformat()
    release = (date.fromisoformat(current)+timedelta(days=30)).isoformat()
    available = (date.fromisoformat(release)+timedelta(days=1)).isoformat()
    unit = 'millions' if symbol == 'INTC' else 'thousands'
    name = 'Intel Corporation' if symbol == 'INTC' else 'KLA Corporation'
    income_title = 'Consolidated Statements of Income' if symbol == 'INTC' else 'Condensed Consolidated Unaudited Statements of Operations'
    ni_table = '<table>'+tr([td(income_title,9)])+tr([td(''),td('Three Months Ended',3),td(''),td('Twelve Months Ended',3),td('')])+tr([td('(In '+unit+', except per share amounts)'),td(day_label(current)),td(''),td(day_label(prior)),td(''),td(day_label(current)),td(''),td(day_label(prior)),td('')])+tr([td('Net revenues'),td('$ 1'),td(''),td('$ 2'),td(''),td('$ 3'),td(''),td('$ 4'),td('')])+tr([td('Net income'),td('9'),td(''),td('8'),td(''),td(ni),td(''),td('7'),td('')])+'</table>'
    bs_table = '<table>'+tr([td('Consolidated Balance Sheets',5)])+tr([td('(In '+unit+')'),td(day_label(current)),td(''),td(day_label(prior)),td('')])+tr([td('Assets'),td('$ 1'),td(''),td('$ 2'),td('')])+tr([td('Total assets'),td('1,234'),td(''),td('987'),td('')])+'</table>'
    raw = ('<html><p>'+name+' (NASDAQ: '+symbol+') '+day_label(release)+'</p>'+ni_table+bs_table+'</html>').encode()
    path = 'work/synthetic_'+symbol+'.html'
    receipt_path = 'work/synthetic_'+symbol+'_receipt.json'
    url = 'https://www.intc.com/news/x' if symbol == 'INTC' else 'https://ir.kla.com/news-events/press-releases/detail/x'
    receipt = dict(sha256=m.sha(raw),url=url,status=200,bytes=len(raw),retrieved_at='2026-10-20T12:00:00+00:00')
    receipt_raw = json.dumps(receipt).encode()
    rref = dict(path=receipt_path,sha256=m.sha(receipt_raw),status=200,retrieved_at=receipt['retrieved_at'],url_and_raw_sha_match=True)
    context = dict(symbol=symbol,event_id=symbol+'_'+release,fiscal_year=year,publication_date=release,available_session=available,official_actual_period_end=current,evidence_path=path,evidence_sha256=m.sha(raw),source_url=url,
                   context=dict(receipt=rref,statement_tables=[],accounting_contexts_numbers_redacted=[]),
                   context_summary=dict(accounting_attention=[],comparison_assets_date=prior,current_annual_period=dict(start=None if symbol=='INTC' else start,end=current,candidate_start_from_prior_end_plus_one=start),
                     net_income_locator=dict(table=0,row=4,label_cell=0,label='Net income',current_header_columns=[5],comparison_header_columns=[7]),
                     assets_locator=dict(table=1,row=3,label_cell=0,label='Total assets',current_header_columns=[1],comparison_header_columns=[3])))
    soup = BeautifulSoup(raw,'html.parser')
    for ti,stop in [(0,3),(1,2)]:
        cells,_ = m.grid_from_html(soup.find_all('table')[ti])
        context['context']['statement_tables'].append(dict(table=ti,header_stop_row=stop,headers=[{k:c[k] for k in ('row','physical_cell','columns','rowspan','text')} for c in cells.values() if c['row'] < stop]))
    p = dict(period_start=start,period_end=current,days_inclusive=(date.fromisoformat(current)-date.fromisoformat(start)).days+1)
    source = dict(path=path,sha256=m.sha(raw),url=url)
    if symbol == 'INTC':
        source['receipt'] = receipt_path
    event = dict(event_id=context['event_id'],symbol=symbol,fiscal_year=year,publication_date=release,available_from=available,source=source,periods=dict(current=p,comparison=dict(period_start=None,period_end=prior,days_inclusive=None)),fields={},gaps=[],context_holds=[],accounting_review_flags=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)
    if symbol == 'INTC':
        event['periods'].update(start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',start_assumption_certified=False)
        event['context_holds'] = [m.INTC_HOLD]
    else:
        event.update(receipt=deepcopy(rref),original_context=deepcopy(context),accounting_context=[])
        if year == 2019:
            event['context_holds'] = [m.ASC_HOLD]
    # These fixture values are literal expected values, not a producer result.
    for field,ti,row,col,value,display in [('annual_net_income',0,4,5,m.signed_text(ni),ni),('current_total_assets',1,3,1,'1234','1,234'),('prior_total_assets',1,3,3,'987','987')]:
        cells,_ = m.grid_from_html(soup.find_all('table')[ti])
        cell = cells[row,col]
        event['fields'][field] = dict(value_decimal=value,displayed_amount=display,scale='1000000' if symbol=='INTC' else '1000',currency_literal='$',currency_iso=None,fact_admitted=False,source=deepcopy(source),available_from=available,as_of=prior if field=='prior_total_assets' else current,period=deepcopy(p) if field=='annual_net_income' else None,
           locator=dict(table=ti,row=row,header_columns=[col],physical_cells=[{k:cell[k] for k in ('row','physical_cell','columns','rowspan','text','markup_sha256')}]))
    policy = dict(periods=dict(INTC_52_53_week_assumption=dict(enabled=True,not_a_quoted_fact=True,assumption_id=m.ASSUMPTION)))
    row = dict(event_id=event['event_id'],symbol=symbol,fiscal_year=year,source_path=path,source_sha256=m.sha(raw))
    reader = MemoryReader({path:raw,receipt_path:receipt_raw})
    sessions = [release,available,(date.fromisoformat(available)+timedelta(days=1)).isoformat()]
    return raw,context,event,row,reader,sessions,policy


def update_context_headers(context,raw):
    soup = BeautifulSoup(raw,'html.parser')
    for spec in context['context']['statement_tables']:
        cells,_ = m.grid_from_html(soup.find_all('table')[spec['table']])
        spec['headers'] = [{k:c[k] for k in ('row','physical_cell','columns','rowspan','text')} for c in cells.values() if c['row'] < spec['header_stop_row']]
    return soup


class SourceTests(unittest.TestCase):
    def read(self, raw=None, context=None, field='annual_net_income', symbol='INTC'):
        if raw is None:
            raw,context,*_ = fixture(symbol)
        return m.source_field(BeautifulSoup(raw,'html.parser'),context,field)
    def test_intc_signed_total(self):
        self.assertEqual(self.read()['value_decimal'],'-31')
    def test_klac_signed_total(self):
        self.assertEqual(self.read(symbol='KLAC')['value_decimal'],'-31')
    def test_all_three_values(self):
        raw,context,*_ = fixture()
        self.assertEqual([self.read(raw,context,f)['value_decimal'] for f in m.FIELDS],['-31','1234','987'])
    def test_zero_income(self):
        raw,context,*_ = fixture(ni='0')
        self.assertEqual(self.read(raw,context)['value_decimal'],'0')
    def test_large_decimal_low_precision(self):
        raw,context,*_ = fixture(ni='(123456789012345678901234.123456789)')
        with localcontext() as ctx:
            ctx.prec = 3
            self.assertEqual(self.read(raw,context)['value_decimal'],'-123456789012345678901234.123456789')
    def test_invalid_monetary_syntax(self):
        for value in ['—','', 'N/A','1 2','(-31)','(31','31)','1,23','31%','31*','$ $31','NaN','Infinity']:
            with self.subTest(value=value):
                raw,context,*_ = fixture()
                raw = raw.replace(b"(31)", value.encode())
                with self.assertRaises(ValueError): self.read(raw,context)
    def test_missing_total_not_attribution(self):
        raw,context,*_ = fixture()
        raw = raw.replace(b'<td>Net income</td>',b'<td>Net income attributable to parent</td>')
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_duplicate_total(self):
        raw,context,*_ = fixture()
        raw = raw.replace(b'</table>',b'<tr><td>Net income</td><td>2</td></tr></table>',1)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_quarter_cannot_replace_annual(self):
        raw,context,*_ = fixture()
        raw = raw.replace(b'Twelve Months Ended',b'Three Months Ended')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_six_and_nine_months_rejected(self):
        for word in [b'Six',b'Nine']:
            raw,context,*_ = fixture()
            raw = raw.replace(b'Twelve',word)
            update_context_headers(context,raw)
            with self.assertRaises(ValueError): self.read(raw,context)
    def test_frozen_header_mutation(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'Twelve',b'Three'),context)
    def test_wrong_locator_columns(self):
        raw,context,*_ = fixture()
        context['context_summary']['net_income_locator']['current_header_columns']=[1]
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_swapped_dates_not_silently_accepted(self):
        raw,context,*_ = fixture()
        raw = raw.replace(b'December 25, 2021',b'December 24, 2021')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_split_month_day_and_year_header(self):
        raw,context,*_ = fixture('KLAC')
        raw = raw.replace(b'Twelve Months Ended',b'Twelve Months Ended June 30,').replace(b'Three Months Ended',b'Three Months Ended June 30,')
        # Only income table dates are split; balance dates remain fully printed.
        first,second = raw.split(b'</table>',1)
        first=first.replace(b'June 30, 2023',b'2023').replace(b'June 30, 2022',b'2022')
        raw=first+b'</table>'+second
        update_context_headers(context,raw)
        self.assertEqual(self.read(raw,context)['value_decimal'],'-31')
    def test_missing_title(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'Consolidated Statements of Income',b'Summary')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_non_gaap_title(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'Consolidated Statements of Income',b'Non-GAAP Consolidated Statements of Income')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_cf_is_not_income_statement(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'Consolidated Statements of Income',b'Consolidated Statements of Cash Flows')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_external_formal_title(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'<table>',b'<p>Consolidated Statements of Income</p><table>',1).replace(b'<td colspan="9">Consolidated Statements of Income</td>',b'<td colspan="9"></td>')
        update_context_headers(context,raw)
        self.assertEqual(self.read(raw,context)['value_decimal'],'-31')
    def test_currency_conflict(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'$ 3',b'EUR 3'),context)
    def test_missing_dollar(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'$',b''),context)
    def test_wrong_scale(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'In millions',b'In thousands')
        update_context_headers(context,raw)
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_cross_period_colspan(self):
        raw,context,*_ = fixture()
        raw=raw.replace(b'<td>(31)</td>',b'<td colspan="3">(31)</td>')
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_rowspan_amount(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'<td>(31)</td>',b'<td rowspan="2">(31)</td>'),context)
    def test_nested_table(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'(31)',b'<table><tr><td>(31)</td></tr></table>'),context)
    def test_negative_assets(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'1,234',b'(1234)'),context,'current_total_assets')
    def test_zero_assets(self):
        raw,context,*_ = fixture()
        with self.assertRaises(ValueError): self.read(raw.replace(b'1,234',b'0'),context,'current_total_assets')
    def test_cross_year_sign(self):
        raw,context,*_ = fixture(ni='31')
        raw=raw.replace(b'<td>31</td><td></td>',b'<td>31</td><td>)</td>')
        with self.assertRaises(ValueError): self.read(raw,context)
    def test_date_groups_and_ambiguity(self):
        headers=[dict(columns=[1],text='June 30, 2023'),dict(columns=[3],text='June 30, 2022')]
        groups,owners=m.derive_date_groups(headers,[dict(columns=[0,1,2,3])],False)
        self.assertEqual(groups,{'2023-06-30':[1],'2022-06-30':[3]})
        self.assertIsNone(owners[2])
    def test_conflicting_date_header(self):
        headers=[dict(columns=[1],text='June 30, 2023 June 30, 2022')]
        with self.assertRaises(ValueError): m.derive_date_groups(headers,[dict(columns=[1])],False)
    def test_three_distinct_dates(self):
        headers=[dict(columns=[i],text='June 30, '+str(2020+i)) for i in range(1,4)]
        with self.assertRaises(ValueError): m.derive_date_groups(headers,[dict(columns=[1,2,3])],False)


class EventTests(unittest.TestCase):
    def review(self,parts):
        raw,context,event,row,reader,sessions,policy=parts
        return m.review_event(reader,row,event,context,sessions,policy)
    def test_intc_event(self):
        result=self.review(fixture())
        self.assertTrue(result['boundary_assumption_verified'])
        self.assertFalse(result['start_is_source_quoted'])
        self.assertTrue(result['clock_verified'])
        self.assertEqual(len(result['boundary_evidence']),3)
    def test_klac_event(self):
        result=self.review(fixture('KLAC'))
        self.assertTrue(result['metadata_verified'])
        self.assertNotIn('boundary_assumption_verified',result)
    def test_intc_371_days(self):
        self.assertTrue(self.review(fixture(current='2022-12-31',prior='2021-12-25'))['boundary_assumption_verified'])
    def test_intc_invalid_duration(self):
        with self.assertRaises(ValueError): self.review(fixture(current='2021-12-31'))
    def test_intc_policy_disabled(self):
        f=fixture();f[-1]['periods']['INTC_52_53_week_assumption']['enabled']=False
        with self.assertRaises(ValueError): self.review(f)
    def test_intc_quoted_start_prohibited(self):
        f=fixture();f[1]['context_summary']['current_annual_period']['start']='2020-12-27'
        with self.assertRaises(ValueError): self.review(f)
    def test_intc_certified_flag_prohibited(self):
        f=fixture();f[2]['periods']['start_assumption_certified']=True
        with self.assertRaises(ValueError): self.review(f)
    def test_contextual_hold_preserved(self):
        f=fixture();f[2]['context_holds']=[]
        with self.assertRaises(ValueError): self.review(f)
    def test_asc606_hold_preserved(self):
        f=fixture('KLAC',year=2019)
        self.assertIn(m.ASC_HOLD,self.review(f)['retained_context_holds'])
        f[2]['context_holds']=[]
        with self.assertRaises(ValueError): self.review(f)
    def test_clock_skipping_first_session(self):
        f=fixture();f[5].insert(1,'2022-01-24')
        # fixture release=2022-01-24; add a genuinely earlier next session by changing availability.
        f=fixture();f[2]['available_from']='2022-01-26';f[1]['available_session']='2022-01-26'
        with self.assertRaises(ValueError): self.review(f)
    def test_outside_calendar_not_certified(self):
        f=list(fixture());f[5]=['2025-01-01','2025-01-02']
        self.assertFalse(self.review(f)['clock_verified'])
    def test_duplicate_calendar(self):
        f=fixture();f[5].append(f[5][-1])
        with self.assertRaises(ValueError): self.review(f)
    def test_prior_asset_date_cannot_backdate_availability(self):
        f=fixture();f[2]['fields']['prior_total_assets']['available_from']='2020-12-26'
        with self.assertRaises(ValueError): self.review(f)
    def test_source_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['source']['sha256']='0'*64
        with self.assertRaises(ValueError): self.review(f)
    def test_amount_sign_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['value_decimal']='31'
        with self.assertRaises(ValueError): self.review(f)
    def test_displayed_sign_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['displayed_amount']='31'
        with self.assertRaises(ValueError): self.review(f)
    def test_units_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['scale']='1000'
        with self.assertRaises(ValueError): self.review(f)
    def test_physical_cell_hash_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['locator']['physical_cells'][0]['markup_sha256']='0'*64
        with self.assertRaises(ValueError): self.review(f)
    def test_physical_cell_number_mismatch(self):
        f=fixture();f[2]['fields']['annual_net_income']['locator']['physical_cells'][0]['physical_cell']=1
        with self.assertRaises(ValueError): self.review(f)
    def test_changed_current_period(self):
        f=fixture();f[2]['periods']['current']['period_start']='2021-01-01'
        with self.assertRaises(ValueError): self.review(f)
    def test_iso_or_admission_not_upgraded(self):
        for key,value in [('currency_iso','USD'),('fact_admitted',True)]:
            f=fixture();f[2]['fields']['annual_net_income'][key]=value
            with self.assertRaises(ValueError): self.review(f)
    def test_unexpected_null_rejected(self):
        f=fixture();f[2]['fields']['annual_net_income']=None
        with self.assertRaises(ValueError): self.review(f)
    def test_receipt_digest_mismatch(self):
        f=fixture();f[1]['context']['receipt']['sha256']='0'*64
        with self.assertRaises(ValueError): self.review(f)
    def test_report_digest_mismatch(self):
        f=fixture();f[3]['source_sha256']='0'*64
        with self.assertRaises(ValueError): self.review(f)
    def test_wrong_security_identity(self):
        f=fixture();f[2]['symbol']='KLAC'
        with self.assertRaises(ValueError): self.review(f)
    def test_original_context_not_mutated(self):
        f=fixture('KLAC');before=deepcopy(f[1]);self.review(f);self.assertEqual(before,f[1])


class PRETests(unittest.TestCase):
    def fixture(self):
        soup=BeautifulSoup('<pre>Condensed Consolidated Unaudited Balance Sheets\nTotal assets    123 456</pre><pre>Condensed Consolidated Unaudited Statements of Operations\nNet income     (123) 45</pre>','html.parser')
        context=dict(symbol='KLAC',fiscal_year=2011,official_actual_period_end='2011-06-30',context_summary=dict(status='preformatted_original_requires_separate_column_geometry_review',net_income_locator=dict(pre_block=1),assets_locator=dict(pre_block=0)))
        event=dict(context_holds=[m.PRE_HOLD],periods=dict(current=dict(period_start=None,period_end='2011-06-30',days_inclusive=None),comparison=dict(period_start=None,period_end=None,days_inclusive=None)),fields=dict.fromkeys(m.FIELDS),gaps=[dict(field=f,reason=m.PRE_HOLD) for f in m.FIELDS])
        return soup,context,event
    def test_three_nulls_preserved_without_parsing(self):
        result=m.review_pre(*self.fixture())
        self.assertEqual(len(result),3)
        self.assertTrue(all(x['null_reason_verified'] and x['source_value_present'] is None for x in result.values()))
    def test_filled_null_rejected(self):
        f=self.fixture();f[2]['fields']['annual_net_income']={'value_decimal':'123'}
        with self.assertRaises(ValueError): m.review_pre(*f)
    def test_missing_original_not_misclassified_as_unparsed(self):
        f=self.fixture();f[0].find_all('pre')[1].decompose()
        with self.assertRaises(ValueError): m.review_pre(*f)
    def test_null_gap_reason_required(self):
        f=self.fixture();f[2]['gaps'][0]['reason']='not found'
        with self.assertRaises(ValueError): m.review_pre(*f)
    def test_pre_period_not_fabricated(self):
        f=self.fixture();f[2]['periods']['current']['period_start']='2010-07-01'
        with self.assertRaises(ValueError): m.review_pre(*f)


class IndependenceTests(unittest.TestCase):
    def test_no_producer_imports(self):
        tree=ast.parse(Path(m.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(name and name.startswith(('after150_intc_extract','after150_klac_extract','after150_source_batch','after149_amzn_extract','after141_amd_source_extract')) for name in imports))
    def test_source_closure(self):
        self.assertEqual(len(m.source_files()),3)
        self.assertIn('work/after149_batch1_independent_review.py',m.source_files())
    def test_missing_frozen_hashes_fail_before_io(self):
        with self.assertRaises(ValueError): m.audit(None,None)
    def test_bad_spans(self):
        for span in ['0','-1','101']:
            with self.assertRaises(ValueError): m.grid_from_html(BeautifulSoup('<table><tr><td colspan="'+span+'">x</td></tr></table>','html.parser').table)
    def test_exact_decimal_parser(self):
        self.assertEqual(m.signed_text('($ 1,234.50)'),'-1234.50')
        self.assertEqual(m.signed_text('− 3'),'-3')


class PopulationTests(unittest.TestCase):
    def fixture(self):
        docs=[]
        for symbol in ('INTC','KLAC'):
            for i in range(16):
                year=2010+i if symbol=='INTC' else 2011+i
                selected=symbol=='KLAC' or year in (2017,2020,2021)
                docs.append(dict(symbol=symbol,fiscal_year=year,event_id=symbol+'_'+str(year)+'-07-31',publication_date=str(year)+'-07-31',available_session=str(year)+'-08-01',official_actual_period_end=str(year)+'-06-30',evidence_path='work/'+symbol+str(year)+'.html',evidence_sha256=str(i%10)*64,html_context_reviewed=selected,gap=None if selected else 'unprocessed'))
        chosen=[d for d in docs if d['html_context_reviewed']]
        plan=dict(records=[dict(event_id=d['event_id']) for d in chosen],catalog_ids=[d['event_id'] for d in docs],catalog_events=32)
        events=[dict(event_id=d['event_id'],fields={f:None if d['symbol']=='KLAC' and d['fiscal_year']==2011 else {} for f in m.FIELDS}) for d in chosen]
        data=dict(events=events,catalog_ids=list(plan['catalog_ids']),counts=dict(events=19,by_symbol={'INTC':3,'KLAC':16},nonnull_fields=54))
        policy=dict(required_fields=list(m.FIELDS),fixed_catalog_metadata=[dict(event_id=d['event_id'],symbol=d['symbol'],fiscal_year=d['fiscal_year'],publication_date=d['publication_date'],available_session=d['available_session'],official_actual_period_end=d['official_actual_period_end'],source_path=d['evidence_path'],source_sha256=d['evidence_sha256'],html_context_reviewed=d['html_context_reviewed'],retained_context_gap=d['gap']) for d in docs])
        return plan,data,dict(documents=docs),policy
    def test_exact19_and32(self):
        self.assertEqual(len(m.verify_population(*self.fixture())),19)
    def test_missing_unprocessed_catalog_event(self):
        f=self.fixture();f[2]['documents'].pop(0)
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_success_only_event_subset(self):
        f=self.fixture();f[1]['events'].pop()
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_reordered_events(self):
        f=self.fixture();f[1]['events'].reverse()
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_duplicate_catalog_id(self):
        f=self.fixture();f[2]['documents'][0]['event_id']=f[2]['documents'][1]['event_id']
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_metadata_policy_mismatch(self):
        f=self.fixture();f[3]['fixed_catalog_metadata'][0]['publication_date']='2010-01-01'
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_nonnull_count_not_assumed(self):
        f=self.fixture();f[1]['counts']['nonnull_fields']=57
        with self.assertRaises(ValueError): m.verify_population(*f)
    def test_hidden_selection_change(self):
        f=self.fixture();f[2]['documents'][0]['html_context_reviewed']=True
        with self.assertRaises(ValueError): m.verify_population(*f)


class AdditionalGeometryTests(unittest.TestCase):
    def test_duplicate_date_anchor_rejected(self):
        h=[dict(columns=[1],text='June 30, 2023'),dict(columns=[1],text='June 30, 2023'),dict(columns=[3],text='June 30, 2022')]
        with self.assertRaises(ValueError): m.derive_date_groups(h,[dict(columns=[1,2,3])],False)
    def test_wide_date_anchor_split_signs(self):
        # Independently manufactured wide three-cell annual spans, not any
        # producer fixture. The negative amount uses separate physical signs.
        raw,context,*_=fixture()
        table='<table><tr><td colspan="8">Consolidated Statements of Income</td></tr><tr><td></td><td colspan="7">Twelve Months Ended</td></tr><tr><td>In millions</td><td colspan="3">December 25, 2021</td><td></td><td colspan="3">December 26, 2020</td></tr><tr><td>Net revenues</td><td>$</td><td>1</td><td></td><td></td><td>$</td><td>2</td><td></td></tr><tr><td>Net income</td><td>($</td><td>31</td><td>)</td><td></td><td>$</td><td>8</td><td></td></tr></table>'
        soup=BeautifulSoup(raw,'html.parser')
        soup.find('table').replace_with(BeautifulSoup(table,'html.parser').table)
        raw=str(soup).encode()
        spec=context['context_summary']['net_income_locator']
        spec.update(current_header_columns=[1,2,3],comparison_header_columns=[5,6,7])
        update_context_headers(context,raw)
        result=m.source_field(soup,context,'annual_net_income')
        self.assertEqual(result['value_decimal'],'-31')
        self.assertEqual([c['physical_cell'] for c in result['physical_cells']],[1,2,3])
    def test_boundary_contradiction_fails(self):
        f=fixture()
        # Direct period contract rejects a stub even if a producer labels it annual.
        f[2]['periods']['current']['days_inclusive']=90
        with self.assertRaises(ValueError): m.periods(f[1],f[2],f[-1])
    def test_receipt_http_failure(self):
        f=fixture();path=f[1]['context']['receipt']['path']
        receipt=json.loads(f[4].files[path]);receipt['status']=404
        raw=json.dumps(receipt).encode();f[4].files[path]=raw
        f[1]['context']['receipt']['sha256']=m.sha(raw)
        with self.assertRaises(ValueError): m.receipt_evidence(f[4],f[1],f[0])
    def test_receipt_url_failure(self):
        f=fixture();path=f[1]['context']['receipt']['path']
        receipt=json.loads(f[4].files[path]);receipt['url']='https://example.com'
        raw=json.dumps(receipt).encode();f[4].files[path]=raw
        f[1]['context']['receipt']['sha256']=m.sha(raw)
        with self.assertRaises(ValueError): m.receipt_evidence(f[4],f[1],f[0])


if __name__=='__main__':
    unittest.main(verbosity=2)
