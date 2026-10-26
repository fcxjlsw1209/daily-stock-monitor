"""ABBV source-only annual revenue parser. No financial features or strategy work."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import profitability141_issuer_statements as layout
import after155_abbv_revenue_audit as audit

ROOT, OUT = audit.ROOT, audit.OUT
SELF='work/after155_abbv_revenue_prepare.py'
SCOPE=OUT+'SOURCE_SCOPE.json'
PROFILE=OUT+'EXTRACTION_PROFILE.json'
TEST=OUT+'FABRICATED_TESTS.json'
CALENDAR='outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
ANNUAL=re.compile(r'Twelve Months Ended (?:December|Dec\.) 31',re.I)
NOTE=re.compile(r'restat\w*|recast\w*|retrospect\w*|reclassif\w*|revis\w*|conform\w*|'
                r'revenue recognition|accounting (?:standard|principle|change)|'
                r'(?:Topic|ASC)\s*606|adopt(?:ed|ion|ing)|separat\w*|'
                r'predecessor|combined financial|discontinued|pro forma|acquisition|acquired|divest\w*',re.I)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def load(rel): return json.loads((ROOT/rel).read_bytes())
def cells(row): return row.find_all(['td','th'],recursive=False)
def txt(tag): return layout.norm(tag.get_text(' ',strip=True))
def assert_inputs(scope):
    for b in scope['inputs']: audit.read(b)
    for s in scope['sources']: audit.read(s['raw']); audit.read(s['receipt'])

def statement_metadata(soup,year):
    """No amount conversion: only headings, labels, header ownership and geometry."""
    candidates=[]
    for ti,t in enumerate(soup.find_all('table')):
        rows,grid,physical=layout.table_grid(t)
        labels=[(ri,pi,txt(c)) for ri,row in enumerate(rows) for pi,c in enumerate(cells(row))
                if re.fullmatch(r'Net (?:sales|revenues)',txt(c),re.I)]
        if not labels: continue
        ri=labels[0][0]
        context=' '.join(txt(r) for r in rows[:ri])
        if not re.search(r'AbbVie Inc\..*Consolidated Statements of Earnings',context,re.I): continue
        if re.search(r'non[ -]?GAAP|\badjusted\b|reconciliation|pro forma',context,re.I):
            raise ValueError('Adjusted formal heading')
        if len(labels)!=1: raise ValueError('Duplicate revenue labels')
        if labels[0][1]!=0: raise ValueError('Revenue label must start statement row')
        if labels[0][2].lower()!=('net sales' if year<=2014 else 'net revenues'):
            raise ValueError('Unregistered annual revenue label')
        if not any(txt(c).lower()=='cost of products sold' for row in rows for c in cells(row)):
            raise ValueError('Consolidated income statement cost context absent')
        units=re.findall(r'\(in millions, except per share data\)',context,re.I)
        if len(units)!=1 or layout.scale_from(context)!=1000000: raise ValueError('Unsupported unit context')
        groups={str(year):[],str(year-1):[]}
        for (r,pi),item in physical.items():
            if r!=ri: continue
            ownership=[]
            for col in item['columns']:
                headers=layout.headers_for(grid,ri,col,ri)
                duration=[v for v in headers if ANNUAL.fullmatch(v)]
                years=[v for v in headers if re.fullmatch(r'\d{4}',v)]
                if len(duration)==1 and len(years)==1 and years[0] in groups:
                    ownership.append((years[0],duration[0],headers))
            if not ownership: continue
            if len(ownership)!=len(item['columns']) or len({v[0] for v in ownership})!=1:
                raise ValueError('Cell crosses annual header ownership')
            y,duration,headers=ownership[0]
            groups[y].append({'physical_cell_index':pi,'logical_columns':item['columns'],
                              'header_literals':headers,'duration_literal':duration})
        if any(not v for v in groups.values()): raise ValueError('Annual current/prior header pair absent')
        candidates.append({'table_index':ti,'table_html_sha256':sha(str(t).encode()),
                           'row_index':ri,'label':labels[0][2],'statement_title_unit_context':context,
                           'unit_literal':units[0], 'header_rows':[[txt(c) for c in cells(r)] for r in rows[:ri]],
                           'groups':groups,
                           'mixed_measure_note':'Separate adjusted EPS rows may occur below revenue. They are neither revenue inputs nor evidence of adjusted revenue.'})
    if len(candidates)!=1: raise ValueError('Formal statement candidate count: '+str(len(candidates)))
    return candidates[0]

def receipt_check(raw,receipt,meta):
    required={'url','final_url','status','retrieved_at','sha256','bytes','content_type','role','ticker','source_representation','request_method','request_payload'}
    if not required.issubset(receipt): raise ValueError('Receipt schema keys absent')
    if receipt['status']!=200 or receipt['sha256']!=sha(raw) or receipt['bytes']!=len(raw): raise ValueError('Receipt raw integrity mismatch')
    if receipt['url']!=meta['source_url'] or receipt['final_url']!=meta['source_url']: raise ValueError('Receipt URL or redirect mismatch')
    if urlparse(receipt['url']).hostname!='news.abbvie.com': raise ValueError('Unregistered source host')
    if receipt['ticker']!='ABBV' or receipt['source_representation']!='original_http_bytes': raise ValueError('Receipt identity/representation mismatch')
    if receipt['request_method']!='GET' or receipt['request_payload'] is not None or not receipt['content_type'].lower().startswith('text/html'): raise ValueError('Receipt transport mismatch')
    stamp=datetime.fromisoformat(receipt['retrieved_at'].replace('Z','+00:00'))
    if stamp.utcoffset() is None: raise ValueError('Naive retrieval clock')
    if stamp.date().isoformat()<meta['publication_date']: raise ValueError('Retrieval precedes release')

def publication(soup,meta,receipt,event,sessions,calendar_start):
    tags=soup.select('.wd_date')
    if len(tags)!=1: raise ValueError('Ambiguous publication date element')
    literal=txt(tags[0]); day=datetime.strptime(literal,'%B %d, %Y').date().isoformat()
    if day!=meta['publication_date'] or not urlparse(meta['source_url']).path.startswith('/'+day+'-'): raise ValueError('Publication date/URL mismatch')
    if event['release_date']!=day or event['fiscal_period_end_actual']!=meta['official_actual_period_end']: raise ValueError('Publication event mismatch')
    if event['evidence_sha256']!=meta['sha256'] or event['source_url']!=meta['source_url']: raise ValueError('Publication event source mismatch')
    if event['precision']!='date_only' or event['publication_timestamp'] is not None: raise ValueError('Unexpected event clock')
    available=layout.first_session_after(day,sessions,calendar_start)
    if available is not None and available!=meta['available_session']: raise ValueError('Pinned session/census mismatch')
    return {'release_date':day,'date_literal':literal,'date_element_html_sha256':sha(str(tags[0]).encode()),
            'precision':'date_only','publication_timestamp':None,
            'publication_timezone':event['publication_timezone'],'publication_timezone_basis':event['publication_timezone_basis'],
            'available_session':available,'catalog_available_session':meta['available_session'],
            'availability_basis':'next pinned session strictly after date' if available else 'held: before pinned calendar start',
            'retrieved_at':receipt['retrieved_at'],'first_public_time_verified':False,'historical_first_public_bytes_verified':False,
            'vintage_policy':'current and comparative facts share this current-release clock; comparative is never backdated'}

def revenue_cells(soup,year,planned):
    actual=statement_metadata(soup,year)
    if actual!=planned: raise ValueError('Frozen statement metadata changed')
    t=soup.find_all('table')[planned['table_index']]
    rows=[r for r in t.find_all('tr') if r.find_parent('table') is t]
    row=cells(rows[planned['row_index']]); result={}
    for y,group in planned['groups'].items():
        numbers=[]; dollar_cells=[]
        for g in group:
            c=row[g['physical_cell_index']]; text=txt(c)
            if text=='$': dollar_cells.append(g['physical_cell_index']); continue
            if text=='': continue
            # Every nonempty annual cell must be an exact amount; no footnote stripping or percentages.
            value=layout.exact_amount(text)
            numbers.append((g,c,text,value))
        if len(numbers)!=1: raise ValueError('Annual group must contain exactly one numeric cell: '+y)
        g,c,text,value=numbers[0]
        if len(dollar_cells)>1: raise ValueError('Multiple currency marker cells')
        if '$' not in text and not dollar_cells: raise ValueError('Annual amount has no local dollar marker')
        start,end=date(int(y),1,1),date(int(y),12,31)
        result['current' if int(y)==year else 'comparative']={
            'measure':'consolidated_total_net_revenue','measure_label':planned['label'],
            'displayed_amount':text,'amount_decimal':value,'scale':1000000,
            'unit_literal':planned['unit_literal'],'currency_literal':'$','currency_iso':None,
            'currency_status':'HOLD_BARE_DOLLAR_NOT_ISO_CERTIFICATION','currency_marker_physical_cells':dollar_cells,
            'period_start':start.isoformat(),'period_end':end.isoformat(),'days_inclusive':(end-start).days+1,
            'duration_literal':g['duration_literal'],'period_start_is_direct_quote':False,
            'period_derivation':'January 1 derived from explicit Twelve Months Ended December/Dec. 31 annual header and standalone year header',
            'header_literals':g['header_literals'],
            'locator':{'table_index_zero_based':planned['table_index'],'row_index_zero_based':planned['row_index'],
                       'physical_cell_index_zero_based':g['physical_cell_index'],'logical_columns_zero_based':g['logical_columns']},
            'cell_html':str(c),'cell_html_sha256':sha(str(c).encode()),
            'row_html_sha256':sha(str(rows[planned['row_index']]).encode()),'table_html_sha256':planned['table_html_sha256'],
            'fact_role':'original_current_year' if int(y)==year else 'same_current_report_comparative',
            'fact_year':int(y),'revenue_admitted':False,'accounting_chain_admitted':False}
    return result

def note_contexts(soup):
    text=txt(soup)
    result=[]
    for m in NOTE.finditer(text):
        a=max(0,m.start()-220); b=min(len(text),m.end()+420)
        result.append({'keyword':m.group(0),'match_start_normalized_text':m.start(),'match_end_normalized_text':m.end(),
                       'excerpt_start_normalized_text':a,'excerpt_end_normalized_text':b,'excerpt':text[a:b],
                       'interpretation':'unclassified accounting/scope lead; neither a confirmed restatement nor comparability clearance'})
    explicit=[{'match':m.group(0),'start_normalized_text':m.start(),'excerpt':text[max(0,m.start()-150):m.end()+150]}
              for m in re.finditer(r'U\.?S\.? dollars|United States dollars|\bUSD\b',text,re.I)]
    gaap=re.search(r'Reported results were prepared in accordance with GAAP and include all revenue and expenses recognized during the period\.',text,re.I)
    return {'normalized_document_text_sha256':sha(text.encode()),'normalization':'BeautifulSoup html.parser get_text(space,strip=True), whitespace collapsed',
            'accounting_boundary_contexts':result,'all_keyword_matches_retained':True,
            'explicit_currency_phrase_leads':explicit,'currency_assignment_from_phrase_automated':False,
            'reported_gaap_policy':None if gaap is None else {'excerpt':gaap.group(0),'start_normalized_text':gaap.start()},
            'accounting_boundary_status':'HOLD_INDEPENDENT_REVIEW_REQUIRED','comparability_asserted':False}

def freeze():
    scope=load(SCOPE);assert_inputs(scope)
    tests=load(TEST)
    if not tests['successful'] or tests['script']['sha256']!=audit.bind(SELF)['sha256']: raise ValueError('Current fabricated tests absent')
    sources=[]
    for s in scope['sources']:
        raw=audit.read(s['raw']);receipt=json.loads(audit.read(s['receipt']))
        receipt_check(raw,receipt,s['metadata'])
        profile=statement_metadata(BeautifulSoup(raw,'html.parser'),s['metadata']['fiscal_year'])
        sources.append({'source':s,'statement_profile':profile,'receipt_schema_keys':sorted(receipt)})
    record={'created_utc':now(),'stage':'FROZEN_AFTER_METADATA_AUDIT_BEFORE_MONETARY_EXTRACTION',
            'symbol':'ABBV','fiscal_years':list(range(2013,2026)),
            'bindings':[audit.bind(p) for p in [SELF,SCOPE,TEST,OUT+'LAYOUT_AUDIT.json',CALENDAR,SESSIONS]],'sources':sources,
            'policy':{'amounts':'Exact displayed decimal with separate scale. No arithmetic ratios, cross-year equality, features, prices, models or strategy admission.',
                      'selection':'Unique AbbVie consolidated statements of earnings, exact Net sales 2013–2014 / Net revenues 2015–2025 row; exact standalone year and Twelve Months Ended December/Dec.31 logical-column ownership. Reject duplicated labels/tables, ambiguous spans, percentages and nonnumeric notes.',
                      'mixed_measures':'Do not mistake separately labeled adjusted EPS beneath the statement for adjusted revenue. Reject adjusted/non-GAAP/pro forma statement heading; do not consume those other rows.',
                      'period':'Derive Jan1 start only from exact annual calendar-end column; retain 365/366 days.',
                      'currency':'Dollar literal only. No inferred ISO currency. Any explicit currency phrase is a review lead, not automated assignment.',
                      'clock':'Both facts receive current report date-only publication clock. Current archive bytes not contemporaneous first-public certification. Calendar prior to 2015-01-01 remains held.',
                      'boundaries':'All bounded normalized-text accounting/scope keyword contexts retained, including revised/recast/retrospective/reclass/adoption/separation/acquisition/discontinued/pro-forma terms. No comparability inference from consecutive years.',
                      'receipt':'Actual status=200 schema; raw byte/hash, exact URL and final URL, news.abbvie.com host, ABBV ticker, GET/no payload, original_http_bytes, HTML and timezone-aware retrieval verified.',
                      'earlier_layout_audit_limit':'LAYOUT_AUDIT included incidental numeric text in nonselected reconciliation pre-label rows. It made no monetary conversion or fact selection; only the formal table metadata below governs extraction.',
                      'admission':'All financial, strategy and chain admission flags remain false; explicit holds are required.'}}
    print(json.dumps(audit.write('EXTRACTION_PROFILE.json',record)))

def extract():
    profile=load(PROFILE);scope=load(SCOPE);assert_inputs(scope)
    for b in profile['bindings']: audit.read(b)
    events=load(audit.EVENTS); cal=load(CALENDAR); sessions_obj=load(SESSIONS)
    if sessions_obj['plan_sha256']!=audit.bind(CALENDAR)['sha256']: raise ValueError('Calendar plan binding mismatch')
    for p,h in cal['bindings'].items():
        if audit.bind(p)['sha256']!=h: raise ValueError('Calendar dependency changed: '+p)
    reports=[]
    for entry in profile['sources']:
        s=entry['source'];meta=s['metadata'];year=meta['fiscal_year']
        record={'symbol':'ABBV','fiscal_year':year,'source':s,'extraction_status':'HOLD','revenue_admitted':False,
                'accounting_chain_admitted':False,'financial_features_computed':False,'strategy_admitted':False}
        try:
            raw=audit.read(s['raw']);receipt=json.loads(audit.read(s['receipt']));receipt_check(raw,receipt,meta)
            soup=BeautifulSoup(raw,'html.parser')
            matches=[e for e in events if e.get('ticker')=='ABBV' and e.get('fiscal_year')==year and e.get('fiscal_quarter')==4]
            if len(matches)!=1: raise ValueError('Annual event absent or duplicated')
            clock=publication(soup,meta,receipt,matches[0],sessions_obj['sessions'],cal['start'])
            facts=revenue_cells(soup,year,entry['statement_profile'])
            notes=note_contexts(soup)
            if notes['reported_gaap_policy'] is None: raise ValueError('Reported GAAP policy absent')
            for fact in facts.values(): fact['publication_clock']=clock;fact['source_raw_sha256']=s['raw']['sha256']
            holds=['CURRENCY_ISO_UNRESOLVED','ACCOUNTING_SCOPE_NOT_INDEPENDENTLY_REVIEWED','HISTORICAL_FIRST_PUBLIC_BYTES_NOT_VERIFIED']
            if clock['available_session'] is None: holds.append('PUBLICATION_PRECEDES_PINNED_CALENDAR')
            record.update({'extraction_status':'EXACT_REPORTED_PAIR_EXTRACTED_ADMISSION_HELD','revenue':facts,
                           'publication_clock':clock,'receipt_verified':True,'statement_profile':entry['statement_profile'],
                           'context':notes,'holds':holds})
        except (ValueError,KeyError,IndexError) as e:
            record['holds']=['SOURCE_EXTRACTION_FAILURE'];record['error']=str(e)
        reports.append(record)
    successful=sum('revenue' in r for r in reports)
    summary={'stage':'SOURCE_ONLY_EXTRACTION_COMPLETE','created_utc':now(),'profile':audit.bind(PROFILE),
             'source_count':len(reports),'exact_report_pairs':successful,'exact_amount_cells':successful*2,
             'failed_sources':len(reports)-successful,'currency_certified':0,'revenue_admitted':0,'accounting_chains_admitted':0,
             'strategy_trials_added':0,'returns_or_prices_read':False,'reports':reports}
    print(json.dumps(audit.write('EXTRACTION.json',summary)))
    print(json.dumps({'source_count':len(reports),'exact_pairs':successful,'failures':[{'year':r['fiscal_year'],'error':r.get('error')} for r in reports if 'revenue' not in r]}))

def fabricated(year=2024,extra='',wide=False):
    # Deliberately invented values: no real report values enter fixtures.
    span=2 if wide else 1
    amt=lambda a: '<td>$</td><td>'+a+'</td>' if wide else '<td>$'+a+'</td>'
    return '<html><div class="wd_date">January 31, 2025</div><table><tr><td colspan="'+str(1+span*4)+'">AbbVie Inc. Consolidated Statements of Earnings</td></tr><tr><td>(in millions, except per share data)</td><td colspan="'+str(span*2)+'">Fourth Quarter Ended December 31</td><td colspan="'+str(span*2)+'">Twelve Months Ended December 31</td></tr><tr><td></td>'+''.join('<td colspan="'+str(span)+'">'+str(y)+'</td>' for y in [year,year-1,year,year-1])+'</tr><tr><td>'+('Net sales' if year<=2014 else 'Net revenues')+'</td>'+amt('1')+amt('2')+amt('3,456.780')+amt('2,345.670')+'</tr><tr><td>Cost of products sold</td><td>0</td></tr>'+extra+'</table></html>'

class Tests(unittest.TestCase):
    def parsed(self,s,year=2024):
        soup=BeautifulSoup(s,'html.parser');return revenue_cells(soup,year,statement_metadata(soup,year))
    def test_annual_not_quarter(self):
        f=self.parsed(fabricated());self.assertEqual(f['current']['amount_decimal'],'3456.780');self.assertEqual(f['comparative']['amount_decimal'],'2345.670')
    def test_split_currency_spans(self): self.assertEqual(self.parsed(fabricated(wide=True))['current']['locator']['logical_columns_zero_based'],[6])
    def test_leap_period(self): self.assertEqual(self.parsed(fabricated())['current']['days_inclusive'],366)
    def test_prior_period_full(self): self.assertEqual(self.parsed(fabricated())['comparative']['period_start'],'2023-01-01')
    def test_old_net_sales(self): self.assertEqual(self.parsed(fabricated(2013),2013)['current']['measure_label'],'Net sales')
    def test_dec_abbreviation(self): self.assertEqual(self.parsed(fabricated().replace('December','Dec.'))['current']['period_end'],'2024-12-31')
    def test_no_iso_inference(self): self.assertIsNone(self.parsed(fabricated())['current']['currency_iso'])
    def test_duplicate_table(self):
        with self.assertRaises(ValueError): self.parsed(fabricated()+fabricated())
    def test_duplicate_revenue(self):
        with self.assertRaises(ValueError): self.parsed(fabricated(extra='<tr><td>Net revenues</td><td>99</td></tr>'))
    def test_adjusted_heading(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('Statements of Earnings','Statements of Earnings (Adjusted)'))
    def test_adjusted_eps_is_separate(self): self.assertEqual(self.parsed(fabricated(extra='<tr><td>Adjusted diluted earnings per share</td><td>99</td></tr>'))['current']['amount_decimal'],'3456.780')
    def test_percentage_rejected(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('3,456.780','3.5%'))
    def test_footnoted_value_rejected(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('3,456.780','3,456.780 a'))
    def test_missing_year_rejected(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('2023','2022'))
    def test_wrong_duration(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('Twelve Months','Nine Months'))
    def test_wrong_scale(self):
        with self.assertRaises(ValueError): self.parsed(fabricated().replace('in millions','in thousands'))
    def test_frozen_table_mutation(self):
        s=BeautifulSoup(fabricated(),'html.parser');p=statement_metadata(s,2024)
        with self.assertRaises(ValueError): revenue_cells(BeautifulSoup(fabricated().replace('3,456.780','6,789.001'),'html.parser'),2024,p)
    def test_boundary_context_offsets(self):
        s=BeautifulSoup('<p>Prior periods were revised for revenue recognition after separation.</p>','html.parser'); n=note_contexts(s)
        self.assertFalse(n['comparability_asserted']);self.assertEqual(len(n['accounting_boundary_contexts']),3)
        for c in n['accounting_boundary_contexts']:self.assertEqual(txt(s)[c['match_start_normalized_text']:c['match_end_normalized_text']],c['keyword'])
    def test_current_clock_not_prior_clock(self):
        raw=b'fabricated';meta={'source_url':'https://news.abbvie.com/2025-01-31-Fabricated','publication_date':'2025-01-31','sha256':sha(raw),'official_actual_period_end':'2024-12-31','available_session':'2025-02-03'}
        event={'release_date':'2025-01-31','fiscal_period_end_actual':'2024-12-31','evidence_sha256':sha(raw),'source_url':meta['source_url'],'precision':'date_only','publication_timestamp':None,'publication_timezone':'America/Chicago','publication_timezone_basis':'fabricated'}
        clock=publication(BeautifulSoup(fabricated(),'html.parser'),meta,{'retrieved_at':'2026-01-01T00:00:00Z'},event,['2025-01-31','2025-02-03'],'2025-01-01')
        self.assertEqual(clock['release_date'],'2025-01-31'); self.assertEqual(clock['available_session'],'2025-02-03')
    def test_receipt_requires_actual_schema_and_host(self):
        raw=b'fabricated';meta={'source_url':'https://news.abbvie.com/a','publication_date':'2025-01-01'}
        rec={'url':meta['source_url'],'final_url':meta['source_url'],'status':200,'sha256':sha(raw),'bytes':len(raw),'retrieved_at':'2025-02-01T00:00:00Z','content_type':'text/html','ticker':'ABBV','role':'earnings_candidate','source_representation':'original_http_bytes','request_method':'GET','request_payload':None}
        receipt_check(raw,rec,meta)
        for k,v in [('status',404),('bytes',0),('final_url','https://other/a'),('ticker','ABT'),('retrieved_at','2025-02-01T00:00:00'),('source_representation','transformed_text')]:
            bad=dict(rec);bad[k]=v
            with self.assertRaises(ValueError):receipt_check(raw,bad,meta)

def tests():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(stream.getvalue())
    print(json.dumps(audit.write('FABRICATED_TESTS.json',{'created_utc':now(),'script':audit.bind(SELF),'successful':result.wasSuccessful(),
                   'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'fabricated_only':True,'output':stream.getvalue()})))
    if not result.wasSuccessful(): raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['test','freeze','extract']);args=p.parse_args()
    {'test':tests,'freeze':freeze,'extract':extract}[args.phase]()
