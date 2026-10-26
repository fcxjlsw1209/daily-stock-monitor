"""Independent original Apple revenue-source review; no producer imports."""
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urljoin, urlparse, unquote
import sys, json, hashlib, re, io, importlib.util
from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
from pypdf import PdfReader

BASE = 'outputs/monthly_aapl_revenue156/v4/'
OUT = 'work/after155_aapl_revenue_independent.json'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
EXPECTED = {
    BASE+'PLAN.json':'3c841c5d80659e5f1911035e027518e51a788f7b7e5afba529d15a0324e3be6b',
    BASE+'DATASET.json':'24eb0618ea6718e074ae8c5eafd7ee1d3abcfe970b895ee256731ff527936e20',
    BASE+'RECEIPT.json':'c6d4d1578b02fac01bec8f566dec0e7795c6b29baa718a745d5bf4b61fa3733c',
    CALENDAR:'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
}
PINS = {}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, expected=None, size=None):
    p=Path(path)
    p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p, 'Redirected path: '+str(p))
    require(ROOT in p.parents or BUNDLE in p.parents, 'Unscoped path: '+str(p))
    raw=p.read_bytes(); h=digest(raw)
    require(expected is None or h==expected, 'Input SHA mismatch: '+str(p))
    require(size is None or size==len(raw), 'Input size mismatch: '+str(p))
    key=str(p.relative_to(ROOT)) if ROOT in p.parents else str(p)
    require(key not in PINS or PINS[key]==h, 'Input changed during review: '+key)
    PINS[key]=h
    return raw

def bound(b):
    return read(b['path'],b['sha256'],b.get('bytes'))

def load(path):
    return json.loads(read(path,EXPECTED.get(path)))

def norm(s):
    return ' '.join(s.split())

def letters(s):
    return r'\s*'.join(re.escape(c) for c in s.replace(' ',''))

DATE = r'(September)\s+(\d{1,2}),\s*(20\d{2})'
AMOUNT = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)'

def dates_in(s):
    return [datetime.strptime(' '.join(m),'%B %d %Y').date().isoformat() for m in re.findall(DATE,s)]

def statement(text):
    """Extract on content-order text; do not use producer's line/geometry locators."""
    text=norm(text).replace('/dollarsign','$')
    title='CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS'
    require(text.count(title)==1 and 'Apple Inc.' in text[:text.index(title)],'Issuer/statement title')
    require('Unaudited' in text or 'UNAUDITED' in text,'Unaudited statement scope')
    cost=re.search(letters('Cost of sales'),text,re.I)
    require(cost is not None,'Cost-of-sales boundary')
    scope=text[:cost.start()]
    marker=re.search(r'Net\s+sales',scope,re.I)
    require(marker is not None,'Revenue start')
    header=scope[:marker.start()]
    require('Three Months Ended' in header and 'Twelve Months Ended' in header,'Annual and quarterly headers')
    require(header.index('Three Months Ended')<header.index('Twelve Months Ended'),'Header ordering')
    require('In millions, except number of' in header,'Millions denomination')
    dates=dates_in(header)
    require(len(dates)==4 and dates[:2]==dates[2:],'Four annual/quarter dates')
    revenue=scope[marker.start():]
    totals=list(re.finditer(letters('Total net sales')+r'(?:\s*\(1\))?\s*(.*)$',revenue,re.I))
    if totals:
        require(len(totals)==1,'Unique total row before cost')
        row=totals[0].group(1)
        products=re.search(r'Products\s+(.*?)\s+Services',revenue,re.I)
        require(products is not None and products.group(1).count('$')==4,'Dollar marks in own statement columns')
        dollar_scope='products_row_same_statement_four_columns'
    else:
        match=re.fullmatch(r'Net\s+sales\s+(.*)',revenue,re.I)
        require(match is not None,'Whole total net sales row')
        row=match.group(1)
        require(row.count('$')==4,'Dollar marks in total row four columns')
        dollar_scope='total_row_four_columns'
    amounts=re.findall(AMOUNT,row)
    require(len(amounts)==4,'Exactly four total revenue amounts')
    require(re.sub(AMOUNT,'',row).replace('$','').strip()=='','Unknown text in total row')
    return {'amounts':amounts,'dates':dates,'dollar_scope':dollar_scope,'header':header,'row':row}

def main():
    require(not (ROOT/OUT).exists(),'Review is write-once')
    plan=load(BASE+'PLAN.json'); receipt=load(BASE+'RECEIPT.json'); data=load(BASE+'DATASET.json'); calendar=load(CALENDAR)
    bound(receipt['dataset']);bound(receipt['source_checks']);bound(receipt['tests'])
    require(receipt['plan_sha256']==EXPECTED[BASE+'PLAN.json']==data['plan_sha256'],'Plan binding')
    for binding in plan['inputs'].values(): bound(binding)
    for path,h in plan['pdf_runtime_hashes'].items(): read(path,h)
    for path,h in calendar['provenance']['source_hashes'].items(): read(path,h)
    spec=importlib.util.spec_from_file_location('aapl_independent_calendar',ROOT/'outputs/monthly_earnings_repair/research_calendar.py')
    calmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(calmod)
    market=calmod.market_calendar(calendar['start'],calendar['end'])
    sessions=[str(d.date()) for d in market.sessions]
    require(sessions==calendar['sessions'],'Frozen calendar rebuild')
    require([r['fiscal_year'] for r in data['rows']]==list(range(2016,2026)),'Ten original years')
    sources={r['fiscal_year']:r for r in plan['sources']}
    results=[]; prior=None; note_count=0
    for r in data['rows']:
        year=r['fiscal_year']; declared=sources[year]
        for key,other in [('source','raw'),('receipt','receipt'),('parent_release','parent_release'),('parent_receipt','parent_receipt')]:
            require(r[key]==declared[other], 'Declared source mapping')
        raw=bound(r['source']); pdfmeta=json.loads(bound(r['receipt']))
        html=bound(r['parent_release']); parentmeta=json.loads(bound(r['parent_receipt']))
        require(raw.startswith(b'%PDF'),'Original PDF representation')
        require(pdfmeta['status']==200 and pdfmeta['content_type']=='application/pdf','PDF HTTP metadata')
        require(pdfmeta['response_sha256']==digest(raw)==pdfmeta['decoded_sha256'] and pdfmeta['response_bytes']==len(raw)==pdfmeta['decoded_bytes'],'PDF byte integrity')
        require(pdfmeta['decoder']=='identity' and pdfmeta['public_unauthenticated'] is True,'PDF representation/authentication')
        require(pdfmeta['url']==pdfmeta['final_url']==r['source_url']==declared['source_url'],'PDF URL binding')
        require(urlparse(r['source_url']).hostname=='www.apple.com' and urlparse(r['source_url']).scheme=='https','Original issuer PDF host')
        require(parentmeta['http_status']==200 and parentmeta['sha256']==digest(html) and parentmeta['size']==len(html),'Parent source integrity')
        require(parentmeta['source_url']==parentmeta['final_url'] and urlparse(parentmeta['source_url']).hostname=='www.apple.com','Parent issuer host')
        soup=BeautifulSoup(html,'html.parser')
        visible=soup.select('.category-eyebrow__date')
        require(len(visible)==1,'Unique visible parent date')
        visible=norm(visible[0].get_text())
        released=datetime.strptime(visible,'%B %d, %Y').date().isoformat()
        published=set(v[:10] for v in re.findall(r'"datePublished"\s*:\s*"([^\"]+)"',html.decode()))
        require(published=={released},'Structured publication date')
        require(released==r['publication']['release_date']==declared['release_date'],'Recorded publication date')
        links={unquote(urljoin(parentmeta['source_url'],a.get('href',''))) for a in soup.find_all('a')}
        require(unquote(r['source_url']) in links,'Direct original PDF link')
        available=next(s for s in sessions if s>released)
        require(available==r['availability']['candidate_first_usable_session'],'Conservative next-session date')
        require(r['publication']['precision']=='date_only' and r['publication']['first_public_version_verified'] is False and r['publication']['historical_web_snapshot_verified'] is False,'Historical-vintage limitations')
        for meta in (pdfmeta,parentmeta):
            retrieved=datetime.fromisoformat(meta['retrieved_at'].replace('Z','+00:00'))
            require(retrieved.utcoffset() is not None and retrieved.date().isoformat()>=released,'Retrieval time')
        reader=PdfReader(io.BytesIO(raw))
        contents=[page.extract_text() for page in reader.pages]
        candidates=[(i,t) for i,t in enumerate(contents) if 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS' in norm(t)]
        require(len(candidates)==1,'Unique original operations page')
        page,content=candidates[0]; parsed=statement(content)
        balance=[(i,t) for i,t in enumerate(contents) if 'CONSOLIDATED BALANCE SHEETS' in norm(t)]
        require(len(balance)==1,'Unique original balance sheet')
        bp,bt=balance[0]; bt=norm(bt)
        require('ASSETS:' in bt,'Balance sheet assets boundary')
        ends=dates_in(bt[:bt.index('ASSETS:')])
        require(len(ends)==2 and parsed['dates'][2:]==ends,'Annual ends from original headers')
        end,previous_end=map(date.fromisoformat,ends)
        start=previous_end+timedelta(days=1); duration=(end-start).days+1
        require(end.year==year and previous_end.year==year-1 and duration in (364,371),'52/53-week fiscal year')
        require(end.weekday()==5 and 24<=end.day<=30 and end.month==9,'Last September Saturday')
        current=r['metadata']['current_period']
        require(current['start']==start.isoformat() and current['end']==end.isoformat() and current['days_inclusive']==duration and current['weeks']==duration//7,'Recorded current duration')
        require(current['start_is_direct_quote'] is False and current['prior_fiscal_end_explicit']==previous_end.isoformat(),'Derived start classification')
        comparative=r['metadata']['comparative_period']
        require(comparative['end']==previous_end.isoformat(),'Comparative end')
        if prior is None:
            require(comparative['start'] is None and comparative['weeks'] is None and comparative['days_inclusive'] is None,'Unknown FY2015 start preserved')
        else:
            require(prior['release_date']<released and prior['end']==previous_end.isoformat(),'Prior original chronological availability')
            require(comparative['start']==prior['start'] and comparative['days_inclusive']==prior['duration'] and comparative['weeks']==prior['duration']//7,'Prior own-year original supplies only period metadata')
            require(comparative['duration_source']==prior['source'] and comparative['duration_source_release_date']==prior['release_date'],'Prior-period provenance')
        # Layout text is checked only for producer locator integrity, not selection or numeric extraction.
        layout=[p.extract_text(extraction_mode='layout') for p in reader.pages]
        require([digest(t.encode()) for t in layout]==r['page_text_sha256'],'Page extraction digest')
        for role,column in [('current',2),('comparative',3)]:
            cell=r['revenue'][role]
            require(cell['displayed_amount']==parsed['amounts'][column] and cell['amount_decimal']==format(Decimal(parsed['amounts'][column].replace(',','')),'f'),'Independent exact revenue amount')
            require(cell['date']==parsed['dates'][column] and cell['header_dates']==parsed['dates'],'Independent annual column identity')
            require(cell['known_at_session']==available and cell['release_date']==released,'Each cell current-release availability')
            require(cell['period']==r['metadata'][role+'_period'],'Cell period metadata')
            require(cell['source']==r['source'] and cell['measure']=='total_net_sales' and cell['scale']==1000000,'Cell source/measure/scale')
            require(cell['currency_literal'] in ('$', '/dollarsign') and cell['currency_literal'] in content and cell['currency_evidence_class']=='dollar_literal_only' and cell['iso_currency'] is None and cell['ISO_currency_verified'] is False,'Dollar-only uncertainty preserved')
            require(cell['vintage_role']==('own_year_original' if role=='current' else 'same_current_release_prior_year_comparative'),'Comparative is not original')
            require(cell['historical_original_event_created']==(role=='current'),'Event vintage boundary')
            require(cell['fact_admitted'] is False and cell['strategy_admitted'] is False,'No premature fact admission')
            require(cell['locator']['page_1based']==page+1 and cell['locator']['financial_column_index_0based']==column,'Independent location selection')
            for textkey,hashkey in [('row_text','row_text_sha256')]:
                require(digest(cell[textkey].encode())==cell[hashkey] and cell[textkey] in layout[page],'Exact row text provenance')
            unit=cell['unit_literal']; require(unit in layout[page] and 'In millions' in unit,'Unit source provenance')
            evidence=cell['currency_evidence']; require(digest(evidence['text'].encode())==evidence['text_sha256'] and evidence['text'] in layout[page],'Dollar evidence provenance')
        for note in r['accounting_note_leads']:
            body=norm(layout[note['page_1based']-1])
            require(body[note['normalized_character_start']:note['normalized_character_end']]==note['excerpt'] and digest(note['excerpt'].encode())==note['excerpt_sha256'],'Exact accounting context provenance')
            require(note['known_at_session']==available and note['source']==r['source'] and note['confirmed_accounting_change'] is False,'Context clock and non-admission')
            note_count+=1
        require(r['fact_admitted'] is False and r['strategy_admitted'] is False and r['accounting_chain_admitted'] is False,'Original limits preserved')
        prior={'source':r['source'],'release_date':released,'start':start.isoformat(),'end':end.isoformat(),'duration':duration}
        results.append({'fiscal_year':year,'source_sha256':digest(raw),'parent_sha256':digest(html),'source_path':r['source']['path'],'operations_page':page+1,'balance_sheet_page':bp+1,'annual_columns':[2,3],'current_amount':parsed['amounts'][2],'comparative_amount':parsed['amounts'][3],'start':start.isoformat(),'end':end.isoformat(),'days':duration,'comparative_start':comparative['start'],'published_date':released,'available_session':available,'available_close':market.session_close(available).isoformat(),'currency_evidence_scope':parsed['dollar_scope'],'iso_currency_certified':False,'all_source_checks_passed':True})
    require(note_count==2,'Declared accounting-context count')
    require(receipt['monetary_cells_extracted']==20 and receipt['held_reports']==0 and receipt['original_reports_extracted']==10,'Producer scope receipt')
    read(str(Path(__file__)))
    for path,h in list(PINS.items()): read(path,h)
    result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'stage':'INDEPENDENT_ORIGINAL_SOURCE_CELL_AND_DATE_REVIEW','input_hashes':PINS,'actual_checks':{'original_reports':10,'annual_pairs':10,'monetary_cells':20,'independent_numeric_selector':'pypdf content-order original statement bounded before cost of sales; never uses producer amount or geometry to choose cells','producer_imports':0,'source_receipt_and_parent_link_integrity':True,'current_and_comparative_annual_columns':True,'calendar_rebuilt_and_publication_dates_verified':True,'full_page_digest_and_row_locator_provenance':True,'accounting_context_offsets':note_count,'comparative_start_unknown_fiscal_years':[2016],'current_53_week_fiscal_years':[x['fiscal_year'] for x in results if x['days']==371]},'reports':results,'findings':[],'limits':['Original-dated reports are currently retrieved bytes; initial-publication byte vintage and absence of later amendments are unverified.','The USD ISO currency is not certified; only reported dollar symbol evidence is verified. Within-issuer currency assumption remains a separate policy decision.','FY2016 comparison has unknown FY2015 start; do not infer or backfill it.','Accounting lead offsets are verified, not accepted as changes; two recasting substrings occur inside forecasting and cannot support continuity conclusions.','No inter-year accounting edge, continuous source chain, financial factor, trading rule or backtest is admitted by this review.'],'admission':{'source_cells_verified':True,'accounting_edges':False,'source_chains':False,'financial_factors':False,'strategy':False},'activity_counts':{'actual_source_cells_verified':20,'financial_ratios':0,'source_chain_rows':0,'new_factors':0,'new_returns':0,'new_models':0,'new_trials':0,'observer_writes':0,'network_requests':0}}
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with (ROOT/OUT).open('xb') as f: f.write(raw)
    print(json.dumps({'path':OUT,'sha256':digest(raw),'passed':True,'reports':10,'cells':20,'notes':note_count,'script_sha256':PINS[str(Path(__file__).relative_to(ROOT))]}))

if __name__=='__main__':
    main()
