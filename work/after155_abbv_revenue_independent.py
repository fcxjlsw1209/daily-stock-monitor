"""Independent bounded ABBV revenue-cell review; no producer imports or ratios."""
from pathlib import Path
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
OUT='work/after155_abbv_revenue_independent.json'
EXTRACT='outputs/monthly_abbv_revenue156/EXTRACTION_V2.json'
PROFILE='outputs/monthly_abbv_revenue156/EXTRACTION_PROFILE_V2.json'
CALENDAR='outputs/monthly_paired_target143/CALENDAR.json'
EXPECTED={EXTRACT:'5c81e1e23eefda6118739b938cd06074119ad9083f827e499cb5567ffe8f93fd',PROFILE:'1262433e45dfb805631b0b24bfd47915536dcd57aec7443caab3ec4ad2e51723',CALENDAR:'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'}
PINS={}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()

def read(path,expected=None,size=None):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and (p==ROOT or ROOT in p.parents),'Nonlocal or redirected source')
    raw=p.read_bytes();h=digest(raw)
    require(expected is None or h==expected,'Changed source '+str(p))
    require(size is None or len(raw)==size,'Changed size '+str(p))
    key=str(p.relative_to(ROOT));require(key not in PINS or PINS[key]==h,'Changed input during audit '+key)
    PINS[key]=h;return raw

def bound(b):return read(b['path'],b['sha256'],b.get('bytes'))
def load(path):return json.loads(read(path,EXPECTED.get(path)))
def norm(t):return ' '.join(t.get_text(' ',strip=True).split())
def direct(row):return row.find_all(['td','th'],recursive=False)

def grid(table):
    rows=[r for r in table.find_all('tr') if r.find_parent('table') is table]
    cells={};positions={}
    for i,row in enumerate(rows):
        col=0
        for j,cell in enumerate(direct(row)):
            while (i,col) in cells:col+=1
            width=int(cell.get('colspan',1));height=int(cell.get('rowspan',1))
            require(width>0 and height>0,'Invalid table span')
            positions[i,j]=list(range(col,col+width))
            for a in range(i,i+height):
                for b in range(col,col+width):
                    require((a,b) not in cells,'Overlapping table spans')
                    cells[a,b]=(i,j,cell)
            col+=width
    return rows,cells,positions

def headers(cells,row,col):
    out=[];seen=set()
    for i in range(row):
        value=cells.get((i,col))
        if value is not None and value[:2] not in seen:
            seen.add(value[:2]);text=norm(value[2])
            if text:out.append(text)
    return out

def parse_amount(text):
    clean=text.replace('$','').strip()
    require(re.fullmatch(r'-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?',clean) is not None,'Ambiguous amount')
    result=Decimal(clean.replace(',',''));require(result.is_finite(),'Nonfinite source amount')
    return format(result,'f')

def statement(soup,year):
    selected=[]
    for ti,table in enumerate(soup.find_all('table')):
        rows,cells,positions=grid(table)
        labels=[(i,j,norm(c)) for i,r in enumerate(rows) for j,c in enumerate(direct(r)) if norm(c).lower() in ('net sales','net revenues')]
        if not labels:continue
        ri=labels[0][0];head=' '.join(norm(r) for r in rows[:ri])
        if not ('AbbVie Inc.' in head and 'Consolidated Statements of Earnings' in head):continue
        require(len(labels)==1 and labels[0][1]==0,'Ambiguous formal revenue row')
        require(not re.search(r'non[ -]?gaap|\badjusted\b|reconciliation|pro forma',head,re.I),'Adjusted statement header')
        require(norm(direct(rows[ri])[0]).lower()==('net sales' if year<=2014 else 'net revenues'),'Wrong total-revenue label')
        require(any(norm(c).lower()=='cost of products sold' for r in rows for c in direct(r)),'Missing earnings-statement context')
        require(len(re.findall(r'\(in millions, except per share data\)',head,re.I))==1,'Ambiguous millions unit')
        selected.append((ti,table,rows,cells,positions,ri,head))
    require(len(selected)==1,'Nonunique annual earnings statement')
    return selected[0]

def main():
    require(not (ROOT/OUT).exists(),'Independent review is write-once')
    data=load(EXTRACT);profile=load(PROFILE);calendar=load(CALENDAR)
    prior=load('outputs/monthly_abbv_revenue156/EXTRACTION_PROFILE.json')
    scope=load('outputs/monthly_abbv_revenue156/SOURCE_SCOPE.json')
    for doc in (profile,prior):
        for b in doc['bindings']:bound(b)
    for b in scope['inputs']:bound(b)
    for source in scope['sources']:bound(source['raw']);bound(source['receipt'])
    for p,h in calendar['provenance']['source_hashes'].items():read(p,h)
    spec=importlib.util.spec_from_file_location('abbv_independent_frozen_calendar',ROOT/'outputs/monthly_earnings_repair/research_calendar.py')
    calmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(calmod)
    market=calmod.market_calendar(calendar['start'],calendar['end'])
    sessions=calendar['sessions'];require(sessions==sorted(set(sessions)),'Invalid frozen sessions')
    require([str(v.date()) for v in market.sessions]==sessions,'Rebuilt frozen calendar differs')
    require([r['fiscal_year'] for r in data['reports']]==list(range(2013,2026)),'Exact13 source years')
    require(data['exact_amount_cells']==26 and data['failed_sources']==0,'Unexpected extraction scope')
    metadata={r['fiscal_year']:r for r in profile['metadata']}
    output=[];clock_gaps=[];context_count=0
    for r in data['reports']:
        year=r['fiscal_year'];source=r['source'];meta=source['metadata'];raw=bound(source['raw']);receipt=json.loads(bound(source['receipt']))
        soup=BeautifulSoup(raw,'html.parser');whole=norm(soup)
        require(receipt['status']==200 and receipt['sha256']==digest(raw) and receipt['bytes']==len(raw),'Receipt integrity')
        url=meta['source_url'];parsed=urlparse(url)
        require(parsed.scheme=='https' and parsed.hostname=='news.abbvie.com','Issuer HTTPS provenance')
        require(receipt['url']==receipt['final_url']==url and receipt['ticker']=='ABBV','Receipt identity')
        require(receipt['source_representation']=='original_http_bytes' and receipt['request_method']=='GET' and receipt['request_payload'] is None,'Receipt representation')
        require(receipt['content_type'].lower().startswith('text/html'),'Receipt MIME')
        retrieved=datetime.fromisoformat(receipt['retrieved_at'].replace('Z','+00:00'));require(retrieved.utcoffset() is not None,'Retrieval timezone')
        dates=soup.select('.wd_date');require(len(dates)==1,'Unique publication date')
        release=datetime.strptime(norm(dates[0]),'%B %d, %Y').date().isoformat()
        require(release==meta['publication_date']==r['publication_clock']['release_date'] and parsed.path.startswith('/'+release+'-'),'Date/source disagreement')
        require(retrieved.date().isoformat()>=release,'Retrieval predates release')
        require(digest(str(dates[0]).encode())==r['publication_clock']['date_element_html_sha256'],'Date element hash')
        canon=metadata[year]['canonical_event_fields'];require(canon['evidence_sha256']==digest(raw) and canon['source_url']==url and canon['publication_date']==release,'Canonical date evidence')
        require(meta['official_actual_period_end']==f'{year}-12-31' and canon['official_actual_period_end']==f'{year}-12-31','Actual fiscal end')
        available=next(s for s in sessions if s>release)
        require(available==meta['available_session']==canon['available_session'],'Independent next-session discrepancy')
        close=market.session_close(available).isoformat();require(close==canon['known_by_at'],'Independent close-clock discrepancy')
        original_clock=r['publication_clock']['available_session']
        require(original_clock==available or (year==2013 and original_clock is None),'Prepared clock discrepancy')
        if original_clock is None:clock_gaps.append({'fiscal_year':year,'prepared_available_session':None,'independently_verified_session':available,'independently_verified_close':close,'original_extraction_unchanged':True,'integration_pending':True})
        require(r['publication_clock']['precision']=='date_only' and r['publication_clock']['publication_timestamp'] is None,'Unwarranted exact publication clock')
        gaap=metadata[year]['reported_gaap_policy'];require(whole[gaap['start_normalized_text']:gaap['end_normalized_text']]==gaap['excerpt'],'GAAP context source mismatch')
        require(re.fullmatch(r'Reported results were prepared in accordance with (?:GAAP|generally accepted accounting principles in the United States \(GAAP\)) and include all revenue and expenses recognized during the period\.',gaap['excerpt'],re.I) is not None,'Missing explicit reported GAAP policy')
        ti,table,rows,cells,positions,ri,head=statement(soup,year)
        require(ti==r['statement_profile']['table_index'] and ri==r['statement_profile']['row_index'],'Independent statement selection mismatch')
        verified={}
        for role,target in (('current',year),('comparative',year-1)):
            candidates=[];markers=[]
            for j,c in enumerate(direct(rows[ri])):
                owned=[]
                for col in positions[ri,j]:
                    hs=headers(cells,ri,col)
                    years=[x for x in hs if re.fullmatch(r'[0-9]{4}',x)]
                    annual=[x for x in hs if re.fullmatch(r'Twelve Months Ended (?:December|Dec\.) 31',x,re.I)]
                    if years==[str(target)] and len(annual)==1:owned.append((col,hs,annual[0]))
                if not owned:continue
                require(len(owned)==len(positions[ri,j]),'Ambiguous annual span')
                text=norm(c)
                if not text:continue
                if text=='$':markers.append(j);continue
                candidates.append((j,c,text,parse_amount(text),owned))
            require(len(candidates)==1,'Unique annual amount required')
            j,c,text,amount,owned=candidates[0];f=r['revenue'][role]
            require('$' in text or len(markers)==1,'No own-column dollar evidence')
            require(len(markers)<=1 and markers==f['currency_marker_physical_cells'],'Currency marker geometry')
            require(text==f['displayed_amount'] and amount==f['amount_decimal'],'Independent exact monetary cell mismatch')
            require(f['locator']=={'table_index_zero_based':ti,'row_index_zero_based':ri,'physical_cell_index_zero_based':j,'logical_columns_zero_based':positions[ri,j]},'Cell geometry mismatch')
            require(f['cell_html']==str(c) and f['cell_html_sha256']==digest(str(c).encode()) and f['row_html_sha256']==digest(str(rows[ri]).encode()) and f['table_html_sha256']==digest(str(table).encode()),'Physical cell/table hashes')
            require(f['period_start']==f'{target}-01-01' and f['period_end']==f'{target}-12-31' and f['days_inclusive']==(date(target,12,31)-date(target,1,1)).days+1,'Annual duration mismatch')
            require(f['scale']==1000000 and f['currency_literal']=='$' and f['currency_iso'] is None,'Scale/currency classification')
            require(f['publication_clock']==r['publication_clock'] and f['source_raw_sha256']==digest(raw),'Current-report clock/source binding')
            require(f['fact_year']==target and f['fact_role']==('original_current_year' if role=='current' else 'same_current_report_comparative'),'Fiscal vintage mismatch')
            require(f['revenue_admitted'] is False and f['accounting_chain_admitted'] is False,'Unwarranted source admission')
            verified[role]={'cell_sha256':f['cell_html_sha256'],'table_index':ti,'row_index':ri,'physical_cell_index':j,'fiscal_year':target,'exact_amount_verified':True,'scale':1000000,'period_verified':True,'own_annual_column_dollar_evidence_verified':True}
        for c in r['context']['accounting_boundary_contexts']:
            require(whole[c['excerpt_start_normalized_text']:c['excerpt_end_normalized_text']]==c['excerpt'] and whole[c['match_start_normalized_text']:c['match_end_normalized_text']]==c['keyword'],'Accounting context offset mismatch')
            context_count+=1
        require(digest(whole.encode())==r['context']['normalized_document_text_sha256'],'Normalized context hash')
        output.append({'fiscal_year':year,'source_path':source['raw']['path'],'source_sha256':digest(raw),'receipt_verified':True,'release_date':release,'original_prepared_available_session':original_clock,'independent_available_session':available,'independent_known_by_close':close,'cells':verified,'reported_gaap_policy_verified':True,'accounting_context_offsets_verified':True,'accounting_comparability_admitted':False,'original_holds_preserved':r['holds']})
    require(len(clock_gaps)==1 and clock_gaps[0]['fiscal_year']==2013,'Unexpected clock-gap scope')
    read(str(Path(__file__)))
    for p,h in list(PINS.items()):require(digest(read(p))==h,'Changed input after source verification')
    result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'stage':'BOUNDED_ACTUAL_SOURCE_CELL_AND_CLOCK_REVIEW','input_hashes':dict(PINS),'actual_checks':{'source_reports':13,'annual_pairs':13,'monetary_cells':26,'source_and_receipt_integrity':True,'formal_gaap_statement_selection_independent':True,'annual_column_geometry_independent':True,'exact_amount_strings_independent':True,'period_scale_dollar_evidence_verified':True,'release_dates_and_calendar_closes_independent':True,'context_offsets_verified':context_count,'producer_imports':0},'reports':output,'clock_gaps':clock_gaps,'currency':{'class':'dollar_literal_only','iso_certified':False,'within_issuer_assumption_admitted_by_this_review':False,'note':'Each selected annual value has dollar evidence in its own column; a separately bound reported-dollar assumption remains required.'},'scope_limits':['The2013 original source date and2014-02-03 close are independently verified using frozen143 calendar; the extraction retains its null clock and integration remains pending.','Accounting keyword excerpts were checked against raw text but no positive inter-report accounting edge was admitted.','Current retrieved original-dated sources do not certify first-public-version bytes or exhaustive amendments.','No cross-year monetary comparison, revenue growth, source-chain grid, factor, return, price or model was calculated.'],'admission':{'source_cells_verified':True,'prepared_all_clocks_resolved':False,'accounting_edges':False,'source_chains':False,'financial_factors':False,'strategy':False,'global_completeness':False},'activity_counts':{'actual_source_cells_verified':26,'new_revenue_ratios':0,'new_source_chain_rows':0,'new_factors':0,'new_returns':0,'new_models':0,'new_trials':0,'observer_writes':0}}
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with (ROOT/OUT).open('xb') as f:f.write(raw)
    print(json.dumps({'path':OUT,'sha256':digest(raw),'passed':True,'reports':13,'cells':26,'context_offsets_verified':context_count,'supplemental_early_clock':clock_gaps}))

if __name__=='__main__':main()
