"""Read-only source-context review; never parse financial cells or quantities."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
FEASIBILITY='work/after153_next_source_feasibility.json'
OUT='work/after153_batch5_accounting_feasibility_review.json'
EXPECTED={'PLTR':list(range(2020,2026)),'PYPL':list(range(2015,2020)),'TMO':list(range(2012,2022))}
TERMS=re.compile(r'spin-off|combined|carve.out|basis of presentation|historical|revenue recognition|lease accounting|pension|results in all periods|reclassif\w*|retrospect\w*|restat\w*|separat\w*|eBay|adopt\w*|accounting polic\w*|change in accounting|discontinued operations|noncontrolling interests|attributable to|U\.S\. GAAP|net income \(loss\)|net loss',re.I)

def require(ok,message):
    if not ok:raise ValueError(message)

def read(name,pins,expected=None):
    path=ROOT/name
    require(not path.is_symlink() and path.resolve().is_relative_to(ROOT),'Local regular input')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    require(expected is None or digest==expected,'Input digest '+name)
    require(name not in pins or pins[name]==digest,'Input changed '+name)
    pins[name]=digest
    return raw

def redact(text):
    return re.sub(r'\d[\d,.]*','[number]',text)

def document(doc,pins):
    raw=read(doc['evidence_path'],pins,doc['evidence_sha256'])
    receipt=doc['receipt'];r=json.loads(read(receipt['path'],pins,receipt['sha256']))
    require(r['sha256']==doc['evidence_sha256'] and r['url']==doc['source_url'] and r.get('status',r.get('http_status'))==200,'Report receipt')
    soup=BeautifulSoup(raw,'html.parser')
    for tag in soup(['script','style']):tag.decompose()
    text=' '.join(soup.get_text(' ',strip=True).split())
    quotes=[];seen=set()
    for match in TERMS.finditer(text):
        start=max(0,match.start()-180);end=min(len(text),match.end()+300)
        if any(abs(start-i)<120 for i in seen):continue
        seen.add(start)
        quotes.append(dict(term=match.group(),path=doc['evidence_path'],sha256=doc['evidence_sha256'],locator=dict(kind='normalized_document_text',start=start,end=end),quote_numbers_redacted=redact(text[start:end])))
    tables=[]
    raw_tables=soup.find_all('table')
    require(all(anchor in text for anchor in doc['identity_anchors']),'Raw identity anchor')
    for candidate in doc['context'].get('statement_candidates',[]):
        index=candidate['table'];table=raw_tables[index]
        rows=[r for r in table.find_all('tr') if r.find_parent('table') is table]
        prefix=' '.join(' '.join(r.get_text(' ',strip=True).split()) for r in rows[:min(7,candidate['header_stop_row']+1)])
        headers=candidate.get('headers',[])
        for head in headers:
            cells=rows[head['row']].find_all(['td','th'],recursive=False)
            require(' '.join(cells[head['physical_cell']].get_text(' ',strip=True).split())==head['text'],'Physical header match')
        tables.append(dict(table=index,metadata=candidate.get('external_metadata',[]),headers=headers,labels=candidate.get('labels',[]),raw_header_context_numbers_redacted=redact(prefix),header_years=sorted(set(re.findall(r'\b20[0-9]{2}\b',prefix))),annual_phrase_present=bool(re.search(r'years? ended|twelve months|full.year',prefix,re.I))))
    return dict(event_id=doc['event_id'],symbol=doc['symbol'],fiscal_year=doc['fiscal_year'],publication_date=doc['publication_date'],period_end=doc['official_actual_period_end'],clock_verified=doc['available_session_verified'],identity=doc['identity_anchors'],source=dict(path=doc['evidence_path'],sha256=doc['evidence_sha256'],url=doc['source_url'],receipt_path=receipt['path'],receipt_sha256=receipt['sha256']),tables=tables,evidence=quotes)

def scan(symbol=None):
    pins={};feas=json.loads(read(FEASIBILITY,pins))
    docs=[d for d in feas['documents'] if d['context'].get('statement_candidates')]
    require({s:[d['fiscal_year'] for d in docs if d['symbol']==s] for s in EXPECTED}==EXPECTED,'Fixed21 ordered candidates')
    selected=[document(d,pins) for d in docs if symbol is None or d['symbol']==symbol]
    return selected,pins,feas

# Reviewed source-table identities, not values or data-dependent selectors.
TABLES={('PLTR',2020):(15,18),('PLTR',2021):(39,43),
        **{('PLTR',y):(9,11) for y in range(2022,2026)},
        ('PYPL',2015):(3,2),**{('PYPL',y):(2,1) for y in range(2016,2019)},('PYPL',2019):(4,3),
        ('TMO',2012):(2,4),('TMO',2013):(2,4),('TMO',2014):(1,2),
        **{('TMO',y):(2,4) for y in range(2015,2018)},('TMO',2018):(3,6),
        ('TMO',2019):(2,4),('TMO',2020):(2,5),('TMO',2021):(5,9)}


def pick_evidence(row,patterns):
    selected=[]
    for pattern in patterns:
        candidates=[e for e in row['evidence'] if re.search(pattern,e['quote_numbers_redacted'],re.I)]
        if candidates:
            evidence=candidates[0]
            if evidence not in selected:selected.append(evidence)
    return selected


def conclusion(row):
    symbol,year=row['symbol'],row['fiscal_year']
    ni_idx,bs_idx=TABLES[(symbol,year)]
    tables={t['table']:t for t in row['tables']};ni,bs=tables[ni_idx],tables[bs_idx]
    unit='thousands' if symbol=='PLTR' else 'millions'
    unit_text=' '.join(ni['metadata']+bs['metadata'])+' '+ni['raw_header_context_numbers_redacted']+' '+bs['raw_header_context_numbers_redacted']
    require(unit in unit_text.lower(),'Declared source monetary unit')
    require(ni['annual_phrase_present'] and str(year) in bs['header_years'] and str(year-1) in bs['header_years'],'Annual/current+prior header context')
    findings=['Use the formal consolidated annual income statement, not quarterly, per-share, highlight, reconciliation or non-GAAP rows.',
              'Use total assets from the two December 31 endpoints in this same report; monetary columns and signs remain unverified.',
              'A January 1 start may be derived from the prior December 31 endpoint only under an explicit recorded policy; this review does not certify an unquoted start.']
    questions=[]
    patterns=[]
    if symbol=='PLTR':
        findings += ['GAAP net loss / net income (loss) / net income labels vary by report; preserve any negative sign and choose the year domain before testing numeric content.',
                     'Monetary scale is thousands; the share and per-share exceptions must not change the NI/assets scale.']
        if year>=2022:
            findings += ['Total net income precedes the separate noncontrolling-interest deduction and common-stockholder line. Use the total NI row for consolidated assets.']
            patterns += ['Less: Net income','noncontrolling interests']
        else:patterns += ['Loss before provision','Loss before provision','Net loss per share']
        if year==2024:
            findings += ['The SAR-related separate presentation concerns adjusted performance; it does not replace GAAP annual NI.']
            patterns += ['separate presentation of SARs']
    elif symbol=='PYPL':
        findings += ['Formal NI/assets are in millions; customer funds and receivables belong to reported total assets. Do not net or remove them to make cross-industry ratios more similar.',
                     'GAAP tables coexist with non-GAAP, FX-neutral, pro-forma, quarterly and EPS presentations; only the formal annual GAAP total is in scope.']
        if year==2015:
            findings += ['Both statements explicitly say Combined and Consolidated. The release contains eBay-related balances, net parent investment, separation charges and historical allocations.']
            questions += ['Resolve the combined/carve-out basis for annual NI and beginning/end assets under a narrowly documented source-use policy before admission. Same-report presence is not by itself proof of a constant standalone reporting perimeter. Do not manufacture standalone history or substitute pro-forma amounts.']
            patterns += ['Combined and Consolidated Balance Sheet','historically allocated','Net parent investment','These are significant expenses related to the separation']
        elif year==2016:
            findings += ['Current statements are consolidated and both selected asset dates follow the separation. Remaining comparative-year EPS or non-GAAP separation notes do not alone invalidate current annual NI and these asset endpoints.']
            patterns += ['These are significant expenses related to the separation','Reconciliation of GAAP']
        else:
            findings += ['No specific restatement or mixed-reporting-basis conflict was identified in the bounded release-context review. This is not certification that the full accounting-change history is absent.']
            patterns += ['non-GAAP measures','historically reported']
    else:
        findings += ['NI/assets scale is millions, while percentage-of-revenue columns and per-share exceptions are adjacent. Monetary-column ownership needs explicit later validation.',
                     'When reported, discontinued operations are components of consolidated total NI. Do not select income from continuing operations or adjusted NI instead.',
                     'Acquisition-related fair-value, integration, policy-conformity, pension and tax charges remain in reported GAAP NI; the non-GAAP exclusions are not a reason to remove them.']
        patterns += ['results in all periods','Income from Continuing Operations.*Discontinued Operations','accounting policies of','results were determined in accordance']
        if year in (2013,2014):
            findings += ['The identified reclassification explicitly refers to prior-period segment information. It does not by itself establish a change in consolidated NI or total-assets basis.']
            patterns += ['segment information has been reclassified']
        if year==2018:
            findings += ['The release explicitly says prior results were restated for immaterial impacts of new pension-accounting guidance. Treat this as a specific comparability question, not a non-GAAP label.']
            questions += ['Determine whether the pension-guidance restatement concerns only comparison-period income presentation or also the selected beginning total-assets basis. Preserve the note and use same-report figures; hold this NI/average-assets use only if the selected-field basis remains unresolved. A comparison-NI change alone is not enough for a permanent issuer-wide hold.']
            patterns += ['have been restated for the immaterial impacts']
        if year==2021:
            findings += ['Total net income and net income attributable to Thermo Fisher are separate. Use the total NI row with consolidated total assets.']
            patterns += ['net income attributable to Thermo']
    if not row['clock_verified']:
        questions += ['Publication lies outside the pinned calendar evidence. Retain the early-clock hold; annual-header evidence does not repair the availability clock.']
    return dict(event_id=row['event_id'],symbol=symbol,fiscal_year=year,source=row['source'],
        identity_anchors_verified_in_raw=row['identity'],publication_date=row['publication_date'],period_end_from_bound_catalog=row['period_end'],
        clock_verified_from_bound_feasibility=row['clock_verified'],
        formal_table_candidates=dict(annual_income=ni_idx,same_report_balance_sheet=bs_idx),
        header_context_verified=True,monetary_unit_context=unit,currency_literal_context='$',currency_iso_certified=False,
        raw_header_evidence=[ni,bs],findings=findings,unresolved_questions=questions,
        evidence=pick_evidence(row,patterns),review_status='context_review_complete_not_value_or_admission_review',
        selected_values_verified=False,field_accounting_admitted=False,fact_admitted=False,model_admitted=False,strategy_admitted=False)


def publish_review():
    rows,pins,feas=scan()
    reviews=[conclusion(row) for row in rows]
    # These are manually assessed scope findings. Search matches only locate evidence.
    require(len(reviews)==21 and len({e['event_id'] for e in reviews})==21,'Complete21 review')
    for row in rows:
        raw=read(row['source']['path'],pins,row['source']['sha256'])
        soup=BeautifulSoup(raw,'html.parser')
        for tag in soup(['script','style']):tag.decompose()
        text=' '.join(soup.get_text(' ',strip=True).split())
        for evidence in next(e for e in reviews if e['event_id']==row['event_id'])['evidence']:
            loc=evidence['locator']
            require(evidence['quote_numbers_redacted']==redact(text[loc['start']:loc['end']]),'Exact redacted source quote')
    read('work/after153_batch5_accounting_feasibility_review.py',pins)
    for path,digest in list(pins.items()):read(path,pins,digest)
    result=dict(schema_version=1,stage='independent_accounting_feasibility_only_not_admission',passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(),source_preparation_order=list(EXPECTED),
        scope='21 declared candidate report contexts within full28 PLTR/PYPL/TMO catalog; no monetary cells extracted or compared',
        input_hashes=pins,runtime=sys.version,event_reviews=reviews,
        counts=dict(candidate_reports=21,by_symbol=dict(Counter(r['symbol'] for r in reviews)),catalog_events=len(feas['documents']),
                    unprocessed_catalog_events=len(feas['documents'])-21,clock_verified_from_feasibility=sum(r['clock_verified'] for r in rows),
                    clock_held_from_feasibility=sum(not r['clock_verified'] for r in rows),numeric_fields_verified=0,new_admissions=0),
        unprocessed_catalog=[dict(event_id=d['event_id'],symbol=d['symbol'],fiscal_year=d['fiscal_year'],source_kind=d['source_kind'],reason='No declared statement candidate; retain latest-event blocking') for d in feas['documents'] if not d['context'].get('statement_candidates')],
        unresolved_source_use_questions=[dict(event_id=r['event_id'],questions=r['unresolved_questions']) for r in reviews if r['unresolved_questions']],
        minimum_later_validation=['Freeze exact records, source/receipt SHA, table/annual-column/label/units and source-use policy before value extraction.',
            'Independently verify total NI and both same-report asset amounts, signs, currencies, annual endpoints and source locators; missing/ambiguous stays null.',
            'Use full28 catalog for latest-event selection before validity. Six PYPL HTMLs without located statements and the TMO index entry remain barriers when latest.',
            'Distinguish field-specific comparability conflicts from irrelevant comparative NI, segment, EPS, CFO/SBC and non-GAAP disclosures. Do not clear or block by keywords alone.',
            'No first-public-version, correction-history, ISO-currency or continuous security-master certification follows from these release copies.'],
        financial_values_extracted=0,ratios_computed=0,factors_computed=0,orders_or_returns_computed=0,new_strategy_trials=0)
    with (ROOT/OUT).open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,counts=result['counts'],output=OUT,sha256=hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--symbol',choices=tuple(EXPECTED));p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.publish:
        publish_review();return
    rows,pins,feas=scan(a.symbol)
    print(json.dumps(dict(events=rows,input_hashes=pins),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
