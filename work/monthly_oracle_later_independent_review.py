"""Independent original Visa source/period/record verifier; no producer imports."""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse
import argparse, calendar, json, math, re, subprocess
from bs4 import BeautifulSoup
import monthly_wmt_crosscheck_independent_review as common
ROOT=Path(__file__).resolve().parent.parent
STUDY='outputs/monthly_sec_oracle_later_admission_v1/'
POLICY='monthly_sec_oracle_later_admission_v1'
CIK='0001341439';CLASS='issuer_hosted_reference_cdn_original_release'
EVIDENCE_CLASS='issuer_original_and_legacy_exact_facts'
CIKS={'ORCL':CIK}
SEMANTICS={'ORCL':{k:'consolidated_total_revenues' for k in ('Revenues','RevenueFromContractWithCustomerExcludingAssessedTax')}}
require=common.require;digest=common.digest;strict_json=common.strict_json

def source_files():return [Path(__file__),ROOT/'work/test_monthly_oracle_later_independent_review.py',ROOT/'work/monthly_oracle_later_independent_parser.py']+common.source_files()
def clean(value):return ' '.join(value.split())
def iso(s):require(type(s)is str and date.fromisoformat(s).isoformat()==s,'ISO date');return date.fromisoformat(s)
def stamp(s):require(datetime.fromisoformat(s.replace('Z','+00:00')).utcoffset() is not None,'aware receipt')
def unique(items,reason):require(len(items)==1,reason);return items[0]
def human(s):
 words=re.fullmatch(r'([A-Za-z]+) (\d{1,2}), (20\d{2})',clean(s));require(words is not None,'human date');month={calendar.month_name[i]:i for i in range(1,13)};require(words[1]in month,'human month');return date(int(words[3]),month[words[1]],int(words[2]))
from monthly_oracle_later_independent_parser import parse as parse_pdf
def blob(d,b,p):
    v=b[p];require(type(v)is bytes and sha256(v).hexdigest()==d['source_hashes'][p],'bound source '+p);return v
def read(d,b,p):return strict_json(blob(d,b,p))
def legacy(meta,raw,url):
    require(meta.get('source')==url and meta.get('sha256')==sha256(raw).hexdigest() and not any(k in meta for k in ('status','status_code','http_status')),'legacy source grade');stamp(meta['retrieved_at'])
def source_context(d,b,issuer):
    issuer=dict(issuer,archives=[dict(name='CIK'+issuer['cik']+'-submissions-001.json',path=issuer['filings_path'],receipt_path=issuer['filings_receipt_path'])]);d=dict(d,**issuer);CIK=issuer['cik'];symbol=issuer['symbol'];require(CIKS.get(symbol)==CIK,'issuer config')
    r=lambda p:read(d,b,p);raw=blob(d,b,d['facts_path']);old=blob(d,b,d['legacy_facts_path']);require(raw==old,'exact fact duplicate');m=r(d['legacy_facts_receipt_path']);url='https://data.sec.gov/api/xbrl/companyfacts/CIK'+CIK+'.json';legacy(m,old,url)
    status=dict(semantic_source_sufficient=True,source_standard='monthly_sec_pg_legacy_exact_semantics_v1',raw_facts_provenance_verified=False,legacy_official_url_hash_time_bound=True,exact_byte_duplicate_bound=True,http_status_recorded=False,modern_http200_provenance_verified=False,facts_path=d['facts_path'],facts_sha256=sha256(raw).hexdigest(),legacy_facts_path=d['legacy_facts_path'],legacy_facts_receipt_path=d['legacy_facts_receipt_path'],source_url=url,retrieved_at=m['retrieved_at'])
    root=r(d['filings_root_path']);cat=r(d['catalogs_path'])[symbol];require(str(root['cik']).zfill(10)==cat['cik']==CIK,'catalog issuer')
    legacy(r(d['filings_root_receipt_path']),blob(d,b,d['filings_root_path']),'https://data.sec.gov/submissions/CIK'+CIK+'.json')
    tables={'recent':root['filings']['recent']};advertised={v['name']:v for v in root['filings']['files']}
    require(len(advertised)==len(root['filings']['files']),'unique advertised archives')
    for item in d['archives']:
        name=item['name'];require(name not in tables and name in advertised,'unique registered archive');archive=r(item['path']);legacy(r(item['receipt_path']),blob(d,b,item['path']),'https://data.sec.gov/submissions/'+name);a=advertised[name]
        require(len(archive['accessionNumber'])==a['filingCount'] and min(archive['filingDate'])==a['filingFrom'] and max(archive['filingDate'])<=a['filingTo'],'archive coverage boundary');tables[name]=archive
    require(set(tables)=={v['name']for v in cat['sources']if v['status']=='present'},'all present catalog tables')
    merged={}
    for key,t in tables.items():
        meta=[v for v in cat['sources'] if v['name']==key];require(len(meta)==1,'catalog source');meta=meta[0];n=len(t['accessionNumber'])
        require(meta['status']=='present' and n==meta['row_count'] and all(type(v)is list and len(v)==n for v in t.values()) and min(t['filingDate'])==meta['actual_min']==meta['range_start'] and max(t['filingDate'])==meta['actual_max']<=meta['range_end'],'catalog columns/coverage')
        for i,acc in enumerate(t['accessionNumber']):
            rawrow={k:t[k][i] for k in ('accessionNumber','filingDate','reportDate','form')}
            if acc in merged:require(merged[acc][0]==rawrow,'conflicting accession')
            else:merged[acc]=(rawrow,[])
            merged[acc][1].append(dict(table=key,row=i))
    # Independently confirm every raw row retained in the certified catalog.
    require(set(merged)=={v['accession'] for v in cat['filings']},'catalog full accession set')
    for v in cat['filings']:
        rawrow,where=merged[v['accession']]
        require(v['form']==rawrow['form'] and v['filing_date']==rawrow['filingDate'] and v['report_date']==rawrow['reportDate'] and v['sources']==sorted(where,key=lambda x:(x['table'],x['row'])),'catalog raw row binding')
    from zoneinfo import ZoneInfo
    snap=datetime.fromisoformat(r(d['filings_root_receipt_path'])['retrieved_at'].replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    require(cat['snapshot_date']==snap,'catalog trusted snapshot')
    ranges=[];holes=[]
    for name,t in tables.items():
        sm=next(v for v in cat['sources']if v['name']==name);end=snap if name=='recent'else advertised[name]['filingTo']
        require(sm['range_end']==end and max(t['filingDate'])<=snap,'source coverage endpoint');ranges.append((iso(sm['range_start']),iso(end)))
    for name,a in sorted(advertised.items()):
        if name not in tables:holes.append(dict(source=name,start=a['filingFrom'],end=a['filingTo'],reason='missing_archive'))
    merged_ranges=[]
    for start,end in sorted(ranges):
        if merged_ranges and start<=merged_ranges[-1][1]+timedelta(days=1):merged_ranges[-1][1]=max(end,merged_ranges[-1][1])
        else:merged_ranges.append([start,end])
    require(cat['holes']==holes and cat['coverage_ranges']==[dict(start=a.isoformat(),end=b.isoformat())for a,b in merged_ranges],'raw coverage and holes')
    return status,tables,cat

def latest(record,cat,tables,sessions):
    signal=record['signal_date'];day=iso(signal);c=record['catalog'];lo=(day-timedelta(days=400)).isoformat()
    require(c['signal_date']==signal and c['coverage_start']==lo and c['coverage_end']==signal and c['coverage_complete'] is True and c['coverage_reasons']==[] and c['coverage_gaps']==[] and c['reasons']==[] and c['tied_accessions']==[],'complete original window')
    require(cat['snapshot_date']>signal and sessions[0]<=lo and signal in sessions,'snapshot and official axis')
    require(all(h['start'] is not None and h['end'] is not None and (h['end']<lo or h['start']>signal) for h in cat['holes']),'overlapping catalog hole')
    cursor=iso(lo)
    for v in sorted(cat['coverage_ranges'],key=lambda v:v['start']):
        if iso(v['end'])<cursor:continue
        require(iso(v['start'])<=cursor,'uncovered catalog date');cursor=max(cursor,iso(v['end'])+timedelta(days=1))
        if cursor>day:break
    require(cursor>day,'catalog range incomplete')
    rows={}
    for name,t in tables.items():
        for i,acc in enumerate(t['accessionNumber']):
            form=t['form'][i];filed=t['filingDate'][i]
            if form not in ('10-Q','10-Q/A','10-K','10-K/A','20-F','20-F/A','40-F','40-F/A') or not lo<=filed<=signal:continue
            j=bisect_right(sessions,filed)
            if j>=len(sessions) or sessions[j]>signal:continue
            rows[acc]=(filed,t['reportDate'][i],form,sessions[j])
    require(rows,'available report');newest=max(v[0] for v in rows.values());winners=[(a,v) for a,v in rows.items() if v[0]==newest];require(len(winners)==1,'ambiguous newest report')
    acc,(filed,end,form,usable)=winners[0];s=c['selected']
    require(c['status']=='selected' and s['accession']==acc and s['filingDate']==filed and s['reportDate']==end and s['form']==form and form in ('10-K','10-Q') and s['first_usable_date']==usable,'latest available original report')
    require(s['metadata_issues']==[] and s['supported_form'] is True and s['filing_age_days']==(day-iso(filed)).days<=180 and s['report_age_days']==(day-iso(end)).days<=240 and end<=filed,'selected age/form/period')


def compare_pair(pair,anchor,parsed):
 if parsed is None:return dict(status='unverified',reason='financial_document_not_parsed',selected_period=None)
 if parsed['identity']['confidential_label_present']:return dict(status='unverified',reason='confidential_version_without_contemporaneous_public_copy',selected_period=None)
 choices=[p for p in parsed['statement']['periods']if p['months_printed']==anchor['fiscal_quarter']*3]
 if len(choices)!=1:return dict(status='unverified',reason='selected_period_not_unique',selected_period=None)
 p=choices[0]
 if p['row_label']!=anchor['statement_label']:return dict(status='unverified',reason='statement_semantic_label_mismatch',selected_period=p)
 for role in ('current','prior'):
  if any(pair[role][key]!=p[role][key]for key in ('start','end','duration_days')):return dict(status='unverified',reason='period_boundary_mismatch',selected_period=p)
  if Decimal(str(pair[role]['val']))!=Decimal(p[role]['revenue_usd']):return dict(status='unverified',reason='printed_value_mismatch',selected_period=p)
 return dict(status='exact_match',reason=None,selected_period=p)
def validate_record(r,a,p,facts,tables,cat,sessions):
 symbol=a['symbol'];cik=a['cik'];require(CIKS[symbol]==cik==r['candidate_cik']==str(facts['cik']).zfill(10) and r['symbol']==symbol,'record entity');latest(r,cat,tables,sessions)
 s=r['catalog']['selected'];require(all(r[k]is False and r['pairing'][k]is False for k in common.FLAGS),'original flags');require(r['pairing']['status']=='paired'and r['pairing']['structural_pair_available']is True and len(r['pairing']['pairs'])in(1,2) and p in r['pairing']['pairs'],'original pair readiness')
 require(s['accession']==s['accessionNumber']==a['accession'] and s['form']==a['form'] and s['report_date']==s['reportDate']==a['report_date'] and s['filing_date']==s['filingDate']==a['filing_date'],'selected aliases')
 where=[dict(table=k,row=i)for k,t in tables.items()for i,v in enumerate(t['accessionNumber'])if v==a['accession']];require(where and sorted(where,key=lambda v:(v['table'],v['row']))==sorted(s['sources'],key=lambda v:(v['table'],v['row'])),'raw filing locators')
 require(s['first_usable_date']==a['first_usable_date'] and a['report_date']<a['release_date']<=a['filing_date']<a['first_usable_date'] and a['fiscal_quarter']in(1,2,3,4) and a['form']==('10-K'if a['fiscal_quarter']==4 else'10-Q'),'chronology and type')
 require(p['taxonomy']=='us-gaap'and p['tag']in SEMANTICS[symbol]and p['unit']=='USD','original semantic context');common.raw_contexts(p,s,facts)
 for role in ('current','prior'):
  v=p[role];start,end=iso(v['start']),iso(v['end']);require(start.day==1 and start<end and type(v['duration_days'])is int and v['duration_days']==(end-start).days+1 and (role!='current'or v['end']==a['report_date']),'original duration')
def admit_record(i,r,entries,status):
 require(status['exact_byte_duplicate_bound']is True and all(status.get(k)is False for k in ('raw_facts_provenance_verified','http_status_recorded')),'source grade unchanged')
 ps=r['pairing']['pairs'];s=r['catalog']['selected'];ids=[digest(dict(accession=s['accession'],pair=p))for p in ps];require(len(ps)in(1,2) and len(ids)==len(set(ids)) and [v['evidence_id']for v in entries]==ids,'ordered complete pairs')
 for p,e in zip(ps,entries):require(e['pair']==p and e['pair_sha256']==digest(p)and e['accession']==s['accession'],'exact pair')
 why=[e['reason']for e in entries if e['status']!='exact_match']
 if not status['semantic_source_sufficient']:why.append('legacy_original_source_not_verified')
 cov=dict(record_index_0based=i,source_record_sha256=digest(r),selected_accession=s['accession'],evidence_ids=ids,status='unverified'if why else'admitted',reasons=list(dict.fromkeys(why)))
 if why:return None,cov
 approved=[]
 for p,e in zip(ps,entries):
  approved.append(dict(evidence_id=e['evidence_id'],pair_sha256=digest(p),semantic_concept=SEMANTICS[r['symbol']][p['tag']],evidence_class=EVIDENCE_CLASS,source_class=CLASS,selected_filing=deepcopy(s),taxonomy=p['taxonomy'],tag=p['tag'],unit=p['unit'],current=deepcopy(p['current']),prior=deepcopy(p['prior']),release_evidence=deepcopy(e['release_evidence']),selected_period=deepcopy(e['selected_period']),source_status=deepcopy(status),checks={k:True for k in ('cached_raw_pair_bound','first_usable_session_verified','values_match','periods_match','release_within_report_filing','source_version_by_selected_filing','all_ordered_original_pairs_bound','original_financial_document_directly_parsed')},original_xbrl_instance_directly_verified=False,original_accession_body_directly_verified=False,period_starts_directly_printed=False,original_inline_duration_contexts_verified=False,continuous_security_identity_verified=False,factor_ready=False))
 return dict(record_index_0based=i,source_record_sha256=digest(r),symbol=r['symbol'],evidence_class=EVIDENCE_CLASS,source_class=CLASS,candidate_cik=r['candidate_cik'],signal_date=r['signal_date'],selected_accession=s['accession'],source_flags={k:r[k]for k in common.FLAGS},semantic_admitted=True,factor_ready=False,evidence_ids=ids,pair_sha256s=[digest(p)for p in ps],approval=approved),cov


def original_source(a,d,b,namespace):
 raw=blob(d,b,a['release_path']);m=read(d,b,a['receipt_path']);url=a['release_url'];u=urlparse(url)
 require(u.scheme=='https' and u.hostname=='s23.q4cdn.com' and u.path.startswith('/440135859/files/') and not u.query and not u.fragment,'unapproved CDN namespace')
 require(m['url']==m['final_url']==url and type(m['status'])is int and m['status']==200 and m['source_representation']=='original_http_bytes' and m['request_method']=='GET' and m['request_payload']is None,'original receipt');stamp(m['retrieved_at'])
 require(m['sha256']==sha256(raw).hexdigest() and m['bytes']==len(raw) and m['content_type']=='application/pdf' and raw.startswith(b'%PDF-'),'PDF bytes')
 return dict(source_path=a['release_path'],source_sha256=sha256(raw).hexdigest(),source_url=url,receipt_path=a['receipt_path'],receipt_sha256=d['source_hashes'][a['receipt_path']],retrieved_at=m['retrieved_at'],http_status_verified=True,transport_response_separately_bound=False,namespace_evidence=namespace,historical_snapshot_verified=False,historical_pdf_bytes_available_verified=False,exact_publication_timestamp_verified=False,filing_catalog_http_status_verified=False,official_archive_outlink_verified=False,source_attribution='Original CDN PDF; present namespace corroborated by Oracle-hosted third-party publication. No exact release outlink or historical-hosting claim.',availability_basis='filing_asof_reconstruction_not_historical_blob_publication')
def extract_pdf(raw):
 code="import io,json,sys,pdfplumber\nfrom pypdf import PdfReader\nassert pdfplumber.__version__=='0.11.9'\nb=sys.stdin.buffer.read();reader=PdfReader(io.BytesIO(b))\nwith pdfplumber.open(io.BytesIO(b)) as f:\n pages=[]\n for i,p in enumerate(f.pages):\n  links=[]\n  for a in reader.pages[i].get('/Annots',[]):\n   act=a.get_object().get('/A',{})\n   if act.get('/URI'):links.append(str(act['/URI']))\n  pages.append(dict(page=i+1,text=p.extract_text(x_tolerance=3,y_tolerance=3)or'',links=links))\n print(json.dumps(dict(engine='pdfplumber',version=pdfplumber.__version__,metadata=f.metadata,total_pages=len(pages),pages=pages),allow_nan=False))"
 result=subprocess.run([common.PDF_PYTHON,'-B','-W','error','-c',code],input=raw,capture_output=True,check=True);require(not result.stderr,'PDF extraction warnings');return strict_json(result.stdout)
def namespace_source(d,b,pdf_reader):
 x=d['namespace_source'];raw=blob(d,b,x['path']);response=blob(d,b,x['response_path']);m=read(d,b,x['receipt_path'])
 require(urlparse(x['url']).hostname=='www.oracle.com' and m['url']==m['final_url']==x['url'] and m['status']==200 and m['public_unauthenticated']is True,'issuer reference receipt');stamp(m['retrieved_at'])
 require(raw==response and m['decoded_sha256']==sha256(raw).hexdigest()==m['response_sha256'] and m['decoded_bytes']==len(raw)==m['response_bytes'] and m['decoder']=='identity' and m['content_encoding']is None,'issuer reference bytes')
 root=Path(x['receipt_path']).parent.parent;require(str(root/m['decoded_path'])==x['path'] and str(root/m['response_path'])==x['response_path'],'issuer reference paths')
 doc=pdf_reader(raw);require(doc['engine']=='pdfplumber' and doc['version']=='0.11.9' and doc['total_pages']==len(doc['pages'])==12,'issuer reference PDF');first=clean(doc['pages'][0]['text']);note=clean(doc['pages'][8]['text'])
 require(all(t in first for t in ('Holger Mueller','December 17, 2025','Constellation Research clients')) and all(t in note for t in ('Oracle Corporation Q4 Fiscal 2025 Financial Results, page 7.','© 2025 Constellation Research Inc.')),'third party authorship and reference')
 links=[t for t in doc['pages'][8]['links']if 'q4cdn'in t];ref='http://s23.q4cdn.com/440135859/files/doc_financials/2025/q4/Q425_Form8K_Exhibit99-1_Earnings_Release_Tables_Final.pdf ';require(links==[ref]*3,'literal original PDF annotations')
 return dict(source_path=x['path'],source_sha256=sha256(raw).hexdigest(),receipt_path=x['receipt_path'],source_url=x['url'],target_host='s23.q4cdn.com',target_tenant='440135859',reference_page=9,raw_annotation_urls=links,author='Holger Mueller',publisher='Constellation Research Inc.',hosting_issuer='Oracle',source_class='current_issuer_hosted_third_party_reference',current_namespace_association_verified=True,issuer_authored_reference=False,exact_release_outlink_verified=False,historical_control_verified=False,historical_pdf_publication_verified=False,reference_document_date='2025-12-17',reference_date_used_for_historical_admission=False)

def source_completion(d,b):
 for path in d['source_completions']:
  c=read(d,b,path);found=False
  for field in ('artifact_hashes','hashes','files'):
   values=c.get(field,{});require(type(values)is dict,'invalid_completion_artifact_map')
   for n,h in values.items():
    found=True;require(not Path(n).is_absolute() and '..'not in Path(n).parts,'invalid_completion_artifact_path');target=n if n.startswith(('outputs/','work/'))else str(Path(path).parent/n);require(d['source_hashes'].get(target)==h and sha256(blob(d,b,target)).hexdigest()==h,'source_completion_artifact_mismatch')
  require(found,'empty_source_completion_artifacts')
  for n,h in c.get('source_hashes',{}).items():require(d['source_hashes'].get(n)==h and sha256(blob(d,b,n)).hexdigest()==h,'source_completion_dependency_mismatch')

def reconstruct(d,b,pdf_reader=None):
 from collections import Counter
 require(d['policy_id']==POLICY and d['source_class']==CLASS and d['evidence_class']==EVIDENCE_CLASS and d['allowed_tag_semantics']==SEMANTICS,'classes');require(d['source_hashes'][d['source_records']['path']]==d['source_records']['sha256'],'records hash');r=lambda p:read(d,b,p);records=r(d['source_records']['path']);sessions=r(d['sessions_path']);require(len(records)==d['expected_record_count']and sessions==sorted(set(sessions))and sessions,'axes')
 require(len({(x['symbol'],x['signal_date'])for x in records})==len(records),'unique keys')
 for rec in records:require(all(rec.get(k)is False and rec['pairing'].get(k)is False for k in common.FLAGS),'original flags')
 anchors=d['scope'];ids=[i for a in anchors for i in a['record_indices']];require(len(anchors)==len({a['accession']for a in anchors})==d['expected_accessions']and len(ids)==len(set(ids))==d['expected_rows']and all(type(i)is int and 0<=i<len(records)for i in ids),'fixed scope');require(sorted(i for i,v in enumerate(records)if v['symbol']in SEMANTICS and i not in ids)==d['excluded_indices'],'all exclusions')
 require(set(ids).isdisjoint(d['previously_admitted_indices']),'previous overlap')
 if d.get('previous_admissions_paths'):require(sorted(x['record_index_0based']for p in d['previous_admissions_paths']for x in r(p)['records']if x['symbol']=='ORCL')==sorted(d['previously_admitted_indices']),'prior admissions exact')
 source_completion(d,b)
 namespace=namespace_source(d,b,pdf_reader or extract_pdf)
 contexts={s:source_context(d,b,issuer)for s,issuer in d['issuers'].items()};require(set(contexts)=={a['symbol']for a in anchors} and set(contexts)<=set(SEMANTICS),'issuer scope');statuses={s:v[0]for s,v in contexts.items()};facts={s:r(issuer['facts_path'])for s,issuer in d['issuers'].items()};releases={};evidence={};rows=[];admitted=[]
 for a in anchors:
  symbol=a['symbol'];acc=a['accession'];source=original_source(a,d,b,namespace);raw=blob(d,b,a['release_path']);parsed=None;reason=None
  try:
   parsed=parse_pdf((pdf_reader or extract_pdf)(raw),a['filing_date']);require(parsed['identity']['release_date']==a['release_date'] and parsed['identity']['symbol']==symbol and parsed['identity']['fiscal_year']==a['fiscal_year'] and parsed['identity']['fiscal_quarter']==a['fiscal_quarter'],'parsed_identity_scope')
  except ValueError as e:reason=str(e);parsed=None
  release=dict(**source,parsed=parsed,parse_reason=reason,financial_evidence_available_date=a['filing_date'],filing_date=a['filing_date'],first_usable_date=a['first_usable_date']);releases[acc]=release;seen={};status,tables,cat=contexts[symbol]
  for i in a['record_indices']:
   rec=records[i];require(digest(rec)==a['record_sha256s'][str(i)],'original record hash');entries=[]
   for pair in rec['pairing']['pairs']:
    validate_record(rec,a,pair,facts[symbol],tables,cat,sessions);eid=digest(dict(accession=acc,pair=pair));seen[eid]=digest(pair);ev=dict(evidence_id=eid,accession=acc,pair_sha256=digest(pair),pair=deepcopy(pair),semantic_concept=SEMANTICS[symbol][pair['tag']],**compare_pair(pair,a,parsed),release_evidence=release)
    if eid in evidence:require(evidence[eid]==ev,'consistent repeated pair')
    evidence[eid]=ev;entries.append(ev)
   sidecar,cov=admit_record(i,rec,entries,status);rows.append(cov)
   if sidecar is not None:admitted.append(sidecar)
  require(seen==a['evidence_pair_hashes'],'all original pairs')
 require(len(evidence)==d['expected_pairs']and sum(len(v['evidence_ids'])for v in rows)==d['expected_pair_occurrences'],'pair counts');rows.sort(key=lambda v:v['record_index_0based']);admitted.sort(key=lambda v:v['record_index_0based'])
 base=dict(schema_version=1,policy_id=POLICY,source_class=CLASS,evidence_class=EVIDENCE_CLASS,raw_facts_provenance_verified=False,source_status=statuses,source_hashes=d['source_hashes'],source_records=d['source_records'],record_count=len(records),factor_ready_count=0,independent_review_pending=True)
 admissions=dict(**base,semantic_admitted_count=len(admitted),records=admitted);ev=dict(**base,releases=releases,evidence=evidence)
 coverage=dict(schema_version=1,policy_id=POLICY,scope_rows=len(rows),scope_accessions=len(releases),unique_pairs=len(evidence),pair_occurrences=sum(len(v['evidence_ids'])for v in rows),semantic_admitted_count=len(admitted),unverified_count=len(rows)-len(admitted),admitted_by_symbol=dict(Counter(v['symbol']for v in admitted)),factor_ready_count=0,admitted_indices=[v['record_index_0based']for v in admitted],excluded_indices=d['excluded_indices'],rows=rows,pair_status_counts=dict(Counter(v['status']for v in evidence.values())),source_flags_unchanged=True,as_reported_comparatives_not_restandardized=True,growth_factor_rank_order_return_computed=False,strategy_trials=0,network_requests=0)
 return {'ADMISSIONS.json':admissions,'EVIDENCE.json':ev,'COVERAGE.json':coverage}
def proof_counts(ev):
 rows=list(ev['releases'].values());p=[r['parsed']for r in rows if r['parsed']is not None]
 return dict(original_releases_loaded=len(rows),original_financial_releases_parsed=len(p),primary_gaap_statements_parsed=sum(len(x['statement']['periods'])for x in p),annual_statements_parsed=sum(t['months_printed']==12 for x in p for t in x['statement']['periods']),all_original_ordered_pairs_required=True,source_versions_checked_against_selected_filing=True,pages_reconstructed=sum(x['extractor']['pages_reconstructed']for x in p),confidential_versions_blocked=sum(x['identity']['confidential_label_present']for x in p))

def audit(root,design_sha,plan_sha):
    root=Path(root);r=lambda p:strict_json((root/p).read_bytes());h=lambda p:sha256((root/p).read_bytes()).hexdigest()
    require(h(STUDY+'DESIGN.json')==design_sha and h(STUDY+'PLAN.json')==plan_sha,'exact registration');d=r(STUDY+'DESIGN.json');plan=r(STUDY+'PLAN.json')
    require(d['policy_id']==POLICY and d['stage']=='frozen_scope_before_actual_build' and d['source_class']==CLASS and d['evidence_class']==EVIDENCE_CLASS and d['allowed_tag_semantics']==SEMANTICS,'metadata classes')
    require(tuple(d[k] for k in ('expected_record_count','expected_rows','expected_accessions','expected_pairs','expected_pair_occurrences'))==(3904,29,10,13,37) and set(d['issuers'])=={'ORCL'},'fixed actual scope')
    require(plan['policy_id']==POLICY and plan['design_sha256']==design_sha and plan['actual_executed'] is False and plan['source_hashes']==d['source_hashes'],'plan contract')
    require(type(plan['synthetic_tests_passed'])is int and plan['synthetic_tests_passed']>0 and h(STUDY+'TESTS.txt')==plan['tests_sha256'],'strict test receipt')
    hashes=common.verify_hashes(root,d['source_hashes']);impl=common.verify_hashes(root,plan['implementation_hashes']);independent=common.verify_hashes(root,plan['independent_review_hashes'])
    auditor={str(p.relative_to(ROOT)):h(str(p.relative_to(ROOT))) for p in source_files()};require(all(independent.get(k)==v or hashes.get(k)==v for k,v in auditor.items()),'independent code closure')
    require(set(impl)=={STUDY+n for n in ('parser.py','admission.py','build.py','test_parser.py','test_admission.py','register.py')},'implementation set')
    require(h(STUDY+'SOURCE_INVENTORY.json')==d['source_inventory_sha256'] and r(STUDY+'SOURCE_INVENTORY.json')['source_hashes']==hashes and hashes[d['source_records']['path']]==d['source_records']['sha256'],'inventory and source records')
    before={STUDY+n:h(STUDY+n) for n in ('DESIGN.json','PLAN.json','TESTS.txt','SOURCE_INVENTORY.json','ADMISSIONS.json','EVIDENCE.json','COVERAGE.json','PROOF.json')}
    structural=r(d['structural_review']);require(structural['passed'] is True and structural['rows_replayed']==3904 and structural['failures']==[] and structural['selected_accession_and_pairs_independently_recomputed'] is True,'structural certification')
    require(all(hashes.get('outputs/monthly_sec_revenue_data_v3/'+k)==v for k,v in structural['artifact_hashes'].items()),'structural certificate closure')
    expected=reconstruct(d,{p:(root/p).read_bytes() for p in hashes})
    for n,v in expected.items():require(r(STUDY+n)==v,'complete independently rebuilt '+n)
    counts=proof_counts(expected['EVIDENCE.json'])
    proof=dict(schema_version=1,policy_id=POLICY,design_sha256=design_sha,plan_sha256=plan_sha,source_hashes=hashes,implementation_hashes=impl,independent_review_hashes=independent,output_hashes={n:h(STUDY+n) for n in expected},**counts,source_rows_unchanged=True,source_class=CLASS,evidence_class=EVIDENCE_CLASS,raw_facts_provenance_verified=False,filing_catalog_http_status_verified=False,exact_byte_duplicate_bound=True,official_archive_outlink_verified=False,period_starts_directly_printed=False,exact_publication_timestamp_verified=False,historical_pdf_bytes_available_verified=False,as_reported_comparatives_not_restandardized=True,factor_ready_count=0,strategy_trials=0,network_requests=0,independent_review_pending=True)
    require(r(STUDY+'PROOF.json')==proof,'complete rebuilt proof')
    for m in (hashes,impl,independent,auditor,before):common.verify_hashes(root,m)
    return dict(schema_version=1,policy_id=POLICY,passed=True,errors=[],design_sha256=design_sha,plan_sha256=plan_sha,source_hashes=hashes,implementation_hashes=impl,independent_review_hashes=independent,auditor_source_hashes=auditor,artifact_hashes=before,summary=expected['COVERAGE.json'],independent_document_counts=counts,all29_original_records_reconstructed=True,all13_original_pairs_and37_occurrences_verified=True,all_original_ordered_pairs_preserved=True,original_financial_pdf_independently_parsed=counts['original_financial_releases_parsed']==10,legacy_source_grades_unchanged=True,latest_filing_independently_selected_from_raw_tables=True,starts_derived_from_printed_calendar_months=True,source_class=CLASS,evidence_class=EVIDENCE_CLASS,raw_facts_provenance_verified=False,filing_catalog_http_status_verified=False,same_document_issuer_ticker_verified=all(v['parsed'] is not None for v in expected['EVIDENCE.json']['releases'].values()),exact_publication_timestamp_verified=False,continuous_security_identity_verified=False,production_modules_imported=False,factor_ready=False,strategy_trials=0,network_requests=0,observer_files_read_or_written=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design-sha',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',default='work/monthly_oracle_later_independent_review.json');a=p.parse_args();result=audit(ROOT,a.design_sha,a.plan_sha);common.immutable_write(ROOT/a.output,result);print(json.dumps(dict(passed=True,output=a.output,admitted=result['summary']['semantic_admitted_count'],unverified=result['summary']['unverified_count'],sha256=sha256((ROOT/a.output).read_bytes()).hexdigest())))
