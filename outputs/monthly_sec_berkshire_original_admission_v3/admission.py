"""Berkshire original-PDF source and exact-record semantic checks; no network."""
from bisect import bisect_right
import calendar
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

_spec=importlib.util.spec_from_file_location('berkshire_statement',Path(__file__).with_name('parser.py'))
parser=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(parser)
POLICY='monthly_sec_berkshire_original_admission_v3'
CIK='0001067983'
SOURCE_CLASS='issuer_original_financial_report'
EVIDENCE_CLASS='issuer_pdf_and_legacy_exact_facts'
SEMANTICS={'Revenues':'consolidated_reported_total_revenues',
    'SalesRevenueNet':'consolidated_sales_and_service_revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax':'consolidated_revenues_from_customer_contracts'}
FLAGS=('factor_ready','identity_verified','semantic_verified')

def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(blob):return hashlib.sha256(blob).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def digest(value):return sha(canonical(value))


def strict_json(blob):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'duplicate_json_key');out[key]=value
        return out
    def bad(value):raise ValueError('nonfinite_json_number')
    return json.loads(blob,object_pairs_hook=pairs,parse_constant=bad)


def iso(value):
    require(type(value) is str and date.fromisoformat(value).isoformat()==value,'invalid_date')
    return date.fromisoformat(value)


def timestamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.utcoffset() is not None,'naive_retrieval_time');return result


def flags(record):
    require(all(record.get(k) is False for k in FLAGS),'original_flags_changed')
    return {k:record[k] for k in FLAGS}


def get_blob(path,documents,hashes):
    blob=documents[path]
    require(type(blob) is bytes and sha(blob)==hashes[path],'bound_source_hash_mismatch: '+path)
    return blob


def get_json(path,documents,hashes):return strict_json(get_blob(path,documents,hashes))


def legacy_receipt(meta,raw,url):
    require(meta.get('source')==url and meta.get('sha256')==sha(raw),'legacy_receipt_mismatch')
    timestamp(meta['retrieved_at'])
    require(not any(k in meta for k in ('http_status','status','status_code')),'unexpected_legacy_http_claim')


def facts_source(design,documents,hashes):
    original=get_blob(design['facts_path'],documents,hashes)
    legacy=get_blob(design['legacy_facts_path'],documents,hashes)
    require(original==legacy,'facts_exact_duplicate_mismatch')
    url='https://data.sec.gov/api/xbrl/companyfacts/CIK'+CIK+'.json'
    meta=get_json(design['legacy_facts_receipt_path'],documents,hashes)
    legacy_receipt(meta,legacy,url)
    return dict(semantic_source_sufficient=True,source_standard='monthly_sec_pg_legacy_exact_semantics_v1',
        raw_facts_provenance_verified=False,legacy_official_url_hash_time_bound=True,exact_byte_duplicate_bound=True,
        http_status_recorded=False,modern_http200_provenance_verified=False,
        facts_path=design['facts_path'],facts_sha256=sha(original),legacy_facts_path=design['legacy_facts_path'],
        legacy_facts_receipt_path=design['legacy_facts_receipt_path'],source_url=url,retrieved_at=meta['retrieved_at'])


def filing_source(design,documents,hashes):
    root=get_json(design['filings_root_path'],documents,hashes)
    require(str(root['cik']).zfill(10)==CIK,'filing_root_cik_mismatch')
    legacy_receipt(get_json(design['filings_root_receipt_path'],documents,hashes),
        get_blob(design['filings_root_path'],documents,hashes),'https://data.sec.gov/submissions/CIK'+CIK+'.json')
    catalog=get_json(design['catalogs_path'],documents,hashes)['BRK.B']
    require(catalog['cik']==CIK,'catalog_cik_mismatch')
    recent=root['filings']['recent'];meta=[s for s in catalog['sources'] if s['name']=='recent']
    require(len(meta)==1 and meta[0]['status']=='present' and
            len(recent['accessionNumber'])==meta[0]['row_count'] and
            all(type(v) is list and len(v)==len(recent['accessionNumber']) for v in recent.values()) and
            min(recent['filingDate'])==meta[0]['actual_min']==meta[0]['range_start'] and
            max(recent['filingDate'])==meta[0]['actual_max']<=meta[0]['range_end'],'recent_catalog_mismatch')
    begin,end=design['needed_coverage_start'],design['needed_coverage_end']
    require(meta[0]['range_start']<=begin<=end<=meta[0]['range_end'] and
            all(h['end']<begin or h['start']>end for h in catalog['holes']),'needed_coverage_gap')
    return {'recent':recent}


def validate_pair(record,pair,facts,sessions,anchor,filings):
    selected=record['catalog']['selected'];end=iso(anchor['report_date'])
    require(record['symbol']=='BRK.B' and record['candidate_cik']==CIK and str(facts['cik']).zfill(10)==CIK,'issuer_mismatch')
    require(record['catalog']['status']=='selected' and record['catalog']['coverage_complete'] is True and
            selected['metadata_issues']==[] and selected['supported_form'] is True and
            record['pairing']['status']=='paired' and record['pairing']['structural_pair_available'] is True,
            'original_structural_not_ready')
    require(selected['accession']==selected['accessionNumber']==anchor['accession'] and
            selected['form']==anchor['form'] and selected['reportDate']==selected['report_date']==anchor['report_date'] and
            selected['filingDate']==selected['filing_date']==anchor['filing_date'],'selected_metadata_mismatch')
    matches=[]
    for name,table in filings.items():
        for i,acc in enumerate(table['accessionNumber']):
            if acc==anchor['accession']:
                require(all(table[k][i]==selected[k] for k in ('accessionNumber','filingDate','reportDate','form')),'raw_filing_mismatch')
                matches.append(dict(table=name,row=i))
    require(matches and sorted(matches,key=lambda x:(x['table'],x['row']))==
            sorted(selected['sources'],key=lambda x:(x['table'],x['row'])),'selected_filing_source_row_mismatch')
    pos=bisect_right(sessions,anchor['filing_date'])
    require(pos<len(sessions) and selected['first_usable_date']==anchor['first_usable_date']==sessions[pos] and
            sessions[pos]<=record['signal_date'] and record['signal_date'] in sessions,'first_usable_session_mismatch')
    require(end.month in (3,6,9,12) and end.day==calendar.monthrange(end.year,end.month)[1] and
            anchor['form']==('10-K' if end.month==12 else '10-Q'),'noncalendar_report_period')
    require(anchor['report_date']<anchor['release_date']<=anchor['filing_date']<anchor['first_usable_date'],
            'release_filing_order_mismatch')
    require(pair['taxonomy']=='us-gaap' and pair['tag'] in SEMANTICS and pair['unit']=='USD','unsupported_pair_semantics')
    rows=facts['facts']['us-gaap'][pair['tag']]['units']['USD']
    for role,year in (('current',end.year),('prior',end.year-1)):
        item=pair[role];start=date(year,1,1);finish=date(year,end.month,end.day)
        require(item['accn']==anchor['accession'] and item['filed']==anchor['filing_date'] and item['form']==anchor['form'] and
                item['start']==start.isoformat() and item['end']==finish.isoformat() and
                type(item['duration_days']) is int and item['duration_days']==(finish-start).days+1,'raw_period_metadata_mismatch')
        indices=item['row_indices']
        require(type(indices) is list and indices and all(type(i) is int and 0<=i<len(rows) for i in indices) and
                indices==sorted(set(indices)) and type(item['row_index']) is int and item['row_index']==indices[0],
                'invalid_raw_indices')
        found=[i for i,row in enumerate(rows) if (row.get('accn'),row.get('start'),row.get('end'))==
               (item['accn'],item['start'],item['end'])]
        raw={k:v for k,v in item.items() if k not in ('row_index','row_indices','duration_days')}
        require(found==indices and all(rows[i]==raw for i in indices),'raw_context_mismatch')
        require(type(item['val']) in (int,float) and Decimal(str(item['val'])).is_finite() and item['val']>=0,'invalid_amount')
    return end.month


def compare_pair(pair,duration,parsed,xbrl):
    if parsed is None:return dict(status='unverified',reason='release_not_parsed',selected_period=None,xbrl_corroboration=None)
    concept=parsed['concepts'].get(pair['tag'])
    found=[] if concept is None else [v for v in concept['periods'] if v['duration_months']==duration and
           all(v[r]['start']==pair[r]['start'] and v[r]['end']==pair[r]['end'] and
               v[r]['duration_days']==pair[r]['duration_days'] for r in ('current','prior'))]
    if len(found)!=1:return dict(status='unverified',reason='required_comparable_period_not_printed',selected_period=None,xbrl_corroboration=None)
    equal=all(found[0][r]['value_usd']==pair[r]['val'] for r in ('current','prior'))
    corroboration=None
    if xbrl is not None:
        try:corroboration=xbrl_pair(xbrl,pair)
        except ValueError as exc:
            return dict(status='unverified',reason='issuer_xbrl_not_corroborated: '+str(exc),selected_period=deepcopy(found[0]),xbrl_corroboration=None)
    if pair['tag']=='SalesRevenueNet' and corroboration is None:
        return dict(status='unverified',reason='rounded_sales_missing_xbrl',selected_period=deepcopy(found[0]),xbrl_corroboration=None)
    return dict(status='exact_match' if equal else 'unverified',reason=None if equal else 'reported_amount_mismatch',
                selected_period=deepcopy(found[0]),xbrl_corroboration=corroboration)


def admit_record(index,record,entries,source_status):
    pairs=record['pairing']['pairs'];selected=record['catalog']['selected']
    expected=[digest(dict(accession=selected['accession'],pair=v)) for v in pairs]
    require(len(pairs) in (1,2) and [e['evidence_id'] for e in entries]==expected,'pair_order_or_count_mismatch')
    for pair,e in zip(pairs,entries):
        require(e['pair']==pair and e['pair_sha256']==digest(pair) and e['accession']==selected['accession'],'evidence_pair_mismatch')
    reasons=[e['reason'] for e in entries if e['status']!='exact_match']
    if not source_status['semantic_source_sufficient']:reasons.append('legacy_facts_source_not_verified')
    coverage=dict(record_index_0based=index,source_record_sha256=digest(record),selected_accession=selected['accession'],
                  evidence_ids=expected,status='unverified' if reasons else 'admitted',reasons=list(dict.fromkeys(reasons)))
    if reasons:return None,coverage
    approved=[]
    for pair,e in zip(pairs,entries):
        approved.append(dict(evidence_id=e['evidence_id'],pair_sha256=digest(pair),semantic_concept=SEMANTICS[pair['tag']],
            evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,
            selected_filing=deepcopy(selected),taxonomy=pair['taxonomy'],tag=pair['tag'],unit=pair['unit'],
            current=deepcopy(pair['current']),prior=deepcopy(pair['prior']),release_evidence=deepcopy(e['release_evidence']),
            selected_period=deepcopy(e['selected_period']),xbrl_corroboration=deepcopy(e['xbrl_corroboration']),source_status=deepcopy(source_status),
            checks=dict(cached_raw_pair_bound=True,first_usable_session_verified=True,values_match=True,periods_match=True,
                        release_within_report_filing=True,original_pdf_directly_parsed=True),
            original_xbrl_instance_directly_verified=e['xbrl_corroboration'] is not None,original_accession_body_directly_verified=False,
            period_starts_directly_printed=False,continuous_security_identity_verified=False,factor_ready=False))
    return dict(record_index_0based=index,source_record_sha256=digest(record),symbol=record['symbol'],
        evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,
        candidate_cik=record['candidate_cik'],signal_date=record['signal_date'],selected_accession=selected['accession'],
        source_flags=flags(record),semantic_admitted=True,factor_ready=False,evidence_ids=expected,
        pair_sha256s=[digest(p) for p in pairs],approval=approved),coverage

# These checks consume already recorded bytes only. No network client is imported.
import gzip
from html.parser import HTMLParser
import re
import subprocess
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

REPORT_BASE='outputs/monthly_berkshire_financial_reports_v1/'
RELEASE_BASE='outputs/monthly_berkshire_original_source_v1/'
NODE='/opt/homebrew/bin/node'

class Anchors(HTMLParser):
    def __init__(self):super().__init__();self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        if tag.lower()=='a':
            values=dict(attrs)
            if 'href' in values:self.hrefs.append(values['href'])

def html_links(blob,base):
    p=Anchors();p.feed(blob.decode('utf-8'));return [(href,urljoin(base,href)) for href in p.hrefs]

def official(url):
    u=urlparse(url);require(u.scheme=='https' and u.netloc=='www.berkshirehathaway.com' and not u.username and not u.password,'unofficial_source_url')

def decode_transfer(raw,encoding):
    if encoding in (None,'','identity'):return raw
    if encoding=='gzip':return gzip.decompress(raw)
    require(encoding=='br','unsupported_transfer_encoding')
    version=subprocess.run([NODE,'--version'],capture_output=True,check=True)
    require(version.stdout.strip()==b'v26.0.0' and not version.stderr,'brotli_runtime_version')
    result=subprocess.run([NODE,'-e',"process.stdout.write(require('node:zlib').brotliDecompressSync(require('node:fs').readFileSync(0)))"],input=raw,capture_output=True,check=True)
    require(not result.stderr,'transfer_decoder_warning');return result.stdout

def transfer_receipt(meta,expected_url,documents,hashes):
    official(expected_url)
    require(meta.get('url')==meta.get('final_url')==expected_url and type(meta.get('status')) is int and meta['status']==200 and
            meta.get('public_unauthenticated') is True and meta.get('request_count')==1,'original_http_receipt_mismatch')
    timestamp(meta['retrieved_at'])
    raw=get_blob(REPORT_BASE+meta['response_path'],documents,hashes)
    decoded=get_blob(REPORT_BASE+meta['decoded_path'],documents,hashes)
    require(sha(raw)==meta['response_sha256'] and len(raw)==meta['response_bytes'] and
            sha(decoded)==meta['decoded_sha256'] and len(decoded)==meta['decoded_bytes'],'transfer_record_mismatch')
    encoding=meta.get('content_encoding')
    require(meta['decoder']=={'br':'node:zlib.brotliDecompressSync','gzip':'python:gzip.decompress'}.get(encoding,'identity'),'transfer_decoder_metadata')
    if encoding=='br':require(meta.get('decoder_version')=='v26.0.0','brotli_recorded_version')
    require(decode_transfer(raw,encoding)==decoded,'transfer_decode_mismatch')
    return decoded

def source_bundle(design,documents,hashes):
    completion=get_json(design['source_completion_path'],documents,hashes)
    require(completion['financial_reports']==35 and completion['earnings_releases']==35 and completion['structured_xbrl_instances']==6 and
            completion['semantic_verified'] is False and completion['factor_ready'] is False and completion['issuer_index_and_intermediate_links_reconstructed'] is True and completion['raw_and_decoded_checksums_verified'] is True,'source_completion_mismatch')
    for name,h in completion['source_hashes'].items():require(hashes.get(name)==h,'source_completion_not_bound')
    plan=get_json(REPORT_BASE+'PLAN.json',documents,hashes);result=get_json(REPORT_BASE+'RESULT.json',documents,hashes)
    releases=get_json(RELEASE_BASE+'RESULT.json',documents,hashes);release_plan=get_json(RELEASE_BASE+'PLAN.json',documents,hashes)
    require(result['plan_sha256']==hashes[REPORT_BASE+'PLAN.json'] and releases['plan_sha256']==hashes[RELEASE_BASE+'PLAN.json'] and
            plan['prior_target_plan_sha256']==hashes[RELEASE_BASE+'PLAN.json'],'source_plan_binding')
    for key,file in [('client_sha256','source_client.py'),('collector_sha256','collect.py')]:require(plan[key]==hashes[REPORT_BASE+file],'collector_binding')
    require(hashes.get(release_plan['source_path'])==release_plan['source_sha256'],'release_plan_source_binding')
    meta=get_json(REPORT_BASE+'index_identity.meta.json',documents,hashes);dec=get_json(REPORT_BASE+'index_decode.json',documents,hashes)
    indexraw=get_blob(REPORT_BASE+'index_identity.response',documents,hashes);index=get_blob(REPORT_BASE+'index_decoded.html',documents,hashes)
    official(meta['url']);require(meta['url']==meta['final_url']==plan['index_url'] and meta['status']==200 and meta['public_unauthenticated'] is True,'index_receipt_mismatch')
    timestamp(meta['retrieved_at'])
    require(meta['sha256']==sha(indexraw)==dec['source_sha256'] and meta['bytes']==len(indexraw) and
            dec['decoded_sha256']==sha(index)==plan['index_decoded_sha256'] and dec['bytes']==len(index) and
            plan['index_receipt_sha256']==hashes[REPORT_BASE+'index_identity.meta.json'],'index_transfer_binding')
    require(dec['decoder']=='node:zlib.brotliDecompressSync' and dec['node_version']=='v26.0.0' and meta['content_encoding']=='br','index_decoder_metadata')
    require(decode_transfer(indexraw,meta['content_encoding'])==index,'index_decode_mismatch')
    extracts=get_json(REPORT_BASE+'EXTRACTIONS.json',documents,hashes);rx=get_json(RELEASE_BASE+'EXTRACTIONS.json',documents,hashes)
    require(extracts['result_sha256']==hashes[REPORT_BASE+'RESULT.json'] and rx['result_sha256']==hashes[RELEASE_BASE+'RESULT.json'],'extraction_result_binding')
    require(len(result['entries'])==len(plan['targets'])==len(releases['results'])==len(release_plan['targets'])==len(extracts['reports'])==len(rx['extractions'])==35,'source_bundle_counts')
    return dict(index_links=html_links(index,meta['url']),plan=plan,result=result,releases=releases,release_plan=release_plan,extracts=extracts,release_extracts=rx)

def only(items,condition,reason):
    found=[v for v in items if condition(v)];require(len(found)==1,reason);return found[0]

def release_source(anchor,design,documents,hashes,bundle):
    acc=anchor['accession'];target=only(bundle['plan']['targets'],lambda v:v['accession']==acc,'target_not_unique')
    for k in ('accession','report_date','release_date','filing_date','record_indices'):require(target[k]==anchor[k],'target_mapping_changed')
    require((target['index_anchor_href'],target['intermediate_url']) in bundle['index_links'],'index_direct_link_missing')
    entry=get_json(anchor['entry_path'],documents,hashes)
    require(entry==only(bundle['result']['entries'],lambda v:v['target']['accession']==acc,'entry_not_unique') and entry['target']==target,'entry_result_mapping')
    intermediate=transfer_receipt(entry['intermediate_receipt'],target['intermediate_url'],documents,hashes)
    links=html_links(intermediate,target['intermediate_url']);selected=entry['selected_pdf_anchor']
    require((selected['href'],selected['url']) in links,'report_direct_link_missing')
    pdf=transfer_receipt(entry['pdf_receipt'],selected['url'],documents,hashes)
    require(entry['pdf_path']==anchor['report_path']==REPORT_BASE+entry['pdf_receipt']['decoded_path'] and pdf.startswith(b'%PDF-'),'report_pdf_mapping')
    instances=[]
    for x in entry['structured_instances']:
        require((x['anchor']['href'],x['anchor']['url']) in links,'xbrl_direct_link_missing')
        raw=transfer_receipt(x['receipt'],x['anchor']['url'],documents,hashes)
        require(x['path']==REPORT_BASE+x['receipt']['decoded_path'],'xbrl_path_mismatch');instances.append(dict(path=x['path'],sha256=sha(raw),source_url=x['anchor']['url']))
    require(len(instances)<=1,'multiple_xbrl_instances')
    r=only(bundle['releases']['results'],lambda v:v['accession']==acc,'release_result_not_unique')
    rt=only(bundle['release_plan']['targets'],lambda v:v['accession']==acc,'release_target_not_unique')
    for k in ('accession','filing_date','release_date','report_date','record_indices','url'):require(r[k]==rt[k]==target[k],'release_target_mapping')
    raw=get_blob(r['evidence_path'],documents,hashes);meta=get_json(r['receipt_path'],documents,hashes);official(r['url'])
    require(r['evidence_path']==anchor['release_path'] and r['http_status']==200 and r['original_pdf_downloaded'] is True and
            r['sha256']==sha(raw)==meta['sha256'] and meta['bytes']==len(raw) and raw.startswith(b'%PDF-') and
            meta['url']==meta['final_url']==r['url'] and meta['status']==200 and meta['public_unauthenticated'] is True,'release_http_mapping')
    timestamp(meta['retrieved_at'])
    if meta['request_count']==0:
        prior=get_json(meta['reused_receipt'],documents,hashes)
        require(hashes[meta['reused_receipt']]==meta['reused_receipt_sha256'] and prior['http_status']==200 and
                prior['source_url']==prior['final_url']==r['url'] and prior['sha256']==sha(raw) and prior['size']==len(raw) and
                prior['retrieved_at']==meta['retrieved_at'] and prior['request_count']==1,'reused_release_receipt_mismatch')
        require(get_blob(design['reused_probe_raw_path'],documents,hashes)==raw,'reused_release_bytes_mismatch')
    else:require(meta['request_count']==1,'release_request_count')
    extraction=only(bundle['extracts']['reports'],lambda v:v['report_date']==anchor['report_date'],'report_extraction_not_unique')
    rex=only(bundle['release_extracts']['extractions'],lambda v:v['source']==r['evidence_path'],'release_extraction_not_unique')
    require(extraction['source_path']==anchor['report_path'] and extraction['text_path']==anchor['report_text_path'] and
            extraction['pages_path']==anchor['report_pages_path'] and extraction['source_sha256']==sha(pdf) and
            extraction['text_sha256']==hashes[anchor['report_text_path']] and extraction['pages_sha256']==hashes[anchor['report_pages_path']] and
            extraction['extractor']=='pypdf' and extraction['extractor_version']=='6.10.0','report_extraction_binding')
    require(rex['text_path']==anchor['release_text_path'] and rex['source_sha256']==sha(raw) and rex['text_sha256']==hashes[anchor['release_text_path']] and
            rex['extractor']=='pypdf' and rex['version']=='6.10.0','release_extraction_binding')
    textreceipt=get_json(rex['text_receipt_path'],documents,hashes)
    require(textreceipt['source_sha256']==sha(raw) and textreceipt['text_sha256']==hashes[anchor['release_text_path']] and
            textreceipt['extractor']=='pypdf' and textreceipt['version']=='6.10.0' and textreceipt['pages']==rex['pages'],'release_text_receipt_binding')
    return dict(source_class=SOURCE_CLASS,evidence_class=EVIDENCE_CLASS,source_path=anchor['report_path'],source_sha256=sha(pdf),source_url=selected['url'],
        report_expected_page_count=extraction['page_count'],release_expected_page_count=rex['pages'],
        report_http_receipt=entry['pdf_receipt'],intermediate_http_receipt=entry['intermediate_receipt'],release_source_path=r['evidence_path'],release_http_receipt=meta,
        official_original_pdf_http200_verified=True,official_index_intermediate_report_links_verified=True,issuer_xbrl_instances=instances,
        publication_time_precision='date_only',first_public_time_verified=False,exact_report_historical_publication_verified=False,
        filing_catalog_source_status='legacy URL/hash/aware-retrieval bound; no HTTP status recorded; recent table covers all required windows')

XBRLI='http://www.xbrl.org/2003/instance'
XSI='http://www.w3.org/2001/XMLSchema-instance'

def xbrl_pair(raw,pair):
    # A corroborating issuer instance is never substituted for a missing printed concept.
    require(b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper(),'unsafe_xml_declaration')
    root=ET.fromstring(raw);require(root.tag=='{'+XBRLI+'}xbrl','xbrl_root_namespace')
    ns={'x':XBRLI};contexts={};units={}
    for c in root.findall('x:context',ns):
        require(c.get('id') not in contexts,'duplicate_context_id');contexts[c.get('id')]=c
    for u in root.findall('x:unit',ns):
        require(u.get('id') not in units,'duplicate_unit_id');units[u.get('id')]=u
    out={}
    for role in ('current','prior'):
        expected=pair[role];matches=[]
        for fact in root:
            if not re.fullmatch(r'\{https?://fasb.org/us-gaap/\d{4}(?:-\d{2}-\d{2})?\}'+pair['tag'],fact.tag):continue
            c=contexts.get(fact.get('contextRef'));u=units.get(fact.get('unitRef'))
            if c is None or u is None:continue
            entity=c.find('x:entity',ns);period=c.find('x:period',ns)
            if entity is None or period is None:continue
            ids=entity.findall('x:identifier',ns)
            if len(ids)!=1 or ids[0].get('scheme') not in ('http://www.sec.gov/CIK','https://www.sec.gov/CIK') or (ids[0].text or '').zfill(10)!=CIK:continue
            if any(v.tag in ('{'+XBRLI+'}segment','{'+XBRLI+'}scenario') or 'xbrldi' in v.tag or v.tag.endswith('}explicitMember') or v.tag.endswith('}typedMember') for v in c.iter()):continue
            if [v.tag for v in period]!=['{'+XBRLI+'}startDate','{'+XBRLI+'}endDate']:continue
            if (period[0].text,period[1].text)!=(expected['start'],expected['end']):continue
            measures=u.findall('x:measure',ns)
            if len(u)!=1 or len(measures)!=1 or measures[0].text not in ('iso4217:USD','USD'):continue
            # Prefix binding is verified from the original document, not accepted by local name alone.
            declarations=dict(re.findall(rb'xmlns:([\w-]+)=["\']([^"\']+)',raw))
            measure=measures[0].text
            if ':' not in measure or declarations.get(measure.split(':')[0].encode())!=b'http://www.xbrl.org/2003/iso4217':continue
            require(fact.get('{'+XSI+'}nil') not in ('true','1'),'nil_xbrl_fact')
            value=Decimal(fact.text or 'NaN');require(value.is_finite(),'nonfinite_xbrl_fact')
            require(value==Decimal(str(expected['val'])),'issuer_xbrl_amount_mismatch')
            decimals=fact.get('decimals');require(decimals is not None and re.fullmatch(r'-?\d+|INF',decimals) is not None,'xbrl_decimals_missing')
            if pair['tag']=='SalesRevenueNet':require(decimals=='-8','sales_xbrl_precision_mismatch')
            matches.append(dict(context_id=fact.get('contextRef'),unit_id=fact.get('unitRef'),namespace=fact.tag.split('}')[0][1:],start=expected['start'],end=expected['end'],value_usd=expected['val'],decimals=decimals,consolidated_no_dimensions=True,entity_cik=CIK))
        require(len(matches)==1,'issuer_xbrl_fact_not_unique');out[role]=matches[0]
    return out
