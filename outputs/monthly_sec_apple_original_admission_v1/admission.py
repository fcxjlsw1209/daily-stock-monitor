"""Apple original-document source and exact-record semantic checks; no network."""
from bisect import bisect_right
import calendar
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

_spec=importlib.util.spec_from_file_location('apple_statement',Path(__file__).with_name('parser.py'))
parser=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(parser)
POLICY='monthly_sec_apple_original_admission_v1'
CIK='0000320193'
SOURCE_CLASS='issuer_original_financial_documents'
EVIDENCE_CLASS='issuer_original_and_http_verified_exact_facts'
SEMANTICS={k:'consolidated_net_sales' for k in ('Revenues','SalesRevenueNet','RevenueFromContractWithCustomerExcludingAssessedTax')}
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


def validate_pair(record,pair,facts,sessions,anchor,filings):
    selected=record['catalog']['selected'];end=iso(anchor['report_date'])
    require(record['catalog']['coverage_complete'] is True,'incomplete_record_coverage')
    require(record['symbol']=='AAPL' and record['candidate_cik']==CIK and str(facts['cik']).zfill(10)==CIK,'issuer_mismatch')
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
    require(end.weekday()==5 and anchor['form']==('10-K' if anchor['fiscal_quarter']==4 else '10-Q'),
            'invalid_fiscal_report_period')
    require(anchor['report_date']<anchor['release_date']<=anchor['filing_date']<anchor['first_usable_date'],
            'release_filing_order_mismatch')
    require(pair['taxonomy']=='us-gaap' and pair['tag'] in SEMANTICS and pair['unit']=='USD','unsupported_pair_semantics')
    rows=facts['facts']['us-gaap'][pair['tag']]['units']['USD']
    for role in ('current','prior'):
        item=pair[role];start=iso(item['start']);finish=iso(item['end'])
        require(start.weekday()==6 and finish.weekday()==5 and start<finish and
                (role!='current' or finish==end),'invalid_raw_fiscal_boundaries')
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
    return 3*anchor['fiscal_quarter']


def admit_record(index,record,entries,source_status):
    pairs=record['pairing']['pairs'];selected=record['catalog']['selected']
    expected=[digest(dict(accession=selected['accession'],pair=v)) for v in pairs]
    require(len(pairs)==1 and [e['evidence_id'] for e in entries]==expected,'pair_order_or_count_mismatch')
    for pair,e in zip(pairs,entries):
        require(e['pair']==pair and e['pair_sha256']==digest(pair) and e['accession']==selected['accession'],'evidence_pair_mismatch')
    reasons=[e['reason'] for e in entries if e['status']!='exact_match']
    for e in entries:
        identity=e['release_evidence']['record_identity']
        require(identity['record_level_security_identity_verified'] is True and identity['continuous_security_identity_verified'] is False,'identity_gate_failed')
    if source_status.get('semantic_source_sufficient') is not True or source_status.get('raw_facts_provenance_verified') is not True:reasons.append('official_exact_source_not_verified')
    coverage=dict(record_index_0based=index,source_record_sha256=digest(record),selected_accession=selected['accession'],
                  evidence_ids=expected,status='unverified' if reasons else 'admitted',reasons=list(dict.fromkeys(reasons)))
    if reasons:return None,coverage
    approved=[]
    for pair,e in zip(pairs,entries):
        approved.append(dict(evidence_id=e['evidence_id'],pair_sha256=digest(pair),semantic_concept=SEMANTICS[pair['tag']],
            evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,
            selected_filing=deepcopy(selected),taxonomy=pair['taxonomy'],tag=pair['tag'],unit=pair['unit'],
            current=deepcopy(pair['current']),prior=deepcopy(pair['prior']),release_evidence=deepcopy(e['release_evidence']),
            selected_period=deepcopy(e['selected_period']),source_status=deepcopy(source_status),
            checks=dict(cached_raw_pair_bound=True,first_usable_session_verified=True,values_match=True,periods_match=True,
                        release_within_report_filing=True,original_financial_document_directly_parsed=True),
            original_xbrl_instance_directly_verified=False,original_accession_body_directly_verified=False,
            period_starts_directly_printed=False,original_inline_duration_contexts_verified=False,continuous_security_identity_verified=False,factor_ready=False))
    return dict(record_index_0based=index,source_record_sha256=digest(record),symbol=record['symbol'],
        evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,
        candidate_cik=record['candidate_cik'],signal_date=record['signal_date'],selected_accession=selected['accession'],
        source_flags=flags(record),semantic_admitted=True,factor_ready=False,evidence_ids=expected,
        pair_sha256s=[digest(p) for p in pairs],approval=approved),coverage


def legacy_receipt(meta,raw,url):
    require(meta.get('source')==url and meta.get('sha256')==sha(raw),'legacy_receipt_mismatch')
    timestamp(meta['retrieved_at'])
    require(not any(k in meta for k in ('http_status','status','status_code')),'unexpected_legacy_http_claim')


def facts_source(design,documents,hashes):
    original=get_blob(design['facts_path'],documents,hashes);fresh=get_blob(design['official_facts_path'],documents,hashes)
    meta=get_json(design['official_facts_receipt_path'],documents,hashes);plan=get_json(design['acquisition_plan_path'],documents,hashes);request=get_json(design['facts_request_path'],documents,hashes)
    url='https://data.sec.gov/api/xbrl/companyfacts/CIK'+CIK+'.json'
    require(original==fresh and str(strict_json(fresh)['cik']).zfill(10)==CIK,'facts_exact_duplicate_mismatch')
    require(meta['http_status']==200 and meta['source']==url and meta['sha256']==sha(fresh) and meta['bytes']==len(fresh) and meta['kind']=='companyfacts','official_facts_receipt')
    timestamp(meta['retrieved_at'])
    require(meta['plan_sha256']==hashes[design['acquisition_plan_path']] and meta['collector_sha256']==hashes[design['acquisition_collector_path']]==plan['collector_sha256'],'facts_acquisition_binding')
    expected=dict(kind='companyfacts',symbol='AAPL',cik=CIK,filename='facts_CIK'+CIK+'.json',url=url)
    require(plan['requests'][request['request_index']]==expected and request['url']==url and request['http_status']==200 and request['sha256']==sha(fresh) and request['bytes']==len(fresh) and request['retrieved_at']==meta['retrieved_at'],'facts_request_binding')
    return dict(semantic_source_sufficient=True,source_standard='official_http200_exact_companyfacts',raw_facts_provenance_verified=True,exact_byte_duplicate_bound=True,http_status_recorded=True,modern_http200_provenance_verified=True,facts_path=design['facts_path'],facts_sha256=sha(original),official_facts_path=design['official_facts_path'],official_facts_receipt_path=design['official_facts_receipt_path'],source_url=url,retrieved_at=meta['retrieved_at'],filing_catalog_http_status_verified=False)


def filing_source(design,documents,hashes):
    root=get_json(design['filings_root_path'],documents,hashes);catalog=get_json(design['catalogs_path'],documents,hashes)['AAPL']
    require(str(root['cik']).zfill(10)==catalog['cik']==CIK,'catalog_identity')
    legacy_receipt(get_json(design['filings_root_receipt_path'],documents,hashes),get_blob(design['filings_root_path'],documents,hashes),'https://data.sec.gov/submissions/CIK'+CIK+'.json')
    name='CIK'+CIK+'-submissions-001.json';table=get_json(design['filings_path'],documents,hashes)
    legacy_receipt(get_json(design['filings_receipt_path'],documents,hashes),get_blob(design['filings_path'],documents,hashes),'https://data.sec.gov/submissions/'+name)
    advertised=[v for v in root['filings']['files'] if v['name']==name]
    require(len(advertised)==1 and len(table['accessionNumber'])==advertised[0]['filingCount'] and min(table['filingDate'])==advertised[0]['filingFrom'] and max(table['filingDate'])<=advertised[0]['filingTo'],'filing_archive_coverage_mismatch')
    tables={'recent':root['filings']['recent'],name:table}
    for key,values in tables.items():
        metas=[s for s in catalog['sources'] if s['name']==key]
        require(len(metas)==1,'catalog_source_count');meta=metas[0];count=len(values['accessionNumber'])
        require(meta['status']=='present' and count==meta['row_count'] and all(type(v)is list and len(v)==count for v in values.values()) and min(values['filingDate'])==meta['actual_min']==meta['range_start'] and max(values['filingDate'])==meta['actual_max']<=meta['range_end'],'catalog_table_mismatch')
    records=get_json(design['source_records']['path'],documents,hashes)
    for anchor in design['scope']:
        for i in anchor['record_indices']:
            r=records[i];begin=r['catalog']['coverage_start'];end=r['catalog']['coverage_end']
            require(any(v['start']<=begin<=end<=v['end'] for v in catalog['coverage_ranges']) and all(h['end']<begin or h['start']>end for h in catalog['holes']),'needed_record_coverage_gap')
    return tables

def calendar_index(releases):
    result={'year_ends':{}}
    for release in releases.values():
        for item in release['publication']['year_ends']:
            add_boundary(result,item,release['publication']['release_date'],release['parent_release_path'],release['parent_release_sha256'])
        if release['parsed'] is not None:
            for item in release['parsed']['year_ends']:
                add_boundary(result,item,release['publication']['release_date'],release['source_path'],release['source_sha256'])
    return result


def add_boundary(result,item,published,path,hash_value):
    row=dict(**item,release_date=published,source_path=path,source_sha256=hash_value)
    values=result['year_ends'].setdefault(str(item['fiscal_year']),[])
    require(all(x['end']==row['end'] for x in values),'conflicting_fiscal_year_boundaries')
    if row not in values:values.append(row)


def compare_pair(pair,duration,parsed,calendar_evidence,available_date):
    if parsed is None:return dict(status='unverified',reason='financial_document_not_parsed',selected_period=None)
    s=parsed['statement'];period=dict(duration_months=max(s['duration_months']),current=dict(end=s['current_end'],revenue_usd=s['current_usd']),prior=dict(end=s['prior_end'],revenue_usd=s['prior_usd']),statement=s)
    if period['duration_months']!=duration or any(period[r]['end']!=pair[r]['end'] for r in ('current','prior')):
        return dict(status='unverified',reason='required_comparable_period_not_printed',selected_period=period)
    for role in ('current','prior'):
        finish=iso(period[role]['end'])
        candidates=[x for rows in calendar_evidence['year_ends'].values() for x in rows if x['end']<finish.isoformat() and x['release_date']<=available_date]
        if not candidates:return dict(status='unverified',reason='missing_causal_prior_fiscal_year_end',selected_period=period)
        latest=max(x['end'] for x in candidates)
        evidence=sorted((x for x in candidates if x['end']==latest),key=lambda x:(x['release_date'],x['source_path'],x['basis'],x['excerpt']))[0]
        start=iso(latest)+timedelta(days=1);days=(finish-start).days+1
        allowed={3:{13,14},6:{26,27},9:{39,40},12:{52,53}}[duration]
        if days%7 or days//7 not in allowed or start.weekday()!=6 or finish.weekday()!=5 or pair[role]['start']!=start.isoformat() or pair[role]['duration_days']!=days:
            return dict(status='unverified',reason='fiscal_start_or_duration_mismatch',selected_period=period)
        period[role].update(start=start.isoformat(),duration_days=days,duration_weeks=days//7,period_start_directly_printed=False,period_start_derivation='day_after_latest_causally_available_explicit_fiscal_year_end',fiscal_boundary_evidence=evidence)
    match=all(period[r]['revenue_usd']==pair[r]['val'] for r in ('current','prior'))
    return dict(status='exact_match' if match else 'unverified',reason=None if match else 'reported_amount_mismatch',selected_period=period)


def original_receipt(raw,meta,url):
    require(urlparse(url).scheme=='https' and urlparse(url).hostname=='www.apple.com' and urlparse(url).path.startswith('/newsroom/'),'unofficial_original_host')
    require(meta.get('source_url')==meta.get('final_url')==url and type(meta.get('http_status'))is int and meta['http_status']==200 and meta.get('sha256')==sha(raw) and meta.get('size')==len(raw) and meta.get('public_unauthenticated')is True and meta.get('content_type','').split(';')[0]=='text/html','original_receipt_mismatch')
    timestamp(meta['retrieved_at'])


def transfer_receipt(raw,decoded,meta,url,host,kind):
    require(meta.get('url')==meta.get('final_url')==url and urlparse(url).hostname==host and urlparse(url).scheme=='https' and type(meta.get('status'))is int and meta['status']==200 and meta.get('public_unauthenticated')is True and meta.get('request_count')==1 and meta.get('content_type','').split(';')[0]==kind,'transfer_receipt_mismatch')
    require(meta['response_sha256']==sha(raw) and meta['response_bytes']==len(raw) and meta['decoded_sha256']==sha(decoded) and meta['decoded_bytes']==len(decoded),'transfer_hash_mismatch');timestamp(meta['retrieved_at'])
    encoding=meta['content_encoding']
    if encoding in (None,'identity',''):require(meta['decoder']=='identity' and raw==decoded,'identity_transfer_mismatch')
    elif encoding=='gzip':
        import gzip
        require(meta['decoder']=='python:gzip.decompress' and gzip.decompress(raw)==decoded,'gzip_transfer_mismatch')
    elif encoding=='br':
        import subprocess
        version=subprocess.run(['node','--version'],check=True,capture_output=True).stdout.decode().strip()
        require(meta['decoder']=='node:zlib.brotliDecompressSync' and meta['decoder_version']==version,'brotli_runtime_mismatch')
        script="let chunks=[];process.stdin.on('data',x=>chunks.push(x));process.stdin.on('end',()=>process.stdout.write(require('node:zlib').brotliDecompressSync(Buffer.concat(chunks))))"
        require(subprocess.run(['node','-e',script],input=raw,capture_output=True,check=True).stdout==decoded,'brotli_transfer_mismatch')
    else:raise ValueError('unsupported_transfer')
    if kind=='application/pdf':require(decoded.startswith(b'%PDF-'),'not_pdf')


def normalized_path(path):
    value=Path(path)
    return str(value.relative_to(Path(__file__).resolve().parents[2])) if value.is_absolute() else path


def check_completion(path,documents,hashes):
    value=get_json(path,documents,hashes)
    for name,digest_value in value['source_hashes'].items():require(hashes.get(normalized_path(name))==digest_value,'completion_source_binding')
    return value


def source_bundle(design,documents,hashes):
    c=design['collection'];plan=get_json(c['plan_path'],documents,hashes);data=get_json(c['result_path'],documents,hashes);complete=check_completion(c['completion_path'],documents,hashes);ex=get_json(c['extractions_path'],documents,hashes)
    require(data['plan_sha256']==hashes[c['plan_path']] and ex['source_result_sha256']==hashes[c['result_path']],'collection_plan_binding')
    require(plan['client_sha256']==hashes[c['client_path']] and plan['collector_sha256']==hashes[c['collector_path']] and ex['extractor_script_sha256']==hashes[c['extractor_path']],'collection_code_binding')
    require(hashes.get(plan['events_path'])==plan['events_sha256'] and hashes.get(plan['records_path'])==plan['records_sha256'],'collection_origin_binding')
    require(complete['semantic_verified'] is False and complete['factor_ready'] is False and complete['strategy_trials']==0,'collection_grade')
    require([e['target'] for e in data['entries']]==plan['targets'] and len(data['entries'])==len(ex['reports'])==design['expected_documents'] and complete['reports']==data['reports']==design['expected_documents'] and complete['candidate_records']==data['candidate_records']==design['expected_rows'],'collection_scope_mismatch')
    require(len({x['report_date'] for x in ex['reports']})==len(ex['reports']),'duplicate_extraction')
    return data,ex


def release_source(anchor,design,documents,hashes,bundle):
    data,ex=bundle;entry=data['entries'][anchor['entry_index']];target=entry['target'];base=Path(design['collection']['plan_path']).parent
    raw=get_blob(anchor['release_path'],documents,hashes);meta=get_json(anchor['receipt_path'],documents,hashes);url=anchor['release_url']
    require(digest(entry)==anchor['entry_sha256'] and entry['path']==anchor['release_path'] and entry['receipt']==meta,'source_entry_binding')
    require(str(base/meta['response_path'])==anchor['response_path'] and str(base/meta['decoded_path'])==anchor['release_path'] and str((base/meta['decoded_path']).with_suffix('.meta.json'))==anchor['receipt_path'],'receipt_path_mismatch')
    transfer_receipt(get_blob(anchor['response_path'],documents,hashes),raw,meta,url,'www.apple.com','application/pdf')
    require(target['record_indices']==anchor.get('record_indices',[]) and target['report_date']==anchor['report_date'] and target['release_date']==anchor['release_date'] and target['url']==url and target['attachment_role']==anchor['attachment_role'],'source_target_mismatch')
    parent=get_blob(target['release_path'],documents,hashes);pm=get_json(target['release_receipt_path'],documents,hashes)
    original_receipt(parent,pm,target['release_url']);publication=parser.publication(parent,target['release_url'])
    require(sha(parent)==target['release_sha256'] and hashes[target['release_receipt_path']]==target['release_receipt_sha256'] and publication['release_date']==anchor['release_date'],'parent_release_binding')
    links=[dict(href=a['href'],label=a.get_text(' ',strip=True),url=urljoin(pm['final_url'],a['href'])) for a in BeautifulSoup(parent,'html.parser').find_all('a',href=True)]
    require(target['anchor'] in links and target['anchor']['label']=='View PDF' and target['anchor']['url']==url,'exact_pdf_outlink_missing')
    extraction=[e for e in ex['reports'] if e['report_date']==anchor['report_date']]
    require(len(extraction)==1 and extraction[0]==anchor['extraction'],'extraction_mapping_mismatch');x=extraction[0]
    require(x['source_path']==anchor['release_path'] and x['source_sha256']==sha(raw) and hashes[x['pages_path']]==x['pages_sha256'] and hashes[x['text_path']]==x['text_sha256'],'extraction_binding_mismatch')
    return dict(source_path=anchor['release_path'],source_sha256=sha(raw),source_url=url,receipt_path=anchor['receipt_path'],receipt_sha256=hashes[anchor['receipt_path']],receipt=deepcopy(meta),retrieved_at=meta['retrieved_at'],parent_release_path=target['release_path'],parent_release_sha256=sha(parent),parent_receipt_path=target['release_receipt_path'],parent_receipt_sha256=hashes[target['release_receipt_path']],parent_url=target['release_url'],pdf_anchor=deepcopy(target['anchor']),publication=publication,official_original_http200_verified=True,original_release_pdf_outlink_verified=True,raw_facts_provenance_verified=True,filing_catalog_http_status_verified=False,historical_web_snapshot_verified=False,first_public_time_verified=False,publication_time_precision='date_only',source_class=SOURCE_CLASS,evidence_class=EVIDENCE_CLASS,format='pdf')


def pdf_payload(anchor,documents,hashes):
    import subprocess
    raw=get_blob(anchor['release_path'],documents,hashes)
    python='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    script="""import io,json,sys,pdfplumber
with pdfplumber.open(io.BytesIO(sys.stdin.buffer.read())) as pdf:
 p=pdf.pages[0]
 print(json.dumps(dict(version=pdfplumber.__version__,page_count=len(pdf.pages),text=p.extract_text(layout=False) or '',chars=[{k:c[k] for k in ('text','x0','x1','top','bottom')} for c in p.chars],bbox=list(p.bbox))))
"""
    process=subprocess.run([python,'-B','-W','error','-c',script],input=raw,capture_output=True,check=True)
    require(not process.stderr,'new_pdf_extraction_warning');payload=strict_json(process.stdout)
    require(payload['version']=='0.11.9' and payload['page_count']==anchor['extraction']['page_count'],'pdf_extractor_or_page_count')
    return payload,dict(extractor='pdfplumber',version=payload['version'],page_count=payload['page_count'],original_pdf_sha256=sha(raw),page1_text_sha256=sha(payload['text'].encode()),page1_geometry_sha256=digest(dict(chars=payload['chars'],bbox=payload['bbox'])),stderr_empty=True,legacy_pypdf_companion_not_used_for_financial_parse=True)


def identity_sources(design,documents,hashes):
    config=design['identity_sources'];complete=check_completion(config['completion_path'],documents,hashes)
    require(complete['historical_identity_anchors_verified'] is True and complete['continuous_security_identity_verified'] is False,'identity_completion_grade')
    found=[]
    for anchor in config['anchors']:
        meta=get_json(anchor['receipt_path'],documents,hashes);raw=get_blob(anchor['path'],documents,hashes)
        transfer_receipt(get_blob(anchor['response_path'],documents,hashes),raw,meta,anchor['url'],'www.nasdaqtrader.com','text/html')
        soup=BeautifulSoup(raw,'html.parser');headers=soup.select('#NewsHeader');bodies=soup.select('#HTA2 > #MainContainer')
        require(len(headers)==len(bodies)==1,'exchange_document_scope');header=parser.norm(headers[0].get_text(' ',strip=True));body=parser.norm(bodies[0].get_text(' ',strip=True))
        if anchor['date']=='2014-06-04':
            require('Wednesday, June 4, 2014' in header and 'Data Technical News #2014 - 16' in header and 'Apple Inc. (Ticker: AAPL)' in body and '7-1 stock split' in body,'exchange_2014_identity')
            rows=[[parser.norm(c.get_text(' ',strip=True)) for c in r.find_all(['td','th'],recursive=False)] for r in bodies[0].select('table tr')]
            require(rows==[['Issue Name','NASDAQ Ticker','NASDAQ Market Category','Record Date','Payment Date','Ex-Date','Round Lot Size'],['Apple Inc.','AAPL','NASDAQ Global Select Market','06/02/2014','06/06/2014','06/09/2014','100 Shares']],'exchange_2014_identity_table')
        else:require(anchor['date']=='2019-03-12' and 'Tuesday, March 12, 2019' in header and 'Apple Inc. (Nasdaq: AAPL) debt securities' in body,'exchange_2019_identity')
        found.append(dict(**anchor,sha256=sha(raw),header=header,identity_at_anchor_verified=True,equity_listing_transfer_claimed=False,continuous_security_identity_verified=False))
    split=config['split'];raw=get_blob(split['path'],documents,hashes);meta=get_json(split['receipt_path'],documents,hashes);original_receipt(raw,meta,split['url']);publication=parser.publication(raw,split['url'])
    soup=BeautifulSoup(raw,'html.parser')
    for x in soup(['script','style','nav','footer']):x.decompose()
    text=parser.norm(soup.get_text(' ',strip=True))
    require(publication['release_date']=='2020-07-30' and 'four-for-one stock split' in text and 'common stock' in text and 'August 31, 2020' in text,'issuer_split_evidence')
    return dict(anchors=found,split=dict(**split,sha256=sha(raw),release_date=publication['release_date'],common_stock_split_verified=True),candidate_cik=CIK,continuous_security_identity_verified=False)


def record_identity(sources,available):
    anchors=[deepcopy(a) for a in sources['anchors'] if a['date']<=available]
    require(anchors and any(a['date']=='2014-06-04' for a in anchors),'missing_causal_security_identity')
    return dict(candidate_cik=CIK,issuer='Apple Inc.',symbol='AAPL',historical_anchors=anchors,split=deepcopy(sources['split']) if sources['split']['release_date']<=available else None,record_level_security_identity_verified=True,continuous_security_identity_verified=False,historical_daily_security_identity_verified=False)
