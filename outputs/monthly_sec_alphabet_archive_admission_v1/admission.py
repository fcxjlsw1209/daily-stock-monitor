"""One-stage archive source, statement and original-record evidence checks."""
import base64
from bisect import bisect_right
import calendar
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import quote

_spec=importlib.util.spec_from_file_location('alphabet_statement',Path(__file__).with_name('parser.py'))
parser=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(parser)
POLICY='monthly_sec_alphabet_archive_admission_v1'
CIK='0001652044'
SOURCE_CLASS='issuer_official_google_cloud_archive_copy'
EVIDENCE_CLASS='primary_document_reconstruction'
SEMANTICS={'Revenues':'consolidated_revenues','RevenueFromContractWithCustomerExcludingAssessedTax':'consolidated_revenues'}
FLAGS=('factor_ready','identity_verified','semantic_verified')
README_URL='https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/search/web-app/README.md'
BUCKET='cloud-samples-data'
PREFIX='gen-app-builder/search/alphabet-investor-pdfs/'
INDEX_URL='https://storage.googleapis.com/storage/v1/b/cloud-samples-data/o?prefix=gen-app-builder%2Fsearch%2Falphabet-investor-pdfs%2F&maxResults=1000'


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


def receipt(meta,blob,url,content_type):
    require(meta.get('source_url')==meta.get('final_url')==url and type(meta.get('http_status')) is int and
            meta['http_status']==200 and meta.get('public_unauthenticated') is True and
            meta.get('sha256')==sha(blob) and type(meta.get('size')) is int and meta['size']==len(blob),
            'source_receipt_mismatch')
    require(meta.get('content_type','').split(';')[0]==content_type,'source_content_type_mismatch')
    timestamp(meta['retrieved_at'])


def archive_context(design,documents,hashes):
    readme=get_blob(design['readme_path'],documents,hashes)
    receipt(get_json(design['readme_receipt_path'],documents,hashes),readme,README_URL,'text/plain')
    marker='- [Alphabet Earnings Reports](https://abc.xyz/investor/)\n      - `gs://'+BUCKET+'/'+PREFIX[:-1]+'`'
    require(readme.decode().count(marker)==1,'official_readme_archive_relation_missing')
    raw=get_blob(design['object_index_path'],documents,hashes)
    receipt(get_json(design['object_index_receipt_path'],documents,hashes),raw,INDEX_URL,'application/json')
    listing=strict_json(raw)
    require(listing.get('kind')=='storage#objects' and not listing.get('nextPageToken'),'incomplete_object_index')
    items=listing['items'];names=[v['name'] for v in items]
    require(len(names)==len(set(names)),'duplicate_archive_object')
    return {v['name']:v for v in items}


def archive_source(anchor,design,documents,hashes,objects,events,manifest):
    path=anchor['release_path'];raw=get_blob(path,documents,hashes)
    require(raw.startswith(b'%PDF-'),'invalid_pdf_magic')
    meta=get_json(anchor['receipt_path'],documents,hashes)
    name=PREFIX+f"{anchor['fiscal_year']}Q{anchor['fiscal_quarter']}_alphabet_earnings_release.pdf"
    url='https://storage.googleapis.com/'+BUCKET+'/'+name
    require(anchor['release_url']==url,'archive_url_name_mismatch')
    receipt(meta,raw,url,'application/pdf')
    obj=objects[name]
    generation=obj['generation']
    require(type(generation) is str and generation.isdigit() and obj.get('kind')=='storage#object' and
            obj.get('bucket')==BUCKET and obj.get('name')==name and obj.get('id')==BUCKET+'/'+name+'/'+generation and
            obj.get('contentType')=='application/pdf','archive_object_identity_mismatch')
    require(obj.get('selfLink')=='https://www.googleapis.com/storage/v1/b/'+BUCKET+'/o/'+quote(name,safe='') and
            obj.get('mediaLink')=='https://storage.googleapis.com/download/storage/v1/b/'+BUCKET+'/o/'+quote(name,safe='')+
                '?generation='+generation+'&alt=media','archive_object_links_mismatch')
    require(obj.get('size')==str(len(raw)) and obj.get('md5Hash')==base64.b64encode(hashlib.md5(raw).digest()).decode(),
            'archive_object_content_mismatch')
    timestamp(obj['timeCreated']);timestamp(obj['updated'])
    matched=[e for e in events if e.get('evidence_path')==path]
    require(len(matched)==1 and digest(matched[0])==anchor['event_sha256'],'event_binding_mismatch')
    event=matched[0]
    require(event['ticker']=='GOOGL' and event['issuer_name_at_release']=='Alphabet Inc.' and
            event['source_class']==SOURCE_CLASS and event['source_url']==url and event['evidence_sha256']==sha(raw) and
            event['release_date']==anchor['release_date'] and event['fiscal_period_end_actual']==anchor['report_date'] and
            (event['fiscal_year'],event['fiscal_quarter'])==(anchor['fiscal_year'],anchor['fiscal_quarter']),
            'event_mapping_mismatch')
    require(event['archive_provenance_evidence_path']==design['readme_path'] and
            event['archive_provenance_sha256']==hashes[design['readme_path']] and
            event['archive_provenance_url']=='https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/web-app/README.md',
            'event_archive_relation_mismatch')
    require(event['publication_timestamp'] is None and event['precision']=='date_only' and
            event['first_public_time_verified'] is False,'unsupported_original_publication_claim')
    entries=[e for e in manifest if e.get('path')==path]
    require(entries==[dict(source_url=url,path=path,sha256=sha(raw))],'evidence_manifest_mismatch')
    return dict(evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,source_path=path,source_sha256=sha(raw),source_url=url,
        text_path=anchor['text_path'],text_sha256=hashes[anchor['text_path']],receipt=deepcopy(meta),
        readme_path=design['readme_path'],readme_sha256=hashes[design['readme_path']],
        object_index_path=design['object_index_path'],object_index_sha256=hashes[design['object_index_path']],
        archive_object=deepcopy(obj),original_ir_direct_link_verified=False,
        archive_upload_used_as_original_availability=False,original_historical_http_success_claimed=False)


def filing_source(design,documents,hashes):
    root=get_json(design['filings_root_path'],documents,hashes)
    require(str(root['cik']).zfill(10)==CIK,'filing_root_cik_mismatch')
    for path,meta_path,name in ((design['filings_root_path'],design['filings_root_receipt_path'],'CIK'+CIK+'.json'),
                                (design['filings_path'],design['filings_receipt_path'],'CIK'+CIK+'-submissions-001.json')):
        raw=get_blob(path,documents,hashes);meta=get_json(meta_path,documents,hashes)
        require(meta.get('source')=='https://data.sec.gov/submissions/'+name and
                meta.get('http_status')==200 and meta.get('sha256')==sha(raw) and meta.get('bytes')==len(raw),
                'filing_receipt_mismatch')
        timestamp(meta['retrieved_at'])
    advertised=[v for v in root['filings']['files'] if v['name']=='CIK'+CIK+'-submissions-001.json']
    table=get_json(design['filings_path'],documents,hashes)
    require(len(advertised)==1 and len(table['accessionNumber'])==advertised[0]['filingCount'] and
            all(len(v)==len(table['accessionNumber']) for v in table.values()) and
            min(table['filingDate'])==advertised[0]['filingFrom'] and max(table['filingDate'])==advertised[0]['filingTo'],
            'filing_archive_coverage_mismatch')
    return table


def validate_pair(record,pair,facts,sessions,anchor,filings):
    selected=record['catalog']['selected'];end=iso(anchor['report_date'])
    require(record['symbol']=='GOOGL' and record['candidate_cik']==CIK and str(facts['cik']).zfill(10)==CIK,'issuer_mismatch')
    require(record['catalog']['status']=='selected' and record['catalog']['coverage_complete'] is True and
            selected['metadata_issues']==[] and selected['supported_form'] is True and
            record['pairing']['status']=='paired' and record['pairing']['structural_pair_available'] is True,
            'original_structural_not_ready')
    require(selected['accession']==selected['accessionNumber']==anchor['accession'] and
            selected['form']==anchor['form'] and selected['reportDate']==selected['report_date']==anchor['report_date'] and
            selected['filingDate']==selected['filing_date']==anchor['filing_date'],'selected_metadata_mismatch')
    source_ids=[i for i,a in enumerate(filings['accessionNumber']) if a==anchor['accession']]
    require(len(source_ids)==1 and selected['sources']==[dict(table='CIK'+CIK+'-submissions-001.json',row=source_ids[0])],
            'selected_filing_source_row_mismatch')
    j=source_ids[0]
    require(all(filings[k][j]==selected[k] for k in ('accessionNumber','filingDate','reportDate','form')),'raw_filing_mismatch')
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


def compare_pair(pair,duration,parsed):
    if parsed is None:return dict(status='unverified',reason='release_not_parsed',selected_period=None)
    found=[v for v in parsed['statement']['periods'] if v['duration_months']==duration and
           all(v[r]['end']==pair[r]['end'] for r in ('current','prior'))]
    if len(found)!=1:return dict(status='unverified',reason='required_comparable_period_not_printed',selected_period=None)
    equal=all(found[0][r]['revenue_usd']==pair[r]['val'] for r in ('current','prior'))
    return dict(status='exact_match' if equal else 'unverified',reason=None if equal else 'reported_amount_mismatch',
                selected_period=deepcopy(found[0]))


def admit_record(index,record,entries,source_status):
    pairs=record['pairing']['pairs'];selected=record['catalog']['selected']
    expected=[digest(dict(accession=selected['accession'],pair=v)) for v in pairs]
    require(len(pairs) in (1,2) and [e['evidence_id'] for e in entries]==expected,'pair_order_or_count_mismatch')
    for pair,e in zip(pairs,entries):
        require(e['pair']==pair and e['pair_sha256']==digest(pair) and e['accession']==selected['accession'],'evidence_pair_mismatch')
    reasons=[e['reason'] for e in entries if e['status']!='exact_match']
    if not source_status['semantic_source_sufficient']:reasons.append('facts_source_provenance_not_admitted')
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
                        release_within_report_filing=True,raw_pdf_reextracted=True),
            original_xbrl_instance_directly_verified=False,original_accession_body_directly_verified=False,
            period_starts_directly_printed=False,continuous_security_identity_verified=False,factor_ready=False))
    return dict(record_index_0based=index,source_record_sha256=digest(record),symbol=record['symbol'],
        evidence_class=EVIDENCE_CLASS,source_class=SOURCE_CLASS,
        candidate_cik=record['candidate_cik'],signal_date=record['signal_date'],selected_accession=selected['accession'],
        source_flags=flags(record),semantic_admitted=True,factor_ready=False,evidence_ids=expected,
        pair_sha256s=[digest(p) for p in pairs],approval=approved),coverage
