"""Offline exact-accession body inventory. No dates-to-lags or market inputs.

Print --rules-only, then bind both digests to --write. Existing outputs are
write-once. Only source metadata are searched; body bytes are read only after an
explicit accession mapping. A local PDF or index hit is not original admission.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse, parse_qs

from after156_event_scope_local_review import Sources, require

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after157_annual_original_body_inventory.py'
TEST='work/test_after157_annual_original_body_inventory.py'
BASE='work/after157_annual_pair_source_feasibility.json'
BASE_SHA='4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242'
OUT='work/after157_annual_original_body_inventory.json'
ERAS=(('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122))
ACC_KEYS=('accessionNumber','accession','accession_number','accn','catalog_accession')
PATH_PAIRS=(('raw_path','raw_sha256'),('path','sha256'),('evidence_path','sha256'),
            ('source_path','source_sha256'),('response_path','response_sha256'))
RULES=dict(version=1,
    population='All5978 original slots, with324 distinct original accessions from266 Tier2 pairs. No reselection or dropping unavailable slots.',
    metadata_scope='Existing JSON receipt/meta/provenance, manifest/catalog/COLLECTION/SOURCE_INVENTORY/EVIDENCE files in monthly output folders except monitor folders; index_union/sources and dynamic_top100/registry_sec; plus output JSON inputs already bound by the annual pair inventory. Filename inventory is fixed before reading metadata. No raw body scan or unrelated-issuer body reads.',
    match='Full SEC Archives/edgar/data CIK+accession+primaryDocument URL, or an explicit accession mapping in one catalog record and its nested source object. Ticker, company name, period end and approximate-year matching are prohibited.',
    references='A URL inside prose or a filing index does not prove body bytes. Paths need an explicit SHA and local source/catalog link. Relative paths resolve against the catalog directory or repository root only; ambiguity blocks.',
    originals='Exact SEC primary URL plus SHA-verified local HTML/SGML/PDF, successful HTTP receipt and complete-body structural evidence produces an original-body candidate for adjudication, never automatic normal-year admission. Non-SEC explicit-accession mirrors stay pending identity unless separately reviewed. FailedHTTP, denial pages, indexes, exhibits and extracted snippets remain unavailable.',
    mirror='An explicit catalog mapping such as source_identity.catalog_accession plus target.raw_path may locate a local PDF, but cannot certify the publisher chain or establish a complete annual pair.',
    timing='Retain the already selected t-42 pair and its cutoff. Current retrieval dates are not historical-first-public proofs; no time fields are reinterpreted and no new clock/identity admission is made.',
    coverage='Report exact-source body candidates, any explicitly accession-mapped bodies, and missing bodies per accession/pair/all original slots. Existing source-only Tier2 is not upgraded.',
    nonactions='No network, lag or lag-change calculation, prices, ranks, factors, orders, returns, models, strategy registration, frozen edits or observer access.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def accession(value):
    if not isinstance(value,str):return None
    value=value.strip()
    if re.fullmatch(r'\d{10}-\d{2}-\d{6}',value):return value
    if re.fullmatch(r'\d{18}',value):return value[:10]+'-'+value[10:12]+'-'+value[12:]
    return None


def sec_identity(url):
    if not isinstance(url,str):return None
    parsed=urlparse(html.unescape(url))
    if parsed.hostname not in ('sec.gov','www.sec.gov','archives.sec.gov'):return None
    path=unquote(parsed.path)
    if path.lower() in ('/ixviewer/doc/action','/ix','/ixviewer/ix.html'):
        doc=parse_qs(parsed.query).get('doc',[None])[0]
        if not doc:return None
        path=unquote(doc)
    match=re.fullmatch(r'/Archives/edgar/data/(\d+)/(\d{18})/(.+)',path,re.I)
    if not match:return None
    return dict(cik=match[1].zfill(10),accession=accession(match[2]),document=match[3])


def metadata_paths(root,base):
    output=root/'outputs'; found=set()
    roots=[p for p in output.iterdir() if p.is_dir() and p.name.startswith('monthly_') and 'monitor' not in p.name.lower()]
    roots += [output/'index_union'/'sources',output/'dynamic_top100'/'registry_sec']
    for folder in roots:
        if not folder.is_dir():continue
        for p in folder.rglob('*.json'):
            if any('monitor' in part.lower() for part in p.relative_to(output).parts):continue
            name=p.name.lower()
            if name.endswith(('.meta.json','.receipt.json','.provenance.json')) or 'manifest' in name or 'catalog' in name or name in ('collection.json','source_inventory.json','evidence.json'):
                found.add(str(p.relative_to(root)))
    for p in base['input_hashes']:
        if p.startswith('outputs/') and p.endswith('.json') and 'monitor' not in p.lower():found.add(p)
    return sorted(found)


def local_paths(root,catalog,value):
    if not isinstance(value,str) or value.startswith(('http:','https:')):return []
    p=Path(value)
    if '..' in p.parts or 'CloudStorage' in p.parts:return []
    candidates=[p] if p.is_absolute() else [root/p,root/Path(catalog).parent/p]
    return sorted({str(p.relative_to(root)) for p in candidates if p.is_relative_to(root) and p.is_file() and p.resolve()==p})


def records(value,targets,pointer='$',inherited=None):
    """Only record-local explicit accessions propagate to nested source objects."""
    if isinstance(value,list):
        for i,item in enumerate(value):yield from records(item,targets,pointer+'[%d]'%i,inherited)
        return
    if not isinstance(value,dict):return
    context=set(inherited or ())
    own={accession(value[k]) for k in ACC_KEYS if k in value}-{None}
    identity=value.get('source_identity')
    if isinstance(identity,dict) and accession(identity.get('catalog_accession')):
        own.add(accession(identity['catalog_accession']))
    if own:context=own
    urls=[]
    for k,v in value.items():
        if isinstance(v,str) and v.startswith(('https://','http://')):
            urls.append(dict(field=k,url=v,sec=sec_identity(v)))
    mapped=context & set(targets)
    mapped |= {u['sec']['accession'] for u in urls if u['sec'] and u['sec']['accession'] in targets}
    if mapped:yield pointer,value,sorted(mapped),urls
    for k,item in value.items():
        key_acc=accession(k)
        next_context={key_acc} if key_acc else context
        yield from records(item,targets,pointer+'.'+k,next_context)


def body_shape(blob):
    if blob.startswith(b'%PDF-'):
        return dict(format='PDF',complete_body_candidate=b'%%EOF' in blob[-2048:],
            structure='PDF signature and EOF only; cover/context and exact identity still require review')
    text=blob.decode('utf-8',errors='replace')
    if text.lstrip().startswith(('{','[')):
        return dict(format='JSON_INDEX_OR_EXTRACT',complete_body_candidate=False,structure='Structured metadata is not the original body')
    plain=html.unescape(re.sub(r'<[^>]*>',' ',text))
    plain=re.sub(r'\s+',' ',plain).upper()
    denied=any(s in plain for s in ('UNDECLARED AUTOMATED TOOL','YOUR REQUEST ORIGINATES FROM','ACCESS DENIED','REQUEST RATE THRESHOLD EXCEEDED'))
    html_body=bool(re.search(r'<(?:!doctype\s+html|html|sec-document)\b',text,re.I))
    full=html_body and not denied and bool(re.search(r'FORM\s+10\s*-\s*K',plain)) and all(x in plain for x in ('PART I','PART II','SIGNATURES'))
    return dict(format='HTML_OR_SGML' if html_body else 'TEXT_OR_OTHER',complete_body_candidate=full,
        denial_page=denied,structure='Requires Form10-K, PartsI/II and Signatures; this is a retrieval screen, not completeness or annual-period adjudication')


def inspect_reference(src,catalog,pointer,node,acc,urls,target):
    refs=[]
    for pk,hk in PATH_PAIRS:
        h=node.get(hk)
        if isinstance(node.get(pk),str) and isinstance(h,str) and re.fullmatch(r'[0-9a-f]{64}',h):
            refs.append((node[pk],h))
    # Standard receipt sidecars may bind an adjacent file without a path field.
    if not refs and isinstance(node.get('sha256'),str) and urls:
        p=src.root/catalog
        for suffix in ('.meta.json','.provenance.json'):
            if p.name.endswith(suffix):
                base=p.name[:-len(suffix)]
                for adjacent in p.parent.glob(base+'*'):
                    if adjacent.is_file() and not adjacent.name.endswith('.json'):
                        refs.append((str(adjacent),node['sha256']))
    if not refs:return []
    strict=[u for u in urls if u['sec'] and u['sec']['accession']==acc and u['sec']['cik']==target['cik'] and u['sec']['document']==target['primaryDocument']]
    sec_other=[u for u in urls if u['sec'] and u['sec']['accession']==acc and u not in strict]
    identity_conflict=any(u['sec'] and (u['sec']['accession']!=acc or u['sec']['cik']!=target['cik']) for u in urls)
    http=node.get('http_status',node.get('status_code'))
    blocked= node.get('accepted_original_source') is False or (isinstance(http,int) and not 200<=http<300)
    found=[]
    for path,h in sorted(set(refs)):
        paths=local_paths(src.root,catalog,path)
        item=dict(catalog=catalog,pointer=pointer,declared_path=path,declared_sha256=h,
            urls=[u['url'] for u in urls],http_status=http,exact_sec_primary_url=bool(strict),
            explicit_accession_mapping=True,selected_accession=acc,local_paths=paths,
            complete_body_candidate=False,exact_source_body_candidate=False,reasons=[])
        if len(paths)!=1:
            item['reasons'].append('MISSING_LOCAL_BODY' if not paths else 'AMBIGUOUS_LOCAL_PATH');found.append(item);continue
        path=paths[0]
        try:blob=src.read_bytes(path,h)
        except ValueError as exc:
            item['reasons'].append('BODY_SHA_MISMATCH');item['detail']=str(exc);found.append(item);continue
        shape=body_shape(blob);item.update(path=path,bytes=len(blob),sha256=h,shape=shape,body_sha_verified=True)
        if blocked:item['reasons'].append('FAILED_HTTP_OR_EXPLICITLY_UNACCEPTED_RESPONSE')
        if identity_conflict:item['reasons'].append('SOURCE_URL_CONTRADICTS_SELECTED_ACCESSION_OR_CIK')
        if sec_other and not strict:item['reasons'].append('SAME_ACCESSION_OTHER_DOCUMENT_OR_CIK')
        if not shape['complete_body_candidate']:item['reasons'].append('NOT_A_COMPLETE_BODY_CANDIDATE')
        item['complete_body_candidate']=shape['complete_body_candidate'] and not blocked and not identity_conflict and not(sec_other and not strict)
        item['exact_source_body_candidate']=item['complete_body_candidate'] and bool(strict) and isinstance(http,int) and 200<=http<300
        if item['complete_body_candidate'] and not item['exact_source_body_candidate']:item['reasons'].append('EXPLICIT_MAPPING_ONLY_PUBLISHER_OR_ORIGINAL_IDENTITY_PENDING')
        found.append(item)
    return found


def selected_targets(base):
    ids={i for r in base['rows'] if r['tier2'] for i in r['selected_event_ids']}
    targets={}
    for eid in sorted(ids):
        e=base['selected_original_and_amendment_metadata'][eid]
        require(e['form']=='10-K','Only previously selected original10-K permitted')
        acc=e['accessionNumber']
        if acc in targets:require(all(targets[acc][k]==e[k] for k in ('cik','primaryDocument','reportDate','filingDate')),'Selected accession conflict')
        else:targets[acc]=dict(e,event_ids=[])
        targets[acc]['event_ids'].append(eid)
    return targets


def summarize(rows):
    out={}
    for name,lo,hi in ERAS:
        group=[r for r in rows if lo<=r['signal_index']<hi]
        out[name]=dict(all_slots=len(group),stock_slots=sum(not r['is_etf'] for r in group),
            prior_tier2_slots=sum(r['prior_tier2'] for r in group),
            both_exact_source_body_candidates=sum(r['both_exact_source_body_candidates'] for r in group),
            both_any_explicitly_mapped_body_candidates=sum(r['both_any_explicitly_mapped_body_candidates'] for r in group),
            any_explicitly_mapped_body_candidate=sum(r['any_explicitly_mapped_body_candidate'] for r in group))
    return out


def run(root,rules_sha,list_sha):
    require(rules_sha==digest(RULES),'Rules digest mismatch')
    src=Sources(root);base=src.read(BASE,BASE_SHA);paths=metadata_paths(root,base)
    require(digest(paths)==list_sha,'Metadata filename inventory changed')
    for p in (SELF,TEST,'work/after156_event_scope_local_review.py','AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):src.read_bytes(p)
    targets=selected_targets(base);findings=defaultdict(list);hits=defaultdict(list);parse_failures=[]
    for p in paths:
        blob=src.read_bytes(p,base['input_hashes'].get(p))
        try:value=json.loads(blob)
        except (ValueError,UnicodeError) as exc:
            parse_failures.append(dict(path=p,reason=type(exc).__name__));continue
        for pointer,node,matched,urls in records(value,targets):
            for acc in matched:
                hits[acc].append(dict(catalog=p,pointer=pointer))
                findings[acc]+=inspect_reference(src,p,pointer,node,acc,urls,targets[acc])
    accessions={};eid_to_acc={}
    for acc,target in sorted(targets.items()):
        found=list({digest(x):x for x in findings[acc]}.values())
        exact=any(x['exact_source_body_candidate'] for x in found)
        anybody=any(x['complete_body_candidate'] for x in found)
        accessions[acc]=dict(selected_metadata=target,
            expected_sec_primary_url='https://www.sec.gov/Archives/edgar/data/%d/%s/%s'%(int(target['cik']),acc.replace('-',''),target['primaryDocument']),
            metadata_hits=hits[acc],references=found,exact_source_body_candidate=exact,
            any_explicitly_mapped_body_candidate=anybody,
            status='EXACT_SOURCE_BODY_CANDIDATE_FOR_CONTEXT_REVIEW' if exact else 'MAPPED_LOCAL_BODY_IDENTITY_PENDING' if anybody else 'NO_VERIFIED_COMPLETE_BODY_IN_SEARCHED_CATALOGS',
            normal_consecutive_annual_admission=False)
        eid_to_acc.update({eid:acc for eid in target['event_ids']})
    rows=[];pairs={}
    for r in base['rows']:
        accs=[eid_to_acc[x] for x in r['selected_event_ids']] if r['tier2'] else []
        exact=len(accs)==2 and all(accessions[a]['exact_source_body_candidate'] for a in accs)
        both=len(accs)==2 and all(accessions[a]['any_explicitly_mapped_body_candidate'] for a in accs)
        anybody=any(accessions[a]['any_explicitly_mapped_body_candidate'] for a in accs)
        row=dict(signal_index=r['signal_index'],candidate_index=r['candidate_index'],symbol=r['symbol'],is_etf=r['is_etf'],
            prior_tier2=r['tier2'],information_cutoff=r['information_cutoff'],selected_accessions=accs,
            both_exact_source_body_candidates=exact,both_any_explicitly_mapped_body_candidates=both,
            any_explicitly_mapped_body_candidate=anybody,normal_consecutive_annual_admission=False,
            unavailable_reasons=r['reasons'] if not r['tier2'] else ['ORIGINAL_BODY_OR_NORMAL_PERIOD_CONTEXT_PENDING'])
        rows.append(row)
        if accs:pairs['|'.join(accs)]=dict(accessions=accs,both_exact_source_body_candidates=exact,both_any_explicitly_mapped_body_candidates=both)
    require(len(rows)==5978 and len(targets)==324 and len(pairs)==266,'Frozen target denominator changed')
    src.recheck()
    return dict(schema_version=1,status='EXACT_ACCESSION_LOCAL_BODY_INVENTORY_NO_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(),runtime_python=sys.version,
        rules=RULES,rules_sha256=rules_sha,metadata_filename_sha256=list_sha,metadata_files=paths,
        input_hashes=dict(sorted(src.hashes.items())),metadata_parse_failures=parse_failures,
        counts=dict(metadata_files=len(paths),selected_accessions=len(targets),selected_pairs=len(pairs),
            accessions_with_metadata_hits=sum(bool(hits[a]) for a in targets),
            exact_source_body_candidates=sum(x['exact_source_body_candidate'] for x in accessions.values()),
            any_explicitly_mapped_body_candidates=sum(x['any_explicitly_mapped_body_candidate'] for x in accessions.values()),
            normal_consecutive_pairs_admitted=0),summary=summarize(rows),accessions=accessions,pairs=pairs,rows=rows,
        activity=dict(network_requests=0,lag_values=0,lag_changes=0,prices=0,factors=0,ranks=0,orders=0,returns=0,new_trials=0,observer_access=0,frozen_writes=0),
        limitations=['Search is exhaustive only over the stated existing metadata filename scope; opaque uncatalogued files are not searched.',
            'Explicitly mapped local bodies can support a later exact-source and ordinary-period review. Body structural markers do not themselves prove consecutive fiscal periods.',
            'Source identity, timing and annual period gates remain unchanged; retrieval is not historical publication.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rules-only',action='store_true');parser.add_argument('--write',action='store_true')
    parser.add_argument('--rules-sha256');parser.add_argument('--metadata-list-sha256');args=parser.parse_args()
    require(args.rules_only!=args.write,'Choose rules-only or write')
    if args.rules_only:
        base=Sources(ROOT).read(BASE,BASE_SHA);paths=metadata_paths(ROOT,base)
        print(json.dumps(dict(rules=RULES,rules_sha256=digest(RULES),metadata_files=len(paths),metadata_list_sha256=digest(paths))))
    else:
        require(not(ROOT/OUT).exists(),'Write-once output exists')
        value=run(ROOT,args.rules_sha256,args.metadata_list_sha256)
        with (ROOT/OUT).open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);handle.write('\n')
        print(json.dumps(dict(path=OUT,sha256=hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest(),counts=value['counts'],summary=value['summary'])))


if __name__=='__main__':main()
