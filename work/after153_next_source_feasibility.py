"""Bounded PLTR/PYPL/TMO source metadata inventory; no amount extraction.

Reads the final three predeclared text-lead issuers from the immutable census.
Only titles, dates, units, labels, source identities and masked notes are emitted.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import after152_next_source_feasibility as prior

ROOT=Path(__file__).resolve().parents[1]
OUT='work/after153_next_source_feasibility.json'
KIND_OUT='work/after153_batch4_source_kind_review.json'
CENSUS=prior.CENSUS
CONCEPT=prior.CONCEPT
SESSIONS=prior.SESSIONS
SYMBOLS=('PLTR','PYPL','TMO')
EXPECTED_COUNTS={'PLTR':6,'PYPL':11,'TMO':11}
IDENTITY={
    'PLTR':re.compile(r'Palantir Technologies(?:,?\s+Inc\.?)?|(?:NYSE|NASDAQ|Nasdaq)\s*[:(]\s*PLTR\)?',re.I),
    'PYPL':re.compile(r'PayPal Holdings,?\s+Inc\.?|(?:NYSE|NASDAQ|Nasdaq)\s*[:(]\s*PYPL\)?',re.I),
    'TMO':re.compile(r'Thermo Fisher Scientific(?:,?\s+Inc\.?)?|(?:NYSE|NASDAQ|Nasdaq)\s*[:(]\s*TMO\)?',re.I),
}
ORIGINAL_KINDS={'issuer_original_release_full_bytes','issuer_original_earnings_release_html',
                'issuer_distributed_original_prnewswire_release'}


def check(ok,message):
    if not ok:raise ValueError(message)


def read_local(name,pins,expected=None):
    p=ROOT/name
    check(not p.is_symlink() and p.resolve().is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Nonlocal source')
    raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    check(expected is None or expected==digest,'Source SHA mismatch: '+name)
    check(name not in pins or pins[name]==digest,'Source changed: '+name)
    pins[name]=digest
    return raw


def batch4_kind_review(census):
    pins={}
    source_plan='outputs/monthly_financial_coverage_batch4_v1/PLAN.json'
    plan=json.loads(read_local(source_plan,pins,'9b1016e51ad22eae4faf13629afeae7e06a8f7efe3b2f0e8d7e77dce0d232643'))
    context_path='work/after152_next_source_context.json'
    context=json.loads(read_local(context_path,pins))
    feasibility_path='work/after152_next_source_feasibility.json'
    feasibility=json.loads(read_local(feasibility_path,pins))
    for name in (context_path,feasibility_path):
        recorded=plan['inputs'].get(name,plan['inputs'].get(str(ROOT/name)))
        check(recorded==pins[name],'Frozen batch4 PLAN does not bind metadata input')
    raw=[r for r in census['annual_documents'] if r['symbol'] in ('MU','PEP')]
    check(len(raw)==len(context['events'])==len(feasibility['documents'])==23,'Batch4 population')
    mapping={
        'issuer_original_release_full_bytes':('HTML_original',True),
        'issuer_distributed_original_prnewswire_release':('HTML_original',True),
        'public_index_observation_not_original_bytes':('index_observation_not_original_report',False),
        'public_indexed_issuer_release_excerpt':('index_observation_not_original_report',False),
    }
    for row,ctx,feas in zip(raw,context['events'],feasibility['documents']):
        check(row['event_id']==ctx['event_id']==feas['event_id'],'Metadata event order')
        kind,original=mapping[row['source_kind']]
        check(ctx['source_kind']==feas['source_kind']==kind,'Metadata source-kind mismatch')
        check(ctx['original_report_bytes_verified'] is original and feas['original_report_bytes_verified'] is original,'Original bytes claim mismatch')
        check(all(ctx.get(k)==v and feas.get(k)==v for k,v in row.items() if k!='source_kind'),'Other frozen census key changed')
    directory='outputs/monthly_financial_coverage_admission_batch4_v1/'
    manifest=json.loads(read_local(directory+'MANIFEST.json',pins))
    dataset=json.loads(read_local(directory+'DATASET.json',pins,manifest['outputs']['DATASET.json']))
    catalog=dataset['latest_event_catalog']
    check([{k:v for k,v in r.items() if k not in ('clock_review','processed_in_batch')} for r in catalog]==raw,'Published catalog provenance changed')
    for name in ('work/after152_batch4_admission.py','work/test_after152_batch4_admission.py'):
        read_local(name,pins)
    return dict(stage='independent_source_kind_metadata_review',passed=True,
        events_verified=23,source_kind_mapping={k:list(v) for k,v in mapping.items()},
        all_other_census_keys_exact=True,published_catalog_original_fields_preserved=True,
        unknown_categories_fail_closed=True,amounts_compared=0,ratios_computed=0,
        scope='Classification normalization for context comparison only; not financial admission or first-public-version certification.',
        input_hashes=pins)


def inspect(row,pins):
    raw=read_local(row['evidence_path'],pins,row['evidence_sha256'])
    p=Path(row['evidence_path']);gaps=[];receipt_info=None;info={};text=''
    if row['source_kind'] not in ORIGINAL_KINDS:
        kind='index_or_unresolved_not_original_report'
        gaps.append('catalog_does_not_identify_original_report_bytes')
    else:
        candidates=[str(p.with_suffix('.meta.json')),str(p)+'.meta.json',str(p.with_suffix('.json'))]
        candidates=list(dict.fromkeys(n for n in candidates if (ROOT/n).is_file()))
        matches=[]
        for name in candidates:
            receipt=json.loads(read_local(name,pins))
            if receipt.get('sha256')==row['evidence_sha256'] and receipt.get('url')==row['source_url'] and receipt.get('status',receipt.get('http_status'))==200:
                matches.append((name,receipt))
        check(len(matches)==1,'Unique matching original receipt required: '+row['event_id'])
        name,receipt=matches[0]
        receipt_info=dict(path=name,sha256=pins[name],status=200,url_and_sha_match=True,
                          retrieved_at=receipt.get('retrieved_at'),kind='original_response_receipt')
        if raw.startswith(b'%PDF'):
            kind='PDF_original_geometry_pending';gaps.append('PDF_title_date_and_visual_geometry_not_reviewed')
        else:
            kind='HTML_original';info,text=prior.html_info(raw)
    identity=sorted(set(m.group(0) for m in IDENTITY[row['symbol']].finditer(text)))
    if not identity:gaps.append('report_local_identity_not_verified')
    statements=info.get('statement_candidates',[])
    if not statements and not info.get('pre_candidates'):
        gaps.append('no_target_statement_table_or_PRE_metadata_located')
    terms=sorted(set(m.group(0).lower() for m in prior.old.ACCOUNTING.finditer(text)))
    snippets=[]
    for match in prior.old.ACCOUNTING.finditer(text):
        quote=prior.masked(text[max(0,match.start()-65):match.end()+175])
        if quote not in snippets:snippets.append(quote)
    return dict(deepcopy(row),source_kind=kind,census_source_kind=row['source_kind'],
                original_report_bytes_verified=kind in ('HTML_original','PDF_original_geometry_pending'),
                receipt=receipt_info,byte_count=len(raw),context=info,identity_anchors=identity,
                accounting_terms=terms,accounting_context_numbers_redacted=snippets,gaps=gaps,
                financial_values_extracted=0,fact_admitted=False,model_admitted=False,strategy_admitted=False)


def build():
    pins={}
    census=json.loads(read_local(CENSUS,pins,prior.old.EXPECTED_CENSUS))
    concept=json.loads(read_local(CONCEPT,pins,'60c3afd837cd32b75a70733b70c777084a9c124cf917e8052628947f1cb7d6ee'))
    check(concept['source_preparation_order']['remaining_text_lead_issuers'][-3:]==list(SYMBOLS),'Predeclared final-three order changed')
    kind_review=batch4_kind_review(census)
    pins.update(kind_review['input_hashes'])
    rows=[r for r in census['annual_documents'] if r['symbol'] in SYMBOLS]
    check(len(rows)==len({r['event_id'] for r in rows})==28 and Counter(r['symbol'] for r in rows)==EXPECTED_COUNTS,'Full final-three census required')
    sessions=prior.normalize_sessions(json.loads(read_local(SESSIONS,pins)))
    docs=[]
    for row in rows:
        doc=inspect(row,pins);session=prior.first_session(row['publication_date'],sessions)
        doc.update(pinned_next_session=session,available_session_verified=session is not None and session==row['available_session'])
        docs.append(doc)
    for name in ('work/after153_next_source_feasibility.py','work/after152_next_source_feasibility.py',
                 'work/after150_next_source_context.py','work/after141_amd_source_extract.py','work/after149_batch1_admission.py'):
        read_local(name,pins)
    for name,expected in list(pins.items()):read_local(name,pins,expected)
    result=dict(schema_version=1,stage='bounded_source_metadata_feasibility_not_admission',
        created_utc=datetime.now(timezone.utc).isoformat(),source_scope=list(SYMBOLS),documents=docs,
        input_hashes=pins,counts=dict(catalog_events=28,by_symbol=dict(Counter(d['symbol'] for d in docs)),
        source_kinds=dict(Counter(d['source_kind'] for d in docs)),clock_verified=sum(d['available_session_verified'] for d in docs),
        with_statement_candidates=sum(bool(d['context'].get('statement_candidates')) for d in docs)),
        limitations=[
          'Full immutable PLTR6/PYPL11/TMO11 census retained; this is not certification of complete issuer history.',
          'This inventory only locates labels, dates, units and report identity. No monetary cell values are parsed or emitted.',
          'Candidate tables can include quarterly, parent-attributable or non-GAAP material; a later declared extractor must prove annual consolidated NI and both asset endpoints.',
          'Copied annual-period dates and publication dates remain catalog metadata until compared with physical headers.',
          'PYPL separation and all accounting/reclassification notes require field-specific review; no automatic financial-sector admission.',
          'Same-report units and identity do not certify currency ISO, first-public version, correction history or a continuous security master.'
        ],financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,
        orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0)
    kind_review['created_utc']=result['created_utc']
    kind_review['input_hashes'][CENSUS]=pins[CENSUS]
    return result,kind_review


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result,review=build()
    if args.write:
        check(not (ROOT/OUT).exists() and not (ROOT/KIND_OUT).exists(),'Write-once metadata outputs already exist')
        for name,obj in ((KIND_OUT,review),(OUT,result)):
            with (ROOT/name).open('x') as stream:
                json.dump(obj,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(counts=result['counts'],kind_review_passed=review['passed'],
        documents=[dict(event_id=d['event_id'],source_kind=d['source_kind'],tables=len(d['context'].get('statement_candidates',[])),
                        pre=len(d['context'].get('pre_candidates',[])),gaps=d['gaps']) for d in result['documents']],
        output=OUT if args.write else None),sort_keys=True))


if __name__=='__main__':main()
