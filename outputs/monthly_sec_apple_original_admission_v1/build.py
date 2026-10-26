"""Create-once Apple original-document evidence sidecar. No network or factors."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
DESIGN_SHA='40cb4b1dd0937516d46b0142ccd3049a5ab87f4ad5fb2a44adc342eef84fc396'
OUTPUTS={'ADMISSIONS.json','EVIDENCE.json','COVERAGE.json','PROOF.json'}
IMPLEMENTATIONS={'parser.py','admission.py','build.py','test_parser.py','test_admission.py','register.py'}

def module():
    spec=importlib.util.spec_from_file_location('apple_admission',P/'admission.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def output_bytes(value):
    import json
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def read_bound(root,hashes):
    a=module();result={}
    for name,digest in hashes.items():
        path=Path(name)
        a.require(not path.is_absolute() and '..' not in path.parts and root.resolve() in (root/path).resolve().parents,
                  'unsafe_source_path')
        blob=(root/path).read_bytes();a.require(a.sha(blob)==digest,'source_hash_mismatch: '+name);result[name]=blob
    return result


def structural_check(design,documents):
    a=module();review=a.strict_json(documents[design['structural_review']])
    a.require(review.get('passed') is True and review.get('rows_replayed')==3904 and review.get('failures')==[] and
              review.get('selected_accession_and_pairs_independently_recomputed') is True,'structural_review_failed')
    expected={'DESIGN.json','PLAN.json','SOURCE_INVENTORY.json','CATALOGS.json','RECORDS.json','COVERAGE.json',
              'VALIDATION.json','PROTECTED.json','SESSIONS.json'}
    a.require(set(review['artifact_hashes'])==expected,'structural_artifacts_changed')
    for name,digest in review['artifact_hashes'].items():
        a.require(design['source_hashes'].get('outputs/monthly_sec_revenue_data_v3/'+name)==digest,'stale_structural_review')


def produce(design,documents):
    a=module();hashes=design['source_hashes'];read=lambda name:a.get_json(name,documents,hashes)
    a.require(design['policy_id']==a.POLICY and design['source_class']==a.SOURCE_CLASS and
              design['evidence_class']==a.EVIDENCE_CLASS and design['allowed_tag_semantics']==a.SEMANTICS,'source_policy_mismatch')
    a.require(hashes[design['source_records']['path']]==design['source_records']['sha256'],'source_records_binding_mismatch')
    records=read(design['source_records']['path']);facts=read(design['facts_path']);sessions=read(design['sessions_path'])
    a.require(len(records)==design['expected_record_count'] and sessions==sorted(set(sessions)) and sessions,'record_count_or_calendar_mismatch')
    for s in sessions:a.iso(s)
    identities=set()
    for record in records:
        key=(record['symbol'],record['signal_date']);a.iso(key[1]);a.require(key not in identities,'duplicate_record')
        identities.add(key);a.flags(record);a.flags(record['pairing'])
    anchors=design['scope'];indices=[i for anchor in anchors for i in anchor['record_indices']]
    a.require(len(anchors)==design['expected_accessions'] and len(indices)==design['expected_rows'] and
              len({x['accession'] for x in anchors})==len(anchors) and len(set(indices))==len(indices),'invalid_scope_counts')
    a.require(all(type(i) is int and 0<=i<len(records) for i in indices),'invalid_scope_index')
    a.require(len(design['calendar_only'])==design['expected_calendar_only_documents'],'calendar_source_count')
    filings=a.filing_source(design,documents,hashes);source_status=a.facts_source(design,documents,hashes)
    bundle=a.source_bundle(design,documents,hashes)
    identity=a.identity_sources(design,documents,hashes)
    releases={};calendar_releases={};evidence={};rows=[];admitted=[]
    for anchor in anchors+design['calendar_only']:
        source=a.release_source(anchor,design,documents,hashes,bundle);parsed=None;reason=None
        source['record_identity']=a.record_identity(identity,source['publication']['release_date'])
        reconstruction=None
        if anchor['attachment_role']=='financial_statement':
            payload,reconstruction=a.pdf_payload(anchor,documents,hashes)
            try:parsed=a.parser.financial(payload)
            except a.parser.ParseError as exc:reason=str(exc)
        else:reason='data_summary_not_used_as_financial_statement'
        if parsed is not None:
            a.require(parsed['statement']['current_end']==anchor['report_date'],'parsed_report_period_mismatch')
            a.require(max(parsed['statement']['duration_months'])==3*source['publication']['fiscal_quarter'],'parsed_duration_mismatch')
        release={**source,'parsed':parsed,'parse_reason':reason,'pdf_reconstruction':reconstruction,
                 'financial_evidence_available_date':source['publication']['release_date'],'role':anchor.get('role','record_evidence')}
        if 'accession' in anchor:
            release.update(filing_date=anchor['filing_date'],first_usable_date=anchor['first_usable_date'])
            releases[anchor['accession']]=release
        else:calendar_releases[anchor['release_path']]=release
    calendar=a.calendar_index({**releases,**calendar_releases})
    for anchor in anchors:
        acc=anchor['accession'];release=releases[acc];observed_pairs={}
        for index in anchor['record_indices']:
            record=records[index];a.require(a.digest(record)==anchor['record_sha256s'][str(index)],'registered_record_changed')
            pairs=record['pairing']['pairs'];a.require(len(pairs)==1,'unexpected_pair_count');entries=[]
            for pair in pairs:
                duration=a.validate_pair(record,pair,facts,sessions,anchor,filings)
                eid=a.digest(dict(accession=acc,pair=pair));observed_pairs[eid]=a.digest(pair)
                e=dict(evidence_id=eid,accession=acc,pair_sha256=a.digest(pair),pair=pair,
                    semantic_concept=a.SEMANTICS[pair['tag']],**a.compare_pair(pair,duration,release['parsed'],calendar,release['financial_evidence_available_date']),release_evidence=release)
                if eid in evidence:a.require(evidence[eid]==e,'conflicting_pair_evidence')
                else:evidence[eid]=e
                entries.append(e)
            sidecar,row=a.admit_record(index,record,entries,source_status);rows.append(row)
            if sidecar is not None:admitted.append(sidecar)
        a.require(observed_pairs==anchor['evidence_pair_hashes'],'registered_pair_scope_changed')
    a.require(len(evidence)==design['expected_pairs'] and sum(len(r['evidence_ids']) for r in rows)==design['expected_pair_occurrences'],'pair_count_changed')
    all_symbol=[i for i,r in enumerate(records) if r['symbol']=='AAPL']
    a.require(sorted(set(all_symbol)-set(indices))==design['excluded_indices'],'excluded_row_scope_changed')
    rows.sort(key=lambda r:r['record_index_0based']);admitted.sort(key=lambda r:r['record_index_0based'])
    common=dict(schema_version=1,policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=True,source_status=source_status,source_hashes=hashes,source_records=design['source_records'],
        record_count=len(records),factor_ready_count=0,independent_review_pending=True)
    admissions={**common,'semantic_admitted_count':len(admitted),'records':admitted}
    ev={**common,'facts_source_status':source_status,'releases':releases,'calendar_only_releases':calendar_releases,
        'fiscal_calendar_evidence':calendar,'historical_identity_sources':identity,'evidence':evidence}
    coverage=dict(schema_version=1,policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=True,record_count=len(records),scope_rows=len(rows),scope_accessions=len(releases),
        calendar_only_documents=len(calendar_releases),unique_pairs=len(evidence),pair_occurrences=sum(len(r['evidence_ids']) for r in rows),
        semantic_admitted_count=len(admitted),unverified_count=len(rows)-len(admitted),factor_ready_count=0,
        admitted_indices=[r['record_index_0based'] for r in admitted],excluded_indices=design['excluded_indices'],rows=rows,
        pair_status_counts=dict(Counter(e['status'] for e in evidence.values())),source_flags_unchanged=True,
        concept_contexts_kept_separate=True,growth_factor_rank_order_return_computed=False,strategy_trials=0,network_requests=0,
        independent_review_pending=True)
    return {'ADMISSIONS.json':output_bytes(admissions),'EVIDENCE.json':output_bytes(ev),'COVERAGE.json':output_bytes(coverage)}


def prepare(expected_design_sha=DESIGN_SHA):
    a=module();design_blob=(P/'DESIGN.json').read_bytes();plan_blob=(P/'PLAN.json').read_bytes()
    a.require(expected_design_sha==DESIGN_SHA==a.sha(design_blob),'design_changed')
    design=a.strict_json(design_blob);plan=a.strict_json(plan_blob)
    a.require(plan['design_sha256']==DESIGN_SHA and plan['actual_executed'] is False and plan['source_hashes']==design['source_hashes'],'plan_mismatch')
    a.require((design['expected_record_count'],design['expected_accessions'],design['expected_rows'],design['expected_calendar_only_documents'])==(3904,40,119,7),'scope_changed')
    tests=(P/'TESTS.txt').read_bytes();inventory=(P/'SOURCE_INVENTORY.json').read_bytes()
    a.require(plan['synthetic_tests_passed']>0 and a.sha(tests)==plan['tests_sha256'],'tests_changed')
    a.require(a.sha(inventory)==design['source_inventory_sha256'] and a.strict_json(inventory)['source_hashes']==design['source_hashes'],'inventory_changed')
    expected={str((P/n).relative_to(ROOT)) for n in IMPLEMENTATIONS}
    a.require(set(plan['implementation_hashes'])==expected,'implementation_set_changed')
    bindings={**design['source_hashes'],**plan['implementation_hashes'],**plan['independent_review_hashes']}
    docs=read_bound(ROOT,bindings);structural_check(design,docs);payloads=produce(design,docs)
    ev=a.strict_json(payloads['EVIDENCE.json'])
    proof=dict(schema_version=1,policy_id=a.POLICY,design_sha256=DESIGN_SHA,plan_sha256=a.sha(plan_blob),
        source_hashes=design['source_hashes'],implementation_hashes=plan['implementation_hashes'],
        independent_review_hashes=plan['independent_review_hashes'],output_hashes={n:a.sha(v) for n,v in payloads.items()},
        original_documents_loaded=47,selected_releases_loaded=40,calendar_only_documents_loaded=7,
        original_releases_parsed=sum(r['parsed'] is not None for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values())),
        original_financial_statements_parsed=sum(r['parsed'] is not None for r in ev['releases'].values()),
        original_pdf_first_pages_reconstructed=sum(r['pdf_reconstruction'] is not None for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values())),
        original_pdf_total_pages_reported=sum(r['pdf_reconstruction']['page_count'] for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values()) if r['pdf_reconstruction'] is not None),
        exact_release_pdf_outlinks_verified=sum(r['original_release_pdf_outlink_verified'] for r in ev['releases'].values()),
        source_rows_unchanged=True,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=True,filing_catalog_http_status_verified=False,
        fiscal_year_starts_derived=True,nominal_months_separate_from_exact_weeks=True,continuous_security_identity_verified=False,period_starts_directly_printed=False,exact_publication_timestamp_verified=False,
        factor_ready_count=0,strategy_trials=0,network_requests=0,independent_review_pending=True)
    payloads['PROOF.json']=output_bytes(proof);read_bound(ROOT,bindings)
    a.require((P/'DESIGN.json').read_bytes()==design_blob and (P/'PLAN.json').read_bytes()==plan_blob and (P/'TESTS.txt').read_bytes()==tests and (P/'SOURCE_INVENTORY.json').read_bytes()==inventory,'registration_changed')
    return payloads


def persist_once(directory,payloads,verify=False):
    a=module();a.require(set(payloads)==OUTPUTS,'output_set_mismatch')
    present=[name for name in payloads if (directory/name).exists()]
    if present:
        a.require(len(present)==len(payloads),'partial_output_set')
        a.require(all((directory/n).read_bytes()==b for n,b in payloads.items()),'existing_output_mismatch')
        return 'verified_existing'
    a.require(not verify,'no_outputs_to_verify')
    for name,blob in payloads.items():
        with (directory/name).open('xb') as f:f.write(blob)
    return 'created_once'


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=('build','verify'))
    cli.add_argument('--design-sha',required=True);args=cli.parse_args()
    payloads=prepare(args.design_sha);status=persist_once(P,payloads,args.action=='verify')
    print(status,module().strict_json(payloads['COVERAGE.json']))
