"""Create-once Intel original-document evidence sidecar. No network or factors."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
DESIGN_SHA='06c1423b71e616fb56150e755176c83ddce80aca145a1aa05aa075fbed805017'
OUTPUTS={'ADMISSIONS.json','EVIDENCE.json','COVERAGE.json','PROOF.json'}
IMPLEMENTATIONS={'parser.py','admission.py','build.py','test_parser.py','test_admission.py','register.py'}

def module():
    spec=importlib.util.spec_from_file_location('intel_admission',P/'admission.py')
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


def produce(design,documents,prior_evidence=None):
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
    collections=a.source_bundle(design,documents,hashes)
    releases={};calendar_releases={};evidence={};rows=[];admitted=[]
    for anchor in anchors+design['calendar_only']:
        source=a.release_source(anchor,design,documents,hashes,collections)
        raw=a.get_blob(anchor['release_path'],documents,hashes);parsed=None;reason=None;reconstruction=None;supplemental=None
        pages=None
        if anchor['format']=='pdf':pages,reconstruction=a.pdf_pages(anchor,documents,hashes)
        try:
            if 'supplemental' in anchor:
                identity=a.parser.release_identity(' '.join(pages))
                parsed=dict(identity=identity,statement=None,fiscal_evidence=dict(year_ends=[],extra_week_declarations=[]))
            else:parsed=a.parser.parse_pdf_pages(pages) if pages is not None else a.parser.parse_html(raw)
        except a.parser.ParseError as exc:reason=str(exc)
        if parsed is not None:
            identity=parsed['identity']
            a.require(identity['release_date']==anchor['release_date'] and (identity['fiscal_year'],identity['fiscal_quarter'])==(anchor['fiscal_year'],anchor['fiscal_quarter']),'parsed_release_mapping_mismatch')
            if parsed['statement'] is not None:a.require(parsed['statement']['periods'][0]['current']['end']==anchor['report_date'],'parsed_report_date_mismatch')
        if 'supplemental' in anchor:
            extra=anchor['supplemental'];extra_source=a.release_source(extra,design,documents,hashes,collections);extra_parsed=None;extra_reason=None
            try:extra_parsed=a.parser.parse_filing_html(a.get_blob(extra['release_path'],documents,hashes),anchor['filing_date'])
            except a.parser.ParseError as exc:extra_reason=str(exc)
            supplemental={**extra_source,'parsed':extra_parsed,'parse_reason':extra_reason,'financial_evidence_available_date':anchor['filing_date'],'availability_basis':'original selected filing date; not earlier press release'}
        release={**source,'parsed':parsed,'parse_reason':reason,'pdf_reconstruction':reconstruction,'supplemental_filing':supplemental,
                 'financial_evidence_available_date':anchor['filing_date'] if supplemental else anchor['release_date'],'role':anchor.get('role','record_evidence')}
        if 'accession' in anchor:
            release.update(filing_date=anchor['filing_date'],first_usable_date=anchor['first_usable_date'])
            releases[anchor['accession']]=release
        else:calendar_releases[anchor['release_path']]=release
    calendar=a.calendar_index({**releases,**calendar_releases})
    prior_calendar=revalidate_prior_calendar(calendar,prior_evidence)
    for anchor in anchors:
        acc=anchor['accession'];release=releases[acc];observed_pairs={}
        for index in anchor['record_indices']:
            record=records[index];a.require(a.digest(record)==anchor['record_sha256s'][str(index)],'registered_record_changed')
            pairs=record['pairing']['pairs'];a.require(len(pairs) in (1,2),'unexpected_pair_count');entries=[]
            for pair in pairs:
                duration=a.validate_pair(record,pair,facts,sessions,anchor,filings)
                eid=a.digest(dict(accession=acc,pair=pair));observed_pairs[eid]=a.digest(pair)
                e=dict(evidence_id=eid,accession=acc,pair_sha256=a.digest(pair),pair=pair,
                    semantic_concept=a.SEMANTICS[pair['tag']],**a.compare_pair(pair,duration,(release['supplemental_filing']['parsed'] if release['supplemental_filing'] else release['parsed']) if release['parsed'] is not None else None,prior_calendar if prior_evidence is not None and prior_evidence['evidence'].get(eid,{}).get('status')=='exact_match' else calendar,release['financial_evidence_available_date']),release_evidence=release)
                if eid in evidence:a.require(evidence[eid]==e,'conflicting_pair_evidence')
                else:evidence[eid]=e
                entries.append(e)
            sidecar,row=a.admit_record(index,record,entries,source_status);rows.append(row)
            if sidecar is not None:admitted.append(sidecar)
        a.require(observed_pairs==anchor['evidence_pair_hashes'],'registered_pair_scope_changed')
    a.require(len(evidence)==design['expected_pairs'] and sum(len(r['evidence_ids']) for r in rows)==design['expected_pair_occurrences'],'pair_count_changed')
    all_symbol=[i for i,r in enumerate(records) if r['symbol']=='INTC']
    a.require(sorted(set(all_symbol)-set(indices))==design['excluded_indices'],'excluded_row_scope_changed')
    rows.sort(key=lambda r:r['record_index_0based']);admitted.sort(key=lambda r:r['record_index_0based'])
    common=dict(schema_version=1,policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=False,source_status=source_status,source_hashes=hashes,source_records=design['source_records'],
        record_count=len(records),factor_ready_count=0,independent_review_pending=True)
    admissions={**common,'semantic_admitted_count':len(admitted),'records':admitted}
    ev={**common,'facts_source_status':source_status,'releases':releases,'calendar_only_releases':calendar_releases,
        'fiscal_calendar_evidence':calendar,'evidence':evidence}
    coverage=dict(schema_version=1,policy_id=a.POLICY,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=False,record_count=len(records),scope_rows=len(rows),scope_accessions=len(releases),
        calendar_only_documents=len(calendar_releases),unique_pairs=len(evidence),pair_occurrences=sum(len(r['evidence_ids']) for r in rows),
        semantic_admitted_count=len(admitted),unverified_count=len(rows)-len(admitted),factor_ready_count=0,
        admitted_indices=[r['record_index_0based'] for r in admitted],excluded_indices=design['excluded_indices'],rows=rows,
        pair_status_counts=dict(Counter(e['status'] for e in evidence.values())),source_flags_unchanged=True,
        concept_contexts_kept_separate=True,growth_factor_rank_order_return_computed=False,strategy_trials=0,network_requests=0,
        independent_review_pending=True)
    return {'ADMISSIONS.json':output_bytes(admissions),'EVIDENCE.json':output_bytes(ev),'COVERAGE.json':output_bytes(coverage)}


def revalidate_prior_calendar(calendar,prior_evidence):
    if prior_evidence is None:return None
    a=module();prior=prior_evidence['fiscal_calendar_evidence']
    a.require(set(prior)==set(calendar),'prior_calendar_kinds_changed')
    for kind,years in prior.items():
        for year,entries in years.items():
            a.require(all(entry in calendar[kind].get(year,[]) for entry in entries),'prior_calendar_source_not_reconstructed')
    return prior


def prior_registration(design,docs):
    a=module();read=lambda p:a.strict_json(docs[p]);hashes=design['source_hashes']
    err=read(design['erratum_path']);old=read(design['prior_design_path']);plan=read(design['prior_plan_path']);complete=read(design['prior_completion_path'])
    a.require(err['kind']=='bounded_identity_adjacency_and_declared_date_transform' and err['prior_design_sha256']==hashes[design['prior_design_path']] and err['prior_plan_sha256']==hashes[design['prior_plan_path']],'erratum_prior_registration')
    for key in ('scope','calendar_only','collections','excluded_indices','expected_record_count','expected_rows','expected_accessions','expected_pairs','expected_pair_occurrences','expected_calendar_only_documents','expected_supplemental_filings','allowed_tag_semantics','source_class','evidence_class'):
        a.require(design[key]==old[key],'erratum_scope_changed')
    for mapping in (err['prior_artifact_hashes'],old['source_hashes'],plan['implementation_hashes'],plan['independent_review_hashes']):
        a.require(all(hashes.get(p)==v for p,v in mapping.items()),'prior_source_closure_mismatch')
    prefix=str(Path(design['prior_design_path']).parent)+'/'
    a.require(all(hashes.get(prefix+n)==h for n,h in complete['artifact_hashes'].items()),'prior_completion_artifact_mismatch')
    reviewed=complete['independent_review'];review=read(design['prior_independent_review_path'])
    a.require(reviewed['path']==design['prior_independent_review_path'] and reviewed['sha256']==hashes[reviewed['path']] and reviewed['passed'] is True and review['passed'] is True and review['errors']==[],'prior_independent_review_mismatch')
    a.require(complete['semantic_admitted_count']==46 and complete['unverified_count']==11 and review['summary']['semantic_admitted_count']==46 and review['summary']['unverified_count']==11,'prior_partial_count_mismatch')
    diag=read(design['diagnosis_path'])
    a.require(err['diagnosis_path']==design['diagnosis_path'] and err['diagnosis_sha256']==hashes[design['diagnosis_path']],'diagnosis_mismatch')
    a.require(all(hashes.get(p)==h for p,h in diag['source_hashes'].items()),'diagnosis_source_mismatch')


def preserve_prior(payloads,prior_admissions,prior_coverage):
    a=module();now=a.strict_json(payloads['ADMISSIONS.json']);coverage=a.strict_json(payloads['COVERAGE.json'])
    prior={r['record_index_0based']:r for r in prior_admissions['records']};current={r['record_index_0based']:r for r in now['records']}
    allowed={r['record_index_0based'] for r in prior_coverage['rows'] if r['status']=='unverified'}
    a.require(all(current.get(i)==row for i,row in prior.items()),'prior_admitted_sidecar_changed')
    a.require(set(current)-set(prior)<=allowed,'addition_not_prior_unverified')
    a.require({r['record_index_0based'] for r in coverage['rows']}=={r['record_index_0based'] for r in prior_coverage['rows']} and coverage['excluded_indices']==prior_coverage['excluded_indices'],'prior_scope_changed')
    return len(prior)


def prepare(expected_design_sha=DESIGN_SHA):
    a=module();design_blob=(P/'DESIGN.json').read_bytes();plan_blob=(P/'PLAN.json').read_bytes()
    a.require(expected_design_sha==DESIGN_SHA==a.sha(design_blob),'design_changed')
    design=a.strict_json(design_blob);plan=a.strict_json(plan_blob)
    a.require(plan['design_sha256']==DESIGN_SHA and plan['actual_executed'] is False and plan['source_hashes']==design['source_hashes'],'plan_mismatch')
    a.require((design['expected_record_count'],design['expected_accessions'],design['expected_rows'],design['expected_calendar_only_documents'])==(3904,20,57,2),'scope_changed')
    tests=(P/'TESTS.txt').read_bytes();inventory=(P/'SOURCE_INVENTORY.json').read_bytes()
    a.require(plan['synthetic_tests_passed']>0 and a.sha(tests)==plan['tests_sha256'],'tests_changed')
    a.require(a.sha(inventory)==design['source_inventory_sha256'] and a.strict_json(inventory)['source_hashes']==design['source_hashes'],'inventory_changed')
    expected={str((P/n).relative_to(ROOT)) for n in IMPLEMENTATIONS}
    a.require(set(plan['implementation_hashes'])==expected,'implementation_set_changed')
    bindings={**design['source_hashes'],**plan['implementation_hashes'],**plan['independent_review_hashes']}
    docs=read_bound(ROOT,bindings);structural_check(design,docs);prior_registration(design,docs)
    payloads=produce(design,docs,a.strict_json(docs[design['prior_evidence_path']]))
    prior_count=preserve_prior(payloads,a.strict_json(docs[design['prior_admissions_path']]),a.strict_json(docs[design['prior_coverage_path']]))
    a.require(prior_count==46,'prior_success_count_changed')
    ev=a.strict_json(payloads['EVIDENCE.json'])
    proof=dict(schema_version=1,policy_id=a.POLICY,design_sha256=DESIGN_SHA,plan_sha256=a.sha(plan_blob),
        source_hashes=design['source_hashes'],implementation_hashes=plan['implementation_hashes'],
        independent_review_hashes=plan['independent_review_hashes'],output_hashes={n:a.sha(v) for n,v in payloads.items()},
        prior_admitted_rows_preserved=prior_count,additions_limited_to_prior_unverified=True,prior_boundary_references_revalidated=True,erratum_sha256=design['source_hashes'][design['erratum_path']],
        original_documents_loaded=23,selected_releases_loaded=20,calendar_only_documents_loaded=2,supplemental_filings_loaded=1,
        original_releases_parsed=sum(r['parsed'] is not None for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values())),
        original_financial_statements_parsed=sum(r['parsed'] is not None and r['parsed']['statement'] is not None for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values()))+sum(r['supplemental_filing'] is not None and r['supplemental_filing']['parsed'] is not None for r in ev['releases'].values()),
        supplemental_filings_parsed=sum(r['supplemental_filing'] is not None and r['supplemental_filing']['parsed'] is not None for r in ev['releases'].values()),
        pdf_documents_reextracted=sum(r['pdf_reconstruction'] is not None for r in list(ev['releases'].values())+list(ev['calendar_only_releases'].values())),
        source_rows_unchanged=True,source_class=a.SOURCE_CLASS,evidence_class=a.EVIDENCE_CLASS,
        raw_facts_provenance_verified=False,filing_catalog_http_status_verified=False,
        fiscal_starts_and_weeks_derived=True,period_starts_directly_printed=False,exact_publication_timestamp_verified=False,
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
