"""Local SEC filing-event source feasibility, with no prices or strategy work."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import sys

from after156_event_scope_local_review import Sources, require

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after156_filing_event_feasibility.json'
BASE = 'outputs/dynamic_top100/registry_sec/'
# Fixed from source identity, before any new counts, factors or outcome inspection.
PILOT = (('AAPL','0000320193'),('MSFT','0000789019'),('ABBV','0001551152'))
BEGIN, END = '2015-01-01', '2026-10-06'
FIELDS = ('accessionNumber','form','filingDate','reportDate','acceptanceDateTime','primaryDocument')
META = 'outputs/monthly_local_linear70/META.json'
SLOTS = 'outputs/monthly_event_scope156/SLOTS.json'
PRE156 = 'outputs/monthly_target70_research/PRE156_1acac72586a8df44bfcb56787e3cbcf91f92b6fd74f3d5ecd9ce893d6d93ae7e.json'


def stamp(raw):
    require(isinstance(raw,str), 'Missing SEC acceptance timestamp')
    value = datetime.fromisoformat(raw.replace('Z','+00:00'))
    require(value.tzinfo is not None, 'Naive SEC acceptance timestamp')
    return value.astimezone(timezone.utc)


def records(raw):
    require(isinstance(raw,dict) and all(isinstance(raw.get(k),list) for k in FIELDS), 'SEC column schema')
    n = len(raw['accessionNumber'])
    require(all(len(v)==n for v in raw.values() if isinstance(v,list)), 'Ragged SEC columns')
    return [{key:raw[key][i] for key in FIELDS} for i in range(n)]


def relevant(entry, begin=BEGIN, end=END):
    return entry['filingFrom'] <= end and entry['filingTo'] >= begin


def self_test():
    calls = 0
    def ok(value):
        nonlocal calls
        require(value,'Synthetic assertion');calls+=1
    ok(stamp('2020-02-01T01:00:00Z')==stamp('2020-01-31T20:00:00-05:00'))
    try:stamp('2020-01-01T10:00:00')
    except ValueError:calls+=1
    else:raise ValueError('Naive clock accepted')
    good={k:['value'] for k in FIELDS};ok(len(records(good))==1)
    bad=dict(good,form=[])
    try:records(bad)
    except ValueError:calls+=1
    else:raise ValueError('Ragged arrays accepted')
    ok(relevant(dict(filingFrom='2014-01-01',filingTo='2015-01-01')))
    ok(not relevant(dict(filingFrom='1994-01-01',filingTo='2014-12-31')))
    ok(not relevant(dict(filingFrom='2026-10-07',filingTo='2027-01-01')))
    ok(relevant(dict(filingFrom=END,filingTo=END)))
    return calls


def load_sec(src, path, url):
    receipt_path = str(Path(path).with_suffix('.provenance.json'))
    receipt = src.read(receipt_path)
    require(receipt['source']==url, 'SEC source URL mismatch')
    retrieved = stamp(receipt['retrieved_at'])
    value = src.read(path,receipt['sha256'])
    return value,dict(path=path,provenance=receipt_path,sha256=receipt['sha256'],
                     source_url=url,retrieved_at=retrieved.isoformat(),
                     source_kind='later_retrieved_SEC_submissions_JSON')


def issuer(src,symbol,cik,slots):
    primary = BASE+cik+'_submissions.json'
    raw, proof = load_sec(src,primary,'https://data.sec.gov/submissions/CIK'+cik+'.json')
    require(str(raw['cik']).zfill(10)==cik and symbol in raw['tickers'],'Fixed CIK/ticker identity')
    current = records(raw['filings']['recent'])
    pages = [dict(proof,kind='recent',rows=len(current),
                  min_filing_date=min(r['filingDate'] for r in current),
                  max_filing_date=max(r['filingDate'] for r in current))]
    allrows = [(r,primary) for r in current]
    missing, warnings = [], []
    declared = raw['filings'].get('files',[])
    for entry in declared:
        name=entry['name']
        require(re.fullmatch('CIK'+cik+r'-submissions-\d+\.json',name) is not None,'Unrecognized SEC page identity')
        if not relevant(entry):
            continue
        path=BASE+'filing_indexes/'+name
        if not src.path(path).is_file() or not src.path(Path(path).with_suffix('.provenance.json')).is_file():
            missing.append(dict(path=path,declared=entry));continue
        chunk,cp=load_sec(src,path,'https://data.sec.gov/submissions/'+name)
        rows=records(chunk)
        require(len(rows)==entry['filingCount'],'Advertised count differs')
        outside=[r for r in rows if not entry['filingFrom']<=r['filingDate']<=entry['filingTo']]
        if outside:
            warnings.append(dict(kind='advertised_page_date_range_disagrees_with_row_filingDate',
                path=path,advertised_from=entry['filingFrom'],advertised_to=entry['filingTo'],
                outside_rows=[{k:r[k] for k in ('accessionNumber','form','filingDate','acceptanceDateTime')} for r in outside],
                handling='Rows retained. Descriptor dates are routing hints, not silently used to discard content.'))
        pages.append(dict(cp,kind='advertised_history_page',rows=len(rows),advertised=entry,
            advertised_count_verified=True,min_filing_date=min(r['filingDate'] for r in rows),
            max_filing_date=max(r['filingDate'] for r in rows)))
        allrows.extend((r,path) for r in rows)
    by_acc={};duplicates=[]
    for row,path in allrows:
        acc=row['accessionNumber']
        require(re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc) is not None,'Invalid accession')
        if acc in by_acc:
            require(by_acc[acc][0]==row,'Conflicting duplicate SEC accession')
            duplicates.append(acc)
        else:by_acc[acc]=(row,path)
    selected=[];amendments=[];other_forms=Counter();invalid=[]
    for row,path in by_acc.values():
        if not BEGIN<=row['filingDate']<=END:continue
        other_forms[row['form']]+=1
        if row['form'] not in ('10-Q','10-K','10-Q/A','10-K/A'):continue
        try:
            accepted=stamp(row['acceptanceDateTime'])
            require(date.fromisoformat(row['filingDate']).isoformat()==row['filingDate'],'Filing date')
            require(date.fromisoformat(row['reportDate']).isoformat()==row['reportDate'],'Report date')
            require(row['reportDate']<=row['filingDate'],'Report after filing')
            require(accepted<=stamp(proof['retrieved_at']),'Future relative to snapshot')
            require(bool(row['primaryDocument']),'Missing primary document')
        except (ValueError,TypeError) as e:
            invalid.append(dict(accession=row['accessionNumber'],reason=str(e)));continue
        target=amendments if row['form'].endswith('/A') else selected
        target.append(dict(row,index_source=path,
            archive_document_url='https://www.sec.gov/Archives/edgar/data/'+str(int(cik))+'/'+
                row['accessionNumber'].replace('-','')+'/'+row['primaryDocument'],
            acceptance_utc_parsed=accepted.isoformat(),
            acceptance_utc_day_minus_filing_date=(accepted.date()-date.fromisoformat(row['filingDate'])).days))
    selected.sort(key=lambda r:(r['acceptanceDateTime'],r['accessionNumber']))
    required=[s for s in slots if s['symbol']==symbol]
    year_counts={str(y):dict(Counter(r['form'] for r in selected if r['filingDate'][:4]==str(y))) for y in range(2015,2027)}
    # Counts refer only to metadata scope. They do not establish information availability or portfolio influence.
    own_counts=dict(total=len(required),early53=sum(s['signal_index']<53 for s in required),
        remaining_first88=sum(53<=s['signal_index']<88 for s in required),later34=sum(s['signal_index']>=88 for s in required))
    temporal_bounds=dict(earliest_required_anchor=min(s['required_anchor_window'][0] for s in required),
        latest_required_anchor=max(s['required_anchor_window'][1] for s in required))
    require(BEGIN<=temporal_bounds['earliest_required_anchor'] and temporal_bounds['latest_required_anchor']<=END,'Pilot extraction bounds')
    return dict(symbol=symbol,cik=cik,sec_name=raw['name'],current_tickers=raw['tickers'],
        pages=pages,advertised_history_pages=len(declared),relevant_advertised_history_pages=sum(relevant(x) for x in declared),
        missing_relevant_pages=missing,advertised_page_files_and_counts_verified=not missing,
        descriptor_date_warnings=warnings,conflicting_accessions=0,identical_duplicate_accessions=duplicates,
        original_filing_events=len(selected),original_forms=dict(Counter(x['form'] for x in selected)),
        original_filings_by_year=year_counts,amendments_excluded_from_original_event_definition=amendments,
        invalid_periodic_rows=invalid,all_forms_in_extraction_dates=dict(sorted(other_forms.items())),
        raw_acceptance_day_offset_counts=dict(Counter(x['acceptance_utc_day_minus_filing_date'] for x in selected)),
        original_monthly_candidate_slots=own_counts,consumed_anchor_bounds=temporal_bounds,
        filing_events=selected,metadata_enumeration_feasible=not missing and not invalid,
        exact_acceptance_instant_independently_verified=False,complete_SEC_history_certified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    checks=self_test();src=Sources();src.read_bytes('work/after156_event_scope_local_review.py')
    meta=src.read(META,'3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
    slots=src.read(SLOTS,'64515d1c5d81f4acd8c38fa6552c57ebdd7b4521769ddc13217b8aa0243a1022')
    require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']) and len(slots)==3904,'Original grid')
    prior=src.read(PRE156);names=[x['directory'] for x in prior['completed']]
    require(len(names)==155,'Pre156 metadata snapshot')
    results=[issuer(src,s,c,slots) for s,c in PILOT]
    refs=['work/after152_additional_mechanism_review.json','work/after149_guidance_audit.json',
        'work/after150_guidance_novelty.json','work/monthly_peer_earnings139_proposal.json',
        'outputs/monthly_earnings_factor_research/PLAN.json','outputs/monthly_incumbent_earnings70/PLAN.json',
        'outputs/monthly_earnings_repair/event_schema.py','outputs/monthly_sec_apple_original_admission_v1/DESIGN.json',
        'outputs/dynamic_top100/collect_registry_filing_history.py','outputs/dynamic_top100/collect_registry_reference_evidence.py',
        'work/after156_filing_event_feasibility.py','AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb']
    for p in refs:src.read_bytes(p)
    result=dict(schema_version=1,stage='READ_ONLY_SEC_FILING_EVENT_FEASIBILITY_AFTER156',
        status='BOUNDED_METADATA_ENUMERATION_FEASIBLE_CLOCK_AND_PROTOCOL_REVIEW_PENDING',
        scope=dict(fixed_issuers=[dict(symbol=s,cik=c) for s,c in PILOT],begin=BEGIN,end=END,
            reason='Same three issuer identities as the previously fixed source pilot; selected independently of156 results. No claim to represent the full73-equity universe.',
            year2026='Through2026-10-06 only, from later-retrieved local snapshots; not all of calendar2026.',
            forms=['10-Q','10-K'],amendments='10-Q/A and10-K/A enumerated separately, not silently merged or treated as originals.',
            foreign_forms='20-F,40-F,6-K and amendments excluded; a separate source and event taxonomy is required.6-K is not synonymous with a quarterly earnings release.'),
        event_definition=dict(candidate='Original SEC10-Q/10-K acceptance event using the preserved acceptanceDateTime field, once its semantics/clock mapping are independently admitted.',
            mechanism='Test whether continuation around periodic filing disclosure contributes distinct confirmation of the unchanged skip42 long-price trend, relative to same-mask non-event formation observations.',
            channel_limit='This is post-filing behavior, not first earnings-news response. Earlier releases and guidance may already convey the information. Price behavior cannot identify the economic cause.',
            proposed_window='For a future protocol only: original210-session formation endingt-42; event post-known offsets2..20, hence relevant anchors[t-271,t-44]. No event-to-price partition or factors computed here.',
            clock_pending='Retain raw timestamp and filingDate. Do not silently substitute either for earnings publication. Verify SEC timestamp timezone semantics and independently compare acceptance headers/indexes; fix at/after-close and non-session behavior before use.'),
        novelty=dict(bounded_direct_SEC_acceptance_price_event_precedent_found=False,
            inspected_completed_directory_count=len(names),
            prior_exact_formation_partition='after152 already proposed all earnings post-known day2..20 formation partition but marked it not ready and recorded zero trials. The aggregation idea is not new.',
            executed_neighbors=['Latest verified earnings-release event10/event21 vs same-age recent returns','Latest-event21 in incumbent episode slot','Same-sector peer event10 age10..63','Unconditional weekly market price delay'],
            SEC_financial_source_use='Apple original financial-source admission uses filingDate for availability, not a demonstrated filing-clock price event experiment.',
            conclusion='A distinct, precisely declared event source definition is supported by this bounded comparison; no exhaustive155-round nonduplication certificate.',
            archive_limit='Prior review records32 missing source packages within an earlier116-package scope; missing files are not evidence that a hypothesis was never tested.',source_refs=refs[:8]),
        issuers=results,metadata_enumeration_feasible=all(r['metadata_enumeration_feasible'] for r in results),
        summary=dict(original_filing_events=sum(r['original_filing_events'] for r in results),
            verified_source_pages=sum(len(r['pages']) for r in results),
            early53_existing_candidate_slots=sum(r['original_monthly_candidate_slots']['early53'] for r in results),
            early53_issuers=[r['symbol'] for r in results if r['original_monthly_candidate_slots']['early53']],
            missing_relevant_advertised_pages=sum(len(r['missing_relevant_pages']) for r in results),
            invalid_periodic_rows=sum(len(r['invalid_periodic_rows']) for r in results),
            source_semantics='Existing candidate-slot availability is metadata only; no factor availability or selection influence has been calculated.'),
        admission=dict(filing_event_clock_admitted=False,full_event_history_complete=False,
            factor_admitted=False,strategy_registered=False,earlier_earnings_event_holds_changed=False),
        source_claim_limits=['Hash-verified SEC-origin JSON establishes the contents of later-retrieved snapshots, not contemporaneous immutable publication evidence.',
            'Having every relevant advertised page and matching row counts closes this snapshot pagination list, not a proof that the SEC dataset has no omitted, corrected or withdrawn filings.',
            'Descriptor date ranges are compared against actual rows and discrepancies retained; routing metadata is not used to discard observed records.',
            'Current ticker-to-CIK match is checked for this fixed three-company scope; historical security identity changes across the full original universe were not audited.',
            'Filing acceptance timestamps are present and parseable but not yet independently checked against original acceptance headers; exact price-event use remains pending.',
            'All original122x49 candidate identities remain unchanged; no new eligibility, rankings, models, portfolios or returns are produced.'],
        next_source_steps=['Independently review source snapshot/page closure and all descriptor boundary disagreements.',
            'Authenticate acceptance-time semantics and a small fixed header sample across2015–2026 before fixing conservative event anchors.',
            'Freeze a filing-only event taxonomy, amendment treatment, duplicate rule, common coverage mask, overlap assignment and matched non-event control before any price partition.',
            'Preserve dominant skip42 momentum,3 fixed monthly holdings,one existing selector pass and preregistered first88 decision; later34 cannot choose the formula.'],
        synthetic_checks=dict(passed=True,count=checks),activity_counts=dict(price_values_read=0,financial_values_extracted=0,
            factor_calculations=0,model_fits=0,ranks=0,orders=0,returns=0,new_trials=0,network_requests=0,frozen_files_modified=0),
        created_utc=datetime.now(timezone.utc).isoformat(),runtime_python=sys.version)
    src.recheck();result['input_hashes']=dict(sorted(src.hashes.items()))
    if args.write:
        with src.path(OUT).open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],metadata_enumeration_feasible=result['metadata_enumeration_feasible'],
        issuers=[{k:r[k] for k in ('symbol','original_filing_events','original_forms','original_filings_by_year','original_monthly_candidate_slots',
            'missing_relevant_pages','invalid_periodic_rows','descriptor_date_warnings')} for r in results]),ensure_ascii=False))


if __name__=='__main__':main()
