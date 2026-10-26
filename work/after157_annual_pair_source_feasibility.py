"""Bounded offline annual-pair source inventory; never computes filing lags.

First run --plan-only. Then bind its SHA with --write --plan-sha256 SHA.
Only administrative dates, source receipts and existing period evidence are read.
No market file, lag value, rank, portfolio, strategy return or registration occurs.
"""
import argparse
from collections import Counter, defaultdict
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from after156_event_scope_local_review import Sources, require
from after156_filing_universe_inventory import rows as sec_rows
from after156_filing_coverage_ceiling import BoundInputs, evaluate
import after156_canonical_chronology as canonical
import after156_exception_preflight as exceptions

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_annual_pair_source_feasibility.py'
TEST = 'work/test_after157_annual_pair_source_feasibility.py'
PLAN = 'work/after157_annual_pair_source_feasibility_plan.json'
OUT = 'work/after157_annual_pair_source_feasibility.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
ISSUERS = 'work/after156_issuer_gate.json'
ANNUAL = 'outputs/monthly_revenue_coverage156/ANNUAL_REPORTS.json'
OLD_FILINGS = 'outputs/monthly_profitability141_local_admission/ANNUAL_FILING_EVENTS.json'
CHAINS = 'outputs/monthly_revenue_pilot_source156/REPORTS.json'
EXTRACTIONS = tuple('outputs/monthly_financial_coverage_batch%d_v1/EXTRACTIONS.json'%i for i in range(1,6))
REVIEWS = tuple('work/after%d_batch%d_independent_review.json'%(148+i,i) for i in range(1,6))
ANNUAL_FORMS = ('10-K','10-K/A','10-KT','10-KT/A')
ORIGINAL_FORMS = ('10-K','10-KT')
ERAS = canonical.ERAS
INPUTS = tuple(dict.fromkeys((INVENTORY, ISSUERS, ANNUAL, OLD_FILINGS, CHAINS,
    *EXTRACTIONS, *REVIEWS, exceptions.META, exceptions.CAL, exceptions.CAL_CODE,
    exceptions.SOURCE_ROWS, exceptions.SOURCE_MANIFEST, exceptions.TZFILE,
    'work/after156_event_scope_local_review.py','work/after156_filing_universe_inventory.py',
    'work/after156_filing_coverage_ceiling.py','work/after156_canonical_chronology.py',
    'work/after156_exception_preflight.py',SELF,TEST,'AGENTS.md','LOCAL_SETUP.md',
    'run_local.sh','local_run.py','local.sandbox.sb')))
RULES = dict(version=1,
    purpose='Source feasibility only; no L, DeltaL, financial ratio, feature, rank, order, return or strategy registration.',
    population='Preserve original122x49=5978 slots, including3904 stock and2074 ETF slots; no successful-source subset.',
    cutoff='Information cutoff is the original full-XNYS close at signal index t-42. Original signal-day decision cutoff is retained only for alignment. Originals, amendments, evidence and raw acceptance sanity all use t-42.',
    provisional_scope='Preceding950 calendar days, latest annual reportDate no older than550 days and adjacent reportDate spacing330..400 days. These are source-inventory scope assumptions, not proof of normal annual periods or a strategy specification.',
    source='Only the96 snapshots already named by the original73-stock inventory, plus their explicitly advertised local pages and original receipts. Re-read all present pages, including locally present pages outside the old2015 window. No web requests, new issuer searches or source substitution.',
    choose_source='For each current-CIK candidate choose the lexicographically first source with root receipt, retrieval through cutoff and all advertised pages overlapping [cutoff-951days,cutoff+1day] present/receipted/count-matched. Choice precedes report or period inspection; a bad chosen source never falls back.',
    latest='Among original10-K/10-KT records with reported filingDate within the provisional950-day scope and no later than t-42, choose the highest fiscal reportDate then the immediately preceding distinct reportDate. Multiple accessions/metadata variants for either date block; transition/latest malformed records block, never select an older clean pair.',
    availability='Each chosen original must have its second XNYS session strictly after filingDate no later than t-42. This is a disclosed administrative-date reconstruction only, not exact acceptance/publication certification. A not-yet-available latest original blocks instead of falling back.',
    exceptions='Use the old strict raw-Z UTC/Eastern sanity alternatives against t-42 close without choosing one. Cross-snapshot CIK+accession form/date/document conflicts block. Known amendments for either selected period by cutoff block. Invalid unlocatable annual records block.',
    identity='Retain the existing explicitly conditional issuer/price-route policy. Known-successor quarantines stay; the whole950-day extent must lie after any stated history boundary. This is not new certified identity admission.',
    period_evidence='Inventory existing annual-release period ends and the current annual intervals of five reviewed financial batches and the reviewed MSFT/ABBV chains. Source/receipt hashes are checked when a hint matches a selected period. They are corroborating metadata only: same symbol/end or near-year spacing is not an accession-specific ordinary-annual or continuity proof.',
    positive_tier3='Requires separately bound accession-specific original cover/context evidence for both selected originals: ordinary annual rather than transition, identified fiscal year and start/end, consecutiveness and unchanged fiscal-year convention, valid by t-42. Existing unresolved141 original-context flags are preserved. This inventory cannot manufacture that missing adjudication.',
    output='Tier1 two observed original10-K report dates; tier2 source/chronology/cutoff/conditional-identity screened pair; tier3 established accession-specific normal consecutive pair. Separate hint coverage never promotes tier3. Full per-slot blockers and selected original metadata retained.',
    limits='Current SEC snapshots and existing release extracts are reconstructed historical evidence. No universal absence of later corrections is asserted; no source, identity, exact clock, price or financial admission is upgraded.')


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    with path.open('x') as handle:
        json.dump(obj,handle,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n')


def plan(root=ROOT):
    src=Sources(root)
    for path in INPUTS:src.read_bytes(path)
    for path,expected in {**canonical.PINS,**exceptions.PINS}.items():
        if path in src.hashes:require(src.hashes[path]==expected,'Frozen input changed: '+path)
    return dict(schema_version=1,status='FIXED_SOURCE_ONLY_INVENTORY_PLAN',
        created_utc=datetime.now(timezone.utc).isoformat(),rules=RULES,rules_sha256=digest(RULES),
        input_hashes=src.hashes,source_inventory=INVENTORY,output=OUT,
        source_collection='Local named snapshot/advertised-page reads only; no downloads',
        strategy_registration=False,actual_lag_or_feature_calculations=0)


def load_snapshots(src, inventory):
    """Reuse old receipt parser/canonical event IDs; retain transition forms too."""
    extended=deepcopy(inventory)
    bound=BoundInputs(src,extended)
    snapshots={}; all_occurrences=[]
    for item in extended['submission_snapshots']:
        root,received,retrieved=bound.proof(item['source'])
        path,cik=item['source']['path'],item['cik']
        require(str(root['cik']).zfill(10)==cik,'Source CIK changed')
        require(root['filings']['files']==[p['advertised'] for p in item['advertised_pages']],'Descriptors changed')
        events={}; pages=[]; consumed=[]; annual_occurrences=[]
        def consume(table, source, receipted, section, count=None):
            records=sec_rows(table)
            require(count is None or len(records)==count,'Advertised array count mismatch')
            consumed.append(dict(path=source,sha256=src.hashes[source],receipted=receipted,rows=len(records)))
            if not receipted:return
            for k,row in enumerate(records):
                occurrence=dict(snapshot=path,source=source,row=k,section=section,source_sha256=src.hashes[source])
                event=dict(cik=cik,**dict(zip(exceptions.FIELDS,row)),occurrence=occurrence)
                all_occurrences.append(event)
                if row[1] in ANNUAL_FORMS:
                    location=dict(source_path=source,section=section,row_index=k,
                        source_sha256=src.hashes[source],source_receipt_verified=True)
                    annual_occurrences.append((row,location))
                    eid=canonical.digest([cik,list(row)])
                    if eid not in events:events[eid]=dict(event,event_id=eid,occurrences=[])
                    events[eid]['occurrences'].append(location)
        consume(root['filings']['recent'],path,received,'filings.recent')
        for page in item['advertised_pages']:
            desc=page['advertised']; p=page['path']; proof=page.get('source')
            present=src.path(p).is_file(); receipted=False; count_match=False
            if present and proof is None:
                receipt=str(Path(p).with_suffix('.provenance.json'))
                if src.path(receipt).is_file():
                    rec=src.read(receipt)
                    extended['input_hashes'][receipt]=src.hashes[receipt]
                    proof=dict(path=p,sha256=rec.get('sha256'),receipt=receipt,
                        source_receipt_verified=True,expected_source_url='https://data.sec.gov/submissions/'+Path(p).name)
            if present and proof is not None:
                raw,receipted,_=bound.proof(proof)
                consume(raw,p,receipted,'historical_page',desc['filingCount']);count_match=True
            pages.append(dict(path=p,lo=desc['filingFrom'],hi=desc['filingTo'],present=present,
                receipt=receipted,count_match=count_match,status='CONSUMED' if count_match else 'UNAVAILABLE_OR_UNRECEIPTED'))
        # Cross-check ordinary annual canonical IDs against the frozen helper.
        old=canonical.build_events(cik,annual_occurrences)
        require({v['event_id'] for v in old.values()}=={e['event_id'] for e in events.values() if e['form'] in ('10-K','10-K/A')},'Canonical annual ID mismatch')
        snapshots[path]=dict(path=path,cik=cik,root_receipted=received,retrieved_date=retrieved,
            pages=pages,events=list(events.values()),consumed_sources=consumed,
            receipted_records=[tuple(e[k] for k in exceptions.FIELDS) for e in events.values() if canonical.valid_day(e['filingDate'])])
    # Include other-form variants of any annual accession, not only annual rows.
    keys={(e['cik'],e['accessionNumber']) for e in all_occurrences if e['form'] in ANNUAL_FORMS}
    return snapshots,[e for e in all_occurrences if (e['cik'],e['accessionNumber']) in keys]


def pair_metadata(events, cutoff, sessions):
    """Pick highest period first, then validate; no filing-minus-period arithmetic."""
    lower=(date.fromisoformat(cutoff)-timedelta(days=950)).isoformat()
    reasons=[]
    if any(not canonical.valid_day(e['filingDate']) for e in events):reasons.append('UNLOCATABLE_ANNUAL_FILING_DATE')
    known=[e for e in events if canonical.valid_day(e['filingDate']) and lower<=e['filingDate']<=cutoff]
    originals=[e for e in known if e['form'] in ORIGINAL_FORMS]
    if any(not canonical.valid_day(e['reportDate']) for e in originals):reasons.append('UNLOCATABLE_ORIGINAL_REPORT_DATE')
    periods=sorted({e['reportDate'] for e in originals if canonical.valid_day(e['reportDate'])},reverse=True)[:2]
    selected=[e for p in periods for e in originals if e['reportDate']==p]
    pair=[next(e for e in selected if e['reportDate']==p) for p in periods]
    tier1=len(periods)==2 and all(any(e['form']=='10-K' and e['reportDate']==p for e in selected) for p in periods)
    if len(periods)<2:reasons.append('FEWER_THAN_TWO_ORIGINAL_REPORT_DATES_IN_SCOPE')
    if any(len([e for e in selected if e['reportDate']==p])!=1 for p in periods):reasons.append('AMBIGUOUS_ORIGINAL_FOR_PERIOD')
    if any(e['form']!='10-K' for e in selected):reasons.append('LATEST_PAIR_CONTAINS_TRANSITION_REPORT')
    for e in selected:
        k=bisect_right(sessions,e['filingDate'])+1
        if k>=len(sessions) or sessions[k]>cutoff:reasons.append('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42')
    if periods and periods[0]<(date.fromisoformat(cutoff)-timedelta(days=550)).isoformat():reasons.append('LATEST_ANNUAL_OUTSIDE_PROVISIONAL_550_DAY_SCOPE')
    if len(periods)==2 and not 330<=(date.fromisoformat(periods[0])-date.fromisoformat(periods[1])).days<=400:
        reasons.append('REPORT_SPACING_OUTSIDE_PROVISIONAL_ANNUAL_SCOPE')
    amendments=[e for e in known if e['form'].endswith('/A') and e['reportDate'] in periods]
    if amendments:reasons.append('KNOWN_SELECTED_PERIOD_AMENDMENT_REQUIRES_ADJUDICATION')
    return dict(tier1=tier1,pair=pair,selected_variants=selected,amendments=amendments,reasons=sorted(set(reasons)))


def information_cutoff(sessions, signal_day):
    """Session positions, not calendar days; the caller binds the full schedule."""
    t=bisect_right(sessions,signal_day)-1
    require(t>=0 and sessions[t]==signal_day,'Signal date absent from original calendar')
    require(t>=42,'Missing t-42 calendar')
    return t-42,sessions[t-42]


def choose_snapshot(snapshots, candidates, cutoff):
    lower=date.fromisoformat(cutoff)-timedelta(days=950)
    guard_lo=(lower-timedelta(days=1)).isoformat()
    guard_hi=(date.fromisoformat(cutoff)+timedelta(days=1)).isoformat()
    tests=[evaluate(snapshots[p],guard_lo,guard_hi) for p in sorted(set(candidates))]
    eligible=[x['path'] for x in tests if x['closed_pages_any_form_diagnostic']]
    return (min(eligible) if eligible else None),tests


def evidence_hints(data):
    hints=defaultdict(list)
    for path,review_path in zip(EXTRACTIONS,REVIEWS):
        review=data[review_path]
        require(review['passed'] is True,'Financial source review not passed')
        for event in data[path]['events']:
            p=event.get('periods',{}).get('current',{})
            if not canonical.valid_day(p.get('period_end')):continue
            hint=dict(kind='REVIEWED_CURRENT_ANNUAL_INTERVAL_METADATA',catalog=path,review=review_path,
                event_id=event['event_id'],symbol=event['symbol'],period_start=p.get('period_start'),
                period_end=p['period_end'],fiscal_year=event.get('fiscal_year'),available=event.get('available_from'),
                basis=event['periods'].get('start_basis'),source=event['source'],accession_specific_normality=False)
            hints[(event['symbol'],p['period_end'])].append(hint)
    for event in data[ANNUAL]:
        p=event.get('official_actual_period_end')
        if canonical.valid_day(p):
            hints[(event['symbol'],p)].append(dict(kind='ANNUAL_END_CATALOG_ONLY',catalog=ANNUAL,
                event_id=event.get('event_id'),symbol=event['symbol'],period_end=p,
                available=event.get('available_session'),source=dict(path=event['path'],sha256=event['sha256']),
                source_kind=event.get('source_kind'),accession_specific_normality=False))
    for symbol,reports in data[CHAINS].items():
        for report in reports or []:
            p=report.get('period_end')
            if canonical.valid_day(p):
                hints[(symbol,p)].append(dict(kind='REVIEWED_ORIGINAL_RELEASE_CHAIN_INTERVAL',catalog=CHAINS,
                    event_id=report['report_id'],symbol=symbol,period_start=report.get('period_start'),period_end=p,
                    fiscal_year=report.get('fiscal_year'),available=report.get('known_by','')[:10],source=report['source'],
                    accession_specific_normality=False))
    return hints


def check_hint(src, hint):
    source=hint['source']; raw=source.get('raw',source)
    src.read_bytes(raw['path'],raw['sha256'])
    receipt=source.get('receipt_path',source.get('receipt'))
    verified=False
    if isinstance(receipt,dict):
        rec=src.read(receipt['path'],receipt['sha256']);verified=rec.get('sha256')==raw['sha256']
    elif isinstance(receipt,str):
        rec=src.read(receipt,source.get('receipt_sha256'));verified=rec.get('sha256')==raw['sha256']
    return dict(hint,source_bytes_sha_verified=True,original_receipt_sha_verified=verified,
        caveat='Original-release/period metadata only; no accession-specific ordinary-annual context adjudication.')


def summaries(rows):
    answer={}
    for name,lo,hi in ERAS:
        group=[r for r in rows if lo<=r['signal_index']<hi and not r['is_etf']]
        item=dict(stock_slots=len(group),all_candidate_slots=(hi-lo)*49,nonexclusive_blockers=dict(Counter(code for r in group for code in r['reasons'])))
        for tier in ('tier1','tier2','tier3'):
            selected=[r for r in group if r[tier]]; months=defaultdict(set)
            for r in selected:months[r['signal_index']].add(r['cik'])
            item[tier]=dict(slots=len(selected),fraction=len(selected)/len(group) if group else None,
                distinct_issuers=len({r['cik'] for r in selected}),months_at_least8_issuers=sum(len(months[i])>=8 for i in range(lo,hi)))
        item['tier2_with_both_period_hints']=sum(r['tier2'] and all(r['period_hint_ids']) for r in group)
        answer[name]=item
    return answer


def audit(root, plan_sha):
    src=Sources(root); proposal=src.read(PLAN,plan_sha)
    require(proposal['rules']==RULES and proposal['rules_sha256']==digest(RULES),'Source plan rules changed')
    require(set(proposal['input_hashes'])==set(INPUTS),'Source plan input closure changed')
    for path,expected in proposal['input_hashes'].items():src.read_bytes(path,expected)
    data={p:src.read(p,proposal['input_hashes'][p]) for p in (INVENTORY,ISSUERS,ANNUAL,OLD_FILINGS,CHAINS,*EXTRACTIONS,*REVIEWS)}
    for path,review_path in zip(EXTRACTIONS,REVIEWS):
        require(data[review_path]['extractions_sha256']==src.hashes[path],'Financial extraction review binding changed')
    meta=src.read(exceptions.META); calendar=src.read(exceptions.CAL)
    signal_cutoffs=exceptions.calendar_cutoffs(src,meta)
    spec=importlib.util.spec_from_file_location('annual_pair_calendar',src.path(exceptions.CAL_CODE))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    schedule=module.market_calendar(calendar['start'],calendar['end']).schedule
    sessions=[str(d.date()) for d in schedule.index];require(sessions==calendar['sessions'],'Full calendar changed')
    eastern=ZoneInfo.from_file(io.BytesIO(src.read_bytes(exceptions.TZFILE)),key='America/New_York')
    snapshots,occurrences=load_snapshots(src,data[INVENTORY]);cross=exceptions.cross_index(occurrences,eastern)
    stocks={s['symbol']:s for s in data[INVENTORY]['stocks']};hints=evidence_hints(data)
    old_context={(e['cik'],e['accession']):e for e in data[OLD_FILINGS]['events']}
    hint_catalog={};rows=[];selected_events={};seen=set()
    for i,day in enumerate(meta['dates']):
        cutoff_position,cutoff=information_cutoff(sessions,day)
        cutoff_close=schedule['close'].iloc[cutoff_position];cutoff_ns=int(cutoff_close.value)
        lower=(date.fromisoformat(cutoff)-timedelta(days=950)).isoformat()
        for j,symbol in enumerate(meta['symbols'][i]):
            require((i,j) not in seen,'Duplicate slot');seen.add((i,j))
            row=dict(signal_index=i,candidate_index=j,symbol=symbol,signal_date=day,
                original_decision_cutoff=signal_cutoffs[i]['decision_cutoff'],information_cutoff_date=cutoff,
                information_cutoff=cutoff_close.isoformat(),information_cutoff_ns=cutoff_ns,
                provisional_source_extent=[lower,cutoff],is_etf=symbol in meta['etfs'],
                selected_snapshot=None,cik=None,tier1=False,tier2=False,tier3=False,
                selected_event_ids=[],period_hint_ids=[],reasons=[],normal_period_admission=False)
            if row['is_etf']:
                row['reasons']=['ETF_PRESERVED_NO_ANNUAL_FILING_FEATURE'];rows.append(row);continue
            stock=stocks[symbol]; candidates=sorted({p for c in stock['exact_current_ciks'] for p in stock['local_snapshot_paths'][c]})
            chosen,tests=choose_snapshot(snapshots,candidates,cutoff)
            row['source_candidates']=tests
            if chosen is None:
                row['reasons']=['SOURCE_EXTENT_OR_CURRENT_CIK_UNAVAILABLE'];rows.append(row);continue
            snap=snapshots[chosen];row.update(selected_snapshot=chosen,cik=snap['cik'])
            result=pair_metadata(snap['events'],cutoff,sessions);row['tier1']=result['tier1']
            reasons=list(result['reasons']);column=data[ISSUERS]['columns'][symbol]
            if not column['candidate']:reasons.append('EXISTING_ISSUER_POLICY:'+column['category'])
            if column['minimum_price_window_start'] and lower<column['minimum_price_window_start']:
                reasons.append('ANNUAL_SOURCE_EXTENT_CROSSES_IDENTITY_BOUNDARY')
            if snap['cik'] not in column['expected_current_ciks']:reasons.append('CURRENT_CIK_CONTRADICTS_FROZEN_ROUTE')
            for event in result['selected_variants']+result['amendments']:
                selected_events[event['event_id']]=event
                reasons+=exceptions.screen(event,cross[(event['cik'],event['accessionNumber'])],cutoff,cutoff_ns)
            for event in result['pair']:
                row['selected_event_ids'].append(event['event_id']);local=[]
                for h in hints[(symbol,event['reportDate'])]:
                    if not canonical.valid_day(h.get('available')) or h['available']>cutoff:continue
                    key=digest(h)
                    if key not in hint_catalog:hint_catalog[key]=check_hint(src,h)
                    local.append(key)
                row['period_hint_ids'].append(local)
            row['known_amendment_event_ids']=[e['event_id'] for e in result['amendments']]
            row['tier2']=result['tier1'] and not reasons
            # Prior141 explicitly retains unresolved original context. Neither its
            # financial admission flag nor release end dates answer this new gate.
            row['existing_accession_context']=[dict(accession=e['accessionNumber'],
                prior141_blockers=old_context.get((e['cik'],e['accessionNumber']),{}).get('admission_blockers'),
                ordinary_consecutive_annual_period_verified=False) for e in result['pair']]
            reasons.append('ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED')
            row['reasons']=sorted(set(reasons));rows.append(row)
    require(len(rows)==5978 and sum(not r['is_etf'] for r in rows)==3904,'Full original grid changed')
    src.recheck()
    return dict(schema_version=1,status='SOURCE_FEASIBILITY_INVENTORY_NOT_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(),runtime_python=sys.version,
        source_plan_sha256=plan_sha,rules=RULES,rules_sha256=digest(RULES),input_hashes=dict(sorted(src.hashes.items())),
        summary=summaries(rows),rows=rows,selected_original_and_amendment_metadata=selected_events,
        period_hint_catalog=hint_catalog,source_summary=dict(snapshots=len(snapshots),
            receipted_annual_accession_keys=len(cross),source_files=len({s['path'] for snap in snapshots.values() for s in snap['consumed_sources']})),
        evidence_gate=dict(tier3_established=False,
            missing='No bound accession-specific ordinary annual, transition exclusion, fiscal-year continuity and unchanged year-end adjudication for the selected pair.',
            existing141_original_context_unverified=sum('original_10K_context_not_verified' in e['admission_blockers'] for e in data[OLD_FILINGS]['events']),
            note='Zero established tier3 is not a claim that no local document could support later focused adjudication. Release interval hints are retained for that review.'),
        activity=dict(actual_filing_lag_values=0,actual_lag_changes=0,price_values_read=0,
            feature_calculations=0,ranks=0,orders=0,returns=0,model_fits=0,new_strategy_trials=0,
            strategy_registration=False,network_requests=0,observer_accesses=0,frozen_files_modified=0),
        limitations=[RULES['limits'],RULES['provisional_scope'],
            'Tier2 reuses a conditional identity assumption over the extended source interval; it does not certify historical issuer/security identity.',
            'This bounded review inventories existing structured annual evidence; it does not newly parse every original annual-report cover or certify no undocumented fiscal-year change.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--write',action='store_true')
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    require(args.plan_only != args.write,'Choose plan-only or write')
    if args.plan_only:
        value=plan();save(ROOT/PLAN,value)
        print(json.dumps(dict(plan=PLAN,sha256=hashlib.sha256((ROOT/PLAN).read_bytes()).hexdigest(),rules_sha256=value['rules_sha256'])))
    else:
        require(bool(args.plan_sha256),'Explicit source-plan SHA required')
        require(not (ROOT/OUT).exists(),'Write-once source report exists')
        value=audit(ROOT,args.plan_sha256);save(ROOT/OUT,value)
        print(json.dumps(dict(report=OUT,sha256=hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest(),summary=value['summary'],activity=value['activity'])))


if __name__=='__main__':main()
