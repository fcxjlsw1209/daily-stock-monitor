"""Canonical, metadata-only SEC original/amendment chronology inspection.

Run --rules-only first, then --write --rules-sha256 <printed digest>.
No price, factor, acceptance-clock interpretation, or identity admission occurs.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import re
import sys

from after156_event_scope_local_review import Sources, require
from after156_filing_universe_inventory import rows as sec_rows
from after156_filing_coverage_ceiling import BoundInputs, evaluate

SELF = 'work/after156_canonical_chronology.py'
OUT = 'work/after156_canonical_chronology.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
COVERAGE = 'work/after156_filing_coverage_ceiling.json'
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
PINS = {
    INVENTORY: 'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
    COVERAGE: '88c13206ad3901ee1cfc211d82fa2147f2068ef0f0018c6afd4bef8b6fd45113',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CAL: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'work/after156_filing_universe_inventory.py': '3251f80ec2c4c56cb9df04b22350fd6ba33b38afff8be7bf0bba6a1c943f9e86',
    'work/after156_filing_coverage_ceiling.py': '8823761cb3e4d8845b19c25e64f60295065c2f2bf784e8c2692bde0efda90e21',
}
FORMS = ('10-Q', '10-K', '10-Q/A', '10-K/A')
ERAS = [('early53', 0, 53), ('recent35', 53, 88), ('first88', 0, 88),
        ('later34', 88, 122), ('all122', 0, 122)]
RULES = {
    'version': 1,
    'population': 'Original 122 signals, 49 candidates per signal, 73 stock identities and 3904 non-ETF slots. Preserve every slot.',
    'mapping': 'Main canonical source candidates are exact current-metadata CIK links from the bound inventory only. This is a hypothesis, never historical identity admission. Other locating CIKs are retained separately, not joined or selected.',
    'envelope': 'Raw filingDate inclusive [calendar(t-272),calendar(t-41)], copied from the bound coverage screen. Required future verified anchors remain [t-271,t-42]. No raw filingDate or raw acceptance string becomes an anchor here.',
    'source_eligibility': 'Verified original root receipt, root retrieved no earlier than envelope end, and every advertised page intersecting the envelope locally present with exact hashes, original receipt and matching array/advertised counts. Source closure remains conditional on advertised descriptors.',
    'selection': 'Sort eligible relative snapshot paths lexicographically, select first. Never rank by original count, amendment count, audit result, performance, or price. Preserve all alternatives and blockers. A selected source with a structural issue remains selected and blocked; never retry a cleaner source.',
    'records': 'Retain exact 10-Q,10-K,10-Q/A,10-K/A only. Other forms never stand in for originals. Preserve six raw metadata fields and every source/row occurrence. Only originally receipted rows may support canonical events or amendment context.',
    'deduplication': 'Within one snapshot, identical CIK plus all six metadata fields is an exact duplicate; collapse to one event and retain every occurrence. Different snapshots remain separate provenance contexts. Exact duplicates are harmless, not extra reports.',
    'order': 'Sort metadata by raw filingDate, accessionNumber, form, reportDate, primaryDocument, raw acceptanceDateTime. This is deterministic display order only, not intraday chronological ordering.',
    'accession_conflict': 'Two different metadata variants for one accession are a blocking conflict when a variant is relevant or supports a relevant amendment. Include every variant reported on or before envelope end; unlocatable variants block conservatively. Do not import future-dated variants into old slots.',
    'report_key': 'Within selected CIK, base form (10-Q or 10-K) plus reportDate. Multiple distinct original accessions for a relevant report key on or before envelope end require adjudication. Multiple amendment accessions alone are not a duplicate-original conflict.',
    'amendment_support': 'A relevant amendment needs one unambiguous same-key original in receipted consumed rows whose raw filingDate is no later than the amendment filingDate. Same reported date is only a possible source match, not a proven temporal order. Retain out-of-envelope original support as context only.',
    'orphan': 'No observed eligible original is an orphan candidate in the consumed source extent, not proof of a truly orphan filing; unparsed older pages or missing pages outside the envelope may contain an original. A bounded orphan candidate blocks structural usability until reviewed.',
    'unlocatable': 'Invalid/missing filingDate in a receipted periodic row cannot be safely placed and blocks every selected slot for that snapshot. Missing/invalid accession, reportDate, primaryDocument or raw acceptance string blocks a relevant event; acceptance strings are not interpreted.',
    'structural_counts': 'Report source-closed slots, source-closed slots without reported structural issues, and that subset with at least two original report keys inside the envelope. None is an admitted feature mask or proves two full original anchor bundles.',
    'non_actions': 'No network, price values, acceptance instant conversion, anchors, E/B/N partition, factors, ranks, orders, returns, new trials, production registration, central-state edits or frozen-file changes.',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def valid_day(value):
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def event_issues(row):
    problems = []
    if not isinstance(row[0], str) or re.fullmatch(r'\d{10}-\d{2}-\d{6}', row[0]) is None:
        problems.append('INVALID_ACCESSION')
    if not valid_day(row[2]):
        problems.append('INVALID_FILING_DATE')
    if not valid_day(row[3]):
        problems.append('INVALID_REPORT_DATE')
    if valid_day(row[2]) and valid_day(row[3]) and row[3] > row[2]:
        problems.append('REPORT_DATE_AFTER_FILING_DATE')
    if not isinstance(row[4], str) or not row[4]:
        problems.append('MISSING_RAW_ACCEPTANCE_STRING')
    if not isinstance(row[5], str) or not row[5]:
        problems.append('MISSING_PRIMARY_DOCUMENT')
    return problems


def key_of(row):
    return (row[1].removesuffix('/A'), row[3])


def ordered(records):
    return sorted(records, key=lambda r: tuple(str(r[i]) for i in (2, 0, 1, 3, 5, 4)))


def build_events(cik, occurrences):
    grouped = defaultdict(list)
    for row, location in occurrences:
        if row[1] in FORMS and location['source_receipt_verified']:
            grouped[row].append(location)
    return {row: dict(event_id=digest([cik, list(row)]), cik=cik,
        accessionNumber=row[0], form=row[1], filingDate=row[2], reportDate=row[3],
        raw_acceptanceDateTime=row[4], primaryDocument=row[5],
        report_key=[cik, *key_of(row)], metadata_issues=event_issues(row),
        occurrences=locations, exact_duplicate_extra_occurrences=len(locations)-1)
        for row, locations in grouped.items()}


def audit(events, lo, hi):
    records = list(events)
    unlocatable = [r for r in records if not valid_day(r[2])]
    relevant = ordered(r for r in records if valid_day(r[2]) and lo <= r[2] <= hi)
    historical = [r for r in records if valid_day(r[2]) and r[2] <= hi]
    originals = [r for r in historical if r[1] in ('10-Q', '10-K')]
    relevant_keys = {key_of(r) for r in relevant}
    relevant_accessions = {r[0] for r in relevant}
    by_key, by_accession = defaultdict(list), defaultdict(list)
    for r in originals:
        by_key[key_of(r)].append(r)
    for r in historical + unlocatable:
        by_accession[r[0]].append(r)
    report_conflicts = []
    for key in sorted(relevant_keys, key=str):
        variants = by_key[key]
        if len({r[0] for r in variants}) > 1:
            report_conflicts.append(dict(report_key=list(key),
                original_event_ids=[events[r]['event_id'] for r in ordered(variants)]))
    amendments, context = [], set()
    for r in relevant:
        if not r[1].endswith('/A'):
            continue
        candidates = ordered(o for o in by_key[key_of(r)] if o[2] <= r[2])
        context.update(o for o in candidates if o not in relevant)
        relevant_accessions.update(o[0] for o in candidates)
        if not candidates:
            status = 'ORPHAN_CANDIDATE_IN_CONSUMED_SOURCE_EXTENT'
        elif len(candidates) != 1 or len({o[0] for o in by_key[key_of(r)]}) > 1:
            status = 'AMBIGUOUS_ORIGINAL_SUPPORT'
        elif events[candidates[0]]['metadata_issues']:
            status = 'INVALID_ORIGINAL_SUPPORT_METADATA'
        else:
            status = 'ONE_OBSERVED_ORIGINAL_METADATA_MATCH_CLOCK_PENDING'
        amendments.append(dict(amendment_event_id=events[r]['event_id'], status=status,
            original_candidate_event_ids=[events[o]['event_id'] for o in candidates],
            same_reported_filing_date=any(o[2] == r[2] for o in candidates),
            true_orphan_proven=False))
    accession_conflicts = [dict(accessionNumber=acc,
        variant_event_ids=[events[r]['event_id'] for r in ordered(by_accession[acc])])
        for acc in sorted(relevant_accessions, key=str) if len(by_accession[acc]) > 1]
    malformed = [dict(event_id=events[r]['event_id'], issues=events[r]['metadata_issues'])
                 for r in ordered(set(relevant) | context) if events[r]['metadata_issues']]
    duplicate_ids = [events[r]['event_id'] for r in relevant
                     if events[r]['exact_duplicate_extra_occurrences'] > 0]
    issues = []
    if unlocatable:
        issues.append('UNLOCATABLE_PERIODIC_METADATA')
    if malformed:
        issues.append('INVALID_RELEVANT_OR_SUPPORT_METADATA')
    if accession_conflicts:
        issues.append('CONFLICTING_ACCESSION_VARIANTS')
    if report_conflicts:
        issues.append('MULTIPLE_ORIGINAL_ACCESSIONS_FOR_REPORT_KEY')
    if any(a['status'] != 'ONE_OBSERVED_ORIGINAL_METADATA_MATCH_CLOCK_PENDING' for a in amendments):
        issues.append('AMENDMENT_ORIGINAL_SUPPORT_UNRESOLVED')
    return dict(event_ids=[events[r]['event_id'] for r in relevant],
        context_original_event_ids=[events[r]['event_id'] for r in ordered(context)],
        form_counts=dict(Counter(r[1] for r in relevant)),
        original_report_key_count=len({key_of(r) for r in relevant if r[1] in ('10-Q', '10-K')}),
        accession_conflicts=accession_conflicts, report_key_conflicts=report_conflicts,
        amendment_support=amendments, invalid_metadata=malformed,
        unlocatable_event_ids=[events[r]['event_id'] for r in ordered(unlocatable)],
        exact_duplicate_event_ids=duplicate_ids,
        exact_duplicate_extra_occurrences=sum(events[r]['exact_duplicate_extra_occurrences'] for r in relevant),
        structural_issues=issues, metadata_structurally_consistent=not issues)


def load_snapshot(inputs, item):
    raw, receipted, retrieved = inputs.proof(item['source'])
    require(str(raw['cik']).zfill(10) == item['cik'], 'CIK changed')
    require(raw['filings'].get('files', []) == [p['advertised'] for p in item['advertised_pages']],
            'Advertised descriptors changed')
    occurrences, pages, source_details = [], [], []

    def consume(table, path, receipt, section, expected_count=None):
        rr = sec_rows(table)  # Checks all arrays, not only the six retained fields.
        require(expected_count is None or len(rr) == expected_count, 'Advertised count mismatch: ' + path)
        source_details.append(dict(path=path, source_receipt_verified=receipt,
            rows=len(rr), periodic_rows=sum(r[1] in FORMS for r in rr),
            sha256=inputs.source.hashes[path]))
        for i, row in enumerate(rr):
            if row[1] in FORMS:
                occurrences.append((row, dict(source_path=path, section=section, row_index=i,
                    source_sha256=inputs.source.hashes[path], source_receipt_verified=receipt)))

    consume(raw['filings']['recent'], item['source']['path'], receipted, 'filings.recent')
    for p in item['advertised_pages']:
        desc, present = p['advertised'], 'source' in p
        if p['status'] == 'MISSING_LOCAL_PAGE':
            require(not inputs.source.path(p['path']).exists(), 'Inventory stale: missing page appeared ' + p['path'])
        if present:
            table, receipt, _ = inputs.proof(p['source'])
            consume(table, p['path'], receipt, 'historical_page', desc['filingCount'])
        else:
            receipt = False
        pages.append(dict(path=p['path'], lo=desc['filingFrom'], hi=desc['filingTo'],
            present=present, receipt=receipt, count_match=present, status=p['status']))
    events = build_events(item['cik'], occurrences)
    return dict(path=item['source']['path'], cik=item['cik'], root_receipted=receipted,
        retrieved_date=retrieved, pages=pages,
        receipted_records=[r for r in events if valid_day(r[2]) and '2015-01-01' <= r[2] <= '2026-10-06'],
        events=events, consumed_sources=source_details,
        unreceipted_periodic_occurrences=sum(not location['source_receipt_verified'] for _, location in occurrences))


def self_test():
    location = dict(source_path='synthetic', section='recent', row_index=0,
                    source_sha256='synthetic', source_receipt_verified=True)
    def record(acc, form, filed, period='2020-03-31', raw='uninterpreted', doc='doc.htm'):
        return (acc, form, filed, period, raw, doc)
    a = record('0000000001-20-000001', '10-Q', '2020-04-20')
    b = record('0000000001-20-000002', '10-Q/A', '2020-05-20')
    events = build_events('1', [(a, location), (a, dict(location, row_index=1)), (b, location)])
    check = audit(events, '2020-04-01', '2020-06-01')
    require(check['metadata_structurally_consistent'] and check['exact_duplicate_extra_occurrences'] == 1
            and len(check['event_ids']) == 2, 'Exact duplicates are not reports or conflicts')
    check = audit(events, '2020-05-01', '2020-06-01')
    require(check['metadata_structurally_consistent'] and len(check['context_original_event_ids']) == 1,
            'Outside-window original context')
    orphan = audit(build_events('1', [(b, location)]), '2020-05-01', '2020-06-01')
    require(not orphan['metadata_structurally_consistent'] and
            not orphan['amendment_support'][0]['true_orphan_proven'], 'Bounded orphan only')
    a2 = record('0000000001-20-000003', '10-Q', '2020-04-21')
    ambiguous = audit(build_events('1', [(a, location), (a2, location), (b, location)]), '2020-05-01', '2020-06-01')
    require(ambiguous['report_key_conflicts'] and not ambiguous['metadata_structurally_consistent'], 'Original key conflict')
    changed = record(a[0], '10-Q', '2020-04-20', doc='different.htm')
    conflict = audit(build_events('1', [(a, location), (changed, location)]), '2020-04-01', '2020-06-01')
    require(conflict['accession_conflicts'], 'Accession variant conflict')
    future = record('0000000001-20-000004', '10-Q', '2020-07-01')
    old = audit(build_events('1', [(a, location), (future, location)]), '2020-04-01', '2020-06-01')
    require(old['metadata_structurally_consistent'], 'Do not import future report conflicts')
    require(not build_events('1', [(record('x', '20-F', '2020-01-01'), location)]), 'Foreign forms excluded')
    require(not build_events('1', [(a, dict(location, source_receipt_verified=False))]), 'Unreceipted support excluded')
    malformed = record('0000000001-20-000005', '10-Q', 'unknown')
    require(not audit(build_events('1', [(malformed, location)]), '2020-04-01', '2020-06-01')
            ['metadata_structurally_consistent'], 'Unlocatable periodic metadata blocks')
    require(build_events('1', [(a, location)])[a]['event_id'] != build_events('2', [(a, location)])[a]['event_id'],
            'CIK event namespace isolated')
    same_day = record('0000000001-20-000006', '10-Q/A', '2020-04-20')
    same = audit(build_events('1', [(a, location), (same_day, location)]), '2020-04-01', '2020-06-01')
    require(same['amendment_support'][0]['same_reported_filing_date'], 'Same day is clock pending')
    wrong_original = record('0000000001-20-000007', '10-K', '2020-04-20')
    wrong = audit(build_events('1', [(wrong_original, location), (b, location)]), '2020-04-01', '2020-06-01')
    require(not wrong['metadata_structurally_consistent'], 'Base form is part of report key')
    return 12


def summarize(rows):
    result = {}
    for name, lo, hi in ERAS:
        group = [r for r in rows if lo <= r['signal_index'] < hi]
        selected = [r for r in group if r['selected_snapshot'] is not None]
        clean = [r for r in selected if r['audit']['metadata_structurally_consistent']]
        two = [r for r in clean if r['audit']['original_report_key_count'] >= 2]
        unclosed = Counter()
        for row in group:
            if row['selected_snapshot'] is not None:
                continue
            candidates = row['unavailable_current_snapshots']
            if not candidates:
                category = 'NO_CURRENT_METADATA_CIK_OR_LOCAL_SNAPSHOT'
            elif any(c['missing_pages'] and not c['source_blockers'] for c in candidates):
                category = 'MISSING_PAGES_ONLY_ON_AT_LEAST_ONE_SNAPSHOT'
            elif any(not c['missing_pages'] for c in candidates):
                category = 'COMPLETE_PAGE_SET_EXISTS_BUT_RECEIPT_OR_AGE_BLOCKED'
            else:
                category = 'PAGES_AND_RECEIPT_OR_AGE_BLOCKED_ON_EVERY_SNAPSHOT'
            unclosed[category] += 1
        result[name] = dict(stock_candidate_slots=len(group), source_closed_slots=len(selected),
            structurally_consistent_slots=len(clean), structurally_consistent_with_two_original_keys=len(two),
            structurally_consistent_two_original_fraction=len(two)/len(group),
            distinct_symbols_with_two_original_keys=len({r['symbol'] for r in two}),
            slots_with_amendments=sum(bool(r['audit']['amendment_support']) for r in selected),
            slots_with_exact_duplicate_occurrences=sum(r['audit']['exact_duplicate_extra_occurrences'] > 0 for r in selected),
            unclosed_slot_classification=dict(unclosed),
            source_closed_original_report_key_count_histogram=dict(Counter(
                str(r['audit']['original_report_key_count']) for r in selected)),
            structural_issue_slot_counts=dict(Counter(issue for r in selected for issue in r['audit']['structural_issues'])),
            admitted_identity_clock_slots=0)
    return result


def verify_serialized(report):
    """Reconstruct each saved audit from the saved raw metadata event pool."""
    pools = {}
    for path, snapshot in report['snapshots'].items():
        pool = {}
        for event in snapshot['events']:
            row = tuple(event[k] for k in ('accessionNumber', 'form', 'filingDate', 'reportDate',
                                          'raw_acceptanceDateTime', 'primaryDocument'))
            require(event['event_id'] == digest([snapshot['cik'], list(row)]), 'Saved event ID mismatch')
            require(event['cik'] == snapshot['cik'] and row[1] in FORMS, 'Saved event identity/form')
            require(all(o['source_receipt_verified'] and
                        report['input_hashes'][o['source_path']] == o['source_sha256']
                        for o in event['occurrences']), 'Saved occurrence provenance')
            require(row not in pool, 'Duplicate saved variant')
            pool[row] = event
        pools[path] = pool
    for row in report['rows']:
        path = row['selected_snapshot']
        if path is not None:
            lo, hi = row['conditional_filing_date_envelope']
            require(audit(pools[path], lo, hi) == row['audit'], 'Saved event pool cannot reconstruct slot audit')
    require(summarize(report['rows']) == report['summary'], 'Saved summary does not reconcile')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rules-only', action='store_true')
    parser.add_argument('--rules-sha256')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--replace-output-sha256')
    args = parser.parse_args()
    rules_sha = digest(RULES)
    if args.rules_only:
        require(not args.write, 'Rules-only cannot write the result')
        print(json.dumps(dict(rules_sha256=rules_sha, rules=RULES), ensure_ascii=False, indent=2))
        return
    require(args.rules_sha256 == rules_sha, 'Review and bind --rules-only digest before source execution')
    checks = self_test()
    src = Sources()
    for path in ('work/after156_filing_universe_inventory.py', 'work/after156_filing_coverage_ceiling.py'):
        src.read_bytes(path, PINS[path])
    inventory = src.read(INVENTORY, PINS[INVENTORY])
    coverage = src.read(COVERAGE, PINS[COVERAGE])
    meta, calendar = src.read(META, PINS[META]), src.read(CAL, PINS[CAL])['sessions']
    require(len(meta['dates']) == 122 and all(len(s) == 49 for s in meta['symbols']) and len(meta['etfs']) == 17,
            'Original grid changed')
    require(calendar == sorted(set(calendar)), 'Calendar order')
    stocks = {r['symbol']: r for r in inventory['stocks']}
    require(len(stocks) == 73, 'Stock universe changed')
    inputs = BoundInputs(src, inventory)
    snapshots = {s['source']['path']: load_snapshot(inputs, s) for s in inventory['submission_snapshots']}
    print('Original hashes, receipts and arrays checked for %d snapshots.' % len(snapshots), file=sys.stderr, flush=True)
    result_rows, selected_paths, used_ids, relevant_ids = [], set(), defaultdict(set), defaultdict(set)
    issue_instances = defaultdict(set)
    for i, (signal, names) in enumerate(zip(meta['dates'], meta['symbols'])):
        t = bisect_left(calendar, signal)
        require(t < len(calendar) and calendar[t] == signal and t >= 272, 'Signal calendar index')
        lo, hi = calendar[t-272], calendar[t-41]
        for j, symbol in enumerate(names):
            if symbol in meta['etfs']:
                continue
            stock = stocks[symbol]
            paths = sorted({p for cik in stock['exact_current_ciks'] for p in stock['local_snapshot_paths'][cik]})
            eligible, unavailable = [], []
            for path in paths:
                tested = evaluate(snapshots[path], lo, hi)
                if not tested['missing_pages'] and not tested['other_source_blockers']:
                    eligible.append(path)
                else:
                    unavailable.append(dict(path=path, missing_pages=tested['missing_pages'],
                                            source_blockers=tested['other_source_blockers']))
            chosen = eligible[0] if eligible else None
            row = dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                conditional_filing_date_envelope=[lo, hi], selected_snapshot=chosen,
                eligible_current_snapshots=eligible, unavailable_current_snapshots=unavailable,
                historical_identity_admitted=False, acceptance_clock_admitted=False,
                source_extent_complete_independently_proven=False)
            if chosen is not None:
                selected_paths.add(chosen)
                inspected = audit(snapshots[chosen]['events'], lo, hi)
                row['audit'] = inspected
                relevant_ids[chosen].update(inspected['event_ids'])
                used_ids[chosen].update(inspected['event_ids'] + inspected['context_original_event_ids']
                                       + inspected['unlocatable_event_ids'])
                for kind in ('accession_conflicts', 'report_key_conflicts'):
                    for issue in inspected[kind]:
                        used_ids[chosen].update(issue.get('variant_event_ids', issue.get('original_event_ids', [])))
                        issue_instances[kind].add(digest([chosen, issue]))
                for amendment in inspected['amendment_support']:
                    if amendment['status'] != 'ONE_OBSERVED_ORIGINAL_METADATA_MATCH_CLOCK_PENDING':
                        issue_instances['unresolved_amendments'].add((chosen, amendment['amendment_event_id']))
            else:
                row['audit'] = None
            result_rows.append(row)
    require(len(result_rows) == 3904, 'Original denominator changed')
    require([(r['signal_index'], r['candidate_index'], r['symbol']) for r in result_rows] ==
            [(r['signal_index'], r['candidate_index'], r['symbol']) for r in coverage['rows']], 'Coverage grid mismatch')
    for row, prior in zip(result_rows, coverage['rows']):
        require(bool(row['selected_snapshot']) == prior['by_identity_scope']['current_metadata_links']
                ['closed_pages_any_form_diagnostic'], 'Source-closure parity with bound coverage screen')
        require(row['selected_snapshot'] is None or row['selected_snapshot'] == min(row['eligible_current_snapshots']),
                'Canonical selector did not choose the lexicographic first source')
    serialized_snapshots = {}
    for path, snapshot in sorted(snapshots.items()):
        all_events = list(snapshot['events'].values())
        selected_events = [v for v in all_events if v['event_id'] in used_ids[path]]
        serialized_snapshots[path] = dict(cik=snapshot['cik'], selected_for_any_slot=path in selected_paths,
            root_receipt_verified=snapshot['root_receipted'], root_retrieved_utc_date=snapshot['retrieved_date'],
            consumed_sources=snapshot['consumed_sources'], advertised_pages=snapshot['pages'],
            observed_receipted_periodic_event_variants=len(all_events),
            unreceipted_periodic_occurrences_excluded=snapshot['unreceipted_periodic_occurrences'],
            selected_relevant_event_variants=len(relevant_ids[path]),
            events=sorted(selected_events, key=lambda e:(str(e['filingDate']), str(e['accessionNumber']), e['event_id'])))
    summary = summarize(result_rows)
    unique_relevant_events = {}
    for path, ids in relevant_ids.items():
        for event in snapshots[path]['events'].values():
            if event['event_id'] in ids:
                unique_relevant_events[event['event_id']] = event
    for path in (SELF, 'work/after156_event_scope_local_review.py', 'AGENTS.md', 'LOCAL_SETUP.md',
                 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        src.read_bytes(path)
    src.recheck()
    report = dict(schema_version=1, status='CANONICAL_CONDITIONAL_SOURCE_METADATA_NOT_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
        rules=RULES, rules_sha256=rules_sha, summary=summary,
        source_summary=dict(inventoried_snapshots=len(snapshots), canonical_snapshots_used=len(selected_paths),
            relevant_event_snapshot_pairs=sum(map(len, relevant_ids.values())),
            serialized_relevant_or_support_event_snapshot_pairs=sum(map(len, used_ids.values())),
            unique_relevant_cik_metadata_variants=len(set().union(*relevant_ids.values())),
            unique_relevant_form_counts=dict(Counter(e['form'] for e in unique_relevant_events.values())),
            unique_structural_issue_instances={k:len(v) for k,v in issue_instances.items()}),
        symbol_source_candidates={symbol:dict(exact_current_ciks=r['exact_current_ciks'],
            unadmitted_locating_ciks=r['non_admitted_locating_ciks'],
            all_inventoried_snapshot_paths=r['local_snapshot_paths'], historical_identity_admitted=False)
            for symbol,r in sorted(stocks.items())},
        snapshots=serialized_snapshots, rows=result_rows,
        remaining_blockers=[
            'Current CIK membership remains an unadmitted historical security/issuer hypothesis. Locating CIK histories are not concatenated.',
            'The filingDate envelope and advertised-page completeness remain conditional; independent original acceptance semantics and chronology completeness need review.',
            'All reported accession/report-key conflicts and bounded orphan candidates require adjudication; absence in consumed rows is not proof of worldwide absence.',
            'Same-day original/amendment ordering and raw acceptance strings need an independently verified clock before anchors can be assigned.',
            'No two complete 19-day bundles, E/B/N counts, finite price mask or beta validity is established here. At least two report keys is only a structural count.',
            'No source/identity/clock gate, feature admission, strategy selection or registration is granted by this artifact.'],
        validation=dict(synthetic_checks_passed=checks, all_consumed_input_hashes_rechecked=True,
            source_closure_parity_with_coverage=True, original_3904_slot_grid_reconciled=True,
            lexicographic_choice_verified_for_every_selected_slot=True,
            rules_digest_required_before_source_execution=True,
            saved_metadata_pool_reconstructs_every_slot_audit=True),
        activity_counts=dict(price_values_read=0, acceptance_instants_interpreted=0, anchors_assigned=0,
            factors=0, ranks=0, orders=0, returns=0, new_trials=0, network_requests=0, frozen_files_modified=0),
        input_hashes=dict(sorted(src.hashes.items())))
    verify_serialized(report)
    output = None
    if args.write:
        blob = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        target = src.path(OUT)
        if target.exists():
            previous = target.read_bytes()
            require(args.replace_output_sha256 is not None and
                    hashlib.sha256(previous).hexdigest() == args.replace_output_sha256,
                    'Refuse to replace an unbound existing task output')
            require(json.loads(previous)['rules_sha256'] == rules_sha, 'Existing task output rules differ')
        else:
            require(args.replace_output_sha256 is None, 'Expected prior output is missing')
        with target.open('wb' if target.exists() else 'xb') as handle:
            handle.write(blob)
        require(target.read_bytes() == blob, 'Output bytes round-trip')
        verify_serialized(json.loads(target.read_bytes()))
        output = dict(path=OUT, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    print(json.dumps(dict(summary=summary, source_summary=report['source_summary'],
        validation=report['validation'], rules_sha256=rules_sha, script_sha256=src.hashes[SELF], output=output), indent=2))


if __name__ == '__main__':
    main()
