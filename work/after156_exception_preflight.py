"""Offline filing-date exception screen, never source or clock admission.

Print --rules-only before binding both digests for the actual metadata run.
Only --write creates the new receipt; existing output is never overwritten.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after156_exception_preflight.py'
OUT = 'work/after156_exception_preflight.json'
PARTITION = 'work/after156_delayed_partition.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
META = 'outputs/monthly_local_linear70/META.json'
CAL_CODE = 'outputs/monthly_earnings_repair/research_calendar.py'
SOURCE_ROWS = 'outputs/monthly_revenue_pilot_source156/ROWS.json'
SOURCE_MANIFEST = 'outputs/monthly_revenue_pilot_source156/MANIFEST.json'
TZFILE = 'work/venv/lib/python3.9/site-packages/pytz/zoneinfo/America/New_York'
PINS = {
    PARTITION: '11fb070a64723752ffedfb81b6fbbb00aacb0ce1ec5e3269e0d560c6e30e9a92',
    'work/after156_delayed_partition.py': '59e8e55083644bce888c0a8ac5134c9039a0d67b36209e2da529f7dd73d083a4',
    'work/after156_canonical_chronology.py': 'd2179d3c993b86366c3ba8f0bdf057a69503ec0d3b9d83a0de4becc9c5db68da',
    'work/after156_event_scope_local_review.py': 'fa31ecd3cb2d53f41355ce00682bfcfba890dab41f17e073aa563ae7c350f892',
    'work/after156_filing_universe_inventory.py': '3251f80ec2c4c56cb9df04b22350fd6ba33b38afff8be7bf0bba6a1c943f9e86',
    'work/after156_filing_coverage_ceiling.py': '8823761cb3e4d8845b19c25e64f60295065c2f2bf784e8c2692bde0efda90e21',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CAL: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    CAL_CODE: '112249b503050b54f62152aa5e15bcde066d4ee30735687c79f1a06a1c61474a',
    SOURCE_ROWS: '1732c3611fdbd14b00ef39d370a95493aec91f1a2c3ee60a2ce2ec3e27080196',
    SOURCE_MANIFEST: 'e4b5b71004186b1262128cc3e8b1286ea639da50ccd426daea572e6af61201bf',
    TZFILE: 'e9ed07d7bee0c76a9d442d091ef1f01668fee7c4f26014c0a868b19fe6c18a95',
}
FIELDS = ('accessionNumber', 'form', 'filingDate', 'reportDate', 'raw_acceptanceDateTime', 'primaryDocument')
CONFLICT_FIELDS = ('form', 'filingDate', 'reportDate', 'primaryDocument')
FORMS = ('10-Q', '10-K', '10-Q/A', '10-K/A')
ERAS = (('early53', 0, 53), ('recent35', 53, 88), ('first88', 0, 88), ('later34', 88, 122), ('all122', 0, 122))
RULES = {
    'version': 1,
    'scope': 'All3904 original stock slots and2074 ETF slots remain. This screen can only remove hypothetical passes, never admit source, identity, price validity, clock or strategy.',
    'cutoff': 'Rebuild original XNYS closes from frozen143 full calendar and original122 META dates; verify all5978 previously bound source ROWS decision_date, decision_cutoff and integer UTC nanoseconds. Include actual early closes and DST.',
    'source': 'Re-read all96 inventory snapshots and all present advertised pages with pinned canonical receipt/count checks; reconcile every periodic event and occurrence to the saved delayed partition. Only rows with original verified receipts enter cross-snapshot evidence.',
    'selected_source': 'Preserve exactly the saved source-closure-then-lexicographic choice. Never select another snapshot after an exception.',
    'consumption': 'Screen every selected audit event_id and context_original_event_id, including originals supporting amendments outside the formation envelope. Missing source/structural or partition failure remains unavailable.',
    'date': 'Require strict real ISO filingDate and reportDate, reportDate<=filingDate, valid accession and nonempty primaryDocument. filingDate cannot exceed the original decision date in New York. Never infer or repair filingDate from acceptance.',
    'acceptance_sanity': 'Strict YYYY-MM-DDTHH:MM:SS[.1to9digits]Z only. Preserve raw bytes. Independently test literal UTC and the same wall time under pinned America/New_York, including both valid DST folds. No interpretation is selected or used as an anchor. Any possible instant after decision_cutoff blocks; uninterpretable/invalid formats or nonexistent Eastern wall time block.',
    'cross_snapshot': 'Across every originally receipted row in all96 snapshots, group by exact CIK+accession. Different form,filingDate,reportDate,primaryDocument block each consuming slot regardless of which snapshot was selected. Include other-form variants for periodic accessions. No latest-version repair or historical backfill.',
    'raw_seconds': 'Different raw acceptance strings alone are recorded separately, with literal-UTC second spreads when parseable; every observed string for a consumed accession must pass both cutoff alternatives. This does not silently change the existing canonical structural decision.',
    'known_corrections': 'Every observed dated metadata variant and amendment is subject to the same original cutoff; a conflicting observed correction blocks rather than overwrites old signals. A later snapshot never establishes a corrected historical version.',
    'limits': 'No external correction-flag search or universal absence-of-error guarantee. Current SEC date metadata remain a disclosed reconstruction assumption, not historical vintage proof, precise acceptance certification or first-earnings-release evidence.',
    'non_actions': 'No price value reads, kernel Admission=True, beta estimates, factors, ranks, orders, returns, registration, trials, network, central-state updates, observer access or frozen writes.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def valid_day(value):
    try:
        return type(value) is str and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def epoch_ns(value, fractional_ns=0):
    delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + fractional_ns


def clock_alternatives(raw, eastern):
    match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z', raw) if type(raw) is str else None
    if match is None:
        return dict(raw=raw, issues=['UNINTERPRETABLE_RAW_ACCEPTANCE'], alternatives=[])
    try:
        naive = datetime.strptime(match[1], '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return dict(raw=raw, issues=['INVALID_RAW_ACCEPTANCE_DATE_TIME'], alternatives=[])
    fraction = int((match[2] or '').ljust(9, '0'))
    alternatives = [dict(interpretation='literal_UTC_sanity_only', utc_ns=epoch_ns(naive.replace(tzinfo=timezone.utc), fraction))]
    eastern_values = set()
    for fold in (0, 1):
        local = naive.replace(tzinfo=eastern, fold=fold)
        if local.astimezone(timezone.utc).astimezone(eastern).replace(tzinfo=None) == naive:
            eastern_values.add(epoch_ns(local, fraction))
    alternatives += [dict(interpretation='Eastern_wall_time_mislabeled_Z_sanity_only', utc_ns=n) for n in sorted(eastern_values)]
    return dict(raw=raw, issues=[] if eastern_values else ['NONEXISTENT_EASTERN_WALL_TIME'], alternatives=alternatives)


def field_issues(event):
    issues = []
    if type(event.get('accessionNumber')) is not str or re.fullmatch(r'\d{10}-\d{2}-\d{6}', event['accessionNumber']) is None:
        issues.append('INVALID_ACCESSION')
    if event.get('form') not in FORMS:
        issues.append('UNEXPECTED_CONSUMED_FORM')
    for key in ('filingDate', 'reportDate'):
        if not valid_day(event.get(key)):
            issues.append('INVALID_' + key.upper())
    if valid_day(event.get('filingDate')) and valid_day(event.get('reportDate')) and event['reportDate'] > event['filingDate']:
        issues.append('REPORT_DATE_AFTER_FILING_DATE')
    if type(event.get('primaryDocument')) is not str or not event['primaryDocument'].strip():
        issues.append('INVALID_PRIMARY_DOCUMENT')
    return issues


def cross_index(occurrences, eastern):
    grouped = defaultdict(list)
    for event in occurrences:
        grouped[(event['cik'], event['accessionNumber'])].append(event)
    result = {}
    for key, events in sorted(grouped.items(), key=str):
        variants = defaultdict(list)
        raw_values = {}
        for event in events:
            variant = {field: event[field] for field in CONFLICT_FIELDS}
            variants[digest(variant)].append(event)
            raw_values[digest(event['raw_acceptanceDateTime'])] = event['raw_acceptanceDateTime']
        clocks = [clock_alternatives(raw, eastern) for _, raw in sorted(raw_values.items())]
        utc = [c['alternatives'][0]['utc_ns'] for c in clocks if c['alternatives']]
        result[key] = dict(cik=key[0], accessionNumber=key[1], metadata_conflict=len(variants) > 1,
            variants=[dict(fields={field: group[0][field] for field in CONFLICT_FIELDS},
                occurrences=[e['occurrence'] for e in group]) for _, group in sorted(variants.items())],
            clock_sanity=clocks, raw_acceptance_variants=len(clocks),
            raw_literal_UTC_spread_seconds=(max(utc)-min(utc))/10**9 if utc else None,
            all_raw_formats_parseable=all(not c['issues'] for c in clocks))
    return result


def screen(event, cross, day, cutoff_ns):
    issues = field_issues(event)
    if valid_day(event.get('filingDate')) and event['filingDate'] > day:
        issues.append('FILING_DATE_AFTER_DECISION_CUTOFF')
    if cross['metadata_conflict']:
        issues.append('CROSS_SNAPSHOT_METADATA_CONFLICT')
    for variant in cross['variants']:
        filing = variant['fields']['filingDate']
        if not valid_day(filing):
            issues.append('INVALID_OBSERVED_VARIANT_FILING_DATE')
        elif filing > day:
            issues.append('OBSERVED_VARIANT_FILING_DATE_AFTER_CUTOFF')
    for check in cross['clock_sanity']:
        issues += check['issues']
        if any(a['utc_ns'] > cutoff_ns for a in check['alternatives']):
            issues.append('POSSIBLE_ACCEPTANCE_AFTER_DECISION_CUTOFF')
    return sorted(set(issues))


def self_test(eastern):
    checks = []
    def check(label, condition):
        require(condition, 'Synthetic check: '+label)
        checks.append(label)
    def event(raw='2020-05-01T16:01:02.003Z', **values):
        return dict(cik='0000000001', accessionNumber='0000000001-20-000001', form='10-Q',
                    filingDate='2020-05-01', reportDate='2020-03-31', primaryDocument='one.htm',
                    raw_acceptanceDateTime=raw, occurrence=dict(snapshot='one', source='one', row=0), **values)
    sample = event()
    key = (sample['cik'], sample['accessionNumber'])
    limit = epoch_ns(datetime(2020, 6, 1, 20, tzinfo=timezone.utc))
    index = cross_index([sample], eastern)
    check('both_clock_alternatives_before_cutoff', screen(sample, index[key], '2020-06-01', limit) == [])
    check('literal_fraction_preserved_to_nanosecond', clock_alternatives('2020-05-01T16:01:02.123456789Z', eastern)['alternatives'][0]['utc_ns'] % 10**9 == 123456789)
    late = dict(sample, raw_acceptanceDateTime='2020-06-02T00:00:00Z')
    check('obviously_postcutoff_blocked', 'POSSIBLE_ACCEPTANCE_AFTER_DECISION_CUTOFF' in screen(late, cross_index([late], eastern)[key], '2020-06-01', limit))
    near = dict(sample, raw_acceptanceDateTime='2020-06-01T17:00:00Z')
    check('UTC_before_but_Eastern_after_blocked', 'POSSIBLE_ACCEPTANCE_AFTER_DECISION_CUTOFF' in screen(near, cross_index([near], eastern)[key], '2020-06-01', limit))
    exact = epoch_ns(datetime(2020, 5, 1, 20, 1, 2, tzinfo=timezone.utc), 3000000)
    check('equality_at_cutoff_allowed', not screen(sample, index[key], '2020-05-01', exact))
    support = dict(sample, filingDate='2020-07-01', raw_acceptanceDateTime='2020-07-01T10:00:00Z')
    check('context_original_uses_same_cutoff', 'FILING_DATE_AFTER_DECISION_CUTOFF' in screen(support, cross_index([support], eastern)[key], '2020-06-01', limit))
    variant = dict(sample, filingDate='2020-05-02', occurrence=dict(snapshot='two', source='two', row=0))
    check('same_accession_different_filing_date_blocks', 'CROSS_SNAPSHOT_METADATA_CONFLICT' in screen(sample, cross_index([sample, variant], eastern)[key], '2020-06-01', limit))
    other_form = dict(sample, form='8-K')
    check('other_form_variant_not_dropped', cross_index([sample, other_form], eastern)[key]['metadata_conflict'])
    same = dict(sample, occurrence=dict(snapshot='two', source='two', row=0))
    check('exact_duplicate_not_conflict', not cross_index([sample, same], eastern)[key]['metadata_conflict'])
    seconds = dict(sample, raw_acceptanceDateTime='2020-05-01T16:01:03.003Z')
    seconds_index = cross_index([sample, seconds], eastern)[key]
    check('raw_second_difference_diagnostic_only', not seconds_index['metadata_conflict'] and seconds_index['raw_literal_UTC_spread_seconds'] == 1 and not screen(sample, seconds_index, '2020-06-01', limit))
    check('later_raw_variant_also_screened', 'POSSIBLE_ACCEPTANCE_AFTER_DECISION_CUTOFF' in screen(sample, cross_index([sample, late], eastern)[key], '2020-06-01', limit))
    for bad in ('2020-05-01', '2020-05-01T16:00:00', '2020-05-01T16:00:00+00:00', '2020-02-30T00:00:00Z', '2020-05-01T16:00:60Z', ' 2020-05-01T16:00:00Z', None):
        bad_event = dict(sample, raw_acceptanceDateTime=bad)
        check('invalid_raw_'+str(bad), bool(screen(bad_event, cross_index([bad_event], eastern)[key], '2020-06-01', limit)))
    check('DST_both_folds_tested', len(clock_alternatives('2020-11-01T01:30:00Z', eastern)['alternatives']) == 3)
    check('DST_nonexistent_wall_time_blocked', bool(clock_alternatives('2020-03-08T02:30:00Z', eastern)['issues']))
    return dict(passed=True, checks=len(checks), names=checks)


def calendar_cutoffs(src, meta):
    import pandas as pd
    calendar = src.read(CAL, PINS[CAL])
    for path, sha in calendar['provenance']['source_hashes'].items():
        src.read_bytes(path, sha)
    spec = importlib.util.spec_from_file_location('exception_preflight_calendar', src.path(CAL_CODE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schedule = module.market_calendar(calendar['start'], calendar['end']).schedule
    require([str(d.date()) for d in schedule.index] == calendar['sessions'], 'Frozen full calendar axis')
    close_map = {str(day.date()): close for day, close in zip(schedule.index, schedule['close'])}
    manifest = src.read(SOURCE_MANIFEST, PINS[SOURCE_MANIFEST])
    reference = manifest['outputs']['ROWS.json']
    rows = json.loads(src.read_bytes(SOURCE_ROWS, reference['sha256'], reference['bytes']))
    require(reference['sha256'] == PINS[SOURCE_ROWS] and len(rows) == 5978, 'Bound original source ROWS')
    seen = set()
    for row in rows:
        i, j = row['signal_index'], row['candidate_index']
        require(type(i) is int and type(j) is int and 0 <= i < 122 and 0 <= j < 49 and (i,j) not in seen, 'Unique original slot')
        seen.add((i,j))
        day, close = meta['dates'][i], close_map[meta['dates'][i]]
        timestamp = pd.Timestamp(row['decision_cutoff'])
        require(row['symbol'] == meta['symbols'][i][j] and row['decision_date'] == day and timestamp.tzinfo is not None
                and type(row['decision_cutoff_ns']) is int and row['decision_cutoff_ns'] == int(close.value) == int(timestamp.value)
                and str(close.tz_convert('America/New_York').date()) == day, 'Original decision cutoff or identity changed')
    return [dict(signal_index=i, decision_date=day, decision_cutoff=close_map[day].isoformat(),
                 decision_cutoff_ns=int(close_map[day].value)) for i,day in enumerate(meta['dates'])]


def summarize(rows):
    result = {}
    for name, lo, hi in ERAS:
        group = [r for r in rows if lo <= r['signal_index'] < hi]
        passes = [r for r in group if r['hypothetical_partition_and_exception_pass']]
        count = Counter(r['signal_index'] for r in passes)
        result[name] = dict(stock_slots=len(group), previous_hypothetical_partition_pass=sum(r['prior_partition_pass'] for r in group),
            exception_screen_pass=sum(r['exception_screen_pass'] for r in group), combined_hypothetical_pass=len(passes),
            newly_blocked_previous_pass=sum(r['prior_partition_pass'] and not r['exception_screen_pass'] for r in group),
            fraction=len(passes)/len(group), distinct_symbols=len({r['symbol'] for r in passes}),
            months_with_at_least8_symbols=sum(count[i] >= 8 for i in range(lo,hi)), minimum_symbols_per_month=min(count[i] for i in range(lo,hi)),
            nonexclusive_blocker_counts=dict(Counter(code for r in group for code in r['reasons'])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules-only', action='store_true')
    parser.add_argument('--rules-sha256')
    parser.add_argument('--script-sha256')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    script_sha = hashlib.sha256((ROOT/SELF).read_bytes()).hexdigest()
    rules_sha = digest(RULES)
    tz_blob = (ROOT/TZFILE).read_bytes()
    require(hashlib.sha256(tz_blob).hexdigest() == PINS[TZFILE], 'Pinned Eastern timezone changed')
    eastern = ZoneInfo.from_file(io.BytesIO(tz_blob), key='America/New_York')
    synthetic = self_test(eastern)
    if args.rules_only:
        require(not args.write, 'Rules-only cannot write real metadata')
        print(json.dumps(dict(rules=RULES, rules_sha256=rules_sha, script_sha256=script_sha, synthetic=synthetic), indent=2))
        return
    require(args.rules_sha256 == rules_sha and args.script_sha256 == script_sha, 'Bind previously displayed rules and script before metadata')
    for path, expected in PINS.items():
        if path.endswith('.py'):
            require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, 'Changed helper: '+path)
    import after156_canonical_chronology as canonical
    src = canonical.Sources()
    for path, expected in PINS.items():
        src.read_bytes(path, expected)
    prior = src.read(PARTITION, PINS[PARTITION])
    for path, expected in prior['input_hashes'].items():
        require(Path(path).suffix not in ('.npz', '.csv', '.gz', '.sqlite'), 'No price/database input permitted')
        src.read_bytes(path, expected)
    require(prior['admission'] == dict(historical_identity=False, clock=False, source_gate=False, factors=False, strategy=False), 'Preflight is hypothetical')
    meta = src.read(META, PINS[META])
    require(len(meta['dates']) == len(meta['symbols']) == 122 and all(len(s) == 49 for s in meta['symbols']), 'Original grid')
    cutoffs = calendar_cutoffs(src, meta)
    inventory = src.read(canonical.INVENTORY, prior['input_hashes'][canonical.INVENTORY])
    inputs = canonical.BoundInputs(src, inventory)
    snapshots = {item['source']['path']: canonical.load_snapshot(inputs, item) for item in inventory['submission_snapshots']}
    require(len(snapshots) == len(prior['snapshots']) == 96 and set(snapshots) == set(prior['snapshots']), 'All original snapshots')
    catalogs, target_keys = {}, set()
    for path, snap in snapshots.items():
        saved = prior['snapshots'][path]
        expected = dict(cik=snap['cik'], selected_for_any_slot=saved['selected_for_any_slot'], consumed_sources=snap['consumed_sources'],
                        advertised_pages=snap['pages'], events=sorted(snap['events'].values(), key=lambda e:e['event_id']))
        require(saved == expected, 'Saved snapshot/source occurrence mismatch: '+path)
        catalogs[path] = {e['event_id']:e for e in snap['events'].values()}
        target_keys.update((e['cik'], e['accessionNumber']) for e in snap['events'].values())
    # Include all forms for an accession seen as periodic anywhere. This catches
    # a cross-snapshot form correction which a periodic-only catalog could hide.
    occurrences, source_cache = [], {}
    for path, snap in sorted(snapshots.items()):
        for reference in snap['consumed_sources']:
            if not reference['source_receipt_verified']:
                continue
            source = reference['path']
            if source not in source_cache:
                raw = src.read(source, reference['sha256'])
                source_cache[source] = canonical.sec_rows(raw['filings']['recent'] if 'filings' in raw else raw)
            for k, values in enumerate(source_cache[source]):
                if (snap['cik'], values[0]) in target_keys:
                    event = dict(zip(FIELDS, values))
                    event.update(cik=snap['cik'], occurrence=dict(snapshot=path, source_path=source, source_sha256=reference['sha256'], row_index=k))
                    occurrences.append(event)
    cross = cross_index(occurrences, eastern)
    rows, seen = [], set()
    for row in prior['rows']:
        i,j = row['signal_index'],row['candidate_index']
        require((i,j) not in seen and row['symbol'] == meta['symbols'][i][j] and row['signal_date'] == meta['dates'][i] and row['symbol'] not in meta['etfs'], 'Original stock identity')
        seen.add((i,j))
        clock = cutoffs[i]
        audited = row['audit']
        reasons, failed = [], []
        relevant = audited['event_ids'] if audited else []
        context = audited['context_original_event_ids'] if audited else []
        chosen = row['selected_snapshot']
        require(chosen == (min(row['eligible_current_snapshots']) if row['eligible_current_snapshots'] else None), 'Canonical source selection changed')
        if chosen is None:
            reasons.append('SOURCE_CLOSURE_UNAVAILABLE')
        else:
            require(audited == canonical.audit(snapshots[chosen]['events'], *row['inclusive_helper_envelope']), 'Original audit changed')
            if not audited['metadata_structurally_consistent']:
                reasons.append('PRIOR_STRUCTURAL_FAILURE')
            for event_id in sorted(set(relevant+context)):
                require(event_id in catalogs[chosen], 'Missing selected event or original support')
                event = catalogs[chosen][event_id]
                key = (event['cik'],event['accessionNumber'])
                require(key in cross, 'Missing receipted accession evidence')
                issues = screen(event, cross[key], clock['decision_date'], clock['decision_cutoff_ns'])
                if issues:
                    reasons += issues
                    failed.append(dict(event_id=event_id, cik=key[0], accessionNumber=key[1], is_context_original=event_id in context, reasons=issues))
        old_pass = row['partition'] is not None and row['partition']['hypothetical_partition_pass'] is True
        reasons = sorted(set(reasons))
        rows.append(dict(signal_index=i, candidate_index=j, symbol=row['symbol'], **{k:v for k,v in clock.items() if k!='signal_index'},
            selected_snapshot=chosen, event_ids=relevant, context_original_event_ids=context,
            failed_events=failed, reasons=reasons, exception_screen_pass=not reasons,
            prior_partition_pass=old_pass, hypothetical_partition_and_exception_pass=old_pass and not reasons,
            historical_identity_admitted=False, actual_clock_admitted=False, source_gate_admitted=False))
    require(len(rows) == 3904 and len(prior['preserved_etf_slots']) == 2074 and len(seen) == 3904, 'Full denominator preserved')
    summary = summarize(rows)
    for path in (SELF, 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        src.read_bytes(path, script_sha if path == SELF else None)
    src.recheck()
    all_cross = [cross[key] for key in sorted(cross, key=str)]
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), status='METADATA_EXCEPTION_SCREEN_COMPLETE_NOT_ADMISSION',
        rules=RULES, rules_sha256=rules_sha, script_sha256=script_sha, runtime_python=sys.version,
        input_hashes=dict(sorted(src.hashes.items())), decision_cutoffs=cutoffs, summary=summary, rows=rows,
        preserved_etf_slots=prior['preserved_etf_slots'], accession_checks=all_cross,
        source_summary=dict(snapshots=len(snapshots), root_receipted_snapshots=sum(s['root_receipted'] for s in snapshots.values()),
            receipted_source_files=len(source_cache), periodic_accession_keys=len(cross), raw_occurrences_checked=len(occurrences),
            metadata_conflict_keys=sum(v['metadata_conflict'] for v in all_cross),
            differing_raw_acceptance_keys=sum(v['raw_acceptance_variants']>1 for v in all_cross)),
        validation=dict(synthetic=synthetic, all_input_hashes_rechecked=True, all_saved_snapshots_reconstructed=True,
            original5978_cutoffs_verified=True, all3904_stock_slots_preserved=True, canonical_selection_unchanged=True),
        limitations=[RULES['limits'], 'No external revision flag or contemporaneous availability guarantee was obtained.',
            'Passing requires only that all observed sanity alternatives precede each historical decision; none is certified or used for F-only anchoring.',
            'No source, historical security identity, numeric price availability or production feature admission is conferred.'],
        admission=dict(historical_identity=False, clock=False, source_gate=False, factors=False, strategy=False),
        activity_counts=dict(price_values_read=0, acceptance_instants_admitted=0, factors=0, beta_estimates=0, ranks=0, orders=0, returns=0,
            registrations=0, new_trials=0, network_requests=0, frozen_files_modified=0))
    output = None
    if args.write:
        blob = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        with src.path(OUT).open('xb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        require(src.path(OUT).read_bytes() == blob, 'Output bytes changed')
        output = dict(path=OUT, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
    print(json.dumps(dict(output=output, rules_sha256=rules_sha, script_sha256=script_sha, summary=summary,
        source_summary=report['source_summary'], validation=report['validation'], input_files_verified=len(src.hashes)), indent=2))


if __name__ == '__main__':
    main()
