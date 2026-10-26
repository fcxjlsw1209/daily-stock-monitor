"""Hypothetical two-session filing-date partitions; metadata only.

First run --rules-only. Then bind both displayed digests with --rules-sha256
and --script-sha256. --write saves one new receipt; existing files are immutable.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after156_delayed_partition.py'
OUT = 'work/after156_delayed_partition.json'
HELPER = 'work/after156_canonical_chronology.py'
HELPER_SHA = 'd2179d3c993b86366c3ba8f0bdf057a69503ec0d3b9d83a0de4becc9c5db68da'
PINS = {
    HELPER: HELPER_SHA,
    'work/after156_event_scope_local_review.py': 'fa31ecd3cb2d53f41355ce00682bfcfba890dab41f17e073aa563ae7c350f892',
    'work/after156_filing_universe_inventory.py': '3251f80ec2c4c56cb9df04b22350fd6ba33b38afff8be7bf0bba6a1c943f9e86',
    'work/after156_filing_coverage_ceiling.py': '8823761cb3e4d8845b19c25e64f60295065c2f2bf784e8c2692bde0efda90e21',
    'work/after156_filing_universe_inventory.json': 'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
    'work/after156_filing_drift_design.json': '4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75',
    'work/after156_delayed_clock_review.json': 'dadf7abdc8ceb2f5ed811be313cd62b76da2e3f5687970007bd6513fe226fee4',
    'work/after156_delayed_clock_validation.json': '67be3d232cbe7857bb6f1b510a32dd67b599ab4b8ef0dff391a7f2e34a49024e',
    'outputs/monthly_local_linear70/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'outputs/monthly_paired_target143/CALENDAR.json': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
}
ERAS = [('early53', 0, 53), ('recent35', 53, 88), ('first88', 0, 88),
        ('later34', 88, 122), ('all122', 0, 122)]
RULES = {
    'version': 1,
    'purpose': 'Hypothetical timing metadata preflight only; no timing assumption adopted, historical identity admitted, price read or strategy registered.',
    'population': 'Preserve all73 original equities,3904 original stock slots and the remaining original122x49 ETF slots. No eligibility or outcome filtering.',
    'source_scope': 'Exact current-metadata CIK links only, explicitly hypothetical historical identity. Never splice CIKs or include locating-only candidates.',
    'canonical_helper_sha256': HELPER_SHA,
    'source_selection': 'For the new inverse envelope, require original receipt, sufficient retrieval date and all intersecting advertised pages present/receipted/count-matching. Choose lexicographically first closed relative snapshot path. Audit and partition only after selecting; never change to an alternative because of structural, bundle or partition failure.',
    'clock': 'F-only: anchor a is the second actual XNYS session with session date strictly greater than raw filingDate F. Raw acceptanceDateTime is preserved but never interpreted. Anchor day0 is that session close; E offsets2..20 remain relative to a.',
    'envelope': 'For required anchors[t-271,t-42], use filing dates in exact half-open[S[t-273],S[t-43]). Pass lo and upper-date-minus-one-calendar-day to canonical inclusive source/audit helpers. Current date labels and advertised-page completeness remain hypotheses.',
    'audit': 'Use pinned canonical audit unchanged: malformed/ambiguous accessions, duplicate original report keys and unsupported amendments block partitioning. Context originals outside envelope are support only, not new anchors.',
    'formation': 'Exactly210 return-end session labels from t-251 through t-42; no actual returns calculated.',
    'ownership': 'Bundle all events on the same a. A bundle with any10-Q/A or10-K/A is an amendment bundle. For each formation day use latest a<=day; if day-a>20 label N. Otherwise amendment labels B0..20; original-only labels B0..1 and E2..20. No overlapping return counted twice.',
    'minimum': 'At least2 original-only anchor bundles must each own their entire19 E days a+2..a+20 inside formation, plus nE>=38 and nN>=80. Partial windows do not count as full bundles. Two reports in one anchor bundle count once.',
    'failure': 'Unclosed source or structural issue yields no partition. Partition minimum failures remain explicit. All hypothetical passes still have actual clock/identity/feature admission false.',
    'summary': 'Report source closure, structural consistency, two original keys, two full bundles and final partition counts by fixed eras; summarize nonexclusive loss reasons. Compare unchanged numeric breadth floors only as hypothetical metadata counts.',
    'non_actions': 'No price arrays, factor kernel, real Admission=True, beta estimates, scores, orders, returns, new trials, registration, central updates, network or observer actions.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def bytes_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helpers():
    # Pin transitive new helper code before executing any of its module bodies.
    for relative, expected in PINS.items():
        if relative.endswith('.py'):
            require(bytes_sha(ROOT / relative) == expected, 'Changed helper: ' + relative)
    import after156_canonical_chronology as canonical
    return canonical


def select_source(snapshots, paths, lo, hi, evaluate):
    closed, rejected = [], []
    for path in sorted(set(paths)):
        result = evaluate(snapshots[path], lo, hi)
        if result['missing_pages'] or result['other_source_blockers']:
            rejected.append(dict(path=path, missing_pages=result['missing_pages'],
                                 source_blockers=result['other_source_blockers']))
        else:
            closed.append(path)
    return (closed[0] if closed else None), closed, rejected


def partition(t, events):
    """Independent metadata algorithm. events already contain hypothetical anchors."""
    left, right = t - 251, t - 42
    require(type(t) is int and t >= 273, 'Valid full-calendar index required')
    grouped = defaultdict(list)
    require(len({event['event_id'] for event in events}) == len(events), 'Duplicate event IDs')
    for event in events:
        a = event['anchor']
        require(type(a) is int and t - 271 <= a <= right, 'Anchor outside required interval')
        require(event['form'] in ('10-Q', '10-K', '10-Q/A', '10-K/A'), 'Unexpected form')
        grouped[a].append(event)
    bundles = [dict(anchor=a, event_ids=sorted(e['event_id'] for e in grouped[a]),
                    forms=sorted(e['form'] for e in grouped[a]),
                    amendment=any(e['form'].endswith('/A') for e in grouped[a]))
               for a in sorted(grouped)]
    anchors = [bundle['anchor'] for bundle in bundles]
    labels, owners = [], []
    for day in range(left, right + 1):
        position = bisect_right(anchors, day) - 1
        owner = bundles[position] if position >= 0 else None
        if owner is None or day - owner['anchor'] > 20:
            labels.append('N')
            owners.append(None)
        else:
            labels.append('B' if owner['amendment'] or day - owner['anchor'] < 2 else 'E')
            owners.append(owner['anchor'])
    for bundle in bundles:
        owned = [left + k for k, (label, a) in enumerate(zip(labels, owners))
                 if label == 'E' and a == bundle['anchor']]
        bundle['owned_E_days'] = owned
        bundle['full19'] = not bundle['amendment'] and owned == list(range(bundle['anchor'] + 2, bundle['anchor'] + 21))
        bundle['left_clipped'] = bundle['anchor'] + 2 < left
        bundle['right_clipped'] = bundle['anchor'] + 20 > right
    counts = {label: labels.count(label) for label in 'EBN'}
    full = sum(bundle['full19'] for bundle in bundles)
    failures = (["FEWER_THAN_TWO_FULL19_ORIGINAL_BUNDLES"] if full < 2 else [])
    failures += ["N_E_BELOW_38"] if counts['E'] < 38 else []
    failures += ["N_N_BELOW_80"] if counts['N'] < 80 else []
    require(sum(counts.values()) == 210, 'Formation coverage changed')
    return dict(first_day=left, last_day=right, labels=''.join(labels), owner_anchor_indices=owners,
                bundles=bundles, counts=counts, full19_original_bundles=full,
                hypothetical_partition_pass=not failures, failure_reasons=failures)


def interval_oracle(t, events):
    """Synthetic cross-check: overwrite each event's interval in anchor order."""
    left, right = t - 251, t - 42
    grouped = defaultdict(list)
    for e in events:
        grouped[e['anchor']].append(e)
    cells = {d: ['N', None] for d in range(left, right + 1)}
    for a, group in sorted(grouped.items()):
        amended = any(e['form'].endswith('/A') for e in group)
        for offset in range(21):
            d = a + offset
            if d in cells:
                cells[d] = ['B' if amended or offset < 2 else 'E', a]
    full = sum(not any(e['form'].endswith('/A') for e in group)
               and all(cells.get(a + k) == ['E', a] for k in range(2, 21))
               for a, group in grouped.items())
    return ''.join(cells[d][0] for d in range(left, right + 1)), [cells[d][1] for d in range(left, right + 1)], full


def self_test():
    t, left, right = 600, 349, 558
    def event(a, form='10-Q', key=None):
        return dict(anchor=a, form=form, event_id=key or str(a) + form)
    fixtures = [[], [event(left), event(left + 50)],
                [event(left - 2), event(right - 20)],
                [event(left - 20), event(right - 20)],
                [event(left), event(right - 19)],
                [event(left), event(right)],
                [event(left), event(left, '10-K')],
                [event(left), event(left, '10-Q/A')],
                [event(left), event(left + 10, '10-Q/A'), event(left + 50)],
                [event(a) for a in range(left, right + 1, 21)]]
    for values in fixtures:
        p, o = partition(t, values), interval_oracle(t, values)
        require((p['labels'], p['owner_anchor_indices'], p['full19_original_bundles']) == o, 'Synthetic oracle mismatch')
    require(partition(t, fixtures[1])['counts'] == dict(E=38, B=4, N=168), 'Separated full bundles')
    require(partition(t, fixtures[2])['hypothetical_partition_pass'], 'Exact full-window boundaries')
    require(partition(t, fixtures[3])['full19_original_bundles'] == 1, 'Left partial is not full')
    require(partition(t, fixtures[4])['full19_original_bundles'] == 1, 'Right partial is not full')
    require(partition(t, fixtures[6])['full19_original_bundles'] == 1, 'Same anchor is one bundle')
    require(partition(t, fixtures[7])['counts']['E'] == 0, 'Same-anchor amendment overrides original')
    require(partition(t, fixtures[8])['full19_original_bundles'] == 1, 'Later amendment interrupts ownership')
    require(partition(t, fixtures[9])['counts']['N'] == 0 and not partition(t, fixtures[9])['hypothetical_partition_pass'], 'Dense events fail non-event minimum')
    cases = 0
    for a in range(left - 20, left + 23, 3):
        for b in range(right - 23, right + 1, 3):
            for amended in (False, True):
                values = [event(a), event(b, '10-K/A' if amended else '10-K')]
                p, o = partition(t, values), interval_oracle(t, values)
                require((p['labels'], p['owner_anchor_indices'], p['full19_original_bundles']) == o, 'Boundary oracle mismatch')
                cases += 1
    for bad in ([event(right + 1)], [event(left - 21)], [event(left), event(left)]):
        try:
            partition(t, bad)
        except ValueError:
            pass
        else:
            raise ValueError('Invalid anchor/event set accepted')
    def closure(snapshot, lo, hi):
        return dict(missing_pages=[] if snapshot['closed'] else ['page'], other_source_blockers=[])
    chosen, options, rejected = select_source({'a': dict(closed=True, full19=0), 'b': dict(closed=True, full19=10)}, ['b', 'a'], '', '', closure)
    require(chosen == 'a' and options == ['a', 'b'] and not rejected, 'No partition-based source choice')
    days = ['2020-07-01', '2020-07-02', '2020-07-07', '2020-07-08', '2020-07-09']
    require(days[bisect_right(days, '2020-07-04') + 1] == '2020-07-08', 'Weekend plus omitted sessions')
    return dict(fixed_partition_fixtures=len(fixtures), independent_oracle_boundary_cases=cases,
                invalid_event_sets_rejected=3, non_partition_source_choice_verified=True,
                weekend_and_closure_anchor_verified=True, passed=True)


def summarize(rows):
    answer = {}
    for era, lo, hi in ERAS:
        group = [r for r in rows if lo <= r['signal_index'] < hi]
        closed = [r for r in group if r['selected_snapshot'] is not None]
        clean = [r for r in closed if r['audit']['metadata_structurally_consistent']]
        keys = [r for r in clean if r['audit']['original_report_key_count'] >= 2]
        full = [r for r in clean if r['partition']['full19_original_bundles'] >= 2]
        passed = [r for r in clean if r['partition']['hypothetical_partition_pass']]
        monthly = Counter(r['signal_index'] for r in passed)
        answer[era] = dict(stock_candidate_slots=len(group), source_closed_slots=len(closed),
            structurally_consistent_slots=len(clean), two_original_key_slots=len(keys),
            two_full19_bundle_slots=len(full), hypothetical_partition_pass_slots=len(passed),
            hypothetical_pass_fraction=len(passed)/len(group),
            distinct_symbols_with_hypothetical_pass=len({r['symbol'] for r in passed}),
            months_with_at_least8_hypothetical_symbols=sum(monthly[i] >= 8 for i in range(lo, hi)),
            minimum_hypothetical_symbols_per_month=min(monthly[i] for i in range(lo, hi)),
            loss_source_to_structure=len(closed)-len(clean), loss_structure_to_two_keys=len(clean)-len(keys),
            loss_two_keys_to_two_full19=len(keys)-len(full), loss_two_full19_to_final_partition=len(full)-len(passed),
            partition_failure_counts=dict(Counter(reason for r in clean for reason in r['partition']['failure_reasons'])),
            full19_count_histogram=dict(Counter(str(r['partition']['full19_original_bundles']) for r in clean)),
            actual_clock_identity_admitted_slots=0)
        require(len(passed) <= len(full) <= len(keys) <= len(clean) <= len(closed), 'Nested stages violated')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules-only', action='store_true')
    parser.add_argument('--rules-sha256')
    parser.add_argument('--script-sha256')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    rules_sha, script_sha = digest(RULES), bytes_sha(ROOT / SELF)
    tests = self_test()
    if args.rules_only:
        require(not args.write, 'Rules-only cannot write actual metadata')
        print(json.dumps(dict(rules=RULES, rules_sha256=rules_sha, script_sha256=script_sha, synthetic_validation=tests), indent=2))
        return
    require(args.rules_sha256 == rules_sha and args.script_sha256 == script_sha,
            'Bind the displayed rules and implementation digests before source execution')
    canonical = load_helpers()
    src = canonical.Sources()
    for path, expected in PINS.items():
        src.read_bytes(path, expected)
    inventory = src.read(canonical.INVENTORY, PINS[canonical.INVENTORY])
    meta, sessions = src.read(canonical.META, PINS[canonical.META]), src.read(canonical.CAL, PINS[canonical.CAL])['sessions']
    require(len(meta['dates']) == 122 and all(len(v) == 49 for v in meta['symbols']) and len(meta['etfs']) == 17, 'Original grid')
    require(sessions == sorted(set(sessions)) and all(date.fromisoformat(d).isoformat() == d for d in sessions), 'Session axis')
    stocks = {row['symbol']: row for row in inventory['stocks']}
    require(len(stocks) == 73, 'Original equity scope')
    source_inputs = canonical.BoundInputs(src, inventory)
    snapshots = {item['source']['path']: canonical.load_snapshot(source_inputs, item) for item in inventory['submission_snapshots']}
    print('Verified %d original snapshots; selecting by source closure and path only.' % len(snapshots), file=sys.stderr, flush=True)
    rows, etfs, used_paths = [], [], set()
    for i, (signal, names) in enumerate(zip(meta['dates'], meta['symbols'])):
        t = bisect_left(sessions, signal)
        require(t < len(sessions) and sessions[t] == signal and t >= 273, 'Signal session index')
        lo, upper = sessions[t-273], sessions[t-43]
        hi = (date.fromisoformat(upper) - timedelta(days=1)).isoformat()
        for j, symbol in enumerate(names):
            if symbol in meta['etfs']:
                etfs.append(dict(signal_index=i, candidate_index=j, symbol=symbol, status='ETF_NEUTRAL_PRESERVED'))
                continue
            stock = stocks[symbol]
            paths = [p for cik in stock['exact_current_ciks'] for p in stock['local_snapshot_paths'][cik]]
            chosen, eligible, unavailable = select_source(snapshots, paths, lo, hi, canonical.evaluate)
            row = dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                signal_session_index=t, half_open_filing_envelope=[lo, upper], inclusive_helper_envelope=[lo, hi],
                selected_snapshot=chosen, eligible_current_snapshots=eligible, unavailable_current_snapshots=unavailable,
                historical_identity_admitted=False, actual_clock_admitted=False, hypothetical_timing=True,
                audit=None, partition=None)
            if chosen is not None:
                used_paths.add(chosen)
                audited = canonical.audit(snapshots[chosen]['events'], lo, hi)
                row['audit'] = audited
                if audited['metadata_structurally_consistent']:
                    catalog = {e['event_id']: e for e in snapshots[chosen]['events'].values()}
                    values = []
                    for identifier in audited['event_ids']:
                        event = catalog[identifier]
                        a = bisect_right(sessions, event['filingDate']) + 1
                        require(a < len(sessions) and t-271 <= a <= t-42, 'Inverse envelope/anchor disagreement')
                        values.append(dict(event_id=identifier, anchor=a, form=event['form']))
                    row['partition'] = partition(t, values)
                    for bundle in row['partition']['bundles']:
                        bundle['hypothetical_anchor_date'] = sessions[bundle['anchor']]
            rows.append(row)
    require(len(rows) == 3904 and len(rows) + len(etfs) == 122*49 and len({r['symbol'] for r in rows}) == 73, 'Full original denominator')
    for row in rows:
        require(row['selected_snapshot'] == (min(row['eligible_current_snapshots']) if row['eligible_current_snapshots'] else None), 'Canonical choice changed')
        if row['partition'] is not None:
            events = [dict(event_id=identifier, anchor=b['anchor'], form=form)
                      for b in row['partition']['bundles'] for identifier, form in zip(b['event_ids'], b['forms'])]
            # The oracle only needs the grouped amendment flag; event/form pairing
            # within a same-anchor bundle has no effect on ownership.
            p, o = row['partition'], interval_oracle(row['signal_session_index'], events)
            require((p['labels'], p['owner_anchor_indices'], p['full19_original_bundles']) == o, 'Actual metadata partition oracle mismatch')
    summary = summarize(rows)
    e, r, f = (summary[k] for k in ('early53', 'recent35', 'first88'))
    floors = dict(first88_fraction=f['hypothetical_pass_fraction'] >= .5,
        early53_fraction=e['hypothetical_pass_fraction'] >= .4, recent35_fraction=r['hypothetical_pass_fraction'] >= .4,
        first88_symbols=f['distinct_symbols_with_hypothetical_pass'] >= 12,
        early53_symbols=e['distinct_symbols_with_hypothetical_pass'] >= 8,
        recent35_symbols=r['distinct_symbols_with_hypothetical_pass'] >= 8,
        early53_months=e['months_with_at_least8_hypothetical_symbols'] >= 40,
        recent35_months=r['months_with_at_least8_hypothetical_symbols'] >= 26)
    for path in (SELF, 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        src.read_bytes(path, script_sha if path == SELF else None)
    src.recheck()
    catalog = {path: dict(cik=s['cik'], selected_for_any_slot=path in used_paths,
        consumed_sources=s['consumed_sources'], advertised_pages=s['pages'],
        events=sorted(s['events'].values(), key=lambda e: e['event_id'])) for path, s in sorted(snapshots.items())}
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
        status='HYPOTHETICAL_TWO_SESSION_METADATA_PARTITIONS_NOT_ADMISSION', rules=RULES,
        rules_sha256=rules_sha, script_sha256=script_sha, summary=summary,
        hypothetical_numeric_floor_comparisons=floors, hypothetical_numeric_floors_met=all(floors.values()),
        snapshots=catalog, rows=rows, preserved_etf_slots=etfs,
        validation=dict(synthetic=tests, all_input_hashes_rechecked=True,
            lexicographic_closed_source_choice_verified=True, original3904_stock_slots_preserved=True,
            independent_interval_oracle_reconciles_every_computed_partition=True,
            inverse_envelope_reconciles_every_hypothetical_anchor=True),
        limitations=['The date clock and current CIK historical identity are explicit hypotheses, not adopted assumptions or positive admission.',
            'Source closure depends on the declared advertised page descriptors and pinned canonical audit; no universal SEC completeness guarantee is asserted.',
            'No price validity, beta estimation or actual usable feature mask is established. Numeric breadth is conditional on timing/identity and all other outstanding gates.',
            'Same-anchor bundles may merge reports; failures never cause a switch to another canonical source.'],
        activity_counts=dict(price_values_read=0, actual_acceptance_instants_interpreted=0,
            admitted_clock_anchors=0, hypothetical_event_slot_anchor_assignments=sum(len(row['audit']['event_ids']) for row in rows if row['partition'] is not None),
            factors=0, ranks=0, orders=0, returns=0, new_trials=0, registrations=0, network_requests=0, frozen_files_modified=0),
        admission=dict(historical_identity=False, clock=False, source_gate=False, factors=False, strategy=False),
        input_hashes=dict(sorted(src.hashes.items())))
    output = None
    if args.write:
        blob = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        with src.path(OUT).open('xb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        require(src.path(OUT).read_bytes() == blob, 'Output roundtrip')
        output = dict(path=OUT, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
    print(json.dumps(dict(output=output, rules_sha256=rules_sha, script_sha256=script_sha,
        summary=summary, hypothetical_numeric_floor_comparisons=floors,
        input_files_verified=len(src.hashes), validation=report['validation']), indent=2))


if __name__ == '__main__':
    main()
