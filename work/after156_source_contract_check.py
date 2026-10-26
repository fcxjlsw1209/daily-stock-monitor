"""Validate the new source-work receipts without running a strategy."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from after156_event_scope_local_review import Sources, require

PINS = {
    'work/after156_canonical_chronology.json': '308bc7268ef36f1b8f4a4db3c76eeabddcb2c85cf65c439edccce6006379b796',
    'work/after156_event_price_identity.json': 'a1e4a42a72b79dc43c20836802e3f5df6265d1ab09c66699e59180fd7ce525fd',
    'work/after156_delayed_clock_validation.json': '67be3d232cbe7857bb6f1b510a32dd67b599ab4b8ef0dff391a7f2e34a49024e',
    'work/after156_filing_drift_design.json': '4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75',
    'outputs/monthly_revenue_persistence156/COMPLETION.json': '97177e6413d2227cb727e9efcc027b0cff68084a8c8e8a899f97d02b573665e2',
}
OUT = 'work/after156_source_contract_check.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    src = Sources()
    reports = {path: src.read(path, sha) for path, sha in PINS.items()}
    decision = src.read('work/after156_filing_clock_decision.json')
    for path, sha in decision['review_inputs'].items():
        src.read_bytes(path, sha)
    # Recheck the referenced input closures; do not execute producers or parse
    # arrays. Hashing a raw price file is not a numerical price examination.
    for path in list(PINS)[:3]:
        for name, sha in reports[path]['input_hashes'].items():
            src.read_bytes(name, sha)
    canonical = reports['work/after156_canonical_chronology.json']
    identity = reports['work/after156_event_price_identity.json']
    clock = reports['work/after156_delayed_clock_validation.json']
    require(len(canonical['rows']) == 3904, 'Original stock-slot denominator')
    counts = canonical['summary']['all122']
    require(counts['source_closed_slots'] == counts['structurally_consistent_slots'] == 2863,
            'Canonical source counts')
    require(counts['structurally_consistent_with_two_original_keys'] == 2732,
            'Two-key source count')
    require(sum(counts['unclosed_slot_classification'].values()) == 1041,
            'All unclosed slots classified')
    require(counts['admitted_identity_clock_slots'] == 0, 'Source closure is not admission')
    for row in canonical['rows']:
        chosen = row['selected_snapshot']
        if chosen is not None:
            require(chosen == min(row['eligible_current_snapshots']), 'Path selection changed')
            require(row['audit']['metadata_structurally_consistent'], 'Unexpected source conflict')
            snap = canonical['snapshots'][chosen]
            events = {e['event_id']: e for e in snap['events']}
            require(all(e in events for e in row['audit']['event_ids']), 'Missing saved event')
            for eid in row['audit']['event_ids']:
                event = events[eid]
                lo, hi = row['conditional_filing_date_envelope']
                require(lo <= event['filingDate'] <= hi, 'Event outside canonical envelope')
                require(event['form'] in ('10-Q', '10-K', '10-Q/A', '10-K/A'), 'Wrong event form')
    require(len(identity['bindings']) == 110, 'Identity binding count')
    require(len({b['selected_provider_symbol'] for b in identity['bindings']}) == 14,
            'Vendor series count')
    require(sum(b['join'] == 'DIRECT_DATED_TICKER_TO_VENDOR_REQUEST'
                for b in identity['bindings']) == 109, 'Direct route count')
    require(all(not b['filing_gate_satisfied'] and not b['historical_price_security_admitted']
                for b in identity['bindings']), 'Identity scope silently upgraded')
    require(clock['passed'] is True and clock['synthetic_validation']['named_checks_passed'] == 18,
            'Clock validation missing')
    require(not any(clock['admission'].values()), 'Clock validation is not source certification')
    require(decision['decision']['registration_permission'] is False, 'Decision registered a strategy')
    state = src.read('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    require(state['registered_queue'] == [] and len(state['completed']) == 156,
            'Concurrent research state changed')
    src.read_bytes('work/after156_source_contract_check.py')
    src.read_bytes('work/after156_event_scope_local_review.py')
    src.recheck()
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Named source receipt closure, canonical selection, identity scope and clock-policy assertions; no independent full chronology reconstruction or financial replay.',
        canonical_slots=3904, source_closed_slots=2863, source_two_original_keys=2732,
        dated_identity_bindings=110, vendor_series=14, exact_clock_or_identity_admitted_slots=0,
        completed_rounds=156, registered_next_rounds=0, input_files_verified=len(src.hashes),
        input_hashes=src.hashes, new_strategy_trials=0, numeric_price_values_read=0,
        factor_calculations=0, orders=0, returns=0, observer_actions=0)
    if args.write:
        with src.path(OUT).open('x') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        out_sha = hashlib.sha256(src.path(OUT).read_bytes()).hexdigest()
    else:
        out_sha = None
    print(json.dumps(dict(passed=True, output=OUT if args.write else None,
        sha256=out_sha, input_files_verified=len(src.hashes), new_trials=0)))


if __name__ == '__main__':
    main()
