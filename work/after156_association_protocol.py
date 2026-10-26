"""Write a separate research protocol; never register or compute a feature."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from after156_event_scope_local_review import Sources, require

OLD = 'work/after156_filing_drift_design.json'
OLD_SHA = '4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75'
OUT = 'work/after156_association_protocol.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    src = Sources()
    old = src.read(OLD, OLD_SHA)
    decision = src.read('work/after156_filing_clock_decision.json')
    require(not decision['decision']['registration_permission'], 'Unexpected prior registration')
    require('decision_cutoff' in decision['assumptions_selected_for_future_protocol_review'][0],
            'Exact historical decision cutoff required')
    p = copy.deepcopy(old)
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),
        status='SEPARATE_RECONSTRUCTED_ASSOCIATION_PROTOCOL_PENDING_FINAL_SOURCE_REVIEW',
        title='Old momentum conditional on reconstructed administrative filing dates',
        prospective_round=None,
        predecessor_exact_clock_protocol=dict(path=OLD, sha256=OLD_SHA, retained_unchanged=True),
        explicit_reconstruction_assumptions=decision['assumptions_selected_for_future_protocol_review'],
        distinction='A new F-only administrative-date association study, not certification or retroactive waiver of the original exact-acceptance design.')
    p['hypothesis'].update(
        economic_proposition='Among old positive trends, strength concentrated near periodic administrative filing dates may rank persistence differently from ordinary residual momentum or event density. This is an association hypothesis under explicit source-vintage and issuer-continuity assumptions.',
        measurement='Use the second XNYS session strictly after reported filingDate as anchor; compare mean beta residuals on anchor offsets2..20 against clean nonfiling days of the same210-session old formation window.',
        not_claimed=['First-publication timing, a pure post-news window, causal filing effects, or original-vintage data.',
                     'Exact reproduction of the author or independent untouched out-of-sample validation.'])
    p['bounded_novelty']['specific_difference'] = (
        'Complete original/amendment administrative filing-date enumeration with a fixed two-session calendar convention, old-window preformation beta and filing/nonfiling contrast. This differs from the previously used latest earnings-release returns. Both late-publication contamination and current-source reconstruction remain disclosed.')
    s = p['systematic_source_scope']
    s['population_rule'] = ('Inspect all73 original equity identities and3904 slots. Use receipt-bound canonical source rows under explicitly reviewed issuer-continuity assumptions and known-transition exclusions; retain every unavailable slot as neutral. No issuer selection by returns.')
    s['sources'] = ('Original SEC submissions roots and all advertised pages intersecting the inverse source interval, exact bytes/receipts, the final source/issuer/exception preflights and frozen price route metadata. Original archive completeness is conditional on source descriptors, not a guarantee against unknown omissions.')
    s['event_schema'] = ['issuer_id','security_id','cik','accessionNumber','form','reportDate',
        'filingDate','raw_acceptanceDateTime','reconstructed_anchor_session','clock_policy_id',
        'decision_cutoff','source_paths_and_sha256','receipt_paths_and_sha256',
        'issuer_continuity_assumption_id','exception_review_id','chronology_review_id']
    s['future_records'] = ('No original/amendment/correction may be treated as known after the original frozen decision_cutoff. A known unresolved source conflict excludes the affected slot, never silently substitutes a later correction into an earlier decision. Unobserved historical revisions are a disclosed data-vintage limitation.')
    p['clock'] = dict(status='RECONSTRUCTED_FILING_DATE_ASSUMPTION_REQUIRES_FINAL_SOURCE_REVIEW',
        id='second_XNYS_session_strictly_after_filingDate_v1',
        anchor_rule='Let F be the accession-bound filingDate; a=second XNYS session whose date is strictly greater than F. No raw timestamp is used to move this anchor.',
        raw_acceptance_sanity='Retain raw timestamp; use independently reviewed exception policy only to reject known cutoff contradictions or malformed required metadata. This is not acceptance-time normalization or public-availability certification.',
        calendar='Original pinned XNYS sessions and original exchange close decision_cutoff; preserve holidays, exceptional closures, DST and early close.',
        inverse_source_interval='[date(S[t-273]),date(S[t-43])) for anchors[t-271,t-42].',
        interpretation='Some labelled return days may precede actual publication. Measured filing-date association is not automatically post-publication drift.',
        no_fallback='The convention is fixed before price features; no alternative anchor, source, sign or issuer subset chosen using returns.')
    p['partition']['same_session_bundle'] = ('Bundle all distinct canonical report keys on the same reconstructed anchor; any amendment makes it a blackout bundle. Known identity or chronology contradictions block the slot.')
    p['partition']['audit'] = ('Persist accessions, F, raw timestamp, reconstructed anchor, exact original cutoff, evidence/assumption IDs, E/B/N labels/owners, clipping flags, bundle counts and210 labels per usable slot.')
    gate = p['gates_before_any_real_factor']
    gate['independent_source_review'] = ('Positive independent review of receipt-bound canonical sources, final issuer/exception intersection, original cutoffs, full unchanged denominators, metadata warmup and the explicit reconstruction assumptions; this is research eligibility, not strict historical certification.')
    gate['broad_coverage_floor']['definition_of_usable'] = ('Intersection of source closure, canonical chronology, reviewed issuer continuity/transition policy, fixed F-only reconstructed clock, known-exception checks, partition minima and metadata-only price warmup. Timestamp-only coverage alone cannot pass this gate.')
    gate['current_status'] = 'Pending final issuer and exception intersection; no actual features or registration authorized yet.'
    p['required_synthetic_cases'][0] = ('Second session strictly after F for session/non-session dates, DST, holidays/exceptional closures, source-envelope endpoints; independently check actual decision close and cutoff contradictions. No exact-acceptance admission is synthesized.')
    # All economic windows, four arms, dose, selection thresholds and execution
    # accounting stay identical to the unexecuted original proposal.
    for key in ('formation_and_nuisance_beta', 'arms', 'fixed_transform',
                'selection_and_evaluation', 'unchanged_framework', 'post_registration_prepare_gate'):
        require(p[key] == old[key], 'Unrelated experiment scope changed: '+key)
    p['reuse_map'][-1]['adapt'] = 'Bind final broad canonical sources, issuer continuity assumptions, known exceptions and F-only calendar policy; old exact-clock admission is not borrowed.'
    src.read_bytes('work/after156_association_protocol.py')
    src.recheck()
    p['input_hashes_at_separate_design'] = dict(src.hashes)
    if args.write:
        raw=(json.dumps(p, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        with src.path(OUT).open('xb') as h:
            h.write(raw)
        print(json.dumps(dict(path=OUT, sha256=hashlib.sha256(raw).hexdigest(), registered=False)))
    else:
        print(json.dumps(p, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
