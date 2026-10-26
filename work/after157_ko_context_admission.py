"""A new conditional source overlay, leaving the original feasibility immutable.

No lag values, financial variables, prices, ranks or returns are calculated.
Semantic body decisions come from separately pinned, accession-specific reviews.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from after157_annual_body_coverage import Reader, require
from after157_annual_body_collect import write_json

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_ko_context_admission.py'
OUT = 'outputs/after157_ko_context_admission_v1/ADMISSION.json'
MISSING_CONTEXT = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
PINS = {
    'work/after157_annual_body_coverage.py': '82a4763b6233d8f5d7d593a261e90a3257788c85037100936e69260e1d002e8c',
    'work/after157_annual_body_collect.py': 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864',
    'work/after157_annual_pair_source_feasibility.json': '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_annual_pair_source_feasibility_plan.json': '5750f95e17d8891f7781962efed7bd841408385cce1abfeee1a345827d3fb57d',
    'work/after157_ko_body_review.json': '0a09cc05afdab1c3c8222ca3538cb8b86fe06c67aaede074a20c9ee52a33e117',
    'work/after157_ko_extension_review.json': '9e7438300eba9d6fff22d596ec3979d94198af04eb29f10d627b27860ac9406f',
    'work/after157_ko_early_bodies_scope.json': '05cdb5d38e50c2d1a06ebedcfd6680d2ef8bddea9923a481385fa052c6e2e0a2',
    'migration/after157_annual_body_source_checkpoint.json': '27c85077304ded881d9d58884e0406c394cfe6ed82d3df0eb7693df22c28377f',
    'work/after157_body_checkpoint_independent.json': 'e7e2bda4838fc0a4d12f4af6d937d69723afa889097ffc3826b3d57835a28843',
}


def admit_rows(rows, verified_pairs, metadata):
    result = []
    for row in rows:
        ids = row['selected_event_ids']
        matched = len(ids) == 2 and len(set(ids)) == 2 and frozenset(ids) in verified_pairs
        if matched:
            require(row['symbol'] == 'KO' and not row['is_etf'] and row['cik'] == '0000021344',
                    'Context joined to wrong issuer')
            require(all(metadata[e]['cik'] == row['cik'] for e in ids), 'Metadata issuer conflict')
        # Retain all original evidence, rather than dropping non-admitted slots.
        reasons = list(row['reasons'])
        unresolved = [r for r in reasons if not (matched and r == MISSING_CONTEXT)]
        admitted = bool(matched and row['tier2'] and not unresolved)
        result.append(dict(original_row=copy.deepcopy(row), body_pair_context_verified=matched,
            resolved_context_reason=[MISSING_CONTEXT] if matched and MISSING_CONTEXT in reasons else [],
            remaining_row_reasons=unresolved, conditional_annual_pair_admitted=admitted,
            old_admission_flags_changed=False))
    return result


def add_review(review, expected_ids, metadata, known, pairs):
    originals = review['originals']
    require(len(originals) == len(expected_ids), 'Unexpected original count')
    require({o['event_id'] for o in originals} == expected_ids, 'Wrong original IDs')
    for original in originals:
        event = original['event_id']
        require(original['exact_original_version_bound'] is True and
                original['normal_annual_context_verified'] is True, 'Original context unresolved')
        match = metadata[event]
        require(match['cik'] == '0000021344' and match['form'] == '10-K', 'Not ordinary KO original')
        for field, raw_field in [('accession', 'accessionNumber'), ('primary_document', 'primaryDocument'),
                                 ('report_date', 'reportDate')]:
            require(original[field] == match[raw_field], 'Original version binding mismatch')
        known.add(event)
    for pair in review['pairs']:
        ids = pair['event_ids']
        require(len(ids) == 2 and len(set(ids)) == 2 and set(ids) <= known, 'Unknown pair IDs')
        require(pair['normal_consecutive_pair_context_verified'] is True and
                pair['exact_original_versions_bound'] is True, 'Pair context unresolved')
        key = frozenset(ids)
        require(key not in pairs, 'Duplicate reviewed pair')
        pairs.add(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--early-review-sha', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    reader = Reader(ROOT)
    reader.pins(PINS)
    reader.read(SELF)
    prior = reader.json('work/after157_annual_pair_source_feasibility.json')
    metadata = prior['selected_original_and_amendment_metadata']
    reviewed = reader.json('work/after157_body_checkpoint_independent.json')
    require(reviewed['passed'] is True, 'Prior independent audit not passed')
    reader.pins(reviewed['input_hashes'])
    old = reader.json('work/after157_ko_body_review.json')
    reader.pins(old['input_hashes'])
    require(old['exact_original_versions_bound'] is True and
            old['normal_consecutive_pair_context_verified'] is True, 'Old pair context unresolved')
    known = set(old['expected_event_ids'])
    require(len(known) == 2 and all(metadata[e]['cik'] == '0000021344' for e in known), 'Old pair changed')
    pairs = {frozenset(known)}
    extension = reader.json('work/after157_ko_extension_review.json')
    reader.pins(extension['input_hashes'])
    extension_ids = {e for e, v in metadata.items() if v['cik'] == '0000021344'
                     and v['form'] == '10-K' and '2018-12-31' <= v['reportDate'] <= '2025-12-31'}
    require(len(extension_ids) == 8, 'Extension metadata changed')
    add_review(extension, extension_ids, metadata, known, pairs)
    early = reader.json('work/after157_ko_early_review.json', args.early_review_sha)
    reader.pins(early['input_hashes'])
    early_scope = reader.json('work/after157_ko_early_bodies_scope.json')
    require(early['input_hashes'].get('work/after157_ko_early_bodies_scope.json') ==
            PINS['work/after157_ko_early_bodies_scope.json'], 'Early review not bound to scope')
    early_ids = {t['expected_original']['event_id'] for t in early_scope['targets']}
    add_review(early, early_ids, metadata, known, pairs)
    require(len(known) == 12 and len(pairs) == 11, 'Expected all twelve KO originals and eleven pairs')
    require(all(review['source_admission_changed'] is False for review in [old, extension, early]),
            'Semantic review unexpectedly changed old admission')
    rows = admit_rows(prior['rows'], pairs, metadata)
    require(len(rows) == 5978 and all(not r['original_row']['tier3'] for r in rows), 'Frozen grid changed')
    summary = {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        part = [r for r in rows if lo <= r['original_row']['signal_index'] < hi]
        accepted = [r for r in part if r['conditional_annual_pair_admitted']]
        summary[name] = dict(all_slots=len(part), stock_slots=sum(not r['original_row']['is_etf'] for r in part),
            conditional_pair_slots=len(accepted), issuers=sorted({r['original_row']['symbol'] for r in accepted}),
            original_tier3_slots=sum(r['original_row']['tier3'] for r in part))
    payload = dict(schema='after157_conditional_ko_annual_pair_source_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='SOURCE_ONLY_CONDITIONAL_OVERLAY_NO_STRATEGY_EXPERIMENT',
        old_admission_flags_changed=False, new_conditional_context_admission=True,
        full_strategy_admission=False, numerical_experiment_registered=False,
        lag_or_feature_values_computed=False, summary=summary, rows=rows,
        verified_event_pairs=[sorted(p) for p in sorted(pairs, key=lambda p: sorted(p))],
        input_hashes=dict(reader.hashes),
        basis='Intersection of original tier2, separately reviewed exact-original normal annual pairs, and no remaining row-level blocker.',
        limits=['Retains all original source, t-42, bad-latest/no-fallback, amendment and conditional identity gates.',
                'Administrative filing-date reconstruction only; no exact publication-clock or historical-vintage certification.',
                'Existing nested original-context and identity evidence remains visible; no financial-statement or general issuer certification.',
                'Coverage is KO only, not a broad cross-sectional sample or a return-improvement result.',
                'Old frozen tier3 flags remain unchanged; this is a new versioned overlay.'])
    reader.reverify()
    if args.write:
        output = reader.path(OUT)[0]
        output.parent.mkdir(exist_ok=True)
        write_json(output, payload)
    print(json.dumps(dict(output=OUT if args.write else None, summary=summary,
                          old_admission_flags_changed=False, numerical_experiment_registered=False)))


if __name__ == '__main__':
    main()
