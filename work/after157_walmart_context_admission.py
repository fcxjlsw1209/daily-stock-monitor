"""Join reviewed WMT annual originals to the unchanged full source grid.

Source-only conditional overlay. No lag, financial factor, rank or return values.
"""
import argparse
import copy
import json
from pathlib import Path

from after157_annual_body_coverage import Reader, require
from after157_annual_body_collect import write_json, utc

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/after157_walmart_context_admission_v1/ADMISSION.json'
MISSING = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
PINS = {
    'work/after157_annual_body_coverage.py': '82a4763b6233d8f5d7d593a261e90a3257788c85037100936e69260e1d002e8c',
    'work/after157_annual_body_collect.py': 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864',
    'work/after157_annual_pair_source_feasibility.json': '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_walmart_bodies_early_scope.json': '4bba503d03789791e8880b1ff50f5d1c4d8e24b97ea7874c8a6df3bb45a15393',
    'work/after157_walmart_bodies_late_scope.json': '11938d1e90419352b83d982647a8be82341582a5c5e9bdf5f695d7d923b6e1c7',
    'outputs/after157_walmart_bodies_early_v1/derived_text/MANIFEST.json': 'ba0614ad2dde21b4d1cd61abd3befe033d338f0afc195d4e0a30ce812863126e',
    'outputs/after157_walmart_bodies_late_v1/derived_text/MANIFEST.json': '0106c446c5a26261df113c55afb1f37c4c3825809b29ec45caacf0807f80f8ff',
    'outputs/after157_ko_context_admission_v1/ADMISSION.json': '6bfab5c682a36deaa503b2027815142c428b70c84b3832f17920ebadd7ec3d9b',
    'work/after157_ko_context_independent.json': '4aab7d30caba3b5d579d8a6b979a65c5e0b650f0841adc2114290925a09108d5',
    'work/after157_walmart_2024_root_review.json': '96100010d393b08bffcdc33298c9e3428f4ead6ef390f98bc85fea49a2d4c264',
}


def collect_pairs(reviews, targets, metadata):
    """Require complete explicit adjudication; a negative review is never rescued."""
    known, accepted, declared, verified = set(), set(), set(), set()
    expected_by_year = {int(t['expected_original']['reportDate'][:4]):
                        t['expected_original']['event_id'] for t in targets}
    require(set(expected_by_year) == set(range(2015, 2027)), 'Unexpected WMT years')
    expected_ids = set(expected_by_year.values())
    require(len(expected_ids) == len(targets) == 12, 'Duplicate WMT targets')
    for target in targets:
        expected = target['expected_original']
        require(all(metadata[expected['event_id']].get(k) == v for k, v in expected.items()),
                'Frozen target metadata changed')
    for review in reviews:
        require(review['source_admission_changed'] is False, 'Review changed prior admission')
        for original in review['originals']:
            event = original['event_id']
            require(event in expected_ids and event not in known, 'Unknown or duplicate original')
            known.add(event)
            item = metadata[event]
            require(item['cik'] == '0000104169' and item['form'] == '10-K', 'Wrong WMT identity')
            require(original['symbol'] == 'WMT' and original['cik'] == item['cik'] and
                    original['form'] == item['form'] and
                    type(original['fiscal_year']) is int and
                    original['fiscal_year'] == int(item['reportDate'][:4]),
                    'Review original identity or fiscal year disagrees with frozen metadata')
            for key, field in [('accession', 'accessionNumber'), ('primary_document', 'primaryDocument'),
                               ('report_date', 'reportDate')]:
                require(original[key] == item[field], 'Original binding mismatch')
            flags = [original['exact_original_version_bound'], original['normal_annual_context_verified']]
            require(all(type(v) is bool for v in flags), 'Nonboolean original judgement')
            if all(flags):
                accepted.add(event)
    require(known == expected_ids, 'Incomplete original review')
    expected_pairs = {frozenset((expected_by_year[y], expected_by_year[y+1])) for y in range(2015, 2026)}
    for review in reviews:
        for pair in review['pairs']:
            ids = pair['event_ids']
            key = frozenset(ids)
            require(len(ids) == len(key) == 2 and key in expected_pairs and key not in declared,
                    'Unexpected, duplicate or nonconsecutive WMT pair')
            declared.add(key)
            flags = [pair['normal_consecutive_pair_context_verified'], pair['exact_original_versions_bound']]
            require(all(type(v) is bool for v in flags), 'Nonboolean pair judgement')
            if all(flags):
                require(key <= accepted, 'Positive pair contains unresolved original')
                verified.add(key)
    require(declared == expected_pairs, 'Incomplete pair review')
    return verified


def join_rows(rows, ko_rows, wmt_pairs, metadata):
    require(len(rows) == len(ko_rows), 'KO grid shape changed')
    out = []
    for original, ko in zip(rows, ko_rows):
        require(ko['original_row'] == original, 'KO original row changed')
        ids = original['selected_event_ids']
        wmt = len(ids) == len(set(ids)) == 2 and frozenset(ids) in wmt_pairs
        if wmt:
            require(not original['is_etf'] and original['symbol'] == 'WMT' and
                    original['cik'] == '0000104169' and
                    all(metadata[e]['cik'] == original['cik'] for e in ids), 'WMT pair joined to wrong row')
        matched = bool(ko['body_pair_context_verified'] or wmt)
        remaining = [reason for reason in original['reasons'] if not (matched and reason == MISSING)]
        out.append(dict(original_row=copy.deepcopy(original), body_pair_context_verified=matched,
            body_context_source='WMT' if wmt else ('KO' if ko['body_pair_context_verified'] else None),
            resolved_context_reason=[MISSING] if matched and MISSING in original['reasons'] else [],
            remaining_row_reasons=remaining,
            conditional_annual_pair_admitted=bool(matched and original['tier2'] and not remaining),
            old_admission_flags_changed=False))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--early-review-sha', required=True)
    parser.add_argument('--late-review-sha', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    reader = Reader(ROOT)
    reader.pins(PINS)
    reader.read('work/after157_walmart_context_admission.py')
    prior = reader.json('work/after157_annual_pair_source_feasibility.json')
    ko = reader.json('outputs/after157_ko_context_admission_v1/ADMISSION.json')
    independent = reader.json('work/after157_ko_context_independent.json')
    require(independent['passed'] is True, 'KO independent review failed')
    reader.pins(independent['input_hashes'])
    root_review = reader.json('work/after157_walmart_2024_root_review.json')
    require(root_review['passed'] is True and root_review['source_admission_changed'] is False and
            all(v is True for v in root_review['checks'].values()), 'FY2024 independent check failed')
    reader.pins(root_review['input_hashes'])
    targets, reviews = [], []
    for batch, digest in [('early', args.early_review_sha), ('late', args.late_review_sha)]:
        scope_path = 'work/after157_walmart_bodies_' + batch + '_scope.json'
        scope = reader.json(scope_path)
        targets.extend(scope['targets'])
        manifest_path = scope['output_dir'] + '/derived_text/MANIFEST.json'
        manifest = reader.json(manifest_path)
        reader.pins(manifest['input_hashes'])
        for record in manifest['records']:
            reader.read(scope['output_dir'] + '/derived_text/' + record['text_file'], record['text_sha256'])
        review = reader.json('work/after157_walmart_' + batch + '_review.json', digest)
        require(review['input_hashes'].get(scope_path) == PINS[scope_path] and
                review['input_hashes'].get(manifest_path) == PINS[manifest_path], 'Review is not bound to batch')
        reader.pins(review['input_hashes'])
        require({r['event_id'] for r in review['originals']} ==
                {t['expected_original']['event_id'] for t in scope['targets']}, 'Review batch identity mismatch')
        reviews.append(review)
    pairs = collect_pairs(reviews, targets, prior['selected_original_and_amendment_metadata'])
    rows = join_rows(prior['rows'], ko['rows'], pairs, prior['selected_original_and_amendment_metadata'])
    require(len(rows) == 5978 and all(r['original_row']['tier3'] is False for r in rows), 'Original grid changed')
    summary = {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        part = [r for r in rows if lo <= r['original_row']['signal_index'] < hi]
        accepted = [r for r in part if r['conditional_annual_pair_admitted']]
        summary[name] = dict(all_slots=len(part), stock_slots=sum(not r['original_row']['is_etf'] for r in part),
            conditional_pair_slots=len(accepted),
            by_issuer={symbol:sum(r['body_context_source'] == symbol for r in accepted) for symbol in ['KO', 'WMT']},
            original_tier3_slots=sum(r['original_row']['tier3'] for r in part))
    payload = dict(schema='after157_ko_wmt_conditional_annual_source_v1', created_utc=utc(),
        status='SOURCE_ONLY_CONDITIONAL_OVERLAY_NO_STRATEGY_EXPERIMENT',
        source_admission_changed=False,
        old_admission_flags_changed=False, new_conditional_context_admission=True,
        full_strategy_admission=False, numerical_experiment_registered=False,
        lag_or_feature_values_computed=False, summary=summary, rows=rows,
        verified_wmt_event_pairs=[sorted(p) for p in sorted(pairs, key=lambda x: sorted(x))],
        input_hashes=dict(reader.hashes),
        limits=['Only removes the exact original annual-context blocker when reviewed; every other gate remains.',
                'Conditional source/identity and reconstructed administrative filing clock, not exact historical vintage certification.',
                'Only KO and WMT; neither broad cross-sectional coverage nor return-improvement evidence.'])
    reader.reverify()
    if args.write:
        path = reader.path(OUT)[0]
        path.parent.mkdir(exist_ok=True)
        write_json(path, payload)
    print(json.dumps(dict(output=OUT if args.write else None, summary=summary, verified_wmt_pairs=len(pairs),
                          numerical_experiment_registered=False)))


if __name__ == '__main__':
    main()
