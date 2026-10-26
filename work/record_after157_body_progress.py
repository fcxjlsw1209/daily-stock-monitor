"""Publish source-only progress and preserve every prior strategy/admission field."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import after157_annual_body_collect as collector
from after157_annual_body_coverage import Reader, require
import monthly_filing_association157_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'migration/after157_annual_body_source_checkpoint.json'
PINS = {
    'work/after157_annual_pair_source_feasibility.json': '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_annual_body_coverage.py': '82a4763b6233d8f5d7d593a261e90a3257788c85037100936e69260e1d002e8c',
    'outputs/after157_annual_period_source_v1/COVERAGE.json': '7fc89ea6c7e72630c07785a3e56877ebf5401fbd960ec60dca3aef17350659b6',
    'work/after157_ko_body_review.json': '0a09cc05afdab1c3c8222ca3538cb8b86fe06c67aaede074a20c9ee52a33e117',
    'work/after157_msft_body_review.json': 'acbf0305cd19f9222a2ce67e9704d4af41d66243f20be494e14095b622285157',
    'work/after157_ko_extension_scope.json': 'bafa244ba87dfe6cea59c4788448bbe0cba45f9a09cf8b3a8940af6c8336242d',
    'work/after157_msft_bridge_scope.json': '68780c8f420b2e035248843fc4fac84adc2e2c9bcbcad6f215c3f7df0bc44241',
    'outputs/after157_ko_annual_extension_v1/COLLECTION.json': '027ffa839d6d32d7ce97b830ca885f28f623f036d6df64daf1e01782e1626ad5',
    'outputs/after157_ko_annual_extension_v1/derived_text/MANIFEST.json': 'a99bec0cd79ac55ac784af52a2c881efc073deb3eb8b93b69027df4b46a15fbe',
    'outputs/after157_msft_bridge_v1/COLLECTION.json': '02f395834a3464008d3dbd2d695471d4acf7d6d636f52f6fa3eeb19a5537a156',
    'work/after157_lag_hypothesis_context.json': '1492c6c41a833e6a7a95b83ef56c59be413134eb17a435a0c4af735652cde72b',
}


def reference(reader, name):
    reader.read(name)
    return dict(path=name, sha256=reader.hashes[name])


def verify_batch(reader, scope_name):
    scope, out, helper = collector.validate_scope(ROOT, scope_name, PINS[scope_name])
    reader.pins(scope['input_hashes'])
    collection_name = scope['output_dir'] + '/COLLECTION.json'
    result = reader.json(collection_name, PINS[collection_name])
    require(result['all_successful'] is True and result['stopped_providers'] == [], 'Collection incomplete')
    require(result['scope_sha256'] == PINS[scope_name] and result['helper_sha256'] == helper,
            'Collection binding mismatch')
    require({r['id'] for r in result['targets']} == {t['id'] for t in scope['targets']}, 'Target set mismatch')
    for target in scope['targets']:
        receipt = collector.verify_existing(out, target, collector.base_binding(target, PINS[scope_name], helper))
        require(receipt is not None and receipt['outcome'] == 'success', 'Missing source receipt')
        summary = [r for r in result['targets'] if r['id'] == target['id']]
        require(len(summary) == 1, 'Duplicate source receipt')
        for name, expected in [(target['receipt_file'], summary[0]['receipt_sha256']),
                               (target['body_file'], receipt['sha256']),
                               ('.attempt_' + target['id'] + '.json', receipt['attempt_sha256'])]:
            reader.read(scope['output_dir'] + '/' + name, expected)
    return scope


def contextual_rows(prior_rows, pairs, metadata):
    """Exact event-pair join only; this never changes an original gate."""
    output = []
    for row in prior_rows:
        ids = row['selected_event_ids']
        pair = frozenset(ids)
        known = len(ids) == 2 and pair in pairs
        if known:
            require(not row['is_etf'] and row['symbol'] == 'KO', 'Body evidence joined to wrong security')
            require(all(metadata[e]['cik'] == row['cik'] == '0000021344' for e in ids), 'Wrong CIK')
        output.append(dict(signal_index=row['signal_index'], candidate_index=row['candidate_index'],
            signal_date=row['signal_date'], symbol=row['symbol'], is_etf=row['is_etf'],
            selected_event_ids=copy.deepcopy(ids), prior_tier2=row['tier2'], prior_tier3=row['tier3'],
            prior_reasons=copy.deepcopy(row['reasons']),
            prior_existing_accession_context=copy.deepcopy(row.get('existing_accession_context', [])),
            body_pair_context_verified=known, source_admission_changed=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ko-review-sha', required=True)
    parser.add_argument('--msft-review-sha', required=True)
    parser.add_argument('--coverage-review-sha', required=True)
    args = parser.parse_args()
    reader = Reader(ROOT)
    reader.pins(PINS)
    reader.read('work/record_after157_body_progress.py')
    prior = reader.json('work/after157_annual_pair_source_feasibility.json')
    first_coverage = reader.json('outputs/after157_annual_period_source_v1/COVERAGE.json')
    reader.pins(first_coverage['input_hashes'])
    review_paths = {
        'work/after157_ko_extension_review.json': args.ko_review_sha,
        'work/after157_msft_bridge_review.json': args.msft_review_sha,
        'work/after157_body_coverage_independent.json': args.coverage_review_sha,
    }
    reviews = {}
    for name, expected in review_paths.items():
        review = reader.json(name, expected)
        reader.pins(review['input_hashes'])
        require(review['source_admission_changed'] is False, 'Review unexpectedly changes admission')
        reviews[name] = review
    coverage_review = reviews['work/after157_body_coverage_independent.json']
    require(coverage_review['passed'] is True, 'Independent first-batch coverage check failed')
    require(coverage_review['input_hashes'].get('outputs/after157_annual_period_source_v1/COVERAGE.json') ==
            PINS['outputs/after157_annual_period_source_v1/COVERAGE.json'], 'Unbound coverage review')
    ko_scope = verify_batch(reader, 'work/after157_ko_extension_scope.json')
    verify_batch(reader, 'work/after157_msft_bridge_scope.json')
    derived_name = 'outputs/after157_ko_annual_extension_v1/derived_text/MANIFEST.json'
    derived = reader.json(derived_name)
    require(derived['scope_sha256'] == PINS['work/after157_ko_extension_scope.json'] and
            derived['collection_sha256'] == PINS['outputs/after157_ko_annual_extension_v1/COLLECTION.json'],
            'Derived source binding mismatch')
    reader.pins(derived['input_helpers'])
    reader.read('work/inspect_after157_ko_extension.py', derived['extractor_sha256'])
    for record in derived['records']:
        reader.read('outputs/after157_ko_annual_extension_v1/derived_text/' + record['text_file'], record['text_sha256'])
    old_ko = reader.json('work/after157_ko_body_review.json')
    require(old_ko['exact_original_versions_bound'] and old_ko['normal_consecutive_pair_context_verified'],
            'Prior KO pair review changed')
    pairs = {frozenset(old_ko['expected_event_ids'])}
    metadata = prior['selected_original_and_amendment_metadata']
    targets = {t['expected_original']['event_id']: t for t in ko_scope['targets']}
    ko_review = reviews['work/after157_ko_extension_review.json']
    for name in ('work/after157_ko_extension_scope.json',
                 'outputs/after157_ko_annual_extension_v1/COLLECTION.json', derived_name,
                 'work/after157_ko_body_review.json'):
        require(ko_review['input_hashes'].get(name) == PINS[name], 'Unbound KO extension review: ' + name)
    originals = {o['event_id']: o for o in ko_review['originals']}
    require(len(ko_review['originals']) == len(originals) == 8 and set(originals) == set(targets),
            'Incomplete or duplicate original review')
    eligible = set(old_ko['expected_event_ids'])
    for event_id, target in targets.items():
        require(all(metadata[event_id].get(k) == v for k, v in target['expected_original'].items()), 'Event changed')
        original = originals[event_id]
        require(type(original['exact_original_version_bound']) is bool and
                type(original['normal_annual_context_verified']) is bool, 'Nonboolean body review')
        if original['exact_original_version_bound'] and original['normal_annual_context_verified']:
            eligible.add(event_id)
    seen_pairs = set()
    for pair_review in ko_review['pairs']:
        ids = pair_review['event_ids']
        require(len(ids) == 2 and len(set(ids)) == 2 and
                set(ids) <= set(targets) | set(old_ko['expected_event_ids']), 'Unknown pair original')
        require(frozenset(ids) not in seen_pairs, 'Duplicate pair review')
        seen_pairs.add(frozenset(ids))
        require(type(pair_review['normal_consecutive_pair_context_verified']) is bool and
                type(pair_review['exact_original_versions_bound']) is bool, 'Nonboolean pair review')
        if pair_review['normal_consecutive_pair_context_verified'] and pair_review['exact_original_versions_bound']:
            require(set(ids) <= eligible, 'Positive pair contains an unverified original')
            pairs.add(frozenset(ids))
    rows = contextual_rows(prior['rows'], pairs, metadata)
    require(len(rows) == 5978 and all(not r['prior_tier3'] for r in rows), 'Prior grid/admission changed')
    summary = {}
    for segment, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        part = [r for r in rows if lo <= r['signal_index'] < hi]
        summary[segment] = dict(candidate_slots=len(part), stock_slots=sum(not r['is_etf'] for r in part),
            documented_pair_slots=sum(r['body_pair_context_verified'] for r in part),
            tier2_with_documented_pair_slots=sum(r['prior_tier2'] and r['body_pair_context_verified'] for r in part),
            prior_tier3_slots=sum(r['prior_tier3'] for r in part))
    body_slots = summary['all122']['tier2_with_documented_pair_slots']
    next_step = (
        '157 rounds / 986 trials remain complete; incumbent67.66617372041686%, below70%. '
        f'Annual source-only body review now covers {body_slots} conditional stock slots; '
        'no original Tier3 or strategy admission has been changed and no lag value was computed. '
        'KO FY2014/FY2015 originals remain outside this batch; the remaining issuers need exact originals '
        'and normal annual-pair context before a lag-change experiment can be registered. '
        'Microsoft year-page retrieval is not an original-version bridge: follow only newly scoped '
        'official links after inspecting the pinned review; do not retry identical pages or bypass provider stops. '
        'Keep t-42/source/identity/amendment gates and full5978 grid. Any numerical experiment requires '
        'prospective registration and comparison to current-lag control; no frozen observer changes.'
    )
    reader.reverify()
    with lifecycle.state_lock(ROOT):
        state, before = lifecycle.state_snapshot(ROOT / lifecycle.STATE)
        require(state['current'] == 'monthly_filing_association157' and
                state['current_status'] == 'completed_independently_verified' and
                len(state['completed']) == 157 and sum(x['trials'] for x in state['completed']) == 986 and
                state['registered_queue'] == [], 'Concurrent experiment state changed')
        require(state['best_training_selected_candidate']['cagr'] == 0.6766617372041686, 'Incumbent changed')
        checkpoint = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
            classification_basis='New official source bodies, bounded bridge-page evidence, independent document reviews and full-grid coverage.',
            completed_rounds=157, strategy_trials=986, new_strategy_trials=0,
            target_reached=False, incumbent=copy.deepcopy(state['best_training_selected_candidate']),
            source_admission_changed=False, source_summary=summary, rows=rows,
            documented_event_pairs=[sorted(pair) for pair in sorted(pairs, key=lambda x: sorted(x))],
            input_hashes=dict(reader.hashes), next_step=next_step,
            limits=['Document context is a side-column, not replacement for original source/identity/clock gates.',
                    'Current retrieval does not certify exact historical publication time or historical data vintage.',
                    'No L/DeltaL, factors, ranks, orders or returns computed in this source-only step.',
                    'Both historical segments were repeatedly examined; neither is pristine out of sample.'])
        path = ROOT / CHECKPOINT
        if path.exists():
            existing = reader.json(CHECKPOINT)
            comparable = lambda value: {k: v for k, v in value.items() if k != 'created_utc'}
            require(comparable(existing) == comparable(checkpoint), 'Conflicting existing checkpoint')
            checkpoint = existing
        else:
            lifecycle.write(path, checkpoint)
        checkpoint_ref = reference(reader, CHECKPOINT)
        if state.get('after157_annual_body_source_checkpoint') == checkpoint_ref:
            print(json.dumps(dict(checkpoint=checkpoint_ref, already_recorded=True,
                                  target_reached=False, new_strategy_trials=0)))
            return
        state.update(after157_annual_body_source_checkpoint=reference(reader, CHECKPOINT),
                     next_step=next_step, last_goal_turn_classification='progress',
                     updated_utc=datetime.now(timezone.utc).isoformat())
        lifecycle.update_state(ROOT / lifecycle.STATE, before, state)
    print(json.dumps(dict(checkpoint=reference(reader, CHECKPOINT), source_summary=summary,
                          target_reached=False, new_strategy_trials=0)))


if __name__ == '__main__':
    main()
