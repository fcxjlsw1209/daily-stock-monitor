"""Publish bounded source progress; never run a factor, strategy, or collector.

The sole historical mutable-state dependency is relocated to an exact-byte
PRE157 archive under the existing lifecycle lock. Original evidence is immutable.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import after157_annual_body_collect as files
from after157_annual_body_coverage import Reader, require
import after157_walmart_collect as walmart
import monthly_filing_association157_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/record_after157_source_extension.py'
TEST = 'work/test_record_after157_source_extension.py'
CHECKPOINT = 'migration/after157_source_extension_checkpoint.json'
REF_KEY = 'after157_source_extension_checkpoint'
HISTORICAL_SHA = 'f0f6571a3f4de4be06aeb957fe03fd13eed11f86b288abafc91094db38ac32b7'
ARCHIVE = 'outputs/monthly_target70_research/PRE157_' + HISTORICAL_SHA + '.json'
KO = 'outputs/after157_ko_context_admission_v1/ADMISSION.json'
KO_REVIEW = 'work/after157_ko_context_independent.json'
MSFT = 'outputs/after157_msft_iframe_v1'
MSFT_SCOPE = 'work/after157_msft_iframe_scope.json'
MSFT_URL = 'https://microsoft.gcs-web.com/financial-information/sec-filings'
WMT_SCOPES = ('work/after157_walmart_bodies_early_scope.json',
              'work/after157_walmart_bodies_late_scope.json')
WMT_OVERLAY = 'outputs/after157_walmart_context_admission_v1/ADMISSION.json'
WMT_OVERLAY_SCHEMA = 'after157_ko_wmt_conditional_annual_source_v1'
PRIOR = 'work/after157_annual_pair_source_feasibility.json'
PRIOR_SHA = '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242'
MISSING_CONTEXT = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
PINS = {
    KO: '6bfab5c682a36deaa503b2027815142c428b70c84b3832f17920ebadd7ec3d9b',
    KO_REVIEW: '4aab7d30caba3b5d579d8a6b979a65c5e0b650f0841adc2114290925a09108d5',
    'work/after157_ko_context_independent.py': 'e354fc583318d6159d88dd27071051c54f973bc09a33247d2a7e9df17ef6c91d',
    MSFT_SCOPE: 'ff1fa90e4fc997593257ec2ffeb370d82c08fa3f178e8611bc4683d305bbbb6c',
    MSFT + '/msft_sec_filings_iframe.receipt.json': '068ab70c6082ba81424d644189a7dcc6e5a19dbcfa55f33cb74bb8089fc0e3cc',
    MSFT + '/STOPPED.json': 'fe5fffb7591e4e0b833700941742264546e157f85ce7bbf0d8ec9d430baa7135',
    MSFT + '/COLLECTION.json': 'fc9f958aa06d29d1697d3aabd854913dd1f4375285c6e9269385aabb13b2fc13',
    WMT_SCOPES[0]: '4bba503d03789791e8880b1ff50f5d1c4d8e24b97ea7874c8a6df3bb45a15393',
    WMT_SCOPES[1]: '11938d1e90419352b83d982647a8be82341582a5c5e9bdf5f695d7d923b6e1c7',
    'outputs/after157_walmart_bodies_early_v1/COLLECTION.json': '1f62e75a5c4ab68cb4f909e16119dc90e671141957d3b02f76b6ac3471dfd432',
    'outputs/after157_walmart_bodies_late_v1/COLLECTION.json': '2e15d04dcfc5b2a9b99bf6025adefa615af710dd57fb7237d3a0137c6aa11995',
    'work/after157_walmart_collect.py': 'dd80db6f8fbc3636640098164f688a1973f37749f2581cea19aa058893a4f592',
    'work/after157_annual_body_collect.py': 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864',
    'work/after157_annual_body_coverage.py': '82a4763b6233d8f5d7d593a261e90a3257788c85037100936e69260e1d002e8c',
    'work/monthly_filing_association157_lifecycle.py': 'eca5c16b9d171caa0a155251f7e4a6a3bbbf6ea9cda9231971377a0dd0075617',
    'work/complete_monthly_same_report145.py': '50e8900d387856293f324da0c826a8c1ed3b45974205baa0ec8a5777089c3c38',
}


def now():
    return datetime.now(timezone.utc).isoformat()


class SourceReader(Reader):
    def read(self, name, expected=None):
        # No generic stale-pin fallback: exactly this one historical state only.
        if name == lifecycle.STATE:
            require(expected == HISTORICAL_SHA, 'Unrecognized historical state pin')
            name = ARCHIVE
        return super().read(name, expected)


def preserve_historical_state(root):
    reader = Reader(root)
    path = reader.path(ARCHIVE)[0]
    if not path.exists():
        raw = reader.read(lifecycle.STATE, HISTORICAL_SHA)
        files.write_once(path, raw)
    reader.read(ARCHIVE, HISTORICAL_SHA)


def ref(reader, name):
    reader.read(name)
    return dict(path=name, sha256=reader.hashes[name])


def validate_state(state):
    rows = state['completed']
    require(state['current'] == 'monthly_filing_association157' and
            state['current_status'] == 'completed_independently_verified' and
            len(rows) == 157 and rows[-1]['directory'] == state['current'] and
            all(type(r['trials']) is int and r['trials'] >= 0 for r in rows) and
            sum(r['trials'] for r in rows) == 986 and state['registered_queue'] == [],
            'Completed/registered experiment state changed')
    require(state['best_training_selected_candidate']['cagr'] == 0.6766617372041686,
            'Incumbent changed')


def verify_ko(reader):
    overlay, review = reader.json(KO), reader.json(KO_REVIEW)
    reader.pins(overlay['input_hashes'])
    reader.pins(review['input_hashes'])
    require(review['passed'] is True and review['source_admission_changed'] is False and
            review['input_hashes'].get(KO) == PINS[KO] and
            len(review['checks']) == 19 and all(v is True for v in review['checks'].values()),
            'KO independent review not passed/bound')
    require(overlay['summary'] == review['summary'], 'KO summary disagrees with review')
    require(len(overlay['rows']) == 5978 and all(r['original_row']['tier3'] is False for r in overlay['rows']),
            'KO frozen grid/admission changed')
    for name, slots, conditional in [('all122', 5978, 109), ('first88', 4312, 81), ('later34', 1666, 28)]:
        summary = overlay['summary'][name]
        require(summary['all_slots'] == slots and summary['conditional_pair_slots'] == conditional and
                summary['original_tier3_slots'] == 0 and summary['issuers'] == ['KO'], 'KO coverage changed')
    require(overlay['old_admission_flags_changed'] is False and overlay['full_strategy_admission'] is False and
            overlay['numerical_experiment_registered'] is False and overlay['lag_or_feature_values_computed'] is False,
            'Unexpected KO admission/activity')
    return dict(status='independently_verified_conditional_body_context', overlay=ref(reader, KO),
                review=ref(reader, KO_REVIEW), summary=copy.deepcopy(overlay['summary']),
                original_tier3_changed=False, full_strategy_admission=False)


def verify_msft_stop(reader):
    scope = reader.json(MSFT_SCOPE)
    reader.pins(scope['input_hashes'])
    receipt_name = MSFT + '/msft_sec_filings_iframe.receipt.json'
    receipt, stop = reader.json(receipt_name), reader.json(MSFT + '/STOPPED.json')
    collection = reader.json(MSFT + '/COLLECTION.json')
    require(scope['url'] == receipt['source_url'] == receipt['final_url'] == stop['source_url'] == MSFT_URL,
            'MSFT failed source URL changed')
    require(receipt['http_status'] == stop['http_status'] == 403 and receipt['outcome'] == 'http_error' and
            receipt['stop_provider'] is True and stop['further_requests_permitted'] is False and
            stop['receipt_sha256'] == collection['receipt_sha256'] == PINS[receipt_name] and
            collection['all_successful'] is False and receipt['redirect_requests'] == 0,
            'MSFT stop/receipt mismatch')
    require(receipt['scope_sha256'] == stop['scope_sha256'] == collection['scope_sha256'] == PINS[MSFT_SCOPE] and
            receipt['helper_sha256'] == stop['helper_sha256'] == collection['helper_sha256'] == scope['helper_sha256'],
            'MSFT scope/helper mismatch')
    raw = reader.read(MSFT + '/' + receipt['body_file'], receipt['sha256'])
    reader.read(MSFT + '/.attempt.json', receipt['attempt_sha256'])
    attempt = reader.json(MSFT + '/.attempt.json')
    require(all(attempt[k] == receipt[k] for k in ('scope_sha256', 'helper_sha256', 'source_url', 'body_file')),
            'MSFT attempt mismatch')
    require(len(raw) == receipt['bytes'] == 424 and stop['retry_after'] is None and
            receipt['headers'].get('Retry-After') is None, 'MSFT failure metadata changed')
    return dict(status='http403_stopped_not_admitted', receipt=ref(reader, receipt_name),
                stop=ref(reader, MSFT + '/STOPPED.json'), http_status=403, bytes=424,
                retry_after=None, further_requests_permitted=False, source_admission_changed=False)


def verify_wmt_batch(reader, scope_name):
    scope, out, helper = walmart.validate_scope(reader.root, scope_name, PINS[scope_name])
    reader.pins(scope['input_hashes'])
    collection_name = scope['output_dir'] + '/COLLECTION.json'
    result = reader.json(collection_name, PINS[collection_name])
    binding = dict(schema=walmart.SCHEMA, scope_sha256=PINS[scope_name], helper_sha256=helper)
    require(reader.json(scope['output_dir'] + '/.collection-scope.json') == binding,
            'WMT collection ownership mismatch')
    require(all(result.get(k) == v for k, v in binding.items()) and result['all_successful'] is True and
            result['stopped_providers'] == [], 'WMT collection incomplete')
    require(len(result['targets']) == len(scope['targets']) == 6 and
            {x['id'] for x in result['targets']} == {t['id'] for t in scope['targets']}, 'WMT target set mismatch')
    targets = []
    for target in scope['targets']:
        receipt = walmart.verify_existing(out, target, walmart.base_binding(target, PINS[scope_name], helper))
        require(receipt is not None and receipt['attempted'] is True and receipt['outcome'] == 'success' and
                receipt['body_truncated_at_limit'] is False, 'Unverified WMT body')
        summaries = [x for x in result['targets'] if x['id'] == target['id']]
        require(len(summaries) == 1 and summaries[0]['outcome'] == 'success' and
                summaries[0]['receipt_file'] == target['receipt_file'], 'WMT summary mismatch')
        receipt_name, raw_name = scope['output_dir'] + '/' + target['receipt_file'], scope['output_dir'] + '/' + target['body_file']
        reader.read(receipt_name, summaries[0]['receipt_sha256'])
        reader.read(raw_name, receipt['sha256'])
        reader.read(scope['output_dir'] + '/.attempt_' + target['id'] + '.json', receipt['attempt_sha256'])
        targets.append(dict(id=target['id'], source_url=receipt['source_url'], final_url=receipt['final_url'],
                            http_status=receipt['http_status'], bytes=receipt['bytes'],
                            raw=ref(reader, raw_name), receipt=ref(reader, receipt_name)))
    return dict(scope=ref(reader, scope_name), collection=ref(reader, collection_name),
                status='collected_not_admitted', successful_targets=len(targets), targets=targets)


def same_json(left, right):
    # Python equality treats True == 1; frozen JSON fields must retain their types.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return canonical(left) == canonical(right)


def verify_combined_overlay(review, prior, ko):
    """Check the fixed overlay schema; semantic body judgments remain external."""
    require(review.get('schema') == WMT_OVERLAY_SCHEMA and
            review.get('status') == 'SOURCE_ONLY_CONDITIONAL_OVERLAY_NO_STRATEGY_EXPERIMENT',
            'Unexpected combined overlay schema/status')
    for key in ('source_admission_changed', 'old_admission_flags_changed', 'full_strategy_admission',
                'numerical_experiment_registered', 'lag_or_feature_values_computed'):
        require(review.get(key) is False, 'Combined overlay admission/activity: ' + key)
    require(review.get('new_conditional_context_admission') is True, 'Missing conditional-only declaration')
    rows, originals, ko_rows = review['rows'], prior['rows'], ko['rows']
    require(all(isinstance(items, list) and len(items) == 5978 for items in (rows, originals, ko_rows)),
            'Combined overlay must retain all 5978 rows')
    declared = review['verified_wmt_event_pairs']
    require(isinstance(declared, list), 'Invalid WMT pair list')
    pairs = set()
    for pair in declared:
        require(isinstance(pair, list) and len(pair) == 2 and
                all(isinstance(event, str) and event for event in pair) and len(set(pair)) == 2,
                'Invalid WMT event pair')
        key = frozenset(pair)
        require(key not in pairs, 'Duplicate WMT event pair')
        pairs.add(key)
    for position, (row, original, previous) in enumerate(zip(rows, originals, ko_rows)):
        i, j = divmod(position, 49)
        require(type(original['signal_index']) is int and type(original['candidate_index']) is int and
                (original['signal_index'], original['candidate_index']) == (i, j), 'Original grid axes changed')
        require(original['tier3'] is False and type(original['tier2']) is bool and
                type(original['is_etf']) is bool, 'Original admission flags changed')
        require(same_json(row['original_row'], original) and same_json(previous['original_row'], original),
                'Combined overlay original row changed')
        ids = original['selected_event_ids']
        wmt = len(ids) == len(set(ids)) == 2 and frozenset(ids) in pairs
        if wmt:
            require(original['symbol'] == 'WMT' and original['cik'] == '0000104169' and
                    original['is_etf'] is False, 'WMT pair joined to wrong issuer')
        require(type(previous['body_pair_context_verified']) is bool, 'Invalid prior KO context flag')
        ko_match = previous['body_pair_context_verified']
        if ko_match:
            require(original['symbol'] == 'KO' and original['is_etf'] is False, 'KO context on wrong issuer')
        matched = ko_match or wmt
        remaining = [reason for reason in original['reasons']
                     if not (matched and reason == MISSING_CONTEXT)]
        expected = dict(original_row=original, body_pair_context_verified=matched,
            body_context_source='WMT' if wmt else ('KO' if ko_match else None),
            resolved_context_reason=[MISSING_CONTEXT] if matched and MISSING_CONTEXT in original['reasons'] else [],
            remaining_row_reasons=remaining,
            conditional_annual_pair_admitted=bool(matched and original['tier2'] and not remaining),
            old_admission_flags_changed=False)
        require(same_json(row, expected), 'Combined overlay context join/blocker mismatch')
    summary = {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        part = rows[lo * 49:hi * 49]
        admitted = [row for row in part if row['conditional_annual_pair_admitted']]
        summary[name] = dict(all_slots=len(part),
            stock_slots=sum(not row['original_row']['is_etf'] for row in part),
            conditional_pair_slots=len(admitted),
            by_issuer={symbol: sum(row['body_context_source'] == symbol for row in admitted)
                       for symbol in ('KO', 'WMT')},
            original_tier3_slots=sum(row['original_row']['tier3'] for row in part))
    require(same_json(review['summary'], summary), 'Combined overlay summary does not match rows')
    return summary


def attach_wmt_review(reader, batches, path=None, expected=None):
    require(bool(path) == bool(expected), 'Both WMT review path and SHA required')
    if path is None:
        return None
    require((path.startswith('work/') and path.endswith('.json')) or path == WMT_OVERLAY,
            'WMT review must be a work JSON or the exact combined overlay path')
    review = reader.json(path, expected)
    require(path != WMT_OVERLAY or review.get('schema') == WMT_OVERLAY_SCHEMA,
            'Exact combined overlay path requires its fixed schema')
    require(review['source_admission_changed'] is False, 'WMT review changed old admission')
    require(review.get('old_admission_flags_changed', False) is False and
            review.get('full_strategy_admission', False) is False, 'WMT review claims broader admission')
    reader.pins(review['input_hashes'])
    required = [item for b in batches for item in [b['scope'], b['collection']]]
    required += [item for b in batches for t in b['targets'] for item in [t['raw'], t['receipt']]]
    require(all(review['input_hashes'].get(r['path']) == r['sha256'] for r in required),
            'WMT review does not bind both complete batches')
    combined = review.get('schema') == WMT_OVERLAY_SCHEMA
    summary = copy.deepcopy(review.get('summary'))
    if combined:
        require(review['input_hashes'].get(PRIOR) == PRIOR_SHA and
                review['input_hashes'].get(KO) == PINS[KO], 'Combined overlay prior/KO pins changed')
        summary = verify_combined_overlay(review, reader.json(PRIOR, PRIOR_SHA), reader.json(KO, PINS[KO]))
    return dict(reference=ref(reader, path), reported_status=review.get('status'),
                reported_conditional_summary=summary, combined_overlay_sanity_verified=combined,
                status='review_attached_no_automatic_admission')


def state_update(state, reference, next_step, timestamp):
    updated = copy.deepcopy(state)
    updated.update({REF_KEY: reference, 'next_step': next_step,
                    'last_goal_turn_classification': 'progress', 'updated_utc': timestamp})
    allowed = {REF_KEY, 'next_step', 'last_goal_turn_classification', 'updated_utc'}
    require({k: v for k, v in state.items() if k not in allowed} ==
            {k: v for k, v in updated.items() if k not in allowed}, 'Unrelated state modification')
    return updated


def commit_checkpoint(reader, state, before, checkpoint):
    path = reader.path(CHECKPOINT)[0]
    if path.exists():
        existing = reader.json(CHECKPOINT)
        content = lambda value: {k: v for k, v in value.items() if k != 'created_utc'}
        require(content(existing) == content(checkpoint), 'Conflicting source checkpoint')
    else:
        require(REF_KEY not in state, 'State points to missing source checkpoint')
        lifecycle.write(path, checkpoint)
    reference = ref(reader, CHECKPOINT)
    if REF_KEY in state:
        require(state[REF_KEY] == reference, 'State checkpoint binding mismatch')
        return dict(checkpoint=reference, already_recorded=True)
    updated = state_update(state, reference, checkpoint['next_step'], now())
    reader.reverify()
    lifecycle.update_state(reader.root / lifecycle.STATE, before, updated)
    return dict(checkpoint=reference, already_recorded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script-sha', required=True)
    parser.add_argument('--wmt-review')
    parser.add_argument('--wmt-review-sha')
    args = parser.parse_args()
    require(bool(args.wmt_review) == bool(args.wmt_review_sha), 'Both WMT review arguments required')
    reader = SourceReader(ROOT)
    reader.read(SELF, args.script_sha)
    reader.read(TEST)
    reader.pins(PINS)
    with lifecycle.state_lock(ROOT):
        state, before = lifecycle.state_snapshot(ROOT / lifecycle.STATE)
        validate_state(state)
        preserve_historical_state(ROOT)
        reader.read(ARCHIVE, HISTORICAL_SHA)
        ko, msft = verify_ko(reader), verify_msft_stop(reader)
        batches = [verify_wmt_batch(reader, name) for name in WMT_SCOPES]
        review = attach_wmt_review(reader, batches, args.wmt_review, args.wmt_review_sha)
        next_step = (
            '157 completed rounds / 986 trials; incumbent CAGR 67.66617372041686%, below70%. '
            'KO exact annual-body context is independently verified for109 conditional slots '
            '(81 first88,28 later34); original Tier3 remains0. WMT12 bodies are collected, '
            + ('with a separately pinned source review attached' if review else 'not yet semantically admitted') + '. '
            'The Microsoft filing iframe returned403; stop that host, with no retry or bypass. '
            'Continue bounded local/source evidence work, or preregister an independently motivated hypothesis '
            'supported by available evidence before any numerical experiment. Preserve source/t-42/identity/'
            'amendment gates and the full5978 denominator; document context is not strategy admission. '
            'No filing-lag values, features, ranks, orders or returns were computed in this step; '
            'no new trials,70% attainment claim, or frozen observer changes.'
        )
        checkpoint = dict(schema='after157_source_extension_checkpoint_v1', created_utc=now(),
            completed_rounds=157, strategy_trials=986, new_strategy_trials=0, target_reached=False,
            incumbent=copy.deepcopy(state['best_training_selected_candidate']),
            current_goal_turn_classification='progress',
            classification_basis='Independent KO full-grid context audit, bounded WMT source collection, and retained MSFT provider stop.',
            source_admission_changed=False, original_tier3_changed=False, full_strategy_admission=False,
            ko=ko, msft=msft, walmart=dict(status='collected_not_admitted' if review is None else 'review_attached_no_automatic_admission',
                successful_targets=sum(b['successful_targets'] for b in batches), batches=batches, review=review),
            input_hashes=dict(reader.hashes),
            historical_pin_aliases={lifecycle.STATE: dict(original_sha256=HISTORICAL_SHA,
                archive_path=ARCHIVE, archive_sha256=HISTORICAL_SHA)},
            next_step=next_step,
            limits=['Source-only conditional context; no historical price identity or exact publication-time certification.',
                    'Original rows and unresolved gates remain unchanged; WMT collection/review is not automatic row admission.',
                    'Both historical segments have been inspected and are not pristine out-of-sample evidence.'],
            activity=dict(network_requests=0, lag_values=0, feature_calculations=0, ranks=0, orders=0,
                          returns=0, strategy_registrations=0, frozen_files_modified=0))
        reader.reverify()
        result = commit_checkpoint(reader, state, before, checkpoint)
    print(json.dumps(dict(result, target_reached=False, new_strategy_trials=0), sort_keys=True))


if __name__ == '__main__':
    main()
