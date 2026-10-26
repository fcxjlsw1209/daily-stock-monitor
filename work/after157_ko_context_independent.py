"""Independent source-only KO overlay audit; no producer imports or numerical inputs.

Hashes all direct overlay inputs and the three semantic reviews/scopes. The older
769-input feasibility audit is inherited through its pinned report, not rerun.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_ko_context_independent.py'
OUT = 'work/after157_ko_context_independent.json'
ADMISSION = 'outputs/after157_ko_context_admission_v1/ADMISSION.json'
PRIOR = 'work/after157_annual_pair_source_feasibility.json'
CENTRAL = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
MISSING = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
REVIEWS = [
    ('work/after157_ko_body_review.json', 'work/after157_annual_body_scope.json', 2),
    ('work/after157_ko_extension_review.json', 'work/after157_ko_extension_scope.json', 8),
    ('work/after157_ko_early_review.json', 'work/after157_ko_early_bodies_scope.json', 2)]
PINS = {
    ADMISSION: '6bfab5c682a36deaa503b2027815142c428b70c84b3832f17920ebadd7ec3d9b',
    PRIOR: '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_ko_body_review.json': '0a09cc05afdab1c3c8222ca3538cb8b86fe06c67aaede074a20c9ee52a33e117',
    'work/after157_ko_extension_review.json': '9e7438300eba9d6fff22d596ec3979d94198af04eb29f10d627b27860ac9406f',
    'work/after157_ko_early_review.json': '2833e52be38d07d070c5468d244baa867e1b86454ca8b88ccb7b81e2e63d7435',
    'work/after157_ko_context_admission.py': '5c773f27fc57ea70099ff8a045a6bb570f61c3d20cfd411d091814b782065233'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def path(self, name):
        need(isinstance(name, str) and name and '\\' not in name and
             all(x not in ('', '.', '..') for x in name.split('/')), 'Unsafe path')
        p = ROOT
        for part in name.split('/'):
            p = p / part
            need(not p.is_symlink(), 'Symlink: ' + name)
        need(p.resolve().is_relative_to(ROOT), 'External path')
        return p

    def verify(self, name, expected):
        need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid SHA')
        need(name not in self.hashes or self.hashes[name] == expected, 'Conflicting shared pin: ' + name)
        need(digest(self.path(name)) == expected, 'SHA mismatch: ' + name)
        self.hashes[name] = expected

    def pins(self, entries):
        need(isinstance(entries, dict) and entries, 'Empty pin map')
        for name, expected in entries.items():
            self.verify(name, expected)

    def document(self, name):
        need(name in self.hashes, 'Unpinned JSON: ' + name)
        return json.loads(self.path(name).read_bytes())


def expected_side_columns(row, pairs):
    ids = row['selected_event_ids']
    pair = tuple(sorted(ids))
    match = len(ids) == 2 and ids[0] != ids[1] and pair in pairs
    if match:
        need(row['symbol'] == 'KO' and row['cik'] == '0000021344' and row['is_etf'] is False,
             'Pair matched wrong issuer/security type')
    remaining = []
    for reason in row['reasons']:
        if reason != MISSING or not match:
            remaining.append(reason)
    return dict(body_pair_context_verified=match,
                conditional_annual_pair_admitted=match and row['tier2'] is True and len(remaining) == 0,
                old_admission_flags_changed=False, remaining_row_reasons=remaining,
                resolved_context_reason=[MISSING] if match and MISSING in row['reasons'] else [])


def compare_rows(originals, overlay, pairs, full_grid=True):
    need(len(originals) == len(overlay), 'Row count changed')
    if full_grid:
        need(len(originals) == 5978, 'Not complete 122x49 grid')
        keys = [(r['signal_index'], r['candidate_index']) for r in originals]
        need(keys == [(i, j) for i in range(122) for j in range(49)], 'Original grid axes changed')
        need(sum(r['is_etf'] is False for r in originals) == 3904, 'Stock denominator changed')
        need(sum(r['is_etf'] is True for r in originals) == 2074, 'ETF denominator changed')
    for index, (old, new) in enumerate(zip(originals, overlay)):
        need(canonical(new['original_row']) == canonical(old), 'Original row changed: ' + str(index))
        need(old['tier3'] is False, 'Original Tier3 changed')
        want = expected_side_columns(old, pairs)
        side = {k: v for k, v in new.items() if k != 'original_row'}
        need(canonical(side) == canonical(want), 'Side-column/original-blocker mismatch: ' + str(index))


def self_tests():
    row = dict(symbol='KO', cik='0000021344', is_etf=False, tier2=True, tier3=False,
               selected_event_ids=['old', 'new'], reasons=[MISSING],
               existing_accession_context=[{'prior141_blockers': ['historical_security_identity_not_verified']}])
    pairs = {('new', 'old')}
    side = expected_side_columns(row, pairs)
    need(side['conditional_annual_pair_admitted'] is True, 'Positive fixture')
    cases = [
        (dict(row, tier2=False), False),
        (dict(row, reasons=[MISSING, 'KNOWN_AMENDMENT']), False),
        (dict(row, selected_event_ids=['old']), False),
        (dict(row, selected_event_ids=['old', 'old']), False),
        (dict(row, selected_event_ids=['old', 'unknown']), False),
        (dict(row, selected_event_ids=['new', 'old']), True)]
    for test, accepted in cases:
        need(expected_side_columns(test, pairs)['conditional_annual_pair_admitted'] is accepted, 'Fixture join')
    original = copy.deepcopy(row)
    overlay = dict(original_row=copy.deepcopy(row), **side)
    compare_rows([row], [overlay], pairs, False)
    need(row == original, 'Test mutated input')
    rejected = 0
    for field, value in [('original_row', dict(row, tier3=True)),
                         ('original_row', dict(row, existing_accession_context=[])),
                         ('conditional_annual_pair_admitted', 1),
                         ('remaining_row_reasons', ['wrong']), ('body_pair_context_verified', False)]:
        changed = copy.deepcopy(overlay)
        changed[field] = value
        try:
            compare_rows([row], [changed], pairs, False)
        except ValueError:
            rejected += 1
    need(rejected == 5, 'Tamper test failed')
    for patch in [dict(symbol='MSFT'), dict(cik='0000789019'), dict(is_etf=True)]:
        try:
            expected_side_columns(dict(row, **patch), pairs)
        except ValueError:
            rejected += 1
    need(rejected == 8, 'Wrong issuer/ETF test failed')
    return {'passed': True, 'synthetic_cases': 16,
            'coverage': ['exact and reversed pairs', 'missing/duplicate/unknown events', 'failed Tier2',
                         'other blockers', 'deep original preservation', 'boolean type tamper',
                         'wrong issuer/ETF', 'nested historical141 preservation']}


def audit(script_sha):
    evidence = Evidence()
    evidence.pins(PINS)
    evidence.verify(SELF, script_sha)
    admission = evidence.document(ADMISSION)
    need(len(admission['input_hashes']) == 135, 'Overlay input scope changed')
    evidence.pins(admission['input_hashes'])
    prior = evidence.document(PRIOR)
    metadata = prior['selected_original_and_amendment_metadata']
    originals, pairs, review_refs = {}, {}, []
    for review_path, scope_path, count in REVIEWS:
        review = evidence.document(review_path)
        for flag in ('exact_original_versions_bound', 'normal_consecutive_pair_context_verified'):
            need(review[flag] is True, 'Unresolved semantic review')
        need(review['source_admission_changed'] is False, 'Semantic review changed old admission')
        evidence.pins(review['input_hashes'])
        scope = evidence.document(scope_path)
        evidence.pins(scope['input_hashes'])
        out = scope['output_dir']
        collection = evidence.document(out + '/COLLECTION.json')
        derived = evidence.document(out + '/derived_text/MANIFEST.json')
        need(collection['all_successful'] is True and collection['stopped_providers'] == [], 'Incomplete collection')
        need(collection['scope_sha256'] == evidence.hashes[scope_path], 'Collection scope mismatch')
        need(derived['scope_sha256'] == evidence.hashes[scope_path] and
             derived['collection_sha256'] == evidence.hashes[out + '/COLLECTION.json'], 'Derived chain mismatch')
        need(derived['admission'] is False and derived['factor_calculations'] == 0, 'Unexpected derived activity')
        need(len(review['originals']) == count, 'Original review count changed')
        for original in review['originals']:
            event = original['event_id']
            need(event not in originals, 'Duplicate reviewed original')
            meta = metadata[event]
            need(original['symbol'] == 'KO' and original['cik'] == meta['cik'] == '0000021344' and
                 original['form'] == meta['form'] == '10-K', 'Not an ordinary KO original')
            for left, right in [('accession', 'accessionNumber'), ('primary_document', 'primaryDocument'),
                                ('report_date', 'reportDate'), ('filing_date_from_frozen_metadata', 'filingDate')]:
                need(original[left] == meta[right], 'Original metadata mismatch: ' + left)
            if count != 2 or review_path != REVIEWS[0][0]:
                need(original['exact_original_version_bound'] is True and
                     original['normal_annual_context_verified'] is True, 'Unresolved individual original')
            targets = [t for t in scope['targets'] if t.get('expected_original', {}).get('event_id') == event]
            need(len(targets) == 1, 'No unique scope original')
            target = targets[0]
            need(all(meta[k] == v for k, v in target['expected_original'].items()), 'Scope event mismatch')
            need(original['raw_path'] == out + '/' + target['body_file'] and
                 original['receipt_path'] == out + '/' + target['receipt_file'], 'Wrong source path')
            for kind in ('raw', 'receipt', 'attempt', 'derived'):
                evidence.verify(original[kind + '_path'], original[kind + '_sha256'])
            receipt = evidence.document(original['receipt_path'])
            attempt = evidence.document(original['attempt_path'])
            url = 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/' + original['accession'] + '/' + original['primary_document']
            need(target['url'] == receipt['source_url'] == receipt['final_url'] == url, 'Exact issuer URL changed')
            need(receipt['http_status'] == 200 and receipt['outcome'] == 'success' and
                 receipt['stop_provider'] is False and receipt['body_truncated_at_limit'] is False and
                 receipt['redirect_chain'] == [], 'Unsuccessful exact body')
            need(receipt['sha256'] == original['raw_sha256'] and
                 receipt['bytes'] == original['raw_bytes'] == evidence.path(original['raw_path']).stat().st_size and
                 receipt['attempt_sha256'] == original['attempt_sha256'], 'Raw/receipt/attempt mismatch')
            for field in ('id', 'provider', 'source_url', 'body_file', 'scope_sha256', 'helper_sha256'):
                need(attempt[field] == receipt[field], 'Attempt binding mismatch')
            need(receipt['scope_sha256'] == evidence.hashes[scope_path] and receipt['id'] == target['id'] and
                 receipt['helper_sha256'] == scope['helper_sha256'], 'Receipt scope mismatch')
            extracted = [r for r in derived['records'] if r['id'] == target['id']]
            need(len(extracted) == 1, 'Derived event mismatch')
            extracted = extracted[0]
            need(extracted['raw_sha256'] == original['raw_sha256'] and
                 extracted['receipt_sha256'] == original['receipt_sha256'] and
                 extracted['text_sha256'] == original['derived_sha256'] and
                 out + '/derived_text/' + extracted['text_file'] == original['derived_path'], 'Derived source mismatch')
            originals[event] = dict(event_id=event, report_date=original['report_date'],
                                    accession=original['accession'], review=review_path,
                                    raw_path=original['raw_path'], raw_sha256=original['raw_sha256'])
        if review_path == REVIEWS[0][0]:
            entries = [dict(event_ids=review['expected_event_ids'], exact_original_versions_bound=True,
                            normal_consecutive_pair_context_verified=True)]
        else:
            entries = review['pairs']
        for entry in entries:
            ids = entry['event_ids']
            need(len(ids) == 2 and len(set(ids)) == 2 and all(x in metadata for x in ids), 'Bad reviewed pair')
            need(entry['exact_original_versions_bound'] is True and
                 entry['normal_consecutive_pair_context_verified'] is True, 'Unresolved pair')
            pair = tuple(sorted(ids))
            need(pair not in pairs, 'Duplicate semantic pair')
            pairs[pair] = review_path
        review_refs.append(dict(path=review_path, sha256=evidence.hashes[review_path], originals=count,
                                pair_count=len(entries)))
    need(len(originals) == 12 and len(pairs) == 11, 'Wrong reviewed event scope')
    chronology = sorted(originals, key=lambda e: originals[e]['report_date'])
    need([originals[e]['report_date'] for e in chronology] == [str(y) + '-12-31' for y in range(2014, 2026)],
         'Unexpected reviewed periods')
    adjacent = {tuple(sorted([a, b])) for a, b in zip(chronology, chronology[1:])}
    need(set(pairs) == adjacent, 'Reviewed pairs do not cover exactly adjacent original periods')
    need(sorted(map(list, pairs)) == admission['verified_event_pairs'], 'Published pair set changed')
    compare_rows(prior['rows'], admission['rows'], set(pairs))
    computed, extra = {}, {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        rows = [r for r in prior['rows'] if lo <= r['signal_index'] < hi]
        sides = [expected_side_columns(r, set(pairs)) for r in rows]
        accepted = [r for r, s in zip(rows, sides) if s['conditional_annual_pair_admitted']]
        computed[name] = dict(all_slots=len(rows), stock_slots=sum(not r['is_etf'] for r in rows),
            conditional_pair_slots=len(accepted), issuers=sorted({r['symbol'] for r in accepted}),
            original_tier3_slots=sum(r['tier3'] for r in rows))
        extra[name] = dict(etf_slots=sum(r['is_etf'] for r in rows),
            matched_document_pair_slots=sum(s['body_pair_context_verified'] for s in sides),
            matched_but_ineligible_slots=sum(s['body_pair_context_verified'] and not s['conditional_annual_pair_admitted'] for s in sides))
    need(canonical(computed) == canonical(admission['summary']), 'Summary mismatch')
    for flag in ('old_admission_flags_changed', 'full_strategy_admission', 'numerical_experiment_registered',
                 'lag_or_feature_values_computed'):
        need(admission[flag] is False, 'Overlay admission/activity flag changed')
    need(admission['new_conditional_context_admission'] is True, 'Conditional scope missing')
    checks = {name: True for name in [
        'all_135_overlay_input_hashes_verified', 'review_and_scope_input_hashes_verified',
        'shared_source_hash_consistency_verified', 'all_12_exact_original_metadata_bindings_verified',
        'all_12_raw_receipt_attempt_and_derived_chains_verified', 'all_11_semantic_review_pairs_bound',
        'full_122_by_49_grid_preserved', 'stock_3904_etf_2074_preserved',
        'all_original_rows_byte_equivalent_after_canonical_json', 'independent_exact_pair_join_verified',
        'only_document_context_reason_resolved_in_side_column', 'other_blockers_and_tier2_gate_preserved',
        'original_tier3_zero_preserved', 'nested_prior141_context_and_identity_preserved',
        'source_selection_cutoffs_amendments_and_no_fallback_rows_preserved',
        'all_three_segment_counts_independently_recomputed', 'no_non_ko_conditional_admission',
        'no_producer_imports', 'inputs_unchanged_after_audit']}
    evidence.pins(dict(evidence.hashes))
    central_sha = admission['input_hashes'][CENTRAL]
    return dict(passed=True, schema='after157_ko_context_independent_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_admission_changed=False,
        old_admission_flags_changed=False, checks=checks, summary=computed, diagnostics=extra,
        input_hashes=evidence.hashes, total_verified_unique_input_files=len(evidence.hashes),
        reviewed_originals=sorted(originals.values(), key=lambda o: o['report_date']),
        semantic_reviews=review_refs,
        verified_event_pairs=[dict(event_ids=list(k), semantic_review=v) for k, v in sorted(pairs.items())],
        mutable_input_pins=[dict(path=CENTRAL, sha256=central_sha, verified_at_original_path=True,
            future_archive_path='outputs/monthly_target70_research/PRE157_' + central_sha + '.json',
            archive_verified_or_created_by_this_audit=False,
            rule='Future central updates require preserving these exact bytes at the stated archive path. This receipt does not pin the future live central state.')],
        synthetic_validation=self_tests(),
        limits=['The 769-member original feasibility input audit is inherited through the exact report SHA, not repeated.',
                'This audit independently verifies joins, source chains and preservation; body semantics remain attributed to the three separately pinned reviews.',
                'Early and FY2016/FY2017 full-year starts retain their reviewed derivation from explicit recurring fiscal-year convention, not newly printed start-date claims.',
                'Conditional issuer/price-route and administrative filing-date reconstruction assumptions remain; no exact public timestamp or historical security certification.',
                'This audit preserves a source-only conditional overlay and does not upgrade original Tier3, certify numerical features, register a strategy, or establish return improvement.'],
        activity=dict(network_requests=0, price_values_read=0, actual_lag_values=0, factors=0,
                      ranks=0, orders=0, returns=0, new_strategy_trials=0, frozen_files_modified=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--script-sha')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.test_only:
        need(not args.write, 'Test-only cannot write')
        print(json.dumps(self_tests(), sort_keys=True))
        return
    need(args.script_sha is not None, 'Pinned script SHA required')
    result = audit(args.script_sha)
    if args.write:
        target = Evidence().path(OUT)
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(dict(passed=result['passed'], summary=result['summary'], diagnostics=result['diagnostics'],
                         verified_files=result['total_verified_unique_input_files'], output=OUT if args.write else None),
                     sort_keys=True))


if __name__ == '__main__':
    main()
