"""Independent metadata-only WMT/KO overlay audit; standard-library imports only.

Reconstructs all row joins independently from frozen events and semantic reviews.
Does not run the overlay producer, read prices, or compute filing lags or returns.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_walmart_context_independent.py'
OUT = 'work/after157_walmart_context_independent.json'
ADMISSION = 'outputs/after157_walmart_context_admission_v1/ADMISSION.json'
PRIOR = 'work/after157_annual_pair_source_feasibility.json'
KO = 'outputs/after157_ko_context_admission_v1/ADMISSION.json'
KO_REVIEW = 'work/after157_ko_context_independent.json'
ROOT_REVIEW = 'work/after157_walmart_2024_root_review.json'
CENTRAL = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
MISSING = 'ACCESSION_SPECIFIC_NORMAL_CONSECUTIVE_ANNUAL_CONTEXT_NOT_ESTABLISHED'
PINS = {
    ADMISSION: '9e90889a1fb4f057438b6ac16a9e234f97cc9ede08c6cb28a6fe190389f3f9d6',
    PRIOR: '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    KO: '6bfab5c682a36deaa503b2027815142c428b70c84b3832f17920ebadd7ec3d9b',
    KO_REVIEW: '4aab7d30caba3b5d579d8a6b979a65c5e0b650f0841adc2114290925a09108d5',
    ROOT_REVIEW: '96100010d393b08bffcdc33298c9e3428f4ead6ef390f98bc85fea49a2d4c264',
    'work/after157_walmart_early_review.json': '3b95d80804f6d71ddb66215972a2c5406fdc9b5e79617c56ee02d1982ae90665',
    'work/after157_walmart_late_review.json': 'f320a6992dbfca98eb2bad79bcc6c736a33f8b0a7de381fd2c428b8ab247248f',
    'work/after157_walmart_context_admission.py': '67ab799dd20e777b80aa9bfd0f9e66185f44d18cbb3da725cd7fca3067c00e10',
    'work/test_after157_walmart_pairs_independent.py': '7f3140e3753a74594990a0c825d67367e3289ef55216843340b47a2eaef2c875',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class Evidence:
    def __init__(self):
        self.hashes, self.resolutions = {}, {}

    def path(self, name):
        need(isinstance(name, str) and name and '\\' not in name and
             all(p not in ('', '.', '..') for p in name.split('/')), 'Unsafe path')
        p = ROOT
        for part in name.split('/'):
            p = p / part
            need(not p.is_symlink(), 'Symlink: ' + name)
        need(p.resolve().is_relative_to(ROOT), 'External path')
        return p

    def verify(self, name, expected):
        need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Bad SHA')
        resolved = name
        if name == CENTRAL and sha(self.path(name)) != expected:
            resolved = 'outputs/monthly_target70_research/PRE157_' + expected + '.json'
        need(resolved not in self.hashes or self.hashes[resolved] == expected, 'Conflicting pin: ' + name)
        need(sha(self.path(resolved)) == expected, 'Hash mismatch: ' + name)
        self.hashes[resolved] = expected
        if resolved != name:
            self.resolutions[name + '@' + expected] = resolved

    def pins(self, entries):
        need(isinstance(entries, dict) and entries, 'Empty pin map')
        for name, expected in entries.items():
            self.verify(name, expected)

    def document(self, name):
        need(name in self.hashes, 'Unpinned document: ' + name)
        return json.loads(self.path(name).read_bytes())

    def lines(self, name):
        need(name in self.hashes, 'Unpinned text: ' + name)
        return self.path(name).read_text().splitlines()


def review_pairs(reviews, targets, metadata):
    """Build a chronological event graph, then compare all declared review edges."""
    expected = {}
    for target in targets:
        item = target['expected_original']
        event = item['event_id']
        need(event not in expected, 'Duplicate target')
        need(all(metadata[event].get(k) == v for k, v in item.items()), 'Target metadata drift')
        need(item['cik'] == '0000104169' and item['form'] == '10-K', 'Wrong target identity')
        expected[event] = item
    ordered = sorted(expected, key=lambda e: expected[e]['reportDate'])
    need([expected[e]['reportDate'] for e in ordered] ==
         [str(y) + '-01-31' for y in range(2015, 2027)], 'Unexpected fiscal chronology')
    reviewed, positive, declared, accepted = {}, set(), {}, set()
    for review in reviews:
        need(review['source_admission_changed'] is False, 'Review changes old admission')
        for original in review['originals']:
            event = original['event_id']
            need(event in expected and event not in reviewed, 'Missing/duplicate original identity')
            item = expected[event]
            need(original['symbol'] == 'WMT' and original['cik'] == item['cik'] and
                 original['form'] == item['form'] and type(original['fiscal_year']) is int and
                 original['report_date'] == str(original['fiscal_year']) + '-01-31', 'Original identity mismatch')
            for a, b in [('accession', 'accessionNumber'), ('primary_document', 'primaryDocument'),
                         ('report_date', 'reportDate'), ('filing_date_from_frozen_metadata', 'filingDate')]:
                need(original[a] == item[b], 'Original metadata mismatch: ' + a)
            flags = [original['exact_original_version_bound'], original['normal_annual_context_verified']]
            need(all(type(v) is bool for v in flags), 'Nonboolean original judgement')
            if all(flags):
                positive.add(event)
            reviewed[event] = original
        for pair in review['pairs']:
            ids = pair['event_ids']
            need(len(ids) == 2 and len(set(ids)) == 2, 'Invalid pair endpoints')
            key = tuple(sorted(ids))
            need(key not in declared, 'Duplicate pair')
            flags = [pair['normal_consecutive_pair_context_verified'], pair['exact_original_versions_bound']]
            need(all(type(v) is bool for v in flags), 'Nonboolean pair judgement')
            declared[key] = all(flags)
    need(set(reviewed) == set(expected), 'Incomplete original review')
    adjacent = {tuple(sorted(ordered[i:i+2])) for i in range(len(ordered) - 1)}
    need(set(declared) == adjacent, 'Incomplete or nonconsecutive pair graph')
    for key, yes in declared.items():
        if yes:
            need(set(key) <= positive, 'Positive pair includes negative original')
            accepted.add(key)
    return accepted, reviewed


def side_columns(row, pairs, with_source=True):
    ids = row['selected_event_ids']
    source = pairs.get(tuple(sorted(ids))) if len(ids) == 2 and len(set(ids)) == 2 else None
    if source:
        need(row['is_etf'] is False and row['symbol'] == source and
             row['cik'] == {'WMT': '0000104169', 'KO': '0000021344'}[source], 'Pair routed to wrong issuer')
    remaining = [reason for reason in row['reasons'] if not (source and reason == MISSING)]
    out = dict(body_pair_context_verified=source is not None,
               resolved_context_reason=[MISSING] if source and MISSING in row['reasons'] else [],
               remaining_row_reasons=remaining,
               conditional_annual_pair_admitted=bool(source and row['tier2'] is True and not remaining),
               old_admission_flags_changed=False)
    if with_source:
        out['body_context_source'] = source
    return out


def compare_rows(originals, overlay, pairs, with_source=True, full=True):
    need(len(originals) == len(overlay), 'Row count changed')
    if full:
        need([(r['signal_index'], r['candidate_index']) for r in originals] ==
             [(i, j) for i in range(122) for j in range(49)], 'Incomplete original grid')
        need(sum(r['is_etf'] is False for r in originals) == 3904 and
             sum(r['is_etf'] is True for r in originals) == 2074, 'Changed stock/ETF denominator')
    expected = []
    for i, (old, new) in enumerate(zip(originals, overlay)):
        need(old['tier3'] is False and old['normal_period_admission'] is False, 'Old admission upgraded')
        want = dict(original_row=old, **side_columns(old, pairs, with_source))
        need(canonical(want) == canonical(new), 'Exact row mismatch at ' + str(i))
        expected.append(want)
    return expected


def self_tests():
    row = dict(symbol='WMT', cik='0000104169', is_etf=False, tier2=True, tier3=False,
               normal_period_admission=False, selected_event_ids=['a', 'b'], reasons=[MISSING],
               existing_accession_context=[{'prior141_blockers': ['historical_security_identity_not_verified']}])
    pairs = {('a', 'b'): 'WMT'}
    cases = [(row, True), (dict(row, selected_event_ids=['b', 'a']), True),
             (dict(row, selected_event_ids=['a', 'a']), False),
             (dict(row, selected_event_ids=['a']), False),
             (dict(row, selected_event_ids=['a', 'c']), False),
             (dict(row, tier2=False), False),
             (dict(row, reasons=[MISSING, 'LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42']), False),
             (dict(row, reasons=[MISSING, 'KNOWN_AMENDMENT']), False)]
    for r, yes in cases:
        need(side_columns(r, pairs)['conditional_annual_pair_admitted'] is yes, 'Synthetic gate failure')
    ko = dict(row, symbol='KO', cik='0000021344')
    need(side_columns(ko, {('a', 'b'): 'KO'})['body_context_source'] == 'KO', 'KO inheritance fixture')
    saved = copy.deepcopy(row)
    output = dict(original_row=copy.deepcopy(row), **side_columns(row, pairs))
    compare_rows([row], [output], pairs, full=False)
    need(row == saved, 'Oracle mutated original')
    mutations = [('original_row', dict(row, tier3=True)),
                 ('original_row', dict(row, existing_accession_context=[])),
                 ('original_row', dict(row, selected_event_ids=['b', 'a'])),
                 ('conditional_annual_pair_admitted', 1), ('remaining_row_reasons', ['x']),
                 ('body_context_source', 'KO'), ('old_admission_flags_changed', True)]
    rejected = 0
    for field, value in mutations:
        mutated = copy.deepcopy(output)
        mutated[field] = value
        try:
            compare_rows([row], [mutated], pairs, full=False)
        except ValueError:
            rejected += 1
    need(rejected == len(mutations), 'Tamper accepted')
    for patch in [dict(symbol='KO'), dict(cik='0000021344'), dict(is_etf=True)]:
        try:
            side_columns(dict(row, **patch), pairs)
        except ValueError:
            rejected += 1
    need(rejected == len(mutations) + 3, 'Wrong issuer accepted')
    return dict(passed=True, synthetic_cases=len(cases) + 2 + len(mutations) + 3,
                checks=['exact/reversed/duplicate/missing event identities', 'KO inheritance',
                        'Tier2/timing/amendment gates', 'deep original and boolean preservation',
                        'wrong issuer/CIK/ETF rejected', 'no original mutation'])


def verify_sources(evidence, targets, originals, scopes):
    for batch, scope, review in scopes:
        base = scope['output_dir']
        scope_name = 'work/after157_walmart_bodies_' + batch + '_scope.json'
        manifest = evidence.document(base + '/derived_text/MANIFEST.json')
        collection = evidence.document(base + '/COLLECTION.json')
        marker = evidence.document(base + '/.collection-scope.json')
        need(collection['all_successful'] is True and collection['stopped_providers'] == [], 'Incomplete collection')
        for doc in [collection, manifest, marker]:
            need(doc['scope_sha256'] == evidence.hashes[scope_name], 'Wrong source scope')
        need(manifest['collection_sha256'] == evidence.hashes[base + '/COLLECTION.json'], 'Wrong collection')
        need(manifest['admission'] is False and manifest['source_admission_changed'] is False and
             manifest['factor_calculations'] == manifest['filing_lag_calculations'] == 0 and
             type(manifest['prices_or_returns_read']) is int and
             manifest['prices_or_returns_read'] == 0, 'Unexpected extraction activity')
        need(len(collection['targets']) == len(manifest['records']) == len(scope['targets']) == 6, 'Source count')
        need({o['event_id'] for o in review['originals']} ==
             {t['expected_original']['event_id'] for t in scope['targets']}, 'Wrong review scope')
        need(set(review['reviewed_target_ids']) == {t['id'] for t in scope['targets']} and
             set(review['expected_event_ids']) == {t['expected_original']['event_id'] for t in scope['targets']},
             'Declared review event/target scope changed')
        for target in scope['targets']:
            item = originals[target['expected_original']['event_id']]
            for kind in ('raw', 'receipt', 'attempt', 'derived'):
                evidence.verify(item[kind + '_path'], item[kind + '_sha256'])
            need(item['raw_path'] == base + '/' + target['body_file'] and
                 item['receipt_path'] == base + '/' + target['receipt_file'], 'Source path mismatch')
            receipt = evidence.document(item['receipt_path'])
            attempt = evidence.document(item['attempt_path'])
            summaries = [r for r in collection['targets'] if r['id'] == target['id']]
            need(len(summaries) == 1 and summaries[0]['outcome'] == 'success' and
                 summaries[0]['receipt_file'] == target['receipt_file'] and
                 summaries[0]['receipt_sha256'] == item['receipt_sha256'], 'Collection target mismatch')
            url = 'https://stock.walmart.com/sec-filings/all-sec-filings/content/' + item['accession'] + '/' + item['primary_document']
            need(target['url'] == receipt['source_url'] == receipt['final_url'] == url, 'Exact body URL mismatch')
            need(receipt['http_status'] == 200 and receipt['outcome'] == 'success' and
                 receipt['stop_provider'] is False and receipt['body_truncated_at_limit'] is False and
                 receipt['redirect_chain'] == [], 'Unsuccessful original response')
            need(receipt['sha256'] == item['raw_sha256'] and
                 receipt['bytes'] == item['raw_bytes'] == evidence.path(item['raw_path']).stat().st_size and
                 receipt['attempt_sha256'] == item['attempt_sha256'], 'Receipt binding mismatch')
            need(receipt['scope_sha256'] == evidence.hashes[scope_name] and receipt['id'] == target['id'] and
                 receipt['helper_sha256'] == scope['helper_sha256'], 'Receipt scope/helper mismatch')
            for field in ('id', 'provider', 'source_url', 'body_file', 'scope_sha256', 'helper_sha256'):
                need(attempt[field] == receipt[field], 'Attempt mismatch: ' + field)
            records = [r for r in manifest['records'] if r['id'] == target['id']]
            need(len(records) == 1, 'Ambiguous derived record')
            record = records[0]
            need(record['expected_original'] == target['expected_original'] and
                 record['raw_sha256'] == item['raw_sha256'] and record['receipt_sha256'] == item['receipt_sha256'] and
                 record['attempt_sha256'] == item['attempt_sha256'] and record['text_sha256'] == item['derived_sha256'] and
                 base + '/derived_text/' + record['text_file'] == item['derived_path'], 'Derived chain mismatch')
            need(record['lines'] == len(evidence.lines(item['derived_path'])), 'Derived line count mismatch')
            pub = target['publisher_evidence']
            lines = evidence.lines(pub['path'])
            need(pub['raw_link_excerpt'] in lines[pub['raw_link_line'] - 1] and
                 pub['raw_label_excerpt'] in lines[pub['raw_label_line'] - 1] and
                 url in pub['raw_link_excerpt'], 'Publisher exact link/label mismatch')
            location = target['original_metadata_location']
            source = evidence.document(location['source'])
            if location['section'] != 'historical_page':
                need(location['section'] == 'filings.recent', 'Unknown original source section')
                source = source['filings']['recent']
            for key in ('accessionNumber', 'form', 'filingDate', 'reportDate', 'primaryDocument'):
                need(source[key][location['row']] == target['expected_original'][key], 'Original metadata row mismatch')


def audit(script_sha):
    e = Evidence()
    e.pins(PINS)
    e.verify(SELF, script_sha)
    for path in ('run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        e.verify(path, sha(e.path(path)))
    admission = e.document(ADMISSION)
    need(len(admission['input_hashes']) == 218, 'Unexpected overlay input count')
    e.pins(admission['input_hashes'])
    prior, ko, ko_review = e.document(PRIOR), e.document(KO), e.document(KO_REVIEW)
    need(ko_review['passed'] is True and all(v is True for v in ko_review['checks'].values()), 'Failed KO review')
    e.pins(ko['input_hashes'])
    e.pins(ko_review['input_hashes'])
    targets, reviews, scopes = [], [], []
    for batch in ('early', 'late'):
        scope_name = 'work/after157_walmart_bodies_' + batch + '_scope.json'
        review_name = 'work/after157_walmart_' + batch + '_review.json'
        scope, review = e.document(scope_name), e.document(review_name)
        for doc in (scope, review, e.document(scope['output_dir'] + '/derived_text/MANIFEST.json')):
            e.pins(doc['input_hashes'])
        targets.extend(scope['targets'])
        reviews.append(review)
        scopes.append((batch, scope, review))
    pairs, originals = review_pairs(reviews, targets, prior['selected_original_and_amendment_metadata'])
    need(sorted(map(list, pairs)) == admission['verified_wmt_event_pairs'], 'Published WMT pair set mismatch')
    verify_sources(e, targets, originals, scopes)
    # KO semantics were reviewed separately; independently replay their exact pairs on every row.
    ko_pairs = {tuple(sorted(x['event_ids'])): 'KO' for x in ko_review['verified_event_pairs']}
    need(len(ko_pairs) == 11 and sorted(map(list, ko_pairs)) == ko['verified_event_pairs'], 'KO pair set mismatch')
    compare_rows(prior['rows'], ko['rows'], ko_pairs, with_source=False)
    combined = dict(ko_pairs)
    need(not (set(combined) & pairs), 'Cross-issuer pair collision')
    combined.update({pair: 'WMT' for pair in pairs})
    expected = compare_rows(prior['rows'], admission['rows'], combined)
    summary, diagnostics = {}, {}
    for label, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        rows = [r for r in expected if lo <= r['original_row']['signal_index'] < hi]
        admitted = Counter(r['body_context_source'] for r in rows if r['conditional_annual_pair_admitted'])
        summary[label] = dict(all_slots=len(rows), stock_slots=sum(r['original_row']['is_etf'] is False for r in rows),
                              conditional_pair_slots=sum(admitted.values()),
                              by_issuer={s: admitted[s] for s in ('KO', 'WMT')},
                              original_tier3_slots=sum(r['original_row']['tier3'] for r in rows))
        diagnostics[label] = dict(etf_slots=sum(r['original_row']['is_etf'] is True for r in rows),
                                 matched_context_slots=sum(r['body_pair_context_verified'] for r in rows),
                                 matched_but_blocked_slots=sum(r['body_pair_context_verified'] and
                                                               not r['conditional_annual_pair_admitted'] for r in rows))
    need(canonical(summary) == canonical(admission['summary']), 'Published summary mismatch')
    blocked = [r for r in expected if r['original_row']['symbol'] == 'WMT' and
               r['original_row']['signal_date'] == '2018-05-31']
    need(len(blocked) == 1 and blocked[0]['original_row']['tier2'] is False and
         blocked[0]['body_pair_context_verified'] is True and blocked[0]['conditional_annual_pair_admitted'] is False and
         blocked[0]['remaining_row_reasons'] == ['LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42'], 'WMT timing blocker lost')
    root_review = e.document(ROOT_REVIEW)
    e.pins(root_review['input_hashes'])
    need(root_review['passed'] is True and root_review['source_admission_changed'] is False and
         len(root_review['checks']) == 13 and all(x is True for x in root_review['checks'].values()), 'FY2024 check failed')
    event2024 = [o for o in originals.values() if o['fiscal_year'] == 2024][0]
    raw = e.lines(event2024['raw_path'])
    primary = 'https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-24-000056/wmt-20240131.htm'
    exhibit = 'https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-24-000056/wmtexhibit21fy24.htm'
    need(primary + '/wmtexhibit21fy24.htm' in raw[10] and primary + '/wmtexhibit21fy24.htm' in raw[11] and
         primary in raw[100] and '10-K' in raw[101] and exhibit in raw[120] and 'EX-21' in raw[121],
         'FY2024 anomaly/corrected navigation evidence mismatch')
    for flag in ('source_admission_changed', 'old_admission_flags_changed', 'full_strategy_admission',
                 'numerical_experiment_registered', 'lag_or_feature_values_computed'):
        need(admission[flag] is False, 'Overstated admission/activity flag')
    need(admission['new_conditional_context_admission'] is True and
         admission['schema'] == 'after157_ko_wmt_conditional_annual_source_v1', 'Wrong overlay scope')
    e.pins(dict(e.hashes))
    central_sha = admission['input_hashes'][CENTRAL]
    return dict(schema='after157_walmart_context_independent_v1', passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_admission_changed=False,
        old_admission_flags_changed=False, full_strategy_admission=False,
        input_hashes=e.hashes, total_verified_unique_input_files=len(e.hashes),
        input_path_resolutions=e.resolutions, overlay_input_files_verified=218,
        checks={k: True for k in ['all_overlay_input_hashes_verified', 'all_twelve_original_chains_verified',
            'reviewed_originals_match_frozen_metadata_and_raw_source_rows', 'exact_publisher_links_verified',
            'complete_eleven_wmt_pairs_verified', 'ko_exact_pair_replay_and_inheritance_verified',
            'all_5978_exact_rows_independently_reconstructed', 'all_original_fields_and_flags_preserved',
            'stock3904_etf2074_axes_preserved', 'only_specific_context_blocker_resolved',
            'other_timing_identity_amendment_selection_gates_preserved', 'wmt_may2018_still_blocked',
            'segment_summaries_independently_recomputed', 'fy2024_discrepancy_retained_and_citations_corrected',
            'original_tier3_zero_preserved', 'no_producer_imports', 'input_hashes_unchanged_after_audit']},
        summary=summary, diagnostics=diagnostics, blocked_wmt_row=blocked[0],
        verified_wmt_event_pairs=sorted(map(list, pairs)), inherited_ko_event_pairs=sorted(map(list, ko_pairs)),
        verified_original_sources=[dict(event_id=o['event_id'], accession=o['accession'], report_date=o['report_date'],
            raw_path=o['raw_path'], raw_sha256=o['raw_sha256']) for o in sorted(originals.values(), key=lambda x: x['report_date'])],
        fy2024_citation_correction=dict(prior_review_unchanged='work/after157_walmart_late_review.json',
            confirming_review=ROOT_REVIEW, primary_navigation_lines=[101, 102], separate_exhibit_navigation_lines=[121, 122],
            canonical_and_og_anomaly_lines=[11, 12], canonical_anomaly_resolved=False,
            interpretation='Positive conditional binding relies on exact official index, response and primary-document mapping, not erroneous canonical/og metadata.'),
        mutable_input_pins=[dict(path=CENTRAL, sha256=central_sha,
            resolved_path=e.resolutions.get(CENTRAL + '@' + central_sha, CENTRAL),
            future_archive_path='outputs/monthly_target70_research/PRE157_' + central_sha + '.json',
            rule='Only the exact SHA-named PRE157 archive may resolve this historical central-state pin after a state update.')],
        synthetic_validation=self_tests(),
        limits=['Original feasibility source/timing reconstruction and KO body semantics are inherited through pinned independent reviews, not newly certified.',
                'WMT body semantics are attributed to the two pinned reviews; this audit verifies exact identity, source chains, event joins and retained gates.',
                'The current official mirror is not a SEC-host byte comparison or complete historical-vintage proof. The FY2024 canonical anomaly remains recorded.',
                'No exact acceptance time or historical price identity is upgraded. Conditional source context is distinct from full strategy admission.',
                'Full original rows, including nested prior141 blockers, remain unchanged. No annual filing lag or numerical strategy value is computed.'],
        activity=dict(network_requests=0, price_values_read=0, lag_values=0, factors=0, ranks=0, orders=0,
                      returns=0, new_strategy_trials=0, frozen_files_modified=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--script-sha')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.test_only:
        need(not args.write, 'Synthetic test cannot write audit')
        print(json.dumps(self_tests(), sort_keys=True))
        return
    need(args.script_sha is not None, 'Explicit script SHA required')
    result = audit(args.script_sha)
    if args.write:
        fd = os.open(str(Evidence().path(OUT)), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(dict(passed=result['passed'], summary=result['summary'], diagnostics=result['diagnostics'],
                         input_files=result['total_verified_unique_input_files'], output=OUT if args.write else None), sort_keys=True))


if __name__ == '__main__':
    main()
