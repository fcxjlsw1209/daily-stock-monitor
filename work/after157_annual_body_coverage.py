"""Offline, source-only body-context overlay. No automatic admission or lag calculation.

The original feasibility report is consumed at its frozen digest. Its historical
769-member input audit is inherited, not repeated here. Every newly collected
raw body, receipt, attempt and derived text, plus all scope/review input pins,
is rehashed. Pinned human reviews supply semantic judgments; this helper checks
their scope and joins exact event IDs, never infers a version from issuer/year.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_annual_body_coverage.py'
PRIOR = 'work/after157_annual_pair_source_feasibility.json'
PRIOR_SHA = '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242'
SCOPE = 'work/after157_annual_body_scope.json'
SCOPE_SHA = 'f49f335509ae9e769750a5a521f1897795aa38b907b430ecadac38cb85cc3709'
OUT = 'outputs/after157_annual_period_source_v1'
COLLECTION = OUT + '/COLLECTION.json'
COLLECTION_SHA = '0fd90aa8642ad340d45b9c7a33a2485ff505eeaf627c9b13985461ff11396838'
DERIVED = OUT + '/derived_text/MANIFEST.json'
DERIVED_SHA = 'd3527fdf4c962c4181375df31f9c4e81aa7122e4a9baac7b916d9b2557f48a25'
COLLECTOR = 'work/after157_annual_body_collect.py'
EXTRACTOR = 'work/inspect_after157_annual_bodies.py'
REVIEWS = {'coca_cola': 'work/after157_ko_body_review.json',
           'microsoft': 'work/after157_msft_body_review.json'}
ISSUERS = {'coca_cola': ('KO', '0000021344', 'investors.coca-colacompany.com'),
           'microsoft': ('MSFT', '0000789019', 'www.microsoft.com')}
SCHEMA = 'after157_annual_body_collection_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, name):
        require(isinstance(name, str) and name and '\\' not in name, 'Invalid path')
        path = Path(name)
        if path.is_absolute():
            require(path.is_relative_to(self.root), 'External path forbidden')
            name = str(path.relative_to(self.root))
        require(all(p not in ('', '.', '..') for p in name.split('/')), 'Path traversal')
        path = self.root
        for part in name.split('/'):
            path /= part
            require(not path.is_symlink(), 'Symlink forbidden')
        return path, name

    def read(self, name, expected=None):
        path, name = self.path(name)
        raw = path.read_bytes()
        actual = sha(raw)
        if expected is not None:
            require(is_sha(expected) and actual == expected, 'SHA mismatch: ' + name)
        require(name not in self.hashes or self.hashes[name] == actual, 'Input changed: ' + name)
        self.hashes[name] = actual
        return raw

    def json(self, name, expected=None):
        return json.loads(self.read(name, expected))

    def pins(self, pins):
        require(isinstance(pins, dict) and pins, 'Missing input pins')
        normalized = {}
        for name, expected in pins.items():
            self.read(name, expected)
            _, local = self.path(name)
            require(local not in normalized, 'Duplicate normalized input pin')
            normalized[local] = expected
        return normalized

    def reverify(self):
        for name, expected in list(self.hashes.items()):
            self.read(name, expected)


def unique_index(records, key):
    require(isinstance(records, list), 'Expected record list')
    result = {}
    for record in records:
        value = record[key]
        require(value not in result, 'Duplicate ' + key)
        result[value] = record
    return result


def name_only(name):
    require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', name),
            'Invalid artifact basename')
    return name


def check_url(url, provider):
    require(isinstance(url, str) and url.isascii() and '\\' not in url
            and not any(ord(c) <= 32 or ord(c) == 127 for c in url), 'Invalid URL')
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname == ISSUERS[provider][2]
            and parsed.username is None and parsed.password is None
            and parsed.port in (None, 443) and not parsed.fragment, 'Wrong issuer URL')


def verify_collection(reader, scope, collection, derived):
    require(scope['schema'] == SCHEMA and scope['output_dir'] == OUT, 'Unexpected scope')
    reader.read(COLLECTOR, scope['helper_sha256'])
    reader.pins(scope['input_hashes'])
    binding = dict(schema=SCHEMA, scope_sha256=SCOPE_SHA, helper_sha256=scope['helper_sha256'])
    require(reader.json(OUT + '/.collection-scope.json') == binding, 'Collection marker mismatch')
    require(all(collection.get(k) == v for k, v in binding.items())
            and collection.get('all_successful') is True and collection.get('stopped_providers') == [],
            'Incomplete collection')
    require(derived['scope_sha256'] == SCOPE_SHA and derived['collection_sha256'] == COLLECTION_SHA
            and derived['admission'] is False and derived['factor_calculations'] == 0,
            'Derived manifest binding mismatch')
    reader.read(EXTRACTOR, derived['extractor_sha256'])
    targets = unique_index(scope['targets'], 'id')
    summaries = unique_index(collection['targets'], 'id')
    texts = unique_index(derived['records'], 'id')
    require(len(targets) == 8 and set(targets) == set(summaries) == set(texts), 'Target set mismatch')
    files = {}
    for key, target in targets.items():
        provider = target['provider']
        require(provider in ISSUERS, 'Unexpected issuer')
        check_url(target['url'], provider)
        rpath = OUT + '/' + name_only(target['receipt_file'])
        bpath = OUT + '/' + name_only(target['body_file'])
        apath = OUT + '/.attempt_' + name_only(key) + '.json'
        summary, text = summaries[key], texts[key]
        require(summary['outcome'] == 'success' and summary['receipt_file'] == target['receipt_file'],
                'Bad target summary')
        receipt = reader.json(rpath, summary['receipt_sha256'])
        target_binding = dict(binding, id=key, provider=provider, source_url=target['url'],
                              body_file=target['body_file'])
        require(all(receipt.get(k) == v for k, v in target_binding.items())
                and receipt.get('attempted') is True and receipt.get('outcome') == 'success'
                and receipt.get('http_status') == 200 and receipt.get('stop_provider') is False
                and receipt.get('body_truncated_at_limit') is False, 'Bad collection receipt')
        check_url(receipt['final_url'], provider)
        current = target['url']
        require(len(receipt['redirect_chain']) <= 5, 'Too many redirects')
        for redirect in receipt['redirect_chain']:
            require(redirect['url'] == current and redirect['http_status'] in (301, 302, 303, 307, 308),
                    'Broken redirect chain')
            check_url(redirect['next_url'], provider)
            current = redirect['next_url']
        require(current == receipt['final_url'], 'Final URL chain mismatch')
        attempt = reader.json(apath, receipt['attempt_sha256'])
        require(all(attempt.get(k) == v for k, v in target_binding.items()), 'Attempt binding mismatch')
        raw = reader.read(bpath, receipt['sha256'])
        require(len(raw) == receipt['bytes'] and 0 < len(raw) <= 30 * 1024 * 1024, 'Raw size mismatch')
        require(text['outcome'] == 'success' and text['http_status'] == 200
                and text['role'] == target['role'] and text['raw_sha256'] == receipt['sha256']
                and text['receipt_sha256'] == summary['receipt_sha256'], 'Text/raw binding mismatch')
        tpath = OUT + '/derived_text/' + name_only(text['text_file'])
        raw_text = reader.read(tpath, text['text_sha256'])
        require(len(raw_text.decode('utf-8').splitlines()) == text['lines'], 'Text line count mismatch')
        files[key] = {p: reader.hashes[p] for p in (bpath, rpath, apath, tpath)}
    require(len({p for group in files.values() for p in group}) == 32, 'Artifact reused by multiple targets')
    return targets, files


def alias_list(review, names):
    values = [review[k] for k in names if k in review]
    require(values and all(isinstance(v, list) and len(v) == len(set(v)) for v in values),
            'Missing/duplicate review identities')
    require(all(set(v) == set(values[0]) for v in values), 'Conflicting review aliases')
    return set(values[0])


def validate_review(review, provider, targets, required_hashes):
    for key in ('exact_original_versions_bound', 'normal_consecutive_pair_context_verified',
                'source_admission_changed'):
        require(type(review.get(key)) is bool, 'Review field must be boolean: ' + key)
    require(review['source_admission_changed'] is False, 'Review must not change admission')
    own = {key: t for key, t in targets.items() if t['provider'] == provider}
    bodies = {key: t for key, t in own.items() if 'expected_original' in t}
    require(len(own) == 4 and len(bodies) == 2, 'Unexpected issuer scope')
    tids = alias_list(review, ('target_ids', 'reviewed_target_ids'))
    eids = alias_list(review, ('expected_event_ids', 'reviewed_event_ids'))
    require(set(bodies).issubset(tids) and tids.issubset(own), 'Review targets differ from issuer scope')
    require(eids == {t['expected_original']['event_id'] for t in bodies.values()}, 'Review event IDs mismatch')
    pins = review['input_hashes']
    require(all(pins.get(path) == expected for path, expected in required_hashes.items()),
            'Review does not bind required evidence')
    return dict(event_ids=sorted(eids), exact_original_versions_bound=review['exact_original_versions_bound'],
                normal_consecutive_pair_context_verified=review['normal_consecutive_pair_context_verified'])


def documents_from_scope(targets, events):
    documents = {}
    for key, target in targets.items():
        if 'expected_original' not in target:
            continue
        expected = target['expected_original']
        event_id = expected['event_id']
        require(event_id in events and event_id not in documents, 'Unknown/duplicate scoped event')
        require(all(events[event_id].get(k) == v for k, v in expected.items()),
                'Scoped original differs from prior event')
        provider = target['provider']
        require(expected['cik'] == ISSUERS[provider][1] and expected['form'] == '10-K',
                'Scoped issuer/form mismatch')
        documents[event_id] = dict(target_id=key, provider=provider, symbol=ISSUERS[provider][0],
                                   expected_original=copy.deepcopy(expected))
    require(len(documents) == 4, 'Expected exactly four original bodies')
    return documents


def overlay_rows(prior_rows, events, documents, reviews):
    require(len(prior_rows) == 5978, 'Original full grid required')
    expected_keys = [(i, j) for i in range(122) for j in range(49)]
    require([(r['signal_index'], r['candidate_index']) for r in prior_rows] == expected_keys,
            'Original grid order/key mismatch')
    dates = []
    for i in range(122):
        month = prior_rows[i * 49:(i + 1) * 49]
        require(all(type(r['is_etf']) is bool for r in month)
                and sum(r['is_etf'] for r in month) == 17, 'Stock/ETF denominator changed')
        require(len({r['symbol'] for r in month}) == 49
                and len({r['signal_date'] for r in month}) == 1, 'Month axis mismatch')
        dates.append(month[0]['signal_date'])
    require(len(set(dates)) == 122 and dates == sorted(dates), 'Signal dates changed')
    result = []
    for row in prior_rows:
        ids = row['selected_event_ids']
        require(isinstance(ids, list) and len(ids) == len(set(ids)) and all(e in events for e in ids),
                'Unknown/duplicate original selected event')
        require(type(row['tier2']) is bool and type(row['tier3']) is bool, 'Invalid original tier')
        require(not row['is_etf'] or not ids, 'ETF acquired filing events')
        matched = [documents[e] for e in ids if e in documents]
        require(all(d['symbol'] == row['symbol'] and d['expected_original']['cik'] == row['cik']
                    for d in matched), 'Event joined to wrong issuer/security row')
        pair_matches = len(ids) == 2 and len(matched) == 2
        verified = False
        pending = copy.deepcopy(row['reasons'])
        if pair_matches:
            providers = {d['provider'] for d in matched}
            require(len(providers) == 1, 'Cross-issuer body pair')
            review = reviews[next(iter(providers))]
            require(set(ids) == set(review['event_ids']), 'Pair differs from reviewed pair')
            verified = review['exact_original_versions_bound'] and review['normal_consecutive_pair_context_verified']
            if not review['exact_original_versions_bound']:
                pending.append('BODY_EXACT_ORIGINAL_VERSION_BINDING_PENDING')
            if not review['normal_consecutive_pair_context_verified']:
                pending.append('BODY_NORMAL_CONSECUTIVE_PAIR_CONTEXT_PENDING')
            if verified:
                pending.append('BODY_CONTEXT_ONLY_PRIOR_ADMISSION_FLAGS_NOT_SUPERSEDED')
        elif not row['is_etf']:
            pending.append('SELECTED_PAIR_NOT_FULLY_COVERED_BY_THIS_BODY_BATCH')
        result.append(dict(signal_index=row['signal_index'], candidate_index=row['candidate_index'],
                           signal_date=row['signal_date'], symbol=row['symbol'], is_etf=row['is_etf'],
                           selected_event_ids=copy.deepcopy(ids), prior_tier2=row['tier2'], prior_tier3=row['tier3'],
                           prior_reasons=copy.deepcopy(row['reasons']),
                           prior_existing_accession_context=copy.deepcopy(row.get('existing_accession_context', [])),
                           body_pair_context_verified=bool(verified), still_pending_reasons=pending))
    return result


def summarize(rows, documents):
    result = {}
    for name, start, end in [('all122', 0, 122), ('first88', 0, 88), ('later34', 88, 122)]:
        subset = [r for r in rows if start <= r['signal_index'] < end]
        count = lambda test: sum(bool(test(r)) for r in subset)
        result[name] = dict(candidate_slots=len(subset), stock_slots=count(lambda r: not r['is_etf']),
            etf_slots=count(lambda r: r['is_etf']), prior_tier2_slots=count(lambda r: r['prior_tier2']),
            prior_tier3_slots=count(lambda r: r['prior_tier3']),
            any_document_match_slots=count(lambda r: any(e in documents for e in r['selected_event_ids'])),
            both_document_match_slots=count(lambda r: len(r['selected_event_ids']) == 2
                                             and all(e in documents for e in r['selected_event_ids'])),
            documented_pair_slots=count(lambda r: r['body_pair_context_verified']),
            tier2_with_documented_pair_slots=count(lambda r: r['prior_tier2'] and r['body_pair_context_verified']))
    return result


def build(ko_review_sha, msft_review_sha, root=ROOT):
    reader = Reader(root)
    reader.read(SELF)
    prior = reader.json(PRIOR, PRIOR_SHA)
    scope = reader.json(SCOPE, SCOPE_SHA)
    collection = reader.json(COLLECTION, COLLECTION_SHA)
    derived = reader.json(DERIVED, DERIVED_SHA)
    targets, files = verify_collection(reader, scope, collection, derived)
    documents = documents_from_scope(targets, prior['selected_original_and_amendment_metadata'])
    review_results = {}
    for provider, expected_sha in [('coca_cola', ko_review_sha), ('microsoft', msft_review_sha)]:
        require(is_sha(expected_sha), 'Explicit review SHA required')
        path = REVIEWS[provider]
        review = reader.json(path, expected_sha)
        review['input_hashes'] = reader.pins(review['input_hashes'])
        required = {PRIOR: PRIOR_SHA, SCOPE: SCOPE_SHA, COLLECTION: COLLECTION_SHA, DERIVED: DERIVED_SHA}
        for key, target in targets.items():
            if target['provider'] == provider and 'expected_original' in target:
                required.update(files[key])
        checked = validate_review(review, provider, targets, required)
        review_results[provider] = dict(checked, path=path, sha256=expected_sha)
    rows = overlay_rows(prior['rows'], prior['selected_original_and_amendment_metadata'], documents, review_results)
    summary = summarize(rows, documents)
    for segment in summary:
        require(summary[segment]['prior_tier2_slots'] == prior['summary'][segment]['tier2']['slots']
                and summary[segment]['prior_tier3_slots'] == prior['summary'][segment]['tier3']['slots'],
                'Original tier count mismatch')
    for event_id, document in documents.items():
        document['evidence_files'] = files[document['target_id']]
        document['review'] = review_results[document['provider']]
    reader.reverify()
    report = dict(schema='after157_annual_body_coverage_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_admission_changed=False,
        new_strategy_trials=0, tier3_upgrades=0, summary=summary, rows=rows,
        documents_by_event_id=documents, reviews=review_results, input_hashes=dict(reader.hashes),
        prior_evidence_gate=copy.deepcopy(prior['evidence_gate']),
        checks=dict(full_original_grid_preserved=True, all_eight_raw_receipt_attempt_text_sets_verified=True,
                    scoped_event_metadata_exactly_matched=True, review_event_and_evidence_bindings_verified=True,
                    original_tiers_reasons_accession_context_preserved=True),
        limits=['Documented pair means both exact original versions and normal consecutive body context passed the pinned review.',
                'Original blockers remain in prior_reasons and still_pending_reasons even for a documented pair; no automatic superseding adjudication.',
                'Original t-42 timing, conditional identity and 141 flags are inherited unchanged; this overlay certifies no new public timestamp or historical identity.',
                'The original feasibility report is SHA-pinned; its historical 769-member audit is not rerun. All new collection and review input pins are rehashed.',
                'No lag, factor, rank, order, return, model fit or strategy registration is computed.'])
    return report, reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ko-review-sha', required=True)
    parser.add_argument('--msft-review-sha', required=True)
    parser.add_argument('--write', action='store_true', help='Write the new COVERAGE.json exclusively')
    args = parser.parse_args()
    report, reader = build(args.ko_review_sha, args.msft_review_sha)
    if args.write:
        path, _ = reader.path(OUT + '/COVERAGE.json')
        raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        reader.reverify()
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(dict(path=str(path), sha256=sha(raw), summary=report['summary'])))
    else:
        print(json.dumps(dict(write=False, summary=report['summary'])))


if __name__ == '__main__':
    main()
