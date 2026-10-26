"""Derive text from the two fixed Walmart body batches, entirely offline.

Source bytes and receipts remain authoritative. Text extraction is a search aid,
not original-version, fiscal-period, identity, clock or strategy admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import after157_walmart_collect as collector
import inspect_after157_annual_bodies as extractor

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    'work/after157_walmart_bodies_early_scope.json': {
        'sha256': '4bba503d03789791e8880b1ff50f5d1c4d8e24b97ea7874c8a6df3bb45a15393',
        'output': 'outputs/after157_walmart_bodies_early_v1',
        'years': tuple(range(2015, 2021)),
    },
    'work/after157_walmart_bodies_late_scope.json': {
        'sha256': '11938d1e90419352b83d982647a8be82341582a5c5e9bdf5f695d7d923b6e1c7',
        'output': 'outputs/after157_walmart_bodies_late_v1',
        'years': tuple(range(2021, 2027)),
    },
}
PINS = {
    'work/after157_walmart_collect.py': 'dd80db6f8fbc3636640098164f688a1973f37749f2581cea19aa058893a4f592',
    'work/inspect_after157_annual_bodies.py': '00ce931d5f4088e8a294c52cd03882e0dbde7c1c8a5556201bb57cda570f10bc',
    'work/after157_annual_body_collect.py': 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        raw = collector.relative(root, name).read_bytes()
        collector.require(digest(raw) == expected, 'Input changed: ' + name)


def prepare(root, scope_name, expected_sha):
    """Validate all immutable evidence, then render in memory without writes."""
    root = Path(root).resolve()
    collector.require(scope_name in SCOPES, 'Only the two frozen Walmart scopes are supported')
    config = SCOPES[scope_name]
    collector.require(expected_sha == config['sha256'], 'Unexpected frozen scope SHA')
    verify_hashes(root, PINS)
    scope, out, helper_sha = collector.validate_scope(root, scope_name, expected_sha)
    collector.require(out == root / config['output'], 'Unexpected collection directory')
    collector.require(not collector.relative(out, '.collection.lock').exists(),
                      'Collection is still running')
    targets = scope['targets']
    collector.require(len(targets) == 6 and all(
        t['provider'] == 'walmart' and t['role'] == 'original_document'
        and t['expected_original']['cik'] == '0000104169'
        and t['expected_original']['form'] == '10-K'
        for t in targets), 'Unexpected targets')
    collector.require(tuple(int(t['expected_original']['reportDate'][:4]) for t in targets)
                      == config['years'], 'Unexpected report years/order')

    hashes = dict(scope['input_hashes'])
    hashes.update(PINS)
    hashes[scope_name] = expected_sha
    self_name = Path(__file__).resolve().relative_to(ROOT).as_posix()
    hashes[self_name] = digest(collector.relative(root, self_name).read_bytes())

    def read_bound(name):
        rel = config['output'] + '/' + name
        raw = collector.relative(root, rel).read_bytes()
        actual = digest(raw)
        collector.require(rel not in hashes or hashes[rel] == actual,
                          'Conflicting source hash: ' + rel)
        hashes[rel] = actual
        return raw

    binding = dict(schema=collector.SCHEMA, scope_sha256=expected_sha,
                   helper_sha256=helper_sha)
    collector.require(json.loads(read_bound('.collection-scope.json')) == binding,
                      'Collection ownership marker mismatch')
    summary_raw = read_bound('COLLECTION.json')
    summary = json.loads(summary_raw)
    receipts, records, rendered = [], [], {}
    for target in targets:
        receipt = collector.verify_existing(out, target,
            collector.base_binding(target, expected_sha, helper_sha))
        collector.require(receipt is not None and receipt['outcome'] == 'success'
                          and receipt['attempted'] is True
                          and receipt['http_status'] == 200
                          and receipt['stop_provider'] is False
                          and receipt['body_truncated_at_limit'] is False,
                          'Missing, unsuccessful or truncated body: ' + target['id'])
        receipt_sha = digest(read_bound(target['receipt_file']))
        attempt_sha = digest(read_bound('.attempt_' + target['id'] + '.json'))
        collector.require(attempt_sha == receipt['attempt_sha256'], 'Attempt SHA mismatch')
        raw = read_bound(target['body_file'])
        collector.require(digest(raw) == receipt['sha256'] and len(raw) == receipt['bytes'],
                          'Raw body changed during verification')
        text, method = extractor.source_text(raw, Path(target['body_file']).suffix)
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
        encoded = ('\n'.join(line for line in lines if line) + '\n').encode('utf-8')
        collector.require(encoded.strip(), 'No searchable text: ' + target['id'])
        name = target['id'] + '.txt'
        rendered[name] = encoded
        receipts.append(dict(id=target['id'], outcome='success',
                             receipt_file=target['receipt_file'], receipt_sha256=receipt_sha))
        records.append(dict(id=target['id'], role=target['role'], outcome='success',
            expected_original=target['expected_original'], source_url=target['url'],
            receipt_file=target['receipt_file'], receipt_sha256=receipt_sha,
            attempt_sha256=attempt_sha, http_status=receipt['http_status'],
            raw_file=target['body_file'], raw_sha256=digest(raw), raw_bytes=len(raw),
            text_file=name, text_sha256=digest(encoded), lines=len(encoded.splitlines()),
            extraction_method=method))
    collector.require(summary == dict(binding, targets=receipts,
                      stopped_providers=[], all_successful=True),
                      'Collection summary differs from exact successful target set')
    collector.require(not collector.relative(out, 'STOPPED_walmart.json').exists(),
                      'Successful collection has a provider stop')
    verify_hashes(root, hashes)
    manifest = dict(schema='after157_walmart_derived_text_v1', scope=scope_name,
        scope_sha256=expected_sha, collection_sha256=digest(summary_raw),
        extractor_sha256=hashes[self_name], input_helpers=PINS, input_hashes=hashes,
        records=records, created_utc=collector.utc(), admission=False,
        source_admission_changed=False, network_requests=0, factor_calculations=0,
        filing_lag_calculations=0, prices_or_returns_read=0,
        note='Search aid only. Raw originals control checkbox/layout, exact version, '
             'identity and complete ordinary/consecutive fiscal-period adjudication.')
    return out, rendered, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True, choices=tuple(SCOPES))
    parser.add_argument('--scope-sha', required=True)
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify and render in memory without creating derived files')
    args = parser.parse_args()
    out, rendered, manifest = prepare(ROOT, args.scope, args.scope_sha)
    destination = collector.relative(out, 'derived_text')
    if not args.verify_only:
        # Validate everything before writing; never replace even a partial derivation.
        destination.mkdir(exist_ok=False)
        for name, raw in rendered.items():
            collector.write_once(destination / name, raw)
        verify_hashes(ROOT, manifest['input_hashes'])
        collector.write_json(destination / 'MANIFEST.json', manifest)
    print(json.dumps(dict(path=str(destination / 'MANIFEST.json'),
        sha256=None if args.verify_only else digest((destination / 'MANIFEST.json').read_bytes()),
        records=len(manifest['records']), verified=True, written=not args.verify_only,
        admission=False), sort_keys=True))


if __name__ == '__main__':
    main()
