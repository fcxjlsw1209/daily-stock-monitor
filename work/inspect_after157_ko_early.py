"""Derive two pinned KO early annual bodies offline; never admit sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import after157_annual_body_collect as collector
import inspect_after157_annual_bodies as extractor

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'work/after157_ko_early_bodies_scope.json'
SCOPE_SHA = '05cdb5d38e50c2d1a06ebedcfd6680d2ef8bddea9923a481385fa052c6e2e0a2'
PINS = {
    'work/after157_annual_body_collect.py': 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864',
    'work/inspect_after157_annual_bodies.py': '00ce931d5f4088e8a294c52cd03882e0dbde7c1c8a5556201bb57cda570f10bc',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-sha', required=True)
    args = parser.parse_args()
    collector.require(args.scope_sha == SCOPE_SHA, 'Unexpected scope digest')
    for name, expected in PINS.items():
        collector.require(digest((ROOT / name).read_bytes()) == expected,
                          'Changed helper: ' + name)
    scope, out, helper_sha = collector.validate_scope(ROOT, SCOPE, SCOPE_SHA)
    collector.require(out == ROOT / 'outputs/after157_ko_early_bodies_v1',
                      'Unexpected output')
    collector.require([t['id'] for t in scope['targets']] ==
                      ['ko2014_10k', 'ko2015_10k'] and
                      all(t['provider'] == 'coca_cola' and
                          t['role'] == 'original_annual_body_candidate'
                          for t in scope['targets']), 'Unexpected targets')
    summary_raw = (out / 'COLLECTION.json').read_bytes()
    summary = json.loads(summary_raw)
    collector.require(summary['scope_sha256'] == SCOPE_SHA and
                      summary['helper_sha256'] == helper_sha and
                      summary['schema'] == collector.SCHEMA and
                      summary['all_successful'] is True and
                      summary['stopped_providers'] == [] and
                      [t['id'] for t in summary['targets']] ==
                      ['ko2014_10k', 'ko2015_10k'], 'Collection mismatch')
    records, rendered = [], {}
    for target in scope['targets']:
        receipt = collector.verify_existing(out, target,
            collector.base_binding(target, SCOPE_SHA, helper_sha))
        collector.require(receipt is not None and receipt['outcome'] == 'success',
                          'Missing successful body')
        rsha = digest((out / target['receipt_file']).read_bytes())
        matching = [r for r in summary['targets'] if r['id'] == target['id']]
        collector.require(len(matching) == 1 and
                          matching[0]['receipt_file'] == target['receipt_file'] and
                          matching[0]['receipt_sha256'] == rsha and
                          matching[0]['outcome'] == 'success',
                          'Collection receipt mismatch')
        raw = (out / target['body_file']).read_bytes()
        text, method = extractor.source_text(raw, Path(target['body_file']).suffix)
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
        encoded = ('\n'.join(line for line in lines if line) + '\n').encode('utf-8')
        name = target['id'] + '.txt'
        rendered[name] = encoded
        records.append(dict(id=target['id'], role=target['role'],
            outcome=receipt['outcome'], receipt_sha256=rsha,
            http_status=receipt['http_status'], raw_sha256=digest(raw),
            text_file=name, text_sha256=digest(encoded),
            lines=len(encoded.splitlines()), extraction_method=method))
    destination = out / 'derived_text'
    destination.mkdir(exist_ok=False)
    for name, raw in rendered.items():
        collector.write_once(destination / name, raw)
    manifest = dict(scope_sha256=SCOPE_SHA,
        collection_sha256=digest(summary_raw),
        extractor_sha256=digest(Path(__file__).read_bytes()), input_helpers=PINS,
        records=records, created_utc=collector.utc(), admission=False,
        factor_calculations=0,
        note='Search aid only; raw originals control version, identity and fiscal-period review.')
    collector.write_json(destination / 'MANIFEST.json', manifest)
    print(json.dumps(dict(path=str(destination / 'MANIFEST.json'),
        sha256=digest((destination / 'MANIFEST.json').read_bytes()),
        records=len(records), admission=False)))


if __name__ == '__main__':
    main()
