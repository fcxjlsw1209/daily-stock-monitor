"""One explicitly scoped Microsoft iframe GET; no redirects, parsing or admission.

Network execution is separate from run_local.sh. The latter runs synthetic tests.
Only immutable filesystem/transport helpers are reused; old HOSTS are not changed.
"""
import argparse
import json
from pathlib import Path
import urllib.request

import after157_annual_body_collect as base

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after157_msft_iframe_collect.py'
BASE = 'work/after157_annual_body_collect.py'
BASE_SHA = 'dbae4f1fde05d1f6794303f1d5fa57370f0bd304cc9c2cc403be3ba6ac6d5864'
SCOPE = 'work/after157_msft_iframe_scope.json'
OUT = 'outputs/after157_msft_iframe_v1'
URL = 'https://microsoft.gcs-web.com/financial-information/sec-filings'
HOST = 'microsoft.gcs-web.com'
SCHEMA = 'after157_msft_iframe_collection_v1'
RAW = 'msft_sec_filings_iframe.html'
RECEIPT = 'msft_sec_filings_iframe.receipt.json'
BRIDGE = 'outputs/after157_msft_bridge_v1'
PARENT_RAW = BRIDGE + '/msft2015_sec_filings_page.html'
PARENT_RECEIPT = BRIDGE + '/msft2015_sec_filings_page.receipt.json'
PARENT_ATTEMPT = BRIDGE + '/.attempt_msft2015_sec_filings_page.json'
PARENT_SCOPE = 'work/after157_msft_bridge_scope.json'
PARENT_REVIEW = 'work/after157_msft_bridge_review.json'
STOP_GUARDS = [BRIDGE + '/STOPPED_microsoft.json',
               'outputs/after157_annual_period_source_v1/STOPPED_microsoft.json']
MAX_BYTES = 30 * 1024 * 1024
TIMEOUT = 35
require = base.require


def validate(root, scope_sha):
    require(base.digest(scope_sha), 'Exact scope SHA required')
    raw = base.relative(root, SCOPE).read_bytes()
    require(base.sha(raw) == scope_sha, 'Scope SHA mismatch')
    scope = json.loads(raw)
    require(scope['schema'] == SCHEMA and scope['url'] == URL and scope['allowed_host'] == HOST
            and scope['output_dir'] == OUT and scope['body_file'] == RAW
            and scope['receipt_file'] == RECEIPT, 'Fixed single-URL scope changed')
    require(scope['max_bytes'] == MAX_BYTES and scope['timeout_seconds'] == TIMEOUT
            and scope['allow_redirects'] is False and scope['source_admission_changed'] is False,
            'Limits or admission policy changed')
    require(scope['abort_if_exists'] == STOP_GUARDS, 'Source stop guards changed')
    for name in STOP_GUARDS:
        require(not base.relative(root, name).exists(), 'Existing Microsoft source stop: ' + name)
    helper_sha = base.sha(base.relative(root, SELF).read_bytes())
    require(scope['helper_sha256'] == helper_sha, 'Helper SHA mismatch')
    pins = scope['input_hashes']
    mandatory = {BASE, PARENT_RAW, PARENT_RECEIPT, PARENT_ATTEMPT, PARENT_SCOPE,
                 PARENT_REVIEW, BRIDGE + '/COLLECTION.json'}
    require(mandatory.issubset(pins) and pins[BASE] == BASE_SHA, 'Missing parent-source/helper pins')
    for name, expected in pins.items():
        require(base.digest(expected) and base.sha(base.relative(root, name).read_bytes()) == expected,
                'Input SHA mismatch: ' + name)
    parent = json.loads(base.relative(root, PARENT_RECEIPT).read_bytes())
    parent_raw = base.relative(root, PARENT_RAW).read_bytes()
    require(parent['sha256'] == pins[PARENT_RAW] and parent['bytes'] == len(parent_raw)
            and parent['http_status'] == 200 and parent['outcome'] == 'success'
            and parent['scope_sha256'] == pins[PARENT_SCOPE] and parent['helper_sha256'] == BASE_SHA
            and parent['attempt_sha256'] == pins[PARENT_ATTEMPT], 'Parent receipt chain mismatch')
    collection = json.loads(base.relative(root, BRIDGE + '/COLLECTION.json').read_bytes())
    matches = [x for x in collection['targets'] if x['id'] == 'msft2015_sec_filings_page']
    require(collection['all_successful'] is True and collection['stopped_providers'] == []
            and len(matches) == 1 and matches[0]['receipt_sha256'] == pins[PARENT_RECEIPT],
            'Parent collection chain mismatch')
    observed = scope['observed_link']
    require(observed == dict(raw_path=PARENT_RAW, raw_line_1based=875, url=URL), 'Observed-link scope changed')
    lines = parent_raw.decode('utf-8').splitlines()
    require(len(lines) >= 875 and ('src="' + URL + '"') in lines[874], 'Exact iframe not on pinned line875')
    return scope, helper_sha


def fetch(request):
    body = bytearray()
    result = dict(source_url=URL, final_url=URL, http_status=None, headers={},
                  request_started_utc=base.utc(), outcome='request_error', stop_provider=True)
    try:
        response = request(urllib.request.Request(URL, headers={
            'User-Agent': 'LocalAnnualSourceResearch/1.0', 'Accept-Encoding': 'identity', 'Accept': 'text/html'}),
            timeout=TIMEOUT)
        try:
            result['final_url'] = response.geturl()
            require(result['final_url'] == URL, 'Transport followed an unauthorized redirect')
            status = response.getcode()
            result['http_status'] = status
            result['headers'] = {k: response.headers.get(k) for k in
                ('Retry-After', 'Location', 'Content-Type', 'Content-Length', 'Content-Encoding')
                if response.headers.get(k) is not None}
            length = response.headers.get('Content-Length')
            if length is not None:
                require(length.isdigit() and int(length) <= MAX_BYTES, 'Invalid/excessive Content-Length')
            while True:
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                require(len(body) <= MAX_BYTES, 'Response exceeds limit')
            require(length is None or len(body) == int(length), 'Incomplete response')
            result['outcome'] = ('success' if status == 200 and body else
                                 'redirect_not_followed' if 300 <= status < 400 else 'http_error')
            result['stop_provider'] = result['outcome'] != 'success'
        finally:
            response.close()
    except Exception as exc:
        result.update(outcome='request_error', stop_provider=True,
                      error_type=type(exc).__name__, error=str(exc))
    result.update(retrieved_utc=base.utc(), redirect_requests=0,
                  body_truncated_at_limit=len(body) > MAX_BYTES)
    return bytes(body[:MAX_BYTES]), result


def collect(scope_sha, root=ROOT, request=None):
    root = Path(root).resolve()
    scope, helper_sha = validate(root, scope_sha)
    out = base.relative(root, OUT)
    binding = dict(schema=SCHEMA, scope_sha256=scope_sha, helper_sha256=helper_sha,
                   source_url=URL, body_file=RAW)
    marker = out / '.collection-scope.json'
    if out.exists():
        require(base.read_json(base.relative(out, marker.name)) == binding, 'Unowned output directory')
    else:
        out.mkdir()
        base.write_json(marker, binding)
    lock = base.relative(out, '.collection.lock')
    base.write_once(lock, b'One URL, no retry\n')
    try:
        attempt_path = base.relative(out, '.attempt.json')
        receipt_path = base.relative(out, RECEIPT)
        raw_path = base.relative(out, RAW)
        stop_path = base.relative(out, 'STOPPED.json')
        if receipt_path.exists():
            receipt = base.read_json(receipt_path)
            require(all(receipt.get(k) == v for k, v in binding.items()), 'Receipt binding mismatch')
            require(base.sha(attempt_path.read_bytes()) == receipt['attempt_sha256'], 'Attempt SHA mismatch')
            require(all(base.read_json(attempt_path).get(k) == v for k, v in binding.items()),
                    'Attempt binding mismatch')
            raw = raw_path.read_bytes()
            require(base.sha(raw) == receipt['sha256'] and len(raw) == receipt['bytes'], 'Body SHA mismatch')
            require(receipt['final_url'] == URL and receipt['redirect_requests'] == 0
                    and receipt['outcome'] in ('success', 'http_error', 'request_error', 'redirect_not_followed'),
                    'Invalid cached outcome')
            require(receipt['stop_provider'] is (receipt['outcome'] != 'success'), 'Cached stop mismatch')
            if receipt['outcome'] == 'success':
                require(receipt['http_status'] == 200 and raw and not receipt['body_truncated_at_limit'],
                        'Invalid cached success')
        else:
            require(not attempt_path.exists() and not raw_path.exists() and not stop_path.exists(),
                    'Prior/incomplete attempt exists; retry forbidden')
            base.write_json(attempt_path, dict(binding, attempt_started_utc=base.utc()))
            raw, details = fetch(request or base.transport())
            receipt = dict(binding, **details, attempt_sha256=base.sha(attempt_path.read_bytes()),
                           bytes=len(raw), sha256=base.sha(raw), source_admission_changed=False)
            base.write_once(raw_path, raw)
            base.write_json(receipt_path, receipt)
        summary = dict(binding, all_successful=receipt['outcome'] == 'success',
                       source_admission_changed=False, receipt_file=RECEIPT,
                       receipt_sha256=base.sha(receipt_path.read_bytes()))
        if receipt['stop_provider']:
            stop = dict(binding, stopped_utc=receipt['retrieved_utc'], reason=receipt['outcome'],
                        http_status=receipt['http_status'], retry_after=receipt['headers'].get('Retry-After'),
                        receipt_sha256=summary['receipt_sha256'], further_requests_permitted=False)
            if stop_path.exists():
                require(base.read_json(stop_path) == stop, 'Existing stop changed')
            else:
                base.write_json(stop_path, stop)
        else:
            require(not stop_path.exists(), 'Contradictory existing stop')
        validate(root, scope_sha)
        summary_path = base.relative(out, 'COLLECTION.json')
        if summary_path.exists():
            require(base.read_json(summary_path) == summary, 'Existing summary changed')
        else:
            base.write_json(summary_path, summary)
        return summary
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-sha', required=True)
    parser.add_argument('--validate-only', action='store_true',
                        help='Verify the pinned scope and local inputs without writes or HTTP')
    args = parser.parse_args()
    if args.validate_only:
        _, helper_sha = validate(ROOT, args.scope_sha)
        print(json.dumps(dict(scope_validated=True, helper_sha256=helper_sha,
                              scope_sha256=args.scope_sha, network_requests=0), sort_keys=True))
        return 0
    result = collect(args.scope_sha)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['all_successful'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
