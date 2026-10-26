"""Write-once, pinned-scope public issuer collection; never a research replay.

Run collection separately from run_local.sh (which intentionally denies network).
Tests inject a transport and run offline through run_local.sh. No content parsing,
financial calculations, retries, cookies, credentials or environment mutations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'after157_annual_body_collection_v1'
MAX_BYTES = 30 * 1024 * 1024
TIMEOUT = 35
HOSTS = {'microsoft': 'www.microsoft.com',
         'coca_cola': 'investors.coca-colacompany.com'}
REDIRECTS = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 5
RESERVED = {'COLLECTION.json', '.collection-scope.json', '.collection.lock'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative(root, value):
    require(isinstance(value, str) and value and '\\' not in value,
            'Invalid relative path')
    parts = value.split('/')
    require(all(p not in ('', '.', '..') for p in parts), 'Path traversal/absolute path')
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink forbidden: ' + value)
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes root')
    return path


def filename(value):
    require(isinstance(value, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value)
            and value not in RESERVED and not value.startswith('STOPPED_'),
            'Invalid/reserved output filename')
    return value


def valid_url(url, provider):
    require(isinstance(url, str) and url.isascii()
            and not any(ord(c) <= 32 or ord(c) == 127 for c in url)
            and '\\' not in url, 'Invalid URL characters')
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname == HOSTS[provider]
            and parsed.username is None and parsed.password is None
            and parsed.port in (None, 443) and not parsed.fragment,
            'URL outside fixed HTTPS provider host')
    return url


def read_json(path):
    return json.loads(path.read_bytes())


def write_once(path, data):
    # O_EXCL also rejects dangling symlinks; never truncate an existing artifact.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    write_once(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def validate_scope(root, scope_relative, expected_sha):
    require(digest(expected_sha), 'Expected scope SHA required')
    source = relative(root, scope_relative)
    raw = source.read_bytes()
    require(sha(raw) == expected_sha, 'Scope SHA mismatch')
    scope = json.loads(raw)
    helper_sha = sha(Path(__file__).read_bytes())
    require(scope.get('schema') == SCHEMA, 'Scope schema mismatch')
    require(scope.get('helper_sha256') == helper_sha, 'Helper SHA mismatch')
    require(scope.get('max_bytes', MAX_BYTES) == MAX_BYTES
            and scope.get('timeout_seconds', TIMEOUT) == TIMEOUT, 'Fixed request limits changed')
    if 'allowed_hosts' in scope:
        require(sorted(scope['allowed_hosts']) == sorted(HOSTS.values()), 'Fixed host list changed')
    out = relative(root, scope['output_dir'])
    parts = scope['output_dir'].split('/')
    require(len(parts) == 2 and parts[0] == 'outputs' and not parts[1].endswith('_monitor'),
            'Output must be a new direct outputs/ child, never a monitor')
    require(isinstance(scope.get('input_hashes'), dict) and scope['input_hashes'],
            'Pinned input_hashes required')
    for name, expected in scope['input_hashes'].items():
        require(digest(expected), 'Invalid input SHA')
        require(sha(relative(root, name).read_bytes()) == expected, 'Input SHA mismatch: ' + name)
    targets = scope.get('targets')
    require(isinstance(targets, list) and 1 <= len(targets) <= 8, 'One to eight targets required')
    ids, names, urls = set(), set(RESERVED), set()
    for target in targets:
        key = target['id']
        require(isinstance(key, str) and re.fullmatch('[a-z][a-z0-9_]*', key)
                and key not in ids, 'Invalid/duplicate target id')
        ids.add(key)
        provider = target['provider']
        require(provider in HOSTS, 'Unknown provider')
        valid_url(target['url'], provider)
        require(target['url'] not in urls, 'Duplicate target URL')
        urls.add(target['url'])
        for field in ('body_file', 'receipt_file'):
            name = filename(target[field])
            require(name not in names, 'Duplicate output filename')
            names.add(name)
    return scope, out, helper_sha


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def transport():
    # Do not inherit environment proxy credentials; do not install a global opener.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(req, timeout):
        try:
            return opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as response:
            return response
    return request


def fetch(target, request):
    current = target['url']
    chain, visited = [], set()
    result = dict(source_url=current, final_url=current, http_status=None,
                  request_started_utc=utc(), retrieved_utc=None, headers={},
                  redirect_chain=chain, outcome='network_error', stop_provider=True)
    body = bytearray()
    try:
        while True:
            valid_url(current, target['provider'])
            require(current not in visited, 'Redirect loop')
            visited.add(current)
            result['final_url'] = current
            req = urllib.request.Request(current, headers={
                'User-Agent': 'LocalAnnualSourceResearch/1.0',
                'Accept-Encoding': 'identity', 'Accept': '*/*'})
            response = request(req, timeout=TIMEOUT)
            try:
                # Our transport never auto-redirects; fail closed for a replacement transport.
                require(response.geturl() == current, 'Transport performed implicit redirect')
                status = response.getcode()
                result['http_status'] = status
                result['headers'] = {k: response.headers.get(k) for k in
                    ('Retry-After', 'Content-Type', 'Content-Length', 'Content-Encoding', 'Location')
                    if response.headers.get(k) is not None}
                if status in REDIRECTS:
                    location = response.headers.get('Location')
                    require(location is not None and len(chain) < MAX_REDIRECTS,
                            'Missing Location or redirect limit')
                    next_url = urllib.parse.urljoin(current, location)
                    chain.append(dict(url=current, http_status=status, location=location,
                                      next_url=next_url, retrieved_utc=utc(),
                                      retry_after=response.headers.get('Retry-After')))
                    # Validate before making ANY request to the destination.
                    valid_url(next_url, target['provider'])
                    current = next_url
                    continue
                length = response.headers.get('Content-Length')
                if length is not None:
                    require(length.isdigit(), 'Invalid Content-Length')
                    require(int(length) <= MAX_BYTES, 'Response exceeds max bytes')
                while True:
                    chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)))
                    if not chunk:
                        break
                    body.extend(chunk)
                    require(len(body) <= MAX_BYTES, 'Response exceeds max bytes')
                require(length is None or int(length) == len(body), 'Incomplete Content-Length body')
                result['outcome'] = 'success' if status == 200 and body else 'http_error'
                result['stop_provider'] = status in (401, 403, 429) or status >= 500
                if status == 200 and not body or status == 206:
                    result.update(outcome='invalid_body', stop_provider=True)
                break
            finally:
                response.close()
    except Exception as exc:
        result.update(outcome='request_error', stop_provider=True,
                      error_type=type(exc).__name__, error=str(exc))
    result['retrieved_utc'] = utc()
    # The limit probe byte is never committed to an artifact.
    result['body_truncated_at_limit'] = len(body) > MAX_BYTES
    return bytes(body[:MAX_BYTES]), result


def base_binding(target, scope_sha, helper_sha):
    return dict(schema=SCHEMA, id=target['id'], provider=target['provider'],
                source_url=target['url'], body_file=target['body_file'],
                scope_sha256=scope_sha, helper_sha256=helper_sha)


def verify_existing(out, target, binding):
    receipt_path = relative(out, target['receipt_file'])
    body_path = relative(out, target['body_file'])
    attempt_path = relative(out, '.attempt_' + target['id'] + '.json')
    if not receipt_path.exists():
        require(not body_path.exists(), 'Body without receipt; no overwrite/retry: ' + target['id'])
        require(not attempt_path.exists(), 'Interrupted attempt without receipt; no retry: ' + target['id'])
        return None
    receipt = read_json(receipt_path)
    require(all(receipt.get(k) == v for k, v in binding.items()), 'Receipt binding mismatch')
    require(receipt.get('outcome') in ('success', 'http_error', 'invalid_body',
                                      'request_error', 'skipped_provider_stopped'), 'Invalid receipt outcome')
    if receipt.get('attempted') is True:
        require(attempt_path.is_file(), 'Attempt marker missing')
        attempt = read_json(attempt_path)
        require(all(attempt.get(k) == v for k, v in binding.items())
                and receipt.get('attempt_sha256') == sha(attempt_path.read_bytes()),
                'Attempt marker binding mismatch')
        require(body_path.is_file(), 'Receipt body missing')
        raw = body_path.read_bytes()
        require(len(raw) == receipt['bytes'] and sha(raw) == receipt['sha256'], 'Existing body SHA mismatch')
        require(len(raw) <= MAX_BYTES, 'Existing body exceeds limit')
        if receipt['outcome'] == 'success':
            require(receipt['http_status'] == 200 and raw and not receipt['stop_provider'],
                    'Malformed successful receipt')
            valid_url(receipt['final_url'], target['provider'])
    else:
        require(receipt.get('attempted') is False and receipt['outcome'] == 'skipped_provider_stopped'
                and not body_path.exists() and not attempt_path.exists(), 'Invalid skipped receipt')
    return receipt


def collect(scope_relative, scope_sha, root=ROOT, request=None):
    root = Path(root).resolve()
    scope, out, helper_sha = validate_scope(root, scope_relative, scope_sha)
    binding = dict(schema=SCHEMA, scope_sha256=scope_sha, helper_sha256=helper_sha)
    marker = out / '.collection-scope.json'
    if out.exists():
        require(out.is_dir() and marker.is_file() and not marker.is_symlink()
                and read_json(marker) == binding, 'Existing output directory is not this pinned collection')
    else:
        out.mkdir()  # Parent outputs/ must already exist. Never adopt an existing directory.
        write_json(marker, binding)
    lock = relative(out, '.collection.lock')
    write_once(lock, b'Exclusive collection in progress\n')
    try:
        receipts, stopped = [], {}
        # Verify all old evidence before any new request, including later targets.
        existing = {t['id']: verify_existing(out, t, base_binding(t, scope_sha, helper_sha))
                    for t in scope['targets']}
        for provider in HOSTS:
            stop_path = relative(out, 'STOPPED_' + provider + '.json')
            if stop_path.exists():
                stop = read_json(stop_path)
                require(all(stop.get(k) == v for k, v in binding.items())
                        and stop.get('provider') == provider, 'Provider stop binding mismatch')
                require(any(r and r['id'] == stop.get('failed_target') and r['provider'] == provider
                            and r.get('stop_provider') and r.get('attempted')
                            and sha((out / t['receipt_file']).read_bytes()) == stop.get('receipt_sha256')
                            for t in scope['targets'] for r in [existing[t['id']]]),
                        'Provider stop lacks exact failed receipt')
                stopped[provider] = stop
        # A crash may have committed a failure receipt just before its stop file.
        # Recover that stop before attempting any new target of that provider.
        for target in scope['targets']:
            receipt = existing[target['id']]
            provider = target['provider']
            if receipt and receipt.get('attempted') and receipt.get('stop_provider') and provider not in stopped:
                stop = dict(binding, provider=provider, failed_target=target['id'],
                            stopped_utc=receipt['retrieved_utc'], outcome=receipt['outcome'],
                            http_status=receipt['http_status'],
                            retry_after=receipt['headers'].get('Retry-After'),
                            receipt_file=target['receipt_file'],
                            receipt_sha256=sha((out / target['receipt_file']).read_bytes()))
                write_json(relative(out, 'STOPPED_' + provider + '.json'), stop)
                stopped[provider] = stop
        request = request or transport()
        for target in scope['targets']:
            provider = target['provider']
            receipt = existing[target['id']]
            if receipt is None:
                receipt = base_binding(target, scope_sha, helper_sha)
                if provider in stopped:
                    receipt.update(attempted=False, outcome='skipped_provider_stopped',
                                   stop_provider=True, failed_target=stopped[provider]['failed_target'],
                                   retrieved_utc=utc(), final_url=None, http_status=None,
                                   bytes=0, sha256=None, headers={}, redirect_chain=[])
                else:
                    attempt_path = relative(out, '.attempt_' + target['id'] + '.json')
                    write_json(attempt_path, dict(receipt, attempt_started_utc=utc()))
                    body, details = fetch(target, request)
                    receipt.update(details, attempted=True, bytes=len(body), sha256=sha(body),
                                   attempt_sha256=sha(attempt_path.read_bytes()))
                    write_once(relative(out, target['body_file']), body)
                write_json(relative(out, target['receipt_file']), receipt)
            if receipt['outcome'] == 'skipped_provider_stopped':
                require(provider in stopped and receipt['failed_target'] == stopped[provider]['failed_target'],
                        'Skipped receipt lacks provider stop')
            elif receipt.get('stop_provider') and provider not in stopped:
                stop = dict(binding, provider=provider, failed_target=target['id'],
                            stopped_utc=receipt['retrieved_utc'], outcome=receipt['outcome'],
                            http_status=receipt['http_status'],
                            retry_after=receipt['headers'].get('Retry-After'),
                            receipt_file=target['receipt_file'],
                            receipt_sha256=sha((out / target['receipt_file']).read_bytes()))
                write_json(relative(out, 'STOPPED_' + provider + '.json'), stop)
                stopped[provider] = stop
            receipts.append(dict(id=target['id'], outcome=receipt['outcome'],
                                 receipt_file=target['receipt_file'],
                                 receipt_sha256=sha((out / target['receipt_file']).read_bytes())))
        validate_scope(root, scope_relative, scope_sha)
        summary = dict(binding, targets=receipts, stopped_providers=sorted(stopped),
                       all_successful=all(r['outcome'] == 'success' for r in receipts))
        summary_path = relative(out, 'COLLECTION.json')
        if summary_path.exists():
            require(read_json(summary_path) == summary, 'Existing summary mismatch')
        else:
            write_json(summary_path, summary)
        return summary
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True, help='Pinned JSON path relative to research root')
    parser.add_argument('--scope-sha', required=True, help='Exact SHA-256 of scope bytes')
    args = parser.parse_args()
    result = collect(args.scope, args.scope_sha)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['all_successful'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
