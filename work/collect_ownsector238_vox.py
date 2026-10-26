"""Bounded, separately authorized VOX acquisition; no source admission or research.

CLI has two independent commands. Never run live collection through run_local.sh;
that wrapper remains offline. Tests inject HTTP/Keychain and use the wrapper.
Each phase gets one durable attempt and at most one HTTP GET. A failed or partial
phase permanently stops this directory. No retry, redirect, fallback or recovery.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ownsector238_vox_sources'
SYMBOL = 'VOX'
START, END = '2015-01-01', '2026-09-24'
HOST = 'api.tiingo.com'
BASE = 'https://' + HOST + '/tiingo/daily/' + SYMBOL
SERVICE = 'daily-stock-monitor.tiingo'
MAX_BYTES = 16 * 1024 * 1024
TIMEOUT = 45
PHASES = ('metadata', 'prices')
SAFE_HEADERS = ('Date', 'Content-Type', 'Content-Length', 'Content-Encoding',
                'Retry-After', 'ETag', 'Last-Modified', 'X-RateLimit-Limit',
                'X-RateLimit-Remaining', 'X-RateLimit-Reset')
PRICE_PARAMS = {'startDate': START, 'endDate': END, 'format': 'json', 'resampleFreq': 'daily'}
SCHEMA = 'ownsector238_vox_acquisition_v1'


class AcquisitionError(RuntimeError):
    """Only fixed, non-secret reason codes may be exposed."""


def require(ok, reason):
    if not ok:
        raise AcquisitionError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def stamp(value):
    require(isinstance(value, datetime) and value.tzinfo is not None, 'invalid_clock')
    return value.astimezone(timezone.utc).isoformat()


def descriptor(phase):
    require(phase in PHASES, 'invalid_phase')
    params = {} if phase == 'metadata' else dict(PRICE_PARAMS)
    endpoint = BASE if phase == 'metadata' else BASE + '/prices'
    url = endpoint + ('?' + urllib.parse.urlencode(params) if params else '')
    return {'phase': phase, 'symbol': SYMBOL, 'endpoint': endpoint, 'parameters': params,
            'request_url': url, 'method': 'GET', 'redirects_permitted': False,
            'retries_permitted': False, 'maximum_gets': 1}


def scope(helper_sha):
    return {'schema': SCHEMA, 'collector_sha256': helper_sha, 'provider_host': HOST,
            'symbol': SYMBOL, 'requested_start': START, 'requested_end': END,
            'phases': {p: descriptor(p) for p in PHASES},
            'source_admitted': False, 'shared_stop_ledger_modified': False}


def safe_directory(path):
    path = Path(path).absolute()
    require('..' not in path.parts, 'invalid_output_path')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'symlink_forbidden')
        require(not any(x in item.name.casefold() for x in ('cloudstorage', 'onedrive')),
                'cloud_path_forbidden')
    require(path.parent.is_dir(), 'output_parent_missing')
    return path


def exists(path):
    return os.path.lexists(str(path))


def read_bytes(path, max_bytes=MAX_BYTES):
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'nonexclusive_input_file')
        require(info.st_size <= max_bytes, 'stored_file_too_large')
        raw = handle.read(max_bytes + 1)
    require(len(raw) <= max_bytes, 'stored_file_too_large')
    return raw


def write_new(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return sha(raw)


def parse_json(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, 'duplicate_json_key')
            obj[key] = value
        return obj
    def nonfinite(_):
        raise AcquisitionError('nonstandard_json_constant')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    except AcquisitionError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise AcquisitionError('invalid_json') from None


def secret_present(value, secret):
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(secret_present(k, secret) or secret_present(v, secret) for k, v in value.items())
    if isinstance(value, list):
        return any(secret_present(x, secret) for x in value)
    return False


def metadata_screen(obj):
    require(isinstance(obj, dict), 'metadata_not_object')
    require(obj.get('ticker') == SYMBOL, 'metadata_ticker_mismatch')
    name = obj.get('name')
    require(isinstance(name, str) and 'vanguard' in name.casefold() and
            ('communication' in name.casefold() or 'telecommunication' in name.casefold()),
            'metadata_name_mismatch')
    exchange = obj.get('exchangeCode')
    require(isinstance(exchange, str) and bool(exchange.strip()), 'metadata_exchange_missing')
    for key in ('startDate', 'endDate'):
        value = obj.get(key)
        require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is not None,
                'metadata_history_date_invalid')
        try:
            date.fromisoformat(value)
        except ValueError:
            raise AcquisitionError('metadata_history_date_invalid') from None
    require(obj['startDate'] <= START and obj['endDate'] >= END, 'metadata_history_insufficient')
    return {'ticker': SYMBOL, 'name': name, 'exchangeCode': exchange,
            'startDate': obj['startDate'], 'endDate': obj['endDate'],
            'identity_history_screen_passed': True, 'exchange_listing_certified': False,
            'history_rows_verified': False, 'source_admitted': False}


def prices_screen(obj):
    require(isinstance(obj, list), 'prices_not_array')
    require(bool(obj), 'prices_empty_array')
    require(all(isinstance(row, dict) and 'adjClose' in row for row in obj), 'prices_adjclose_missing')
    return {'rows_acquired': len(obj), 'adjClose_key_present_in_each_row': True,
            'numeric_values_verified': False, 'dates_unique_verified': False,
            'date_window_verified': False, 'session_coverage_verified': False,
            'adjustment_vintage_verified': False, 'source_admitted': False}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_once(req, timeout):
    # Empty ProxyHandler avoids environment proxy routing; no retry-capable wrapper.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as response:
        return response


def read_keychain():
    try:
        result = subprocess.run(['/usr/bin/security', 'find-generic-password', '-a',
                                 getpass.getuser(), '-s', SERVICE, '-w'],
                                capture_output=True, text=True, timeout=30, check=False)
        require(result.returncode == 0, 'credential_unavailable')
        token = result.stdout.strip()
        require(re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', token) is not None, 'credential_format_invalid')
        return token
    except AcquisitionError:
        raise
    except Exception:
        raise AcquisitionError('credential_unavailable') from None


def retry_after_time(raw, now):
    if raw is None:
        return None
    try:
        stripped = raw.strip()
        if re.fullmatch(r'\d+', stripped):
            future = now + timedelta(seconds=int(stripped))
        else:
            future = parsedate_to_datetime(stripped)
            if future.tzinfo is None:
                return None
        return stamp(future)
    except (ValueError, TypeError, OverflowError):
        return None


def stopped(out, phase, helper_sha, reason, clock, details):
    now = clock()
    safe_headers = details.get('safe_response_headers', {})
    retry = safe_headers.get('Retry-After')
    status = details.get('http_status')
    record = {'schema': SCHEMA, 'collector_sha256': helper_sha, 'provider_host': HOST,
              'phase': phase, 'reason_code': reason, 'stopped_utc': stamp(now),
              'stopped_date_utc': now.astimezone(timezone.utc).date().isoformat(),
              'stopped_date_america_los_angeles': now.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat(),
              'http_status': status, 'safe_response_headers': safe_headers,
              'retry_after_raw': retry, 'retry_after_earliest_utc': retry_after_time(retry, now),
              'permanent_directory_stop': True, 'further_requests_permitted': False,
              'provider_stop_today_required': status == 429,
              'provider_stop_scope': 'Tiingo account/provider for the recorded day; coordination is external' if status == 429 else None,
              'other_scripts_automatically_blocked': False,
              'policy': 'No retry or redirect. This directory remains stopped even after Retry-After. No shared ledger is modified.',
              'request': descriptor(phase), 'source_admitted': False,
              'raw_response_sha256': details.get('raw_response_sha256')}
    if not exists(out / 'STOPPED.json'):
        write_new(out / 'STOPPED.json', record)


def fetch(req, request, secret, details):
    response = request(req, TIMEOUT)
    try:
        require(response.geturl() == req.full_url, 'redirect_forbidden')
        status = response.getcode()
        require(type(status) is int, 'http_status_invalid')
        details['http_status'] = status
        headers = {name: response.headers.get(name) for name in SAFE_HEADERS
                   if response.headers.get(name) is not None}
        require(all(isinstance(v, str) for v in headers.values()), 'header_type_invalid')
        require(not secret_present(headers, secret), 'credential_echo_in_headers')
        details['safe_response_headers'] = headers
        require(status == 200, 'http_status_not_200')
        length = headers.get('Content-Length')
        require(length is None or length.isdigit() and int(length) <= MAX_BYTES,
                'response_length_invalid')
        require(headers.get('Content-Encoding', 'identity').casefold() in ('identity', ''),
                'response_encoding_unsupported')
        body = bytearray()
        while True:
            chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)))
            require(isinstance(chunk, bytes), 'response_chunk_invalid')
            if not chunk:
                break
            body.extend(chunk)
            require(len(body) <= MAX_BYTES, 'response_too_large')
        require(length is None or len(body) == int(length), 'response_length_mismatch')
        raw = bytes(body)
        require(secret.encode('utf-8') not in raw, 'credential_echo_in_body')
        obj = parse_json(raw)
        require(not secret_present(obj, secret), 'credential_echo_in_body')
        return raw, obj
    finally:
        response.close()


def binding(phase, helper_sha, metadata_sha):
    return dict(descriptor(phase), collector_sha256=helper_sha,
                operator_metadata_review_sha256=metadata_sha)


def verify_cache(out, phase, helper_sha, metadata_sha):
    expected = binding(phase, helper_sha, metadata_sha)
    attempt_raw = read_bytes(out / (phase + '.attempt.json'))
    attempt = parse_json(attempt_raw)
    require(isinstance(attempt, dict) and attempt.get('binding') == expected,
            'cache_attempt_binding_mismatch')
    raw = read_bytes(out / (phase + '.raw.json'))
    receipt_raw = read_bytes(out / (phase + '.receipt.json'))
    receipt = parse_json(receipt_raw)
    require(isinstance(receipt, dict) and receipt.get('binding') == expected,
            'cache_receipt_binding_mismatch')
    require(receipt.get('attempt_sha256') == sha(attempt_raw) and
            receipt.get('raw_sha256') == sha(raw) and receipt.get('raw_bytes') == len(raw),
            'cache_digest_mismatch')
    obj = parse_json(raw)
    screen = metadata_screen(obj) if phase == 'metadata' else prices_screen(obj)
    require(receipt.get('screen') == screen and receipt.get('outcome') == 'ACQUIRED_NOT_ADMITTED' and
            receipt.get('http_status') == 200 and receipt.get('request_count') == 1 and
            receipt.get('source_admitted') is False, 'cache_receipt_invalid')
    return {'phase': phase, 'outcome': 'CACHE_VERIFIED_NOT_ADMITTED',
            'raw_sha256': sha(raw), 'receipt_sha256': sha(receipt_raw),
            'screen': screen, 'http_requests_this_invocation': 0, 'source_admitted': False}


def collect(phase, *, expected_sha256, metadata_sha256=None, output=OUT,
            keychain=None, request=None, clock=None):
    """Dependency injection is for offline tests; CLI fixes output and all targets."""
    descriptor(phase)
    helper_sha = sha(read_bytes(Path(__file__).resolve()))
    require(expected_sha256 == helper_sha, 'collector_sha256_mismatch')
    require(phase != 'prices' or isinstance(metadata_sha256, str) and
            re.fullmatch(r'[0-9a-f]{64}', metadata_sha256) is not None, 'metadata_review_sha256_required')
    require(phase != 'metadata' or metadata_sha256 is None, 'unexpected_metadata_review_sha256')
    keychain, request, clock = keychain or read_keychain, request or request_once, clock or utc
    out = safe_directory(output)
    created = False
    try:
        out.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        require(out.is_dir(), 'output_not_directory')
    lock = out / '.lock'
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise AcquisitionError('directory_locked_no_request') from None
    details = {}
    try:
        require(not exists(out / 'STOPPED.json'), 'directory_permanently_stopped')
        if created:
            write_new(out / 'SCOPE.json', scope(helper_sha))
        require(parse_json(read_bytes(out / 'SCOPE.json')) == scope(helper_sha), 'scope_mismatch')
        allowed = {'SCOPE.json', '.lock'} | {p + suffix for p in PHASES
                  for suffix in ('.attempt.json', '.raw.json', '.receipt.json')}
        require(set(p.name for p in out.iterdir()) <= allowed, 'unexpected_directory_member')
        for p in PHASES:
            present = [exists(out / (p + suffix)) for suffix in ('.attempt.json', '.raw.json', '.receipt.json')]
            require(not any(present) or all(present), 'partial_phase_permanent_stop')
        cached_metadata = None
        if exists(out / 'metadata.receipt.json'):
            cached_metadata = verify_cache(out, 'metadata', helper_sha, None)
        if phase == 'prices':
            require(cached_metadata is not None, 'metadata_success_required')
            require(cached_metadata['raw_sha256'] == metadata_sha256, 'metadata_review_sha256_mismatch')
        if exists(out / 'prices.receipt.json'):
            require(cached_metadata is not None, 'metadata_success_required')
            cached_prices = verify_cache(out, 'prices', helper_sha, cached_metadata['raw_sha256'])
            if phase == 'prices':
                return cached_prices
        if phase == 'metadata' and cached_metadata is not None:
            return cached_metadata
        started = stamp(clock())
        fixed = binding(phase, helper_sha, metadata_sha256)
        attempt_sha = write_new(out / (phase + '.attempt.json'), {
            'schema': SCHEMA, 'binding': fixed, 'attempt_started_utc': started,
            'attempt_budget_consumed': True, 'http_gets_maximum': 1,
            'policy': 'Any incomplete attempt permanently blocks further requests.'})
        # Persist the attempt before credentials or HTTP; never persist the token.
        secret = keychain()
        require(isinstance(secret, str) and re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', secret) is not None,
                'credential_format_invalid')
        req = urllib.request.Request(fixed['request_url'], method='GET', headers={
            'Authorization': 'Token ' + secret, 'Accept': 'application/json',
            'Accept-Encoding': 'identity', 'User-Agent': 'LocalBoundedSourceResearch/238'})
        raw, obj = fetch(req, request, secret, details)
        details['raw_response_sha256'] = write_new(out / (phase + '.raw.json'), raw)
        screen = metadata_screen(obj) if phase == 'metadata' else prices_screen(obj)
        receipt = {'schema': SCHEMA, 'binding': fixed, 'attempt_sha256': attempt_sha,
                   'request_started_utc': started, 'request_finished_utc': stamp(clock()),
                   'outcome': 'ACQUIRED_NOT_ADMITTED', 'http_status': 200,
                   'safe_response_headers': details['safe_response_headers'],
                   'raw_sha256': sha(raw), 'raw_bytes': len(raw), 'request_count': 1,
                   'redirects_followed': 0, 'screen': screen, 'source_admitted': False,
                   'credential_persisted': False, 'features_computed': False,
                   'orders_computed': False, 'returns_computed': False}
        receipt_sha = write_new(out / (phase + '.receipt.json'), receipt)
        return {'phase': phase, 'outcome': receipt['outcome'], 'raw_sha256': sha(raw),
                'receipt_sha256': receipt_sha, 'screen': screen,
                'http_requests_this_invocation': 1, 'source_admitted': False}
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, AcquisitionError) else 'acquisition_failed'
        # No exception dumps, request repr, Keychain output or HTTP error body.
        try:
            stopped(out, phase, helper_sha, reason, clock, details)
        except BaseException:
            # The attempt and/or incomplete directory remains fail-closed even if
            # the filesystem cannot persist STOP. Never try HTTP again here.
            raise AcquisitionError('stop_persistence_failed_directory_unusable') from None
        raise AcquisitionError(reason) from None
    finally:
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    for phase in PHASES:
        sub = commands.add_parser(phase)
        sub.add_argument('--script-sha256', required=True)
        if phase == 'prices':
            sub.add_argument('--metadata-sha256', required=True,
                             help='Exact successful raw metadata SHA reviewed before this separate command')
    args = parser.parse_args(argv)
    try:
        result = collect(args.phase, expected_sha256=args.script_sha256,
                         metadata_sha256=getattr(args, 'metadata_sha256', None))
    except AcquisitionError as exc:
        print(json.dumps({'outcome': 'STOPPED_OR_BLOCKED', 'reason_code': str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
