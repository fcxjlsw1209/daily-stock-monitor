"""Five fixed public Boeing pages: separate collection, then offline metadata parsing.

No prices, factors, return calculations or source admission. No redirects/retries.
Tests and scope preparation run through run_local.sh. --collect is a separate
explicitly authorized acquisition command using the same local Python binary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_ba_publication_collect.py'
OUT = 'outputs/after158_ba_publication_v1'
SCHEMA = 'after158_ba_publication_collection_v1'
MAX_BYTES = 8 * 1024 * 1024
TIMEOUT = 35
HOST = 'boeing.mediaroom.com'
URLS = [
 'https://boeing.mediaroom.com/2015-01-28-Boeing-Reports-Record-2014-Revenue-Core-EPS-and-Backlog-and-Provides-2015-Guidance',
 'https://boeing.mediaroom.com/2016-01-27-Boeing-Reports-Fourth-Quarter-Results-and-Provides-2016-Guidance',
 'https://boeing.mediaroom.com/2017-01-25-Boeing-Reports-Fourth-Quarter-Results-and-Provides-2017-Guidance',
 'https://boeing.mediaroom.com/2018-01-31-Boeing-Reports-Record-2017-Results-and-Provides-2018-Guidance',
 'https://boeing.mediaroom.com/2019-01-30-Boeing-Reports-Record-2018-Results-and-Provides-2019-Guidance',
]
FIXED = {
 'work/after158_ba_source_plan.json': '4b7067b5a0e60aece954ba6617d4aa050375d323c6d13fd3899eabdadba7f422',
 'work/after158_ba_original_statement_review.json': '3c95f49ab51407e0b089638d2301f32520e3b1d35a227d9e7f1bd8dd7841e47d',
 'work/after158_ba_independent_review.json': 'a26da326feafc899773e8c17897b7d6fee91c17f68614d720ab0cff988f84b74',
 'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/BA/415de853ff772b56c253.bin': '5171d32b01c90b751bf68496a26b082511a0287e153d46ca5c0cde986026da06',
}
STOP_PATHS = [
 'outputs/monthly_earnings_repair/batch_health_energy_foreign/SOURCE_STOPS.json',
 'outputs/monthly_earnings_repair/ROUND3_SOURCE_STOPS.json',
 'outputs/monthly_earnings_repair/warmup/SOURCE_STOPS.json',
 'outputs/after157_msft_iframe_v1/STOPPED.json',
 'outputs/monthly_aapl_annual_notes_after156_source_v1/STOPPED_publisher.json',
 'outputs/monthly_event_tsla_wire156/STOPPED_2026-10-20.json',
 'work/after155_sec_gap_discovery_collection.json',
]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def local(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative, 'Invalid path')
    parts = relative.split('/')
    require(all(p not in ('', '.', '..') for p in parts), 'Path escape')
    p = Path(root)
    for part in parts:
        require('onedrive' not in part.lower() and 'cloudstorage' not in part.lower(), 'Cloud path')
        p = p / part
        require(not p.is_symlink(), 'Symlink forbidden')
    require(p.resolve().is_relative_to(Path(root).resolve()), 'Path escape')
    return p

def read(path, expected=None):
    require(not path.is_symlink(), 'Symlink input')
    a = path.stat()
    data = path.read_bytes()
    b = path.stat()
    require((a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_ino, b.st_size, b.st_mtime_ns), 'Changed while reading')
    require(expected is None or sha(data) == expected, 'Hash mismatch: ' + str(path))
    return data

def write(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as h:
        h.write(data)
        h.flush()
        os.fsync(h.fileno())
    return sha(data)

def valid_url(url):
    require(url in URLS, 'Only the five fixed original URLs are authorized')
    p = urllib.parse.urlsplit(url)
    require(p.scheme == 'https' and p.hostname == HOST and not p.username and not p.password and not p.fragment, 'URL scope')

def verify_inputs(scope):
    for name, digest in scope['input_hashes'].items():
        raw = read(local(ROOT, name), digest)
        if name in STOP_PATHS:
            require(HOST not in raw.decode('utf-8').lower(), 'Source stop found; manual review required')

def freeze_scope(script_sha):
    read(local(ROOT, SELF), script_sha)
    pins = dict(FIXED)
    pins[SELF] = script_sha
    receipt = 'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/BA/415de853ff772b56c253.meta.json'
    for name in STOP_PATHS + [receipt]:
        pins[name] = sha(read(local(ROOT, name)))
    plan = json.loads(read(local(ROOT, 'work/after158_ba_source_plan.json'), FIXED['work/after158_ba_source_plan.json']))
    pins.update(plan['source_hashes'])
    targets = []
    for idx, target in enumerate(plan['targets']):
        require(target['fiscal_year'] == 2014 + idx, 'Plan target order')
        targets.append(dict(id='FY%d' % target['fiscal_year'], fiscal_year=target['fiscal_year'], url=URLS[idx],
                            expected_dateline=target['catalog_publication_date'],
                            original_pdf_url=target['url'], original_pdf_path=target['paths']['raw'],
                            original_pdf_sha256=plan['source_hashes'][target['paths']['raw']]))
    scope = dict(schema=SCHEMA, created_utc=utc(), helper_sha256=script_sha, input_hashes=pins,
                 output_dir=OUT, host=HOST, targets=targets, max_bytes=MAX_BYTES, timeout_seconds=TIMEOUT,
                 redirects_allowed=False, retries_allowed=False,
                 purpose='Official dated release page metadata and explicit links only; no source admission.',
                 cache_search='No exact target URL found in the bounded existing BA raw cache and candidate metadata. This is not an exhaustive cache absence claim.',
                 source_stops_checked=STOP_PATHS,
                 archival_limit='New current acquisition, never a contemporaneous archive capture.',
                 source_admission_changed=False, new_strategy_trials=0)
    verify_inputs(scope)
    out = local(ROOT, OUT)
    require(not out.exists(), 'Output directory already exists; never replace scope')
    out.mkdir()
    digest = write(out / 'SCOPE.json', scope)
    print(json.dumps(dict(path=OUT + '/SCOPE.json', sha256=digest, helper_sha256=script_sha)))

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def transport(req, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as response:
        return response

def fetch(url, request):
    valid_url(url)
    result = dict(source_url=url, final_url=url, request_started_utc=utc(), retrieved_utc=None,
                  http_status=None, headers={}, outcome='request_error', stop_source=True, redirects_followed=0)
    body = bytearray()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'LocalSourceResearch/1.0', 'Accept': 'text/html', 'Accept-Encoding': 'identity'})
        response = request(req, timeout=TIMEOUT)
        try:
            require(response.geturl() == url, 'Implicit redirect forbidden')
            result['http_status'] = response.getcode()
            for key in ('Date', 'Last-Modified', 'Content-Type', 'Content-Length', 'Content-Encoding', 'ETag', 'Cache-Control', 'Location', 'Retry-After', 'Server'):
                if response.headers.get(key) is not None:
                    result['headers'][key] = response.headers.get(key)
            length = response.headers.get('Content-Length')
            require(length is None or length.isdigit() and int(length) <= MAX_BYTES, 'Invalid/oversize Content-Length')
            while True:
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                require(len(body) <= MAX_BYTES, 'Body exceeds size bound')
            require(length is None or int(length) == len(body), 'Incomplete body')
            require(response.headers.get('Content-Encoding', 'identity').lower() in ('identity', ''), 'Unexpected content encoding')
            result['outcome'] = 'success' if response.getcode() == 200 and body else 'http_error'
            result['stop_source'] = result['outcome'] != 'success'
        finally:
            response.close()
    except Exception as e:
        result.update(outcome='request_error', stop_source=True, error_type=type(e).__name__, error=str(e))
    result['retrieved_utc'] = utc()
    result['body_truncated_at_limit'] = len(body) > MAX_BYTES
    return bytes(body[:MAX_BYTES]), result

def binding(target, scope_sha, helper_sha):
    return dict(schema=SCHEMA, target_id=target['id'], source_url=target['url'], scope_sha256=scope_sha, helper_sha256=helper_sha)

def verify_existing(out, target, scope_sha, helper_sha):
    key = target['id']
    receipt_path, body_path, attempt_path = [local(out, key + suffix) for suffix in ('.receipt.json', '.html', '.attempt.json')]
    if not receipt_path.exists():
        require(not body_path.exists() and not attempt_path.exists(), 'Interrupted attempt cannot be retried: ' + key)
        return None
    r = json.loads(read(receipt_path))
    require(all(r.get(k) == v for k, v in binding(target, scope_sha, helper_sha).items()), 'Receipt identity')
    if r['attempted']:
        read(body_path, r['body_sha256'])
        require(body_path.stat().st_size == r['bytes'], 'Body size')
        attempt = json.loads(read(attempt_path, r['attempt_sha256']))
        require(all(attempt.get(k) == v for k, v in binding(target, scope_sha, helper_sha).items()), 'Attempt identity')
        if r['outcome'] == 'success':
            require(r['http_status'] == 200 and r['final_url'] == target['url'] and not r['stop_source'], 'Success receipt malformed')
    else:
        require(r['outcome'] == 'skipped_source_stopped' and not body_path.exists() and not attempt_path.exists(), 'Skipped receipt malformed')
    return r

def collect_core(out, scope, scope_sha, helper_sha, request, verify):
    old = {t['id']: verify_existing(out, t, scope_sha, helper_sha) for t in scope['targets']}
    failed = next((r for r in old.values() if r and r['attempted'] and r['stop_source']), None)
    stop_path = local(out, 'STOPPED.json')
    if stop_path.exists():
        stop = json.loads(read(stop_path))
        require(failed and stop['failed_target'] == failed['target_id'], 'Stop has no failed receipt')
        read(local(out, failed['target_id'] + '.receipt.json'), stop['receipt_sha256'])
    receipts = []
    for target in scope['targets']:
        verify(scope)
        key = target['id']
        r = old[key]
        if r is None:
            r = binding(target, scope_sha, helper_sha)
            if failed:
                r.update(attempted=False, outcome='skipped_source_stopped', stop_source=True,
                         failed_target=failed['target_id'], recorded_utc=utc())
            else:
                attempt_sha = write(local(out, key + '.attempt.json'), dict(r, attempt_started_utc=utc()))
                body, info = fetch(target['url'], request)
                r.update(info, attempted=True, bytes=len(body), body_sha256=sha(body), attempt_sha256=attempt_sha)
                write(local(out, key + '.html'), body)
            write(local(out, key + '.receipt.json'), r)
        if r['attempted'] and r['stop_source'] and not failed:
            failed = r
        if failed and not stop_path.exists():
            write(stop_path, dict(schema=SCHEMA, source=HOST, failed_target=failed['target_id'],
                                  http_status=failed['http_status'], stopped_utc=failed['retrieved_utc'],
                                  retry_after=failed['headers'].get('Retry-After'), further_requests_permitted=False,
                                  receipt_sha256=sha(read(local(out, failed['target_id'] + '.receipt.json')))))
        receipts.append(dict(target_id=key, outcome=r['outcome'], attempted=r['attempted'],
                             receipt_file=key + '.receipt.json', receipt_sha256=sha(read(local(out, key + '.receipt.json')))))
    verify(scope)
    result = dict(schema=SCHEMA, scope_sha256=scope_sha, helper_sha256=helper_sha, targets=receipts,
                  all_successful=all(r['outcome'] == 'success' for r in receipts),
                  source_admission_changed=False, new_strategy_trials=0)
    path = local(out, 'COLLECTION.json')
    if path.exists():
        require(json.loads(read(path)) == result, 'Summary changed')
    else:
        write(path, result)
    return result

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.visible = []
        self.metadata = []
        self.links = []
        self.scripts = []
        self.in_ld = False
        self.ld_parts = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ('script', 'style', 'noscript'):
            self.hidden += 1
        if tag == 'script' and d.get('type', '').lower() == 'application/ld+json':
            self.in_ld, self.ld_parts = True, []
        if tag == 'meta':
            name = d.get('property') or d.get('name') or d.get('itemprop') or ''
            if any(word in name.lower() for word in ('date', 'publish', 'modif', 'creat', 'title')):
                self.metadata.append(dict(name=name, content=d.get('content'), line=self.getpos()[0]))
        if tag == 'time':
            self.metadata.append(dict(name='time_element', attributes=d, line=self.getpos()[0]))
        if tag == 'a' and d.get('href'):
            self.links.append(dict(href=d['href'], line=self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag == 'script' and self.in_ld:
            self.scripts.append(''.join(self.ld_parts))
            self.in_ld = False
        if tag in ('script', 'style', 'noscript'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, text):
        if self.in_ld:
            self.ld_parts.append(text)
        if not self.hidden:
            self.visible.append(text)

def structured_dates(obj, path='$'):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ('datepublished', 'datemodified', 'datecreated', 'uploaddate'):
                out.append(dict(path=path + '.' + k, name=k, value=v))
            out.extend(structured_dates(v, path + '.' + k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(structured_dates(v, path + '[%d]' % i))
    return out

def parse_page(raw, url):
    parser = PageParser()
    parser.feed(raw.decode('utf-8', errors='replace'))
    visible = re.sub(r'\s+', ' ', ' '.join(parser.visible)).strip()
    dates = []
    pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+20\d{2}\b'
    for m in re.finditer(pattern, visible, re.I):
        dates.append(dict(text=m.group(), offset=m.start(), context=visible[max(0, m.start() - 90):m.end() + 170]))
    ld, errors = [], []
    for i, script in enumerate(parser.scripts):
        try:
            ld.extend(structured_dates(json.loads(script), '$ld[%d]' % i))
        except ValueError:
            errors.append(i)
    pdfs = []
    for link in parser.links:
        resolved = urllib.parse.urljoin(url, link['href'])
        if '.pdf' in urllib.parse.urlsplit(resolved).path.lower():
            pdfs.append(dict(link, resolved_url=resolved))
    publication = [m for m in parser.metadata if 'publish' in m['name'].lower()]
    publication += [m for m in ld if m['name'].lower() == 'datepublished']
    return dict(visible_text=visible, visible_date_candidates=dates, date_and_title_metadata=parser.metadata,
                structured_date_fields=ld, malformed_ld_script_indices=errors, explicit_publication_metadata_candidates=publication,
                pdf_links=pdfs, source_admission_changed=False,
                interpretation='Visible dates and explicit publication-named fields are candidates for source review. Modified, created, HTTP Date and Last-Modified are not promoted to publication time. No historical capture or exact first-publication instant is certified.')

def load_scope(scope_sha, script_sha):
    scope = json.loads(read(local(ROOT, OUT + '/SCOPE.json'), scope_sha))
    read(local(ROOT, SELF), script_sha)
    require(scope['helper_sha256'] == script_sha and scope['schema'] == SCHEMA, 'Scope/helper mismatch')
    require([t['url'] for t in scope['targets']] == URLS, 'Target list changed')
    require(scope['max_bytes'] == MAX_BYTES and scope['timeout_seconds'] == TIMEOUT, 'Limits changed')
    verify_inputs(scope)
    return scope

def derive(scope_sha, script_sha):
    scope = load_scope(scope_sha, script_sha)
    out = local(ROOT, OUT)
    collection_raw = read(out / 'COLLECTION.json')
    collection = json.loads(collection_raw)
    require(collection['scope_sha256'] == scope_sha and collection['helper_sha256'] == script_sha, 'Collection binding')
    records = []
    for target, entry in zip(scope['targets'], collection['targets']):
        require(entry['target_id'] == target['id'], 'Collection order')
        read(out / entry['receipt_file'], entry['receipt_sha256'])
        r = verify_existing(out, target, scope_sha, script_sha)
        if r['outcome'] != 'success':
            records.append(dict(target_id=target['id'], status='not_parsed_unsuccessful_acquisition'))
            continue
        raw = read(out / (target['id'] + '.html'), r['body_sha256'])
        parsed = parse_page(raw, target['url'])
        parsed.update(target_id=target['id'], source_url=target['url'], raw_sha256=r['body_sha256'],
                      receipt_sha256=entry['receipt_sha256'], expected_dateline=target['expected_dateline'],
                      expected_original_pdf_url=target['original_pdf_url'],
                      expected_original_pdf_exact_link_present=any(x['resolved_url'] == target['original_pdf_url'] for x in parsed['pdf_links']))
        name = target['id'] + '.parsed.json'
        digest = write(out / name, parsed)
        records.append(dict(target_id=target['id'], path=name, sha256=digest,
                            visible_date_candidates=parsed['visible_date_candidates'],
                            explicit_publication_metadata_candidates=parsed['explicit_publication_metadata_candidates'],
                            pdf_links=parsed['pdf_links'], exact_original_pdf_link=parsed['expected_original_pdf_exact_link_present']))
    verify_inputs(scope)
    value = dict(schema='after158_ba_publication_metadata_v1', scope_sha256=scope_sha, helper_sha256=script_sha,
                 collection_sha256=sha(collection_raw), records=records, source_admission_changed=False,
                 first_publication_instant_certified=False, contemporaneous_capture=False, new_strategy_trials=0)
    digest = write(out / 'METADATA.json', value)
    print(json.dumps(dict(path=OUT + '/METADATA.json', sha256=digest, records=records)))

class FakeResponse(io.BytesIO):
    def __init__(self, url, status=200, raw=b'<html>ok</html>', headers=None):
        super().__init__(raw)
        self.url, self.status, self.headers = url, status, headers or {}
    def geturl(self):
        return self.url
    def getcode(self):
        return self.status

class Tests(unittest.TestCase):
    def test_scope_forbids_other_hosts_and_extra_paths(self):
        for u in ('https://www.sec.gov/x', URLS[0] + '?x=1', URLS[0].replace('https:', 'http:')):
            with self.assertRaises(ValueError):
                valid_url(u)

    def test_redirect_not_followed(self):
        calls = []
        def req(r, timeout):
            calls.append(r.full_url)
            return FakeResponse(r.full_url, 302, b'redirect', {'Location': 'https://www.sec.gov/x'})
        _, r = fetch(URLS[0], req)
        self.assertEqual(calls, [URLS[0]])
        self.assertTrue(r['stop_source'])

    def test_modified_and_created_are_not_publication(self):
        p = parse_page(b'<meta property="article:modified_time" content="2026-01-01"><script type="application/ld+json">{"dateCreated":"2025-01-01","dateModified":"2026-01-01"}</script><p>CHICAGO, Jan. 28, 2015</p>', URLS[0])
        self.assertEqual(p['explicit_publication_metadata_candidates'], [])
        self.assertEqual(p['visible_date_candidates'][0]['text'], 'Jan. 28, 2015')
        self.assertEqual(len(p['structured_date_fields']), 2)

    def test_explicit_publication_and_pdf_link(self):
        p = parse_page(b'<meta property="article:published_time" content="2015-01-28T07:30:00-06:00"><a href="/a.pdf">PDF</a><script type="application/ld+json">{"datePublished":"2015-01-28"}</script>', URLS[0])
        self.assertEqual(len(p['explicit_publication_metadata_candidates']), 2)
        self.assertEqual(p['pdf_links'][0]['resolved_url'], 'https://' + HOST + '/a.pdf')

    def test_error_stops_rest_and_reuses_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            calls = []
            targets = [dict(id='FY%d' % (2014+i), url=u) for i, u in enumerate(URLS)]
            scope = dict(targets=targets)
            def req(r, timeout):
                calls.append(r.full_url)
                return FakeResponse(r.full_url, 403, b'forbidden', {'Retry-After': '600'})
            first = collect_core(out, scope, 's', 'h', req, lambda s: None)
            self.assertEqual(len(calls), 1)
            self.assertFalse(first['all_successful'])
            self.assertEqual(first, collect_core(out, scope, 's', 'h', req, lambda s: None))
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads((out/'STOPPED.json').read_bytes())['retry_after'], '600')

    def test_orphan_attempt_and_tampering_fail(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            target = dict(id='FY2014', url=URLS[0])
            write(out/'FY2014.attempt.json', {})
            with self.assertRaises(ValueError):
                verify_existing(out, target, 's', 'h')
            with self.assertRaises(ValueError):
                read(out/'FY2014.attempt.json', '0'*64)
            with self.assertRaises(FileExistsError):
                write(out/'FY2014.attempt.json', {})

    def test_paths_are_confined(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ('../x', '/tmp/x', 'OneDrive/x'):
                with self.assertRaises(ValueError):
                    local(Path(d), path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-only', action='store_true')
    p.add_argument('--freeze-scope', action='store_true')
    p.add_argument('--collect', action='store_true')
    p.add_argument('--derive', action='store_true')
    p.add_argument('--script-sha256')
    p.add_argument('--scope-sha256')
    args = p.parse_args()
    if args.test_only:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if r.wasSuccessful() else 1
    require(args.script_sha256, 'Pinned script SHA required')
    if args.freeze_scope:
        freeze_scope(args.script_sha256)
        return 0
    require(args.scope_sha256, 'Pinned scope SHA required')
    if args.derive:
        derive(args.scope_sha256, args.script_sha256)
        return 0
    require(args.collect, 'Choose one mode')
    scope = load_scope(args.scope_sha256, args.script_sha256)
    out = local(ROOT, OUT)
    lock = out / '.collection.lock'
    write(lock, b'Exclusive collector\n')
    try:
        result = collect_core(out, scope, args.scope_sha256, args.script_sha256, transport, verify_inputs)
    finally:
        lock.unlink()
    print(json.dumps(dict(path=OUT + '/COLLECTION.json', sha256=sha(read(out / 'COLLECTION.json')), **result)))
    return 0 if result['all_successful'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
