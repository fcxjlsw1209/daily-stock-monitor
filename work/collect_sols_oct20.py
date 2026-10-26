"""Bounded two-URL Nasdaq source acquisition and separate offline metadata review.

No strategy, performance calculation, security admission, retries or redirects.
Run --test-only/--review with run_local.sh; --collect is separate acquisition.
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
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/collect_sols_oct20.py'
OUT = 'outputs/index_union/sources/oct20_sols_availability'
TARGETS = [
 ('highlights', 'https://www.nasdaq.com/articles/global-indexes/2025-nasdaq-100-reconstitution-and-performance-highlights'),
 ('annual', 'https://www.nasdaq.com/2025-Nasdaq-100-Annual-Reconstitution'),
]
CONTEXT = {
 'outputs/index_union/sources/oct08_validation/ndx_2025_reconstitution.pdf': '74f20d5f11bbaca127454e9902a59dc9021e1e93803b576e8cdead2710233dd4',
 'outputs/index_union/sources/oct08_validation/ndx_2025_reconstitution.provenance.json': '9d5e30599d170fa2f966ee24850c54fa8798640213839b8caf2e3dedc91e3cef',
}
MAX_BYTES = 8 * 1024 * 1024
TIMEOUT = 35

def require(ok, text):
    if not ok:
        raise ValueError(text)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def local(relative):
    parts = relative.split('/')
    require(all(x not in ('', '.', '..') for x in parts), 'Invalid path')
    p = ROOT
    for part in parts:
        require('onedrive' not in part.lower() and 'cloudstorage' not in part.lower(), 'Cloud path')
        p /= part
        require(not p.is_symlink(), 'Symlink forbidden')
    require(p.resolve().is_relative_to(ROOT.resolve()), 'Path escape')
    return p

def read(path, expected=None):
    require(not path.is_symlink(), 'Symlink input')
    raw = path.read_bytes()
    require(expected is None or sha(raw) == expected, 'Hash mismatch: ' + str(path))
    return raw

def write(path, obj):
    raw = obj if isinstance(obj, bytes) else (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as h:
        h.write(raw)
        h.flush()
        os.fsync(h.fileno())
    return sha(raw)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def request_once(req, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e

def fetch(url, request):
    require(url in [u for _, u in TARGETS], 'Only fixed source URLs permitted')
    r = dict(source_url=url, final_url=url, request_started_utc=utc(), retrieved_utc=None,
             http_status=None, headers={}, outcome='request_error', stop_source=True,
             request_count=1, redirects_followed=0)
    body = bytearray()
    try:
        response = request(urllib.request.Request(url, headers={
            'User-Agent': 'LocalSourceResearch/1.0', 'Accept-Encoding': 'identity', 'Accept': 'text/html'}), TIMEOUT)
        try:
            require(response.geturl() == url, 'Implicit redirect forbidden')
            r['http_status'] = response.getcode()
            for k in ('Date', 'Last-Modified', 'Content-Type', 'Content-Length', 'Content-Encoding', 'Retry-After', 'Location', 'ETag', 'Cache-Control', 'Server'):
                if response.headers.get(k) is not None:
                    r['headers'][k] = response.headers.get(k)
            length = response.headers.get('Content-Length')
            require(length is None or length.isdigit() and int(length) <= MAX_BYTES, 'Invalid or excessive length')
            while True:
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                require(len(body) <= MAX_BYTES, 'Maximum bytes exceeded')
            require(length is None or len(body) == int(length), 'Incomplete response')
            r['outcome'] = 'success' if r['http_status'] == 200 and body else 'http_error'
            r['stop_source'] = r['outcome'] != 'success'
        finally:
            response.close()
    except Exception as e:
        r.update(error_type=type(e).__name__, error=str(e), outcome='request_error', stop_source=True)
    r.update(retrieved_utc=utc(), body_truncated_at_limit=len(body) > MAX_BYTES)
    return bytes(body[:MAX_BYTES]), r

def collect_targets(out, helper_sha, request):
    stopped = None
    records = []
    for key, url in TARGETS:
        r = dict(key=key, source_url=url, helper_sha256=helper_sha)
        if stopped:
            r.update(attempted=False, outcome='not_attempted_source_stopped', request_count=0,
                     failed_target=stopped, recorded_utc=utc())
        else:
            attempt_sha = write(out / (key + '.attempt.json'), dict(r, request_started_utc=utc()))
            body, details = fetch(url, request)
            r.update(details, attempted=True, attempt_sha256=attempt_sha, bytes=len(body), sha256=sha(body))
            write(out / (key + '.html'), body)
        receipt_sha = write(out / (key + '.receipt.json'), r)
        records.append(dict(key=key, outcome=r['outcome'], receipt_sha256=receipt_sha))
        if r.get('stop_source') and not stopped:
            stopped = key
            write(out / 'STOPPED.json', dict(host='www.nasdaq.com', failed_target=key,
                  stopped_utc=r['retrieved_utc'], http_status=r['http_status'],
                  retry_after=r['headers'].get('Retry-After'), receipt_sha256=receipt_sha,
                  further_requests_permitted=False,
                  policy='No retries; all remaining URLs on this source are skipped. Retry-After is retained without early retry.'))
    result = dict(schema='sols_oct20_source_collection_v1', helper_sha256=helper_sha,
                  records=records, all_successful=all(x['outcome'] == 'success' for x in records),
                  source_admission_changed=False, identity_connection_admitted=False)
    write(out / 'COLLECTION.json', result)
    return result

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hide = 0
        self.visible, self.meta, self.times, self.json_scripts = [], [], [], []
        self.in_ld, self.ld = False, []
        self.in_row, self.row = False, []
        self.rows = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ('script', 'style', 'noscript'):
            self.hide += 1
        if tag == 'script' and d.get('type') == 'application/ld+json':
            self.in_ld, self.ld = True, []
        if tag == 'meta':
            name = d.get('property') or d.get('name') or d.get('itemprop') or ''
            if any(x in name.lower() for x in ('date', 'publish', 'modif', 'creat', 'title')):
                self.meta.append(dict(name=name, content=d.get('content'), line=self.getpos()[0]))
        if tag == 'time':
            self.times.append(dict(attributes=d, line=self.getpos()[0]))
        if tag == 'tr':
            self.in_row, self.row = True, []
    def handle_endtag(self, tag):
        if tag == 'script' and self.in_ld:
            self.json_scripts.append(''.join(self.ld))
            self.in_ld = False
        if tag in ('script', 'style', 'noscript'):
            self.hide = max(0, self.hide - 1)
        if tag == 'tr' and self.in_row:
            self.rows.append(' '.join(self.row))
            self.in_row = False
    def handle_data(self, s):
        if self.in_ld:
            self.ld.append(s)
        if not self.hide:
            self.visible.append(s)
            if self.in_row:
                self.row.append(s)

def dates(obj, path='$'):
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ('datepublished', 'datemodified', 'datecreated'):
                result.append(dict(path=path + '.' + k, value=v))
            result.extend(dates(v, path + '.' + k))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            result.extend(dates(v, path + '[%d]' % idx))
    return result

def parse(raw):
    p = Parser()
    p.feed(raw.decode('utf-8', errors='replace'))
    text = re.sub(r'\s+', ' ', ' '.join(p.visible)).strip()
    fields, malformed = [], []
    for idx, script in enumerate(p.json_scripts):
        try:
            fields.extend(dates(json.loads(script), '$ld[%d]' % idx))
        except ValueError:
            malformed.append(idx)
    hits = [dict(symbol=m.group(), offset=m.start(), literal_context=text[max(0, m.start()-180):m.end()+220])
            for m in re.finditer(r'\bSOLS(?:V)?\b', text)]
    rows = [re.sub(r'\s+', ' ', x).strip() for x in p.rows if re.search(r'\bSOLS(?:V)?\b', x)]
    pub = [m.group() for m in re.finditer(r'(?:Published|Updated|Last updated)\s*.{0,120}', text, re.I)]
    return dict(date_metadata=p.meta, time_elements=p.times, structured_date_fields=fields,
                malformed_ld_json=malformed, visible_publication_label_contexts=pub,
                symbol_literal_contexts=hits, symbol_table_rows=rows,
                both_symbols_present=all(re.search(r'\b' + x + r'\b', text) for x in ('SOLS', 'SOLSV')),
                identity_connection_admitted=False,
                interpretation='Literal source fields only. Modified/created/HTTP dates are not publication times; co-occurrence or coincident dates do not establish when-issued identity.')

def review(helper_sha):
    read(local(SELF), helper_sha)
    out = local(OUT)
    coll_raw = read(out / 'COLLECTION.json')
    coll = json.loads(coll_raw)
    require(coll['helper_sha256'] == helper_sha, 'Helper binding')
    pins = {SELF: helper_sha, OUT + '/COLLECTION.json': sha(coll_raw), **CONTEXT}
    records = []
    for (key, url), item in zip(TARGETS, coll['records']):
        require(item['key'] == key, 'Target order')
        name = OUT + '/' + key + '.receipt.json'
        r = json.loads(read(local(name), item['receipt_sha256']))
        pins[name] = item['receipt_sha256']
        require(r['source_url'] == url and r['helper_sha256'] == helper_sha, 'Receipt identity')
        record = dict(key=key, source_url=url, outcome=r['outcome'])
        if r['attempted']:
            path = OUT + '/' + key + '.html'
            pins[path] = r['sha256']
            pins[OUT + '/' + key + '.attempt.json'] = r['attempt_sha256']
            raw = read(local(path), r['sha256'])
            require(len(raw) == r['bytes'], 'Body size')
            record.update(http_status=r['http_status'], retrieved_utc=r['retrieved_utc'], headers=r['headers'], bytes=r['bytes'], sha256=r['sha256'])
            if r['outcome'] == 'success':
                record['literal_metadata'] = parse(raw)
        records.append(record)
    for path, digest in pins.items():
        read(local(path), digest)
    result = dict(schema='sols_oct20_literal_source_review_v1', created_utc=utc(), input_hashes=pins,
                  records=records, source_admission_changed=False, identity_connection_admitted=False,
                  existing_pdf_not_downloaded=True, financial_or_performance_calculations=0,
                  scope='Two fixed Nasdaq pages; exact source acquisition and literal metadata only. Current retrieval is not contemporaneous historical capture.')
    digest = write(out / 'REVIEW.json', result)
    print(json.dumps(dict(path=OUT + '/REVIEW.json', sha256=digest, records=records)))

class Response(io.BytesIO):
    def __init__(self, url, code, body, headers=None):
        super().__init__(body)
        self.url, self.code, self.headers = url, code, headers or {}
    def geturl(self): return self.url
    def getcode(self): return self.code

class Tests(unittest.TestCase):
    def test_429_stops_second_and_retains_retry_after(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def req(r, timeout):
                calls.append(r.full_url)
                return Response(r.full_url, 429, b'limited', {'Retry-After': '3600'})
            result = collect_targets(Path(d), 'h', req)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['records'][1]['outcome'], 'not_attempted_source_stopped')
            self.assertEqual(json.loads((Path(d)/'STOPPED.json').read_bytes())['retry_after'], '3600')
    def test_redirect_not_followed(self):
        calls = []
        def req(r, timeout):
            calls.append(r.full_url)
            return Response(r.full_url, 302, b'redirect', {'Location':'https://example.com'})
        _, r = fetch(TARGETS[0][1], req)
        self.assertTrue(r['stop_source'])
        self.assertEqual(len(calls), 1)
    def test_literal_cooccurrence_never_admits_identity(self):
        r = parse(b'<table><tr><td>SOLSV</td><td>2025-10-20</td></tr><tr><td>SOLS</td><td>2025-10-20</td></tr></table>')
        self.assertTrue(r['both_symbols_present'])
        self.assertFalse(r['identity_connection_admitted'])
        self.assertEqual(len(r['symbol_table_rows']), 2)
    def test_date_fields_keep_different_semantics(self):
        r = parse(b'<meta name="dateModified" content="2026-10-20"><script type="application/ld+json">{"datePublished":"2026-01-05","dateCreated":"2025-12-30"}</script>')
        self.assertEqual(r['date_metadata'][0]['name'], 'dateModified')
        self.assertEqual(len(r['structured_date_fields']), 2)
    def test_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x'
            write(p, b'a')
            with self.assertRaises(FileExistsError): write(p, b'b')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-only', action='store_true')
    p.add_argument('--collect', action='store_true')
    p.add_argument('--review', action='store_true')
    p.add_argument('--script-sha256')
    args = p.parse_args()
    if args.test_only:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if r.wasSuccessful() else 1
    require(args.script_sha256, 'Script SHA required')
    read(local(SELF), args.script_sha256)
    if args.review:
        review(args.script_sha256)
        return 0
    require(args.collect, 'Choose one mode')
    for path, digest in CONTEXT.items():
        read(local(path), digest)
    out = local(OUT)
    require(not out.exists(), 'Collection directory exists; no overwrite/retry')
    out.mkdir()
    write(out/'SCOPE.json', dict(targets=TARGETS, helper_sha256=args.script_sha256,
          max_bytes=MAX_BYTES, timeout_seconds=TIMEOUT, no_redirects=True, no_retries=True,
          existing_context=CONTEXT, checked_stops='Scoped index_union STATUS/queue/recent FOLLOWUP checks and root confirmation: no current Nasdaq429 known.'))
    r = collect_targets(out, args.script_sha256, request_once)
    print(json.dumps(dict(path=OUT+'/COLLECTION.json', sha256=sha(read(out/'COLLECTION.json')), **r)))
    return 0 if r['all_successful'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
