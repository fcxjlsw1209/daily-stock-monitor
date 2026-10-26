"""Source-only SEC two-quarter probe; run separately from offline research.

Requires an existing, genuine SEC_USER_AGENT or EDGAR_USER_AGENT declaration.
No invented contact, browser impersonation, retry, redirect, alternate host,
factor calculation, price input, holding filter, or registration is supported.
--self-test uses only in-memory synthetic archives and temporary directories.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'outputs/monthly_form4_source_probe_v1'
LANDING = 'https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets'
TARGETS = ('2016 Q3 345', '2021 Q1 345')
ENV_NAMES = ('SEC_USER_AGENT', 'EDGAR_USER_AGENT', 'SEC_EDGAR_USER_AGENT')
PRIOR_CONFIG = 'outputs/monthly_aapl_annual_notes156/PLAN.json'
TABLES = {'SUBMISSION', 'REPORTINGOWNER', 'NONDERIV_TRANS', 'NONDERIV_HOLDING',
          'DERIV_TRANS', 'DERIV_HOLDING', 'FOOTNOTES', 'OWNER_SIGNATURE'}
DATE_FIELDS = {'FILING_DATE', 'PERIOD_OF_REPORT', 'DATE_OF_ORIG_SUB',
               'TRANS_DATE', 'DEEMED_EXECUTION_DATE'}
HEADERS = ('Content-Type', 'Content-Length', 'Date', 'Last-Modified', 'ETag', 'Retry-After')
MAX_PACKAGE = 64 * 1024 * 1024
MAX_UNPACKED = 512 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def valid_ua(value):
    if not isinstance(value, str) or not (10 <= len(value) <= 512):
        return False
    if not value.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return False
    match = re.search(r'[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', value)
    return bool(match and match.start() > 0 and match.group(1).lower() not in
                {'example.com', 'example.org', 'example.net', 'invalid.test', 'localhost.test'})


def existing_ua(root, environ):
    checks = []
    for name in ENV_NAMES:
        value = environ.get(name)
        okay = valid_ua(value)
        checks.append({'source': 'environment:' + name, 'present': value is not None,
                       'has_valid_declaration_shape': okay})
        if okay:
            return value, checks
    path = root / PRIOR_CONFIG
    value = None
    if path.is_file() and not path.is_symlink():
        try:
            value = json.loads(path.read_text())['request_policy']['user_agent']
        except (ValueError, KeyError, TypeError):
            pass
    checks.append({'source': PRIOR_CONFIG + '#/request_policy/user_agent',
                   'present': value is not None,
                   'has_valid_declaration_shape': valid_ua(value)})
    return (value if valid_ua(value) else None), checks


def official_url(url):
    parsed = urllib.parse.urlsplit(url)
    decoded_path = urllib.parse.unquote(parsed.path)
    if (parsed.scheme != 'https' or parsed.netloc != 'www.sec.gov' or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            not parsed.path.startswith('/') or '\\' in decoded_path or
            '..' in PurePosixPath(decoded_path).parts):
        raise ValueError('noncanonical_official_url')
    return url


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.href = None
        self.parts = []
        self.matches = {label: [] for label in TARGETS}

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.href = dict(attrs).get('href')
            self.parts = []

    def handle_data(self, data):
        if self.href is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'a' and self.href is not None:
            label = ' '.join(''.join(self.parts).split())
            if label in self.matches:
                raw_path = urllib.parse.unquote(urllib.parse.urlsplit(self.href).path)
                if '\\' in raw_path or '..' in PurePosixPath(raw_path).parts:
                    raise ValueError('unsafe_source_href')
                url = official_url(urllib.parse.urljoin(LANDING, self.href))
                if not urllib.parse.urlsplit(url).path.lower().endswith('.zip'):
                    raise ValueError('target_is_not_zip')
                self.matches[label].append(url)
            self.href = None
            self.parts = []


def exact_links(raw):
    parser = Links()
    parser.feed(raw.decode('utf-8-sig'))
    if any(len(values) != 1 for values in parser.matches.values()):
        raise ValueError('missing_or_ambiguous_official_quarter_link')
    result = {label: values[0] for label, values in parser.matches.items()}
    if len(set(result.values())) != 2:
        raise ValueError('quarter_links_not_distinct')
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(url, ua, out, name, limit, opener):
    official_url(url)
    receipt = {'original_url': url, 'started_at_utc': utc(), 'method': 'GET',
               'http_status': None, 'response_headers': {}, 'complete': False,
               'redirects_followed': 0, 'error_type': None}
    request = urllib.request.Request(url, headers={
        'User-Agent': ua, 'Accept-Encoding': 'identity', 'Accept': '*/*'})
    chunks = []
    size = 0
    response = None
    try:
        try:
            response = opener.open(request, timeout=60)
        except urllib.error.HTTPError as exc:
            response = exc
        receipt['http_status'] = response.code
        receipt['response_url'] = response.geturl()
        receipt['response_headers'] = {k: response.headers[k] for k in HEADERS if k in response.headers}
        read_limit = limit if response.code == 200 else min(limit, 256 * 1024)
        while size <= read_limit:
            chunk = response.read(min(65536, read_limit + 1 - size))
            if not chunk:
                receipt['complete'] = True
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > read_limit:
            receipt['error_type'] = 'response_size_limit'
        if response.code != 200:
            receipt['error_type'] = 'http_not_200'
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
    finally:
        if response is not None:
            response.close()
    raw = b''.join(chunks)
    receipt.update(finished_at_utc=utc(), bytes=len(raw), sha256=sha(raw), raw_path=name)
    with (out / name).open('xb') as stream:
        stream.write(raw)
    save_json(out / (name + '.http.json'), receipt)
    return raw, receipt


def safe_members(archive):
    members = archive.infolist()
    if not members or len(members) > 64:
        raise ValueError('zip_member_count')
    names = set()
    total = 0
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        mode = (member.external_attr >> 16) & 0o170000
        if (not name or '\\' in name or '\x00' in name or path.is_absolute() or
                '..' in path.parts or ':' in name or name in names or
                mode not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1):
            raise ValueError('unsafe_zip_member')
        names.add(name)
        total += member.file_size
        if member.file_size > MAX_UNPACKED or total > MAX_UNPACKED:
            raise ValueError('zip_uncompressed_limit')
        if member.file_size and (not member.compress_size or member.file_size / member.compress_size > 1000):
            raise ValueError('zip_ratio_limit')
    return members


def date_update(target, text):
    if not text:
        target['missing'] += 1
        return
    try:
        value = datetime.strptime(text, '%d-%b-%Y').date().isoformat()
    except ValueError:
        target['invalid'] += 1
        return
    target['valid'] += 1
    target['min'] = min(target['min'], value) if target['min'] else value
    target['max'] = max(target['max'], value) if target['max'] else value


def categorical(value, allowed):
    return value if value in allowed else ('EMPTY' if not value else 'INVALID_OR_UNRECOGNIZED')


def audit_zip(raw):
    result = {'extraction': 'streamed_only_no_files_extracted', 'safety': {}, 'tables': {},
              'members': [], 'transaction_factor_computed': False, 'holding_filter_applied': False}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = safe_members(archive)
        table_members = {}
        for member in members:
            if member.is_dir():
                continue
            stem = PurePosixPath(member.filename).stem.upper()
            if stem in TABLES:
                if stem in table_members:
                    raise ValueError('duplicate_table')
                table_members[stem] = member
        if set(table_members) != TABLES:
            raise ValueError('missing_expected_table')
        forms = {}
        ordered = sorted((m for m in members if not m.is_dir()),
                         key=lambda m: (m != table_members['SUBMISSION'], m.filename))
        for member in ordered:
            data = archive.read(member)  # full reads verify ZIP CRC, after all member guards
            if len(data) != member.file_size:
                raise ValueError('zip_member_size_mismatch')
            result['members'].append({'path': member.filename, 'bytes': len(data), 'sha256': sha(data), 'crc_verified': True})
            table = PurePosixPath(member.filename).stem.upper()
            if table not in TABLES:
                continue
            rows = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''), delimiter='\t')
            fields = rows.fieldnames
            if not fields or len(set(fields)) != len(fields):
                raise ValueError('invalid_table_header')
            required = {'ACCESSION_NUMBER'}
            if table == 'SUBMISSION':
                required |= {'DOCUMENT_TYPE', 'FILING_DATE'}
            if table == 'NONDERIV_TRANS':
                required |= {'TRANS_CODE', 'TRANS_DATE', 'TRANS_ACQUIRED_DISP_CD'}
            if not required.issubset(fields):
                raise ValueError('missing_required_columns')
            summary = {'columns': fields, 'rows': 0, 'date_ranges': {}, 'document_types': {},
                       'transaction_codes': {}, 'p_rows_by_document_type': {}, 'p_rows_by_acquired_disposed': {}}
            for field in sorted(DATE_FIELDS.intersection(fields)):
                summary['date_ranges'][field] = {'valid': 0, 'invalid': 0, 'missing': 0, 'min': None, 'max': None}
            docs, codes, pdocs, pacq = Counter(), Counter(), Counter(), Counter()
            for row in rows:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('malformed_tsv_row')
                summary['rows'] += 1
                for field, values in summary['date_ranges'].items():
                    date_update(values, row[field])
                if table == 'SUBMISSION':
                    accession = row['ACCESSION_NUMBER']
                    if accession in forms:
                        raise ValueError('duplicate_submission_accession')
                    form = categorical(row['DOCUMENT_TYPE'], {'3', '3/A', '4', '4/A', '5', '5/A'})
                    forms[accession] = form
                    docs[form] += 1
                if table == 'NONDERIV_TRANS':
                    code = categorical(row['TRANS_CODE'], set('ACDEFGHIJLMOPSUWXZK'))
                    codes[code] += 1
                    if code == 'P':
                        pdocs[forms.get(row['ACCESSION_NUMBER'], 'UNMATCHED_SUBMISSION')] += 1
                        pacq[categorical(row['TRANS_ACQUIRED_DISP_CD'], {'A', 'D'})] += 1
            summary.update(document_types=dict(docs), transaction_codes=dict(codes),
                           p_rows_by_document_type=dict(pdocs), p_rows_by_acquired_disposed=dict(pacq))
            result['tables'][table] = summary
        result['safety'] = {'members_checked_before_decompression': True, 'crc_all_members_passed': True,
                            'member_count': len(members), 'uncompressed_bytes': sum(m.file_size for m in members),
                            'path_traversal_symlink_encryption_size_ratio_guards': True}
    return result


def collect(root=ROOT, environ=None, opener=None):
    out = root / OUTPUT
    out.mkdir()  # exclusive: never overwrite, resume, or replay a prior source attempt
    state = {'schema': 'form4_source_probe_v1', 'created_at_utc': utc(), 'status': 'preflight',
             'source_page': LANDING, 'requested_quarter_labels': list(TARGETS), 'resolved_urls': {},
             'network_requests': 0, 'quarter_packages_downloaded': 0, 'requests': [],
             'no_factor_or_return_calculation': True, 'no_observer_access': True,
             'source_only_not_registered_or_admitted': True,
             'ua_requirement_reference': 'https://www.sec.gov/about/webmaster-frequently-asked-questions'}
    try:
        ua, checks = existing_ua(root, os.environ if environ is None else environ)
        state['ua_configuration_checks'] = checks
        if ua is None:
            state['status'] = 'blocked_missing_compliant_user_agent'
            state['required_future_configuration'] = 'Set a genuine identifying SEC_USER_AGENT or EDGAR_USER_AGENT with an authorized contact email; never invent a contact. The collector records no UA value.'
            return state
        state['ua_value_recorded'] = False
        opener = opener or urllib.request.build_opener(NoRedirect())
        queue = [('source_page.html', LANDING, 4 * 1024 * 1024)]
        for name, url, limit in queue:
            if state['network_requests']:
                time.sleep(1)
            state['network_requests'] += 1
            raw, receipt = fetch(url, ua, out, name, limit, opener)
            state['requests'].append(receipt)
            if receipt['http_status'] in (403, 429):
                state['status'] = 'blocked_sec_http_' + str(receipt['http_status'])
                state['stop_sec_source_for_utc_day'] = receipt['finished_at_utc'][:10]
                save_json(out / 'STOP_SEC_SOURCE_UTC_DAY.json', {'date_utc': state['stop_sec_source_for_utc_day'],
                          'status': receipt['http_status'], 'no_retry_or_alternate_route': True})
                return state
            if receipt['http_status'] != 200 or not receipt['complete'] or receipt['error_type']:
                state['status'] = 'blocked_http_or_transport_failure'
                return state
            if name == 'source_page.html':
                state['resolved_urls'] = exact_links(raw)
                queue.extend((label.replace(' ', '_') + '.zip', state['resolved_urls'][label], MAX_PACKAGE)
                             for label in TARGETS)
            else:
                state['quarter_packages_downloaded'] += 1
                save_json(out / (name + '.audit.json'), audit_zip(raw))
        state['status'] = 'source_schema_audit_complete_not_admitted'
        return state
    except Exception as exc:
        state['status'] = 'blocked_local_validation'
        state['error_type'] = type(exc).__name__
        return state
    finally:
        state['finished_at_utc'] = utc()
        save_json(out / 'STATUS.json', state)


class SyntheticTests(unittest.TestCase):
    def test_missing_ua_makes_zero_requests_and_never_overwrites(self):
        class Never:
            def open(self, *args, **kwargs):
                raise AssertionError('network_must_not_run')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'outputs').mkdir()
            result = collect(root, {}, Never())
            self.assertEqual(result['status'], 'blocked_missing_compliant_user_agent')
            self.assertEqual(result['network_requests'], 0)
            self.assertEqual(json.loads((root / OUTPUT / 'STATUS.json').read_text()), result)
            with self.assertRaises(FileExistsError):
                collect(root, {}, Never())

    def test_ua_shape_does_not_accept_placeholders_or_injection(self):
        for value in (None, '', 'research', 'Research a@example.com', 'Research a@valid.org\nX: y'):
            self.assertFalse(valid_ua(value))
        self.assertTrue(valid_ua('Synthetic test robot audit@synthetic-fixture.org'))

    def test_official_link_resolution_and_rejection(self):
        raw = b'<a href="/files/a.zip">2016 Q3 345</a><a href="/files/b.zip">2021 Q1 345</a>'
        self.assertEqual(exact_links(raw)[TARGETS[0]], 'https://www.sec.gov/files/a.zip')
        for bad in (raw.replace(b'/files/a.zip', b'https://elsewhere.invalid/a.zip'), raw + raw,
                    raw.replace(b'/files/a.zip', b'/../a.zip'), raw.replace(b'/files/a.zip', b'/a.zip?token=x')):
            with self.assertRaises(ValueError):
                exact_links(bad)

    def test_http_403_and_429_stop_before_any_package(self):
        for code in (403, 429):
            class Blocked:
                calls = 0
                def open(self, request, timeout):
                    self.calls += 1
                    raise urllib.error.HTTPError(request.full_url, code, 'blocked', {}, io.BytesIO(b'synthetic refusal'))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'outputs').mkdir()
                opener = Blocked()
                result = collect(root, {'SEC_USER_AGENT': 'Synthetic test audit@synthetic-fixture.org'}, opener)
                self.assertEqual(opener.calls, 1)
                self.assertEqual(result['network_requests'], 1)
                self.assertEqual(result['quarter_packages_downloaded'], 0)
                self.assertTrue((root / OUTPUT / 'STOP_SEC_SOURCE_UTC_DAY.json').is_file())
                self.assertEqual(result['requests'][0]['sha256'], sha(b'synthetic refusal'))
                self.assertNotIn('audit@', json.dumps(result))

    def test_all_member_guards_precede_decompression(self):
        for name, mode in (('../escape', 0), ('/absolute', 0), ('a\\b', 0), ('link', stat.S_IFLNK)):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                member = zipfile.ZipInfo(name)
                member.external_attr = mode << 16
                archive.writestr(member, b'payload')
            with self.assertRaises(ValueError):
                audit_zip(buffer.getvalue())

    def test_package_refusal_stops_before_second_quarter(self):
        page = b'<a href="/files/a.zip">2016 Q3 345</a><a href="/files/b.zip">2021 Q1 345</a>'
        class Page(io.BytesIO):
            code = 200
            headers = {}
            def geturl(self):
                return LANDING
        class RefusePackage:
            calls = []
            def open(self, request, timeout):
                self.calls.append(request.full_url)
                if len(self.calls) == 1:
                    return Page(page)
                raise urllib.error.HTTPError(request.full_url, 403, 'blocked', {}, io.BytesIO(b'no'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'outputs').mkdir()
            opener = RefusePackage()
            result = collect(root, {'SEC_USER_AGENT': 'Synthetic test audit@synthetic-fixture.org'}, opener)
            self.assertEqual(opener.calls, [LANDING, 'https://www.sec.gov/files/a.zip'])
            self.assertEqual(result['status'], 'blocked_sec_http_403')
            self.assertEqual(result['quarter_packages_downloaded'], 0)

    def test_duplicate_members_and_expansion_limits(self):
        class Archive:
            def __init__(self, members):
                self.members = members
            def infolist(self):
                return self.members
        member = zipfile.ZipInfo('table.tsv')
        with self.assertRaises(ValueError):
            safe_members(Archive([member, member]))
        member.file_size = MAX_UNPACKED + 1
        with self.assertRaises(ValueError):
            safe_members(Archive([member]))
        member.file_size, member.compress_size = 1001, 1
        with self.assertRaises(ValueError):
            safe_members(Archive([member]))

    def test_schema_dates_amendments_and_p_counts_no_owner_multiplication(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for table in sorted(TABLES):
                text = 'ACCESSION_NUMBER\nx\n'
                if table == 'SUBMISSION':
                    text = 'ACCESSION_NUMBER\tDOCUMENT_TYPE\tFILING_DATE\nx\t4\t01-JAN-2021\ny\t4/A\t02-JAN-2021\n'
                elif table == 'NONDERIV_TRANS':
                    text = 'ACCESSION_NUMBER\tTRANS_CODE\tTRANS_DATE\tTRANS_ACQUIRED_DISP_CD\nx\tP\t31-DEC-2020\tA\ny\tP\t31-DEC-2020\tA\nx\tS\tBAD\tD\n'
                elif table == 'REPORTINGOWNER':
                    text = 'ACCESSION_NUMBER\nx\nx\nx\n'
                archive.writestr(table + '.tsv', text)
        result = audit_zip(buffer.getvalue())
        summary = result['tables']['NONDERIV_TRANS']
        self.assertEqual(summary['transaction_codes'], {'P': 2, 'S': 1})
        self.assertEqual(summary['p_rows_by_document_type'], {'4': 1, '4/A': 1})
        self.assertEqual(summary['date_ranges']['TRANS_DATE']['invalid'], 1)
        self.assertEqual(summary['date_ranges']['TRANS_DATE']['min'], '2020-12-31')
        self.assertTrue(result['safety']['crc_all_members_passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    result = collect()
    print(json.dumps({k: result[k] for k in ('status', 'network_requests', 'quarter_packages_downloaded')}))
    return 0 if result['status'] == 'source_schema_audit_complete_not_admitted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
