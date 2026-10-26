"""Offline source/coverage validation only; no price arrays, factors or returns.

--validate is an explicit, write-once source operation. Run only after approval.
The standalone synthetic test log must bind the current validator/test bytes.
"""
import argparse
from collections import Counter
import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys

OUT = 'outputs/after159_vvix_source'
PLAN = 'work/after159_vvix_source_plan.json'
PLAN_SHA = '3b7ad5ef022bf8626acce9292438508866c6a83d31f1ba28b2153faba2e98be8'
COLLECTOR = 'work/after159_vvix_collect.py'
COLLECTOR_SHA = 'e1ce3a9156f1468d5d1bc968150131db36ce97981a23dff0a005ccdcb7c722b1'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
META = 'outputs/monthly_local_linear70/META.json'
META_SHA = '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'
INITIAL = '2016-08-17'
HEADERS = {'VVIX': ['DATE', 'VVIX'], 'VIX': ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']}
FILENAMES = {'historical_index_page': 'historical_index_page.html', 'VVIX': 'VVIX_History.csv',
             'VIX': 'VIX_History.csv', 'methodology': 'methodology.pdf', 'launch_release': 'launch_release.html'}
MIMES = {'historical_index_page': 'text/html', 'VVIX': 'text/csv', 'VIX': 'text/csv',
         'methodology': 'application/pdf', 'launch_release': 'text/html'}
TEST = 'work/test_after159_vvix_validate.py'
SELF = 'work/after159_vvix_validate.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ordered_iso(values, label):
    require(type(values) is list and values, 'Nonempty '+label)
    require(all(type(v) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', v) and
                date.fromisoformat(v).isoformat() == v for v in values), 'Exact ISO dates: '+label)
    require(all(a < b for a, b in zip(values, values[1:])), 'Unique ascending '+label)


def contract():
    return dict(schema='after159_vvix_source_validation_v1', purpose='source_only_not_strategy_admission',
        sources=['VVIX', 'VIX'], schemas=HEADERS, values_consumed={'VVIX': 'VVIX', 'VIX': 'CLOSE'},
        window_sessions=252, end='immediately_previous_frozen_official_session', initial_partial=INITIAL,
        missing_policy='explicit failure for required sessions; no fill, substitution or changed denominator',
        future_policy='retain diagnostics separately; out-of-window values cannot affect window coverage',
        chronology='all source rows must have unique strictly ascending real MM/DD/YYYY dates',
        noncalendar_policy='report original line/date/cells; do not insert into official-session windows',
        methodology_scope='current downloaded document only; no original-vintage certification',
        new_features=0, new_ranks=0, new_returns=0, new_orders=0, strategy_registered=False)


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.json_scripts = []
        self._script = None
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(dict(url=attrs['href'], line=self.getpos()[0]))
        if tag == 'script' and attrs.get('type', '').lower() == 'application/ld+json':
            self._script = dict(line=self.getpos()[0], text='')
    def handle_data(self, value):
        if self._script is not None:
            self._script['text'] += value
    def handle_endtag(self, tag):
        if tag == 'script' and self._script is not None:
            self.json_scripts.append(self._script)
            self._script = None


def official_links(raw, urls):
    page = Page(); page.feed(raw.decode('utf-8'))
    matched = {name: [x for x in page.links if x['url'] == urls[name]] for name in HEADERS}
    require(all(matched.values()), 'Official page must link both exact original CSV URLs as anchors')
    return matched


def launch_evidence(raw, url, first_signal):
    page = Page(); page.feed(raw.decode('utf-8'))
    found = []
    for script in page.json_scripts:
        obj = json.loads(script['text'])
        if not isinstance(obj, dict) or obj.get('@type') != 'NewsArticle':
            continue
        if obj.get('mainEntityOfPage', {}).get('@id') != url:
            continue
        title = unescape(obj.get('headline', ''))
        description = unescape(obj.get('description', ''))
        require('VIX of VIX' in title and 'CBOE' in title and 'began publishing' in description,
                'Specific VVIX launch announcement required')
        stamp = obj['datePublished']
        parsed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None and
                parsed.date().isoformat() < first_signal, 'Launch announcement must predate sample')
        found.append(dict(line=script['line'], headline=title, datePublished=stamp,
                          dateModified=obj.get('dateModified'), description=description))
    require(len(found) == 1, 'One exact-URL launch article required')
    return found[0]


def numeric_status(literal):
    try:
        x = Decimal(literal.strip())
    except InvalidOperation:
        return 'not_numeric'
    if not x.is_finite():
        return 'nonfinite'
    if x <= 0:
        return 'nonpositive'
    return None


def parse_csv(name, raw):
    """No financial transformation: retain literal rows and value-validity flags."""
    require(name in HEADERS, 'Fixed series name')
    reader = csv.reader(io.StringIO(raw.decode('utf-8'), newline=''), strict=True)
    header = next(reader, None)
    require(header == HEADERS[name], 'Exact observed CSV header required: '+name)
    rows, errors, order_errors = [], [], []
    last = None
    for cells in reader:
        line = reader.line_num
        if len(cells) != len(header):
            errors.append(dict(line=line, reason='column_count', cells=cells)); continue
        literal = cells[0]
        try:
            require(re.fullmatch(r'\d{2}/\d{2}/\d{4}', literal), 'date shape')
            day = datetime.strptime(literal, '%m/%d/%Y').date().isoformat()
        except ValueError:
            errors.append(dict(line=line, reason='invalid_date', cells=cells)); continue
        if last is not None and day <= last:
            order_errors.append(dict(line=line, date=day, previous_date=last))
        last = day
        status = {k: numeric_status(v) for k, v in zip(header[1:], cells[1:])}
        rows.append(dict(line=line, date=day, cells=cells, value_status=status,
                         consumed_value_status=status['VVIX' if name == 'VVIX' else 'CLOSE']))
    require(rows or errors, 'Empty CSV history: '+name)
    counts = Counter(row['date'] for row in rows)
    duplicates = sorted(day for day, n in counts.items() if n > 1)
    return dict(name=name, header=header, rows=rows, schema_errors=errors,
                order_errors=order_errors, duplicate_dates=duplicates,
                schema_and_chronology_passed=not (errors or order_errors or duplicates))


def coverage(series, sessions, signals, initial=INITIAL):
    """Metadata/value-validity windows only. Future rows never substitute for gaps."""
    ordered_iso(sessions, 'official sessions'); ordered_iso(signals, 'signals')
    require(signals[0] == initial, 'Initial partial must be first and remain neutral')
    require(set(series) == set(HEADERS), 'Two fixed series required')
    index = {d: i for i, d in enumerate(sessions)}
    require(all(d in index for d in signals), 'Every original signal must be an exact official session')
    windows = []; consumed = set()
    by_source = {}
    for name, data in series.items():
        lookup = {}
        for row in data['rows']:
            lookup.setdefault(row['date'], []).append(row)
        by_source[name] = lookup
    for i, signal in enumerate(signals):
        if i == 0:
            windows.append(dict(signal_index=i, signal_date=signal, initial_partial=True,
                                required_sessions=0, first_required_date=None, last_required_date=None,
                                passed=True, status='neutral_initial_partial', sources={}))
            continue
        t = index[signal]
        needed = sessions[max(0, t-252):t]
        consumed.update(needed)
        complete = len(needed) == 252
        sources = {}
        for name, lookup in by_source.items():
            missing = [d for d in needed if d not in lookup]
            bad = []
            for d in needed:
                rows = lookup.get(d, [])
                if len(rows) > 1:
                    bad.append(dict(date=d, reason='duplicate_date', lines=[r['line'] for r in rows]))
                elif rows and rows[0]['consumed_value_status'] is not None:
                    r = rows[0]
                    bad.append(dict(date=d, reason=r['consumed_value_status'], line=r['line'], cells=r['cells']))
            sources[name] = dict(passed=complete and not missing and not bad,
                                 missing_dates=missing, invalid_required_rows=bad,
                                 valid_required_sessions=len(needed)-len(missing)-len(bad))
        windows.append(dict(signal_index=i, signal_date=signal, initial_partial=False,
            required_sessions=252, available_calendar_sessions=len(needed),
            first_required_date=needed[0] if needed else None, last_required_date=needed[-1] if needed else None,
            required_sessions_sha256=sha_bytes(json.dumps(needed, separators=(',', ':')).encode()),
            passed=all(s['passed'] for s in sources.values()), sources=sources))
    last_cutoff = max(consumed) if consumed else None
    diagnostics = {}
    calendar_set = set(sessions)
    for name, data in series.items():
        rows = data['rows']
        def details(r):
            return {k: r[k] for k in ('line', 'date', 'cells', 'value_status')}
        noncalendar = [details(r) for r in rows if sessions[0] <= r['date'] <= sessions[-1] and r['date'] not in calendar_set]
        outside = [r['date'] for r in rows if not sessions[0] <= r['date'] <= sessions[-1]]
        future = [r for r in rows if last_cutoff is not None and r['date'] > last_cutoff]
        invalid = [r for r in rows if r['consumed_value_status'] is not None]
        diagnostics[name] = dict(header=data['header'], row_count=len(rows),
            first_date=min(r['date'] for r in rows) if rows else None,
            last_date=max(r['date'] for r in rows) if rows else None,
            schema_and_chronology_passed=data['schema_and_chronology_passed'],
            schema_errors=data['schema_errors'], order_errors=data['order_errors'], duplicate_dates=data['duplicate_dates'],
            noncalendar_rows=noncalendar, outside_calendar_extent_dates=outside,
            future_beyond_last_required=dict(row_count=len(future), first_date=min((r['date'] for r in future), default=None),
                last_date=max((r['date'] for r in future), default=None),
                invalid_rows=[details(r) for r in future if r['consumed_value_status'] is not None]),
            invalid_consumed_rows=[details(r) for r in invalid if r['date'] in consumed],
            invalid_unconsumed_rows=[details(r) for r in invalid if r['date'] not in consumed],
            unused_OHLC_issues=[details(r) for r in rows if name == 'VIX' and any(r['value_status'][k] for k in ('OPEN', 'HIGH', 'LOW'))],
            covered_noninitial_signals=sum(w['sources'][name]['passed'] for w in windows[1:]))
    return dict(windows=windows, sources=diagnostics, coverage_passed=all(w['passed'] for w in windows),
        schema_and_chronology_passed=all(d['schema_and_chronology_passed'] for d in series.values()),
        signals_checked=len(signals), noninitial_signals=len(signals)-1,
        first_required_date=min(consumed) if consumed else None, last_required_date=last_cutoff,
        distinct_required_sessions=len(consumed))


def collection_records(plan, collection, plan_sha, read_bytes):
    require(plan['schema'] == 'after159_vvix_source_collection_v1' and plan['output_directory'] == OUT and
            plan['fixed_scope'] == ['VVIX', 'VIX'] and set(plan['sources']) == set(FILENAMES), 'Fixed source plan')
    require(collection.get('complete') is True and collection['plan_sha256'] == plan_sha and
            set(collection['sources']) == set(FILENAMES) and not collection.get('stop_for_429'), 'Complete exact collection')
    bodies = {}
    for key, name in FILENAMES.items():
        record = collection['sources'][key]
        require(type(record.get('status')) is int and record['status'] == 200 and not record.get('error'), 'Source HTTP success: '+key)
        require(record['requested_url'] == record['final_url'] == plan['sources'][key], 'Exact source URL: '+key)
        require(record['file'] == OUT+'/'+name, 'Exact source path: '+key)
        mime = record['headers']['Content-Type'].split(';')[0].strip().lower()
        require(mime == MIMES[key], 'Source MIME: '+key)
        stamp = datetime.fromisoformat(record['retrieved_utc'].replace('Z', '+00:00'))
        require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, 'UTC collection receipt')
        body = read_bytes(record['file'])
        require(type(record['bytes']) is int and len(body) == record['bytes'] and
                sha_bytes(body) == record['sha256'], 'Source byte/hash mismatch: '+key)
        if key == 'methodology':
            require(body.startswith(b'%PDF-'), 'Actual methodology PDF required')
        bodies[key] = body
    return bodies


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.seen = {}
    def path(self, name):
        p = Path(name); p = (p if p.is_absolute() else self.root/p).resolve()
        require(not any(s in str(p) for s in ('CloudStorage', 'OneDrive', 'stock-observer-runtime')),
                'Forbidden path')
        return p
    def bytes(self, name):
        p = self.path(name); value = p.read_bytes(); digest = sha_bytes(value)
        require(str(p) not in self.seen or self.seen[str(p)] == digest, 'Input changed during validation: '+str(p))
        self.seen[str(p)] = digest
        return value
    def bind(self, name, digest):
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'Explicit SHA256')
        require(sha_bytes(self.bytes(name)) == digest, 'Input SHA mismatch: '+str(name))
    def json(self, name):
        def unique(items):
            out = {}
            for k, v in items:
                require(k not in out, 'Duplicate JSON key'); out[k] = v
            return out
        return json.loads(self.bytes(name), object_pairs_hook=unique)
    def refresh(self):
        for name in list(self.seen):
            self.bytes(name)


def tests_receipt(reader, log_path, log_sha):
    reader.bind(log_path, log_sha)
    text = reader.bytes(log_path).decode()
    markers = [line.split('=', 1)[1] for line in text.splitlines() if line.startswith('SYNTHETIC_TEST_RECEIPT=')]
    require(len(markers) == 1 and '\nOK\n' in text, 'Successful synthetic test log required')
    receipt = json.loads(markers[0])
    counts = re.findall(r'Ran (\d+) tests? in ', text)
    require(len(counts) == 1 and int(counts[0]) == receipt['tests_run'] and receipt['tests_run'] >= 15 and
            receipt['passed'] is True and receipt['synthetic_only'] is True and receipt['hashes_unchanged'] is True,
            'Complete synthetic test receipt')
    require(set(receipt['implementation_hashes']) == {SELF, TEST}, 'Exact validator/test closure')
    for name, digest in receipt['implementation_hashes'].items():
        reader.bind(name, digest)
    return receipt


def validate(root, collection_sha, log_path, log_sha):
    reader = Reader(root); destination = reader.path(OUT)/'VALIDATION.json'
    require(not destination.exists(), 'Write-once VALIDATION already exists')
    tested = tests_receipt(reader, log_path, log_sha)
    for name, digest in ((PLAN, PLAN_SHA), (COLLECTOR, COLLECTOR_SHA),
                         (CALENDAR, CALENDAR_SHA), (META, META_SHA), (OUT+'/COLLECTION.json', collection_sha)):
        reader.bind(name, digest)
    plan, collection, calendar, meta = [reader.json(p) for p in (PLAN, OUT+'/COLLECTION.json', CALENDAR, META)]
    require(len(calendar['provenance']['source_hashes']) == 9, 'All nine frozen calendar dependencies required')
    for name, digest in calendar['provenance']['source_hashes'].items():
        reader.bind(name, digest)
    for name in ('run_local.sh', 'local_run.py', 'local.sandbox.sb', sys.executable):
        reader.bytes(name)
    require(len(meta['dates']) == len(meta['symbols']) == len(meta['ends']) == 122 and
            all(len(s) == len(set(s)) == 49 for s in meta['symbols']) and
            meta['dates'][0] == INITIAL and meta['dates'][-1] == '2026-08-31', 'Frozen original122x49 axes')
    bodies = collection_records(plan, collection, PLAN_SHA, reader.bytes)
    links = official_links(bodies['historical_index_page'], plan['sources'])
    launch = launch_evidence(bodies['launch_release'], plan['sources']['launch_release'], meta['dates'][0])
    data = {name: parse_csv(name, bodies[name]) for name in HEADERS}
    report = coverage(data, calendar['sessions'], meta['dates'])
    report.update(contract=contract(), plan_sha256=PLAN_SHA, collection_sha256=collection_sha,
        passed=report['coverage_passed'] and report['schema_and_chronology_passed'],
        official_csv_links=links, launch_evidence=launch, synthetic_tests=tested,
        original_meta_ends_last=meta['ends'][-1], calendar_start=calendar['start'], calendar_end=calendar['end'],
        methodology=dict(raw_sha256=collection['sources']['methodology']['sha256'],
                         scope='Current document hash verified; historical revision completeness not certified'),
        limitations=[plan['timing_assumption'],
            'A current historical download and pre-sample launch do not certify original as-published vintages.',
            'Noncalendar rows and outside-calendar-extent dates are retained as diagnostics, never substituted.',
            'Coverage is not source admission for a strategy; no ranks, features, returns, orders or selection computed.'],
        source_admission_changed=False, new_features=0, new_ranks=0, new_returns=0, new_orders=0)
    reader.refresh(); report['input_hashes'] = dict(reader.seen)
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    return dict(passed=report['passed'], output=str(destination), sha256=sha_bytes(destination.read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--collection-sha'); p.add_argument('--tests-log'); p.add_argument('--tests-sha')
    args = p.parse_args()
    if not args.validate:
        print(json.dumps(contract(), indent=2)); return
    require(args.collection_sha and args.tests_log and args.tests_sha, 'Collection SHA and bound synthetic log required')
    result = validate(Path(__file__).resolve().parents[1], args.collection_sha, args.tests_log, args.tests_sha)
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
