"""Two issuer-distributed CAT release sources; separate from stopped IR/SEC requests."""
import argparse
from bisect import bisect_right
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from after155_broadcom_gap_source import ROOT, Text, digest, read, save, utc

OUT = ROOT / 'outputs/monthly_event_cat_wire156'
TARGETS = [
    dict(key='CAT_2025Q1', fiscal_year=2025, fiscal_quarter=1, publication_date='2025-04-30',
         period_end='2025-03-31', ordinal='first', dateline_month='April', dateline_day=30,
         period_month='March', period_day=31,
         url='https://www.prnewswire.com/news-releases/caterpillar-reports-first-quarter-2025-results-302442518.html'),
    dict(key='CAT_2025Q2', fiscal_year=2025, fiscal_quarter=2, publication_date='2025-08-05',
         period_end='2025-06-30', ordinal='second', dateline_month='Aug(?:ust)?\\.?', dateline_day=5,
         period_month='June', period_day=30,
         url='https://www.prnewswire.com/news-releases/caterpillar-reports-second-quarter-2025-results-302521882.html')
]


def freeze():
    inputs = [
        'work/after155_cat_wire_source.py', 'work/after155_broadcom_gap_source.py',
        'work/after155_sec_gap_discovery.json', 'work/after155_sec_gap_discovery_collection.json',
        'outputs/monthly_event_gap156_cat_hsbc/COLLECTION.json',
        'outputs/monthly_earnings_repair/EVENTS.json',
        'outputs/monthly_event_scope156/EVENTS.json',
        'outputs/monthly_event_patch156/ADDED_EVENTS.json',
        'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json',
        'outputs/monthly_paired_target143/CALENDAR.json',
        'outputs/monthly_earnings_repair/event_schema.py',
        'outputs/monthly_earnings_repair/research_calendar.py',
        'outputs/index_union/CALENDAR_EXCEPTIONS.json',
        'outputs/index_union/sources/closure_20250109.html',
        'work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py'
    ]
    save(OUT / 'SOURCE_SPEC.json', dict(created_utc=utc(), targets=TARGETS,
        source_class='issuer_distributed_original_release_PRNewswire',
        authorization='Separate two-URL source batch explicitly authorized by parent task; this is issuer-distributed content, not an alternate route to stopped SEC or issuer IR URLs.',
        input_hashes={rel: digest((ROOT / rel).read_bytes()) for rel in inputs},
        network_policy='One normal GET per frozen URL; no retry. Any 403 or 429 stops all subsequent requests to this source. Preserve Retry-After and response bytes.',
        SEC_requests_permitted=False, IR_requests_permitted=False,
        events_admitted=False, calendar_complete=False,
        financial_values_extracted=0, price_values_read=0, factor_calculations=0, returns=0))


def checked_spec():
    spec = read(OUT / 'SOURCE_SPEC.json')
    for rel, expected in spec['input_hashes'].items():
        assert digest((ROOT / rel).read_bytes()) == expected, rel
    return spec


def collect():
    spec = checked_spec()
    assert not (OUT / 'COLLECTION.json').exists(), 'Collection is write-once'
    rows = []
    stopped = False
    for target in spec['targets']:
        row = dict(key=target['key'], requested_url=target['url'], request_utc=utc())
        body = None
        if stopped:
            row['status'] = 'not_attempted_after_source_stop'
        else:
            try:
                request = urllib.request.Request(target['url'], headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; historical-research/1.0)'})
                with urllib.request.urlopen(request, timeout=35) as response:
                    body = response.read(8 * 1024 * 1024 + 1)
                    assert len(body) <= 8 * 1024 * 1024
                    row.update(status='downloaded', http_status=response.status, final_url=response.url,
                        content_type=response.headers.get('Content-Type'), retry_after=response.headers.get('Retry-After'),
                        etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))
            except urllib.error.HTTPError as error:
                body = error.read(8 * 1024 * 1024)
                row.update(status='http_error', http_status=error.code, final_url=error.url,
                    content_type=error.headers.get('Content-Type'), retry_after=error.headers.get('Retry-After'))
                stopped = error.code in (403, 429)
            except (urllib.error.URLError, TimeoutError) as error:
                row.update(status='network_error', error_type=type(error).__name__, error=str(error))
                stopped = True
        if body is not None:
            raw = OUT / (target['key'] + '.html')
            with raw.open('xb') as stream:
                stream.write(body)
            row.update(path=str(raw.relative_to(ROOT)), sha256=digest(body), bytes=len(body))
        if stopped:
            row['source_stop_utc_date'] = row['request_utc'][:10]
        save(OUT / (target['key'] + '.receipt.json'), row)
        rows.append(row)
    save(OUT / 'COLLECTION.json', dict(created_utc=utc(), source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes()), receipts=rows))
    print([(row['key'], row['status'], row.get('http_status')) for row in rows])


def hit(pattern, plain):
    matched = re.search(pattern, plain, flags=re.I)
    return dict(pattern=pattern, found=bool(matched), normalized_offset=matched.start() if matched else None,
                quote=matched.group() if matched else None)


def inspect():
    spec = checked_spec()
    sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
    from event_schema import normalize_event
    from research_calendar import market_calendar
    calendar = market_calendar(start='2025-01-01', end='2026-01-01')
    frozen_sessions = read(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    catalogs = {path: read(ROOT / path) for path in (
        'outputs/monthly_earnings_repair/EVENTS.json', 'outputs/monthly_event_scope156/EVENTS.json',
        'outputs/monthly_event_patch156/ADDED_EVENTS.json')}
    gaps = read(ROOT / 'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json')['quarter_items']
    collection = read(OUT / 'COLLECTION.json')
    results = []
    for target, receipt in zip(spec['targets'], collection['receipts']):
        assert target['key'] == receipt['key']
        row = dict(key=target['key'], receipt=receipt, source_checks={}, candidate=None)
        if receipt['status'] == 'downloaded':
            blob = (ROOT / receipt['path']).read_bytes()
            assert digest(blob) == receipt['sha256']
            raw = blob.decode('utf-8', errors='replace')
            parser = Text(); parser.feed(raw)
            plain = ' '.join(' '.join(parser.parts).split())
            checks = {
                'title': r'Caterpillar Reports ' + target['ordinal'] + r'-Quarter 2025 Results',
                'issuer': r'Caterpillar Inc\.\s*\(NYSE:\s*CAT\)',
                'provided_by': r'News provided by\s+Caterpillar Inc\.',
                'dateline': r'IRVING, Texas,\s*' + target['dateline_month'] + r'\s+' + str(target['dateline_day']) + r', 2025',
                'actual_results': r'announced ' + target['ordinal'] + r'-quarter 2025 results',
                'period_header': r'Three Months Ended\s+' + target['period_month'] + r'\s+' + str(target['period_day']) + r',?\s+2025'
            }
            row['source_checks'] = {name: hit(pattern, plain) for name, pattern in checks.items()}
            row['raw_line_hints'] = [i+1 for i, line in enumerate(raw.splitlines()) if
                ('IRVING' in line or 'Three Months Ended' in line or 'News provided by' in line)]
            row['source_display_timestamp'] = hit(r'(?:Apr(?:il)?|Aug(?:ust)?)\.?\s+' + str(target['dateline_day']) + r', 2025,\s*\d{1,2}:\d{2}\s*ET', plain)
            if all(check['found'] for check in row['source_checks'].values()):
                event = normalize_event(dict(symbol='CAT', official=dict(actual_period_end=target['period_end'],
                    publication_date=target['publication_date'], publication_timezone='America/Chicago',
                    source_url=target['url'], verified=True, publication_kind='newswire_release')), calendar=calendar)
                assert event['status'] == 'ready'
                assert event['available_session'] == frozen_sessions[bisect_right(frozen_sessions, target['publication_date'])]
                event.update(event_id='CAT_' + target['publication_date'], fiscal_year=2025,
                    fiscal_quarter=target['fiscal_quarter'], evidence_path=receipt['path'], evidence_sha256=receipt['sha256'],
                    source_kind='issuer_distributed_original_earnings_release', status='proposal_pending_independent_review',
                    event_admitted=False, first_public_time_verified=False,
                    publication_timezone_basis='Conservative issuer-local date interpretation from IRVING, Texas dateline; displayed ET time retained but not asserted as first-public time.',
                    historical_archive_limitation='Later-retrieved issuer-distributed original release, not a contemporaneous immutable capture.')
                row['candidate'] = event
        row['catalog_conflicts'] = {path: [dict(event_id=e.get('event_id'), date=e.get('publication_date')) for e in events
            if e.get('symbol') == 'CAT' and (e.get('publication_date') == target['publication_date'] or
            e.get('official_actual_period_end') == target['period_end'] or
            (e.get('fiscal_year'), e.get('fiscal_quarter')) == (2025, target['fiscal_quarter']))]
            for path, events in catalogs.items()}
        row['affected_gap_records'] = [gap for gap in gaps if gap['symbol'] == 'CAT' and
            '2025Q' + str(target['fiscal_quarter']) in gap['intermediate_fiscal_labels']]
        row['ready_for_independent_source_review'] = row['candidate'] is not None and not any(row['catalog_conflicts'].values())
        results.append(row)
    save(OUT / 'SOURCE_PROPOSAL.json', dict(created_utc=utc(), results=results,
        source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes()),
        collection_sha256=digest((OUT / 'COLLECTION.json').read_bytes()),
        scope='Only two CAT actual results releases, pending independent review; no event ingestion or complete calendar assertion.',
        events_admitted=False, calendar_complete=False, financial_values_extracted=0, price_values_read=0,
        factor_calculations=0, returns=0, SEC_stop_preserved=True, IR_failures_preserved=True))
    print([(row['key'], row['ready_for_independent_source_review'], {k:v['found'] for k,v in row['source_checks'].items()}) for row in results])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'collect', 'inspect'))
    {'freeze': freeze, 'collect': collect, 'inspect': inspect}[parser.parse_args().phase]()
