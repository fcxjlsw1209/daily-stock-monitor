"""Three bounded release-source leads; no event ingestion or financial calculations."""
import argparse
from bisect import bisect_right
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from after155_broadcom_gap_source import Text, digest, read, save, utc, ROOT

OUT = ROOT / 'outputs/monthly_event_gap156_crm_sap_tsla'
CACHE = 'outputs/monthly_earnings_repair/batch_technology_industrial/TSLA/raw/de691e5b9b29307fe7de'
ITEMS = [
    dict(name='crm', symbol='CRM', fiscal_year=2020, fiscal_quarter=3,
         publication_date='2019-12-03', period_end='2019-10-31', timezone='America/Los_Angeles',
         url='https://www.salesforce.com/news/press-releases/2019/12/03/salesforce-announces-record-third-quarter-fiscal-2020-results/?bc=OTH',
         before='2019-08-22', after='2020-02-25',
         title=r'Salesforce Announces Record Third Quarter Fiscal 2020 Results',
         issuer=r'Salesforce\s*\(NYSE:\s*CRM\)',
         dateline=r'SAN FRANCISCO, Calif\.\s*[—–-]\s*Dec\. 3, 2019',
         actual=r'today announced results for its fiscal third quarter ended October 31, 2019',
         period=r'third quarter ended October 31, 2019',
         release_kind='actual_quarterly_results'),
    dict(name='sap', symbol='SAP', fiscal_year=2019, fiscal_quarter=4,
         publication_date='2020-01-28', period_end='2019-12-31', timezone='Europe/Berlin',
         url='https://news.sap.com/germany/2020/01/sap-veroeffentlicht-vorlaeufige-ergebnisse-fuer-das-vierte-quartal-und-das-gesamtjahr-2019/',
         before='2019-10-21', after='2020-04-21',
         title=r'SAP veröffentlicht vorläufige Ergebnisse für das vierte Quartal und das Gesamtjahr 2019',
         issuer=r'Die SAP SE',
         dateline=r'28\.\s*Januar 2020',
         actual=r'Die SAP SE hat heute die vorläufigen Ergebnisse für das vierte Quartal und das Gesamtjahr 2019.{0,80}veröffentlicht',
         period=r'Gesamtjahr 2019\s*\(1\.\s*Januar\s*[–—-]\s*31\.\s*Dezember\)',
         release_kind='actual_preliminary_quarterly_and_annual_results'),
    dict(name='tsla', symbol='TSLA', fiscal_year=2019, fiscal_quarter=3,
         publication_date='2019-10-23', period_end='2019-09-30', timezone='America/Los_Angeles',
         url='https://ir.tesla.com/press-release/tesla-releases-third-quarter-2019-financial-results',
         before='2019-07-24', after='2020-01-29',
         title=r'Tesla Releases Third Quarter 2019 Financial Results',
         issuer=r'Tesla has released its financial results',
         dateline=r'PALO ALTO, Calif\., Oct\. 23, 2019',
         actual=r'Tesla has released its financial results for the third quarter of 2019',
         period=None, release_kind='actual_quarterly_results_with_cached_update_letter'),
]


def freeze():
    inputs = [
        'work/after155_broadcom_gap_source.py', str(Path(__file__).relative_to(ROOT)),
        'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json',
        'outputs/monthly_earnings_repair/EVENTS.json',
        'outputs/monthly_event_scope156/EVENTS.json',
        'outputs/monthly_event_patch156/ADDED_EVENTS.json',
        'outputs/monthly_paired_target143/CALENDAR.json',
        'outputs/monthly_earnings_repair/event_schema.py',
        'outputs/monthly_earnings_repair/research_calendar.py',
        'outputs/index_union/CALENDAR_EXCEPTIONS.json',
        'outputs/index_union/sources/closure_20250109.html',
        'work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py',
        'outputs/monthly_earnings_repair/batch_technology_industrial/SOURCE_STOPS.json',
        'outputs/monthly_earnings_repair/batch_health_energy_foreign/SOURCE_STOPS.json',
    ] + [CACHE + suffix for suffix in ('.bin', '.meta.json', '.txt', '.text.meta.json')]
    save(OUT / 'SOURCE_SPEC.json', dict(
        created_utc=utc(), items=ITEMS,
        input_hashes={p: digest((ROOT / p).read_bytes()) for p in inputs},
        discovery='Primary issuer search results identify three actual releases. Search snippets are discovery leads only, not original response bytes or final admission evidence.',
        cache_search=dict(
            CRM='Bounded CRM raw cache search found adjacent Q2/Q4 FY2020 releases, no Q3 target.',
            SAP='Bounded SAP raw metadata/text search found Q3 2019 indexed material, not the Q4 target.',
            TSLA='Reuse existing issuer-hosted Q3 2019 PDF and hash-bound extracted text; acquire only release dateline page.'),
        targets_frozen_before_download=True, max_requests=3,
        network_policy='One attempt per listed URL; no retry or 403 bypass. Any 429 ends this batch for the UTC day. Preserve Retry-After. Previously recorded host stops apply on their UTC date.',
        source_only=True, events_admitted=False, calendar_complete=False,
        financial_values_extracted=0, price_values_read=0, factor_calculations=0, strategy_trials=0))


def spec():
    result = read(OUT / 'SOURCE_SPEC.json')
    for rel, expected in result['input_hashes'].items():
        assert digest((ROOT / rel).read_bytes()) == expected, rel
    return result


def collect():
    value = spec()
    assert not (OUT / 'COLLECTION.json').exists(), 'Collection is write-once; reuse prior receipts.'
    day = utc()[:10]
    # Prior stops are inspected without changing their historical files.
    stopped_hosts = set()
    for rel in value['input_hashes']:
        if not rel.endswith('SOURCE_STOPS.json'):
            continue
        prior = read(ROOT / rel)
        prior = list(prior.values()) if isinstance(prior, dict) else prior
        for row in prior:
            observed = row.get('utc_date') or row.get('retrieved_at', '')[:10]
            if observed == day and (row.get('status') or row.get('http_status')) in (403, 429):
                stopped_hosts.add(row.get('host') or urllib.parse.urlparse(row['url']).netloc)
    receipts = []
    stop_all = (OUT / ('STOPPED_' + day + '.json')).exists()
    for item in value['items']:
        receipt_path = OUT / (item['name'] + '.receipt.json')
        if receipt_path.exists():
            row = read(receipt_path)
            receipts.append(row)
            if row.get('http_status') == 429:
                stop_all = True
            continue
        row = dict(name=item['name'], requested_url=item['url'], attempted_utc=utc())
        host = urllib.parse.urlparse(item['url']).netloc
        if stop_all or host in stopped_hosts:
            row.update(status='not_attempted_due_source_stop', stop_utc_date=day)
        else:
            try:
                request = urllib.request.Request(item['url'], headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; historical-research/1.0)'})
                with urllib.request.urlopen(request, timeout=35) as response:
                    blob = response.read(8 * 1024 * 1024 + 1)
                    assert len(blob) <= 8 * 1024 * 1024, 'response exceeds bounded size'
                    row.update(http_status=response.status, final_url=response.url,
                               content_type=response.headers.get('Content-Type'),
                               etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))
                path = OUT / (item['name'] + '.html')
                with path.open('xb') as stream:
                    stream.write(blob)
                row.update(status='downloaded', path=str(path.relative_to(ROOT)), sha256=digest(blob), bytes=len(blob))
            except urllib.error.HTTPError as error:
                row.update(status='http_error', http_status=error.code,
                           final_url=error.url, retry_after=error.headers.get('Retry-After'))
                if error.code in (403, 429):
                    stopped_hosts.add(host)
                if error.code == 429:
                    stop_all = True
                    save(OUT / ('STOPPED_' + day + '.json'), dict(created_utc=utc(), reason='HTTP429', host=host, retry_after=row['retry_after']))
            except (urllib.error.URLError, TimeoutError) as error:
                row.update(status='network_error', error_type=type(error).__name__, error=str(error))
        save(receipt_path, row)
        receipts.append(row)
    save(OUT / 'COLLECTION.json', dict(created_utc=utc(), receipts=receipts,
        reused_cached_source=dict(path=CACHE + '.bin', metadata_path=CACHE + '.meta.json',
                                  extracted_text=CACHE + '.txt', extracted_text_metadata=CACHE + '.text.meta.json'),
        source_requests_attempted=sum(r['status'] != 'not_attempted_due_source_stop' for r in receipts)))
    print([(r['name'], r['status'], r.get('http_status')) for r in receipts])


def match(pattern, text):
    hit = re.search(pattern, text, re.IGNORECASE) if pattern else None
    return dict(found=bool(hit), normalized_text_offset=hit.start() if hit else None,
                excerpt=hit.group() if hit else None, pattern=pattern)


def inspect():
    value = spec()
    collection = read(OUT / 'COLLECTION.json')
    sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
    from event_schema import normalize_event
    from research_calendar import market_calendar
    calendar = market_calendar(start='2019-01-01', end='2021-01-01')
    sessions = read(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    remaining = read(ROOT / 'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json')['quarter_items']
    catalogs = {p: read(ROOT / p) for p in (
        'outputs/monthly_earnings_repair/EVENTS.json',
        'outputs/monthly_event_scope156/EVENTS.json',
        'outputs/monthly_event_patch156/ADDED_EVENTS.json')}
    meta = read(ROOT / (CACHE + '.meta.json'))
    text_meta = read(ROOT / (CACHE + '.text.meta.json'))
    assert digest((ROOT / (CACHE + '.bin')).read_bytes()) == meta['sha256'] == text_meta['source_sha256']
    assert digest((ROOT / (CACHE + '.txt')).read_bytes()) == text_meta['text_sha256']
    lines = (ROOT / (CACHE + '.txt')).read_text().splitlines()
    # Only the report identity and statement column headers are extracted.
    cached_identity = match(r'Q3\s+2019\s+Update', lines[0])
    header_lines = [dict(line=i + 1, excerpt=' '.join(line.split())) for i, line in enumerate(lines)
                    if 'Three  months ended' in line or 'Sep 30, 2019' in line]
    cached_period = any('Sep 30, 2019' in r['excerpt'] for r in header_lines) and any('Three months ended' in r['excerpt'] for r in header_lines)
    rows = []
    for item, receipt in zip(value['items'], collection['receipts']):
        assert item['name'] == receipt['name']
        row = dict(symbol=item['symbol'], fiscal_year=item['fiscal_year'], fiscal_quarter=item['fiscal_quarter'],
                   release_kind=item['release_kind'], receipt=receipt, source_checks={}, candidate=None)
        if receipt['status'] == 'downloaded':
            blob = (ROOT / receipt['path']).read_bytes()
            assert digest(blob) == receipt['sha256']
            raw = blob.decode('utf-8', errors='replace')
            parser = Text(); parser.feed(raw)
            plain = ' '.join(' '.join(parser.parts).split())
            row['source_checks'] = {field: match(item[field], plain) for field in ('title', 'issuer', 'dateline', 'actual', 'period') if item[field]}
            if item['name'] == 'tsla':
                row['source_checks']['cached_period'] = dict(found=cached_period and cached_identity['found'],
                    period_end='2019-09-30', source_path=CACHE + '.bin', source_sha256=meta['sha256'],
                    text_path=CACHE + '.txt', text_sha256=text_meta['text_sha256'],
                    identity=cached_identity, statement_header_locators=header_lines,
                    note='Q3 2019 report and three-month statement header establish period; webcast schedule is not used as publication proof.')
            row['raw_html_line_hints'] = [i + 1 for i, line in enumerate(raw.splitlines())
                if re.search(r'2019-12-03|2020-01-28|2019-10-23|Dec\. 3, 2019|28\. Januar 2020|Oct\. 23, 2019', line)]
            if all(check['found'] for check in row['source_checks'].values()):
                event = normalize_event(dict(symbol=item['symbol'], asset_type='equity', official=dict(
                    actual_period_end=item['period_end'], publication_date=item['publication_date'],
                    publication_timezone=item['timezone'], source_url=item['url'],
                    verified=True, publication_kind='issuer_release')), calendar=calendar)
                assert event['status'] == 'ready', event
                assert event['available_session'] == sessions[bisect_right(sessions, item['publication_date'])]
                assert item['before'] < item['publication_date'] < item['after']
                event.update(event_id=item['symbol'] + '_' + item['publication_date'],
                    fiscal_year=item['fiscal_year'], fiscal_quarter=item['fiscal_quarter'],
                    status='candidate_pending_independent_review', single_event_source_admitted=False,
                    first_public_time_verified=False, evidence_path=receipt['path'], evidence_sha256=receipt['sha256'],
                    publication_timezone_basis='Conservative issuer-local date from dateline location; exact release clock and CMS timezone not established.',
                    historical_archive_limitation='Current retrieved issuer original-page bytes, not a contemporaneous immutable capture.',
                    release_kind=item['release_kind'])
                row['candidate'] = event
        label = str(item['fiscal_year']) + 'Q' + str(item['fiscal_quarter'])
        row['affected_gap_records'] = [gap for gap in remaining if gap['symbol'] == item['symbol'] and label in gap['intermediate_fiscal_labels']]
        row['catalog_conflicts'] = {path: [dict(event_id=e.get('event_id'), publication_date=e.get('publication_date'),
              fiscal_year=e.get('fiscal_year'), fiscal_quarter=e.get('fiscal_quarter'), period_end=e.get('official_actual_period_end'))
              for e in events if e.get('symbol') == item['symbol'] and (
                  e.get('publication_date') == item['publication_date'] or e.get('official_actual_period_end') == item['period_end'] or
                  (e.get('fiscal_year'), e.get('fiscal_quarter')) == (item['fiscal_year'], item['fiscal_quarter']))]
              for path, events in catalogs.items()}
        row['proposal_ready_for_independent_review'] = row['candidate'] is not None and not any(row['catalog_conflicts'].values())
        rows.append(row)
    save(OUT / 'SOURCE_PROPOSAL.json', dict(created_utc=utc(), source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes()),
        collection_sha256=digest((OUT / 'COLLECTION.json').read_bytes()), results=rows,
        stage='SOURCE_ONLY_CANDIDATES_PENDING_INDEPENDENT_REVIEW', events_admitted=False, calendar_complete=False,
        scope='Only the three named missing releases. Resolving an intervening-quarter gap does not enumerate the full archive or certify window completeness.',
        publication_policy='Historical issuer dateline; conservative next full XNYS session close. Never substitute a webcast schedule, preannouncement, dateModified or current retrieval time for release publication.',
        financial_values_extracted=0, price_values_read=0, factor_calculations=0, strategy_trials=0,
        old_events_modified=False))
    print([(r['symbol'], r['proposal_ready_for_independent_review'], {k: v['found'] for k, v in r['source_checks'].items()}) for r in rows])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'collect', 'inspect'))
    {'freeze': freeze, 'collect': collect, 'inspect': inspect}[parser.parse_args().phase]()
