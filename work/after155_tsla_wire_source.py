"""One separately authorized issuer-distributed Tesla release; source metadata only."""
import argparse
from bisect import bisect_right
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from bs4 import BeautifulSoup
from after155_broadcom_gap_source import ROOT, digest, read, save, utc

OUT = ROOT / 'outputs/monthly_event_tsla_wire156'
URL = 'https://www.globenewswire.com/news-release/2019/10/23/1934484/0/en/Tesla-Releases-Third-Quarter-2019-Financial-Results.html'
CACHE = 'outputs/monthly_earnings_repair/batch_technology_industrial/TSLA/raw/de691e5b9b29307fe7de'
CATALOGS = ('outputs/monthly_earnings_repair/EVENTS.json',
            'outputs/monthly_event_scope156/EVENTS.json',
            'outputs/monthly_event_patch156/ADDED_EVENTS.json')


def freeze():
    paths = list(CATALOGS) + [
        str(Path(__file__).relative_to(ROOT)), 'work/after155_broadcom_gap_source.py',
        'outputs/monthly_event_gap156_crm_sap_tsla/tsla.receipt.json',
        'work/after155_sec_gap_discovery.json', 'work/after155_sec_gap_discovery_collection.json',
        'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json',
        'outputs/monthly_paired_target143/CALENDAR.json',
        'outputs/monthly_earnings_repair/event_schema.py',
        'outputs/monthly_earnings_repair/research_calendar.py',
        'outputs/index_union/CALENDAR_EXCEPTIONS.json',
        'outputs/index_union/sources/closure_20250109.html',
        'work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py',
        'run_local.sh', 'local_run.py', 'local.sandbox.sb',
    ] + [CACHE + suffix for suffix in ('.bin', '.meta.json', '.txt', '.text.meta.json')]
    save(OUT / 'SOURCE_SPEC.json', dict(schema_version='tsla-wire-source/v1', created_utc=utc(),
        target=dict(symbol='TSLA', fiscal_year=2019, fiscal_quarter=3, expected_publication_date='2019-10-23',
                    expected_period_end='2019-09-30', url=URL),
        source_class='issuer_distributed_original_release_GlobeNewswire',
        authorization='Separately authorized single public GlobeNewswire source supplied by parent task. Not a retry or alternate route to rejected SEC or Tesla IR endpoints.',
        source_selection='Missing 2019Q3 label between original 2019-07-24 and 2020-01-29 events; no price or return selection.',
        input_hashes={p: digest((ROOT / p).read_bytes()) for p in paths},
        network_policy='One normal GET to this exact URL, no redirects or retries. Preserve response bytes, status, headers and Retry-After. Any 403/429 stops this source for the UTC day.',
        cached_pdf_policy='Reuse existing official PDF and its hash-bound extracted text solely for quarter identity and period-end headers; no new PDF or linked-document request.',
        clock_policy='Retain displayed wire time as wire metadata only. Use original date interpreted conservatively in issuer-local America/Los_Angeles time and next full XNYS session EOD; never assert the wire time was first publication.',
        SEC_requests_permitted=False, IR_requests_permitted=False,
        events_admitted=False, complete_calendar=False, price_values_read=0, financial_values_extracted=0,
        factor_calculations=0, returns_calculated=0))


def checked_spec():
    spec = read(OUT / 'SOURCE_SPEC.json')
    assert spec['target']['url'] == URL
    for path, expected in spec['input_hashes'].items():
        assert digest((ROOT / path).read_bytes()) == expected, path
    return spec


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def collect():
    checked_spec()
    # A started attempt, even if interrupted, prevents a second network request.
    save(OUT / 'ATTEMPT_STARTED.json', dict(created_utc=utc(), url=URL, max_requests=1))
    row = dict(requested_url=URL, request_utc=utc(), request_method='GET', retries=0, redirects_allowed=False)
    blob = None
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0 (compatible; historical-research/1.0)'})
        opener = urllib.request.build_opener(NoRedirect)
        with opener.open(request, timeout=35) as response:
            blob = response.read(8 * 1024 * 1024 + 1)
            row.update(http_status=response.status, final_url=response.url, status='downloaded',
                       content_type=response.headers.get('Content-Type'), retry_after=response.headers.get('Retry-After'),
                       etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))
            if len(blob) > 8 * 1024 * 1024:
                row['status'] = 'response_exceeds_limit'
    except urllib.error.HTTPError as error:
        blob = error.read(8 * 1024 * 1024)
        row.update(status='http_error', http_status=error.code, final_url=error.url,
                   content_type=error.headers.get('Content-Type'), retry_after=error.headers.get('Retry-After'),
                   location=error.headers.get('Location'))
    except (urllib.error.URLError, TimeoutError) as error:
        row.update(status='network_error', error_type=type(error).__name__, error=str(error))
    if blob is not None:
        path = OUT / 'issuer_distributed_release.html'
        with path.open('xb') as stream:
            stream.write(blob)
        row.update(path=str(path.relative_to(ROOT)), sha256=digest(blob), bytes=len(blob))
    if row.get('http_status') in (403, 429):
        save(OUT / ('STOPPED_' + row['request_utc'][:10] + '.json'), dict(created_utc=utc(),
            host='www.globenewswire.com', reason='HTTP' + str(row['http_status']), retry_after=row.get('retry_after'),
            further_requests_permitted=False))
    save(OUT / 'RECEIPT.json', row)
    save(OUT / 'COLLECTION.json', dict(created_utc=utc(), request_count=1, receipts=[row],
        source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes())))
    print({key: row.get(key) for key in ('status', 'http_status', 'bytes', 'sha256')})


def hit(pattern, text):
    match = re.search(pattern, text, re.I)
    return dict(found=bool(match), pattern=pattern, normalized_offset=match.start() if match else None,
                excerpt=match.group() if match else None)


def inspect():
    spec = checked_spec()
    collection = read(OUT / 'COLLECTION.json')
    assert collection['source_spec_sha256'] == digest((OUT / 'SOURCE_SPEC.json').read_bytes())
    receipt = read(OUT / 'RECEIPT.json')
    assert collection['receipts'] == [receipt]
    meta = read(ROOT / (CACHE + '.meta.json'))
    text_meta = read(ROOT / (CACHE + '.text.meta.json'))
    assert digest((ROOT / (CACHE + '.bin')).read_bytes()) == meta['sha256'] == text_meta['source_sha256']
    assert digest((ROOT / (CACHE + '.txt')).read_bytes()) == text_meta['text_sha256']
    assert meta['status'] == 200 and meta['content_type'] == 'application/pdf'
    lines = (ROOT / (CACHE + '.txt')).read_text().splitlines()
    identity = hit(r'Q3\s+2019\s+Update', lines[0])
    headers = [dict(line=i+1, text=' '.join(line.split())) for i, line in enumerate(lines)
               if 'Three  months ended' in line or ('Sep 30, 2019' in line and 'In millions of USD or shares' in line)]
    period_ok = identity['found'] and any('Three months ended' in r['text'] for r in headers) and any('Sep 30, 2019' in r['text'] for r in headers)
    pdf_evidence = dict(found=period_ok, source_path=CACHE + '.bin', source_sha256=meta['sha256'],
        source_url=meta['url'], receipt_path=CACHE + '.meta.json', text_path=CACHE + '.txt',
        text_sha256=text_meta['text_sha256'], cached_identity=identity, statement_headers=headers,
        period_end='2019-09-30', extraction_scope='Identity and period headers only; existing hash-bound PDF text reused.')
    checks = {}
    candidate = None
    display_time = None
    line_hints = []
    if receipt['status'] == 'downloaded' and receipt['http_status'] == 200:
        blob = (ROOT / receipt['path']).read_bytes()
        assert digest(blob) == receipt['sha256']
        raw = blob.decode('utf-8', errors='replace')
        soup = BeautifulSoup(raw, 'html.parser')
        for hidden in soup(['script', 'style', 'noscript']):
            hidden.decompose()
        plain = ' '.join(soup.get_text(' ', strip=True).split())
        checks = {name: hit(pattern, plain) for name, pattern in dict(
            title=r'Tesla Releases Third Quarter 2019 Financial Results',
            issuer=r'Source:\s*Tesla,?\s*Inc\.?',
            actual_release=r'Tesla has released its financial results for the third quarter of 2019',
            dateline=r'PALO ALTO, Calif\.,?\s*Oct(?:ober)?\.?\s*23,?\s*2019',
        ).items()}
        checks['cached_official_pdf_period'] = pdf_evidence
        display_time = hit(r'Oct(?:ober)?\.?\s*23,?\s*2019\s*16:54\s*(?:ET|EST|EDT)', plain)
        line_hints = [i+1 for i, line in enumerate(raw.splitlines())
                      if 'Tesla has released' in line or 'PALO ALTO' in line or '16:54' in line]
        if all(check['found'] for check in checks.values()):
            sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
            from event_schema import normalize_event
            from research_calendar import market_calendar
            calendar = market_calendar(start='2019-01-01', end='2020-02-01')
            candidate = normalize_event(dict(symbol='TSLA', official=dict(actual_period_end='2019-09-30',
                publication_date='2019-10-23', publication_timezone='America/Los_Angeles', source_url=URL,
                verified=True, publication_kind='newswire_release')), calendar=calendar)
            sessions = read(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
            assert candidate['status'] == 'ready'
            assert candidate['available_session'] == sessions[bisect_right(sessions, '2019-10-23')] == '2019-10-24'
            assert candidate['known_by_at'] == '2019-10-24T20:00:00+00:00'
            candidate.update(event_id='TSLA_2019-10-23', fiscal_year=2019, fiscal_quarter=3,
                status='source_proposal_pending_independent_review', event_admitted=False,
                evidence_path=receipt['path'], evidence_sha256=receipt['sha256'],
                source_kind='issuer_distributed_actual_results_release_with_cached_official_update',
                first_public_time_verified=False, publication_timezone_basis='Conservative issuer-local date from PALO ALTO dateline; displayed wire clock is retained only as source metadata.',
                historical_capture_limitation='Later-retrieved original issuer-distributed release and existing official PDF; not a contemporaneous immutable capture.')
    catalogs = {p: read(ROOT / p) for p in CATALOGS}
    conflicts = {p: [dict(event_id=e.get('event_id'), publication_date=e.get('publication_date'),
            period_end=e.get('official_actual_period_end'), fiscal_year=e.get('fiscal_year'), fiscal_quarter=e.get('fiscal_quarter'))
        for e in events if e.get('symbol') == 'TSLA' and (e.get('publication_date') == '2019-10-23' or
            e.get('official_actual_period_end') == '2019-09-30' or (e.get('fiscal_year'), e.get('fiscal_quarter')) == (2019,3))]
        for p, events in catalogs.items()}
    gaps = [g for g in read(ROOT / 'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json')['quarter_items']
            if g['symbol'] == 'TSLA' and '2019Q3' in g['intermediate_fiscal_labels']]
    save(OUT / 'SOURCE_PROPOSAL.json', dict(schema_version='tsla-wire-source/v1', created_utc=utc(),
        source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes()),
        collection_sha256=digest((OUT / 'COLLECTION.json').read_bytes()),
        receipt_sha256=digest((OUT / 'RECEIPT.json').read_bytes()),
        source_checks=checks, displayed_wire_time=display_time, raw_html_line_hints=line_hints,
        cached_pdf_evidence=pdf_evidence, candidate=candidate, catalog_conflicts=conflicts, affected_gap_records=gaps,
        ready_for_independent_source_review=candidate is not None and not any(conflicts.values()),
        scope='Only the TSLA 2019Q3 actual release. Does not establish complete historical calendar, first-ever release time or event admission.',
        SEC_and_IR_stop_records_preserved=True, events_admitted=False, complete_calendar=False,
        financial_values_extracted=0, price_values_read=0, factor_calculations=0, returns_calculated=0))
    print(dict(candidate_ready=candidate is not None, conflicts=any(conflicts.values()),
               checks={k:v['found'] for k,v in checks.items()}, displayed_wire_time=display_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'collect', 'inspect'))
    {'freeze': freeze, 'collect': collect, 'inspect': inspect}[parser.parse_args().phase]()
