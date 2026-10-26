"""Read-only independent HSBC source and availability checks; no calculations of financial values."""
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCH = 'outputs/monthly_event_gap156_cat_hsbc/'
CACHE = 'outputs/monthly_earnings_repair/batch_financial_consumer/HSBC/raw/'
RUNTIME = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(RUNTIME))
sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
from pypdf import PdfReader
import pypdf
import pypdfium2
from research_calendar import market_calendar
from event_schema import normalize_event


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def load(rel):
    return json.loads((ROOT / rel).read_text())


def norm(value):
    return ' '.join(value.split())


def main():
    proposal_raw = (ROOT / (BATCH + 'PROPOSAL.json')).read_bytes()
    spec_raw = (ROOT / (BATCH + 'SOURCE_SPEC.json')).read_bytes()
    assert sha(proposal_raw) == 'f3819ca884fb7a35fcd8cb7265e2e6224a93e4c20bd3ada1ee917bf3a6f15137'
    assert sha(spec_raw) == 'ee52950089a463576fefafb57d0ef4ad12557d9fbe919bc9fdb5f4590d456344'
    proposal = json.loads(proposal_raw)
    spec = json.loads(spec_raw)
    collection = load(BATCH + 'COLLECTION.json')
    catalogs = {name: load(name) for name in (
        'outputs/monthly_earnings_repair/EVENTS.json',
        'outputs/monthly_event_scope156/EVENTS.json',
        'outputs/monthly_event_patch156/ADDED_EVENTS.json')}
    remaining = load('outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json')
    frozen_sessions = load('outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    cal = market_calendar(start='2016-01-01', end='2018-01-01')
    observed = []
    inputs = set(catalogs) | {BATCH + f for f in ('PROPOSAL.json', 'SOURCE_SPEC.json', 'COLLECTION.json')}
    inputs |= {'outputs/monthly_event_overlay_scope156/REMAINING_ITEMS.json',
               'outputs/monthly_paired_target143/CALENDAR.json',
               'outputs/monthly_earnings_repair/research_calendar.py',
               'outputs/monthly_earnings_repair/event_schema.py',
               'outputs/index_union/CALENDAR_EXCEPTIONS.json',
               'outputs/index_union/sources/closure_20250109.html',
               'work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py',
               'run_local.sh', 'local_run.py', 'local.sandbox.sb',
               str(Path(__file__).relative_to(ROOT))}
    targets = [
        dict(key='HSBC_2016H1', file='dc7a551d806bc6656921.pdf', date='2016-08-03',
             end='2016-06-30', quarter=2, date_text='3 August 2016', title='2016 INTERIM RESULTS',
             period='Half-year to 30 Jun', actual='we performed reasonably well in the first half',
             period_type='interim_half_year_results'),
        dict(key='HSBC_2016FY', file='a12baac8837504183928.pdf', date='2017-02-21',
             end='2016-12-31', quarter=4, date_text='21 February 2017', title='2016 RESULTS',
             period='Year ended 31 Dec', actual='Financial performance',
             period_type='annual_results'),
    ]
    for target in targets:
        pdf_path = CACHE + target['file']
        meta_path = pdf_path + '.meta.json'
        inputs |= {pdf_path, meta_path, BATCH + target['key'] + '_page1.png'}
        blob = (ROOT / pdf_path).read_bytes()
        meta = load(meta_path)
        assert meta['status'] == 200 and meta['content_type'] == 'application/pdf'
        assert sha(blob) == meta['sha256'] == spec['input_hashes'][pdf_path]
        assert len(blob) == meta['bytes'] and blob.startswith(b'%PDF-')
        assert sha((ROOT / meta_path).read_bytes()) == spec['input_hashes'][meta_path]
        receipt = next(r for r in collection['receipts'] if r['key'] == target['key'])
        assert receipt['sha256'] == sha(blob) and receipt['original_receipt_sha256'] == sha((ROOT / meta_path).read_bytes())
        assert receipt['url'] == meta['url'] and receipt['retrieved_utc'] == meta['retrieved_at']
        reader = PdfReader(io.BytesIO(blob))
        page_text = reader.pages[0].extract_text(extraction_mode='layout')
        normalized = norm(page_text)
        evidence = {}
        for field, needle in [('date', target['date_text']), ('issuer', 'HSBC HOLDINGS PLC'),
                              ('title', target['title']), ('period', target['period']),
                              ('actual_results', target['actual'])]:
            offset = normalized.casefold().find(needle.casefold())
            assert offset >= 0, (target['key'], field)
            evidence[field] = dict(page_1based=1, normalized_offset=offset,
                                   quote=normalized[offset:offset + len(needle)])
        page_lines = page_text.splitlines()
        at = next(i for i, line in enumerate(page_lines) if target['period'].casefold() in norm(line).casefold())
        evidence['period_column_header'] = dict(page_1based=1, lines=[norm(line) for line in page_lines[at:at+3]])
        assert '2016' in ' '.join(evidence['period_column_header']['lines'])
        render = io.BytesIO()
        document = pypdfium2.PdfDocument(blob)
        document[0].render(scale=1.25).to_pil().save(render, format='PNG')
        png_path = BATCH + target['key'] + '_page1.png'
        assert sha(render.getvalue()) == sha((ROOT / png_path).read_bytes()), 'Independent page render differs'
        expected = next(e for e in proposal['events'] if e['publication_date'] == target['date'])
        normalized_event = normalize_event(dict(symbol='HSBC', asset_type='equity', official=dict(
            actual_period_end=target['end'], publication_date=target['date'],
            publication_timezone='Europe/London', source_url=meta['url'], verified=True,
            publication_kind='issuer_release')), calendar=cal)
        assert normalized_event['status'] == 'ready'
        for key in ('publication_date', 'official_actual_period_end', 'available_session', 'date_only_upper_bound_at'):
            assert normalized_event[key] == expected[key], key
        assert normalized_event['known_by_at'] == expected['proposed_known_by_at']
        next_session = frozen_sessions[bisect_right(frozen_sessions, target['date'])]
        assert next_session == normalized_event['available_session']
        close = cal.session_close(next_session)
        assert close.tz_convert('America/New_York').strftime('%H:%M') == '16:00'
        conflicts = {}
        for path, events in catalogs.items():
            conflicts[path] = [dict(event_id=e.get('event_id'), publication_date=e.get('publication_date'),
                                   fiscal_year=e.get('fiscal_year'), fiscal_quarter=e.get('fiscal_quarter'),
                                   period_end=e.get('official_actual_period_end')) for e in events
                if e.get('symbol') == 'HSBC' and (e.get('event_id') == 'HSBC_' + target['date'] or
                    e.get('publication_date') == target['date'] or e.get('official_actual_period_end') == target['end'] or
                    (e.get('fiscal_year'), e.get('fiscal_quarter')) == (2016, target['quarter']))]
        assert not any(conflicts.values())
        gap_records = [r for r in remaining['quarter_items'] if r['symbol'] == 'HSBC' and
                       ('2016Q' + str(target['quarter'])) in r['intermediate_fiscal_labels']]
        assert gap_records and all(r['before_publication'] < target['date'] < r['after_publication'] for r in gap_records)
        observed.append(dict(event_id='HSBC_' + target['date'], fiscal_year=2016, endpoint_quarter=target['quarter'],
            period_type=target['period_type'], pdf_path=pdf_path, pdf_sha256=sha(blob), pdf_pages=len(reader.pages),
            bytes=len(blob), original_receipt=meta, original_receipt_path=meta_path,
            original_receipt_sha256=sha((ROOT / meta_path).read_bytes()),
            independently_extracted_evidence=evidence,
            independent_render_matches_producer_png=True, rendered_page_path=png_path,
            rendered_page_sha256=sha(render.getvalue()), availability=normalized_event,
            frozen_calendar_next_session=next_session, actual_xnys_session_close=close.isoformat(),
            xnys_early_close=False, conflict_checks=conflicts, affected_gap_records=gap_records))
    cat_receipts = [dict(key=r['key'], http_status=r['http_status'], status=r['status'],
                         content_type=r.get('content_type')) for r in collection['receipts'] if r['key'].startswith('CAT')]
    assert len(cat_receipts) == 2 and all(r['http_status'] == 403 for r in cat_receipts)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method='Independent pypdf page-1 extraction and PDFium in-memory rendering from original bytes; no producer marker/extraction code called. Clock recomputed with frozen research calendar and event policy.',
        input_hashes={p: sha((ROOT / p).read_bytes()) for p in sorted(inputs)},
        runtime=dict(python=sys.version, pypdf_version=pypdf.__version__,
            pypdf_module_sha256=sha(Path(pypdf.__file__).read_bytes()),
            pypdfium2_module_sha256=sha(Path(pypdfium2.__file__).read_bytes())),
        checked_events=observed, cat_exclusion=cat_receipts, network_requests=0,
        financial_values_extracted=0, price_values_read=0, factor_calculations=0, returns_calculated=0,
        old_artifacts_modified=False)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
