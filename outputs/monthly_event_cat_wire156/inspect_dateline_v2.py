"""Offline parser correction for whitespace between adjacent HTML span nodes."""
import json
from pathlib import Path
import sys
from bisect import bisect_right

import after155_cat_wire_source as source
from after155_broadcom_gap_source import Text, digest, read, save, utc

ROOT = source.ROOT
OUT = source.OUT


def main():
    spec = source.checked_spec()
    original = read(OUT / 'SOURCE_PROPOSAL.json')
    sys.path.insert(0, str(ROOT / 'outputs/monthly_earnings_repair'))
    from event_schema import normalize_event
    from research_calendar import market_calendar
    calendar = market_calendar(start='2025-01-01', end='2026-01-01')
    sessions = read(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    rows = []
    for target, previous in zip(spec['targets'], original['results']):
        assert target['key'] == previous['key']
        receipt = previous['receipt']
        assert receipt['status'] == 'downloaded' and receipt['http_status'] == 200
        raw = (ROOT / receipt['path']).read_bytes()
        assert digest(raw) == receipt['sha256']
        parser = Text(); parser.feed(raw.decode('utf-8'))
        plain = ' '.join(' '.join(parser.parts).split())
        row = json.loads(json.dumps(previous))
        pattern = r'IRVING,\s*Texas\s*,\s*' + target['dateline_month'] + r'\s+0?' + str(target['dateline_day']) + r',\s*2025'
        row['source_checks']['dateline'] = source.hit(pattern, plain)
        assert all(check['found'] for check in row['source_checks'].values())
        row['source_display_timestamp'] = source.hit(
            r'(?:Apr(?:il)?|Aug(?:ust)?)\.?\s+0?' + str(target['dateline_day']) + r',\s*2025,\s*\d{1,2}:\d{2}\s*ET', plain)
        assert row['source_display_timestamp']['found']
        event = normalize_event(dict(symbol='CAT', official=dict(actual_period_end=target['period_end'],
            publication_date=target['publication_date'], publication_timezone='America/Chicago',
            source_url=target['url'], verified=True, publication_kind='newswire_release')), calendar=calendar)
        assert event['status'] == 'ready'
        assert event['available_session'] == sessions[bisect_right(sessions, target['publication_date'])]
        assert not any(row['catalog_conflicts'].values())
        assert row['affected_gap_records']
        assert all(gap['before_publication'] < target['publication_date'] < gap['after_publication'] for gap in row['affected_gap_records'])
        event.update(event_id='CAT_' + target['publication_date'], fiscal_year=2025,
            fiscal_quarter=target['fiscal_quarter'], evidence_path=receipt['path'], evidence_sha256=receipt['sha256'],
            source_kind='issuer_distributed_original_earnings_release', status='proposal_pending_independent_review',
            event_admitted=False, first_public_time_verified=False,
            source_display_timestamp=row['source_display_timestamp']['quote'],
            source_display_timestamp_used_as_first_public_time=False,
            publication_timezone_basis='Conservative issuer-local date interpretation from IRVING, Texas dateline; printed ET release time retained, not certified as the first public instant.',
            historical_archive_limitation='Later-retrieved issuer-distributed original release, not a contemporaneous immutable capture.')
        row['candidate'] = event
        row['ready_for_independent_source_review'] = True
        rows.append(row)
    save(OUT / 'SOURCE_PROPOSAL_V2.json', dict(created_utc=utc(), results=rows,
        source_spec_sha256=digest((OUT / 'SOURCE_SPEC.json').read_bytes()),
        collection_sha256=digest((OUT / 'COLLECTION.json').read_bytes()),
        original_proposal_preserved=True, original_proposal_sha256=digest((OUT / 'SOURCE_PROPOSAL.json').read_bytes()),
        parser_path=str(Path(__file__).relative_to(ROOT)), parser_sha256=digest(Path(__file__).read_bytes()),
        revision='Only tolerate span-boundary whitespace before dateline comma and zero-padded displayed day; originals, target dates, periods and acquisition scope unchanged.',
        scope='Two CAT issuer-distributed actual results releases pending independent review; no event ingestion or complete calendar assertion.',
        events_admitted=False, calendar_complete=False, network_requests=0, financial_values_extracted=0,
        price_values_read=0, factor_calculations=0, returns=0, SEC_stop_preserved=True, IR_failures_preserved=True))
    print([(row['key'], row['candidate']['known_by_at']) for row in rows])


if __name__ == '__main__':
    main()
