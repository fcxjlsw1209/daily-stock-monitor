"""Offline original-source, provenance and requested-period coverage checks."""
from collections import Counter
import json
from collector import P, read, write, digest, now
from collect_events import parse_event, RESEARCH_CUTOFF


def main():
    events = read('events.json', [])
    manifest = read('manifest.json', {})
    catalog = read('release_catalog.json', {})
    expected = {(2014, 4)} | {(y, q) for y in range(2015, 2026) for q in range(1, 5)} | {(2026, 1), (2026, 2)}
    keys = [(e['fiscal_year'], e['fiscal_quarter']) for e in events]
    assert len(keys) == len(set(keys)), 'Duplicate quarter'
    assert [e['release_date'] for e in events] == sorted(e['release_date'] for e in events)
    for url, record in manifest.items():
        assert digest((P / record['evidence_path']).read_bytes()) == record['sha256'], url
    for event in events:
        record = manifest[event['source_url']]
        assert record['http_status'] == 200
        assert digest((P / event['evidence_path']).read_bytes()) == event['evidence_sha256'] == record['sha256']
        reparsed = parse_event(record, (P / event['evidence_path']).read_text(), {'archive_sources': event['archive_source_urls']})
        assert reparsed == event, event['source_url']
        assert event['precision'] == 'date_only' and event['publication_timestamp'] is None
        assert event['publication_timezone'] == 'America/Los_Angeles'
        z = event['publication_timezone_evidence']
        assert digest((P / z['evidence_path']).read_bytes()) == z['sha256']
        assert '2015-01-01' <= event['release_date'] <= RESEARCH_CUTOFF
        assert not event['filing_date_used_as_publication'] and not event['acceptance_timestamp_used_as_publication']
    absent = sorted(expected - set(keys))
    pending = read('pending.json', [])
    summary = dict(validated_at=now(), ticker='AMZN', validation_passed=True,
        release_date_scope=['2015-01-01', RESEARCH_CUTOFF], expected_quarter_count=len(expected),
        accepted_event_count=len(events), complete_fiscal_quarter_sequence=not absent,
        missing_fiscal_quarters=[{'fiscal_year': y, 'fiscal_quarter': q} for y, q in absent],
        unexpected_fiscal_quarters=sorted(set(keys) - expected),
        earliest_release=min((e['release_date'] for e in events), default=None),
        latest_release=max((e['release_date'] for e in events), default=None),
        releases_by_calendar_year=dict(sorted(Counter(e['release_date'][:4] for e in events).items())),
        original_dateline_overrides_migrated_header=[{'release_date': e['release_date'], 'header_date': e['migrated_header_date'], 'url': e['source_url']} for e in events if e['migrated_header_date_conflicts_with_original']],
        archive_pages_enumerated=len(catalog['archives']), catalog_release_count=len(catalog['releases']),
        pending_catalog_entries=len(pending), pending_reasons=dict(Counter(e['reason'] for e in pending)),
        all_source_checksums_verified=True, all_events_reparsed_identically=True,
        saved_source_records=len(manifest), http_status_counts=dict(Counter(str(r['http_status']) for r in manifest.values())),
        blocked_sources=read('blocked_sources.json', {}), alpha_vantage_requests=0,
        sec_candidate_count=len(read('sec_candidates.json', {})['candidates']),
        sec_filing_or_acceptance_dates_used_as_release_dates=0,
        input_hashes={name: digest((P / name).read_bytes()) for name in ['collector.py', 'build_catalog.py', 'collect_events.py', 'validate_events.py', 'events.json', 'manifest.json', 'release_catalog.json', 'pending.json', 'sec_candidates.json']})
    assert not summary['unexpected_fiscal_quarters']
    write('VALIDATION.json', summary)
    print(json.dumps({k: summary[k] for k in ['validation_passed', 'expected_quarter_count', 'accepted_event_count', 'missing_fiscal_quarters', 'original_dateline_overrides_migrated_header', 'http_status_counts']}, indent=2))


if __name__ == '__main__':
    main()
