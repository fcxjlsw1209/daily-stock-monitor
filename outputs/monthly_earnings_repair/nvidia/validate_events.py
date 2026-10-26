"""Offline integrity, coverage and original-source reconciliation checks."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
from bs4 import BeautifulSoup
from collector import P, read, write, digest, now
from collect_events import parse_event


def main():
    events = read('events.json', [])
    catalog = read('release_catalog.json', {})
    manifest = read('manifest.json', {})
    expected = {(2015, 4)} | {(year, quarter) for year in range(2016, 2027) for quarter in range(1, 5)} | {(2027, 1), (2027, 2)}
    issues = []
    for url, record in manifest.items():
        path = P / record['evidence_path']
        assert path.exists() and digest(path.read_bytes()) == record['sha256'], url
    assert catalog['completed_pages'] == catalog['archive_last_page'] == 18
    assert all(row['http_status'] == 200 for row in catalog['archives'])
    keys = [(event['fiscal_year'], event['fiscal_quarter']) for event in events]
    assert len(keys) == len(set(keys)), 'Duplicate fiscal quarters'
    release_dates = [event['release_date'] for event in events]
    assert release_dates == sorted(release_dates) and len(release_dates) == len(set(release_dates))
    for event in events:
        record = manifest[event['source_url']]
        assert record['http_status'] == 200
        path = P / event['evidence_path']
        assert digest(path.read_bytes()) == event['evidence_sha256'] == record['sha256']
        reparsed = parse_event(record, path.read_text(), {'archive_sources': event['archive_source_urls']})
        for key in ['release_date', 'fiscal_period_end_actual', 'fiscal_year', 'fiscal_quarter']:
            assert reparsed[key] == event[key], (event['source_url'], key)
        assert '2015-01-01' <= event['release_date'] <= '2026-10-12'
        assert event['fiscal_period_end_actual'] < event['release_date']
        zone_evidence = event['publication_timezone_evidence']
        assert digest((P / zone_evidence['evidence_path']).read_bytes()) == zone_evidence['sha256']
        if event['precision'] == 'date_only':
            assert event['publication_timestamp'] is None
            assert event['publication_timezone'] == 'America/Los_Angeles'
        else:
            assert event['publication_timezone'] == 'America/New_York'
            stamp = datetime.fromisoformat(event['publication_timestamp'])
            assert stamp.tzinfo is not None and stamp.date().isoformat() == event['release_date']
            evidence = event['timestamp_evidence']
            observation = P / evidence['evidence_path']
            assert digest(observation.read_bytes()) == evidence['sha256']
            obj = json.loads(observation.read_text())
            assert obj['exact_publisher_time_visible'] is True
            assert obj['issuer_fiscal_period_end'] == event['fiscal_period_end_actual']
        assert not event['global_first_public_time_verified']
        assert not event['conference_call_time_used_as_publication']
        assert not event['modified_time_used_as_publication']
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    summary = dict(validated_at=now(), ticker='NVDA', release_date_scope=['2015-01-01', '2026-10-12'],
        requested_fiscal_quarters_count=len(expected), observed_event_count=len(events),
        earliest_release=min(release_dates) if release_dates else None,
        latest_release=max(release_dates) if release_dates else None,
        complete_fiscal_quarter_sequence=not missing and not extra,
        missing_fiscal_quarters=missing, unexpected_fiscal_quarters=extra,
        releases_by_calendar_year=dict(sorted(Counter(event['release_date'][:4] for event in events).items())),
        timestamp_precision_counts=dict(Counter(event['precision'] for event in events)),
        all_manifest_bytes_verified=True, original_article_reparse_match=True,
        archive_pages_enumerated=catalog['completed_pages'], catalog_release_urls=len(catalog['releases']),
        saved_source_records=len(manifest), http_status_counts=dict(Counter(str(v['http_status']) for v in manifest.values())),
        parse_issues=read('parse_issues.json', []),
        collection_gaps=['Exact publication times remain unknown for date-only issuer pages; no midnight is inferred',
            'Two reused original newswire timestamps establish publisher time, not earliest disclosure across every channel'],
        alpha_vantage_requests=0, blocked_sources=read('blocked_sources.json', {}),
        input_hashes={name: digest((P / name).read_bytes()) for name in ['events.json', 'manifest.json', 'release_catalog.json', 'collector.py', 'build_catalog.py', 'collect_events.py']})
    write('VALIDATION.json', summary)
    print(json.dumps({key: summary[key] for key in ['observed_event_count', 'complete_fiscal_quarter_sequence', 'missing_fiscal_quarters', 'unexpected_fiscal_quarters', 'timestamp_precision_counts', 'earliest_release', 'latest_release', 'http_status_counts']}, indent=2))
    if missing or extra or summary['parse_issues']:
        raise SystemExit('Coverage or parse gaps remain; see VALIDATION.json')


if __name__ == '__main__':
    main()
