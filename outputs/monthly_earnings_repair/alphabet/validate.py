"""Offline evidence, fiscal sequence and common-schema validation."""
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys

from collect import AS_OF, ARCHIVE_DOC_PATH, INDEX, extract_pdf, parse_text, targets
from fetch import HERE, ROOT, RAW, dump, sha


def main():
    events_path = HERE / 'events.json'
    events = json.loads(events_path.read_text())
    wanted = {(x['fiscal_year'], x['fiscal_quarter']): x for x in targets()}
    errors = []; evidence_manifest = []
    assert len(events) == len({(e['fiscal_year'], e['fiscal_quarter']) for e in events}), 'duplicate fiscal quarter'
    sys.path.insert(0, str(HERE.parent))
    from event_schema import normalize_event
    normalized = []
    for e in events:
        try:
            target = wanted[(e['fiscal_year'], e['fiscal_quarter'])]
            p = ROOT / e['evidence_path']; b = p.read_bytes()
            meta = json.loads(p.with_suffix('.meta.json').read_text())
            assert b.startswith(b'%PDF-') and meta['http_status'] == 200
            assert sha(b) == e['evidence_sha256'] == meta['sha256']
            text = extract_pdf(p)
            assert parse_text(text, target, e['source_url'], e['evidence_path'], sha(b)) == e
            assert e['release_date'] <= AS_OF and e['fiscal_period_end_actual'] < e['release_date']
            assert e['publication_timezone'] == 'America/Los_Angeles'
            assert e['publication_timezone_evidence']['cms_timezone_observed'] is False
            assert e['publication_timestamp'] is None
            expected_issuer = 'Google Inc.' if (e['fiscal_year'], e['fiscal_quarter']) <= (2015, 2) else 'Alphabet Inc.'
            assert e['issuer_name_at_release'] == expected_issuer, 'Original predecessor/successor issuer mismatch'
            assert expected_issuer.lower() in e['issuer_identity_evidence'].lower()
            if 'archive_copy' in e['source_class']:
                assert e['archive_provenance_sha256'] == sha(ARCHIVE_DOC_PATH.read_bytes())
                assert 'alphabet-investor-pdfs' in ARCHIVE_DOC_PATH.read_text()
            n = normalize_event({'symbol': 'GOOGL', 'asset_type': 'equity', 'vendor': {},
                'official': {'verified': True, 'actual_period_end': e['fiscal_period_end_actual'],
                    'publication_date': e['release_date'], 'publication_at': None,
                    'publication_timezone': e['publication_timezone'],
                    'publication_timezone_basis': e['publication_timezone_basis'],
                    'source_url': e['source_url'], 'publication_kind': 'release'}})
            assert n['status'] == 'ready', 'common event schema rejected: ' + str(n)
            assert n['available_session'] > e['release_date']
            normalized.append(n)
            evidence_manifest.append({'source_url': e['source_url'], 'path': e['evidence_path'], 'sha256': sha(b)})
        except (AssertionError, ValueError, KeyError) as err:
            errors.append({'fiscal_year': e['fiscal_year'], 'fiscal_quarter': e['fiscal_quarter'], 'error': str(err)})
    got = {(e['fiscal_year'], e['fiscal_quarter']) for e in events}
    expected = {(y, q) for y in range(2011, 2027) for q in range(1, 5 if y < 2026 else 3)}
    gaps = sorted(expected - got)
    report = {
        'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'research_cutoff': AS_OF,
        'verified_event_count': len(events), 'expected_fiscal_quarters': len(expected),
        'first_release_date': min(e['release_date'] for e in events), 'last_release_date': max(e['release_date'] for e in events),
        'release_dates_2015_to_cutoff': sum(e['release_date'] >= '2015-01-01' for e in events),
        'release_dates_2016_to_cutoff': sum(e['release_date'] >= '2016-01-01' for e in events),
        'fiscal_coverage': 'FY2011Q1 through FY2026Q2', 'source_classes': dict(Counter(e['source_class'] for e in events)),
        'date_only_count': sum(e['precision'] == 'date_only' for e in events), 'exact_publication_timestamps': 0,
        'explicit_timezone_count': sum(e['publication_timezone'] == 'America/Los_Angeles' for e in events),
        'original_announcing_issuers': dict(Counter(e['issuer_name_at_release'] for e in events)),
        'schema_ready_count': len(normalized), 'missing_fiscal_quarters': gaps, 'validation_errors': errors,
        'release_date_and_actual_fiscal_period_ready': not gaps and not errors,
        'events_sha256': sha(events_path.read_bytes()), 'public_archive_index_sha256': sha(INDEX.read_bytes()),
        'official_archive_documentation_sha256': sha(ARCHIVE_DOC_PATH.read_bytes()),
        'network_requests_from_validator': 0,
        'limitations': ['date-only timing requires conservative next-session availability',
            'historical copies are preserved Google official sample documents, not a claim that inaccessible legacy IR URLs currently resolve',
            'no point-in-time analyst estimates, EPS-surprise labels or advance-announcement availability',
            'does not claim all historical universe issuers are covered or an earnings factor has been backtested'],
    }
    dump(HERE / 'schema_events.json', normalized)
    dump(HERE / 'EVIDENCE_MANIFEST.json', evidence_manifest)
    dump(HERE / 'VALIDATION.json', report)
    print(json.dumps(report, indent=2))
    if gaps or errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
