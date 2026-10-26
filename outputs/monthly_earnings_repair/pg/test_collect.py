"""Offline semantic and full-cache provenance tests. No network requests."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('pg_collection',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def release(date='October 24, 2025',quarter='first',fy=2026,end='September 30, 2025'):
    return 'News Release P&G ANNOUNCES '+quarter.upper()+' QUARTER RESULTS. CINCINNATI, '+date+' - The Procter & Gamble Company reported '+quarter+' quarter fiscal year '+str(fy)+' results. Three Months Ended '+end+' Net Sales.'

class ParserTests(unittest.TestCase):
    def test_fiscal_year_not_calendar_year(self):
        r=c.parse_release(release());self.assertEqual((r['fiscal_year'],r['fiscal_quarter'],r['fiscal_period_end_actual']),(2026,1,'2025-09-30'))
        r=c.parse_release(release('January 22, 2026','second',2026,'December 31, 2025'))
        self.assertEqual((r['fiscal_year'],r['fiscal_quarter']),(2026,2))
    def test_publication_date_is_body_dateline_not_metadata(self):
        r=c.parse_release('Received: 10/25/2025 12:00:00. dateModified: October 26, 2025. '+release())
        self.assertEqual(r['release_date'],'2025-10-24');self.assertNotIn('publication_timestamp',r)
    def test_no_dateline_no_event_even_with_filing_and_call_dates(self):
        with self.assertRaisesRegex(ValueError,'dateline'):
            c.parse_release('Received October 24, 2025. P&G first quarter fiscal year 2026 reported earnings. Conference call October 24, 2025. Three Months Ended September 30, 2025.')
    def test_scheduled_release_rejected(self):
        with self.assertRaisesRegex(ValueError,'Scheduled'):
            c.parse_release(release().replace('reported','will release reported'))
    def test_conflicting_quarter_and_period_rejected(self):
        with self.assertRaisesRegex(ValueError,'disagree'):
            c.parse_release(release(end='June 30, 2025'))
    def test_actual_period_and_fiscal_year_required(self):
        with self.assertRaisesRegex(ValueError,'calendar date'):
            c.parse_release(release().replace('Three Months Ended September 30, 2025',''))
        with self.assertRaisesRegex(ValueError,'fiscal year'):
            c.parse_release(release().replace('fiscal year 2026','fiscal year 2025'))
    def test_date_spacing_and_heading_identity(self):
        r=c.parse_release(release().replace('September 30, 2025','September 30,2025').replace('reported first quarter','reported quarterly'))
        self.assertEqual(r['fiscal_quarter'],1)
    def test_same_release_issuer_period_statement_requires_same_date(self):
        body=release().replace('Three Months Ended September 30, 2025','')
        wrapper='On October 24, 2025, The Procter & Gamble Company issued a news release with respect to earnings for the quarter ended September 30, 2025. '
        self.assertEqual(c.parse_release(wrapper+body)['fiscal_period_end_actual'],'2025-09-30')
        with self.assertRaisesRegex(ValueError,'calendar date'):
            c.parse_release(wrapper.replace('October 24','October 25')+body)
    def test_cache_stem_collision_cannot_overwrite_existing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'probe.meta.json').write_text(json.dumps(dict(url='https://example.test/old.pdf',sha256='a'*64,status=200)))
            with patch.object(c,'RAW',root):
                with self.assertRaisesRegex(ValueError,'reserved'):
                    c.fetch('https://example.test/new.html','probe','.html')
            self.assertEqual(json.loads((root/'probe.meta.json').read_text())['url'],'https://example.test/old.pdf')

    def test_full_cache_and_normalization(self):
        sys.path.insert(0,str(c.HERE.parent));from event_schema import normalize_event
        rows=json.loads((c.HERE/'events.json').read_text());self.assertEqual(len(rows),63)
        keys=set()
        for r in rows:
            key=(r['fiscal_year'],r['fiscal_quarter']);self.assertNotIn(key,keys);keys.add(key)
            self.assertEqual(r['ticker'],'PG');self.assertEqual(r['precision'],'date_only');self.assertIsNone(r['publication_timestamp']);self.assertFalse(r['first_public_time_verified'])
            p=c.ROOT/r['evidence_path'];b=p.read_bytes();meta=json.loads(p.with_suffix('.meta.json').read_text())
            self.assertEqual(c.sha(b),r['evidence_sha256']);self.assertEqual(meta['sha256'],r['evidence_sha256']);self.assertEqual(meta['url'],r['source_url']);self.assertEqual(meta['status'],200)
            parsed=c.parse_release(c.html_text(b))
            for field in ('release_date','fiscal_year','fiscal_quarter','fiscal_period_end_actual'):self.assertEqual(r[field],parsed[field])
            for proof in [r['fiscal_period_evidence'],r['fiscal_identity_evidence'],r['publication_timezone_evidence'],r['official_archive_evidence']]+r['release_date_evidence']+r['additional_evidence']:
                self.assertEqual(c.sha((c.ROOT/proof['evidence_path']).read_bytes()),proof['sha256'])
            event=normalize_event(dict(symbol='PG',asset_type='equity',official=dict(verified=True,publication_kind='release',actual_period_end=r['fiscal_period_end_actual'],publication_date=r['release_date'],publication_timezone=r['publication_timezone'],source_url=r['source_url'])))
            self.assertEqual(event['status'],'ready');self.assertIsNone(event['publication_at']);self.assertGreater(event['available_session'],r['release_date'])

if __name__=='__main__':unittest.main()
