"""Offline evidence admission, source cache and fiscal-year tests for Visa."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('visa_collect',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
spec=importlib.util.spec_from_file_location('earnings_event_schema',c.HERE.parent/'event_schema.py')
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)

class VisaTests(unittest.TestCase):
    def test_fiscal_q1_is_previous_december(self):
        parsed=c.parse_release('Visa Reports Fiscal first quarter 2011. SAN FRANCISCO, CA, February 2, 2011 - financial results for the Company’s fiscal first quarter 2011 ended December 31, 2010.',2011,1)
        self.assertEqual(parsed['fiscal_period_end_actual'],'2010-12-31')
        self.assertEqual(parsed['release_date'],'2011-02-02')

    def test_body_requires_correct_fiscal_year_and_quarter(self):
        text='Visa Reports Fiscal first quarter 2024. San Francisco, CA, January 25, 2024 - ended December 31, 2023.'
        with self.assertRaisesRegex(ValueError,'Fiscal year/quarter'):
            c.parse_release(text,2023,1)
        with self.assertRaisesRegex(ValueError,'Fiscal year/quarter'):
            c.parse_release(text,2024,2)

    def test_body_period_end_cannot_be_inferred(self):
        text='Visa Reports Fiscal first quarter 2024. San Francisco, CA, January 25, 2024.'
        with self.assertRaisesRegex(ValueError,'Actual fiscal-period'):
            c.parse_release(text,2024,1)

    def test_conference_or_scheduled_date_not_a_dateline(self):
        text='Visa Reports Fiscal first quarter 2024 ended December 31, 2023. Conference call January 25, 2024 at 5 PM ET.'
        with self.assertRaisesRegex(ValueError,'dateline'):
            c.parse_release(text,2024,1)

    def test_scheduled_announcement_with_dateline_is_not_release(self):
        text='Visa to Announce Fiscal First Quarter 2024 Results. San Francisco, January 10, 2024 - results for quarter ended December 31, 2023 will be released January 25, 2024.'
        with self.assertRaisesRegex(ValueError,'Actual results release headline'):c.parse_release(text,2024,1)

    def test_html_cms_timestamp_and_head_metadata_excluded(self):
        parser=c.BodyParser();parser.feed('<html><head><meta name="datePublished" content="2017-04-20T20:05:21"><script>San Francisco, April 20, 2017</script></head><body><p>Visa Reports Fiscal second quarter 2017 ended March 31, 2017</p><script>San Francisco, April 20, 2017</script></body></html>')
        text=' '.join(parser.parts)
        self.assertNotIn('San Francisco',text)
        with self.assertRaisesRegex(ValueError,'dateline'):c.parse_release(text,2017,2)

    def test_wire_body_dateline_and_fiscal_period(self):
        text='SAN FRANCISCO --(BUSINESS WIRE)--Apr. 20, 2017-- Visa Inc. Reports Fiscal second quarter 2017. Visa today announced financial results for the Company’s fiscal second quarter 2017, ended March 31, 2017.'
        self.assertEqual(c.parse_release(text,2017,2)['release_date'],'2017-04-20')

    def test_san_francisco_dateline_without_state_suffix(self):
        text='Visa Reports Fiscal Fourth Quarter and Full-Year 2025 Results. San Francisco, October 28, 2025 - financial results for the Company’s fiscal fourth quarter and full-year 2025 ended September 30, 2025.'
        self.assertEqual(c.parse_release(text,2025,4)['release_date'],'2025-10-28')

    def test_repurchase_as_of_date_cannot_establish_fiscal_end(self):
        text='Visa Reports Fiscal First Quarter 2024 Results. San Francisco, CA, January 25, 2024. Outstanding repurchase authorization as of December 31, 2023 was $10 billion.'
        with self.assertRaisesRegex(ValueError,'company financial-results'):
            c.parse_release(text,2024,1)

    def test_payment_volume_date_cannot_establish_fiscal_end(self):
        text='Visa Reports Fiscal First Quarter 2024 Results. San Francisco, CA, January 25, 2024. Payments volume for the three months ended December 31, 2023 increased 8%.'
        with self.assertRaisesRegex(ValueError,'company financial-results'):
            c.parse_release(text,2024,1)

    def test_unrelated_date_cannot_rescue_wrong_statement_period(self):
        text='Visa Reports Fiscal First Quarter 2024 Results. San Francisco, CA, January 25, 2024. Repurchases as of December 31, 2023. Consolidated Statements of Operations (Unaudited) Three Months Ended September 30, 2023 2022 Net revenues $1 $2.'
        with self.assertRaisesRegex(ValueError,'company financial-results'):
            c.parse_release(text,2024,1)

    def test_statement_requires_current_and_prior_year_columns(self):
        text='Visa Reports Fiscal First Quarter 2024 Results. San Francisco, CA, January 25, 2024. Consolidated Statements of Operations (Unaudited) Three Months Ended December 31, 2022 2021 Net revenues $1 $2.'
        with self.assertRaisesRegex(ValueError,'company financial-results'):
            c.parse_release(text,2024,1)

    def test_original_2013q1_repurchase_is_replaced_by_company_fiscal_statement(self):
        parsed=c.parse_release((c.RAW/'2013Q1.txt').read_text(),2013,1)
        self.assertIn('financial results for the Company',parsed['period_excerpt'])
        self.assertIn('fiscal first quarter 2013 ended December 31, 2012',parsed['period_excerpt'])
        self.assertNotIn('repurchase',parsed['period_excerpt'])

    def test_layout_spacing_artifacts_do_not_change_dateline(self):
        parsed=c.parse_release((c.RAW/'2011Q1.txt').read_text(),2011,1)
        self.assertEqual(parsed['release_date'],'2011-02-02')
        self.assertEqual(parsed['location'].lower(),'san francisco')
        parsed=c.parse_release((c.RAW/'2017Q1.txt').read_text(),2017,1)
        self.assertEqual(parsed['release_date'],'2017-02-02')

    def test_complete_cache_is_unique_and_hash_verified(self):
        rows=json.loads((c.HERE/'events.json').read_text());self.assertEqual(len(rows),63)
        ids={(r['fiscal_year'],r['fiscal_quarter']) for r in rows}
        self.assertEqual(ids,{(y,q) for y in range(2011,2027) for q in range(1,5) if y<2026 or q<=3})
        for row in rows:
            self.assertEqual(row['ticker'],'V');self.assertEqual(row['precision'],'date_only')
            self.assertIsNone(row['publication_timestamp']);self.assertFalse(row['first_public_time_verified'])
            self.assertEqual(row['publication_timezone'],'America/Los_Angeles')
            self.assertLess(row['fiscal_period_end_actual'],row['release_date']);self.assertLessEqual(row['release_date'],c.CUTOFF)
            p=c.ROOT/row['evidence_path'];self.assertEqual(c.sha(p.read_bytes()),row['evidence_sha256'])
            meta=json.loads(p.with_suffix('.meta.json').read_text())
            self.assertEqual(meta['url'],row['source_url']);self.assertEqual(meta['sha256'],row['evidence_sha256']);self.assertEqual(meta['status'],200)
            self.assertEqual(meta['retrieved_at'],row['retrieved_at']);self.assertTrue(meta['retrieved_at'].endswith('+00:00'))
            tp=p.with_suffix('.txt');tm=json.loads(p.with_suffix('.text.meta.json').read_text())
            self.assertEqual(tm['source_sha256'],row['evidence_sha256']);self.assertEqual(tm['text_sha256'],c.sha(tp.read_bytes()))
            self.assertNotIn('Payments volume',row['fiscal_period_evidence']['excerpt'])
            self.assertNotIn('repurchase',row['fiscal_period_evidence']['excerpt'].lower())
            self.assertEqual(row['fiscal_period_evidence']['basis'],c.parse_release(tp.read_text(),row['fiscal_year'],row['fiscal_quarter'])['period_basis'])
            parsed=c.parse_release(tp.read_text(),row['fiscal_year'],row['fiscal_quarter'])
            self.assertEqual(parsed['release_date'],row['release_date']);self.assertEqual(parsed['fiscal_period_end_actual'],row['fiscal_period_end_actual'])
            for proof in [row['fiscal_period_evidence'],row['identity_evidence'],row['publication_timezone_evidence']]+row['release_date_evidence']:
                self.assertEqual(proof['sha256'],c.sha((c.ROOT/proof['evidence_path']).read_bytes()))

    def test_all_fetches_are_cached_and_offline(self):
        with patch.object(c.urllib.request,'urlopen',side_effect=AssertionError('Network must not run')):
            for s in json.loads((c.HERE/'sources.json').read_text()):
                c.fetch(s['url'],f"{s['year']}Q{s['quarter']}",'.html' if s.get('format')=='html' else '.pdf')

    def test_all_events_normalize_conservatively_under_shared_schema(self):
        rows=json.loads((c.HERE/'events.json').read_text())
        for row in rows:
            event=schema.normalize_event(dict(symbol='V',asset_type='equity',official=dict(actual_period_end=row['fiscal_period_end_actual'],publication_at=None,publication_date=row['release_date'],publication_timezone=row['publication_timezone'],source_url=row['source_url'],verified=True,publication_kind='release')))
            self.assertEqual(event['status'],'ready',event['issues']);self.assertEqual(event['precision'],'date_only')
            self.assertFalse(event['known_by_is_actual_publication']);self.assertIsNone(event['publication_at'])
            self.assertGreater(event['available_session'],event['publication_date'])

if __name__=='__main__':unittest.main()
