"""Offline admission tests for original Walmart release evidence and fiscal dates."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('walmart_collect',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class WalmartEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items={(z['fiscal_year'],z['fiscal_quarter']):z for z in c.archive_items((c.RAW/'archive.html').read_bytes())}
        cls.events=json.loads((c.HERE/'events.json').read_text())

    def parsed(self,year,quarter,text=None):
        if text is None:text=(c.RAW/f'{year}Q{quarter}.txt').read_text()
        return c.parse_release(text,self.items[year,quarter])

    def test_archive_has_all_63_requested_regular_quarters(self):
        self.assertEqual(len(self.items),63)
        self.assertEqual(min(self.items),(2011,4));self.assertEqual(max(self.items),(2027,2))

    def test_2011_original_dateline_and_january_year_end(self):
        out=self.parsed(2011,4)
        self.assertEqual(out['release_date'],'2011-02-22')
        self.assertIn('Jan. 31, 2011',out['period_excerpt'])
        self.assertIn('dateline',out['release_basis'])

    def test_fy2012_q1_is_calendar_2011(self):
        self.assertEqual(self.items[2012,1]['fiscal_period_end_actual'],'2011-04-30')
        self.assertEqual(self.parsed(2012,1)['release_date'],'2011-05-17')

    def test_fy2027_q2_is_calendar_2026(self):
        self.assertEqual(self.items[2027,2]['fiscal_period_end_actual'],'2026-07-31')
        self.assertEqual(self.parsed(2027,2)['release_date'],'2026-08-20')

    def test_original_footer_date_is_calendar_date_not_clock(self):
        out=self.parsed(2016,1)
        self.assertEqual(out['release_date'],'2015-05-19')
        self.assertIn('footer',out['release_basis'])
        self.assertNotIn('publication_timestamp',out)

    def test_13_week_comp_sales_end_is_not_consolidated_quarter_end(self):
        text=(c.RAW/'2016Q1.txt').read_text()
        self.assertIn('period ended May 1, 2015',c.clean(text))
        self.assertIn('April 30,',self.parsed(2016,1)['period_excerpt'])
        wrong=dict(self.items[2016,1],fiscal_period_end_actual='2015-05-01')
        with self.assertRaisesRegex(ValueError,'fiscal label/end conflict'):
            c.parse_release(text,wrong)

    def test_mislabeling_fy_as_calendar_year_rejected(self):
        text=(c.RAW/'2027Q2.txt').read_text()
        wrong=dict(self.items[2027,2],fiscal_year=2026)
        with self.assertRaisesRegex(ValueError,'fiscal label/end conflict'):
            c.parse_release(text,wrong)

    def test_missing_original_date_cannot_use_call_or_cms(self):
        text='Walmart reports Q1 FY16 results. DateModified May 19, 2015. Conference call May 19, 2015 8:00 AM EDT.'
        with self.assertRaisesRegex(ValueError,'CMS/call'):
            self.parsed(2016,1,text)

    def test_conflicting_original_release_dates_rejected(self):
        text=(c.RAW/'2016Q1.txt').read_text()+' NYSE: WMT May 20, 2015 stock.walmart.com'
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            self.parsed(2016,1,text)

    def test_quarter_identity_must_be_in_opening(self):
        text='Walmart reports Q2 FY16 results.'+' blank '*400+'Q1 FY16 NYSE: WMT May 19, 2015 stock.walmart.com'
        with self.assertRaisesRegex(ValueError,'headline'):
            self.parsed(2016,1,text)

    def test_comp_date_without_consolidated_table_cannot_confirm_end(self):
        text='Walmart reports Q1 FY16 results. NYSE: WMT May 19, 2015 stock.walmart.com. 13-week comp sales period ended April 30, 2015.'
        with self.assertRaisesRegex(ValueError,'income statement absent'):
            self.parsed(2016,1,text)

    def test_all_cached_evidence_hashes_and_dates(self):
        self.assertEqual(len(self.events),63)
        keys=set()
        for row in self.events:
            key=(row['fiscal_year'],row['fiscal_quarter'])
            self.assertNotIn(key,keys);keys.add(key)
            self.assertEqual(row['ticker'],'WMT')
            self.assertEqual(row['precision'],'date_only')
            self.assertIsNone(row['publication_timestamp'])
            self.assertFalse(row['first_public_time_verified'])
            self.assertEqual(row['publication_timezone'],'America/Chicago')
            self.assertLess(row['fiscal_period_end_actual'],row['release_date'])
            self.assertTrue(c.START<=row['release_date']<=c.CUTOFF)
            self.assertTrue(row['source_url'].startswith('https://stock.walmart.com/'))
            self.assertEqual(c.sha((c.ROOT/row['evidence_path']).read_bytes()),row['evidence_sha256'])
            self.assertEqual(c.sha((c.ROOT/row['body_text_evidence_path']).read_bytes()),row['body_text_sha256'])
            proofs=[row['official_archive_evidence'],row['fiscal_period_evidence'],row['publication_timezone_evidence']]+row['release_date_evidence']+row['additional_evidence']
            for proof in proofs:
                self.assertEqual(c.sha((c.ROOT/proof['evidence_path']).read_bytes()),proof['sha256'])
            self.assertEqual(self.parsed(*key)['release_date'],row['release_date'])

    def test_offline_replay_reuses_complete_cache_without_network(self):
        with patch.object(c.urllib.request,'urlopen',side_effect=AssertionError('No network allowed')):
            for item in self.items.values():
                label=f"{item['fiscal_year']}Q{item['fiscal_quarter']}"
                body,path,meta=c.fetch(item['pdf_url'],label,'.pdf')
                self.assertEqual(c.sha(body),meta['sha256'])

    def test_shared_event_schema_admits_all_date_only_under_policy(self):
        spec=importlib.util.spec_from_file_location('event_schema_wmt',c.HERE.parent/'event_schema.py')
        schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
        for row in self.events:
            normalized=schema.normalize_event(dict(symbol='WMT',asset_type='equity',official=dict(
                actual_period_end=row['fiscal_period_end_actual'],publication_date=row['release_date'],
                publication_timezone=row['publication_timezone'],source_url=row['source_url'],
                verified=True,publication_kind='release')))
            self.assertEqual(normalized['status'],'ready')
            self.assertFalse(normalized['known_by_is_actual_publication'])
            self.assertGreater(normalized['available_session'],row['release_date'])
            self.assertIsNone(normalized['publication_at'])

if __name__=='__main__':unittest.main()
