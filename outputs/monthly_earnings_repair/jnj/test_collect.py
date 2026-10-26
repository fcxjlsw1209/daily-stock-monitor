"""Offline regression and corpus-audit tests; no network requests."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

spec=importlib.util.spec_from_file_location('jnj_collect',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class ReleaseParserTests(unittest.TestCase):
 def test_cms_migration_date_ignored(self):
  t='Johnson & Johnson Reports 2015 Third-Quarter Results February 23, 2017 NEW BRUNSWICK, N.J., Oct. 13, 2015 /PRNewswire/ -- Johnson & Johnson today announced sales'
  self.assertEqual(c.parse_release(t,2015,3)['release_date'],'2015-10-13')
 def test_conference_call_clock_never_timestamp(self):
  t='Johnson & Johnson Reports 2015 Third-Quarter Results NEW BRUNSWICK, N.J., Oct. 13, 2015 /PRNewswire/ -- Johnson & Johnson today announced sales. Conference call today at 8:30 a.m. ET'
  got=c.parse_release(t,2015,3)
  self.assertEqual(got['release_date'],'2015-10-13');self.assertNotIn('publication_timestamp',got)
 def test_upcoming_meeting_cannot_replace_pdf_header_date(self):
  t='NEWS RELEASE Johnson & Johnson reports Q1 2026 results, raises 2026 outlook 2026-04-14 Company announces planned Enterprise Business Review for December 8, 2026 NEW BRUNSWICK, N.J.--(BUSINESS WIRE)-- Johnson & Johnson today announced results'
  self.assertEqual(c.parse_release(t,2026,1)['release_date'],'2026-04-14')
  with self.assertRaisesRegex(ValueError,'publication date absent'):
   c.parse_release(t.replace('NEWS RELEASE','Webpage').replace('2026-04-14',''),2026,1)
 def test_original_businesswire_dateline(self):
  t='Johnson & Johnson Reports Q2 2022 Results NEW BRUNSWICK, N.J.--(BUSINESS WIRE)--Jul. 19, 2022-- Johnson & Johnson today announced results'
  self.assertEqual(c.parse_release(t,2022,2)['release_date'],'2022-07-19')
 def test_original_numeric_date_header(self):
  t='NEWS RELEASE Johnson & Johnson Reports Q2 2023 Results 7/20/2023 Significant milestones NEW BRUNSWICK, N.J.--(BUSINESS WIRE)-- Johnson & Johnson today announced results'
  self.assertEqual(c.parse_release(t,2023,2)['release_date'],'2023-07-20')
 def test_schedule_not_release(self):
  t='Johnson & Johnson Reports Q2 2023 Results NEW BRUNSWICK, N.J., June 1, 2023 Johnson & Johnson will release results July 20, 2023.'
  with self.assertRaisesRegex(ValueError,'actual results'):c.parse_release(t,2023,2)
 def test_wrong_fy_or_quarter_rejected(self):
  with self.assertRaisesRegex(ValueError,'identity'):c.parse_release('Johnson & Johnson Reports 2015 First-Quarter Results NEW BRUNSWICK, N.J., April 14, 2015 today announced',2014,1)

class DownloadTests(unittest.TestCase):
 def test_403_and_429_stop_host_and_preserve_retry_after(self):
  for code in (403,429):
   with self.subTest(code=code),tempfile.TemporaryDirectory() as td:
    here=Path(td);raw=here/'raw';raw.mkdir();url='https://example.test/a.pdf'
    err=urllib.error.HTTPError(url,code,'blocked',{'Retry-After':'3600'},None)
    with patch.object(c,'HERE',here),patch.object(c,'RAW',raw),patch.object(c.urllib.request,'urlopen',side_effect=err) as request:
     with self.assertRaises(urllib.error.HTTPError):c.fetch(url,'a')
     with self.assertRaisesRegex(RuntimeError,'source stopped'):c.fetch('https://example.test/b.pdf','b')
     self.assertEqual(request.call_count,1)
    stop=json.loads((here/'DIRECT_SOURCE_STOPS.json').read_text())[0]
    self.assertEqual(stop['http_status'],code);self.assertEqual(stop['retry_after'],'3600')
 def test_cache_checksum_mismatch_rejected_without_network(self):
  with tempfile.TemporaryDirectory() as td:
   raw=Path(td);p=raw/'a.pdf';p.write_bytes(b'%PDF bad')
   c.dump(raw/'a.meta.json',dict(url='https://example.test/a.pdf',sha256='0'*64))
   with patch.object(c,'RAW',raw),patch.object(c.urllib.request,'urlopen') as request:
    with self.assertRaisesRegex(ValueError,'checksum'):c.fetch('https://example.test/a.pdf','a')
    request.assert_not_called()

class CorpusTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=json.loads((c.HERE/'events.json').read_text())
 def test_complete_quarter_identity_and_dates(self):
  self.assertEqual(len(self.rows),63)
  expected={(y,q) for y in range(2011,2026) for q in range(1,5)}|{(2010,4),(2026,1),(2026,2)}
  self.assertEqual({(x['fiscal_year'],x['fiscal_quarter']) for x in self.rows},expected)
  self.assertEqual(len({x['release_date'] for x in self.rows}),63)
  for x in self.rows:
   self.assertEqual(x['ticker'],'JNJ');self.assertLess(x['fiscal_period_end_actual'],x['release_date']);self.assertLessEqual(x['release_date'],c.CUTOFF)
   d=dt.date.fromisoformat(x['release_date']);q=x['fiscal_quarter']
   self.assertEqual(d.year,x['fiscal_year']+(q==4));self.assertEqual(d.month,{1:4,2:7,3:10,4:1}[q])
 def test_noncalendar_fiscal_ends_preserved(self):
  rows={(x['fiscal_year'],x['fiscal_quarter']):x for x in self.rows}
  for key,actual in {(2015,3):'2015-09-27',(2015,4):'2016-01-03',(2016,1):'2016-04-03',(2016,2):'2016-07-03',(2021,1):'2021-04-04',(2026,2):'2026-06-28'}.items():
   self.assertEqual(rows[key]['fiscal_period_end_actual'],actual)
 def test_all_evidence_bytes_and_representation(self):
  for row in self.rows:
   p=c.ROOT/row['evidence_path'];self.assertEqual(c.sha(p.read_bytes()),row['evidence_sha256'])
   if row['raw_original_bytes_available']:
    self.assertTrue(p.read_bytes().startswith(b'%PDF'));self.assertEqual(row['raw_original_bytes_sha256'],row['evidence_sha256'])
    self.assertEqual(row['source_representation'],'original_pdf_bytes')
   else:
    self.assertIsNone(row['raw_original_bytes_sha256']);self.assertIn('not_original_html',row['source_representation'])
   for proof in row['release_date_evidence']+[row['fiscal_period_evidence']]+row['additional_evidence']:
    self.assertEqual(c.sha((c.ROOT/proof['evidence_path']).read_bytes()),proof['sha256'])
 def test_conservative_precision_timezone_and_no_fabricated_instant(self):
  for row in self.rows:
   self.assertEqual(row['precision'],'date_only');self.assertIsNone(row['publication_timestamp']);self.assertFalse(row['first_public_time_verified']);self.assertEqual(row['publication_timezone'],'America/New_York')
 def test_reparse_publication_from_saved_source(self):
  for row in self.rows:
   p=c.ROOT/row['evidence_path'];t=c.pdf_text(p) if p.suffix=='.pdf' else json.loads(p.read_text())['response']
   got=c.parse_release(t,row['fiscal_year'],row['fiscal_quarter'])
   self.assertEqual(got['release_date'],row['release_date'])

if __name__=='__main__':unittest.main()
