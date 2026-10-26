"""Offline admission and evidence-integrity regression tests. No HTTP requests."""
import collections,datetime as dt,sys,unittest
from pathlib import Path
import collect as c
import build as b
sys.path.insert(0,str(c.HERE.parent))
from event_schema import normalize_event

class EvidenceAdmission(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.events=c.load(c.HERE/'events.json',[])
  cls.by={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']):e for e in cls.events}
 def test_unique_identity_and_date_bounds(self):
  self.assertGreater(len(self.events),800)
  self.assertEqual(len(self.events),len(self.by))
  for e in self.events:
   with self.subTest(key=(e['ticker'],e['fiscal_year'],e['fiscal_quarter'])):
    self.assertIn(e['ticker'],c.TICKERS);self.assertIn(e['fiscal_quarter'],range(1,5))
    self.assertTrue('2011-01-01'<=e['release_date']<=c.CUTOFF)
    days=(dt.date.fromisoformat(e['release_date'])-dt.date.fromisoformat(e['fiscal_period_end_actual'])).days
    self.assertTrue(0<days<=120)
 def test_all_evidence_hashes_and_public_index_disclosure(self):
  checked={}
  for e in self.events:
   proofs=[dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256']),e['fiscal_period_evidence']]+e['release_date_evidence']+e['additional_evidence']
   for proof in proofs:
    path=c.ROOT/proof['evidence_path']
    if path not in checked:checked[path]=c.sha(path.read_bytes())
    self.assertEqual(checked[path],proof['sha256'],str(path))
   if e['source_representation']=='public_search_index_excerpt':
    self.assertFalse(e['raw_original_bytes_available']);self.assertIsNone(e['raw_original_bytes_sha256'])
    self.assertIn('NOT downloaded',e['historical_limitation'])
   else:
    self.assertTrue(e['raw_original_bytes_available']);self.assertEqual(e['raw_original_bytes_sha256'],e['evidence_sha256'])
   self.assertIsNotNone(dt.datetime.fromisoformat(e['retrieved_at'].replace('Z','+00:00')).tzinfo)
 def test_publication_date_only_and_central_schema(self):
  for e in self.events:
   self.assertIsNone(e['publication_timestamp']);self.assertEqual(e['precision'],'date_only')
   self.assertFalse(e['first_public_time_verified'])
   normalized=normalize_event(dict(symbol=e['ticker'],asset_type='equity',official=dict(actual_period_end=e['fiscal_period_end_actual'],publication_date=e['release_date'],publication_timezone=e['publication_timezone'],source_url=e['source_url'],verified=True,publication_kind='release')))
   self.assertEqual(normalized['status'],'ready',str(normalized))
   self.assertEqual(normalized['precision'],'date_only');self.assertFalse(normalized['known_by_is_actual_publication'])
   self.assertGreater(normalized['available_session'],e['release_date'])
 def test_original_date_excerpt_round_trip(self):
  for e in self.events:
   p=c.ROOT/e['evidence_path'];tx=b.text(p,e['ticker'])
   rd,excerpt=b.release(e['ticker'],tx,e['fiscal_year'],e['fiscal_quarter'])
   self.assertEqual(rd,e['release_date']);self.assertEqual(excerpt,e['release_date_evidence'][0]['excerpt'])
 def test_no_period_from_quarter_name(self):
  for t in ('DHR','ASML','PFE','SAP','ORCL'):
   with self.assertRaises(ValueError):b.period(t,'Company reports first-quarter 2025 earnings.',2025,1,dict(url='fixture'),Path(__file__),False)
 def test_novo_annual_header_does_not_override_interim_report_calendar(self):
  for y in [2013,2014,2015]:
   event=self.by[('NVO',y,3)];tx=b.text(c.ROOT/event['evidence_path'],'NVO')
   self.assertEqual(b.identity('NVO',tx,{})[:2],(y,3))
   self.assertEqual(b.release('NVO',tx,y,3)[0],{2013:'2013-10-31',2014:'2014-10-30',2015:'2015-10-29'}[y])
  annual=self.by[('NVO',2015,4)];self.assertEqual(annual['release_date'],'2016-02-03');self.assertEqual(annual['fiscal_period_end_actual'],'2015-12-31')
 def test_full_ge_novo_sequences(self):
  expected={(2010,4)}|{(y,q) for y in range(2011,2026) for q in range(1,5)}|{(2026,1),(2026,2)}
  for t in ['GE','NVO']:self.assertEqual({(y,q) for ticker,y,q in self.by if ticker==t},expected)
 def test_no_cms_or_call_date_as_publication(self):
  with self.assertRaises(ValueError):b.release('SAP','dateModified January 27, 2026; earnings conference call January 28, 2026.',2025,4)
 def test_toyota_release_is_not_sec_filing_date(self):
  for y,q,rd,end in [(2024,1,'2023-08-01','2023-06-30'),(2012,1,'2011-08-02','2011-06-30'),(2011,4,'2011-05-11','2011-03-31'),(2011,3,'2011-02-08','2010-12-31')]:
   e=self.by[('TM',y,q)];self.assertEqual((e['release_date'],e['fiscal_period_end_actual']),(rd,end))
  self.assertEqual(sum(e['ticker']=='TM' for e in self.events),63)
 def test_non_calendar_periods_retained(self):
  dhr=self.by[('DHR',2026,2)];self.assertEqual(dhr['fiscal_period_end_actual'],'2026-06-26')
  rows=[e for e in self.events if e['ticker']=='PFE' and e['fiscal_quarter']<4]
  self.assertTrue(any(e['fiscal_period_end_actual']!=b.expected_period('PFE',e['fiscal_year'],e['fiscal_quarter']) for e in rows))
 def test_wrong_china_mobile_archive_document_rejected(self):
  with self.assertRaisesRegex(ValueError,'different issuer'):b.identity('CHL','EVERGREEN INTERNATIONAL HOLDINGS 2011 Annual Results',{})
 def test_shell_predecessor_mapping_explicit(self):
  for e in self.events:
   if e['ticker']=='SHEL' and e['release_date']<'2022-01-21':
    self.assertEqual(e['issuer_name_at_release'],'Royal Dutch Shell plc')
    self.assertIn('RDS.A and RDS.B',e['historical_ticker_context'])
    self.assertEqual(e['publication_timezone'],'Europe/Amsterdam')
    self.assertTrue(any(p['role']=='issuer_identity_continuity' for p in e['additional_evidence']))
 def test_coverage_status_counts(self):
  st=c.load(c.HERE/'STATUS.json',{});self.assertEqual(st['verified_events'],len(self.events))
  self.assertEqual(st['events_by_ticker'],{t:sum(e['ticker']==t for e in self.events) for t in c.TICKERS})
if __name__=='__main__':unittest.main(verbosity=2)
