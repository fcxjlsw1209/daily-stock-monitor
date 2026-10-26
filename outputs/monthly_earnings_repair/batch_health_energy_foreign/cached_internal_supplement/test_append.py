import unittest,sys,datetime as dt
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import collect as c,build as b
sys.path.insert(0,str(HERE.parent.parent));from event_schema import normalize_event
class CachedRepair(unittest.TestCase):
 def setUp(self):self.e=c.load(HERE/'events.json',[])[0]
 def test_current_year_identity_precedes_comparative_prior_year(self):
  e=self.e;tx=b.text(c.ROOT/e['evidence_path'],'ABBV');self.assertIn('fourth-quarter 2014',tx)
  self.assertEqual(b.identity('ABBV',tx,{})[:2],(2015,4));self.assertEqual(b.release('ABBV',tx,2015,4)[0],'2016-01-29')
  self.assertEqual(b.period('ABBV',tx,2015,4,dict(url=e['source_url']),c.ROOT/e['evidence_path'])[0],'2015-12-31')
 def test_original_hashes_and_utc_fetch_metadata(self):
  e=self.e
  for p in [dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256']),e['fiscal_period_evidence']]+e['release_date_evidence']+e['additional_evidence']:self.assertEqual(c.sha((c.ROOT/p['evidence_path']).read_bytes()),p['sha256'])
  self.assertTrue(e['raw_original_bytes_available']);self.assertIsNotNone(dt.datetime.fromisoformat(e['retrieved_at']).tzinfo)
 def test_central_schema_next_session_and_scope(self):
  e=self.e;self.assertEqual(len(c.load(HERE/'events.json',[])),1);self.assertIsNone(e['publication_timestamp'])
  n=normalize_event(dict(symbol='ABBV',asset_type='equity',official=dict(actual_period_end=e['fiscal_period_end_actual'],publication_date=e['release_date'],publication_timezone=e['publication_timezone'],source_url=e['source_url'],verified=True,publication_kind='release')))
  self.assertEqual(n['status'],'ready');self.assertEqual(n['available_session'],'2016-02-01');self.assertEqual(n['precision'],'date_only')
if __name__=='__main__':unittest.main(verbosity=2)
