import unittest,sys,datetime as dt
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import collect as c,build as b
sys.path.insert(0,str(HERE.parent.parent));from event_schema import normalize_event
class InternalRepair(unittest.TestCase):
 def setUp(self):self.events=c.load(HERE/'events.json',[])
 def test_explicit_original_identity_and_dates(self):
  self.assertEqual({(e['fiscal_year'],e['fiscal_quarter'],e['release_date'],e['fiscal_period_end_actual']) for e in self.events},{(2015,1,'2015-04-23','2015-03-31'),(2016,1,'2016-04-21','2016-03-31')})
  for e in self.events:
   p=c.ROOT/e['evidence_path'];tx=b.text(p,'NVS');y,q=e['fiscal_year'],e['fiscal_quarter'];self.assertEqual(b.identity('NVS',tx,{})[:2],(y,q));self.assertEqual(b.release('NVS',tx,y,q)[0],e['release_date']);self.assertEqual(b.period('NVS',tx,y,q,dict(url=e['source_url']),p)[0],e['fiscal_period_end_actual'])
 def test_original_hashes_and_utc_fetch_metadata(self):
  for e in self.events:
   for p in [dict(evidence_path=e['evidence_path'],sha256=e['evidence_sha256']),e['fiscal_period_evidence']]+e['release_date_evidence']+e['additional_evidence']:self.assertEqual(c.sha((c.ROOT/p['evidence_path']).read_bytes()),p['sha256'])
   self.assertTrue(e['raw_original_bytes_available']);self.assertIsNotNone(dt.datetime.fromisoformat(e['retrieved_at']).tzinfo)
 def test_central_date_only_schema(self):
  for e in self.events:
   self.assertEqual(e['ticker'],'NVS');self.assertIsNone(e['publication_timestamp']);self.assertEqual(e['publication_timezone'],'Europe/Zurich')
   n=normalize_event(dict(symbol='NVS',asset_type='equity',official=dict(actual_period_end=e['fiscal_period_end_actual'],publication_date=e['release_date'],publication_timezone=e['publication_timezone'],source_url=e['source_url'],verified=True,publication_kind='release')))
   self.assertEqual(n['status'],'ready');self.assertGreater(n['available_session'],e['release_date']);self.assertEqual(n['precision'],'date_only')
if __name__=='__main__':unittest.main(verbosity=2)
