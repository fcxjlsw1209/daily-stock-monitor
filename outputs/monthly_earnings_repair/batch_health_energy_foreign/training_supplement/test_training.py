import sys,unittest,datetime as dt
from pathlib import Path
import build_training as z
sys.path.insert(0,str(z.HERE.parent.parent));from event_schema import normalize_event
class TrainingEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.es=z.c.load(z.HERE/'events.json',[])
 def test_exact_bounded_quarter_coverage(self):
  expected={(t,y,q) for t,rows in z.TARGETS.items() for y,q in rows};actual={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']) for e in self.es}
  self.assertEqual(expected,actual);self.assertEqual(len(actual),len(self.es));self.assertEqual(z.c.load(z.HERE/'PENDING.json',[]),[])
 def test_every_source_hash_and_upstream_capture(self):
  for e in self.es:
   for p in e['release_date_evidence']+[e['fiscal_period_evidence']]:
    f=z.c.ROOT/p['evidence_path'];self.assertEqual(z.c.sha(f.read_bytes()),p['sha256'])
    m=z.c.load(f,{});up=z.c.ROOT/m['upstream_capture_path'];self.assertEqual(z.c.sha(up.read_bytes()),m['upstream_capture_sha256'])
    self.assertFalse(m['original_bytes_available']);self.assertIn('not_original_bytes',m['source_representation'])
   self.assertEqual(z.c.sha((z.c.ROOT/e['evidence_path']).read_bytes()),e['evidence_sha256'])
 def test_original_dateline_round_trip(self):
  for e in self.es:
   tx=z.b.text(z.c.ROOT/e['evidence_path'],e['ticker']);rd,excerpt,zone=z.release(e['ticker'],tx)
   self.assertEqual((rd,zone),(e['release_date'],e['publication_timezone']));self.assertEqual(excerpt,e['release_date_evidence'][0]['excerpt'])
   self.assertLess(e['fiscal_period_end_actual'],rd)
   self.assertIsNotNone(dt.datetime.fromisoformat(e['retrieved_at'].replace('Z','+00:00')).tzinfo)
 def test_date_only_central_schema_and_scope(self):
  for e in self.es:
   self.assertEqual(e['research_scope'],'training_candidate_window_and_predecessor');self.assertIsNone(e['publication_timestamp']);self.assertFalse(e['first_public_time_verified']);self.assertIsNone(e['raw_original_bytes_sha256'])
   n=normalize_event(dict(symbol=e['ticker'],asset_type='equity',official=dict(actual_period_end=e['fiscal_period_end_actual'],publication_date=e['release_date'],publication_timezone=e['publication_timezone'],source_url=e['source_url'],verified=True,publication_kind='release')))
   self.assertEqual(n['status'],'ready');self.assertGreater(n['available_session'],e['release_date'])
 def test_no_filing_or_call_timestamp_substitution(self):
  for t in z.TARGETS:
   with self.assertRaises(ValueError):z.release(t,'SEC acceptance date January 26, 2011. Earnings call January 27, 2011.')
if __name__=='__main__':unittest.main(verbosity=2)
