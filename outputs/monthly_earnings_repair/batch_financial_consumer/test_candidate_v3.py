import json,unittest
import core as c
import test_collect
class CandidateV3(test_collect.ExportAudit):
 @classmethod
 def setUpClass(cls):cls.rows=json.loads((c.HERE/'candidate_supplement_v3/events.json').read_text())
 def test_unique_dates_and_all_assigned_issuers_have_rows(self):
  keys=[(e['ticker'],e['fiscal_year'],e['fiscal_quarter'])for e in self.rows];self.assertEqual(len(keys),5);self.assertEqual(len(set(keys)),5);self.assertEqual(set(keys),{('PM',2012,2),('WFC',2012,2),('WFC',2012,3),('WFC',2013,1),('WFC',2014,3)})
 def test_original_archive_is_explicitly_classified(self):
  r=next(e for e in self.rows if e['ticker']=='WFC'and e['fiscal_year']==2014);self.assertTrue(r['original_source_fully_cached']);self.assertFalse(r['indexed_source_only']);self.assertIn('public_archive_full_original',r['source_class']);self.assertEqual(r['release_date'],'2014-10-14');self.assertEqual(len(r['release_date_evidence']),2)
 def test_weekday_is_part_of_actual_publication_header(self):
  r=next(e for e in self.rows if e['ticker']=='WFC'and e['fiscal_year']==2013);self.assertEqual(r['release_date'],'2013-04-12');self.assertTrue(any('Friday, April' in p['excerpt']for p in r['release_date_evidence']))
if __name__=='__main__':unittest.main()
