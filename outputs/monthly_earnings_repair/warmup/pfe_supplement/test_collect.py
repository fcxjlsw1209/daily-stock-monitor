import unittest
from validate import *

class PfizerTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.es=json.loads((HERE/'events.json').read_text())
 def find(self,y,q):return next(e for e in self.es if (e['fiscal_year'],e['fiscal_quarter'])==(y,q))
 def test_all_original_bytes_hashes_schema_and_parse(self):
  for e in self.es:
   with self.subTest(fy=e['fiscal_year'],q=e['fiscal_quarter']):validate_event(e)
 def test_original_dateline_beats_shifted_CMS_day(self):
  e=self.find(2011,1);r=parse('Published May 2, 2011 21:30 CMS '+extract(ROOT/e['evidence_path']));self.assertEqual(r['release_date'],'2011-05-03')
 def test_us_52_week_fiscal_ends_not_calendar_or_foreign_ends(self):
  for y,q,end in [(2010,3,'2010-10-03'),(2011,1,'2011-04-03'),(2011,2,'2011-07-03'),(2011,3,'2011-10-02'),(2012,1,'2012-04-01'),(2012,2,'2012-07-01'),(2014,2,'2014-06-29'),(2014,3,'2014-09-28')]:
   with self.subTest(fy=y,q=q):self.assertEqual(self.find(y,q)['fiscal_period_end_actual'],end)
  self.assertIn('May 25, 2014',self.find(2014,2)['fiscal_period_evidence']['excerpt'])
 def test_no_period_imputation_from_headline_or_url(self):
  with self.assertRaisesRegex(ValueError,'period note absent'):
   parse('PFIZER REPORTS FIRST-QUARTER 2011 RESULTS NEW YORK, N.Y., Tuesday, May 3, 2011 - Pfizer Inc. fiscal period ends unknown')
 def test_overseas_note_alone_is_not_us_period(self):
  e=self.find(2014,2);t=extract(ROOT/e['evidence_path']);t=re.sub(r'financial\s+statements\s+present','financial statements OMITTED',t,flags=re.I)
  with self.assertRaisesRegex(ValueError,'period note absent'):parse(t)
 def test_future_or_conference_headline_rejected(self):
  e=self.find(2014,2);t=clean(extract(ROOT/e['evidence_path'])).replace('PFIZER REPORTS SECOND-QUARTER 2014 RESULTS','PFIZER TO REPORT SECOND-QUARTER 2014 RESULTS')
  with self.assertRaisesRegex(ValueError,'headline absent'):parse(t)
 def test_quarter_identity_comes_from_body_not_reissued_filename(self):
  e=self.find(2015,3);self.assertIn('Q4_2015',e['source_url']);self.assertEqual(e['fiscal_quarter'],3);self.assertEqual(e['release_date'],'2015-10-27');self.assertIn('THIRD-QUARTER',e['fiscal_identity_evidence']['excerpt']);self.assertEqual(e['publication_version'],'archival_reissue_filename')
 def test_corrected_version_not_republished_event_or_value_factor(self):
  e=self.find(2020,1);self.assertEqual(e['release_date'],'2020-04-28');self.assertEqual(e['source_revision_date'],'2020-05-05');self.assertEqual(e['fiscal_period_end_actual'],'2020-03-29');self.assertIn('40 basis points',e['source_revision_evidence']['excerpt']);self.assertIn('no revised financial values',e['event_scope'])
 def test_secondary_public_index_not_mislabeled_original(self):
  for e in [self.find(2015,3),self.find(2020,1)]:
   self.assertEqual(e['corroborating_public_index_evidence']['source_class'],'official_public_index_excerpt_not_original_bytes')
   self.assertTrue((ROOT/e['evidence_path']).read_bytes().startswith(b'%PDF'))
 def test_first_training_candidate_uses_2010q3(self):
  a=json.loads((HERE/'CANDIDATE_AUDIT.json').read_text());r=a['first_month_predecessor'];self.assertEqual((r['signal_date'],r['latest_release']),('2011-01-31','2010-11-02'));self.assertTrue(self.find(2010,3)['training_predecessor_only']);self.assertEqual(normalized(self.find(2010,4))['available_session'],'2011-02-02')
 def test_month_end_date_only_cannot_use_same_month_close(self):
  e=self.find(2012,2);self.assertEqual(e['release_date'],'2012-07-31');self.assertEqual(normalized(e)['available_session'],'2012-08-01')
 def test_complete_sequence_without_duplicate_frozen_base(self):
  p=json.loads((HERE/'PENDING.json').read_text());old=[e for e in json.loads((ROOT/p['baseline']).read_text()) if e['ticker']=='PFE'];keys={(e['fiscal_year'],e['fiscal_quarter']) for e in old+self.es};self.assertEqual(len(keys),len(old)+len(self.es));self.assertEqual(len(keys),64);self.assertEqual(p['missing_target_quarters'],[])
 def test_clock_not_fabricated(self):
  for e in self.es:self.assertIsNone(e['publication_timestamp']);self.assertEqual(e['precision'],'date_only');self.assertFalse(e['first_public_time_verified'])
if __name__=='__main__':unittest.main()

