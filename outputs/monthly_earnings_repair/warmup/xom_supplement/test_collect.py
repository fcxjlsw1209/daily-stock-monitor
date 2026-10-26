import unittest
from validate import *

class XomTests(unittest.TestCase):
 def setUp(self):self.es=json.load(open(HERE/'events.json'))
 def test_all_original_bytes_hashes_and_schema(self):
  for e in self.es:
   with self.subTest(fy=e['fiscal_year'],q=e['fiscal_quarter']):validate_event(e)
 def test_original_exhibit_release_not_filing_header(self):
  e=next(e for e in self.es if (e['fiscal_year'],e['fiscal_quarter'])==(2011,3));t=extract(ROOT/e['evidence_path']);r=parse('Date of report January 1, 2020 '+t);self.assertEqual(r['release_date'],'2011-10-27');self.assertIn('FOR IMMEDIATE RELEASE',r['release_excerpt'])
 def test_no_period_imputation_from_quarter_name(self):
  with self.assertRaisesRegex(ValueError,'financial statement absent'):
   parse('Exxon Mobil Corporation announces estimated First Quarter 2011 Results Released April 28, 2011')
 def test_capital_amount_not_misread_as_day(self):
  e=next(e for e in self.es if (e['fiscal_year'],e['fiscal_quarter'])==(2011,2));self.assertEqual(e['fiscal_period_end_actual'],'2011-06-30');self.assertIn('June 175,822',e['fiscal_period_evidence']['excerpt'])
 def test_fourth_quarter_headline_beats_forward_outlook(self):
  for y in [2014,2015]:
   e=next(e for e in self.es if (e['fiscal_year'],e['fiscal_quarter'])==(y,4));self.assertEqual(e['fiscal_period_end_actual'],str(y)+'-12-31')
 def test_first_training_month_uses_previous_release(self):
  a=json.load(open(HERE/'CANDIDATE_AUDIT.json'));r=a['first_month_predecessor'];self.assertEqual((r['signal_date'],r['latest_release']),('2011-01-31','2010-10-28'))
  q4=next(e for e in self.es if (e['fiscal_year'],e['fiscal_quarter'])==(2010,4));self.assertEqual(normalized(q4)['available_session'],'2011-02-01')
 def test_sequence_complete_and_no_duplicate_existing_quarters(self):
  p=json.load(open(HERE/'PENDING.json'));old=[e for e in json.load(open(ROOT/p['baseline'])) if e['ticker']=='XOM'];keys={(e['fiscal_year'],e['fiscal_quarter']) for e in old+self.es};self.assertEqual(len(keys),len(old)+len(self.es));self.assertEqual(len(keys),64)
 def test_original_release_footer_does_not_infer_clock(self):
  for e in self.es:self.assertIsNone(e['publication_timestamp']);self.assertEqual(e['precision'],'date_only')
 def test_no_earlier_training_backfill_without_marker(self):
  e=next(e for e in self.es if e['release_date']<'2011-01-01');self.assertTrue(e['training_predecessor_only']);self.assertEqual(e['release_date'],'2010-10-28')

if __name__=='__main__':unittest.main()
