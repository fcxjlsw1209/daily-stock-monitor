"""Offline Costco period/date/identity and immutable-source regression checks."""
import unittest
from validate import *

class CostcoTests(unittest.TestCase):
 def test_monthly_sales_end_is_not_quarter_end(self):
  t='Costco Wholesale Corporation Reports Second Quarter and Year-to-Date Operating Results for Fiscal Year 2011 and February Sales Results ISSAQUAH, Wash., March 2, 2011 -- Costco announced its operating results for the second quarter (twelve weeks) and first half (24 weeks) of fiscal 2011, both ended February 13, 2011, and February sales results for the four weeks ended February 27, 2011.'
  r=parse(t);self.assertEqual(r['fiscal_period_end_actual'],'2011-02-13');self.assertEqual(r['fiscal_quarter_weeks'],12)
 def test_sales_only_release_is_rejected(self):
  t='Costco Wholesale Corporation Reports First Quarter Fiscal 2014 Sales Results ISSAQUAH, Wash., December 5, 2013 -- Costco announced sales for the first quarter (twelve weeks) ended November 24, 2013.'
  with self.assertRaisesRegex(ValueError,'sales-only'):parse(t)
 def test_fiscal_fourth_quarter_can_be_seventeen_weeks(self):
  e=next(e for e in json.load(open(HERE/'events.json')) if (e['fiscal_year'],e['fiscal_quarter'])==(2023,4))
  self.assertEqual((e['fiscal_quarter_weeks'],e['fiscal_period_end_actual']),(17,'2023-09-03'))
 def test_fiscal_identity_must_match_observed_end(self):
  with self.assertRaisesRegex(ValueError,'Fiscal title disagrees'):
   parse('Costco Wholesale Reports First Quarter Fiscal 2025 Operating Results ISSAQUAH, Wash., December 14, 2023 -- Costco announced operating results for the first quarter (twelve weeks) ended November 26, 2023.')
 def test_future_or_scheduled_release_is_not_admitted(self):
  with self.assertRaises(ValueError):parse('Costco Wholesale Reports Fourth Quarter Fiscal 2026 Operating Results ISSAQUAH, Wash., September 25, 2026 -- Costco announced operating results for the 16-week fourth quarter ended August 30, 2026.')
 def test_all_hashes_and_conservative_availability(self):
  for e in json.load(open(HERE/'ALL_VERIFIED_CANDIDATES.json')):
   with self.subTest(year=e['fiscal_year'],quarter=e['fiscal_quarter']):validate(e)
 def test_all_original_events_independently_audited_one_correction(self):
  aud=json.load(open(HERE/'ORIGINAL_35_AUDIT.json'));self.assertEqual(len(aud),35);self.assertEqual(Counter(r['status'] for r in aud),dict(matches=34,confirmed_correction=1))
  j=json.load(open(HERE/'CORRECTIONS.json'));p=ROOT/j['original_input'];self.assertEqual(sha(p.read_bytes()),j['original_input_sha256']);self.assertIn('financial_consumer_687',str(p));r=j['corrections'][0]
  self.assertEqual((r['release_date'],r['old_end'],r['verified_end']),('2011-03-02','2011-02-27','2011-02-13'))
 def test_additions_are_unique_and_absent_from_immutable_source(self):
  es=json.load(open(HERE/'events.json'));j=json.load(open(HERE/'CORRECTIONS.json'));old=json.load(open(ROOT/j['original_input']));oldkeys={(e['fiscal_year'],e['fiscal_quarter']) for e in old if e['ticker']=='COST'};keys=[(e['fiscal_year'],e['fiscal_quarter']) for e in es]
  self.assertEqual(len(keys),19);self.assertEqual(len(set(keys)),19);self.assertFalse(set(keys)&oldkeys)
 def test_full_candidate_sequence_and_all_candidate_rows(self):
  obs={(e['fiscal_year'],e['fiscal_quarter']) for e in json.load(open(HERE/'ALL_VERIFIED_CANDIDATES.json'))};self.assertTrue({(y,q) for y in range(2018,2027) for q in range(1,5)}<=obs)
  a=json.load(open(HERE/'CANDIDATE_AUDIT.json'));self.assertEqual(a['counts'],dict(monthly=dict(rows=52,review_rows=0),training=dict(rows=44,review_rows=0)))
 def test_public_index_and_original_bytes_are_not_conflated(self):
  es=json.load(open(HERE/'events.json'));self.assertEqual(sum('not_full_bytes' in e['source_class'] for e in es),4)
  for e in es:
   if 'not_full_bytes' in e['source_class']:self.assertIn('not original',e['evidence_limitation'])
   else:self.assertTrue((ROOT/e['evidence_path']).read_bytes().startswith(b'%PDF'))

if __name__=='__main__':unittest.main()
