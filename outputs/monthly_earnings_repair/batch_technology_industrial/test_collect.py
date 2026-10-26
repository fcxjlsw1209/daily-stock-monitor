"""Offline financial-evidence admission regression tests."""
import unittest,json,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect import parse,clean,HERE
from finalize import validate_event
from event_schema import normalize_event
class AdmissionTests(unittest.TestCase):
 def test_original_dateline_ignores_cms(self):
  t='dateModified 2026-10-01 Intel Reports Second Quarter 2016 Results SANTA CLARA, Calif., July 20, 2016. Intel reported results for the second quarter 2016 ended July 2, 2016.'
  e=parse('INTC',t);self.assertEqual(e['release_date'],'2016-07-20');self.assertEqual(e['fiscal_period_end_actual'],'2016-07-02')
 def test_calendar_guess_is_not_period_evidence(self):
  with self.assertRaisesRegex(ValueError,'Actual current fiscal period absent'):
   parse('INTC','Intel Reports Second Quarter 2025 Results SANTA CLARA, Calif., July 24, 2025. Intel reported second quarter 2025 results.')
 def test_future_schedule_rejected(self):
  with self.assertRaises(ValueError):parse('INTC','Intel to Report Second Quarter 2026 Results SANTA CLARA, Calif., July 1, 2026. Intel will release second quarter 2026 results ended June 27, 2026.')
 def test_preliminary_not_regular_event(self):
  with self.assertRaisesRegex(ValueError,'Preliminary'):
   parse('AMD','AMD Reports Preliminary Second Quarter 2015 Results SUNNYVALE, Calif., July 6, 2015. AMD reported second quarter 2015 ended June 27, 2015.')
 def test_regular_results_with_updated_outlook_remain_admissible(self):
  e=parse('ACN','Accenture Reports Strong Second Quarter Fiscal 2015 Results and Updates Full-Year Outlook NEW YORK, March 26, 2015. Accenture reported second quarter fiscal 2015 results for the quarter ended February 28, 2015.')
  self.assertEqual((e['fiscal_year'],e['fiscal_quarter'],e['release_date']),(2015,2,'2015-03-26'))
 def test_wrong_fiscal_year_rejected(self):
  with self.assertRaisesRegex(ValueError,'label disagrees'):
   parse('CRM','Salesforce Announces Fourth Quarter Fiscal 2025 Results SAN FRANCISCO, California, February 28, 2024. Salesforce reported fourth quarter fiscal 2025 results ended January 31, 2024.')
 def test_non_calendar_fiscal_label(self):
  e=parse('CRM','Salesforce Announces Strong Fourth Quarter and Full Fiscal 2023 Results San Francisco - March 1, 2023 - Salesforce announced results for its fourth quarter and full fiscal year ended January 31, 2023.')
  self.assertEqual((e['fiscal_year'],e['fiscal_quarter'],e['fiscal_period_end_actual']),(2023,4,'2023-01-31'))
 def test_date_only_does_not_backfill_release_day(self):
  e=normalize_event(dict(symbol='TSLA',official=dict(verified=True,publication_kind='release',publication_date='2026-07-22',publication_timezone='America/Chicago',actual_period_end='2026-06-30',source_url='https://ir.tesla.com/')))
  self.assertEqual(e['status'],'ready');self.assertEqual(e['available_session'],'2026-07-23');self.assertFalse(e['known_by_is_actual_publication']);self.assertIsNone(e['publication_at'])
 def test_training_predecessors_keep_actual_2010_date(self):
  for k,fy,q in [('HPQ',2010,4),('CSCO',2011,1)]:
   e=next(e for e in json.loads((HERE/k/'events.json').read_text()) if (e['fiscal_year'],e['fiscal_quarter'])==(fy,q))
   self.assertTrue(e['training_predecessor_only']);self.assertLess(e['release_date'],'2011-01-01');validate_event(e)
 def test_qualcomm_non_calendar_fy(self):
  e=next(e for e in json.loads((HERE/'QCOM/events.json').read_text()) if (e['fiscal_year'],e['fiscal_quarter'])==(2011,1))
  self.assertEqual((e['release_date'],e['fiscal_period_end_actual']),('2011-01-26','2010-12-26'))
 def test_targeted_training_quarter_inventories_have_no_internal_holes(self):
  ranges={'IBM':((2010,4),(2016,2)),'QCOM':((2011,1),(2014,2)),'HPQ':((2010,4),(2011,1)),'CSCO':((2011,1),(2026,4)),'INTC':((2010,4),(2026,2))}
  for k,(a,b) in ranges.items():
   observed={(e['fiscal_year'],e['fiscal_quarter']) for e in json.loads((HERE/k/'events.json').read_text())}
   expected={(n//4,n%4+1) for n in range(a[0]*4+a[1]-1,b[0]*4+b[1])}
   self.assertFalse(expected-observed,(k,expected-observed))
 def test_all_evidence_and_schema(self):
  events=json.loads((HERE/'events.json').read_text());self.assertGreaterEqual(len(events),666)
  for e in events:
   with self.subTest(ticker=e['ticker'],year=e['fiscal_year'],q=e['fiscal_quarter']):validate_event(e)
 def test_duplicate_free(self):
  es=json.loads((HERE/'events.json').read_text());keys=[(e['ticker'],e['fiscal_year'],e['fiscal_quarter']) for e in es];self.assertEqual(len(keys),len(set(keys)))
 def test_public_index_never_claims_original_bytes(self):
  es=json.loads((HERE/'events.json').read_text())
  for e in es:
   if 'public_index' in e['source_class']:
    self.assertIn('evidence_limitation',e)
    if e['evidence_path'].endswith('.json'):self.assertTrue('not' in e['evidence_limitation'].lower())
 def test_tesla_notice_date_not_acceptance(self):
  es=json.loads((HERE/'TSLA/events.json').read_text());e=next(e for e in es if e['fiscal_year']==2020 and e['fiscal_quarter']==1)
  self.assertEqual((e['release_date'],e['fiscal_period_end_actual']),('2020-04-29','2020-03-31'));self.assertIn('released its financial results',e['release_date_evidence'][0]['excerpt'])
 def test_spcx_does_not_leak_into_pre_release_months(self):
  es=json.loads((HERE/'SPCX/events.json').read_text());self.assertEqual(len(es),1);e=es[0];self.assertEqual(e['release_date'],'2026-08-04');self.assertEqual(e['fiscal_period_end_actual'],'2026-06-30')
  n=validate_event(e);self.assertGreater(n['available_session'],'2026-07-31')
if __name__=='__main__':unittest.main()
