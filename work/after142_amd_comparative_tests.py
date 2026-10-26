"""Synthetic-only source tests: no issuer cache, prices, rankings or returns."""
import copy
import unittest
from unittest import mock
from datetime import date,timedelta
from bs4 import BeautifulSoup
import after142_amd_comparative_source as m
from after141_amd_source_extract_tests import fixture as old_fixture


def fixture(**kw):
 raw,spec=old_fixture(**kw)
 release=spec['release_date'];day=date.fromisoformat(release)
 anchor=('SANTA CLARA, Calif., '+day.strftime('%B %d, %Y')+' -- AMD (NASDAQ:AMD) reports results. Advanced Micro Devices, Inc.')
 raw=raw.replace(b'<html>',('<html><p>'+anchor+'</p>').encode())
 spec.update(symbol='AMD',raw_path='synthetic.html');spec.pop('raw_html',None)
 receipt={'url':spec['source_url'],'final_url':spec['source_url'],'status':200,'content_type':'text/html','retrieved_at':'2026-10-19T12:00:00Z','public_unauthenticated':True}
 event={'ticker':'AMD','fiscal_year':spec['fiscal_year'],'fiscal_quarter':4,'source_url':spec['source_url'],'evidence_path':spec['raw_path'],'release_date':release,'fiscal_period_end_actual':spec['fiscal_year_end'],'source_retrieved_at':receipt['retrieved_at']}
 sessions=[release]+[(day+timedelta(days=i)).isoformat() for i in range(1,4)]
 return rebind(raw,spec,receipt,[event],sessions)


def rebind(raw,spec,receipt,events,sessions):
 spec=copy.deepcopy(spec);receipt=copy.deepcopy(receipt);events=copy.deepcopy(events)
 spec['raw_sha256']=m.sha(raw);receipt['sha256']=m.sha(raw)
 for event in events:event['evidence_sha256']=m.sha(raw)
 return raw,spec,receipt,events,list(sessions)


def mutate(args,fn):return rebind(fn(args[0]),*args[1:])


class ComparativeTests(unittest.TestCase):
 def test_four_cells_and_same_release_availability(self):
  args=fixture();r=m.extract_candidate(*args)
  self.assertEqual(r['current']['gross_amount']['amount_decimal'],'1234.50')
  self.assertEqual(r['comparison']['gross_amount']['amount_decimal'],'987')
  self.assertEqual(r['current']['total_assets']['amount_decimal'],'9876')
  self.assertEqual(r['comparison']['total_assets']['amount_decimal'],'8000')
  for group in ('current','comparison'):
   for f in r[group].values():self.assertEqual(f['available_from'],'2023-02-01')
  self.assertEqual(r['comparison_period']['end'],'2021-12-25');self.assertIsNone(r['comparison_period']['start'])
  self.assertTrue(m.audit_candidate(args[0],r)['passed'])

 def test_shifted_income_and_interim_balance(self):
  args=fixture(quarters=3,interim_balance_date='2022-09-24',external=True)
  r=m.extract_candidate(*args);self.assertEqual(r['comparison']['gross_amount']['amount_decimal'],'987')
  self.assertEqual(r['comparison']['total_assets']['date'],'2021-12-25')
  self.assertTrue(m.audit_candidate(args[0],r)['passed'])

 def test_no_mutation(self):
  args=fixture();before=copy.deepcopy(args);m.extract_candidate(*args);self.assertEqual(args,before)

 def test_no_currency_or_vintage_or_admission_upgrade(self):
  r=m.extract_candidate(*fixture())
  self.assertFalse(r['fact_admitted']);self.assertFalse(r['strategy_admitted']);self.assertFalse(r['historical_first_version_verified'])
  self.assertFalse(r['publication_anchor']['continuous_security_identity_verified'])
  self.assertFalse(r['publication_anchor']['original_exact_clock_verified'])
  self.assertFalse(r['scope_review']['admitted_for_factor_use'])
  for group in ('current','comparison'):
   for f in r[group].values():self.assertIsNone(f['currency_iso'])

 def test_comparison_percent_missing_and_nonpositive_assets(self):
  for change in (lambda r:r.replace(b'>987<',b'>42%<'),lambda r:r.replace(b'>987<',b'>NaN<'),lambda r:r.replace(b'>8,000<',b'>0<')):
   with self.subTest(change=change),self.assertRaises(ValueError):m.extract_candidate(*mutate(fixture(),change))

 def test_prior_decimal_and_negative_preserved(self):
  args=mutate(fixture(),lambda r:r.replace(b'>987<',b'>(987.25)<'))
  r=m.extract_candidate(*args);self.assertEqual(r['comparison']['gross_amount']['amount_decimal'],'-987.25')
  self.assertTrue(m.audit_candidate(args[0],r)['passed'])

 def test_old_margin_is_monetary_not_percent(self):
  r=m.extract_candidate(*fixture(year=2015,end='2015-12-26',prior='2014-12-27'))
  self.assertEqual(r['comparison']['gross_amount']['measure_label'],'Gross margin')
  self.assertEqual(r['comparison']['gross_amount']['amount_decimal'],'987')

 def test_2016_needs_accounting_notes(self):
  args=fixture(year=2016,end='2016-12-31',prior='2015-12-26')
  with self.assertRaisesRegex(ValueError,'accounting'):m.extract_candidate(*args)
  args=mutate(args,lambda r:r.replace(b'</html>',b'<p>Amounts reflected adoption of FASB ASU 2015-17 and FASB ASU 2015-03.</p></html>'))
  self.assertTrue(m.extract_candidate(*args)['scope_review']['eligible_for_candidate_review'])

 def test_2018_excluded_even_when_numbers_look_clean(self):
  r=m.extract_candidate(*fixture(year=2018,end='2018-12-29',prior='2017-12-30'))
  self.assertFalse(r['scope_review']['eligible_for_candidate_review']);self.assertTrue(r['scope_review']['blockers'])

 def test_2025_excluded_without_needing_a_keyword(self):
  r=m.extract_candidate(*fixture(year=2025,end='2025-12-27',prior='2024-12-28'))
  self.assertFalse(r['scope_review']['eligible_for_candidate_review'])

 def test_other_year_discontinued_also_excluded(self):
  r=m.extract_candidate(*fixture(extra='<p>Income from discontinued operations</p>'))
  self.assertFalse(r['scope_review']['eligible_for_candidate_review'])

 def test_2020_comparative_not_prior_release(self):
  r=m.extract_candidate(*fixture(year=2020,end='2020-12-26',prior='2019-12-28'))
  self.assertEqual(r['comparison']['gross_amount']['available_from'],'2021-02-01')
  self.assertEqual(r['fiscal_year'],2020);self.assertEqual(r['comparison_period']['end'],'2019-12-28')

 def test_2019_not_synthesized(self):
  with self.assertRaisesRegex(ValueError,'scope'):m.extract_candidate(*fixture(year=2019,end='2019-12-28',prior='2018-12-29'))

 def test_missing_identity_wrong_date_and_ambiguous_dates(self):
  changes=(lambda r:r.replace(b'AMD (NASDAQ:AMD)',b'AMD'),lambda r:r.replace(b'January 31, 2023 -- AMD',b'January 30, 2023 -- AMD'),lambda r:r.replace(b'</html>',b'<p>SANTA CLARA, Calif., January 30, 2023 -- AMD (NASDAQ:AMD)</p></html>'))
  for fn in changes:
   with self.subTest(fn=fn),self.assertRaises(ValueError):m.extract_candidate(*mutate(fixture(),fn))

 def test_original_dateline_in_script_does_not_count(self):
  args=mutate(fixture(),lambda r:r.replace(b'<p>SANTA',b'<script>SANTA').replace(b'Devices, Inc.</p>',b'Devices, Inc.</script>'))
  with self.assertRaisesRegex(ValueError,'anchor'):m.extract_candidate(*args)

 def test_prior_annual_header_required(self):
  args=mutate(fixture(),lambda r:r.replace(b'Year Ended',b'Three Months Ended'))
  with self.assertRaises(ValueError):m.extract_candidate(*args)

 def test_wrong_scale_other_currency_missing_dollar(self):
  for fn in (lambda r:r.replace(b'Millions',b'Thousands and millions'),lambda r:r.replace(b'Millions',b'EUR Millions'),lambda r:r.replace(b'$',b'')):
   with self.subTest(fn=fn),self.assertRaises(ValueError):m.extract_candidate(*mutate(fixture(),fn))

 def test_receipt_fail_closed(self):
  for key,value in [('status',404),('sha256','0'*64),('url','https://example.com'),('final_url','https://example.com'),('public_unauthenticated',False),('content_type','text/plain'),('retrieved_at',None)]:
   args=list(fixture());args[2]=dict(args[2],**{key:value})
   with self.subTest(key=key),self.assertRaises(ValueError):m.extract_candidate(*args)

 def test_raw_hash_fail_closed(self):
  args=list(fixture());args[1]['raw_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'SHA'):m.extract_candidate(*args)

 def test_catalog_fail_closed(self):
  for key,value in [('source_url','https://example.com'),('evidence_path','elsewhere'),('evidence_sha256','0'*64),('release_date','2023-01-30'),('fiscal_period_end_actual','2022-12-30'),('source_retrieved_at','elsewhere')]:
   args=list(fixture());args[3][0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.extract_candidate(*args)
  for events in ([],[fixture()[3][0]]*2):
   args=list(fixture());args[3]=events
   with self.assertRaises(ValueError):m.extract_candidate(*args)

 def test_calendar_fail_closed(self):
  for sessions in ([],['2023-02-01','2023-01-31'],['2023-02-01']*2,['2023-01-31'],['not-a-day']):
   args=list(fixture());args[4]=sessions
   with self.subTest(sessions=sessions),self.assertRaises(ValueError):m.extract_candidate(*args)

 def test_audit_detects_tampered_prior_value_and_date(self):
  args=fixture();r=m.extract_candidate(*args)
  for key,value in [('amount_decimal','777'),('displayed_amount','777'),('date','2020-12-31'),('source_raw_sha256','0'*64),('available_from','2022-02-01')]:
   c=copy.deepcopy(r);c['comparison']['gross_amount'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.audit_candidate(args[0],c)

 def test_audit_detects_removed_blocker_and_false_admission(self):
  args=fixture(year=2018,end='2018-12-29',prior='2017-12-30');r=m.extract_candidate(*args)
  c=copy.deepcopy(r);c['scope_review']['blockers']=[]
  with self.assertRaises(ValueError):m.audit_candidate(args[0],c)
  for key in ('fact_admitted','strategy_admitted'):
   c=copy.deepcopy(r);c[key]=True
   with self.assertRaises(ValueError):m.audit_candidate(args[0],c)

 def test_source_functions_do_not_open_files_or_network(self):
  args=fixture()
  with mock.patch('builtins.open',side_effect=AssertionError('I/O')),mock.patch('socket.socket',side_effect=AssertionError('network')):
   r=m.extract_candidate(*args);self.assertTrue(m.audit_candidate(args[0],r)['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
