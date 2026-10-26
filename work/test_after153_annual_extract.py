"""Generated source cells only; never reads a real report."""
import copy
import hashlib
import unittest
from bs4 import BeautifulSoup
import after153_annual_extract as p
import after153_next_source_context as ctx
from test_after153_next_source_context import fixture as tables


def fixture(symbol='PLTR',year=None,percent=False,split=False):
 year=year or {'PLTR':2024,'PYPL':2018,'TMO':2020}[symbol]
 profile=p.PROFILES[symbol]
 raw=('<p>'+profile['issuer']+' (NYSE: '+symbol+')</p>'+tables(year,split=split,percent=percent)).encode()
 digest=hashlib.sha256(raw).hexdigest();source=dict(path='synthetic/report.html',sha256=digest,url='https://'+profile['hosts'][0]+'/report.html',receipt_path='synthetic/receipt.json',receipt_sha256='b'*64)
 parsed=[x for i,t in enumerate(BeautifulSoup(raw,'html.parser').find_all('table')) if (x:=ctx.table_context(t,i)) is not None]
 selected,gaps=ctx.choose_pair(parsed,year,f'{year}-12-31');assert not gaps
 c=dict(symbol=symbol,fiscal_year=year,event_id=f'{symbol}_{year+1}-02-01',publication_date=f'{year+1}-02-01',available_session=f'{year+1}-02-02',available_session_verified=True,
  context_status=p.reference.STATUS,source_kind='HTML_original',format='html_or_text',original_report_bytes_verified=True,
  source=source,evidence_path=source['path'],evidence_sha256=digest,source_url=source['url'],receipt=dict(path=source['receipt_path'],sha256=source['receipt_sha256']),
  identity_anchors=[profile['issuer']],official_actual_period_end=f'{year}-12-31',context_holds=['not_certified','accounting_pending'],**selected)
 for field in c['fields'].values():field.update(source=copy.deepcopy(source),available_from=c['available_session'],availability_verified=True)
 return raw,c


def refresh(raw,c):
 c=copy.deepcopy(c);digest=hashlib.sha256(raw).hexdigest();c['source']['sha256']=c['evidence_sha256']=digest
 for f in c['fields'].values():f['source']['sha256']=digest
 return c


class ParserTests(unittest.TestCase):
 def assert_gap(self,result,field=None):
  self.assertFalse(result['complete_required_fields'])
  if field:self.assertIsNone(result['fields'][field])
 def test_annual_not_quarter(self):
  raw,c=fixture();r=p.extract_document(raw,c);self.assertTrue(r['complete_required_fields']);self.assertEqual(r['fields']['annual_net_income']['value_decimal'],'873')
 def test_three_issuer_profiles(self):
  for s in p.PROFILES:
   with self.subTest(s=s):raw,c=fixture(s);self.assertTrue(p.extract_document(raw,c)['complete_required_fields'])
 def test_split_titles(self):
  raw,c=fixture('TMO',split=True);self.assertTrue(p.extract_document(raw,c)['complete_required_fields'])
 def test_percentage_columns_excluded(self):
  raw,c=fixture('TMO',percent=True);r=p.extract_document(raw,c);self.assertTrue(r['complete_required_fields']);self.assertEqual(r['fields']['annual_net_income']['value_decimal'],'873')
 def test_negative_income(self):
  raw,c=fixture();raw=raw.replace(b'$ 873',b'$ (873)');c=refresh(raw,c);self.assertEqual(p.extract_document(raw,c)['fields']['annual_net_income']['value_decimal'],'-873')
 def test_zero_income(self):
  raw,c=fixture();raw=raw.replace(b'$ 873',b'$ 0');c=refresh(raw,c);self.assertEqual(p.extract_document(raw,c)['fields']['annual_net_income']['value_decimal'],'0')
 def test_original_context_unchanged(self):
  raw,c=fixture();saved=copy.deepcopy(c);p.extract_document(raw,c);self.assertEqual(c,saved)
 def test_holds_never_cleared(self):
  raw,c=fixture();r=p.extract_document(raw,c);self.assertEqual(r['context_holds'],c['context_holds'])
 def test_all_admission_flags_false(self):
  raw,c=fixture();r=p.extract_document(raw,c)
  for flag in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','first_public_time_verified','complete_correction_history_verified','certified_daily_security_master'):self.assertIs(r[flag],False)
 def test_field_and_source_provenance(self):
  raw,c=fixture();r=p.extract_document(raw,c)
  for field in r['fields'].values():self.assertEqual(field['source'],c['source']);self.assertTrue(field['locator']['physical_cells']);self.assertTrue(field['locator']['header_evidence'])
 def test_raw_hash_mismatch_hard_fails(self):
  raw,c=fixture()
  with self.assertRaises(p.SourceError):p.extract_document(raw+b' ',c)
 def test_receipt_mismatch_hard_fails(self):
  raw,c=fixture();c['receipt']['sha256']='c'*64
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_identity_hard_fails(self):
  raw,c=fixture();c['identity_anchors']=['wrong issuer']
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_other_year_hard_fails(self):
  raw,c=fixture();c['fiscal_year']=1999
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_other_symbol_hard_fails(self):
  raw,c=fixture();c['symbol']='SPY'
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_external_host_hard_fails(self):
  raw,c=fixture();c['source']['url']='https://bad.invalid/report'
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_traversal_hard_fails(self):
  raw,c=fixture();c['source']['path']='../report.html'
  with self.assertRaises(p.SourceError):p.extract_document(raw,c)
 def test_unknown_clock_is_not_certified(self):
  raw,c=fixture();c['available_session_verified']=False
  for f in c['fields'].values():f['availability_verified']=False
  r=p.extract_document(raw,c);self.assertTrue(r['complete_required_fields']);self.assertFalse(r['first_public_time_verified'])
 def test_invalid_availability_is_gap(self):
  raw,c=fixture();c['available_session']=c['publication_date'];self.assert_gap(p.extract_document(raw,c))
 def test_367_days_rejected(self):
  raw,c=fixture();c['periods']['days_inclusive']=367;self.assert_gap(p.extract_document(raw,c))
 def test_non_calendar_end_rejected(self):
  raw,c=fixture();c['periods']['current_end']='2024-12-30';self.assert_gap(p.extract_document(raw,c))
 def test_start_not_quoted(self):
  raw,c=fixture();r=p.extract_document(raw,c);self.assertFalse(r['periods']['start_assumption_certified']);self.assertEqual(r['periods']['current']['days_inclusive'],366)
 def test_fabricated_approval_rejected(self):
  raw,c=fixture();c['periods']['start_assumption_approved']=True;self.assert_gap(p.extract_document(raw,c))
 def test_changed_period_column_rejected(self):
  raw,c=fixture();c['fields']['annual_net_income']['logical_columns']=[1];self.assert_gap(p.extract_document(raw,c),'annual_net_income')
 def test_changed_scale_rejected(self):
  raw,c=fixture();c['fields']['annual_net_income']['scale']='1000';self.assert_gap(p.extract_document(raw,c),'annual_net_income')
 def test_currency_iso_not_inferred(self):
  raw,c=fixture();r=p.extract_document(raw,c);self.assertIsNone(r['fields']['annual_net_income']['currency_iso'])
 def test_parent_attributable_not_used(self):
  raw,c=fixture();c['fields']['annual_net_income']['row']+=1;self.assert_gap(p.extract_document(raw,c),'annual_net_income')
 def test_bad_number_retained_as_gap(self):
  raw,c=fixture();raw=raw.replace(b'$ 873',b'not available');c=refresh(raw,c);self.assert_gap(p.extract_document(raw,c),'annual_net_income')
 def test_other_currency_rejected(self):
  raw,c=fixture();raw=raw.replace(b'$ 873',b'EUR 873');c=refresh(raw,c);self.assert_gap(p.extract_document(raw,c),'annual_net_income')
 def test_source_clock_applies_to_prior_assets(self):
  raw,c=fixture();r=p.extract_document(raw,c);self.assertEqual(r['fields']['prior_total_assets']['available_from'],c['available_session']);self.assertEqual(r['fields']['prior_total_assets']['as_of'],'2023-12-31')

if __name__=='__main__':unittest.main(verbosity=2)
