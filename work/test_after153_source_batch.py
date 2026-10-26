"""Pure generated metadata and cell results for batch5 contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import after153_source_batch as s
from test_after153_annual_extract import fixture


def context_fixture():
 events=[]
 for symbol,year in s.FULL:
  if (symbol,year) in s.SELECTED:
   _,c=fixture(symbol,year)
  else:
   c=dict(symbol=symbol,fiscal_year=year,event_id=f'{symbol}_{year+1}-02-01',publication_date=f'{year+1}-02-01',available_session=f'{year+1}-02-02',available_session_verified=True,
    official_actual_period_end=f'{year}-12-31',source=dict(path='synthetic.html',sha256='a'*64,url='https://example.invalid',receipt_path=None,receipt_sha256=None),
    source_kind='index_only',original_report_bytes_verified=False,context_status='not_located',context_holds=['missing'])
  c['census_source_kind']='issuer_original_release_full_bytes' if c['source_kind']=='HTML_original' else 'index_source'
  events.append(c)
 return dict(events=events)


def inputs():
 d={k:'a'*64 for k in s.BASE_INPUTS};d[s.CONTEXT]=s.CONTEXT_SHA;d[s.FEASIBILITY]=s.FEASIBILITY_SHA;return d


class SourceTests(unittest.TestCase):
 def test_fixed_scope21_catalog28(self):
  c=context_fixture();self.assertEqual(len(s.records(c)),21);self.assertEqual(len(s.catalog_metadata(c)),28)
 def test_missing_catalog_fails(self):
  c=context_fixture();c['events'].pop()
  with self.assertRaises(ValueError):s.records(c)
 def test_duplicate_catalog_fails(self):
  c=context_fixture();c['events'][-1]=copy.deepcopy(c['events'][0])
  with self.assertRaises(ValueError):s.records(c)
 def test_reordered_catalog_fails(self):
  c=context_fixture();c['events'].reverse()
  with self.assertRaises(ValueError):s.records(c)
 def test_missing_candidate_fails(self):
  c=context_fixture();c['events'][0]['context_status']='missing'
  with self.assertRaises(ValueError):s.records(c)
 def test_extra_candidate_fails(self):
  c=context_fixture();c['events'][-1]['context_status']=s.READY
  with self.assertRaises(ValueError):s.records(c)
 def test_index_candidate_fails(self):
  c=context_fixture();c['events'][0]['source_kind']='index'
  with self.assertRaises(ValueError):s.records(c)
 def test_policy_roundtrip(self):
  c=context_fixture();p=s.make_policy(c,inputs(),'fixed');s.validate_policy(json.loads(json.dumps(p)),c)
 def test_policy_boundary_367_rejected(self):
  c=context_fixture();p=s.make_policy(c,inputs(),'fixed');p['periods']['allowed_days'].append(367)
  with self.assertRaises(ValueError):s.validate_policy(p,c)
 def test_policy_drift_rejected(self):
  c=context_fixture();d=inputs();d[s.CONTEXT]='f'*64
  with self.assertRaises(ValueError):s.make_policy(c,d)
 def test_policy_missing_parent_source_rejected(self):
  c=context_fixture();d=inputs();d.pop(next(k for k in d if 'MANIFEST' in k))
  with self.assertRaises(ValueError):s.make_policy(c,d)
 def test_policy_catalog_projection12_fields(self):
  self.assertTrue(all(len(x)==12 for x in s.catalog_metadata(context_fixture())))
 def test_original_catalog_deep_preserved(self):
  c=context_fixture();raw=[dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source_kind=e['census_source_kind']) for e in c['events']]
  saved=copy.deepcopy(raw);r=s.original_catalog(dict(annual_documents=raw),c);self.assertEqual(r,saved);self.assertEqual(raw,saved)
 def test_original_kind_drift_rejected(self):
  c=context_fixture();raw=[dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source_kind='wrong') for e in c['events']]
  with self.assertRaises(ValueError):s.original_catalog(dict(annual_documents=raw),c)
 def test_actual_parser_generated_result_validates(self):
  for sym in s.parser.PROFILES:
   with self.subTest(sym=sym):raw,c=fixture(sym);s.validate_record(s.parser.extract_document(raw,c),c)
 def test_loss_allowed(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['annual_net_income']['value_decimal']='-10';s.validate_record(r,c)
 def test_zero_allowed(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['annual_net_income']['value_decimal']='0';s.validate_record(r,c)
 def test_null_with_gap_retained(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['annual_net_income']=None;r['gaps']=[dict(field='annual_net_income',reason='unparseable')];r['complete_required_fields']=False;s.validate_record(r,c)
 def test_null_no_gap_rejected(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['annual_net_income']=None
  with self.assertRaises(ValueError):s.validate_record(r,c)
 def test_certification_flag_rejected(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['first_public_time_verified']=True
  with self.assertRaises(ValueError):s.validate_record(r,c)
 def test_holds_drift_rejected(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['context_holds']=[]
  with self.assertRaises(ValueError):s.validate_record(r,c)
 def test_source_drift_rejected(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['prior_total_assets']['source']['path']='other.html'
  with self.assertRaises(ValueError):s.validate_record(r,c)
 def test_field_date_drift_rejected(self):
  raw,c=fixture();r=s.parser.extract_document(raw,c);r['fields']['prior_total_assets']['as_of']='2024-12-31'
  with self.assertRaises(ValueError):s.validate_record(r,c)
 def test_extract_requires_exact_plan_sha_before_reads(self):
  with patch.object(s.shared,'sha',return_value='a'*64):
   with self.assertRaises(ValueError):s.run('b'*64)
 def test_freeze_is_write_once(self):
  with tempfile.TemporaryDirectory() as d,patch.object(s,'OUT',Path(d)):
   Path(d,'PLAN.json').write_text('{}')
   with self.assertRaises(ValueError):s.freeze()

if __name__=='__main__':unittest.main(verbosity=2)
