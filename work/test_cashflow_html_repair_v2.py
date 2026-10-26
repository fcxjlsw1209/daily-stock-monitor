"""Synthetic representation repairs; no real monetary source extraction."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cashflow_html_repair_v2 as m
from test_cashflow_source_preflight_v1 import PERIODS, raw, spec, table

def combined():
 bs=table('bs');cf=table()
 # Explicit second statement heading occurs inside the same physical table.
 return bs.replace('</table>', '<tr><td>CONSOLIDATED STATEMENTS OF CASH FLOWS</td></tr><tr><td>(in millions)</td></tr>'+cf[cf.index('</caption>')+10:]).encode()

def mapped(content):
 s=spec(content);s['locator_inventory']=m.inventory(content,PERIODS)
 return m.map_event(s,content)

class RepairTests(unittest.TestCase):
 def generated_plan(self,d):
  content=raw();Path(d,'source.html').write_bytes(content)
  for p in (m.SELF,m.TEST):
   Path(d,p).parent.mkdir(parents=True,exist_ok=True);Path(d,p).write_text('synthetic code')
  specs=[];old=[];original_specs=[]
  for i in range(34):
   s=spec(content);s['event_id']=str(i);s['symbol']='AMD'if i<10 else'MSFT'if i<22 else'NVDA'
   original_specs.append(deepcopy(s));old.append(m.v1.map_event(s,content,{}))
   s['locator_inventory']=m.inventory(content,PERIODS);specs.append(s)
  plan=dict(stage='frozen_HTML_repair_plan',runtime=sys.version,events=specs,source_code_hashes=m.code_hashes(d),source_hashes={'source.html':m.io.hash_file(d,'source.html')})
  m.io.publish(d,m.OLD+'/RAW_FACTS.json',{'events':old})
  m.io.publish(d,m.OLD+'/PLAN.json',{'events':original_specs})
  m.io.publish(d,m.OUT+'/PLAN.json',plan)
  return m.io.hash_file(d,m.OUT+'/PLAN.json')
 def test_generated_build_preserves_all_old_values(self):
  with tempfile.TemporaryDirectory()as d:
   p=self.generated_plan(d);data,audit=m.products(d,p)
   self.assertEqual(data['counts']['existing_v1_cells_preserved'],204)
   self.assertEqual(data['counts']['newly_parsed_cells'],0)
   self.assertTrue(audit['all_inherited_rejections_preserved'])
 def test_frozen_source_change_stops_before_extract(self):
  with tempfile.TemporaryDirectory()as d:
   p=self.generated_plan(d);Path(d,'source.html').write_bytes(b'changed')
   with patch.object(m,'extract',side_effect=AssertionError('should not extract')):
    with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.products(d,p)
 def test_frozen_code_change_stops_before_extract(self):
  with tempfile.TemporaryDirectory()as d:
   p=self.generated_plan(d);Path(d,m.SELF).write_text('changed')
   with patch.object(m,'extract',side_effect=AssertionError('should not extract')):
    with self.assertRaisesRegex(ValueError,'code/runtime changed'):m.products(d,p)
 def test_duplicate_event_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   self.generated_plan(d);old=m.io.read(d,m.OLD+'/PLAN.json');events=deepcopy(old['events']);events[-1]=events[0]
   with self.assertRaisesRegex(ValueError,'canonical34'):m.validate_specs(events,old)
 def test_identity_and_available_date_cannot_change(self):
  with tempfile.TemporaryDirectory()as d:
   self.generated_plan(d);old=m.io.read(d,m.OLD+'/PLAN.json')
   for key,value in (('available_from','2099-01-01'),('symbol','OTHER'),('release_date','2099-01-01')):
    events=deepcopy(old['events']);events[0][key]=value
    with self.assertRaisesRegex(ValueError,'canonical metadata'):m.validate_specs(events,old)
 def test_unexpected_event_key_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   self.generated_plan(d);old=m.io.read(d,m.OLD+'/PLAN.json');events=deepcopy(old['events']);events[0]['unknown']=True
   with self.assertRaisesRegex(ValueError,'metadata keys'):m.validate_specs(events,old)
 def test_freeze_requires_metadata_draft_stage(self):
  with tempfile.TemporaryDirectory()as d:
   m.io.publish(d,m.OUT+'/DRAFT.json',{'stage':'other','schema':m.schema()})
   with self.assertRaisesRegex(ValueError,'metadata-only'):m.freeze(d,m.io.hash_file(d,m.OUT+'/DRAFT.json'))
 def test_v1_unchanged_standard_six_values(self):
  out=mapped(raw());old=m.v1.map_event(spec(),raw(),{})
  for side in m.v1.SIDES:
   for f in m.v1.FIELDS:
    self.assertEqual(out['fields'][side][f]['value_decimal'],old['fields'][side][f]['value_decimal'])
    self.assertEqual(out['fields'][side][f]['parsed_cell_sha256'],old['fields'][side][f]['parsed_cell_sha256'])
 def test_explicit_in_table_cf_heading_splits_statements(self):
  out=mapped(combined())
  self.assertEqual(out['fields']['current']['net_income']['value_decimal'],'314159')
  self.assertEqual(out['fields']['current']['total_assets']['value_decimal'],'567890')
  self.assertGreater(out['fields']['current']['net_income']['locator']['row_index_zero_based'],out['fields']['current']['total_assets']['locator']['row_index_zero_based'])
  self.assertEqual(out['fields']['current']['net_income']['locator']['table_index_zero_based'],0)
 def test_initial_title_after_blank_or_text_rows_not_split(self):
  for prefix in ('<tr><td></td></tr>','<tr><td>Unaudited financial statements</td></tr>'):
   text='<table>'+prefix+'<tr><td>CONSOLIDATED STATEMENTS OF CASH FLOWS</td></tr>'+table()[table().index('</caption>')+10:]
   self.assertEqual(m.inventory(text.encode(),PERIODS)['statement_segments'],[])
 def test_no_title_does_not_infer_cashflow_segment(self):
  content=combined().replace(b'CONSOLIDATED STATEMENTS OF CASH FLOWS',b'Other section')
  self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_metadata_inventory_never_parses_money(self):
  with patch.object(m.v1.html,'exact_amount',side_effect=AssertionError('metadata parsed money')):
   metadata=m.inventory(combined(),PERIODS)
  self.assertNotIn('314,159',m.io.encoded(metadata).decode())
  self.assertTrue(metadata['statement_segments'])
 def test_segment_requires_own_units(self):
  content=combined().replace(b'<tr><td>(in millions)</td></tr>',b'')
  self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_segment_requires_annual_columns(self):
  content=combined().replace(b'Years Ended',b'Three Months Ended')
  self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_duplicate_formal_segments_remain_ambiguous(self):
  content=(combined().decode().replace('</table>','')+'<tr><td>CONSOLIDATED STATEMENTS OF CASH FLOWS</td></tr><tr><td>(in millions)</td></tr>'+table()[table().index('</caption>')+10:]).encode()
  self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_cross_boundary_rowspan_blocks_split(self):
  content=combined().replace(b'<td>Assets</td>',b'<td rowspan="99">Assets</td>')
  inventory=m.inventory(content,PERIODS)
  self.assertEqual(inventory['statement_segments'],[])
  self.assertTrue(any(g['reason']=='segmentation_rejected_cross_boundary_span'for g in inventory['metadata_gaps']))
 def test_spaced_comma_literal_preserved(self):
  content=raw().replace(b'314,159',b'314 , 159');cell=mapped(content)['fields']['current']['net_income']
  self.assertEqual(cell['displayed_amount'],'$ 314 , 159')
  self.assertEqual(cell['normalized_monetary_literal'],'$ 314,159')
  self.assertEqual(cell['value_decimal'],'314159')
  self.assertTrue(cell['comma_whitespace_normalized'])
 def test_negative_multiple_spaced_groups(self):
  content=raw().replace(b'(27,182)',b'(1 , 234 , 567.50)');cell=mapped(content)['fields']['comparison']['net_income']
  self.assertEqual(cell['value_decimal'],'-1234567.50')
 def test_malformed_groups_not_repaired(self):
  for value in (b'12 , 34',b'1 234',b'1 , 234 . 50',b'NaN',b'2%',b'( -2 )'):
   with self.subTest(value=value):
    content=raw().replace(b'314,159',value);self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_exact_zero_remains_zero(self):
  self.assertEqual(mapped(raw())['fields']['comparison']['operating_cash_flow']['value_decimal'],'0')
 def test_no_new_ni_scope_or_summary_substitution(self):
  content=combined().replace(b'<td>Net income</td>',b'<td>Net income attributable to parent</td>')
  self.assertIsNone(mapped(content)['fields']['current']['net_income'])
 def test_all_six_positions_and_prior_rejections_retained(self):
  out=mapped(combined())
  self.assertEqual(out['inherited_rejection_codes'],['previous_field_rejected'])
  self.assertEqual(sum(len(x)for x in out['fields'].values()),6)
  self.assertFalse(out['fact_admitted']);self.assertFalse(out['model_admitted']);self.assertFalse(out['strategy_admitted'])
 def test_frozen_locator_change_rejected(self):
  s=spec(raw());s['locator_inventory']=m.inventory(raw(),PERIODS)
  s['locator_inventory']['candidates']['current']['net_income'][0]['row_index_zero_based']=999
  with self.assertRaisesRegex(ValueError,'locators changed'):m.map_event(s,raw())
 def test_missing_plan_or_bad_sha_blocks_build(self):
  with tempfile.TemporaryDirectory()as d:
   for value in (None,'','X'*64):
    with self.assertRaises(ValueError):m.products(d,value)
   with self.assertRaises(FileNotFoundError):m.products(d,'a'*64)
 def test_schema_scope_is_representation_only(self):
  s=m.schema();self.assertEqual(s['events'],34);self.assertEqual(s['maximum_cells'],204)
  self.assertIn('No new sources',s['exclusions'])

if __name__=='__main__':
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
 text=stream.getvalue();print(text,end='')
 if '--write-report'in sys.argv:
  m.io.publish(m.ROOT,m.OUT+'/TESTS.json',dict(passed=result.wasSuccessful(),synthetic_only=True,tests_run=result.testsRun,
   failures=len(result.failures),errors=len(result.errors),real_new_source_cells_extracted=0,source_code_hashes=m.code_hashes(m.ROOT),log=text))
 raise SystemExit(0 if result.wasSuccessful()else 1)
