"""Synthetic validation only; no market or financial source files loaded."""
import copy, json, tempfile, unittest
from pathlib import Path
import wc194_nvda_amd_extract as m

def fixture(ap='Accounts payable',annual='Year Ended',period='December 31, 2024',units='(In millions)',extra='',admitted=True,scope=False,ar='(23)',inv='7',pay='—'):
 rows=[f'<tr><td colspan="4">EXAMPLE CONSOLIDATED STATEMENTS OF CASH FLOWS {units}</td></tr>',
  f'<tr><td></td><td>Three Months Ended</td><td>{annual}</td><td>Year Ended</td></tr>',
  f'<tr><td></td><td>{period}</td><td>{period}</td><td>December 31, 2023</td></tr>',
  '<tr><td>Net income</td><td>$ 99</td><td>$ 100</td><td>$ 88</td></tr>',
  '<tr><td colspan="4">Changes in operating assets and liabilities, net of acquisitions:</td></tr>',
  f'<tr><td>Accounts receivable, net</td><td>999</td><td>{ar}</td><td>(77)</td></tr>',
  f'<tr><td>Inventories</td><td>888</td><td>{inv}</td><td>4</td></tr>',
  f'<tr><td>{ap}</td><td>777</td><td>{pay}</td><td>5</td></tr>']
 if extra:rows.append(extra)
 if scope:rows += ['<tr><td>Net cash provided by operating activities of continuing operations</td><td>1</td><td>400</td><td>1</td></tr>',
                  '<tr><td>Net cash provided by operating activities of discontinued operations</td><td>1</td><td>100</td><td>1</td></tr>']
 idx=len(rows);rows += ['<tr><td>Net cash provided by operating activities</td><td>$ 70</td><td>$ 500</td><td>$ 300</td></tr>',
  '<tr><td>Cash flows from investing activities</td><td></td><td></td><td></td></tr>']
 raw=('<table>'+''.join(rows)+'</table>').encode()
 loc={'format':'html','table_index_zero_based':0,'row_index_zero_based':idx,'logical_columns':[2],
      'row_label':'Net cash provided by operating activities','period_end':'2024-12-31','scale':'1000000'}
 source={'path':'synthetic.html','sha256':m.sha(raw),'url':'https://example.invalid/report'}
 target={'event_id':'AMD:FY2024:2025-02-04','symbol':'AMD','inherited_admitted':admitted,
         'inherited_available_from':'2025-02-05','current_cfo_locator':loc,'source':source}
 perioddata={'period_start':'2024-01-01','period_end':'2024-12-31','days_inclusive':366}
 old={'fiscal_year':2024,'release_date':'2025-02-04','periods':{'current':perioddata},
      'fields':{'current':{'operating_cash_flow':{'locator':loc,'value_decimal':'500','scale':'1000000','period':perioddata}}},
      'cash_quality_admission':{'reasons':[] if admitted else ['old_identity_hold']}}
 return raw,target,old

class SignedTests(unittest.TestCase):
 def test_signed_numeric(self):
  for literal,expected in [('12','12'),('$ (1,234 )','-1234'),('-3','-3'),('0','0'),('1.25','1.25'),('−8','-8')]:self.assertEqual(m.exact_signed(literal),expected)
 def test_confirmed_dash(self):
  for d in ('—','–','-','$ —'):self.assertEqual(m.exact_signed(d),'0')
 def test_ambiguous_rejected(self):
  for d in ('','(1','1)','(-1)','1 2','1,23','not available','(—)','$'):
   with self.assertRaises(ValueError):m.exact_signed(d)

class ExtractTests(unittest.TestCase):
 def get(self,**kwargs):return m.extract(*fixture(**kwargs))
 def test_exact_annual_sign_not_quarter(self):
  r=self.get();self.assertTrue(r['extraction_verified']);self.assertEqual(r['status'],'verified')
  self.assertEqual([r['fields'][k]['signed_value_decimal'] for k in m.KEYS],['-23','7','0'])
  self.assertFalse(r['component_sum_computed']);self.assertFalse(r['fact_admitted'])
 def test_direct_currency_preserved_missing(self):
  f=self.get()['fields']['accounts_receivable'];self.assertIsNone(f['currency_literal'])
  self.assertEqual(f['statement_currency_evidence']['literal'],'$');self.assertIsNone(f['currency_iso'])
 def test_complete_row_hash_source_and_headers(self):
  raw,t,o=fixture();r=m.extract(raw,t,o);f=r['fields']['inventories']
  self.assertEqual(m.sha(f['row_html'].encode()),f['row_html_sha256']);self.assertEqual(m.sha(f['row_text'].encode()),f['row_text_sha256'])
  self.assertEqual(f['source'],t['source']);self.assertEqual(f['locator']['logical_columns'],[2]);self.assertTrue(f['locator']['annual_header_evidence'])
 def test_inherited_hold_retained(self):
  r=self.get(admitted=False);self.assertTrue(r['extraction_verified']);self.assertEqual(r['status'],'held');self.assertIn('old_identity_hold',r['hold_reasons'])
 def test_combined_payable_rejected(self):
  r=self.get(ap='Accounts payable, accrued liabilities and other');self.assertIsNone(r['fields']['accounts_payable']);self.assertIsNotNone(r['fields']['inventories'])
  self.assertFalse(r['extraction_verified']);self.assertIn('combined_or_other_component_label',r['component_hold_reasons']['accounts_payable'])
 def test_nontrade_ar_rejected(self):
  raw,t,o=fixture();raw=raw.replace(b'Accounts receivable, net',b'Vendor non-trade receivables');r=m.extract(raw,t,o)
  self.assertIsNone(r['fields']['accounts_receivable'])
 def test_inventory_adjustment_does_not_replace(self):
  raw,t,o=fixture();raw=raw.replace(b'<td>Inventories</td>',b'<td>Amortization of inventory fair value adjustment</td>');r=m.extract(raw,t,o)
  self.assertIsNone(r['fields']['inventories'])
 def test_duplicate_exact_component_held(self):
  r=self.get(extra='<tr><td>Inventories</td><td>1</td><td>2</td><td>3</td></tr>');self.assertIsNone(r['fields']['inventories'])
  self.assertIn('multiple_component_rows_ambiguous_scope',r['component_hold_reasons']['inventories'])
 def test_no_scope_merge(self):
  r=self.get(scope=True);self.assertTrue(all(v is None for v in r['fields'].values()));self.assertTrue(r['scope_conflict_evidence'])
  self.assertEqual(r['candidate_evidence']['accounts_receivable'][0]['annual_cell']['signed_value_decimal'],'-23')
 def test_missing_component_not_zero(self):
  r=self.get(inv='');self.assertIsNone(r['fields']['inventories'])
 def test_missing_locator_no_fallback(self):
  raw,t,o=fixture();t['current_cfo_locator']=None;r=m.extract(raw,t,o);self.assertTrue(all(v is None for v in r['fields'].values()))
 def test_no_annual_header(self):
  r=self.get(annual='Three Months Ended');self.assertFalse(r['extraction_verified']);self.assertIn('not_current_annual_column',r['component_hold_reasons']['inventories'])
 def test_wrong_year_rejected(self):
  r=self.get(period='December 31, 2023');self.assertFalse(r['extraction_verified'])
 def test_wrong_unit_rejected(self):
  self.assertFalse(self.get(units='(In thousands)')['extraction_verified'])
 def test_cashflow_title_required(self):
  raw,t,o=fixture();raw=raw.replace(b'STATEMENTS OF CASH FLOWS',b'BALANCE SHEETS');self.assertFalse(m.extract(raw,t,o)['extraction_verified'])
 def test_heading_required(self):
  raw,t,o=fixture();raw=raw.replace(b'Changes in operating assets and liabilities, net of acquisitions:',b'Unrelated heading');self.assertFalse(m.extract(raw,t,o)['extraction_verified'])
 def test_cfo_must_reconcile_original(self):
  raw,t,o=fixture();o['fields']['current']['operating_cash_flow']['value_decimal']='501';r=m.extract(raw,t,o);self.assertFalse(r['extraction_verified'])
 def test_not_inherited_table_does_not_override(self):
  raw,t,o=fixture();raw+=b'<table><tr><td>Accounts receivable</td><td>12345</td></tr></table>';r=m.extract(raw,t,o)
  self.assertEqual(r['fields']['accounts_receivable']['signed_value_decimal'],'-23');self.assertEqual(len(r['candidate_evidence']['accounts_receivable']),2)
 def test_sign_split_and_colspan(self):
  raw,t,o=fixture(ar='(23');raw=raw.replace(b'<td>(23</td><td>(77)</td>',b'<td>(23</td><td>)</td><td>(77)</td>');r=m.extract(raw,t,o)
  self.assertEqual(r['fields']['accounts_receivable']['signed_value_decimal'],'-23');self.assertEqual(r['fields']['accounts_receivable']['locator']['logical_columns'],[2,3])
 def test_inputs_not_mutated(self):
  raw,t,o=fixture();before=copy.deepcopy((t,o));m.extract(raw,t,o);self.assertEqual((t,o),before)

class GovernanceTests(unittest.TestCase):
 def test_hash_and_path_validation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'x').write_bytes(b'x');self.assertEqual(m.source_bytes(root,{'path':'x','sha256':m.sha(b'x')}),b'x')
   with self.assertRaises(ValueError):m.source_bytes(root,{'path':'x','sha256':'0'*64})
   with self.assertRaises(ValueError):m.source_bytes(root,{'path':'../x','sha256':'0'*64})
 def test_receipt_binding(self):
  s={'url':'u','sha256':'a','receipt_entry_key':'u'};e={'u':{'url':'u','sha256':'a','status':200}}
  self.assertEqual(m.validate_receipt(s,e)['status'],200)
  e['u']['status']=404
  with self.assertRaises(ValueError):m.validate_receipt(s,e)
 def test_write_once(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';m.write_once(p,{'a':1})
   with self.assertRaises(FileExistsError):m.write_once(p,{'a':2})
   self.assertEqual(json.loads(p.read_text()),{'a':1})
if __name__=='__main__':unittest.main(verbosity=2)
