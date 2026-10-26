"""Pure synthetic WC194 original-source parser tests; no sums or financial ratios."""
import copy,io,json,unittest
from decimal import Decimal
import wc194_independent as a
b=a.b

def fixture(aplabel='Accounts payable',ar='(7)',inv='5',ap='(3)',duplicate=False,year=2020):
 labels=[('Accounts receivable, net',ar),('Inventories',inv),(aplabel,ap)]
 if duplicate:labels.append(('Accounts payable',ap))
 rows='<tr><td></td><td>Three Months Ended December 31, 2020</td><td>Year Ended December 31, 2020</td></tr><tr><td>Net income</td><td>3</td><td>8</td></tr><tr><td>Changes in operating assets and liabilities:</td><td></td><td></td></tr>'
 for label,value in labels:rows+=f'<tr><td>{label}</td><td>999</td><td>{value}</td></tr>'
 cfrow=3+len(labels);rows+='<tr><td>Net cash from operations</td><td>10</td><td>12</td></tr><tr><td>Investing activities:</td><td></td><td></td></tr>'
 raw=('<table><caption>CASH FLOWS STATEMENTS (In millions)</caption>'+rows+'</table>').encode()
 loc=dict(format='html',table_index_zero_based=0,row_index_zero_based=cfrow,row_label='Net cash from operations',logical_columns=[2],header_literals=['Year Ended December 31, 2020'],period_end=f'{year}-12-31',scale='1000000',statement_title='CASH FLOWS STATEMENTS')
 f=dict(source={'sha256':b.sha(raw)},locator=loc,value_decimal='12',scale='1000000',currency_literal='$')
 e={'fields':{'current':{'operating_cash_flow':f,'total_assets':None},'comparison':{'total_assets':None}},'periods':{'current':{'period_end':'2020-12-31'}}}
 return raw,e

class SourceTests(unittest.TestCase):
 def test_signed_three_components_and_annual(self):
  raw,e=fixture();fields,holds,_,pairs=a.html_components(raw,e)
  self.assertEqual({k:fields[k]['signed_value_decimal'] for k in a.KEYS},{'accounts_receivable':'-7','inventories':'5','accounts_payable':'-3'})
  self.assertFalse(holds);self.assertTrue(pairs['current_operating_cash_flow']['checked'])
 def test_positive_sign_not_flipped(self):
  raw,e=fixture(ar='7',inv='(5)',ap='3');f,_,_,_=a.html_components(raw,e)
  self.assertEqual(f['accounts_receivable']['signed_value_decimal'],'7');self.assertEqual(f['inventories']['signed_value_decimal'],'-5')
 def test_combined_ap_partial_hold(self):
  raw,e=fixture('Accounts payable, accrued liabilities and other');f,h,_,_=a.html_components(raw,e)
  self.assertIsNone(f['accounts_payable']);self.assertIsNotNone(f['accounts_receivable']);self.assertIsNotNone(f['inventories']);self.assertEqual(h,['accounts_payable_combined_scope_not_separable'])
 def test_duplicate_ap_hold(self):
  raw,e=fixture(duplicate=True);f,h,_,_=a.html_components(raw,e);self.assertIsNone(f['accounts_payable']);self.assertTrue(h)
 def test_bad_year_rejected(self):
  raw,e=fixture(year=2021)
  with self.assertRaises(ValueError):a.html_components(raw,e)
 def test_quarter_column_rejected(self):
  raw,e=fixture();loc=e['fields']['current']['operating_cash_flow']['locator'];loc['logical_columns']=[1];e['fields']['current']['operating_cash_flow']['value_decimal']='10'
  with self.assertRaises(ValueError):a.html_components(raw,e)
 def test_blank_not_zero(self):
  for amount in ('','—','-','N/A'):
   raw,e=fixture(ar=amount)
   with self.subTest(amount=amount),self.assertRaises(ValueError):a.html_components(raw,e)
 def test_explicit_zero_valid(self):
  raw,e=fixture(ar='0');f,_,_,_=a.html_components(raw,e);self.assertEqual(f['accounts_receivable']['signed_value_decimal'],'0')
 def test_missing_original_cfo_all_held(self):
  raw,e=fixture();e['fields']['current']['operating_cash_flow']=None;f,h,_,_=a.html_components(raw,e)
  self.assertTrue(all(v is None for v in f.values()));self.assertEqual(h,['missing_formal_current_annual_cashflow_column'])
 def test_wrong_cfo_amount_rejected(self):
  raw,e=fixture();e['fields']['current']['operating_cash_flow']['value_decimal']='13'
  with self.assertRaises(ValueError):a.html_components(raw,e)
 def test_wrong_cfo_source_rejected(self):
  raw,e=fixture();e['fields']['current']['operating_cash_flow']['source']['sha256']='wrong'
  with self.assertRaises(ValueError):a.html_components(raw,e)
 def test_balance_sheet_only_cannot_substitute(self):
  raw,e=fixture();raw=raw.replace(b'<td>Changes in operating assets and liabilities:</td>',b'<td>Balance sheets</td>');e['fields']['current']['operating_cash_flow']['source']['sha256']=b.sha(raw)
  with self.assertRaises(ValueError):a.html_components(raw,e)
 def test_label_prefix_not_pure(self):
  for label in ('Accounts payable and other liabilities','Vendor accounts receivable','Payables to related parties','Inventories and advances'):
   self.assertFalse(any(a.exact_component_label(label,k) for k in a.KEYS))
 def test_split_sign_cells(self):
  h=b.Tables();h.feed('<table><tr><td>Accounts payable</td><td>(</td><td>3</td><td>)</td></tr></table>')
  self.assertEqual(b.number(b.selected(h.tables[0],0,[1,2,3])[0]),Decimal('-3'))
 def test_right_parenthesis_only_extension(self):
  h=b.Tables();h.feed('<table><tr><td>Accounts payable</td><td>(3</td><td>)</td><td>9</td></tr></table>')
  lit,v,_,cols=a.signed_cell(h.tables[0],0,[1]);self.assertEqual(v,Decimal('-3'));self.assertEqual(cols,[1,2])
 def test_left_parenthesis_only_extension(self):
  h=b.Tables();h.feed('<table><tr><td>Accounts payable</td><td>(</td><td>3</td><td>)</td><td>9</td></tr></table>')
  self.assertEqual(a.signed_cell(h.tables[0],0,[2])[1],Decimal('-3'))
 def test_no_numeric_neighbour_extension(self):
  h=b.Tables();h.feed('<table><tr><td>Accounts payable</td><td>(3</td><td>9</td></tr></table>')
  with self.assertRaises(ValueError):a.signed_cell(h.tables[0],0,[1])
 def test_continuing_components_total_cfo_scope_hold(self):
  raw,e=fixture();raw=raw.replace(b'<tr><td>Net cash from operations</td>',b'<tr><td>Net cash provided by operating activities of continuing operations</td><td>5</td><td>9</td></tr><tr><td>Net cash from operations</td>')
  e['fields']['current']['operating_cash_flow']['source']['sha256']=b.sha(raw);e['fields']['current']['operating_cash_flow']['locator']['row_index_zero_based']+=1
  f,h,ev,_=a.html_components(raw,e);self.assertTrue(all(x is None for x in f.values()));self.assertIn('scope_conflict',ev[-1]);self.assertEqual(h,['current_cfo_total_vs_continuing_component_scope_conflict'])
 def test_mixed_span_rejected(self):
  h=b.Tables();h.feed('<table><tr><td>Accounts payable</td><td colspan="2">3</td></tr></table>')
  with self.assertRaises(ValueError):b.selected(h.tables[0],0,[1])
 def test_release_date_not_same_day(self):self.assertEqual(b.next_session('2020-01-03',['2020-01-03','2020-01-06']),'2020-01-06')
 def test_preflight_lead_not_admission(self):
  self.assertEqual(a.preliminary_status({'symbol':'AMD','fiscal_year':2016}),'accounts_payable_combined_with_accrued_liabilities_and_other')

if __name__=='__main__':unittest.main(verbosity=2)
