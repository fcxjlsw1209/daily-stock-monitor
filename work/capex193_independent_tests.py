"""Synthetic-only tests for independent source extraction and producer comparison."""
import copy,io,json,unittest
from decimal import Decimal
from pathlib import Path
import capex193_independent as a
import capex193_independent_review as r

def fixture(label='Additions to property and equipment',amount='(7)',duplicate=False):
 rows='<tr><td></td><td>Three Months Ended December 31, 2020</td><td>Year Ended December 31, 2020</td></tr><tr><td>Net income</td><td>3</td><td>8</td></tr><tr><td>Net cash from operations</td><td>5</td><td>12</td></tr><tr><td>Investing:</td><td></td><td></td></tr>'
 line=f'<tr><td>{label}</td><td>(2)</td><td>{amount}</td></tr>'
 raw=('<table><caption>CASH FLOWS STATEMENTS (In millions)</caption>'+rows+line+(line if duplicate else '')+'</table>').encode()
 loc=dict(format='html',table_index_zero_based=0,row_index_zero_based=2,row_label='Net cash from operations',logical_columns=[2],header_literals=['Year Ended December 31, 2020'],period_end='2020-12-31',scale='1000000',statement_title='CASH FLOWS STATEMENTS')
 field=dict(source={'sha256':a.sha(raw)},locator=loc,value_decimal='12',scale='1000000')
 e={'fields':{'current':{'operating_cash_flow':field,'total_assets':None},'comparison':{'total_assets':None}}}
 return raw,e

def comparison_fixture(held=False):
 raw,event=fixture();cap,_,_,_=a.html_extract(raw,event)
 e=dict(event_id='S:2020',symbol='S',fiscal_year=2020,release_date='2021-02-01',available_from='2021-02-02',inherited_admitted=not held,source={'path':'raw','sha256':'abc'},source_sha256='abc',cash_ppe_verified=True,inherited_hold_reasons=['old hold'] if held else [],positive_outflow_decimal='7',signed_value_decimal='-7',scale='1000000',annual_period_end='2020-12-31',capex_evidence=cap,paired_cfo_verified=True,paired_assets_verified=True)
 f=dict(signed_literal='(7)',positive_outflow_decimal='7',signed_value_decimal='-7',scale='1000000',row_label='Additions to property and equipment',row_text=cap['row_text'],row_text_sha256=cap['row_text_sha256'],period={'period_end':'2020-12-31'},locator=dict(format='html',period_end='2020-12-31',table_index_zero_based=0,row_index_zero_based=4,logical_columns=[2]))
 p={k:copy.deepcopy(e[k]) for k in ('event_id','symbol','fiscal_year','release_date','available_from','inherited_admitted','source')};p.update(extraction_verified=True,status='held' if held else 'verified',fact_admitted=False,strategy_admitted=False,inherited_holds=e['inherited_hold_reasons'][:],hold_reasons=e['inherited_hold_reasons'][:],field=f)
 return e,p

class SourceTests(unittest.TestCase):
 def test_parentheses_spaces(self):self.assertEqual(a.number('$ ( 1,234 )'),Decimal('-1234'))
 def test_explicit_zero(self):self.assertEqual(a.number('0'),0)
 def test_missing_not_zero(self):
  for x in ('','—','-','N/A','1 2?'):
   with self.subTest(x=x),self.assertRaises(ValueError):a.number(x)
 def test_malformed_comma_rejected(self):
  with self.assertRaises(ValueError):a.number('1,23')
 def test_split_paren_cells(self):
  h=a.Tables();h.feed('<table><tr><td>A</td><td>(</td><td>12</td><td>)</td></tr></table>');lit,cells=a.selected(h.tables[0],0,[1,2,3]);self.assertEqual(a.number(lit),-12)
 def test_spanning_annual_and_quarter_rejected(self):
  h=a.Tables();h.feed('<table><tr><td>A</td><td colspan="2">12</td></tr></table>')
  with self.assertRaises(ValueError):a.selected(h.tables[0],0,[1])
 def test_colspan_own_annual_allowed(self):
  h=a.Tables();h.feed('<table><tr><td>A</td><td colspan="2">(12)</td></tr></table>');self.assertEqual(a.number(a.selected(h.tables[0],0,[1,2])[0]),-12)
 def test_annual_not_quarter(self):
  raw,e=fixture();cap,holds,_,pairs=a.html_extract(raw,e);self.assertEqual(cap['positive_outflow_decimal'],'7');self.assertFalse(holds);self.assertTrue(pairs['current_operating_cash_flow']['checked'])
 def test_wrong_annual_year_rejected(self):
  raw,e=fixture();e['fields']['current']['operating_cash_flow']['locator']['period_end']='2021-12-31'
  with self.assertRaises(ValueError):a.html_extract(raw,e)
 def test_combined_intangible_hold(self):
  raw,e=fixture('Purchases of property and equipment and intangible assets');cap,holds,_,_=a.html_extract(raw,e);self.assertIsNone(cap);self.assertEqual(holds,['combined_ppe_and_intangible_not_separable'])
 def test_two_scope_rows_hold(self):
  raw,e=fixture(duplicate=True);cap,holds,_,_=a.html_extract(raw,e);self.assertIsNone(cap);self.assertEqual(holds,['multiple_scope_pp_and_e_rows'])
 def test_unsigned_positive_rejected(self):
  raw,e=fixture(amount='7')
  with self.assertRaises(ValueError):a.html_extract(raw,e)
 def test_missing_capex_cell_rejected(self):
  raw,e=fixture(amount='')
  with self.assertRaises(ValueError):a.html_extract(raw,e)
 def test_original_cfo_tamper_rejected(self):
  raw,e=fixture();e['fields']['current']['operating_cash_flow']['value_decimal']='13'
  with self.assertRaises(ValueError):a.html_extract(raw,e)
 def test_clock_strict_after_release(self):self.assertEqual(a.next_session('2020-01-03',['2020-01-03','2020-01-06']),'2020-01-06')
 def test_clock_horizon_fail(self):
  with self.assertRaises(ValueError):a.next_session('2020-01-06',['2020-01-03'])

class CompareTests(unittest.TestCase):
 def test_exact(self):
  e,p=comparison_fixture();self.assertTrue(r.compare_records([e],[p])[0]['producer_comparison_passed'])
 def test_inherited_hold_retained(self):
  e,p=comparison_fixture(True);self.assertEqual(r.compare_records([e],[p])[0]['producer_status'],'held')
 def test_explicit_inherited_prefix_retained(self):
  e,p=comparison_fixture(True);p['hold_reasons']=['inherited:old hold'];self.assertTrue(r.compare_records([e],[p]))
 def test_promotion_of_old_hold_rejected(self):
  e,p=comparison_fixture(True);p['status']='verified'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_missing_hold_rejected(self):
  e,p=comparison_fixture(True);p['hold_reasons']=[]
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_amount_tamper_rejected(self):
  e,p=comparison_fixture();p['field']['positive_outflow_decimal']='8'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_column_tamper_rejected(self):
  e,p=comparison_fixture();p['field']['locator']['logical_columns']=[1]
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_source_tamper_rejected(self):
  e,p=comparison_fixture();p['source']['sha256']='wrong'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_coverage_rejected(self):
  e,p=comparison_fixture()
  for ps in ([],[p,p]):
   with self.assertRaises(ValueError):r.compare_records([e],ps)
 def test_paired_source_required(self):
  e,p=comparison_fixture();e['paired_assets_verified']=False
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_missing_field_hold_not_zero(self):
  e,p=comparison_fixture();e['cash_ppe_verified']=False;p.update(extraction_verified=False,status='held',field=None,hold_reasons=['missing']);self.assertTrue(r.compare_records([e],[p]))
  p['field']={}
  with self.assertRaises(ValueError):r.compare_records([e],[p])

if __name__=='__main__':
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__('__main__')));print(stream.getvalue())
 if result.wasSuccessful():
  pins={p:a.sha(a.path(p).read_bytes()) for p in ('work/capex193_independent.py','work/capex193_independent_review.py','work/capex193_independent_tests.py')}
  a.publish('TESTS_02.json',dict(passed=True,tests_run=result.testsRun,synthetic_only=True,code_sha256=pins,log=stream.getvalue()))
 else:raise SystemExit(1)
