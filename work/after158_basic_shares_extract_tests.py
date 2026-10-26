"""Synthetic source extraction tests. No original financial source reads."""
import unittest
import io
import sys
from copy import deepcopy
from bs4 import BeautifulSoup
import after158_basic_shares_extract as m

class Shares(unittest.TestCase):
 def fixture(self,quarter='3',prior='40',caption='INCOME STATEMENTS (In millions, except per share data)',annual='Year Ended'):
  raw=('''<table><caption>%s</caption><tr><td></td><td colspan="2">Three Months Ended</td><td colspan="2">%s</td></tr>
  <tr><td></td><td>December 31, 2023</td><td>December 31, 2022</td><td>December 31, 2023</td><td>December 31, 2022</td></tr>
  <tr><td>Revenue</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>
  <tr><td>Net income per share:</td></tr><tr><td>Basic</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
  <tr><td>Shares used in computing earnings per share:</td></tr>
  <tr><td>Basic</td><td>%s</td><td>4</td><td>50</td><td>%s</td></tr>
  <tr><td>Diluted</td><td>99</td><td>99</td><td>99</td><td>99</td></tr></table>'''%(caption,annual,quarter,prior)).encode()
  event={'fiscal_year':2023,'periods':{'current':{'period_end':'2023-12-31'},'comparison':{'period_end':'2022-12-31'}},'source':{'sha256':m.hashlib.sha256(raw).hexdigest()}}
  entry={'inventory':m.inv.html_inventory(raw,event)}
  return raw,event,entry
 def test_whole_table_selects_annual_basic(self):
  cells,_=m.html_extract(*self.fixture());self.assertEqual(cells['current']['value_decimal'],'50');self.assertEqual(cells['comparison']['value_decimal'],'40');self.assertEqual(cells['current']['unit']['scale'],1000000)
 def test_quarter_perturbation_invariant(self):
  a,_=m.html_extract(*self.fixture(quarter='999999'));b,_=m.html_extract(*self.fixture(quarter='0'));self.assertEqual(a['current']['value_in_shares'],b['current']['value_in_shares'])
 def test_whole_table_missing_annual_value_not_zero(self):
  with self.assertRaises(ValueError):m.html_extract(*self.fixture(prior='—'))
 def test_formal_table_not_prior_nongaap_narrative(self):
  raw,e,i=self.fixture();raw=b'<p>Non-GAAP reconciliation discussion</p>'+raw
  c,_=m.html_extract(raw,e,i);self.assertEqual(c['current']['value_decimal'],'50')
 def test_nongaap_table_rejected(self):
  with self.assertRaises(ValueError):m.html_extract(*self.fixture(caption='NON-GAAP INCOME STATEMENTS (In millions)'))
 def test_no_unit_invented(self):
  with self.assertRaises(ValueError):m.html_extract(*self.fixture(caption='INCOME STATEMENTS'))
 def test_no_annual_mapping_rejected(self):
  with self.assertRaises(ValueError):m.html_extract(*self.fixture(annual='Three Months Ended'))
 def test_multiple_candidates_not_selected_by_value(self):
  raw,e,i=self.fixture();i['inventory']['candidates']*=2
  with self.assertRaises(ValueError):m.html_extract(raw,e,i)
 def test_units_share_exception(self):
  x=m.unit_scale([{'text':'(In millions, except number of shares which are reflected in thousands and per share amounts)'}]);self.assertEqual(x['scale'],1000);self.assertTrue(x['share_specific'])
 def test_general_units(self):self.assertEqual(m.unit_scale([{'text':'(In millions, except per share data)'}])['scale'],1000000)
 def test_unknown_unit(self):
  with self.assertRaisesRegex(ValueError,'missing'):m.unit_scale([{'text':'Basic shares'}])
 def test_conflicting_unit(self):
  with self.assertRaisesRegex(ValueError,'conflicting'):m.unit_scale([{'text':'In thousands'},{'text':'In millions'}])
 def test_zero_is_real(self):self.assertEqual(m.amount('0'),'0')
 def test_decimal(self):self.assertEqual(m.amount('1,234.50'),'1234.50')
 def test_no_eps_reverse_or_dashes(self):
  for x in ('—','-','(2)','-2','$3','4%','1 234','12,34','nan','inf'):
   with self.subTest(x=x),self.assertRaises(ValueError):m.amount(x)
 def test_basic_semantics(self):
  self.assertEqual(m.inv.label_type('Basic',['Shares used in computing earnings per share:']),'basic_eps_denominator_weighting_implicit')
  self.assertEqual(m.inv.label_type('Basic',['Weighted average shares outstanding:']),'explicit_basic_weighted_average_shares')
 def test_diluted_and_eps_rejected(self):
  self.assertIsNone(m.inv.label_type('Diluted',['Weighted average shares outstanding:']))
  self.assertIsNone(m.inv.label_type('Basic',['Net income per share:']))
 def test_annual_exact_date(self):m.validate_annual([{'text':'Year Ended'},{'text':'December 31, 2023'}],'2023-12-31')
 def test_quarter_ambiguous(self):
  for duration in ('Three Months Ended','Year Ended Three Months Ended'):
   with self.assertRaises(ValueError):m.validate_annual([{'text':duration},{'text':'December 31, 2023'}],'2023-12-31')
 def test_wrong_year(self):
  with self.assertRaises(ValueError):m.validate_annual([{'text':'Year Ended'},{'text':'December 31, 2022'}],'2023-12-31')
 def test_split_disclosure_not_automatic(self):
  r=m.split_evidence([({'line':1},'All share amounts reflect a four-for-one stock split. Other facts.')]);self.assertEqual(len(r['evidence']),1);self.assertFalse(r['automatic_adjustment_applied']);self.assertFalse(r['complete_split_history_certified'])
 def test_no_split_keyword_not_certified(self):self.assertEqual(m.split_evidence([])['status'],'no_split_keyword_found_not_certified_absence')
 def test_colspan_physical_identity(self):
  _,g,r=m.inv.table_grid(BeautifulSoup('<table><tr><td>Basic</td><td colspan="3">42</td></tr></table>','html.parser').table);self.assertIs(g[0,1],g[0,3]);self.assertEqual(len(r[0]),2)
 def test_unit_normalization_only(self):
  e={'periods':{'current':{'period_end':'2023-12-31'}},'source':{'sha256':'a'*64}}
  c=m.cell('1.5',e,'current','basic_eps_denominator_weighting_implicit',{'scale':1000},{});self.assertEqual(c['value_in_shares'],'1500.0');self.assertFalse(c['weighted_average_explicit']);self.assertFalse(c['source_admission_changed'])
if __name__=='__main__':
 if sys.argv[1:]==['--record']:
  pins=m.implementation_hashes();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Shares));m.verify(pins)
  m.write_once(m.OUT+'/TESTS.json',{'passed':result.wasSuccessful(),'synthetic_only':True,'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'implementation_hashes':pins,'log':stream.getvalue(),'actual_financial_sources_read':False})
  print(stream.getvalue());sys.exit(0 if result.wasSuccessful()else 1)
 else:unittest.main()
