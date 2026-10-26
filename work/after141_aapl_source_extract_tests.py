from copy import deepcopy
import unittest
import after141_aapl_source_extract as m

def fixture():
 op='''Apple Inc.
CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)
(In millions, except shares and per share amounts)
Three Months Ended                  Twelve Months Ended
September 28, September 30, September 28, September 30,
2024 2023 2024 2023
Net sales $ 5 $ 4 $ 20 $ 18
Gross margin 2 1 8 7
'''
 bs='''Apple Inc.
CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)
(In millions, except shares and par value)
September 28, September 30,
2024 2023
Cash $ 4 $ 3
Total assets $ 30 $ 27
'''
 return [op,bs],dict(fiscal_year=2024,fiscal_year_end='2024-09-28',release_date='2024-10-31')

class AppleTests(unittest.TestCase):
 def test_current_annual_and_current_instant(self):
  t,s=fixture();x=m.parse_layout_pages(t,s)
  self.assertEqual(x['gross_amount']['amount_decimal'],'8');self.assertEqual(x['total_assets']['amount_decimal'],'30')
  self.assertEqual(x['annual_period']['days_inclusive'],364);self.assertFalse(x['fact_admitted'])
 def test_zero_negative_gross_retained(self):
  for value in ['0','(8)','-8']:
   t,s=fixture();t[0]=t[0].replace('2 1 8 7','2 1 '+value+' 7')
   self.assertEqual(m.parse_layout_pages(t,s)['gross_amount']['amount_decimal'],m.decimal_cell(value))
 def test_percentage_rejected(self):
  t,s=fixture();t[0]=t[0].replace('2 1 8 7','2 1 8% 7')
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_non_gaap_rejected(self):
  t,s=fixture();t[0]+='NON-GAAP RECONCILIATION'
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_quarter_only_rejected(self):
  t,s=fixture();t[0]=t[0].replace('Twelve Months Ended','Three Months Ended')
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_split_duration_glyphs_supported(self):
  t,s=fixture();t[0]=t[0].replace('Twelve Months Ended','Twe l ve M o n t h s E n d e d')
  self.assertEqual(m.parse_layout_pages(t,s)['gross_amount']['amount_decimal'],'8')
 def test_dollarsign_glyph_supported(self):
  t,s=fixture();t=[p.replace('$','/dollarsign')for p in t]
  self.assertEqual(m.parse_layout_pages(t,s)['total_assets']['amount_decimal'],'30')
 def test_spaced_glyph_label_supported(self):
  t,s=fixture();t[1]=t[1].replace('Total assets','To    t  a l  a s s e t s')
  self.assertEqual(m.parse_layout_pages(t,s)['total_assets']['amount_decimal'],'30')
 def test_wrong_year_end_rejected(self):
  t,s=fixture();s['fiscal_year_end']='2024-09-27'
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_mismatched_asset_prior_end_rejected(self):
  t,s=fixture();t[1]=t[1].replace('September 30','September 29')
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_duplicate_row_rejected(self):
  t,s=fixture();t[0]+='Gross margin 2 1 8 7\n'
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_missing_currency_or_scale_rejected(self):
  for old,new in [('$',''),('In millions','In thousands')]:
   t,s=fixture();t=[p.replace(old,new)for p in t]
   with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_nonpositive_assets_rejected(self):
  t,s=fixture();t[1]=t[1].replace('$ 30','$ 0')
  with self.assertRaises(ValueError):m.parse_layout_pages(t,s)
 def test_bad_amounts_rejected(self):
  for c in ['NaN','Infinity','True','1,23','(-5)','1 234','12%']:
   with self.subTest(c=c),self.assertRaises(ValueError):m.decimal_cell(c)
 def test_source_input_not_mutated(self):
  t,s=fixture();before=deepcopy((t,s));m.parse_layout_pages(t,s);self.assertEqual((t,s),before)

if __name__=='__main__':unittest.main()
