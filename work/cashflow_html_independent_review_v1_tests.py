"""Synthetic tests for the independent HTML reviewer; no producer imports."""
import unittest
import cashflow_html_independent_review_v1 as r

PERIODS={s:{'period_end':end}for s,end in (('current','2024-12-31'),('comparison','2023-12-31'))}
def source(quarter=False,label='Net income'):
 duration='Three Months Ended'if quarter else'Years Ended'
 return ('<table><caption>CONSOLIDATED STATEMENTS OF CASH FLOWS (in millions)</caption>'
  '<tr><td></td><td colspan="2">'+duration+'</td></tr>'
  '<tr><td></td><td>December 31, 2024</td><td>December 31, 2023</td></tr>'
  '<tr><td>'+label+'</td><td>$ 101</td><td>$ (51)</td></tr>'
  '<tr><td>Adjustments to reconcile net income to cash from operations:</td></tr>'
  '<tr><td>Net cash from operations</td><td>202</td><td>0</td></tr></table>').encode()

class ReviewTests(unittest.TestCase):
 def test_exact_negative_and_zero(self):
  self.assertEqual(r.exact_money('$ (1,234.50)'),'-1234.50')
  self.assertEqual(r.exact_money('0'),'0')
 def test_invalid_numbers_rejected(self):
  for s in ('NaN','Infinity','1 2','(2','( -3 )','1%'):
   with self.subTest(value=s),self.assertRaises(ValueError):r.exact_money(s)
 def test_current_and_prior_are_independent_columns(self):
  c,_,_=r.fresh_candidates(source(),PERIODS)
  self.assertEqual(c['current']['net_income'][0]['value_decimal'],'101')
  self.assertEqual(c['comparison']['net_income'][0]['value_decimal'],'-51')
 def test_quarter_excluded(self):
  c,_,_=r.fresh_candidates(source(True),PERIODS)
  self.assertEqual(c['current']['net_income'],[])
 def test_attribution_not_substituted(self):
  c,_,_=r.fresh_candidates(source(label='Net income attributable to parent'),PERIODS)
  self.assertEqual(c['current']['net_income'],[])
 def test_duplicate_formal_tables_ambiguous(self):
  c,_,_=r.fresh_candidates(source()+source(),PERIODS)
  self.assertEqual(len(c['current']['operating_cash_flow']),2)
 def test_missing_cell_stays_gap(self):
  self.assertEqual(r.compare_cell(None,[])['status'],'producer_gap')
 def test_declared_value_mismatch_detected(self):
  c,_,_=r.fresh_candidates(source(),PERIODS);candidate=c['current']['net_income'][0]
  cell={k:candidate[k]for k in ('displayed_amount','value_decimal','scale','currency_literal','parsed_cell_sha256')}
  cell.update(fact_admitted=False,locator=dict(table_index_zero_based=0,row_index_zero_based=2,physical_cell_indices_zero_based=[1],header_literals=candidate['headers'],
   row_label=candidate['row_label'],statement_title=candidate['title'],logical_columns=candidate['logical_columns'],scale=candidate['scale']))
  self.assertEqual(r.compare_cell(cell,[candidate])['status'],'matched')
  cell['value_decimal']='102'
  self.assertIn('value_decimal_mismatch',r.compare_cell(cell,[candidate])['failures'])
 def test_cross_period_cell_rejected(self):
  raw=source().replace(b'<td>$ 101</td><td>$ (51)</td>',b'<td colspan="2">$ 101</td>')
  c,_,_=r.fresh_candidates(raw,PERIODS)
  self.assertEqual(c['current']['net_income'],[])
 def test_restatement_notes_not_cleared(self):
  _,_,notes=r.fresh_candidates(source()+b'<p>Prior periods have been recast.</p>',PERIODS)
  self.assertEqual(notes[0]['text'],'Prior periods have been recast.')

if __name__=='__main__':unittest.main(verbosity=2)
