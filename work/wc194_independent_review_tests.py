"""Synthetic negative tests for independent WC194 source review schema."""
import copy, unittest
import wc194_independent as a
import wc194_independent_review as r
from wc194_independent_tests import fixture

def pair(held=False,combined=False):
 raw,event=fixture('Accounts payable, accrued liabilities and other' if combined else 'Accounts payable')
 fields,reasons,_,_=a.html_components(raw,event)
 source={'path':'synthetic.html','sha256':a.b.sha(raw)}
 e=dict(event_id='S:FY2020:2021-02-01',symbol='S',fiscal_year=2020,release_date='2021-02-01',available_from='2021-02-02',inherited_admitted=not held,inherited_hold_reasons=['original_hold'] if held else [],extraction_verified=not combined,source=source,source_sha256=source['sha256'],fields=fields,components={},paired_cfo_verified=True,paired_assets_verified=True)
 for k,g in fields.items():
  if g:g['source']=source
  e['components'][k]=dict(verified=g is not None,signed_value_decimal=g['signed_value_decimal'] if g else None,scale=g['scale'] if g else None,annual_period_end=g['annual_period_end'] if g else None)
 p={k:copy.deepcopy(e[k]) for k in ('event_id','symbol','fiscal_year','release_date','available_from','inherited_admitted','source','extraction_verified')}
 p.update(status='held' if held or combined else 'verified',fact_admitted=False,strategy_admitted=False,inherited_hold_reasons=e['inherited_hold_reasons'][:],hold_reasons=e['inherited_hold_reasons']+reasons,fields={})
 for k,g in fields.items():
  if g is None:p['fields'][k]=None;continue
  loc=g['locator'];p['fields'][k]=dict(copy.deepcopy(g),available_from=e['available_from'],currency_literal=None)
  p['fields'][k]['locator']=dict(format='html',table_index_zero_based=loc['table'],row_index_zero_based=loc['row'],logical_columns=loc['logical_columns'],period_end=g['annual_period_end'],original_cfo_locator=loc['original_cfo_locator'])
  p['fields'][k]['statement_currency_evidence']=dict(format='html',table_index_zero_based=loc['table'],original_cfo_locator=loc['original_cfo_locator'])
 return e,p

class ComparisonTests(unittest.TestCase):
 def test_all_fields(self):
  e,p=pair();self.assertTrue(r.compare_records([e],[p])[0]['producer_comparison_passed'])
 def test_partial_components_retained(self):
  e,p=pair(combined=True);out=r.compare_records([e],[p])[0];self.assertTrue(out['components']['accounts_receivable']['verified']);self.assertFalse(out['components']['accounts_payable']['verified'])
 def test_inherited_hold_cannot_promote(self):
  e,p=pair(held=True);self.assertEqual(r.compare_records([e],[p])[0]['producer_status'],'held');p['status']='verified'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_hold_cannot_disappear(self):
  e,p=pair(held=True);p['hold_reasons']=[]
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_signed_literal_tamper(self):
  e,p=pair();p['fields']['accounts_receivable']['signed_literal']='7'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_amount_tamper(self):
  e,p=pair();p['fields']['inventories']['signed_value_decimal']='6'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_missing_not_zero(self):
  e,p=pair(combined=True);p['fields']['accounts_payable']={'signed_value_decimal':'0'}
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_combined_label_rejected(self):
  e,p=pair();p['fields']['accounts_payable']['row_label']='Accounts payable and accrued liabilities'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_current_annual_column(self):
  e,p=pair();p['fields']['accounts_payable']['locator']['logical_columns']=[1]
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_statement_currency_wrong_table(self):
  e,p=pair();p['fields']['accounts_payable']['statement_currency_evidence']['table_index_zero_based']=1
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_source_digest(self):
  e,p=pair();p['fields']['accounts_payable']['source']['sha256']='bad'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_period(self):
  e,p=pair();p['fields']['accounts_receivable']['period']['period_end']='2021-12-31'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_clock(self):
  e,p=pair();p['available_from']=p['release_date']
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_false_iso_certification(self):
  e,p=pair();p['fields']['accounts_receivable']['currency_iso']='USD'
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_paired_original_assets(self):
  e,p=pair();e['paired_assets_verified']=False
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_duplicate_and_missing(self):
  e,p=pair()
  for rows in ([],[p,p]):
   with self.assertRaises(ValueError):r.compare_records([e],rows)
 def test_unknown_component(self):
  e,p=pair();p['fields']['vendor_nontrade']={}
  with self.assertRaises(ValueError):r.compare_records([e],[p])
 def test_row_hash(self):
  e,p=pair();p['fields']['inventories']['row_text']='Inventories 1 5'
  with self.assertRaises(ValueError):r.compare_records([e],[p])

if __name__=='__main__':unittest.main(verbosity=2)
