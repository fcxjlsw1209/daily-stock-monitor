"""Source-only receipt-schema tests: synthetic metadata, no financial values."""
import unittest
from copy import deepcopy
import capex193_assemble as a
class Receipts(unittest.TestCase):
 def setUp(self):
  ids=['AAPL:'+str(i) for i in range(10)]
  self.original={'events':[{'symbol':'AAPL','event_id':i} for i in ids]}
  self.visual={'passed':True,'report_count':10,'visual_review_pending':False,'extraction_sha256':a.PARTS['outputs/monthly_capex193_source/msft_aapl/EXTRACTION.json'],'records':[]}
  self.review={'passed':True,'source_only':True,'producer_refs':deepcopy(a.PARTS),'scope_sha256':a.SCOPE_SHA,'calendar_sha256':a.CALENDAR_SHA,'runtime':{'manifest_sha256':a.RUNTIME_SHA,'historical_runtime_currently_passes':False},'independent_visual_reviews':[]}
  for k in ('financial_ratios','factors','returns','strategy_admissions','fact_admissions'):self.visual[k]=0
  for k in ('ratios_computed','factors_computed','returns_computed'):self.review[k]=0
  for i in ids:
   self.visual['records'].append(dict(event_id=i,source={'sha256':'a'*64},render={'path':i+'.png','sha256':'b'*64},pure_ppe_row_distinct_from_acquisitions_and_intangibles=True,legible_unclipped_annual_columns=True,extracted_value_matches_visual=True,producer_visual_review_completed=True))
   self.review['independent_visual_reviews'].append(dict(event_id=i,source_sha256='a'*64,render_path=i+'.png',render_sha256='b'*64,current_annual_column_first=True,units_millions=True,pure_cash_ppe_separate_from_intangibles=True,cfo_matches_source=True,independent_visual_review=True))
 def check(self):a.validate_reviews(self.review,self.visual,self.original)
 def test_actual_receipt_schema(self):self.check()
 def test_missing_visual(self):
  self.visual['records'].pop()
  with self.assertRaises(ValueError):self.check()
 def test_source_review_failure(self):
  self.review['passed']=False
  with self.assertRaises(ValueError):self.check()
 def test_different_producer(self):
  self.review['producer_refs'].popitem()
  with self.assertRaises(ValueError):self.check()
 def test_wrong_runtime(self):
  self.review['runtime']['manifest_sha256']='bad'
  with self.assertRaises(ValueError):self.check()
 def test_historical_failure_not_pass(self):
  self.review['runtime']['historical_runtime_currently_passes']=True
  with self.assertRaises(ValueError):self.check()
 def test_returns_prohibited(self):
  self.review['returns_computed']=1
  with self.assertRaises(ValueError):self.check()
 def test_render_changed(self):
  self.review['independent_visual_reviews'][0]['render_sha256']='bad'
  with self.assertRaises(ValueError):self.check()
 def test_not_independent(self):
  self.review['independent_visual_reviews'][0]['independent_visual_review']=False
  with self.assertRaises(ValueError):self.check()
if __name__=='__main__':unittest.main(verbosity=2)
