import unittest
from copy import deepcopy
import wc194_assemble as a
class Validate(unittest.TestCase):
    def setUp(self):
        self.o={'events':[dict(event_id='AAPL:'+str(i),symbol='AAPL') for i in range(10)]}
        self.v=dict(passed=True,visual_review_pending=False,report_count=10,current_cells_reviewed=30,extraction_path=next(iter(a.PARTS)),extraction_sha256=next(iter(a.PARTS.values())),records=[],component_sums=0,financial_ratios=0,factors=0,orders=0,returns=0,fact_admissions=0,strategy_admissions=0)
        for e in self.o['events']:
            self.v['records'].append(dict(e,source={'sha256':'a'*64},pure_trade_ar_separate_from_vendor_nontrade=True,inventories_and_ap_distinct_from_other_assets_and_liabilities=True,all_three_rows_within_operating_cashflow_changes=True,legible_unclipped_annual_columns=True,producer_visual_review_completed=True))
        self.r=dict(passed=True,source_only=True,scope_sha256=a.SCOPE_SHA,concept_sha256=a.CONCEPT_SHA,producer_refs=a.PARTS,producer_visual_refs={a.VISUAL:a.VISUAL_SHA},runtime=dict(manifest_sha256=a.RUNTIME_SHA,historical_runtime_currently_passes=False),component_sum_computed=False,ratios_computed=False,factors_computed=False,orders_computed=False,returns_computed=False,independent_visual_reviews=[dict(e,visual_source='independent_generated_PDF_page_directly_viewed',units_millions=True,three_distinct_operating_reconciliation_rows=True,signs_and_values_match_independent_extraction=True,current_annual_column=0,source_sha256='a'*64) for e in self.o['events']],events=[dict(e,review_passed=True,producer_comparison_passed=True) for e in self.o['events']])
    def run_check(self):a.validate_reviews(self.r,self.v,self.o)
    def test_complete(self):self.run_check()
    def test_missing_independent_visual(self):
        self.r['independent_visual_reviews'].pop()
        with self.assertRaises(ValueError):self.run_check()
    def test_pending_producer_visual(self):
        self.v['visual_review_pending']=True
        with self.assertRaises(ValueError):self.run_check()
    def test_changed_producer(self):
        self.r['producer_refs']={}
        with self.assertRaises(ValueError):self.run_check()
    def test_wrong_source(self):
        self.r['independent_visual_reviews'][0]['source_sha256']='b'*64
        with self.assertRaises(ValueError):self.run_check()
    def test_incomplete_comparison(self):
        self.r['events'][0]['producer_comparison_passed']=False
        with self.assertRaises(ValueError):self.run_check()
    def test_source_only(self):
        self.r['component_sum_computed']=True
        with self.assertRaises(ValueError):self.run_check()
    def test_no_retroactive_old_runtime_pass(self):
        self.r['runtime']['historical_runtime_currently_passes']=True
        with self.assertRaises(ValueError):self.run_check()
if __name__=='__main__':unittest.main()
