import unittest
import pandas as pd
from reconcile_share_evidence import reconcile

class ShareCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(cik='0001137411', taxonomy='dei', concept='EntityCommonStockSharesOutstanding', reported_as_of='2010-01-18', filed='2010-01-28', accessions='0001144204-10-004114', shares=157218806000000, available_no_earlier_than='2010-01-29T05:00:00+00:00', usable_for_rank=False)
        self.fix = {k:self.row[k] for k in ('cik','taxonomy','concept','reported_as_of','filed')}
        self.fix.update(accession=self.row['accessions'], observed_shares=self.row['shares'], corrected_shares=157218806, source='https://www.sec.gov/Archives/edgar/data/1137411/000114420410004114/v172298_10q.htm', verification='original_filing_cover')
    def test_exact_correction_preserves_provenance_and_availability(self):
        source=pd.DataFrame([self.row]); result=reconcile(source,[self.fix])
        self.assertEqual(result.iloc[0].shares,157218806)
        self.assertEqual(result.iloc[0].original_shares,self.row['shares'])
        self.assertEqual(source.iloc[0].shares,self.row['shares'])
        self.assertEqual(result.iloc[0].available_no_earlier_than,self.row['available_no_earlier_than'])
        self.assertFalse(result.iloc[0].usable_for_rank)
    def test_cover_corrects_date_without_backdating_availability(self):
        result=reconcile(pd.DataFrame([self.row]),[{**self.fix,'corrected_reported_as_of':'2010-01-16'}])
        self.assertEqual(result.iloc[0].reported_as_of,'2010-01-16')
        self.assertEqual(result.iloc[0].original_reported_as_of,'2010-01-18')
        self.assertEqual(result.iloc[0].available_no_earlier_than,self.row['available_no_earlier_than'])
        self.assertFalse(result.iloc[0].usable_for_rank)
    def test_future_context_rejected(self):
        with self.assertRaises(ValueError):
            reconcile(pd.DataFrame([self.row]),[{**self.fix,'corrected_reported_as_of':'2011-01-01'}])
    def test_no_guess_for_unverified_scale(self):
        result=reconcile(pd.DataFrame([self.row]),[])
        self.assertTrue(result.iloc[0].scale_review_required)
        self.assertEqual(result.iloc[0].shares,self.row['shares'])
    def test_different_accession_never_matches(self):
        row={**self.row,'accessions':self.row['accessions']+'0'}
        self.assertEqual(reconcile(pd.DataFrame([row]),[self.fix]).iloc[0].shares,self.row['shares'])
    def test_changed_source_value_rejects(self):
        with self.assertRaises(ValueError):reconcile(pd.DataFrame([{**self.row,'shares':123}]),[self.fix])
    def test_missing_evidence_rejects(self):
        with self.assertRaises(ValueError):reconcile(pd.DataFrame([self.row]),[{**self.fix,'verification':''}])
    def test_invalid_replacement_rejects(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):reconcile(pd.DataFrame([self.row]),[{**self.fix,'corrected_shares':value}])

if __name__=='__main__':unittest.main()
