import unittest
import pandas as pd
from normalize_expanded_evidence import normalize
class DisclosureTests(unittest.TestCase):
 def row(self,**kw):
  x=dict(cik='123',taxonomy='dei',concept='EntityCommonStockSharesOutstanding',end='2020-01-01',filed='2020-02-01',val=100,accn='a',source_file='one');x.update(kw);return x
 def test_duplicate_sources_preserve_one_event(self):
  d=normalize(pd.DataFrame([self.row(),self.row(source_file='two')]))
  self.assertEqual(len(d),1);self.assertEqual(d.iloc[0].source_file,'one;two')
 def test_cross_source_conflict_is_not_resolved_arbitrarily(self):
  d=normalize(pd.DataFrame([self.row(),self.row(val=200,source_file='two')]))
  self.assertTrue(d.iloc[0].conflict);self.assertTrue(pd.isna(d.iloc[0].shares))
 def test_later_filing_is_separate_revision(self):
  d=normalize(pd.DataFrame([self.row(),self.row(val=200,filed='2020-03-01')]))
  self.assertEqual(len(d),2);self.assertFalse(d.usable_for_rank.any())
 def test_future_period_and_weighted_shares_are_excluded(self):
  d=normalize(pd.DataFrame([self.row(),self.row(end='2021-01-01'),self.row(concept='WeightedAverageNumberOfSharesOutstandingBasic')]))
  self.assertEqual(len(d),1);self.assertEqual(d.iloc[0].available_no_earlier_than,'2020-02-02T05:00:00+00:00')
