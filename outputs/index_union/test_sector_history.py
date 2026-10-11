import unittest
import pandas as pd
from sector_history import asof_sectors,benchmark_returns

class SectorHistoryTests(unittest.TestCase):
    def row(self):
        return dict(security_id='A',sector='Technology',effective_from='2010-01-01',effective_to='2011-01-01',known_at='2010-01-05T23:59:59-05:00',verified=True,source_sha256='test')
    def test_late_publication_cannot_backfill_prior_signals(self):
        days=pd.bdate_range('2010-01-04',periods=4);r=asof_sectors([self.row()],days,['A'])
        self.assertTrue(r.A.iloc[:2].isna().all());self.assertEqual(r.A.iloc[2],'Technology')
    def test_missing_or_expired_classification_stays_unknown(self):
        days=pd.bdate_range('2011-01-03',periods=2);r=asof_sectors([self.row()],days,['A','B'])
        self.assertTrue(r.isna().all().all())
    def test_conflicting_evidence_rejected(self):
        a=self.row();b=dict(a,sector='Financial Services')
        with self.assertRaises(ValueError):asof_sectors([a,b],pd.bdate_range('2010-01-06',periods=1),['A'])
    def test_etf_not_yet_in_existence_is_not_backfilled(self):
        days=pd.bdate_range('2010-01-01',periods=80)
        sector=pd.DataFrame('Communication Services',index=days,columns=['A']);etf=pd.DataFrame({'XLC':[float('nan')]*70+[100.]*10},index=days)
        self.assertTrue(benchmark_returns(sector,etf,{'Communication Services':'XLC'}).isna().all().all())

if __name__=='__main__':unittest.main()
