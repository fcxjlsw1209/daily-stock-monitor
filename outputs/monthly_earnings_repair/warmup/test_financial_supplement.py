"""Offline facts/identity regression checks for the independently added batch."""
import hashlib,json,sys,unittest
from pathlib import Path
from unittest import mock
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import financial_normalize as f

class FinancialSupplementTests(unittest.TestCase):
    def test_no_unresolved_originals_and_no_duplicate_quarters(self):
        self.assertEqual(json.loads((f.OUT/'PENDING.json').read_text()),[])
        rows=json.loads((f.OUT/'events.json').read_text())+json.loads((P.parent/'integrated_batches/financial_consumer_687/events.json').read_text())
        keys=[(r['ticker'],r['fiscal_year'],r['fiscal_quarter']) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))

    def test_mcdonalds_63_consecutive_quarters(self):
        rows=json.loads((f.OUT/'events.json').read_text())+json.loads((P.parent/'integrated_batches/financial_consumer_687/events.json').read_text())
        quarters={r['fiscal_year']*4+r['fiscal_quarter']-1 for r in rows if r['ticker']=='MCD'}
        self.assertEqual(quarters,set(range(2010*4+3,2026*4+2)))

    def test_date_only_index_observation_is_explicit(self):
        for r in json.loads((f.OUT/'events.json').read_text()):
            self.assertIsNone(r['publication_timestamp'])
            self.assertEqual(r['precision'],'date_only')
            self.assertEqual(hashlib.sha256(Path(r['evidence_path']).read_bytes()).hexdigest(),r['sha256'])
            if r['source_kind'].startswith('public_index'):
                self.assertFalse(r['original_response_available'])
                self.assertTrue(r['publication_evidence']);self.assertTrue(r['period_evidence'])

    def test_tmo_actual_week_ends_not_calendar_end(self):
        rows={r['fiscal_quarter']:r for r in json.loads((f.OUT/'events.json').read_text()) if r['ticker']=='TMO'}
        self.assertEqual(rows[1]['fiscal_period_end_actual'],'2022-04-02')
        self.assertEqual(rows[2]['fiscal_period_end_actual'],'2022-07-02')
        self.assertEqual(rows[3]['fiscal_period_end_actual'],'2022-10-01')

    def test_mastercard_original_exhibit_not_8k_filing_date(self):
        rows=json.loads((f.OUT/'MA_MANIFEST.json').read_text())
        r=next(x for x in rows if x['year']==2020)
        e=f.parse(r)
        self.assertEqual(e['release_date'],'2021-01-28')
        self.assertEqual(e['fiscal_period_end_actual'],'2020-12-31')
        with self.assertRaisesRegex(ValueError,'checksum'):
            f.parse(dict(r,meta=dict(r['meta'],sha256='0'*64)))

    def test_comparison_period_does_not_override_current_year(self):
        r=next(x for x in json.loads((f.OUT/'MA_MANIFEST.json').read_text()) if x['year']==2013 and x['quarter']==4)
        self.assertEqual(f.parse(r)['fiscal_period_end_actual'],'2013-12-31')
        with self.assertRaisesRegex(ValueError,'identity'):
            f.parse(dict(r,year=2014))

if __name__=='__main__':unittest.main()
