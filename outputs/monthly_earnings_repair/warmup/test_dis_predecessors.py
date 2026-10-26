import hashlib,json,sys,unittest,copy
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import dis_normalize as dis
class DisneyAndPredecessorTests(unittest.TestCase):
    def test_full_fiscal_sequence_without_notice_dates(self):
        rs=json.loads((P/'dis_supplement/events.json').read_text())
        rs += [r for r in json.loads((P.parent/'integrated_batches/financial_consumer_628/events.json').read_text()) if r['ticker']=='DIS']
        self.assertEqual(len(rs),63)
        self.assertEqual({(r['fiscal_year'],r['fiscal_quarter']) for r in rs},{(y,q) for y in range(2011,2027) for q in range(1,5) if (y,q)<=(2026,3)})
        for r in rs:
            self.assertEqual(hashlib.sha256(Path(r['evidence_path']).read_bytes()).hexdigest(),r.get('sha256') or r['evidence_sha256'])
            self.assertIsNone(r['publication_timestamp'])
            self.assertLess(r['fiscal_period_end_actual'],r['release_date'])
    def test_image_only_first_pages_use_actual_date_and_period(self):
        rs=json.loads((P/'dis_supplement/MANIFEST.json').read_text())
        for y,q,release,end in [(2018,4,'2018-11-08','2018-09-29'),(2022,2,'2022-05-11','2022-04-02')]:
            r=dis.parse(next(x for x in rs if (x['year'],x['quarter'])==(y,q)))
            self.assertEqual(r['release_date'],release);self.assertEqual(r['fiscal_period_end_actual'],end)
            self.assertIsNotNone(r['visual_first_page_evidence'])
    def test_original_shareholder_letter_not_comparison_period(self):
        rs=json.loads((P/'dis_supplement/MANIFEST.json').read_text())
        for q,d in [(2,'2026-03-28'),(3,'2026-06-27')]:
            r=dis.parse(next(x for x in rs if (x['year'],x['quarter'])==(2026,q)))
            self.assertEqual(r['fiscal_period_end_actual'],d)
    def test_checksum_failure_is_not_indexed_date_override(self):
        r=copy.deepcopy(next(x for x in json.loads((P/'dis_supplement/MANIFEST.json').read_text()) if (x['year'],x['quarter'])==(2024,4)))
        r['meta']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum'):dis.parse(r)
    def test_predecessors_keep_fiscal_identity(self):
        rs={r['ticker']:r for r in json.loads((P/'predecessors/events.json').read_text())}
        self.assertEqual(rs['WMT']['fiscal_year'],2011);self.assertEqual(rs['WMT']['fiscal_quarter'],3)
        self.assertEqual(rs['WMT']['fiscal_period_end_actual'],'2010-10-31')
        self.assertEqual(rs['BRK.B']['fiscal_period_end_actual'],'2010-09-30')
        for r in rs.values():
            self.assertTrue(r['training_predecessor_only'])
            self.assertIsNone(r['publication_timestamp'])
            self.assertEqual(hashlib.sha256(Path(r['evidence_path']).read_bytes()).hexdigest(),r['sha256'])
if __name__=='__main__':unittest.main()
