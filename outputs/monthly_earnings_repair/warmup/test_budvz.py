"""Source and publication semantics for the BUD/VZ continuation, offline only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import bud_normalize as bud
import vz_normalize as vz

class EvidenceTests(unittest.TestCase):
    def test_all_evidence_files_match(self):
        for name in ['bud','vz','cvx']:
            for e in json.loads((P/name/'events.json').read_text()):
                self.assertEqual(hashlib.sha256(Path(e['evidence_path']).read_bytes()).hexdigest(),e['evidence_sha256'])
                for a in e.get('additional_evidence',[]):
                    self.assertEqual(hashlib.sha256(Path(a['evidence_path']).read_bytes()).hexdigest(),a['sha256'])

    def test_verified_quarters_are_contiguous_and_unique(self):
        for name,expected in [('bud',36),('vz',63),('cvx',63)]:
            e=json.loads((P/name/'events.json').read_text())
            q=sorted(x['fiscal_year']*4+x['fiscal_quarter']-1 for x in e)
            self.assertEqual(len(q),expected)
            self.assertEqual(q,list(range(q[0],q[-1]+1)))

    def test_original_news_date_not_archival_verification_date(self):
        e=bud.early_q1()
        self.assertEqual(e['release_date'],'2011-05-04')
        self.assertEqual(e['fiscal_period_end_actual'],'2011-03-31')
        self.assertIn('no later financial values',e['period_basis'])
        self.assertIsNone(e['publication_timestamp'])

    def test_index_observation_is_not_claimed_original_bytes(self):
        e=next(e for e in json.loads((P/'vz/events.json').read_text()) if e['release_date']=='2017-01-24')
        self.assertEqual(e['source_kind'],'public_index_observation_not_original_bytes')
        self.assertIn('not original HTML',e['evidence_limitation'])
        self.assertEqual(e['fiscal_period_end_actual'],'2016-12-31')

    def test_call_timestamp_and_cms_update_do_not_supply_release_date(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'news.html'
            p.write_text('<h1>Verizon reports first quarter results</h1><meta name="dateModified" content="2020-04-24">Webcast April 24, 2020 at 8:30 AM; quarter ended March 31, 2020')
            link=dict(path=str(p),url='https://example.com/news/quarter',meta=dict(http_status=200,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            self.assertIsNone(vz.publication(link))

    def test_pdf_filename_does_not_override_original_body(self):
        a=json.loads((P/'vz/SUPPLEMENTAL_LINKS.json').read_text())[0]
        self.assertIn('10-21-24',a['url'])
        self.assertEqual(str(vz.publication(a)[0]),'2024-10-22')

    def test_fiscal_period_is_not_imputed_from_quarter_name(self):
        with self.assertRaises(ValueError):
            bud.parse('Brussels, 4 May 2011 Anheuser-Busch InBev reports First Quarter 2011 Results',Path('fixture'),{})

if __name__=='__main__':unittest.main()
