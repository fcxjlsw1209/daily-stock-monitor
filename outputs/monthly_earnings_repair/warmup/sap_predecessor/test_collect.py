import unittest,copy
from collect import *

class SAPPredecessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=json.loads((HERE/'events.json').read_text())[0]
        cls.pages=json.loads(PDF.with_suffix('.pages.json').read_text())
        cls.publication=json.loads(PUBLICATION.read_text())
    def test_original_report_bytes_and_extraction_hashes(self):
        self.assertTrue(PDF.read_bytes().startswith(b'%PDF'))
        self.assertEqual(sha(PDF.read_bytes()),self.e['evidence_sha256'])
        m=json.loads(PDF.with_suffix('.text.meta.json').read_text())
        self.assertEqual(m['source_sha256'],self.e['evidence_sha256'])
        self.assertEqual(m['extracted_pages_sha256'],sha(PDF.with_suffix('.pages.json').read_bytes()))
        self.assertEqual(len(self.pages),55)
    def test_all_event_proof_hashes(self):
        for p in self.e['release_date_evidence']+self.e['additional_evidence']+[self.e['fiscal_identity_evidence'],self.e['fiscal_period_evidence']]:
            self.assertEqual(sha((ROOT/p['evidence_path']).read_bytes()),p['sha256'])
    def test_observed_release_and_actual_quarter(self):
        r=parse(self.pages,self.publication)
        self.assertEqual((r['release_date'],r['fiscal_year'],r['fiscal_quarter'],r['fiscal_period_end_actual']),('2012-10-24',2012,3,'2012-09-30'))
    def test_imprint_alone_cannot_supply_release(self):
        p=copy.deepcopy(self.publication);p.pop('dateline')
        with self.assertRaisesRegex(ValueError,'Independent issuer-issued'):parse(self.pages,p)
    def test_future_notice_is_not_completed_results(self):
        p=copy.deepcopy(self.publication);p['headline']='SAP will announce third quarter results'
        with self.assertRaisesRegex(ValueError,'completed-results'):parse(self.pages,p)
    def test_no_calendar_period_imputation(self):
        p=copy.deepcopy(self.pages);p[24]['text']=p[24]['text'].replace('September 30','DATE OMITTED')
        with self.assertRaisesRegex(ValueError,'period table absent'):parse(p,self.publication)
    def test_wrong_current_year_rejected(self):
        p=copy.deepcopy(self.pages);p[24]['text']=p[24]['text'].replace('Note 2012 2011','Note 2011 2010')
        with self.assertRaisesRegex(ValueError,'year conflict'):parse(p,self.publication)
    def test_source_classes_are_transparent(self):
        self.assertIn('third_party_mirror',self.e['source_class'])
        self.assertFalse(self.e['download_host_is_official'])
        self.assertIn('not_original_bytes',self.e['fiscal_identity_evidence']['source_class'])
        self.assertFalse(self.publication['original_html_downloaded'])
        self.assertIn('third-party PDF mirror',self.e['issuer_identity_note'])
        self.assertIn('report imprint only corroborates',self.e['evidence_limitation'])
    def test_day_only_schema_and_same_day_exclusion(self):
        n=normalized(self.e)
        self.assertEqual(n['status'],'ready');self.assertEqual(n['available_session'],'2012-10-25')
        self.assertIsNone(self.e['publication_timestamp']);self.assertFalse(n['known_by_is_actual_publication'])
        self.assertFalse(event_available(n,'2012-10-24T20:00:00+00:00'))
        self.assertTrue(event_available(n,'2012-12-31T21:00:00+00:00'))
    def test_actual_training_candidate_exists(self):
        meta=json.loads((ROOT/'outputs/monthly_architecture_restart/META.json').read_text())
        self.assertTrue(any(d=='2012-12-31' and 'SAP' in syms for d,syms in zip(meta['training']['dates'],meta['training']['symbols'])))
        self.assertTrue(self.e['training_predecessor_only'])
    def test_failed_sources_preserved_and_stopped(self):
        stops=json.loads((HERE/'SOURCE_STOPS.json').read_text())
        self.assertIn('www.sap.com',stops);self.assertEqual(stops['www.pressebox.de']['status'],403)
        inventory=json.loads((HERE/'SOURCE_INVENTORY.json').read_text())
        self.assertTrue(any(x['status']==403 for x in inventory))

if __name__=='__main__':unittest.main()
