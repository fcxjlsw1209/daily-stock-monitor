import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import pep_normalize as model

class PepsiCoEvidenceTests(unittest.TestCase):
    def test_contiguous_63_quarters_and_raw_evidence_integrity(self):
        rows=json.loads((P/'pep_supplement/events.json').read_text())
        extra=json.loads((P.parent/'integrated_batches/financial_consumer_628/events.json').read_text())
        rows += [x for x in extra if x['ticker']=='PEP']
        self.assertEqual(len(rows),64)
        self.assertEqual({(x['fiscal_year'],x['fiscal_quarter']) for x in rows},
            {(y,q) for y in range(2010,2027) for q in range(1,5) if (2010,3)<=(y,q)<=(2026,2)})
        for x in rows:
            p=Path(x['evidence_path']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),x.get('sha256') or x['evidence_sha256'])
            self.assertIsNone(x['publication_timestamp']);self.assertLess(x['fiscal_period_end_actual'],x['release_date'])

    def test_dates_are_actual_fiscal_ends_not_quarter_end_assumptions(self):
        rows={(x['fiscal_year'],x['fiscal_quarter']):x for x in json.loads((P/'pep_supplement/events.json').read_text())}
        self.assertEqual(rows[2010,4]['fiscal_period_end_actual'],'2010-12-25')
        self.assertEqual(rows[2013,4]['fiscal_period_end_actual'],'2013-12-28')
        self.assertEqual(rows[2020,1]['release_date'],'2020-04-28')
        self.assertNotEqual(rows[2020,1]['release_date'],'2020-04-21')

    def test_preannouncement_title_rejected_even_with_original_body(self):
        row=copy.deepcopy(json.loads((P/'pep_supplement/MANIFEST.json').read_text())[0])
        row['title']='PepsiCo Announces Timing of Second Quarter 2019 Results'
        with self.assertRaises(ValueError):model.parse_original(row)

    def test_quarter_2012_statement_ignores_prior_audited_year_in_parentheses(self):
        rows=json.loads((P/'pep_supplement/MANIFEST.json').read_text())
        row=next(x for x in rows if 'Fourth Quarter and Full Year 2012' in x['title'])
        self.assertEqual(model.parse_original(row)['fiscal_period_end_actual'],'2012-12-29')

    def test_date_only_index_is_labeled_not_original_pdf(self):
        for row in json.loads((P/'pep_supplement/events.json').read_text()):
            if 'index' in row['source_kind']:
                self.assertIn('original PDF bytes unavailable',row['evidence_limitation'])
                self.assertFalse(row['first_public_time_verified'])
if __name__=='__main__':unittest.main()
