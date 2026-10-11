import json
import hashlib
import unittest
from pathlib import Path
from extract_sp_sector_observations import extract_legacy, REVIEWED_NAMES

P=Path(__file__).resolve().parent

class SpSectorObservationTests(unittest.TestCase):
    def test_mixed_index_table_does_not_leak_other_indices(self):
        rows=extract_legacy((P/'sources/sp_releases/dc288f36c1bc08a2.html').read_text())
        self.assertEqual({(r['company'],r['action']) for r in rows},{('Dow','addition'),('Halliburton','deletion')})
    def test_action_does_not_carry_across_headers(self):
        html='<table><tr><td>S&P 100 INDEX GICS</td></tr><tr><td>ADDED</td><td>A</td><td>Energy</td></tr><tr><td>S&P 500 INDEX</td></tr><tr><td>DELETED</td><td>B</td><td>Energy</td></tr><tr><td>S&P 100 INDEX</td></tr><tr><td>C</td><td>Energy</td></tr></table>'
        self.assertEqual([r['company'] for r in extract_legacy(html)],['A'])
    def test_reused_ticker_review_is_source_scoped(self):
        self.assertEqual(REVIEWED_NAMES['6da13df59135e564.html']['Actavis'],['ACT'])
        self.assertNotIn('ACT',REVIEWED_NAMES)
    def test_two_fox_classes_have_same_sector_not_new_fox(self):
        x=json.loads((P/'SP_HISTORICAL_SECTOR_OBSERVATIONS.json').read_text())
        rows=[r for r in x['observations'] if r['company']=='Twenty-First Century Fox']
        self.assertEqual({r['symbol'] for r in rows},{'TFCF','TFCFA'})
        self.assertEqual({r['sector'] for r in rows},{'Communication Services'})
    def test_each_observation_retains_exact_event_time_and_source(self):
        events=[]
        for f in ['sp100_modern_event_rows.json','sp100_legacy_reviewed_rows.json','sp100_supplemental_reviewed_rows.json']:
            events+=json.loads((P/f).read_text())
        x=json.loads((P/'SP_HISTORICAL_SECTOR_OBSERVATIONS.json').read_text())
        self.assertFalse(x['complete_sector_history']); self.assertEqual(x['unmapped'],[])
        for row in x['observations']:
            self.assertEqual(hashlib.sha256((P/row['source_file']).read_bytes()).hexdigest(),row['source_sha256'])
            candidates=[e for e in events if e['source_file']==row['source_file'] and e.get('ticker',e.get('ticker_as_announced_or_effective'))==row['symbol'] and e['action']==row['action']]
            self.assertTrue(any(e['effective_session']==row['effective_session'] and e['published_at']==row['known_at'] for e in candidates))
            self.assertFalse(row['permanent_identity_verified'])

if __name__=='__main__':unittest.main()
