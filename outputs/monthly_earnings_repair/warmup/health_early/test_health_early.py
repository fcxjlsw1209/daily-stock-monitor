import hashlib
import json
from pathlib import Path
import sys
import unittest

P = Path(__file__).resolve().parent
REPAIR = P.parents[1]
sys.path.insert(0, str(REPAIR))
from event_schema import normalize_event
from research_calendar import market_calendar


class EarlyHealthEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((P / 'events.json').read_text())

    def test_original_and_observed_evidence_are_distinct_and_hash_validated(self):
        self.assertEqual(len(self.rows), 13)
        self.assertEqual(sum(r['original_response_available'] for r in self.rows), 6)
        for row in self.rows:
            self.assertEqual(hashlib.sha256(Path(row['evidence_path']).read_bytes()).hexdigest(), row['sha256'])
            self.assertIsNone(row['publication_timestamp'])
            for extra in row.get('additional_evidence', []):
                self.assertEqual(hashlib.sha256(Path(extra['evidence_path']).read_bytes()).hexdigest(), extra['sha256'])
            if not row['original_response_available']:
                self.assertIn('not original response bytes', row['limitation'])

    def test_merck_actual_fiscal_ends_and_oracle_noncalendar_year(self):
        lookup = {(r['ticker'], r['fiscal_year'], r['fiscal_quarter']): r for r in self.rows}
        self.assertEqual(lookup['MRK', 2011, 1]['fiscal_period_end_actual'], '2011-03-31')
        self.assertTrue(lookup['MRK', 2011, 1]['additional_evidence'])
        self.assertEqual(lookup['ORCL', 2011, 2]['fiscal_period_end_actual'], '2010-11-30')
        self.assertEqual(lookup['ORCL', 2016, 3]['fiscal_period_end_actual'], '2016-02-29')
        self.assertEqual(lookup['ORCL', 2015, 1]['release_date'], '2014-09-18')

    def test_no_collision_with_frozen_base(self):
        old = json.loads((REPAIR / 'integrated_batches/health_energy_foreign_900/events.json').read_text())
        key = lambda r: (r['ticker'], r['fiscal_year'], r['fiscal_quarter'])
        self.assertEqual(len({key(r) for r in self.rows}), len(self.rows))
        self.assertFalse({key(r) for r in old} & {key(r) for r in self.rows})

    def test_date_only_release_never_enters_same_day_signal(self):
        cal = market_calendar(start='2010-01-01', end='2030-12-31')
        for row in self.rows:
            event = normalize_event(dict(symbol=row['ticker'], asset_type='equity', official=dict(
                actual_period_end=row['fiscal_period_end_actual'], publication_date=row['release_date'],
                publication_at=None, publication_timezone=row['publication_timezone'],
                publication_kind='release', source_url=row['source_url'], verified=True)), cal)
            self.assertGreater(event['known_by_at'][:10], row['release_date'])


if __name__ == '__main__':
    unittest.main()
