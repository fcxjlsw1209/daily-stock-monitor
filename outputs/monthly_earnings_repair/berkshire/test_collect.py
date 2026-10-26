"""Evidence/timing regression checks using cached Berkshire observations only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent))
spec = importlib.util.spec_from_file_location('berkshire_collector_test', P/'collect.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
from event_schema import normalize_event


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name)/'observed.json'
        self.file.write_text('synthetic test evidence')
        self.row = dict(url='https://www.berkshirehathaway.com/news/test.pdf',
            source_kind='public_index_observation_of_original_issuer_release_not_full_pdf',
            opening_excerpt='BERKSHIRE HATHAWAY INC. NEWS RELEASE FOR IMMEDIATE RELEASE May 1, 2021 Omaha, NE (BRK.A; BRK.B) First quarters of 2021 and 2020.',
            period_excerpt='On March 31, 2021 there were Class A equivalent shares outstanding.')

    def test_original_date_is_not_a_clock_time(self):
        event = collector.parse(self.row, self.file)
        self.assertEqual(event['release_date'], '2021-05-01')
        self.assertEqual(event['fiscal_period_end_actual'], '2021-03-31')
        self.assertIsNone(event['publication_timestamp'])
        self.assertIn('not directly downloaded', event['evidence_limitation'])

    def test_scheduled_and_preliminary_are_not_regular_events(self):
        for word in ('will be', 'preliminary', 'information regarding'):
            row = dict(self.row, opening_excerpt=self.row['opening_excerpt']+' '+word)
            with self.assertRaisesRegex(ValueError, 'Scheduled/preliminary'):
                collector.parse(row, self.file)

    def test_issuer_dateline_quarter_and_period_are_required(self):
        for field, value in [('opening_excerpt', 'Other Corp May 1, 2021'),
                             ('period_excerpt', 'March 31, 2020')]:
            row = dict(self.row, **{field: value})
            with self.assertRaises(ValueError):
                collector.parse(row, self.file)
        for old, new in [('Omaha, NE', 'Unknown place'),
                         ('First quarters of 2021 and 2020.', ''),
                         ('FOR IMMEDIATE RELEASE', 'Conference call')]:
            row = dict(self.row, opening_excerpt=self.row['opening_excerpt'].replace(old, new))
            with self.assertRaises(ValueError):
                collector.parse(row, self.file)

    def test_separate_10q_proves_period_not_release_date(self):
        row = dict(self.row, period_excerpt='', period_confirmation=dict(
            release_url=self.row['url'], url='https://www.berkshirehathaway.com/qtrly/test.pdf',
            opening_excerpt='FORM 10-Q For the quarterly period ended March 31, 2021 BERKSHIRE HATHAWAY INC.'))
        result = collector.parse(row, self.file)
        self.assertEqual(result['release_date'], '2021-05-01')
        self.assertIn('Filing date is not used', result['period_basis'])
        for old, new in [('2021', '2020'), ('BERKSHIRE HATHAWAY INC.', 'OTHER INC.')]:
            wrong = copy.deepcopy(row)
            wrong['period_confirmation']['opening_excerpt'] = wrong['period_confirmation']['opening_excerpt'].replace(old, new)
            with self.assertRaises(ValueError):
                collector.parse(wrong, self.file)

    def test_corrected_archive_is_labeled(self):
        row = dict(self.row, opening_excerpt=self.row['opening_excerpt'].replace('NEWS RELEASE', 'CORRECTED NEWS RELEASE'))
        self.assertEqual(collector.parse(row, self.file)['publication_version'], 'corrected_current_archive')

    def test_all_cached_events_parse_and_hash_and_weekends(self):
        events = json.loads((P/'events.json').read_text())
        self.assertEqual(len(events), 61)
        for event in events:
            path = Path(event['evidence_path'])
            self.assertEqual(collector.sha(path), event['evidence_sha256'])
            regenerated = collector.parse(event['evidence_excerpt']['original_excerpt'], path)
            self.assertEqual(regenerated, event)
            normalized = normalize_event(dict(symbol='BRK.B', official=dict(
                actual_period_end=event['fiscal_period_end_actual'], publication_date=event['release_date'],
                publication_timezone=event['publication_timezone'], source_url=event['source_url'],
                verified=True, publication_kind='release')))
            self.assertEqual(normalized['status'], 'ready')
            self.assertGreater(normalized['available_session'], event['release_date'])
            if event['release_date'] == '2021-05-01':
                self.assertEqual(normalized['available_session'], '2021-05-03')


if __name__ == '__main__':
    unittest.main()
