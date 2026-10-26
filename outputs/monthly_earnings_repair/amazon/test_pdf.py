import unittest
from pathlib import Path
from collect_pdf import parse, legacy_events


class OriginalReleaseTests(unittest.TestCase):
    def fixture(self, date='February 5, 2026', ended='December 31, 2025'):
        return ('AMAZON.COM ANNOUNCES FOURTH QUARTER RESULTS\n'
                'SEATTLE—(BUSINESS WIRE) '+date+'—Amazon.com, Inc. (NASDAQ: AMZN) '
                'today announced financial results for its fourth quarter ended '+ended+'.')

    def parse(self, text, expected=(2025,4)):
        return parse(text,Path('fixture.pdf'),dict(url='https://example.test/official',sha256='abc',retrieved_at='2026-10-12'),expected)

    def test_content_supplies_date_not_preannouncement_metadata(self):
        row=self.parse('datePublished=2026-01-22\n'+self.fixture())
        self.assertEqual(row['release_date'],'2026-02-05')
        self.assertEqual(row['fiscal_period_end_actual'],'2025-12-31')
        self.assertIsNone(row['publication_timestamp'])

    def test_conflicting_fiscal_expected_rejected(self):
        with self.assertRaisesRegex(ValueError,'Fiscal identity'):self.parse(self.fixture(),(2024,4))

    def test_missing_dateline_rejected(self):
        with self.assertRaisesRegex(ValueError,'dateline'):self.parse(self.fixture().replace('SEATTLE','PORTLAND'))

    def test_conference_call_cannot_supply_release_date(self):
        with self.assertRaises(ValueError):self.parse('AMAZON.COM ANNOUNCES FOURTH QUARTER RESULTS Conference call February 5, 2026 quarter ended December 31, 2025')

    def test_future_release_rejected(self):
        with self.assertRaises(ValueError):self.parse(self.fixture(date='February 5, 2027',ended='December 31, 2026'),(2026,4))

    def test_non_fiscal_quarter_end_rejected(self):
        with self.assertRaisesRegex(ValueError,'Fiscal end'):self.parse(self.fixture(ended='December 30, 2025'))

    def test_public_index_observations_keep_distinct_source_kind(self):
        events=legacy_events()
        self.assertEqual(len(events),5)
        self.assertEqual({x['precision'] for x in events},{'date_only'})
        self.assertTrue(all('not_full_html' in x['source_kind'] for x in events))


if __name__=='__main__':unittest.main()
