import json
from pathlib import Path
import unittest

from collect import parse_text


class ParserTests(unittest.TestCase):
    def text(self, date='July 22, 2026', period='June 30, 2026', dateline='MOUNTAIN VIEW, Calif.'):
        return ('Alphabet Announces Second Quarter 2026 Results\n' + dateline + ' – ' + date +
                ' – Alphabet Inc. today announced financial results for the quarter ended ' + period +
                '.\nWebcast begins today 1:30 PM PT / 4:30 PM ET.')

    def parse(self, text=None, target=None):
        return parse_text(text or self.text(), target or dict(fiscal_year=2026, fiscal_quarter=2),
            'https://s206.q4cdn.com/479360582/files/release.pdf', 'raw/release.pdf', 'abc')

    def test_real_period_and_release_separate(self):
        e = self.parse()
        self.assertEqual(e['release_date'], '2026-07-22')
        self.assertEqual(e['fiscal_period_end_actual'], '2026-06-30')
        self.assertEqual((e['fiscal_year'], e['fiscal_quarter']), (2026, 2))

    def test_no_invented_time_from_conference_call(self):
        e = self.parse()
        self.assertIsNone(e['publication_timestamp'])
        self.assertEqual(e['precision'], 'date_only')
        self.assertFalse(e['first_public_time_verified'])
        self.assertEqual(e['publication_timezone'], 'America/Los_Angeles')
        self.assertFalse(e['publication_timezone_evidence']['cms_timezone_observed'])

    def test_future_release_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.parse(self.text(date='October 22, 2026'))

    def test_wrong_quarter_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requested fiscal quarter'):
            self.parse(target=dict(fiscal_year=2026, fiscal_quarter=1))

    def test_missing_dateline_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dateline'):
            self.parse(self.text(dateline='Unknown'))

    def test_wrong_actual_period_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Actual fiscal period'):
            self.parse(self.text(period='March 31, 2026'))

    def test_old_google_predecessor_and_extracted_dash(self):
        s = ('Google Announces Fourth Quarter and Fiscal Year 2011 Results\nMOUNTAIN VIEW, Calif. ± January 19, 2012 - '
            'Google Inc. announced results for the quarter and the fiscal year ended December 31, 2011.')
        e = self.parse(s, dict(fiscal_year=2011, fiscal_quarter=4))
        self.assertEqual(e['issuer_name_at_release'], 'Google Inc.')
        self.assertEqual(e['release_date'], '2012-01-19')

    def cached_release(self, year, quarter):
        here = Path(__file__).resolve().parent
        root = here.parents[2]
        event = next(e for e in json.loads((here / 'events.json').read_text())
                     if (e['fiscal_year'], e['fiscal_quarter']) == (year, quarter))
        text = (root / event['evidence_path']).with_suffix('.txt').read_text()
        return parse_text(text, dict(fiscal_year=year, fiscal_quarter=quarter),
                          event['source_url'], event['evidence_path'], event['evidence_sha256'])

    def test_actual_2011q1_uppercase_google_identity(self):
        e = self.cached_release(2011, 1)
        self.assertEqual(e['issuer_name_at_release'], 'Google Inc.')
        self.assertIn('GOOGLE ANNOUNCES', e['headline'])
        self.assertIn('Google Inc.', e['issuer_identity_evidence'])
        self.assertEqual(e['release_date'], '2011-04-14')
        self.assertEqual(e['fiscal_period_end_actual'], '2011-03-31')

    def test_actual_2026q2_alphabet_identity(self):
        e = self.cached_release(2026, 2)
        self.assertEqual(e['issuer_name_at_release'], 'Alphabet Inc.')
        self.assertIn('Alphabet Inc.', e['issuer_identity_evidence'])
        self.assertEqual(e['release_date'], '2026-07-22')
        self.assertEqual(e['fiscal_period_end_actual'], '2026-06-30')

    def test_later_google_mentions_do_not_change_alphabet_identity(self):
        e = self.parse(self.text() + ' Google Inc. was the predecessor issuer.')
        self.assertEqual(e['issuer_name_at_release'], 'Alphabet Inc.')

    def test_unknown_announcing_issuer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'announcing issuer absent'):
            self.parse(self.text().replace('Alphabet Inc.', 'Other Company Inc.'))

    def test_headline_body_issuer_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.parse(self.text().replace('Alphabet Inc.', 'Google Inc.'))


if __name__ == '__main__':
    unittest.main()
