"""Tests target corrupted archive dates and false event evidence."""
import unittest
from collect_events import parse_event


class ParseTests(unittest.TestCase):
    def html(self, body, header='April 27, 2016'):
        return '<h1 class="PressReleasePage-headline">Amazon.com Announces First Quarter Sales</h1><div class="PressReleasePage-datePublished">' + header + '</div><div class="PressReleasePage-articleBody">' + body + '</div>'

    def parse(self, body, header='April 27, 2016'):
        return parse_event({'url': 'https://press.aboutamazon.com/example', 'evidence_path': 'raw/example.html', 'sha256': 'hash', 'retrieved_at': 'test'}, self.html(body, header), {'archive_sources': []})

    def test_original_dateline_wins_over_migrated_header(self):
        e = self.parse('SEATTLE--(BUSINESS WIRE)--Apr. 28, 2016-- Amazon announced first quarter ended March 31, 2016.')
        self.assertEqual(e['release_date'], '2016-04-28')
        self.assertTrue(e['migrated_header_date_conflicts_with_original'])
        self.assertIsNone(e['publication_timestamp'])
        self.assertEqual(e['publication_timezone'], 'America/Los_Angeles')

    def test_truncated_original_body_is_not_repaired_from_header(self):
        with self.assertRaisesRegex(ValueError, 'dateline missing'):
            self.parse('uarter ended March 31, 2016.')

    def test_guidance_or_call_date_does_not_replace_release_date(self):
        with self.assertRaisesRegex(ValueError, 'dateline missing'):
            self.parse('SEATTLE--(BUSINESS WIRE)-- first quarter ended March 31, 2016. Expectations as of Apr. 28, 2016. Call at 2 p.m.')

    def test_fiscal_year_is_actual_period_year(self):
        e = self.parse('SEATTLE--(BUSINESS WIRE)--Jan. 29, 2015-- fourth quarter ended December 31, 2014.', 'January 29, 2015')
        self.assertEqual((e['fiscal_year'], e['fiscal_quarter']), (2014, 4))

    def test_invalid_quarter_period_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.parse('SEATTLE--(BUSINESS WIRE)--Apr. 28, 2016-- second quarter ended March 31, 2016.')

    def test_next_year_comparative_period_rejected(self):
        with self.assertRaisesRegex(ValueError, '100 days'):
            self.parse('SEATTLE--(BUSINESS WIRE)--Apr. 28, 2016-- first quarter ended March 31, 2015.')


if __name__ == '__main__':
    unittest.main()
