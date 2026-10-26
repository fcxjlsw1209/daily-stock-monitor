"""Synthetic abbreviated-date regression checks for frozen-v1 replacement."""
import unittest
from liquidity162_audit_v2 import release_evidence, iso_date

class DottedDates(unittest.TestCase):
    def test_dotted_and_full_literals(self):
        for literal in ('Jan. 27, 2019','Jan 27, 2019','January 27, 2019'):
            self.assertEqual(iso_date(literal), '2019-01-27')
        self.assertEqual(iso_date('Sept. 24, 2026'), '2026-09-24')

    def test_period_evidence_keeps_raw_literal(self):
        row=dict(ticker='NVDA',fiscal_year=2019,release_date='2019-02-14',fiscal_period_end='2019-01-27')
        html=('<div class="article-title">Fourth Quarter and Fiscal 2019</div>'
              '<div class="article-date">February 14, 2019</div>'
              '<div class="article-body">Quarter ended Jan. 27, 2019.</div>')
        self.assertEqual(release_evidence(html,row)['period_end_literal'],'Jan. 27, 2019')
        with self.assertRaises(ValueError):
            release_evidence(html.replace('Jan. 27','Jan. 26'),row)

    def test_invalid_date_still_fails(self):
        for literal in ('Jan. 32, 2019','Junk. 27, 2019','Jan 27, 19'):
            with self.assertRaises(ValueError): iso_date(literal)

if __name__=='__main__': unittest.main()
