import unittest
from parse_sec_foreign_roster import parse


class ForeignRosterTests(unittest.TestCase):
    def test_equity_and_non_equity_are_distinct(self):
        rows, bad = parse({1: 'Barclays plc United Kingdom NYSE',
                           2: 'Barclays Bank plc United Kingdom NYSE - Preferred',
                           3: 'Credit Suisse AG Switzerland NYSE - Debt',
                           4: 'Example Canada OTC'})
        self.assertEqual(bad, [])
        self.assertEqual([r['exchange_equity_candidate'] for r in rows],
                         [True, False, False, False])

    def test_country_words_inside_company_name(self):
        rows, bad = parse({1: 'China Mobile Ltd. Hong Kong NYSE',
                           2: 'Royal Bank of Canada Canada NYSE'})
        self.assertFalse(bad)
        self.assertEqual(rows[0]['name'], 'China Mobile Ltd.')
        self.assertEqual(rows[1]['name'], 'Royal Bank of Canada')

    def test_unrecognized_rows_remain_visible(self):
        rows, bad = parse({1: 'Example Unknownland Mystery Market'})
        self.assertEqual(rows, [])
        self.assertEqual(len(bad), 1)

    def test_snapshot_does_not_certify_daily_eligibility(self):
        rows, _ = parse({1: 'Example Israel Global Mkt'})
        self.assertEqual(rows[0]['as_of'], '2009-12-31')
        self.assertFalse(rows[0]['daily_eligibility_certified'])


if __name__ == '__main__':
    unittest.main()
