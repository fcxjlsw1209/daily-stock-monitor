"""Bounded synthetic parser checks; no market input or outcome calculations."""
import unittest
import shell_pool_source168_extract as s


class TableParser(unittest.TestCase):
    def test_legacy_three_columns_exclude_sterling_deferred(self):
        text='''Issued and fully paid
Ordinary shares of €0.07 each Sterling deferred shares
Number of shares A B of £1 each
At January 1, 2015 1,000 2,000 50,000
Scrip dividends 10 - -
Repurchases of shares (5) - -
At December 31, 2015 1,005 2,000 50,000
'''
        v=s.parse_table(text,2015,4)
        self.assertEqual((v['A'],v['B'],v['total_derived']),(1005,2000,3005))
        self.assertTrue(v['rollforward']['agrees'])

    def test_combined_table_excludes_nominal_amounts(self):
        text='''ISSUED AND FULLY PAID ORDINARY SHARES OF €0.07 EACH1
Number of shares Nominal value ($ million)
A B A B Total
At January 1, 2019 1,000 2,000 7 14 21
Repurchases of shares (20) (5) (1) (1) (2)
At March 31, 2019 980 1,995 6 13 19
At January 1, 2018 1,000 2,000 7 14 21
'''
        v=s.parse_table(text,2019,1)
        self.assertEqual((v['A'],v['B']),(980,1995))
        self.assertEqual(v['rollforward']['movements'][0]['A'],-20)

    def test_year_end_not_weighted_eps_or_comparative(self):
        text='''Weighted average number of shares
Basic earnings per share 9,999 8,888
ISSUED AND FULLY PAID ORDINARY SHARES OF €0.07 EACH1
Number of shares Nominal value ($ million)
A B A B Total
At January 1, 2018 1,000 2,000 7 14 21
At June 30, 2018 1,000 2,000 7 14 21
At January 1, 2017 9,000 8,000 7 14 21
Scrip dividends 1,000 1,000 7 14 21
At June 30, 2017 10,000 9,000 7 14 21
'''
        v=s.parse_table(text,2018,2)
        self.assertEqual((v['A'],v['B']),(1000,2000))
        self.assertEqual(v['rollforward']['movements'],[])

    def test_issued_acquisition_footnote_and_wrapped_label(self):
        text='''ISSUED AND FULLY PAID
Number of shares A B of £1 each
At January 1, 2016 1,000 2,000 50,000
Shares issued for the 100 200 0
acquisition of BG Group plc1
At March 31, 2016 1,100 2,200 50,000
'''
        v=s.parse_table(text,2016,1)
        self.assertEqual((v['A'],v['B']),(1100,2200))

    def test_footnoted_issuance_label_ignores_note_number(self):
        text='''ISSUED AND FULLY PAID ORDINARY SHARES OF €0.07 EACH1
Number of shares A B A B Total
At January 1, 2016 1,000 2,000 7 14 21
Shares issued2 100 200 1 2 3
At December 31, 2016 1,100 2,200 8 16 24
'''
        self.assertEqual(s.parse_table(text,2016,4)['A'],1100)

    def test_inconsistent_rollforward_fails(self):
        text='''ISSUED AND FULLY PAID
At January 1, 2015 1,000 2,000 50,000
Scrip dividends 20 - -
At December 31, 2015 1,010 2,000 50,000
'''
        with self.assertRaisesRegex(AssertionError,'Rollforward'):s.parse_table(text,2015,4)

    def test_unknown_movement_fails_instead_of_dropping(self):
        text='''ISSUED AND FULLY PAID
At January 1, 2015 1,000 2,000 50,000
Mystery share action 10 - -
At December 31, 2015 1,010 2,000 50,000
'''
        with self.assertRaisesRegex(AssertionError,'unrecognized'):s.parse_table(text,2015,4)

    def test_dash_is_only_zero_when_explicit_cell(self):
        self.assertEqual(s.amount('—'),0)
        self.assertEqual(s.amount('-'),0)
        self.assertEqual(s.amount('(1,000)'),-1000)
        for value in ('','n/a','1.1',' 0'):
            with self.assertRaises(AssertionError):s.amount(value)

    def test_date_cannot_masquerade_as_missing_numeric_column(self):
        for row in ('At January 1, 2015 1,000 2,000',
                    'At January 1, 2015 1,000 2,000 50,000 1',
                    'Basic EPS 1,000 2,000 50,000'):
            with self.assertRaises(AssertionError):s.row_values(row,3)


if __name__=='__main__':unittest.main()
