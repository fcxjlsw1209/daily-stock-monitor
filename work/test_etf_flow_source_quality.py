import unittest
from decimal import Decimal
from etf_flow_source_quality import positive_decimal, coverage, accounting_checks, parse_ishares

class SourceQualityTests(unittest.TestCase):
    def test_placeholders_are_not_zero(self):
        for value in (None, '-', '', 'NaN', 'Infinity', 0, -1, True):
            self.assertIsNone(positive_decimal(value))
        self.assertEqual(positive_decimal('1,000.25'), Decimal('1000.25'))

    def test_missing_and_extra_dates_are_explicit(self):
        result = coverage([{'date':'2024-03-29','nav':'1','shares_outstanding':'-'}],
                          ['2024-03-28'], '2024-03-28', '2024-03-29')
        self.assertEqual(result['absent_sessions'], ['2024-03-28'])
        self.assertEqual(result['source_non_session_dates'], ['2024-03-29'])
        self.assertFalse(result['complete_nav_and_shares'])

    def test_invalid_session_field_not_filled(self):
        r = coverage([{'date':'2024-03-28','nav':1,'shares_outstanding':'-'}],
                     ['2024-03-28'], '2024-03-28', '2024-03-28')
        self.assertEqual(r['invalid_session_fields']['shares_outstanding'], ['2024-03-28'])

    def test_duplicates_fail(self):
        with self.assertRaises(ValueError):
            coverage([{'date':'2024-01-02'}] * 2, [], '', '')

    def test_matched_units_pass(self):
        r = accounting_checks([{'date':'d','nav':'50','shares_outstanding':'20','net_assets':'1000'}], ['d'])
        self.assertEqual(r['checked_rows'], 1)
        self.assertEqual(r['coarse_unit_screen_failures'], [])

    def test_split_scale_mismatch_fails(self):
        r = accounting_checks([{'date':'d','nav':'50','shares_outstanding':'10','net_assets':'1000'}], ['d'])
        self.assertEqual(len(r['coarse_unit_screen_failures']), 1)

    def test_no_aum_inference(self):
        r = accounting_checks([{'date':'d','nav':50,'shares_outstanding':20}], ['d'])
        self.assertEqual(r['unavailable_dates'], ['d'])
        self.assertFalse(r['derived_aum_or_share_proxy_used'])

    def test_wrong_xml_sheet_fails(self):
        with self.assertRaises(ValueError):
            parse_ishares(b'<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"/>')

if __name__ == '__main__':
    unittest.main()
