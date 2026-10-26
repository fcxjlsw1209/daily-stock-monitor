import json
from copy import deepcopy
from pathlib import Path
import unittest
import after142_aapl_comparative_source as m
from after141_aapl_source_extract_tests import AppleTests, fixture

class ComparisonTests(unittest.TestCase):
    def test_exact_comparison_column_and_date(self):
        texts,s=fixture();x=m.parse_comparisons(texts,s)
        self.assertEqual(x['comparison_gross_amount']['amount_decimal'],'7')
        self.assertEqual(x['comparison_total_assets']['amount_decimal'],'27')
        self.assertEqual(x['comparison_gross_amount']['date'],'2023-09-30')
        self.assertEqual(x['comparison_gross_amount']['locator']['financial_column_index_0based'],3)
    def test_prior_duration_is_not_invented(self):
        t,s=fixture();x=m.parse_comparisons(t,s)
        self.assertIsNone(x['comparison_annual_period']['days_inclusive'])
        self.assertIsNone(x['comparison_annual_period']['start'])
    def test_no_mutation(self):
        t,s=fixture();old=deepcopy((t,s));m.parse_comparisons(t,s);self.assertEqual((t,s),old)
    def test_nonpositive_prior_assets(self):
        for value in ['0','-27','(27)']:
            t,s=fixture();t[1]=t[1].replace('$ 27','$ '+value)
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'comparison assets'):m.parse_comparisons(t,s)
    def test_prior_negative_gross_kept(self):
        t,s=fixture();t[0]=t[0].replace('2 1 8 7','2 1 8 (7)')
        self.assertEqual(m.parse_comparisons(t,s)['comparison_gross_amount']['amount_decimal'],'-7')
    def test_no_admission_implied(self):
        t,s=fixture();x=m.parse_comparisons(t,s)
        self.assertFalse(x['fact_admitted']);self.assertFalse(x['strategy_admitted'])
    def test_late_comparison_is_current_release_event(self):
        ss=['2023-09-29','2023-10-02','2024-10-31','2024-11-01','2024-11-04']
        self.assertEqual(m.available('2024-10-31',ss),'2024-11-01')
    def test_weekend_next_session(self):
        self.assertEqual(m.available('2024-11-02',['2024-11-01','2024-11-04']),'2024-11-04')
    def test_out_of_bounds_calendar(self):
        for d in ['2023-01-01','2024-11-05']:
            with self.subTest(day=d),self.assertRaises(ValueError):m.available(d,['2024-11-01','2024-11-04'])
    def test_bad_calendar(self):
        with self.assertRaises(ValueError):m.available('2024-11-01',['2024-11-04','2024-11-01'])
    def test_mismatched_comparison_header(self):
        t,s=fixture();x=m.original.parse_layout_pages(t,s)
        with self.assertRaisesRegex(ValueError,'date mismatch'):m.prior_cell(x['gross_amount'],3,'2022-10-01')
    def test_current_cell_not_changed(self):
        t,s=fixture();x=m.original.parse_layout_pages(t,s);b=deepcopy(x)
        m.prior_cell(x['gross_amount'],3,'2023-09-30');self.assertEqual(x,b)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    m.OUT.mkdir(exist_ok=True)
    m.write_new(m.OUT/'TESTS.json',{'passed':True,'synthetic_only':True,'tests_run':result.testsRun,
        'source_sha256':m.sha(Path(m.__file__).read_bytes()),'test_sha256':m.sha(Path(__file__).read_bytes())})
