"""Synthetic checks for the source-only coordinator; no actual-source extraction."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd

from liquidity162_audit import release_evidence, verify_schema, FIELDS, CODE, PLAN_SHA, check_registry


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.msft = dict(ticker='MSFT', fiscal_year=2020, release_date='2020-07-22', fiscal_period_end='2020-06-30')
        self.html = '<p>Earnings Release FY20 Q4</p><p>REDMOND, Wash. — July 22, 2020 fiscal year ended June 30, 2020</p>'

    def test_original_dateline(self):
        self.assertEqual(release_evidence(self.html, self.msft)['period_end_literal'], 'June 30, 2020')

    def test_modified_date_not_publication(self):
        with self.assertRaises(ValueError):
            release_evidence(self.html.replace('REDMOND, Wash.', 'dateModified'), self.msft)

    def test_future_fiscal_wrong(self):
        with self.assertRaises(ValueError):
            release_evidence(self.html.replace('June 30, 2020', 'June 30, 2021'), self.msft)

    def test_fiscal_year_wrong(self):
        with self.assertRaises(ValueError):
            release_evidence(self.html.replace('FY20', 'FY21'), self.msft)

    def test_nvda_identity(self):
        row = dict(ticker='NVDA', fiscal_year=2020, release_date='2020-02-13', fiscal_period_end='2020-01-26')
        html = ('<div class="article-title">Fourth Quarter and Fiscal 2020</div>'
                '<div class="article-date">February 13, 2020</div>'
                '<div class="article-body">Quarter ended January 26, 2020.</div>')
        self.assertTrue(release_evidence(html, row)['reparsed_from_original_html'])
        with self.assertRaises(ValueError):
            release_evidence(html.replace('Fourth', 'Third'), row)

    def test_candidate_cannot_drop_short_debt(self):
        amount = {'value_reported': '3', 'scale': 1000, 'raw_cells': ['Cash', '3'], 'row_index': 1}
        row = dict(ticker='MSFT', status='candidate', fields={k:copy.deepcopy(amount) for k in FIELDS},
                   current_debt_components=['short_term_debt'], noncurrent_debt_components=['long_term_debt'])
        verify_schema(row)
        row['fields']['short_term_debt'] = None
        with self.assertRaises(ValueError):
            verify_schema(row)
        row['status'] = 'rejected'
        verify_schema(row)

    def test_current_noncurrent_disjoint(self):
        row = dict(ticker='NVDA', status='rejected', fields={k:None for k in FIELDS},
                   current_debt_components=['long_term_debt'], noncurrent_debt_components=['long_term_debt'])
        with self.assertRaises(ValueError):
            verify_schema(row)

    def test_known_by_after_release_and_weekend(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'outputs/monthly_earnings_repair'))
        from event_schema import normalize_event
        schedule = pd.DataFrame({'close': pd.to_datetime(['2020-07-24T20:00:00Z', '2020-07-27T20:00:00Z'])},
                                index=pd.to_datetime(['2020-07-24','2020-07-27']))
        result = normalize_event({'symbol':'TEST','asset_type':'equity','official':{
            'verified':True,'actual_period_end':'2020-06-30','publication_date':'2020-07-24',
            'publication_timezone':'America/Los_Angeles','source_url':'https://example.test/release',
            'publication_kind':'release'}}, SimpleNamespace(schedule=schedule))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['available_session'], '2020-07-27')
        self.assertEqual(result['known_by_at'], '2020-07-27T20:00:00+00:00')
        self.assertFalse(result['known_by_is_actual_publication'])

    def test_registry_requires_complete_tested_code_and_runtime(self):
        hashes = {p:'synthetic-digest' for p in CODE}
        runtime = {'python':'synthetic-runtime'}
        code = dict(source_plan_sha256=PLAN_SHA, code_hashes=hashes.copy(), runtime=runtime)
        validation = dict(code_hashes=hashes.copy(), runtime=runtime, passed=True, synthetic_only=True)
        check_registry(code, validation, hashes, runtime)
        for mutate in ('missing_code', 'changed_code', 'changed_test', 'runtime', 'untested'):
            c,v = copy.deepcopy(code),copy.deepcopy(validation)
            if mutate=='missing_code': c['code_hashes'].pop(CODE[0])
            if mutate=='changed_code': c['code_hashes'][CODE[0]]='changed'
            if mutate=='changed_test': v['code_hashes'][CODE[0]]='changed'
            if mutate=='runtime': v['runtime']={'python':'different'}
            if mutate=='untested': v['passed']=False
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                check_registry(c,v,hashes,runtime)


if __name__ == '__main__':
    unittest.main()
