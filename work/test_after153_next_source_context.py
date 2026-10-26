"""Synthetic-only context tests; no historical source or financial values read."""
import copy
import json
import unittest
from bs4 import BeautifulSoup
import after153_next_source_context as m


def fixture(year=2024, quarter=True, split=False, percent=False):
    title=('<tr><td>Condensed Consolidated</td></tr><tr><td>Statement of Income (unaudited)</td></tr>' if split else '<tr><td colspan="5">Condensed Consolidated Statements of Income</td></tr>')
    if percent:
        header=f'<tr><td></td><td colspan="4">Year Ended</td></tr><tr><td></td><td>December 31,</td><td>% of</td><td>December 31,</td><td>% of</td></tr><tr><td>(In millions)</td><td>{year}</td><td>Revenues</td><td>{year-1}</td><td>Revenues</td></tr>'
        values='<td>$ 873</td><td>19%</td><td>$ 652</td><td>13%</td>'
    elif quarter:
        header=f'<tr><td colspan="5">(In millions)</td></tr><tr><td></td><td colspan="2">Three Months Ended December 31,</td><td colspan="2">Years Ended December 31,</td></tr><tr><td></td><td>{year}</td><td>{year-1}</td><td>{year}</td><td>{year-1}</td></tr>'
        values='<td>$ 111</td><td>$ 112</td><td>$ 873</td><td>$ 652</td>'
    else:
        header=f'<tr><td colspan="3">(In millions)</td></tr><tr><td></td><td colspan="2">Years Ended December 31,</td></tr><tr><td></td><td>{year}</td><td>{year-1}</td></tr>'
        values='<td>$ 873</td><td>$ 652</td>'
    income='<table>'+title+header+'<tr><td>Revenues</td>'+values+'</tr><tr><td>Net income</td>'+values+'</tr><tr><td>Net income attributable to common stockholders</td>'+values+'</tr></table>'
    balance=f'<table><tr><td colspan="3">Condensed Consolidated Balance Sheets</td></tr><tr><td colspan="3">(In millions)</td></tr><tr><td></td><td colspan="2">As of December 31,</td></tr><tr><td></td><td>{year}</td><td>{year-1}</td></tr><tr><td>Assets</td><td></td><td></td></tr><tr><td>Total assets</td><td>$ 9973</td><td>$ 9962</td></tr></table>'
    return income+balance


def contexts(html):
    soup=BeautifulSoup(html,'html.parser')
    return [x for i,t in enumerate(soup.find_all('table')) if (x:=m.table_context(t,i)) is not None]


def choose(html,year=2024):
    return m.choose_pair(contexts(html),year,f'{year}-12-31')


class ContextTests(unittest.TestCase):
    def test_compound_annual_not_quarter(self):
        data,gaps=choose(fixture());self.assertFalse(gaps)
        self.assertEqual(data['fields']['annual_net_income']['logical_columns'],[3])
    def test_split_title(self):
        data,gaps=choose(fixture(split=True));self.assertFalse(gaps)
        self.assertEqual(data['fields']['annual_net_income']['logical_columns'],[3])
    def test_percentage_revenue_header_not_body(self):
        data,gaps=choose(fixture(split=True,percent=True));self.assertFalse(gaps)
        self.assertEqual(data['fields']['annual_net_income']['logical_columns'],[1])
    def test_leap_year_366(self):
        data,_=choose(fixture());self.assertEqual(data['periods']['days_inclusive'],366)
    def test_standard_year_365(self):
        data,_=choose(fixture(2025),2025);self.assertEqual(data['periods']['days_inclusive'],365)
    def test_start_assumption_not_certified(self):
        data,_=choose(fixture());self.assertFalse(data['periods']['start_is_source_quoted']);self.assertFalse(data['periods']['start_assumption_approved'])
    def test_currency_iso_not_inferred(self):
        data,_=choose(fixture());self.assertIsNone(data['fields']['annual_net_income']['currency_iso'])
    def test_metadata_contains_no_amounts(self):
        text=json.dumps(contexts(fixture()))
        for number in ('873','652','9973','9962','111','112'):self.assertNotIn(number,text)
    def test_parent_attributable_not_substitute(self):
        self.assertIsNone(choose(fixture().replace('<td>Net income</td>','<td>Operating income</td>'))[0])
    def test_parent_rows_recorded_but_excluded(self):
        data,_=choose(fixture());self.assertEqual(data['fields']['annual_net_income']['label'],'Net income');self.assertEqual(len(data['excluded_parent_NI_labels']),1)
    def test_duplicate_income_ambiguous(self):
        tables=contexts(fixture());self.assertIsNone(m.choose_pair(tables+[copy.deepcopy(tables[0])],2024,'2024-12-31')[0])
    def test_no_annual_header(self):
        self.assertIsNone(choose(fixture().replace('Years Ended','Three Months Ended'))[0])
    def test_no_scale(self):
        self.assertIsNone(choose(fixture().replace('(In millions)',''))[0])
    def test_conflicting_scale(self):
        self.assertIsNone(choose(fixture().replace('(In millions)','(In millions and thousands)'))[0])
    def test_wrong_endpoint(self):
        self.assertIsNone(m.choose_pair(contexts(fixture()),2024,'2024-12-30')[0])
    def test_incorrect_previous_year(self):
        self.assertIsNone(choose(fixture().replace('2023','2022'))[0])
    def test_split_title_gap_rejected(self):
        html=fixture(split=True).replace('<tr><td>Statement of Income','<tr><td>Irrelevant</td></tr><tr><td>Statement of Income')
        self.assertIsNone(choose(html)[0])
    def test_unknown_non_temporal_digits(self):
        self.assertFalse(m.temporal('Net income 873'));self.assertFalse(m.temporal('As of $ 2024'))
    def test_header_date_binding_full(self):
        data,_=choose(fixture());self.assertEqual(data['periods']['annual_date_binding'],'same_report_full_date_headers')


if __name__=='__main__':unittest.main(verbosity=2)
