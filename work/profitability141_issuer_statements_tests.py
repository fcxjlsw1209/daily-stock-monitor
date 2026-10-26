import unittest
from bs4 import BeautifulSoup
import profitability141_issuer_statements as m

class StatementTests(unittest.TestCase):
    def fixture(self,amount='12',caption='INCOME STATEMENTS (In millions)'):
        soup=BeautifulSoup('<table><caption>'+caption+'</caption><tr><th></th><th>Twelve Months Ended June 30,</th><th>Twelve Months Ended June 30,</th></tr><tr><td></td><td>2025</td><td>2024</td></tr><tr><td>Revenue:</td><td>$100</td><td>$90</td></tr><tr><td>Cost of revenue</td><td>88</td><td>80</td></tr><tr><td>Gross margin</td><td>'+amount+'</td><td>10</td></tr></table>','html.parser')
        spec={'kind':'annual_gross_amount','table_index_zero_based':0,'row_index_zero_based':4,'row_label':'Gross margin','matching_cells':[{'source_cell_index':1,'grid_columns':[1],'header_literals':['Twelve Months Ended June 30,','2025']}],'units':{'literal':'(In millions)'}}
        return soup,spec
    def test_exact_annual_column(self):
        soup,spec=self.fixture()
        row=m.extract_cell(soup,spec,'MSFT','2025-06-30')
        self.assertEqual(row['amount_decimal'],'12')
        self.assertEqual(row['header_literals'],['Twelve Months Ended June 30,','2025'])
    def test_percentage_despite_monetary_table_rejected(self):
        soup,spec=self.fixture('12%')
        with self.assertRaises(ValueError):m.extract_cell(soup,spec,'MSFT','2025-06-30')
    def test_non_gaap_annual_table_rejected(self):
        soup,spec=self.fixture(caption='Non-GAAP INCOME STATEMENTS (In millions)')
        with self.assertRaises(ValueError):m.extract_cell(soup,spec,'MSFT','2025-06-30')
    def test_wrong_period_rejected(self):
        soup,spec=self.fixture()
        with self.assertRaises(ValueError):m.extract_cell(soup,spec,'MSFT','2024-06-30')
    def test_duplicate_row_rejected(self):
        soup,spec=self.fixture();soup.table.append(BeautifulSoup('<tr><td>Gross margin</td><td>13</td><td>11</td></tr>','html.parser').tr)
        with self.assertRaises(ValueError):m.extract_cell(soup,spec,'MSFT','2025-06-30')
    def test_amount_and_scale(self):
        self.assertEqual(m.exact_amount('$ (1,234.50)'),'-1234.50')
        self.assertEqual(m.scale_from('(In thousands, except per share data)'),1000)
        self.assertEqual(m.scale_from('(In millions)'),1000000)
    def test_percent_blank_bad_comma_rejected(self):
        for s in ['71.2%','—','','1,23','NaN','Infinity']:
            with self.assertRaises(ValueError):m.exact_amount(s)
    def test_scale_conflict(self):
        for s in ['In millions and in thousands','amount']:
            with self.assertRaises(ValueError):m.scale_from(s)
    def test_non_gaap_rejected(self):
        with self.assertRaises(ValueError):m.validate_statement('Non-GAAP INCOME STATEMENTS','Gross margin','annual_gross_amount','MSFT')
        with self.assertRaises(ValueError):m.validate_statement('INCOME STATEMENTS','As adjusted cost of revenue','annual_gross_amount','MSFT')
    def test_summary_rejected(self):
        with self.assertRaises(ValueError):m.validate_statement('FY2026 Summary GAAP','Gross profit','annual_gross_amount','NVDA')
    def test_spans(self):
        soup=BeautifulSoup('<table><tr><th rowspan="2">Item</th><th colspan="2">Twelve Months Ended</th></tr><tr><th>2026</th><th>2025</th></tr><tr><td>Gross profit</td><td>10</td><td>8</td></tr></table>','html.parser')
        rows,grid,physical=m.table_grid(soup.table)
        self.assertEqual(physical[(2,1)]['columns'],[1])
        self.assertEqual(m.headers_for(grid,2,1,2),['Twelve Months Ended','2026'])
    def test_calendar_dates(self):
        p=m.derive_period('2016-06-30','2015-06-30','MSFT')
        self.assertEqual(p['start'],'2015-07-01')
        self.assertEqual(p['days_inclusive'],366)
        p=m.derive_period('2016-01-31','2015-01-25','NVDA')
        self.assertEqual(p['start'],'2015-01-26')
        self.assertEqual(p['days_inclusive'],371)
        self.assertFalse(p['start_is_direct_quote'])
    def test_not_annual(self):
        with self.assertRaises(ValueError):m.derive_period('2016-01-31','2015-10-31','NVDA')
    def test_next_session_weekend_holiday(self):
        sessions=['2025-01-08','2025-01-10','2025-01-13']
        self.assertEqual(m.first_session_after('2025-01-08',sessions),'2025-01-10')
        self.assertEqual(m.first_session_after('2025-01-11',sessions),'2025-01-13')
        self.assertIsNone(m.first_session_after('2025-01-13',sessions))
    def test_hash_and_paths(self):
        with self.assertRaises(ValueError):m.read_pinned({'path':m.PLAN,'sha256':'0'*64})
        with self.assertRaises(ValueError):m.safe_path('../escape')
    def test_original_receipt_mismatch(self):
        with self.assertRaises(ValueError):m.receipt_check(b'x',{'http_status':200,'sha256':'0'*64,'url':'https://www.microsoft.com/'},{'source_url':'https://www.microsoft.com/'})
    def test_split_header_dates(self):
        self.assertEqual(m.header_dates(['Twelve Months Ended','January 31,','2016']),['2016-01-31'])

if __name__=='__main__':unittest.main()
