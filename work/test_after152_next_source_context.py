"""Synthetic-only checks for MU/PEP metadata redaction and annual ownership."""
import json
import unittest
from bs4 import BeautifulSoup
import after152_next_source_feasibility as f
import after152_next_source_context as c


def table(kind='income',parent=False,quarter=False):
    title='CONSOLIDATED STATEMENTS OF INCOME' if kind=='income' else 'CONSOLIDATED BALANCE SHEETS'
    label='Net income attributable to parent' if parent else 'Net income' if kind=='income' else 'Total assets'
    period='Three Months Ended' if quarter else 'Years Ended'
    return BeautifulSoup('<table><tr><th colspan="3">'+title+'</th></tr><tr><th colspan="3">In millions</th></tr>'
      +('<tr><th></th><th colspan="2">'+period+'</th></tr>' if kind=='income' else '')
      +'<tr><th></th><th>December 31, 2020</th><th>December 31, 2019</th></tr>'
      +'<tr><td>'+('Revenue' if kind=='income' else 'Assets')+'</td><td>$</td><td></td></tr>'
      +'<tr><td>'+label+'</td><td>987654321</td><td>(876543210)</td></tr></table>','html.parser').table


class MetadataTests(unittest.TestCase):
    def test_monetary_values_never_leave_table_scan(self):
        result,_=f.html_info(str(table()).encode());text=json.dumps(result)
        self.assertNotIn('987654321',text);self.assertNotIn('876543210',text)

    def test_amounts_are_not_metadata(self):
        for text in ('987654321','$2020','margin 30%','income 987 years ended December 31, 2020'):
            self.assertFalse(f.is_metadata(text))

    def test_redaction_removes_dates_and_financial_digits_in_prose(self):
        self.assertNotRegex(f.masked('Restated 2020 from $1,234.56 to (4,500) million'),r'\d')

    def test_current_prior_dates_and_units_without_amounts(self):
        inc=c.table_context(table(),0);bs=c.table_context(table('balance'),1)
        result,gaps=c.choose_pair([inc,bs],2020,'2020-12-31')
        self.assertEqual(gaps,[]);self.assertEqual(result['periods']['proposed_current_start'],'2020-01-01')
        self.assertEqual(result['fields']['annual_net_income']['scale'],'1000000')
        self.assertNotIn('987654321',json.dumps(result));self.assertNotIn('876543210',json.dumps(result))
        self.assertFalse(result['periods']['start_assumption_approved'])

    def test_parent_only_cannot_supply_consolidated_ni(self):
        self.assertIsNone(c.table_context(table(parent=True),0))

    def test_quarter_does_not_supply_annual_column(self):
        result,gaps=c.choose_pair([c.table_context(table(quarter=True),0),c.table_context(table('balance'),1)],2020,'2020-12-31')
        self.assertIsNone(result);self.assertTrue(gaps)

    def test_duplicate_formal_table_is_ambiguous(self):
        inc=c.table_context(table(),0);bs=c.table_context(table('balance'),1)
        self.assertIsNone(c.choose_pair([inc,inc,bs],2020,'2020-12-31')[0])

    def test_slash_headers_are_dates_not_amounts(self):
        self.assertEqual(c.parsed_dates(['12/27/2014']),['2014-12-27'])
        self.assertEqual(c.parsed_dates(['12/27/14']),['2014-12-27'])
        self.assertEqual(c.parsed_dates(['1,234,567']),[])
        self.assertEqual(c.parsed_dates(['99/99/2014']),[])

    def test_unit_caption_may_contain_dollar_symbol_or_audit_date(self):
        self.assertTrue(c.units('($ and shares in millions, except per share amounts)'))
        self.assertTrue(c.units('(in millions, unaudited except year-ended 12/31/2019 amounts)'))
        self.assertFalse(c.units('Revenue $123 million'))

    def test_caption_prior_date_does_not_contaminate_current_columns(self):
        tab=table()
        tab.find_all('tr')[1].th.string='(in millions, unaudited except year-ended 12/31/2019 amounts)'
        parsed=c.table_context(tab,0)
        self.assertIn('2020-12-31',parsed['date_columns'])
        self.assertIn('2019-12-31',parsed['date_columns'])
        self.assertEqual(parsed['scale'],'1000000')

    def test_wrong_date_or_missing_scale_is_gap(self):
        inc=c.table_context(table(),0);bs=c.table_context(table('balance'),1)
        self.assertIsNone(c.choose_pair([inc,bs],2020,'2020-12-30')[0])
        bs['scale']=None
        self.assertEqual(c.choose_pair([inc,bs],2020,'2020-12-31')[1],['unit_or_currency_context_unresolved'])


if __name__=='__main__':unittest.main(verbosity=2)
