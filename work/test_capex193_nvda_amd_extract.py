"""Synthetic-only parser and fail-closed source semantics tests."""
import copy, json, tempfile, unittest
from pathlib import Path
import capex193_nvda_amd_extract as m

def fixture(label='Purchases of property, plant and equipment', amount='(96 )', admitted=True):
    html=f'''<html><p>CONSOLIDATED STATEMENTS OF CASH FLOWS</p><p>(In millions) (Unaudited)</p>
<table><tr><td></td><td>Three Months Ended</td><td>Year Ended</td></tr>
<tr><td></td><td>December 31, 2020</td><td>December 31, 2020</td></tr>
<tr><td colspan="3">Cash flows from operating activities:</td></tr>
<tr><td>Net income</td><td>$ 4</td><td>$ 9</td></tr>
<tr><td>Net cash provided by operating activities</td><td>10</td><td>20</td></tr>
<tr><td>Cash flows from investing activities:</td></tr>
<tr><td>{label}</td><td>(3)</td><td>{amount}</td></tr></table></html>'''.encode()
    loc={'table_index_zero_based':0,'row_index_zero_based':4,'logical_columns':[2],'period_end':'2020-12-31'}
    t={'event_id':'AMD:FY2020:2021-01-20','symbol':'AMD','source':{},'inherited_admitted':admitted,'inherited_available_from':'2021-01-21','current_cfo_locator':loc}
    o={'fiscal_year':2020,'release_date':'2021-01-20','cash_quality_admission':{'reasons':[] if admitted else ['old_hold']},'inherited_rejection_codes':[] if admitted else ['old_code'],'periods':{'current':{'period_start':'2020-01-01','period_end':'2020-12-31','days_inclusive':366}}}
    return html,t,o

class Tests(unittest.TestCase):
    def test_exact_sign_and_grammar(self):
        self.assertEqual(m.exact_signed('$ (1,234 )'),'-1234')
        self.assertEqual(m.exact_signed('(0)'), '0')
        for bad in ('—','1, 234','(-23)','23%','23 extra','(23'):
            with self.assertRaises(ValueError):m.exact_signed(bad)
    def test_unique_annual_outflow(self):
        b,t,o=fixture();r=m.extract(b,t,o)
        self.assertTrue(r['extraction_verified']);self.assertEqual(r['status'],'verified')
        self.assertEqual(r['capex']['positive_outflow_decimal'],'96')
        self.assertEqual(r['capex']['locator']['physical_cell_indices_zero_based'],[2])
        self.assertEqual(r['capex']['row_text_sha256'],m.sha(r['capex']['row_text'].encode()))
    def test_old_hold_not_overridden(self):
        b,t,o=fixture(admitted=False);r=m.extract(b,t,o)
        self.assertTrue(r['extraction_verified']);self.assertEqual(r['status'],'held')
        self.assertIn('old_code',r['hold_reasons']);self.assertFalse(r['fact_admitted'])
    def test_combined_intangibles_held(self):
        b,t,o=fixture('Purchases of property and equipment and intangible assets')
        r=m.extract(b,t,o);self.assertIsNone(r['capex']);self.assertFalse(r['extraction_verified'])
        self.assertEqual(r['candidate_evidence'][0]['annual_amount']['positive_outflow_decimal'],'96')
    def test_principal_not_total_purchases(self):
        b,t,o=fixture('Principal payments on property and equipment')
        r=m.extract(b,t,o);self.assertIsNone(r['capex'])
        self.assertEqual(r['candidate_evidence'][0]['classification'],'principal_payment_not_total_purchases')
    def test_free_cash_flow_only_never_substituted(self):
        b,t,o=fixture();t['current_cfo_locator']=None
        r=m.extract(b,t,o);self.assertIsNone(r['capex']);self.assertEqual(len(r['candidate_evidence']),1)
    def test_annual_column_not_quarter_or_wrong_year(self):
        b,t,o=fixture();t['current_cfo_locator']['logical_columns']=[1]
        r=m.extract(b,t,o);self.assertIsNone(r['capex'])
        self.assertEqual(r['candidate_evidence'][0]['annual_amount_hold_reason'],'not annual column')
        b,t,o=fixture();t['current_cfo_locator']['period_end']='2019-12-31'
        self.assertIsNone(m.extract(b,t,o)['capex'])
    def test_unsigned_payment_held(self):
        b,t,o=fixture(amount='96');self.assertIsNone(m.extract(b,t,o)['capex'])
    def test_duplicate_scopes_held_not_summed(self):
        b,t,o=fixture();b=b.replace(b'</table>',b'<tr><td>Cash flows from discontinued operations:</td></tr><tr><td>Purchases of property, plant and equipment</td><td>(1)</td><td>(4)</td></tr></table>')
        r=m.extract(b,t,o);self.assertIsNone(r['capex']);self.assertEqual(len(r['candidate_evidence']),2)
        self.assertIn('multiple_scope_pp_and_e_rows_no_single_consolidated_literal',r['hold_reasons'])
    def test_split_parenthesis_cell(self):
        b,t,o=fixture();b=b.replace(b'<td>(96 )</td>',b'<td>(96</td><td>)</td>')
        r=m.extract(b,t,o);self.assertEqual(r['capex']['positive_outflow_decimal'],'96')
        self.assertEqual(r['capex']['locator']['physical_cell_indices_zero_based'],[2,3])
    def test_span_grid_and_source_hash(self):
        soup=m.BeautifulSoup('<table><tr><td colspan="2">label</td><td>(3)</td></tr></table>','html.parser')
        rows,g,p=m.table_grid(soup.table);self.assertIs(g[(0,0)],g[(0,1)])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a').write_bytes(b'hello');self.assertEqual(m.source_bytes(root,{'path':'a','sha256':m.sha(b'hello')}),b'hello')
            with self.assertRaises(ValueError):m.source_bytes(root,{'path':'a','sha256':'wrong'})
    def test_receipt_binding_and_exclusive_write(self):
        s={'sha256':'abc','url':'https://example.com'};r={'sha256':'abc','url':s['url'],'status':200}
        m.validate_receipt(s,r)
        for bad in ({**r,'sha256':'bad'},{**r,'status':403},{**r,'url':'wrong'}):
            with self.assertRaises(ValueError):m.validate_receipt(s,bad)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';m.write_once(p,{'one':1})
            with self.assertRaises(FileExistsError):m.write_once(p,{'two':2})
            self.assertEqual(json.loads(p.read_text()),{'one':1})

if __name__=='__main__':unittest.main()
