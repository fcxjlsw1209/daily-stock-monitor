import json
import unittest
import after158_basic_shares_inventory as m

E={'fiscal_year':2024,'periods':{'current':{'period_end':'2024-12-31'},'comparison':{'period_end':'2023-12-31'}}}

def table(duration='Years ended',basic='Basic',heading='Weighted average shares outstanding',values=('87654321','98765432')):
    return ('<h2>Consolidated Statements of Income</h2><p>Shares in millions</p><table><tr><th></th><th colspan="2">'+duration+'</th></tr><tr><th></th><th>2024</th><th>2023</th></tr><tr><td>'+heading+'</td><td></td><td></td></tr><tr><td>'+basic+'</td><td>'+values[0]+'</td><td>'+values[1]+'</td></tr></table>').encode()

class Tests(unittest.TestCase):
    def test_annual_pair(self):
        r=m.html_inventory(table(),E);self.assertEqual(m.summarize(r)['status'],'explicit_annual_pair_locator_present')
    def test_quarter_never_annual(self):
        r=m.html_inventory(table(duration='Three months ended'),E);self.assertFalse(r['candidates'][0]['same_report_annual_pair_locator_present'])
    def test_diluted_not_basic(self):self.assertEqual(m.html_inventory(table(basic='Diluted'),E)['candidates'],[])
    def test_eps_value_not_share_count(self):self.assertEqual(m.html_inventory(table(heading='Net income per common share'),E)['candidates'],[])
    def test_basic_eps_row_is_rejected(self):self.assertEqual(m.html_inventory(table(basic='Basic earnings per share'),E)['candidates'],[])
    def test_eps_heading_stops_old_share_heading(self):
        self.assertIsNone(m.label_type('Basic',['Weighted average shares outstanding','Net income per common share']))
    def test_implicit_weighting_is_flagged(self):
        r=m.html_inventory(table(heading='Shares used in computing earnings per share'),E)
        self.assertIn('weighted_average_semantics_implicit_needs_review',r['candidates'][0]['blockers'])
    def test_share_amounts_do_not_leak(self):
        r=m.html_inventory(table(),E);encoded=json.dumps(r)
        self.assertNotIn('87654321',encoded);self.assertNotIn('98765432',encoded)
    def test_end_balance_is_not_weighted(self):self.assertIsNone(m.label_type('Shares outstanding',[]))
    def test_mixed_periods_need_own_columns(self):
        raw=b'<p>Shares in millions</p><table><tr><td></td><td colspan="2">Three months ended</td><td colspan="2">Years ended</td></tr><tr><td></td><td>2024</td><td>2023</td><td>2024</td><td>2023</td></tr><tr><td>Weighted average shares outstanding</td></tr><tr><td>Basic</td><td>1</td><td>2</td><td>87654321</td><td>98765432</td></tr></table>'
        c=m.html_inventory(raw,E)['candidates'][0];self.assertTrue(c['same_report_annual_pair_locator_present']);self.assertEqual(c['side_locators']['current'][0]['logical_column'],3)
    def test_mixed_shared_header_ambiguous(self):
        c=m.html_inventory(table(duration='Three months ended and years ended'),E)['candidates'][0];self.assertFalse(c['same_report_annual_pair_locator_present'])
    def test_missing_prior_not_substituted(self):
        c=m.html_inventory(table().replace(b'2023',b'2022'),E)['candidates'][0];self.assertFalse(c['same_report_annual_pair_locator_present'])
    def test_pdf_is_quantity_free_and_geometry_pending(self):
        text='Consolidated Statements of Income\n(In millions)\nYears ended\n2024 2023\nWeighted average shares outstanding\nBasic    87654321  98765432\nDiluted   22222222  33333333\n'
        r=m.pdf_inventory(text,E);s=json.dumps(r);self.assertNotIn('87654321',s);self.assertNotIn('98765432',s);self.assertFalse(r['candidates'][0]['same_report_annual_pair_locator_present'])
    def test_split_flags_do_not_leak_ratio(self):
        r=m.word_flags('A 87654321-for-98765432 stock split was retroactively adjusted.');s=json.dumps(r)
        self.assertTrue(r['split_context']);self.assertNotIn('87654321',s)
    def test_bad_span_reported_not_crashed(self):
        raw=table().replace(b'colspan="2"',b'colspan="9000"');self.assertTrue(m.html_inventory(raw,E)['structure_notes'])
    def test_no_old_cfo_gate(self):self.assertNotIn('required_field_completeness',m.rules()['prior_holds'])
    def test_all_candidates_retained_ambiguity(self):
        r=m.html_inventory(table()+table(),E);self.assertEqual(len(r['candidates']),2);self.assertEqual(m.summarize(r)['status'],'multiple_annual_pair_candidates_ambiguous')

if __name__=='__main__':unittest.main()
