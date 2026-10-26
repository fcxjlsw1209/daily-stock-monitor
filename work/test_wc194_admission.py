from copy import deepcopy
import hashlib
import unittest
import wc194_admission as a

class Gate(unittest.TestCase):
    def setUp(self):
        src={'path':'test.html','sha256':'a'*64};loc={'format':'html','table_index_zero_based':1,'row_index_zero_based':9,'period_end':'2020-12-31'}
        self.o=dict(event_id='X:2020',symbol='X',release_date='2021-01-01',available_from='2021-01-04',source=src,cash_quality_admission=dict(admitted=True,reasons=[]),periods={'current':{'period_end':'2020-12-31'}},fields={'current':{'operating_cash_flow':dict(source=src,scale='1000000',currency_literal='$',locator=loc)}})
        self.t=dict(event_id='X:2020',source=src,inherited_admitted=True,inherited_available_from='2021-01-04')
        self.p={k:self.o[k] for k in a.META};self.p.update(source=src,inherited_admitted=True,status='verified',extraction_verified=True,hold_reasons=[],fields={})
        self.i=dict(event_id='X:2020',source_sha256='a'*64,review_passed=True,paired_cfo_verified=True,paired_assets_verified=True,components={})
        for n,(key,label,val) in enumerate(zip(a.KEYS,['Accounts receivable, net','Inventories','Accounts payable'],['-8','0','12'])):
            row=label+' '+val
            self.p['fields'][key]=dict(component=key,row_label=label,row_text=row,row_text_sha256=hashlib.sha256(row.encode()).hexdigest(),signed_literal=val,signed_value_decimal=val,scale='1000000',currency_literal=None,currency_iso=None,statement_currency_literal='$',currency_scope='same formal cash-flow table; ISO unverified',statement_currency_evidence=dict(format='html',table_index_zero_based=1,original_cfo_locator=deepcopy(loc)),locator=dict(loc,row_index_zero_based=n+3,original_cfo_locator=deepcopy(loc),operating_changes_heading_row_zero_based=2),fact_admitted=False)
            self.i['components'][key]=dict(verified=True,signed_value_decimal=val,scale='1000000',annual_period_end='2020-12-31')
        self.sessions=['2020-12-31','2021-01-04','2021-01-05']
    def run_gate(self):return a.check_record(self.o,self.t,self.p,self.i,self.sessions)
    def test_good_signed_not_summed(self):
        old=deepcopy(self.o);r=self.run_gate();self.assertTrue(r['trade_working_capital_admission']['admitted']);self.assertEqual(old,self.o);self.assertEqual(r['trade_working_capital_source_evidence'],self.p);self.assertNotIn('W',r)
    def test_old_hold_preserved(self):
        self.o['cash_quality_admission']['admitted']=self.t['inherited_admitted']=self.p['inherited_admitted']=False;self.p.update(status='held',hold_reasons=['old']);self.assertFalse(self.run_gate()['trade_working_capital_admission']['admitted'])
    def test_partial_not_zero(self):
        k='accounts_payable';self.p['fields'][k]=None;self.i['components'][k]=dict(verified=False,signed_value_decimal=None,scale=None,annual_period_end=None);self.p.update(status='held',hold_reasons=['combined'],extraction_verified=False);r=self.run_gate();self.assertIsNone(r['trade_working_capital'][k]);self.assertIsNotNone(r['trade_working_capital']['inventories']);self.assertFalse(r['trade_working_capital_admission']['admitted'])
    def test_literal_sign_reject(self):
        self.p['fields']['accounts_receivable']['signed_literal']='8'
        with self.assertRaisesRegex(ValueError,'literal/value'):self.run_gate()
    def test_parentheses(self):
        self.p['fields']['accounts_receivable']['signed_literal']='(8)';self.assertTrue(self.run_gate()['trade_working_capital_admission']['admitted'])
    def test_explicit_dash_zero(self):
        self.p['fields']['inventories']['signed_literal']='—';self.assertTrue(self.run_gate()['trade_working_capital_admission']['admitted'])
    def test_combined_label(self):
        self.p['fields']['accounts_payable']['row_label']='Accounts payable, accrued liabilities and other'
        with self.assertRaisesRegex(ValueError,'Mixed'):self.run_gate()
    def test_vendor_nontrade_rejected(self):
        self.p['fields']['accounts_receivable']['row_label']='Vendor non-trade receivables'
        with self.assertRaisesRegex(ValueError,'Mixed'):self.run_gate()
    def test_table_currency_evidence_alternative(self):
        f=self.p['fields']['inventories'];f.pop('statement_currency_literal');f.pop('currency_scope');f['statement_currency_evidence'].update(literal='$',rows=[dict(row_text='Cash $ 10',row_html_sha256='b'*64)]);self.assertTrue(self.run_gate()['trade_working_capital_admission']['admitted'])
    def test_alternative_currency_missing(self):
        f=self.p['fields']['inventories'];f.pop('statement_currency_literal')
        with self.assertRaisesRegex(ValueError,'dollar evidence'):self.run_gate()
    def test_currency_conflict(self):
        self.p['fields']['inventories']['currency_literal']='EUR'
        with self.assertRaisesRegex(ValueError,'Currency conflict'):self.run_gate()
    def test_unverified_iso_not_added(self):
        self.p['fields']['inventories']['currency_iso']='USD'
        with self.assertRaisesRegex(ValueError,'Currency conflict'):self.run_gate()
    def test_currency_segment_conflict(self):
        self.p['fields']['inventories']['statement_currency_evidence']['statement_segment']={'start_row':50}
        with self.assertRaisesRegex(ValueError,'segment'):self.run_gate()
    def test_same_table_not_enough_outside_cfo(self):
        self.p['fields']['inventories']['locator']['row_index_zero_based']=10
        with self.assertRaisesRegex(ValueError,'Outside'):self.run_gate()
    def test_comparison_period_rejected(self):
        self.p['fields']['inventories']['locator']['period_end']='2019-12-31'
        with self.assertRaisesRegex(ValueError,'annual period'):self.run_gate()
    def test_unit_conflict(self):
        self.p['fields']['inventories']['scale']=self.i['components']['inventories']['scale']='1000'
        with self.assertRaisesRegex(ValueError,'CFO source/unit'):self.run_gate()
    def test_independent_disagreement(self):
        self.i['components']['inventories']['signed_value_decimal']='1'
        with self.assertRaisesRegex(ValueError,'Independent component'):self.run_gate()
    def test_source_disagreement(self):
        self.i['source_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'review'):self.run_gate()
    def test_clock_disagreement(self):
        self.sessions.insert(1,'2021-01-02')
        with self.assertRaisesRegex(ValueError,'First session'):self.run_gate()
    def test_duplicate_rows(self):
        self.p['fields']['inventories']['locator']['row_index_zero_based']=3
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_gate()
    def test_rowhash(self):
        self.p['fields']['inventories']['row_text']='wrong'
        with self.assertRaisesRegex(ValueError,'Row text'):self.run_gate()
    def test_assets_unverified(self):
        self.i['paired_assets_verified']=False
        with self.assertRaisesRegex(ValueError,'assets'):self.run_gate()
    def test_duplicate_census(self):
        with self.assertRaisesRegex(ValueError,'census'):a.indexed([self.p,self.p],[self.p['event_id']])
    def test_missing_census(self):
        with self.assertRaisesRegex(ValueError,'census'):a.indexed([],['missing'])
    def test_pdf_geometry(self):
        cfo=self.o['fields']['current']['operating_cash_flow'];cfo['locator']=dict(format='pdf',page_1based=3,line_1based=20,period_end='2020-12-31')
        for j,f in enumerate(self.p['fields'].values()):
            f.update(currency_scope='same formal cash-flow page; ISO unverified',statement_currency_evidence=dict(format='pdf',page_1based=3,original_cfo_locator=deepcopy(cfo['locator'])),pdfium_text_agrees=True,pdfium_geometry={'current_column':0},locator=dict(cfo['locator'],line_1based=12+j,operating_changes_heading_line_1based=10,original_cfo_locator=deepcopy(cfo['locator'])))
        self.assertTrue(self.run_gate()['trade_working_capital_admission']['admitted']);self.p['fields']['inventories']['pdfium_text_agrees']=False
        with self.assertRaisesRegex(ValueError,'geometry'):self.run_gate()
    def test_malformed_sign(self):
        for x in ['(5','5)','(-5)','NaN','1,23','']:
            with self.assertRaises(ValueError):a.literal_value(x)

if __name__=='__main__':unittest.main()
