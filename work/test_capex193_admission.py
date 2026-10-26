from copy import deepcopy
import hashlib
import unittest
import capex193_admission as a

class Gate(unittest.TestCase):
    def setUp(self):
        self.src={'path':'a.html','sha256':'a'*64}
        self.o={'event_id':'X:FY2020:2021-01-01','symbol':'X','release_date':'2021-01-01','available_from':'2021-01-04','cash_quality_admission':{'admitted':True,'reasons':[]},'periods':{'current':{'period_end':'2020-12-31'}},'fields':{'current':{'operating_cash_flow':{'source':self.src,'scale':'1000000','currency_literal':'$'}}}}
        self.t={'event_id':self.o['event_id'],'source':self.src,'inherited_admitted':True,'inherited_available_from':'2021-01-04'}
        f={'signed_literal':'(12)','signed_value_decimal':'-12','positive_outflow_decimal':'12','scale':'1000000','currency_literal':'$','row_text':'Additions to property and equipment (12)','row_label':'Additions to property and equipment','locator':{'period_end':'2020-12-31'}}
        f['row_text_sha256']=hashlib.sha256(f['row_text'].encode()).hexdigest()
        self.p={k:self.o[k] for k in a.META};self.p.update(source=self.src,inherited_admitted=True,status='verified',hold_reasons=[],extraction_verified=True,field=f)
        self.i={'event_id':self.o['event_id'],'source_sha256':'a'*64,'review_passed':True,'cash_ppe_verified':True,'paired_cfo_verified':True,'paired_assets_verified':True,'positive_outflow_decimal':'12','signed_value_decimal':'-12','scale':'1000000','annual_period_end':'2020-12-31'}
        self.s=['2020-12-31','2021-01-04','2021-01-05']
    def run_gate(self):return a.check_record(self.o,self.t,self.p,self.i,self.s)
    def test_good_preserves_original(self):
        prior=deepcopy(self.o);r=self.run_gate();self.assertTrue(r['cash_ppe_admission']['admitted']);self.assertEqual(prior,self.o)
    def test_inherited_hold_cannot_reopen(self):
        self.o['cash_quality_admission']={'admitted':False,'reasons':['old']};self.t['inherited_admitted']=self.p['inherited_admitted']=False
        self.p.update(status='held',hold_reasons=['old']);self.assertFalse(self.run_gate()['cash_ppe_admission']['admitted'])
    def test_missing_is_not_zero(self):
        self.p.update(field=None,extraction_verified=False,status='held',hold_reasons=['absent']);self.i['cash_ppe_verified']=False
        self.assertIsNone(self.run_gate()['cash_ppe'])
    def test_wrong_clock(self):
        self.s.insert(1,'2021-01-02')
        with self.assertRaisesRegex(ValueError,'first session'):self.run_gate()
    def test_source_changed(self):
        self.i['source_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'source binding'):self.run_gate()
    def test_positive_inflow_cannot_absolute(self):
        self.p['field']['signed_value_decimal']='12'
        with self.assertRaisesRegex(ValueError,'sign/scale'):self.run_gate()
    def test_combined_row_held(self):
        self.p['field']['row_label']+=' and intangible assets'
        with self.assertRaisesRegex(ValueError,'Excluded'):self.run_gate()
    def test_quarter_column(self):
        self.p['field']['locator']['period_end']='2020-09-30'
        with self.assertRaisesRegex(ValueError,'Annual period'):self.run_gate()
    def test_cfo_mismatch(self):
        self.o['fields']['current']['operating_cash_flow']['scale']='1000'
        with self.assertRaisesRegex(ValueError,'CFO unit'):self.run_gate()
    def test_unreviewed_assets(self):
        self.i['paired_assets_verified']=False
        with self.assertRaisesRegex(ValueError,'assets'):self.run_gate()
    def test_missing_independent(self):
        self.i['review_passed']=False
        with self.assertRaisesRegex(ValueError,'independent event'):self.run_gate()
    def test_amount_disagreement(self):
        self.i['positive_outflow_decimal']='13'
        with self.assertRaisesRegex(ValueError,'amount differs'):self.run_gate()
    def test_duplicate_census(self):
        with self.assertRaisesRegex(ValueError,'census'):a.indexed([self.p,self.p],[self.o['event_id']])
    def test_missing_census(self):
        with self.assertRaisesRegex(ValueError,'census'):a.indexed([],['missing'])
    def test_nonfinite_decimal(self):
        with self.assertRaises(ValueError):a.amount('NaN')
    def test_ppe_acquisition_label_allowed(self):
        self.p['field']['row_label']='Payments for acquisition of property, plant and equipment'
        self.assertTrue(self.run_gate()['cash_ppe_admission']['admitted'])
    def test_business_acquisition_rejected(self):
        self.p['field']['row_label']='Payments for property and equipment and acquisitions of businesses'
        with self.assertRaisesRegex(ValueError,'Excluded'):self.run_gate()
    def test_statement_currency_preserved(self):
        self.p['field'].update(currency_literal=None,statement_currency_literal='$',currency_scope='same formal cash-flow table; ISO unverified')
        loc={'format':'html','table_index_zero_based':3}
        self.o['fields']['current']['operating_cash_flow']['locator']=loc
        self.p['field']['locator'].update(**loc,original_cfo_locator=deepcopy(loc))
        r=self.run_gate();self.assertTrue(r['cash_ppe_admission']['admitted']);self.assertIsNone(r['cash_ppe']['currency_literal'])
    def test_statement_currency_requires_scope(self):
        self.p['field'].update(currency_literal=None,statement_currency_literal='$')
        with self.assertRaisesRegex(ValueError,'statement currency'):self.run_gate()
    def test_conflicting_explicit_currency(self):
        self.p['field'].update(currency_literal='EUR',statement_currency_literal='$',currency_scope='same formal cash-flow table; ISO unverified')
        with self.assertRaisesRegex(ValueError,'currency mismatch'):self.run_gate()
    def test_statement_currency_wrong_table(self):
        self.test_statement_currency_preserved()
        self.p['field']['locator']['table_index_zero_based']=4
        with self.assertRaisesRegex(ValueError,'table differs'):self.run_gate()
    def test_statement_currency_pdf_page(self):
        self.p['field'].update(currency_literal=None,statement_currency_literal='$',currency_scope='same formal cash-flow page; ISO unverified')
        loc={'format':'pdf','page_1based':3}
        self.o['fields']['current']['operating_cash_flow']['locator']=loc
        self.p['field']['locator'].update(**loc,original_cfo_locator=deepcopy(loc))
        self.assertTrue(self.run_gate()['cash_ppe_admission']['admitted'])
        self.p['field']['locator']['page_1based']=4
        with self.assertRaisesRegex(ValueError,'page differs'):self.run_gate()
    def test_statement_currency_reject_conflict(self):
        self.p['field']['statement_currency_literal']='EUR'
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.run_gate()
if __name__=='__main__':unittest.main(verbosity=2)
