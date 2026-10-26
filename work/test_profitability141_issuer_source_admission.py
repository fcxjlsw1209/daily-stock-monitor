import copy
from datetime import date,timedelta
import unittest
import profitability141_issuer_source_admission as m


def fixture():
    raw={'symbol':'NVDA','cik':'0001045810','fiscal_year_label':2025,
         'availability':{'release_date':'2025-02-26','first_conservative_session_after_release_date':'2025-02-27'},
         'annual_period':{'start':'2024-01-29','end':'2025-01-26','days_inclusive':364},
         'currency':{'USD_verified':False,'iso_currency':None,'both_statement_currency_labels_match':True},
         'gaap':{'selected_formal_consolidated_statements':True,'non_gaap_tables_excluded':True},
         'source':{'raw_sha256':'a'*64,'source_url':'https://nvidianews.nvidia.com/a','receipt_and_event_verified':True},
         'raw_fact_pair_extracted':True,
         'gross_amount':{'measure_label':'Gross profit','date':'2025-01-26','currency_label':'$','scale':1000,'amount_decimal':'120.50'},
         'total_assets':{'measure_label':'Total assets','date':'2025-01-26','currency_label':'$','scale':1000,'amount_decimal':'300'}}
    anchor={'symbol':'NVDA','fiscal_year_label':2025,'release_date':'2025-02-26','raw':{'sha256':'a'*64},'source_url':'https://nvidianews.nvidia.com/a','issuer_ticker_literal':'NVIDIA (NASDAQ: NVDA)'}
    return raw,anchor,['2025-02-26','2025-02-27','2025-02-28']

class AdmissionTests(unittest.TestCase):
    def test_exact_amounts_and_separate_scales_preserved(self):
        r,a,s=fixture();e=m.make_event(r,a,s)
        self.assertTrue(e['admitted']);self.assertEqual(e['gross_profit'],{'value':'120.50','scale':'1000'})
        self.assertIsNone(e['currency_iso']);self.assertFalse(e['first_public_version_verified'])
    def test_causal_identity_bounds(self):
        r,a,s=fixture();e=m.make_event(r,a,s)
        self.assertEqual(e['identity_valid_from'],'2025-02-27')
        self.assertEqual(e['identity_valid_through'],(date(2025,1,26)+timedelta(days=550)).isoformat())
    def test_bad_newer_event_retained(self):
        r,a,s=fixture();r['total_assets']['amount_decimal']='0'
        d=m.build_dataset([r],[a],s,['NVDA','MSFT','SPY'])
        self.assertEqual(len(d['events']['NVDA']),1);self.assertFalse(d['events']['NVDA'][0]['admitted'])
        self.assertEqual(d['events']['NVDA'][0]['available_from'],'2025-02-27')
        self.assertEqual(d['events']['SPY'],[])
    def test_missing_identity_not_dropped(self):
        r,a,s=fixture();d=m.build_dataset([r],[],s,['NVDA'])
        self.assertFalse(d['events']['NVDA'][0]['admitted'])
    def test_future_append_does_not_change_old_interval_or_record(self):
        r,a,s=fixture();old=m.build_dataset([r],[a],s,['NVDA'])['events']['NVDA'][0]
        future=copy.deepcopy(r);future['fiscal_year_label']=2026
        future['availability']={'release_date':'2026-02-25','first_conservative_session_after_release_date':'2026-02-26'}
        future['annual_period']={'start':'2025-01-27','end':'2026-01-25','days_inclusive':364}
        for k in ['gross_amount','total_assets']:future[k]['date']='2026-01-25'
        af=copy.deepcopy(a);af.update(fiscal_year_label=2026,release_date='2026-02-25')
        new=m.build_dataset([r,future],[a,af],s+['2026-02-25','2026-02-26'],['NVDA'])
        self.assertEqual(new['events']['NVDA'][0],old)
    def test_negative_and_zero_gross_profit_allowed(self):
        for amount in ['0','-10']:
            r,a,s=fixture();r['gross_amount']['amount_decimal']=amount
            self.assertTrue(m.make_event(r,a,s)['admitted'])
    def test_nonfinite_and_bool_rejected(self):
        for amount in ['NaN','Infinity',True,1]:
            r,a,s=fixture();r['gross_amount']['amount_decimal']=amount
            self.assertFalse(m.make_event(r,a,s)['admitted'])
    def test_wrong_measure_rejected(self):
        r,a,s=fixture();r['gross_amount']['measure_label']='Revenue'
        self.assertFalse(m.make_event(r,a,s)['admitted'])
    def test_same_day_use_rejected(self):
        r,a,s=fixture();r['availability']['first_conservative_session_after_release_date']='2025-02-26'
        self.assertFalse(m.make_event(r,a,s)['admitted'])
    def test_wrong_identity_or_source_rejected(self):
        r,a,s=fixture();a['raw']['sha256']='b'*64
        self.assertFalse(m.make_event(r,a,s)['admitted'])
    def test_duplicate_event_and_anchor_fail(self):
        r,a,s=fixture()
        with self.assertRaises(ValueError):m.build_dataset([r,r],[a],s,['NVDA'])
        with self.assertRaises(ValueError):m.build_dataset([r],[a,a],s,['NVDA'])
    def test_mismatched_end_rejected(self):
        r,a,s=fixture();r['total_assets']['date']='2025-01-25'
        self.assertFalse(m.make_event(r,a,s)['admitted'])
    def test_outside_universe_fails(self):
        r,a,s=fixture()
        with self.assertRaises(ValueError):m.build_dataset([r],[a],s,['MSFT'])
    def test_source_and_caller_unchanged(self):
        r,a,s=fixture();before=copy.deepcopy((r,a,s));m.make_event(r,a,s)
        self.assertEqual((r,a,s),before)

if __name__=='__main__':unittest.main()
