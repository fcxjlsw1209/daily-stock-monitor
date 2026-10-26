"""Synthetic financial statements only; no admitted dataset or market data reads."""
from copy import deepcopy
from datetime import date,timedelta
from fractions import Fraction
from decimal import localcontext
import hashlib,json,unittest
import numpy as np
import monthly_cash_ppe193_independent as o


def report(symbol='MSFT',identifier='oracle'):
    period=lambda y:dict(period_start=str(y)+'-01-01',period_end=str(y)+'-12-31',days_inclusive=366 if y==2024 else 365)
    periods={'current':period(2025),'comparison':period(2024)}
    source={'path':'synthetic/annual.html','sha256':'b'*64}
    e=dict(event_id=identifier,symbol=symbol,release_date='2026-01-29',available_from='2026-01-30',
        identity=dict(issuer_id='company-'+symbol,security_id=symbol,identity_valid_from='2026-01-30',
                      identity_valid_through=str(date(2025,12,31)+timedelta(days=550))),
        periods=periods,source=source,inherited_rejection_codes=[],
        cash_quality_admission={'admitted':True,'definition':o.OLD_DEFINITION,'reasons':[]},
        cash_ppe_admission={'admitted':True,'definition':o.DEFINITION,'reasons':[]})
    e['fields']={}
    for side in ('current','comparison'):
        e['fields'][side]={}
        for field,value in [('operating_cash_flow','42.5'),('net_income','-3'),('total_assets','240' if side=='current' else '160')]:
            e['fields'][side][field]=dict(value_decimal=value,scale='1000',currency_literal='$',source=deepcopy(source),
                period=deepcopy(periods[side]),available_from=e['available_from'])
    loc=dict(format='html',table_index_zero_based=3,statement_segment='cash-flows')
    e['fields']['current']['operating_cash_flow']['locator']=deepcopy(loc)
    row='Additions to property and equipment (7.5)'
    p=dict(signed_literal='(7.5)',signed_value_decimal='-7.5',positive_outflow_decimal='7.5',scale='1000',
        currency_literal=None,statement_currency_literal='$',currency_iso=None,
        currency_scope='same formal cash-flow table; ISO unverified',row_label='Additions to property and equipment',
        row_text=row,row_text_sha256=hashlib.sha256(row.encode()).hexdigest(),
        locator=dict(loc,original_cfo_locator=deepcopy(loc),period_end='2025-12-31'))
    e['cash_ppe_source_evidence']={k:deepcopy(e[k]) for k in ('event_id','symbol','source','release_date','available_from')}
    e['cash_ppe_source_evidence'].update(inherited_admitted=True,extraction_verified=True,status='verified',hold_reasons=[],field=deepcopy(p))
    e['cash_ppe']=dict(p,source=deepcopy(source),period=deepcopy(periods['current']),available_from=e['available_from'],fact_admitted=False)
    return e


def sync(e):
    p=e['cash_ppe_source_evidence']
    for key in ('event_id','symbol','source','release_date','available_from'):p[key]=deepcopy(e[key])
    key='field' if 'field' in p else 'capex'
    for name in list(p[key]):p[key][name]=deepcopy(e['cash_ppe'][name])
    return e


def fixture():
    names=['MSFT','AAPL','AMD','NVDA']+['STOCK'+str(i) for i in range(28)]+['ETF'+str(i) for i in range(17)]
    dates=['2026-01-30','2026-03-31','2026-05-29']
    meta=dict(dates=dates,symbols=[names[:] for _ in dates],etfs=names[32:])
    data=dict(schema_version=1,method=o.METHOD,conditional_research_only=True,events=[report()])
    return meta,data


class Tests(unittest.TestCase):
    def evaluate(self,e=None,signal='2026-03-31'):return o.evaluate_oracle(report() if e is None else e,signal)
    def test_closed_form_and_decimal_context(self):
        with localcontext() as context:
            context.prec=2;r=self.evaluate()
        self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],float(Fraction(7,47)))
        self.assertEqual(r['control_bounded'],float(Fraction(17,97)))
    def test_signed_cash_and_outflow(self):
        for cash in ('-100','0','2','7.5','42.5'):
            e=report();e['fields']['current']['operating_cash_flow']['value_decimal']=cash;r=self.evaluate(e)
            for key,n in [('candidate_bounded',Fraction(cash)-Fraction('7.5')),('control_bounded',Fraction(cash))]:
                self.assertEqual(r[key],float(n/(200+abs(n))))
    def test_scale_and_exact_cancellation(self):
        e=report()
        for side in ('current','comparison'):
            for c in e['fields'][side].values():c['scale']='1'
        e['cash_ppe']['scale']='1';sync(e);self.assertEqual(self.evaluate(e),self.evaluate())
        huge=10**100;e['fields']['current']['operating_cash_flow']['value_decimal']=str(huge+1)
        e['cash_ppe'].update(positive_outflow_decimal=str(huge),signed_value_decimal=str(-huge));sync(e)
        self.assertEqual(self.evaluate(e)['candidate_bounded'],float(Fraction(1,201)))
    def test_zero_outflow_is_valid_not_missing(self):
        e=report();e['cash_ppe'].update(positive_outflow_decimal='0',signed_value_decimal='0');sync(e)
        r=self.evaluate(e);self.assertTrue(r['available']);self.assertEqual(r['candidate_bounded'],r['control_bounded'])
        e['cash_ppe']=None;self.assertFalse(self.evaluate(e)['available'])
    def test_admissions_are_strict_both_required(self):
        for key in ('cash_quality_admission','cash_ppe_admission'):
            for mutation in ({'admitted':1},{'admitted':False},{'reasons':['hold']},{'definition':'wrong'},{'extra':True}):
                e=report();e[key].update(mutation);self.assertFalse(self.evaluate(e)['available'])
    def test_original_ni_gate_not_used_in_value(self):
        e=report();e['fields']['current']['net_income']['value_decimal']='1e100';self.assertEqual(self.evaluate(e),self.evaluate())
        e['fields']['current']['net_income']=None;self.assertEqual(self.evaluate(e)['reason'],'missing_current_net_income')
        e=report();e['fields']['comparison']['net_income']=object();e['fields']['comparison']['operating_cash_flow']=object()
        self.assertEqual(self.evaluate(e),self.evaluate())
    def test_source_clock_period_immutable_evidence(self):
        for key,value in [('source',{}),('available_from','2026-01-31'),('period',{}),('fact_admitted',True)]:
            e=report();e['cash_ppe'][key]=value;self.assertFalse(self.evaluate(e)['available'])
        e=report();e['cash_ppe']['signed_value_decimal']='-8';self.assertFalse(self.evaluate(e)['available'])
        e=report();e['cash_ppe_source_evidence']['extraction_verified']=False;self.assertFalse(self.evaluate(e)['available'])
    def test_statement_currency_structure(self):
        self.assertTrue(self.evaluate()['available'])
        for key,value in [('table_index_zero_based',4),('table_index_zero_based',True),('statement_segment','other'),('original_cfo_locator',{}),('format','pdf')]:
            e=report();e['cash_ppe']['locator'][key]=value;sync(e);self.assertFalse(self.evaluate(e)['available'])
        e=report();loc=dict(format='pdf',page_1based=8)
        e['fields']['current']['operating_cash_flow']['locator']=deepcopy(loc)
        e['cash_ppe']['locator']=dict(loc,original_cfo_locator=deepcopy(loc),period_end='2025-12-31')
        e['cash_ppe']['currency_scope']='same formal cash-flow page; ISO unverified';sync(e)
        self.assertTrue(self.evaluate(e)['available'])
        e['cash_ppe']['locator']['page_1based']=0;sync(e);self.assertFalse(self.evaluate(e)['available'])
    def test_real_label_semantics_without_real_values(self):
        e=report();e['cash_ppe']['row_label']='Payments for acquisition of property, plant and equipment';sync(e)
        self.assertTrue(self.evaluate(e)['available'])
        for bad in ('Purchases of property and equipment and intangible assets','Payments for acquisition of property and equipment and businesses',
                    'Principal payments for property and equipment','Free cash flow'):
            e['cash_ppe']['row_label']=bad;sync(e)
            self.assertFalse(self.evaluate(e)['available'])
    def test_source_sign_currency_and_rowhash(self):
        for k,v in [('positive_outflow_decimal','-1'),('signed_value_decimal','1'),('scale','0'),('scale','1'),
                    ('currency_literal','EUR'),('statement_currency_literal','EUR'),('row_text_sha256','0'*64),('positive_outflow_decimal','NaN')]:
            e=report();e['cash_ppe'][k]=v;sync(e);self.assertFalse(self.evaluate(e)['available'],k)
    def test_annual_expiry_and_identity(self):
        end=date(2025,12,31)
        self.assertTrue(self.evaluate(signal=str(end+timedelta(days=550)))['available'])
        self.assertFalse(self.evaluate(signal=str(end+timedelta(days=551)))['available'])
        for key,value in [('security_id','AMD'),('identity_valid_from','2026-01-29'),('identity_valid_through','2030-01-01')]:
            e=report();e['identity'][key]=value;self.assertFalse(self.evaluate(e)['available'])
        e=report();e['periods']['current']['days_inclusive']=True;self.assertFalse(self.evaluate(e)['available'])
    def test_latest_hold_blocks_without_fallback(self):
        meta,data=fixture();new=report(identifier='new');new['available_from']='2026-03-01';new['cash_ppe_admission']['admitted']=False
        data['events'].append(new);built=o.independent_features(meta,data)
        self.assertTrue(built['arrays']['common_available'][0,0]);self.assertFalse(built['arrays']['common_available'][1:,0].any())
        self.assertEqual(built['audit']['cells'][49]['current_event_id'],'new')
    def test_future_poison_and_all_prefixes(self):
        meta,data=fixture();full=o.independent_features(meta,data)
        data['events'].append(dict(event_id='future',symbol='MSFT',available_from='2030-01-02',cash_ppe=object()))
        for n in range(4):
            short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]);got=o.independent_features(short,data)
            for key in o.KEYS:np.testing.assert_array_equal(got['arrays'][key],full['arrays'][key][:n])
            self.assertEqual(got['audit']['cells'],full['audit']['cells'][:n*49])
    def test_duplicate_ambiguity_etf(self):
        meta,data=fixture();data['events'].append(deepcopy(data['events'][0]));self.assertTrue(o.independent_features(meta,data)['arrays']['common_available'][:,0].all())
        data['events'][-1]['event_id']='other';self.assertFalse(o.independent_features(meta,data)['arrays']['common_available'].any())
        data['events']=[report('ETF0')];self.assertFalse(o.independent_features(meta,data)['arrays']['common_available'].any())
    def test_verifier_rejects_corrupt_arrays_and_audit(self):
        meta,data=fixture();built=o.independent_features(meta,data);receipt=o.verify_features(meta,data,built)
        self.assertEqual(receipt['available_cells'],3);self.assertEqual(receipt['checked_cells'],147)
        self.assertEqual(receipt['width'],receipt['symbols_per_event']);self.assertEqual(receipt['symbols_per_event'],49)
        self.assertEqual(receipt['predictive_model_fits'],0);self.assertEqual(receipt['new_model_fits'],0)
        for key in ('candidate_bounded','control_bounded','common_available'):
            bad=deepcopy(built);bad['arrays'][key][0,0]=False if key=='common_available' else .6
            with self.assertRaises(ValueError):o.verify_features(meta,data,bad)
        bad=deepcopy(built);bad['audit']['cells'][0]['current_days']=1
        with self.assertRaises(ValueError):o.verify_features(meta,data,bad)
    def test_delta_eligibility_including_initial(self):
        meta,data=fixture();a=o.independent_features(meta,data)['arrays'];eligible=np.ones((3,49),bool);eligible[1,0]=False
        ds=o.independent_deltas(a,eligible);receipt=o.verify_deltas(a,ds,eligible)
        self.assertTrue(receipt['eligible_only']);self.assertNotEqual(ds['cash_after_ppe'][0,0],0)
        self.assertEqual(ds['cash_after_ppe'][1,0],0);self.assertEqual(receipt['changed_cells']['cash_after_ppe'],2)
        ds['cash_after_ppe'][0,0]=0
        with self.assertRaises(ValueError):o.verify_deltas(a,ds,eligible)
        with self.assertRaises(ValueError):o.independent_deltas(a,eligible.astype(int))
        with self.assertRaises(ValueError):o.independent_deltas(a,eligible,.02)
    def test_schema_and_no_mutation(self):
        meta,data=fixture();saved=deepcopy((meta,data));built=o.independent_features(meta,data);self.assertEqual((meta,data),saved)
        json.dumps(built['audit'],allow_nan=False)
        data['schema_version']=True
        with self.assertRaises(ValueError):o.independent_features(meta,data)
        meta,data=fixture();meta['symbols'][0][1]='MSFT'
        with self.assertRaises(ValueError):o.independent_features(meta,data)
    def test_pinned_accounting_no_producer_dependency(self):
        helper=o.frozen_helper();self.assertTrue(callable(o.accounting().audit_path))
        self.assertNotIn('capex193_factor',o.__dict__)
        self.assertEqual(helper.FROZEN_HELPER_SHA,'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc')
    def test_full_producer_cross_on_both_synthetic_fixtures(self):
        import capex193_factor as producer
        import test_capex193_factor as producer_tests
        for make in (fixture,producer_tests.fixture):
            meta,data=make()
            for n in range(len(meta['dates'])+1):
                short=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
                built=producer.build_features(short,data);o.verify_features(short,data,built)
            for mutation in ('missing','held','currency','label','identity'):
                m,d=make();e=d['events'][0]
                if mutation=='missing':e['cash_ppe']=None
                elif mutation=='held':e['cash_quality_admission']['admitted']=False
                elif mutation=='currency':e['cash_ppe']['currency_literal']='EUR'
                elif mutation=='label':e['cash_ppe']['row_label']='Purchases of property and equipment and intangible assets'
                else:e['identity']['security_id']='WRONG'
                o.verify_features(m,d,producer.build_features(m,d))

if __name__=='__main__':unittest.main()
