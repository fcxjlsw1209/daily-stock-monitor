"""Synthetic-only tests: no actual CSCO report amounts or real input files."""
from copy import deepcopy
from datetime import date,timedelta
import hashlib
import json
import unittest
from unittest.mock import patch
import after149_csco_extract as m


def cf(ni='$ 500',prior='$ 7777',title='CONSOLIDATED STATEMENTS OF CASH FLOWS',duration='Fiscal Year Ended',label='Net income',extra='',units='millions',quarter=False):
    if quarter:
        return ('<table><caption>'+title+' (In '+units+')</caption>'
         '<tr><td></td><td colspan="2">Three Months Ended</td><td colspan="2">Fiscal Year Ended</td></tr>'
         '<tr><td></td><td>July 26, 2025</td><td>July 27, 2024</td><td>July 26, 2025</td><td>July 27, 2024</td></tr>'
         '<tr><td>'+label+'</td><td>$ 999999</td><td>$ 888888</td><td>'+ni+'</td><td>'+prior+'</td></tr>'
         '<tr><td>Adjustments to reconcile net income:</td></tr>'
         '<tr><td>Share-based compensation expense</td><td>111</td><td>112</td><td>333333</td><td>334</td></tr>'
         '<tr><td>Net cash provided by operating activities</td><td>1111</td><td>2222</td><td>444444</td><td>4444</td></tr>'+extra+'</table>')
    return ('<table><caption>'+title+' (In '+units+')</caption>'
         '<tr><td></td><td colspan="2">'+duration+'</td></tr>'
         '<tr><td></td><td>July 26, 2025</td><td>July 27, 2024</td></tr>'
         '<tr><td>'+label+'</td><td>'+ni+'</td><td>'+prior+'</td></tr>'
         '<tr><td>Adjustments to reconcile net income:</td></tr>'
         '<tr><td>Share-based compensation expense</td><td>333333</td><td>334</td></tr>'
         '<tr><td>Excess tax benefits from share-based compensation</td><td>444444</td><td>445</td></tr>'
         '<tr><td>Net cash provided by operating activities</td><td>555555</td><td>556</td></tr>'+extra+'</table>')


def bs(current='$ 2000',prior='$ 1500',units='millions',label='TOTAL ASSETS',title='CONDENSED CONSOLIDATED BALANCE SHEETS'):
    return ('<table><caption>'+title+' (In '+units+')</caption>'
       '<tr><td></td><td>July 26, 2025</td><td>July 27, 2024</td></tr>'
       '<tr><td>ASSETS</td></tr><tr><td>'+label+'</td><td>'+current+'</td><td>'+prior+'</td></tr></table>')


def fixture(cash=None,balance=None):
    raw=('<p>CISCO SYSTEMS, INC. (NASDAQ: CSCO)</p>'+ (cf()if cash is None else cash)+(bs()if balance is None else balance)).encode()
    context={'event_id':'CSCO_2025-08-13','fiscal_year':2025,'release_date':'2025-08-13','available_session':'2025-08-14',
      'source':{'path':'synthetic.html','sha256':hashlib.sha256(raw).hexdigest(),'url':'https://newsroom.cisco.com/synthetic','receipt_path':'synthetic.meta.json'},
      'source_issuer_name_found':True,'source_listing_reference':'NASDAQ: CSCO',
      'periods':{'current':{'period_start':'2024-07-28','period_end':'2025-07-26','days_inclusive':364},'comparison':{'period_start':'2023-07-30','period_end':'2024-07-27','days_inclusive':364}},
      'release_metadata':{'ticker':'CSCO','release_date':'2025-08-13','fiscal_year':2025,'fiscal_quarter':4,'evidence_path':'synthetic.html','evidence_sha256':hashlib.sha256(raw).hexdigest(),'fiscal_period_end_actual':'2025-07-26','first_public_time_verified':False},
      'calendar_verification':{'status':'first_strictly_later_pinned_session_verified'},'accounting_context':[],
      'tables':[{'table':0,'title':['CONSOLIDATED STATEMENTS OF CASH FLOWS'],'labels':[{'row':2,'label':'Net income'}],
                 'columns':[{'column':1,'dates':['2025-07-26'],'annual':True,'quarter':False},{'column':2,'dates':['2024-07-27'],'annual':True,'quarter':False}]},
                {'table':1,'title':['CONDENSED CONSOLIDATED BALANCE SHEETS'],'labels':[{'row':2,'label':'TOTAL ASSETS'}],
                 'columns':[{'column':1,'dates':['2025-07-26'],'annual':False,'quarter':False},{'column':2,'dates':['2024-07-27'],'annual':False,'quarter':False}]}]}
    return raw,context


def change_year(raw,e,year,release='08-13'):
    # Generated date fixtures with explicitly computed annual boundaries.
    raw=raw.replace(b'2025',str(year).encode()).replace(b'2024',str(year-1).encode())
    e=deepcopy(e);e['fiscal_year']=year;e['release_date']=str(year)+'-'+release;e['available_session']=str(year)+'-08-14';e['event_id']='CSCO_'+e['release_date']
    for side,yr,end,start in [('current',year,'07-26','07-28'),('comparison',year-1,'07-27','07-30')]:
        p=e['periods'][side];p['period_end']=f'{yr}-{end}';p['period_start']=f'{yr-1}-{start}';p['days_inclusive']=(date.fromisoformat(p['period_end'])-date.fromisoformat(p['period_start'])).days+1
    for t in e['tables']:
        for c in t['columns']:c['dates']=[s.replace('2025',str(year)).replace('2024',str(year-1))for s in c['dates']]
    e['release_metadata'].update(fiscal_year=year,release_date=e['release_date'],fiscal_period_end_actual=e['periods']['current']['period_end'])
    return rehash(raw,e)


def rehash(raw,e):
    e=deepcopy(e);e['source']['sha256']=hashlib.sha256(raw).hexdigest();e['release_metadata']['evidence_sha256']=e['source']['sha256'];return raw,e


class ExtractTests(unittest.TestCase):
    def test_complete_and_provenance(self):
        raw,e=fixture();before=deepcopy(e);r=m.extract_document(raw,e)
        self.assertEqual(e,before);self.assertFalse(r['gaps']);self.assertEqual(r['original_context'],e)
        self.assertEqual([r['fields'][f]['value_decimal']for f in m.FIELDS],['500','2000','1500'])
        for v in r['fields'].values():
            self.assertEqual(v['scale'],'1000000');self.assertEqual(v['source'],e['source']);self.assertEqual(v['available_from'],e['available_session'])
            self.assertEqual(v['currency_literal'],'$');self.assertIsNone(v['currency_iso']);self.assertFalse(v['fact_admitted'])
            self.assertTrue(v['locator']['physical_cells']);self.assertTrue(v['locator']['header_cells'])
        self.assertEqual(r['fields']['annual_net_income']['period'],e['periods']['current'])
        for k in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','currency_iso_verified'):self.assertIs(r[k],False)
    def test_only_three_requested_values(self):
        r=m.extract_document(*fixture());text=json.dumps(r)
        for excluded in ('7777','333333','444444','555555'):self.assertNotIn(excluded,text)
        self.assertEqual(set(r['fields']),set(m.FIELDS))
    def test_signed_zeros_and_commas(self):
        for v,wanted in [('$ (1,234)','-1234'),('$ -1,234','-1234'),('$ \u22121,234','-1234'),('$ 0','0'),('$ 1 , 234.50','1234.50')]:
            with self.subTest(v=v):self.assertEqual(m.extract_document(*fixture(cf(ni=v)))['fields']['annual_net_income']['value_decimal'],wanted)
    def test_bad_amounts_stay_gaps(self):
        for v in ('$ -','$ 1 234','$ 12,34','$ NaN','$ 5%','$ 100 (a)','$ (-4)','$ (4',' $ 2/3'):
            with self.subTest(v=v):
                r=m.extract_document(*fixture(cf(ni=v)));self.assertIsNone(r['fields']['annual_net_income']);self.assertTrue(r['gaps'])
                self.assertIsNotNone(r['fields']['current_total_assets'])
    def test_no_quarter_as_annual(self):
        r=m.extract_document(*fixture(cf(duration='Three Months Ended')));self.assertIsNone(r['fields']['annual_net_income'])
    def test_quarter_and_annual_separate_columns(self):
        raw,e=fixture(cf(quarter=True));e['tables'][0]['columns']=[{'column':3,'dates':['2025-07-26'],'annual':True,'quarter':False},{'column':4,'dates':['2024-07-27'],'annual':True,'quarter':False}]
        r=m.extract_document(raw,e);self.assertEqual(r['fields']['annual_net_income']['value_decimal'],'500');self.assertEqual(r['fields']['annual_net_income']['locator']['logical_columns'],[3])
    def test_parent_income_not_substitute(self):
        for label in ('Net income attributable to common shareholders','Income from continuing operations','Net income attributable to Cisco','Income before taxes'):
            with self.subTest(label=label):self.assertIsNone(m.extract_document(*fixture(cf(label=label)))['fields']['annual_net_income'])
    def test_net_loss_supported(self):
        raw,e=fixture(cf(label='Net loss',ni='$ (5)'));e['tables'][0]['labels'][0]['label']='Net loss'
        self.assertEqual(m.extract_document(raw,e)['fields']['annual_net_income']['value_decimal'],'-5')
    def test_preceding_pretax_or_continuing_scope_rejected(self):
        for label in ('Income before income taxes','Income from continuing operations','Earnings before taxes'):
            cash=cf().replace('<tr><td>Net income</td>', '<tr><td>'+label+'</td><td>$ 1</td><td>$ 2</td></tr><tr><td>Net income</td>')
            raw,e=fixture(cash);e['tables'][0]['labels'][0]['row']=3
            with self.subTest(label=label):self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_non_gaap_does_not_substitute(self):
        r=m.extract_document(*fixture(cf(title='NON-GAAP CONSOLIDATED STATEMENTS OF CASH FLOWS')));self.assertIsNone(r['fields']['annual_net_income'])
    def test_duplicate_formal_statements_ambiguous(self):
        raw,e=fixture(cf()+cf());e['tables'][1]['table']=2
        self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_duplicate_net_income_ambiguous(self):
        self.assertIsNone(m.extract_document(*fixture(cf(extra='<tr><td>Net income</td><td>$ 10</td><td>$ 11</td></tr>')))['fields']['annual_net_income'])
    def test_missing_adjustment_or_CFO_scope(self):
        for before,after in [('Adjustments to reconcile net income:','Other items'),('Net cash provided by operating activities','Something else')]:
            raw,e=fixture(cf().replace(before,after));self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_wrong_period_cannot_substitute(self):
        raw,e=fixture(cf().replace('July 26, 2025','July 19, 2025'));self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_latest_current_enforced(self):
        raw,e=fixture(cf().replace('July 27, 2024','July 25, 2026'));self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_unsupported_currency(self):
        r=m.extract_document(*fixture(cf(ni='EUR 500')));self.assertIsNone(r['fields']['annual_net_income'])
    def test_missing_dollar_not_inferred(self):
        r=m.extract_document(*fixture(cf(ni='500',prior='7777')));self.assertIsNone(r['fields']['annual_net_income'])
    def test_ambiguous_units(self):
        self.assertIsNone(m.extract_document(*fixture(cf(units='millions and thousands')))['fields']['annual_net_income'])
    def test_thousands_preserved_no_scaling_arithmetic(self):
        r=m.extract_document(*fixture(cf(units='thousands')));self.assertEqual(r['fields']['annual_net_income']['value_decimal'],'500');self.assertEqual(r['fields']['annual_net_income']['scale'],'1000')
    def test_dollar_and_amount_separate_cells_assets(self):
        balance=('<table><caption>CONSOLIDATED BALANCE SHEETS (In millions)</caption><tr><td colspan="2"></td><td colspan="2">July 26, 2025</td><td></td><td colspan="2">July 27, 2024</td></tr>'
        '<tr><td colspan="2">ASSETS</td></tr><tr><td colspan="2">TOTAL ASSETS</td><td>$</td><td>2,000</td><td></td><td>$</td><td>1,500</td></tr></table>')
        raw,e=fixture(balance=balance);e['tables'][1]['columns']=[{'column':i,'dates':[d],'annual':False,'quarter':False}for d,ix in [('2025-07-26',[2,3]),('2024-07-27',[5,6])]for i in ix]
        r=m.extract_document(raw,e);self.assertEqual(r['fields']['current_total_assets']['value_decimal'],'2000');self.assertEqual(r['fields']['prior_total_assets']['value_decimal'],'1500')
    def test_cross_date_colspan_rejected(self):
        balance=bs().replace('<td>$ 2000</td><td>$ 1500</td>','<td colspan="2">$ 2000</td>')
        r=m.extract_document(*fixture(balance=balance));self.assertIsNone(r['fields']['current_total_assets']);self.assertIsNone(r['fields']['prior_total_assets'])
    def test_cross_row_span_rejected(self):
        r=m.extract_document(*fixture(cf(),bs().replace('<td>$ 2000</td>','<td rowspan="2">$ 2000</td>')))
        self.assertIsNone(r['fields']['current_total_assets'])
    def test_nested_financial_table_rejected(self):
        raw,e=fixture(cf(extra='<tr><td><table><tr><td>Nested</td></tr></table></td></tr>'));e['tables'][1]['table']=2
        self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_context_row_or_columns_changed(self):
        for mode in ('row','column'):
            raw,e=fixture()
            if mode=='row':e['tables'][0]['labels'][0]['row']=3
            else:e['tables'][0]['columns'][0]['column']=2
            self.assertIsNone(m.extract_document(raw,e)['fields']['annual_net_income'])
    def test_source_hash_wrong_raises(self):
        raw,e=fixture()
        with self.assertRaisesRegex(ValueError,'SHA'):m.extract_document(raw+b' ',e)
    def test_identity_context_raises(self):
        for key,value in [('event_id','AMD_2025-08-13'),('source_issuer_name_found',False),('fiscal_year',True)]:
            raw,e=fixture();e[key]=value
            with self.assertRaises(ValueError):m.extract_document(raw,e)
    def test_raw_identity_raises(self):
        raw,e=fixture();raw,e=rehash(raw.replace(b'NASDAQ: CSCO',b'NASDAQ: XXXX'),e)
        with self.assertRaisesRegex(ValueError,'identity'):m.extract_document(raw,e)
    def test_release_metadata_mismatch(self):
        raw,e=fixture();e['release_metadata']['fiscal_quarter']=3
        with self.assertRaises(ValueError):m.extract_document(raw,e)
    def test_no_future_availability_or_end(self):
        for v in ('2025-08-13','2025-08-12'):
            raw,e=fixture();e['available_session']=v
            with self.assertRaises(ValueError):m.extract_document(raw,e)
    def test_duration_and_contiguity_raises(self):
        raw,e=fixture();e['periods']['current']['days_inclusive']=365
        with self.assertRaises(ValueError):m.extract_document(raw,e)
        raw,e=fixture();e['periods']['current'].update(period_start='2024-07-29',days_inclusive=363)
        with self.assertRaises(ValueError):m.extract_document(raw,e)
    def test_FY2011_holds_preserved(self):
        raw,e=change_year(*fixture(),2011);e['periods']['comparison'].update(period_start=None,days_inclusive=None);e['calendar_verification']['status']='outside_pinned_calendar_range_existing_catalog_date_only'
        r=m.extract_document(raw,e);self.assertEqual(len(r['context_holds']),2);self.assertFalse(r['gaps'])
    def test_unknown_comparative_start_other_year_raises(self):
        raw,e=fixture();e['periods']['comparison'].update(period_start=None,days_inclusive=None)
        with self.assertRaises(ValueError):m.extract_document(raw,e)
    def test_FY2019_holds_cannot_be_dropped(self):
        r=m.extract_document(*change_year(*fixture(),2019));self.assertIn('ASC606_modified_retrospective_NI_asset_comparability_unresolved',r['context_holds']);self.assertFalse(r['fact_admitted'])
    def test_unknown_calendar_does_not_guess(self):
        raw,e=fixture();e.pop('calendar_verification');r=m.extract_document(raw,e);self.assertIn('strict_next_session_calendar_not_independently_verified',r['context_holds'])
    def test_pure_no_files_or_mutation(self):
        raw,e=fixture();before=deepcopy(e)
        with patch('builtins.open',side_effect=AssertionError('I/O forbidden')):r=m.extract_document(raw,e)
        self.assertEqual(before,e);self.assertEqual(r['financial_ratios_computed'],0)

if __name__=='__main__':unittest.main(verbosity=2)
